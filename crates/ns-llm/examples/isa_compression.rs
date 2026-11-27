//! cargo run --release -p ns-llm --features custom-rwkv --example isa_compression -- \
//!   [dataset.jsonl] [vocab.bin]
//!
//! Mede a compressão REAL da ISA sobre um dataset JSONL verificado usando o `TokTrie` do RWKV-7:
//! - Valida em cada linha o round-trip exato `decode_one(encode(...))` e a gramática `expect()`;
//!   retorna `ExitCode::FAILURE` se qualquer amostra falhar.
//! - Separa literais nativos (`Expect::Lit`, ex.: texto livre / expressão) de fallbacks de
//!   ponteiro (`Expect::Ref | RefOrEnd -> Sym::Lit`) para reportar a taxa real `Ptr%`.
//! - Compara JSON vs. ISA original vs. ISA `@slot` (com `ProposeAnswer.text` vazio, como no
//!   `Orchestrator` atual) e reporta `p50`/`p95` de passos ISA.
//!
//! Variáveis úteis:
//! - `NS_DATASET_PATH`: caminho padrão do dataset (padrão: `artifacts/verified_dataset.jsonl`).
//! - `NS_VOCAB_BIN`: caminho do `vocab.bin` (autodetecta `rwkv7_g1_01b_isa` ou `rwkv7_g1_01b_base`).
//! - `NS_MS_PER_TOK`: opcional; se definido (ex.: `2.4`), exibe economia estimada em `ms/step`.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ns_core::action::LlmAction;
use ns_llm::isa::{Action, Bank, Expect, Sym, decode_one, encode, expect};
use ns_llm::isa_adapter::{CostMeter, bank_add_text};
use ns_llm::load_vocab_trie;
use serde::Deserialize;
use serde_json::Value;

const MAX_LOGGED_ERRORS: usize = 5;

#[derive(Debug, Deserialize)]
struct DatasetRow {
    scenario: String,
    prompt: String,
    action: LlmAction,
}

#[derive(Default)]
struct ScenarioStats {
    meter: CostMeter,
    slot_meter: CostMeter,
    isa_steps_per_action: Vec<usize>,
    ptr_hits: usize,
    ptr_misses: usize,
    native_lits: usize,
    roundtrip_errors: usize,
    grammar_errors: usize,
}

impl ScenarioStats {
    fn ptr_hit_pct(&self) -> f64 {
        let total = self.ptr_hits + self.ptr_misses;
        if total == 0 {
            100.0
        } else {
            100.0 * (self.ptr_hits as f64) / (total as f64)
        }
    }
}

/// Constrói o `Bank` a partir de tudo o que já apareceu no `prompt`:
/// - payloads JSON em `TOOL: {...}` e `SYSTEM: CANDIDATES: [...]`
/// - termos/identificadores presentes em `USER:` e demais linhas `SYSTEM:`
fn build_bank_from_prompt(prompt: &str) -> Bank {
    let mut bank = Bank::new();

    for line in prompt.lines() {
        let trimmed = line.trim();
        if let Some(json_part) = trimmed.strip_prefix("TOOL:") {
            if let Ok(val) = serde_json::from_str::<Value>(json_part.trim()) {
                collect_json_strings(&val, &mut bank);
            }
        } else if let Some(user_part) = trimmed.strip_prefix("USER:") {
            bank_add_text(&mut bank, user_part);
        } else if let Some(sys_part) = trimmed.strip_prefix("SYSTEM:") {
            let rest = sys_part.trim();
            if let Some(cand_part) = rest.strip_prefix("CANDIDATES:") {
                if let Ok(val) = serde_json::from_str::<Value>(cand_part.trim()) {
                    collect_json_strings(&val, &mut bank);
                }
            } else {
                bank_add_text(&mut bank, rest);
            }
        }
    }

    bank
}

fn collect_json_strings(val: &Value, bank: &mut Bank) {
    match val {
        Value::String(s) if !s.is_empty() => {
            let _ = bank.intern(s);
        }
        Value::Array(arr) => {
            for item in arr {
                collect_json_strings(item, bank);
            }
        }
        Value::Object(map) => {
            for v in map.values() {
                collect_json_strings(v, bank);
            }
        }
        _ => {}
    }
}

fn as_slot_action(action: &LlmAction) -> LlmAction {
    match action {
        LlmAction::ProposeAnswer { cited_facts, .. } => LlmAction::ProposeAnswer {
            text: String::new(),
            cited_facts: cited_facts.clone(),
        },
        other => other.clone(),
    }
}

fn percentile(sorted: &[usize], pct: usize) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = sorted.len().saturating_sub(1).saturating_mul(pct) / 100;
    sorted[idx]
}

fn resolve_dataset_path() -> PathBuf {
    if let Some(arg) = std::env::args().nth(1) {
        return PathBuf::from(arg);
    }
    if let Ok(env_path) = std::env::var("NS_DATASET_PATH") {
        let trimmed = env_path.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    PathBuf::from("artifacts/verified_dataset.jsonl")
}

fn resolve_vocab_path() -> PathBuf {
    if let Some(arg) = std::env::args().nth(2) {
        return PathBuf::from(arg);
    }
    if let Ok(env_path) = std::env::var("NS_VOCAB_BIN") {
        let trimmed = env_path.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    for candidate in [
        "artifacts/rwkv7_g1_01b_isa/vocab.bin",
        "artifacts/rwkv7_g1_01b_base/vocab.bin",
    ] {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return path;
        }
    }
    PathBuf::from("artifacts/rwkv7_g1_01b_isa/vocab.bin")
}

fn print_row(name: &str, stats: &ScenarioStats, ms_per_tok: Option<f64>) {
    let m = &stats.meter;
    let s = &stats.slot_meter;
    let denom = m.steps.max(1) as f64;

    let avg_j = (m.json as f64) / denom;
    let avg_i = (m.isa as f64) / denom;
    let ratio = (m.json as f64) / (m.isa.max(1) as f64);

    let avg_slot = (s.isa as f64) / denom;
    let ratio_slot = (m.json as f64) / (s.isa.max(1) as f64);

    let avg_jc = (m.json_ctl as f64) / denom;
    let avg_ic = (m.isa_ctl as f64) / denom;
    let ratio_c = (m.json_ctl as f64) / (m.isa_ctl.max(1) as f64);

    let mut sorted = stats.isa_steps_per_action.clone();
    sorted.sort_unstable();
    let p50 = percentile(&sorted, 50);
    let p95 = percentile(&sorted, 95);

    let status = if stats.roundtrip_errors == 0 && stats.grammar_errors == 0 {
        ""
    } else {
        "  <-- ERRO"
    };

    if let Some(ms) = ms_per_tok {
        let saved_ms = (avg_j - avg_slot) * ms;
        println!(
            "{name:<20} | {:>5} | {:>5.1}/{:>5.1} ({:>4.2}x) | {:>4.1} ({:>4.2}x) | {:>5.1}/{:>4.1} ({:>4.2}x) | {:>5.1}% | {:>2}/{:>2} | {:>+6.1} ms{status}",
            m.steps,
            avg_j,
            avg_i,
            ratio,
            avg_slot,
            ratio_slot,
            avg_jc,
            avg_ic,
            ratio_c,
            stats.ptr_hit_pct(),
            p50,
            p95,
            -saved_ms,
        );
    } else {
        println!(
            "{name:<20} | {:>5} | {:>5.1}/{:>5.1} ({:>4.2}x) | {:>4.1} ({:>4.2}x) | {:>5.1}/{:>4.1} ({:>4.2}x) | {:>5.1}% | {:>2}/{:>2}{status}",
            m.steps,
            avg_j,
            avg_i,
            ratio,
            avg_slot,
            ratio_slot,
            avg_jc,
            avg_ic,
            ratio_c,
            stats.ptr_hit_pct(),
            p50,
            p95,
        );
    }
}

fn run(
    dataset_path: &Path,
    vocab_path: &Path,
    ms_per_tok: Option<f64>,
) -> Result<bool, Box<dyn std::error::Error>> {
    let trie = load_vocab_trie(vocab_path, 0)?;
    let count_real_tokens = |text: &str| -> usize { trie.greedy_tokenize(text.as_bytes()).len() };

    let file = File::open(dataset_path)?;
    let reader = BufReader::new(file);

    let mut by_scenario: BTreeMap<String, ScenarioStats> = BTreeMap::new();
    let mut total = ScenarioStats::default();
    let mut logged_errors = 0_usize;

    for (line_no, line_res) in reader.lines().enumerate() {
        let line = line_res?;
        if line.trim().is_empty() {
            continue;
        }
        let row: DatasetRow = serde_json::from_str(&line)?;
        let bank = build_bank_from_prompt(&row.prompt);

        let isa_action = Action::from(&row.action);
        let syms = encode(&isa_action, &bank);

        let rt_ok = match decode_one(&syms, &bank) {
            Ok((decoded, used)) => used == syms.len() && LlmAction::from(decoded) == *action,
            Err(_) => false,
        };
        if !rt_ok {
            acc.roundtrip_errors += 1;
        }

        let mut ptr_hits = 0_usize;
        let mut ptr_misses = 0_usize;
        let mut native_lits = 0_usize;
        let mut grammar_ok = true;

        for (idx, sym) in syms.iter().enumerate() {
            match expect(&syms[..idx]) {
                Ok(Expect::Op) if matches!(sym, Sym::Op(_)) => {}
                Ok(Expect::Num) if matches!(sym, Sym::Num(_)) => {}
                Ok(Expect::Lit) if matches!(sym, Sym::Lit(_)) => native_lits += 1,
                Ok(Expect::Ref) => match sym {
                    Sym::Ptr(_) => ptr_hits += 1,
                    Sym::Lit(_) => ptr_misses += 1,
                    _ => grammar_ok = false,
                },
                Ok(Expect::RefOrEnd) => match sym {
                    Sym::Ptr(_) => ptr_hits += 1,
                    Sym::Lit(_) => ptr_misses += 1,
                    Sym::End => {}
                    _ => grammar_ok = false,
                },
                _ => grammar_ok = false,
            }
        }
        if !matches!(expect(&syms), Ok(Expect::Op)) {
            grammar_ok = false;
        }

        if (!rt_ok || !grammar_ok) && logged_errors < MAX_LOGGED_ERRORS {
            eprintln!(
                "invariante falhou na linha {} (cenário '{}'): round_trip={rt_ok}, grammar={grammar_ok}",
                line_no + 1,
                row.scenario
            );
            logged_errors += 1;
        }

        let slot_action = as_slot_action(&row.action);
        let mut single = CostMeter::default();
        single.add(&row.action, &bank, &count_real_tokens);
        let isa_step_cost = single.isa;

        let entry = by_scenario.entry(row.scenario).or_default();
        for acc in [entry, &mut total] {
            acc.meter.add(&row.action, &bank, &count_real_tokens);
            acc.slot_meter.add(&slot_action, &bank, &count_real_tokens);
            acc.isa_steps_per_action.push(isa_step_cost);
            acc.ptr_hits += ptr_hits;
            acc.ptr_misses += ptr_misses;
            acc.native_lits += native_lits;
            if !rt_ok {
                acc.roundtrip_errors += 1;
            }
            if !grammar_ok {
                acc.grammar_errors += 1;
            }
        }
    }

    let line_width = if ms_per_tok.is_some() { 114 } else { 102 };

    println!();
    println!("Compressão REAL da ISA com TokTrie RWKV-7 World (65.536 tokens)");
    println!(
        "dataset: {} | vocab: {}",
        dataset_path.display(),
        vocab_path.display()
    );
    if let Some(ms) = ms_per_tok {
        println!(
            "{:<20} | {:>5} | {:^19} | {:^12} | {:^18} | {:>6} | {:>5} | {:>9}",
            "Cenário",
            "Steps",
            "JSON/ISA (x)",
            "@slot (x)",
            "CTL json/isa (x)",
            "Ptr%",
            "p50/95",
            format!("@{ms:.2}ms")
        );
    } else {
        println!(
            "{:<20} | {:>5} | {:^19} | {:^12} | {:^18} | {:>6} | {:>5}",
            "Cenário", "Steps", "JSON/ISA (x)", "@slot (x)", "CTL json/isa (x)", "Ptr%", "p50/95"
        );
    }
    println!("{}", "-".repeat(line_width));
    for (name, stats) in &by_scenario {
        print_row(name, stats, ms_per_tok);
    }
    println!("{}", "-".repeat(line_width));
    print_row("TOTAL", &total, ms_per_tok);

    let saved_isa_pct = if total.meter.json == 0 {
        0.0
    } else {
        100.0 * (1.0 - (total.meter.isa as f64) / (total.meter.json as f64))
    };
    let saved_slot_pct = if total.meter.json == 0 {
        0.0
    } else {
        100.0 * (1.0 - (total.slot_meter.isa as f64) / (total.meter.json as f64))
    };

    println!();
    println!(
        "Referências (Ref): {} Ptr hits | {} Lit fallbacks ({:.1}% acerto) | Literais nativos (Lit): {}",
        total.ptr_hits,
        total.ptr_misses,
        total.ptr_hit_pct(),
        total.native_lits,
    );
    println!(
        "Economia de passos: {:.1}% (ISA texto livre) | {:.1}% (ISA @slot) | Invariantes: round-trip err={} | grammar err={}",
        saved_isa_pct, saved_slot_pct, total.roundtrip_errors, total.grammar_errors,
    );

    Ok(total.roundtrip_errors == 0 && total.grammar_errors == 0)
}

fn main() -> ExitCode {
    let dataset_path = resolve_dataset_path();
    let vocab_path = resolve_vocab_path();
    let ms_per_tok = std::env::var("NS_MS_PER_TOK")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0);

    match run(&dataset_path, &vocab_path, ms_per_tok) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => {
            eprintln!("invariantes do codec ISA falharam em uma ou mais amostras");
            ExitCode::FAILURE
        }
        Err(err) => {
            eprintln!(
                "erro ao executar isa_compression (dataset='{}', vocab='{}'): {err}",
                dataset_path.display(),
                vocab_path.display()
            );
            ExitCode::FAILURE
        }
    }
}
