use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use ns_core::action::LlmAction;
use ns_llm::isa::{Action, Bank, Sym, decode_one, encode};
use ns_llm::isa_adapter::{CostMeter, bank_add_text};
use ns_llm::load_vocab_trie;
use serde::Deserialize;
use serde_json::Value;

const RWKV7_01B_MS_PER_TOK: f64 = 7.52;

#[derive(Debug, Deserialize)]
struct DatasetRow {
    scenario: String,
    prompt: String,
    action: LlmAction,
}

#[derive(Default)]
struct ScenarioStats {
    meter: CostMeter,
    ptr_hits: usize,
    lit_fallbacks: usize,
}

/// Constrói um `Bank` realista extraindo apenas o que já apareceu no `prompt`:
/// 1) todas as strings presentes nos payloads `TOOL: {...}`
/// 2) palavras/identificadores presentes nas linhas `USER: ...` (via `bank_add_text`).
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

fn print_row(name: &str, m: &CostMeter) {
    let denom = m.steps.max(1) as f64;
    let avg_j = m.json as f64 / denom;
    let avg_i = m.isa as f64 / denom;
    let ratio = (m.json as f64) / (m.isa.max(1) as f64);

    let avg_jc = m.json_ctl as f64 / denom;
    let avg_ic = m.isa_ctl as f64 / denom;
    let ratio_c = (m.json_ctl as f64) / (m.isa_ctl.max(1) as f64);

    let saved_ms = (avg_j - avg_i) * RWKV7_01B_MS_PER_TOK;

    println!(
        "{name:<20} | {:>5} | {:>5.1} / {:>5.1} ({:>4.2}x) | {:>5.1} / {:>5.1} ({:>4.2}x) | {:>+7.1} ms/step",
        m.steps, avg_j, avg_i, ratio, avg_jc, avg_ic, ratio_c, -saved_ms
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset_path = std::env::args().nth(1).map_or_else(
        || PathBuf::from("artifacts/verified_dataset.jsonl"),
        PathBuf::from,
    );
    let vocab_path = std::env::args().nth(2).map_or_else(
        || PathBuf::from("artifacts/rwkv7_g1_01b_base/vocab.bin"),
        PathBuf::from,
    );

    let trie = load_vocab_trie(&vocab_path, 0)?;
    let count_real_tokens = |text: &str| -> usize { trie.greedy_tokenize(text.as_bytes()).len() };

    let file = File::open(&dataset_path)?;
    let reader = BufReader::new(file);

    let mut by_scenario: BTreeMap<String, ScenarioStats> = BTreeMap::new();
    let mut total = ScenarioStats::default();

    for line_res in reader.lines() {
        let line = line_res?;
        if line.trim().is_empty() {
            continue;
        }
        let row: DatasetRow = serde_json::from_str(&line)?;
        let bank = build_bank_from_prompt(&row.prompt);

        let isa_action = Action::from(&row.action);
        let syms = encode(&isa_action, &bank);
        let (decoded, used) = decode_one(&syms, &bank)?;
        assert_eq!(LlmAction::from(decoded), row.action, "round-trip diverged");
        assert_eq!(used, syms.len());

        let ptrs = syms.iter().filter(|s| matches!(s, Sym::Ptr(_))).count();
        let lits = syms.iter().filter(|s| matches!(s, Sym::Lit(_))).count();

        let entry = by_scenario.entry(row.scenario).or_default();
        for acc in [entry, &mut total] {
            acc.meter.add(&row.action, &bank, &count_real_tokens);
            acc.ptr_hits += ptrs;
            acc.lit_fallbacks += lits;
        }
    }

    println!();
    println!(
        "Medição REAL com TokTrie RWKV-7 World (65.536 tokens) | Dataset: {}",
        dataset_path.display()
    );
    println!(
        "{:<20} | {:>5} | {:^21} | {:^21} | {:>15}",
        "Cenário", "Steps", "TOTAL json/isa (x)", "CONTROLE json/isa (x)", "Ganho @7.52ms"
    );
    println!("{}", "-".repeat(94));
    for (name, stats) in &by_scenario {
        print_row(name, &stats.meter);
    }
    println!("{}", "-".repeat(94));
    print_row("TOTAL", &total.meter);
    println!();
    println!(
        "Ponteiros resolvidos (Ptr): {} | Literais (Lit, incl. texto livre/expressões): {}",
        total.ptr_hits, total.lit_fallbacks
    );

    Ok(())
}
