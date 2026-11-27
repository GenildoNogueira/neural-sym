//! cargo run --release -p ns-llm --features custom-rwkv --example isa_measure
//! (também compila sem `--features custom-rwkv`, usando `approx_tokens` como proxy)
//!
//! Replica os cenários do `dataset_gen.rs` (incluindo `@slot` e shortlist de 1 passo) com os
//! tipos reais de `LlmAction`, mede JSON vs. ISA e valida os invariantes do codec:
//! - round-trip exato `decode_one(encode(a, bank), bank) == a`;
//! - gramática `expect()` nunca rejeita o próximo símbolo válido e fecha em `Expect::Op`.
//!
//! Variáveis úteis:
//! - `NS_ISA_CATALOG_N`: número de entidades sintéticas no catálogo (padrão: 80).
//! - `NS_VOCAB_BIN`: caminho explícito para `vocab.bin` (com `--features custom-rwkv`, tenta
//!   automaticamente `artifacts/rwkv7_g1_01b_isa/vocab.bin` e `artifacts/rwkv7_g1_01b_base/vocab.bin`).

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::ExitCode;

use ns_core::action::LlmAction;
use ns_core::fact::{Fact, FactId};
use ns_llm::isa::{Action, Bank, Expect, Sym, approx_tokens, decode_one, encode, expect};
use ns_llm::isa_adapter::{CostMeter, bank_add_fact, bank_add_text};

#[derive(Default)]
struct ScenarioAcc {
    trajectories: usize,
    meter: CostMeter,
    ptr_hits: usize,
    ptr_misses: usize,
    native_lits: usize,
    roundtrip_errors: usize,
    grammar_errors: usize,
}

impl ScenarioAcc {
    fn merge(&mut self, other: &Self) {
        self.trajectories += other.trajectories;
        self.meter.merge(&other.meter);
        self.ptr_hits += other.ptr_hits;
        self.ptr_misses += other.ptr_misses;
        self.native_lits += other.native_lits;
        self.roundtrip_errors += other.roundtrip_errors;
        self.grammar_errors += other.grammar_errors;
    }

    fn ptr_hit_pct(&self) -> f64 {
        let total_refs = self.ptr_hits + self.ptr_misses;
        if total_refs == 0 {
            100.0
        } else {
            100.0 * (self.ptr_hits as f64) / (total_refs as f64)
        }
    }
}

fn catalog(n: usize) -> Vec<Fact> {
    let mut v = Vec::with_capacity(n.saturating_mul(3));
    for i in 0..n {
        let subject = format!("plano_{i:03}");
        let next = format!("plano_{:03}", (i + 1) % n.max(1));
        let reais = 49_u64 + (i as u64 * 15) % 450;
        let cents = (i as u64 * 10) % 100;
        let mk = |id: String, rel: &str, obj: String, src: &str| Fact {
            id: FactId::new(id),
            subject: subject.clone(),
            relation: rel.to_owned(),
            object: obj,
            source_ref: src.to_owned(),
        };
        v.push(mk(
            format!("fact-preco-{i:03}"),
            "preco_mensal",
            format!("{reais}.{cents:02}"),
            "synthetic#catalogo",
        ));
        v.push(mk(
            format!("fact-taxa-{i:03}"),
            "taxa_servico",
            format!("0.{:02}", 5 + (i % 20)),
            "synthetic#catalogo",
        ));
        v.push(mk(
            format!("fact-upgrade-{i:03}"),
            "upgrade_para",
            next,
            "synthetic#grafo",
        ));
    }
    v
}

fn verify_and_classify(action: &LlmAction, bank: &Bank, acc: &mut ScenarioAcc) {
    let isa_action = Action::from(action);
    let syms = encode(&isa_action, bank);

    let rt_ok = match decode_one(&syms, bank) {
        Ok((decoded, used)) => used == syms.len() && LlmAction::from(decoded) == *action,
        Err(_) => false,
    };
    if !rt_ok {
        acc.roundtrip_errors += 1;
    }

    for (idx, sym) in syms.iter().enumerate() {
        match expect(&syms[..idx]) {
            Ok(Expect::Op) if matches!(sym, Sym::Op(_)) => {}
            Ok(Expect::Num) if matches!(sym, Sym::Num(_)) => {}
            Ok(Expect::Lit) if matches!(sym, Sym::Lit(_)) => {
                acc.native_lits += 1;
            }
            Ok(Expect::Ref) => match sym {
                Sym::Ptr(_) => acc.ptr_hits += 1,
                Sym::Lit(_) => acc.ptr_misses += 1,
                _ => acc.grammar_errors += 1,
            },
            Ok(Expect::RefOrEnd) => match sym {
                Sym::Ptr(_) => acc.ptr_hits += 1,
                Sym::Lit(_) => acc.ptr_misses += 1,
                Sym::End => {}
                _ => acc.grammar_errors += 1,
            },
            _ => {
                acc.grammar_errors += 1;
            }
        }
    }

    if !matches!(expect(&syms), Ok(Expect::Op)) {
        acc.grammar_errors += 1;
    }
}

/// Acompanha uma trajetória completa: o banco cresce com o prompt e com cada resultado de TOOL.
struct Traj<'a, F: Fn(&str) -> usize> {
    bank: Bank,
    acc: ScenarioAcc,
    count_tokens: &'a F,
}

impl<'a, F: Fn(&str) -> usize> Traj<'a, F> {
    fn new(prompt: &str, count_tokens: &'a F) -> Self {
        let mut bank = Bank::new();
        bank_add_text(&mut bank, prompt);
        Self {
            bank,
            acc: ScenarioAcc {
                trajectories: 1,
                ..ScenarioAcc::default()
            },
            count_tokens,
        }
    }

    fn emit(&mut self, action: LlmAction) {
        verify_and_classify(&action, &self.bank, &mut self.acc);
        self.acc.meter.add(&action, &self.bank, self.count_tokens);
    }

    fn tool_result(&mut self, facts: &[&Fact]) {
        for f in facts {
            bank_add_fact(&mut self.bank, f);
        }
    }
}

fn print_row(name: &str, a: &ScenarioAcc) {
    let ratio = |num: usize, den: usize| (num as f64) / (den.max(1) as f64);
    let n = a.trajectories.max(1) as f64;
    let m = &a.meter;
    let status = if a.roundtrip_errors == 0 && a.grammar_errors == 0 {
        ""
    } else {
        "  <-- ERRO"
    };

    println!(
        "{name:<18} | {:>4} | {:>6.1} | {:>6.1} | {:>5.2}x | {:>6.1} | {:>6.1} | {:>5.2}x | {:>5.1}%{status}",
        a.trajectories,
        (m.json as f64) / n,
        (m.isa as f64) / n,
        ratio(m.json, m.isa),
        (m.json_ctl as f64) / n,
        (m.isa_ctl as f64) / n,
        ratio(m.json_ctl, m.isa_ctl),
        a.ptr_hit_pct(),
    );
}

#[cfg(feature = "custom-rwkv")]
fn resolve_vocab_path() -> Option<PathBuf> {
    if let Some(arg) = std::env::args().nth(1) {
        return Some(PathBuf::from(arg));
    }
    if let Ok(env_path) = std::env::var("NS_VOCAB_BIN") {
        let trimmed = env_path.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    for candidate in [
        "artifacts/rwkv7_g1_01b_isa/vocab.bin",
        "artifacts/rwkv7_g1_01b_base/vocab.bin",
    ] {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn run_measure<F: Fn(&str) -> usize>(n: usize, tok_label: &str, count_tokens: F) -> ExitCode {
    let facts = catalog(n);
    let subjects: HashSet<&str> = facts.iter().map(|f| f.subject.as_str()).collect();
    let by_subject = |s: &str| facts.iter().filter(|f| f.subject == s).collect::<Vec<_>>();

    let mut scenarios: [(&str, ScenarioAcc); 6] = [
        ("direct_lookup", ScenarioAcc::default()),
        ("direct_slot", ScenarioAcc::default()),
        ("shortlist_1step", ScenarioAcc::default()),
        ("search_discovery", ScenarioAcc::default()),
        ("math_exact", ScenarioAcc::default()),
        ("graph_traversal", ScenarioAcc::default()),
    ];

    for (idx, f) in facts.iter().enumerate() {
        // 1. direct_lookup (texto livre na resposta)
        {
            let mut t = Traj::new(
                &format!("Qual o(a) {} de {}?", f.relation, f.subject),
                &count_tokens,
            );
            t.emit(LlmAction::QueryFact {
                entity: f.subject.clone(),
                relation: f.relation.clone(),
            });
            t.tool_result(&[f]);
            t.emit(LlmAction::ProposeAnswer {
                text: format!(
                    "O valor de {} para {} e {}.",
                    f.relation, f.subject, f.object
                ),
                cited_facts: vec![f.id.clone()],
            });
            scenarios[0].1.merge(&t.acc);
        }

        // 2. direct_slot (modo atual: <ANSWER> "" #ptr <END>, sem head 65k)
        {
            let mut t = Traj::new(
                &format!("Qual o(a) {} de {}?", f.relation, f.subject),
                &count_tokens,
            );
            t.emit(LlmAction::QueryFact {
                entity: f.subject.clone(),
                relation: f.relation.clone(),
            });
            t.tool_result(&[f]);
            t.emit(LlmAction::ProposeAnswer {
                text: String::new(),
                cited_facts: vec![f.id.clone()],
            });
            scenarios[1].1.merge(&t.acc);
        }

        // 3. shortlist_1step (Peça D: fato já vem em CANDIDATES antes do passo 0 + @slot)
        {
            let mut t = Traj::new(
                &format!("Qual o(a) {} de {}?", f.relation, f.subject),
                &count_tokens,
            );
            t.tool_result(&[f]);
            t.emit(LlmAction::ProposeAnswer {
                text: String::new(),
                cited_facts: vec![f.id.clone()],
            });
            scenarios[2].1.merge(&t.acc);
        }

        // 4. search_discovery (prompt só traz formas com espaço -> testa fallback/busca)
        {
            let mut t = Traj::new(
                &format!(
                    "Pesquise na base qual e o dado de {} relacionado a {}.",
                    f.relation.replace('_', " "),
                    f.subject.replace('_', " ")
                ),
                &count_tokens,
            );
            t.emit(LlmAction::QueryFact {
                entity: f.subject.replace('_', " "),
                relation: "valor_desconhecido".into(),
            });
            t.emit(LlmAction::SearchFacts {
                query: f.subject.clone(),
                limit: 10,
            });
            t.tool_result(&[f]);
            t.emit(LlmAction::ProposeAnswer {
                text: format!(
                    "Encontrado via busca: {} de {} e {}.",
                    f.relation, f.subject, f.object
                ),
                cited_facts: vec![f.id.clone()],
            });
            scenarios[3].1.merge(&t.acc);
        }

        // 5. math_exact (fatos numéricos)
        if f.object.parse::<f64>().is_ok() {
            let m = 2 + idx % 11;
            let mut t = Traj::new(
                &format!(
                    "Calcule {m} vezes {} de {} com precisao exata.",
                    f.relation, f.subject
                ),
                &count_tokens,
            );
            t.emit(LlmAction::QueryFact {
                entity: f.subject.clone(),
                relation: f.relation.clone(),
            });
            t.tool_result(&[f]);
            t.emit(LlmAction::EvalMath {
                expression: format!("{} * {m}", f.object),
            });
            t.emit(LlmAction::ProposeAnswer {
                text: format!(
                    "O total exato para {m} unidades de {} ({}) e 000.00.",
                    f.subject, f.relation
                ),
                cited_facts: vec![f.id.clone()],
            });
            scenarios[4].1.merge(&t.acc);
        }

        // 6. graph_traversal
        if subjects.contains(f.object.as_str()) && f.object != f.subject {
            let mut reached = by_subject(&f.subject);
            reached.extend(by_subject(&f.object));
            let mut t = Traj::new(
                &format!(
                    "Percorra o grafo a partir de {} em 2 saltos e resuma as relacoes.",
                    f.subject
                ),
                &count_tokens,
            );
            t.emit(LlmAction::TraverseGraph {
                start_entity: f.subject.clone(),
                max_hops: 2,
            });
            t.tool_result(&reached);
            t.emit(LlmAction::ProposeAnswer {
                text: format!(
                    "A partir de {}, o grafo alcanca {} fatos em 2 saltos incluindo {} -> {}.",
                    f.subject,
                    reached.len(),
                    f.relation,
                    f.object
                ),
                cited_facts: reached.iter().map(|r| r.id.clone()).collect(),
            });
            scenarios[5].1.merge(&t.acc);
        }
    }

    println!();
    println!("Tokens por trajetória (JSON = serde_json real do LlmAction vs. ISA)");
    println!(
        "tokenizador: {tok_label} | entidades no catálogo: {n} ({} fatos)",
        facts.len()
    );
    println!(
        "{:<18} | {:>4} | {:^23} | {:^23} | {:>6}",
        "Cenário", "traj", "TOTAL json / isa / x", "CONTROLE json / isa / x", "Ptr%"
    );
    println!("{}", "-".repeat(92));

    let mut total = ScenarioAcc::default();
    for (name, acc) in &scenarios {
        print_row(name, acc);
        total.merge(acc);
    }
    println!("{}", "-".repeat(92));
    print_row("TOTAL", &total);

    let saved_pct = if total.meter.json == 0 {
        0.0
    } else {
        100.0 * (1.0 - (total.meter.isa as f64) / (total.meter.json as f64))
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
        "Economia total de passos: {:.1}% | Invariantes: round-trip err={} | grammar err={}",
        saved_pct, total.roundtrip_errors, total.grammar_errors,
    );

    if total.roundtrip_errors == 0 && total.grammar_errors == 0 {
        ExitCode::SUCCESS
    } else {
        eprintln!("invariantes do codec ISA falharam");
        ExitCode::FAILURE
    }
}

fn main() -> ExitCode {
    let n = std::env::var("NS_ISA_CATALOG_N")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(80);

    #[cfg(feature = "custom-rwkv")]
    if let Some(path) = resolve_vocab_path() {
        match ns_llm::load_vocab_trie(&path, 0) {
            Ok(trie) => {
                let label = format!("TokTrie RWKV-7 World ({})", path.display());
                return run_measure(n, &label, |text| {
                    trie.greedy_tokenize(text.as_bytes()).len()
                });
            }
            Err(err) => {
                eprintln!(
                    "aviso: não foi possível carregar '{}': {err}; usando approx_tokens",
                    path.display()
                );
            }
        }
    }

    run_measure(
        n,
        "approx_tokens (proxy — compile com --features custom-rwkv e NS_VOCAB_BIN para real)",
        approx_tokens,
    )
}
