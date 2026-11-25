//! cargo run --release -p ns-llm --example isa_measure
//! Replica os cenários do `dataset_gen.rs` (sem reparos) com os tipos reais e mede JSON vs ISA.
//! O contador de tokens ainda é o proxy `approx_tokens`; troque por um tokenizer real.

use ns_core::action::LlmAction;
use ns_core::fact::{Fact, FactId};
use ns_llm::isa::{Bank, approx_tokens};
use ns_llm::isa_adapter::{CostMeter, bank_add_fact, bank_add_text};

fn catalog(n: usize) -> Vec<Fact> {
    let mut v = Vec::new();
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

/// Acompanha a trajetória: o banco cresce com o prompt e com cada resultado de TOOL.
struct Traj {
    bank: Bank,
    meter: CostMeter,
}

impl Traj {
    fn new(prompt: &str) -> Self {
        let mut bank = Bank::new();
        bank_add_text(&mut bank, prompt);
        Self {
            bank,
            meter: CostMeter::default(),
        }
    }
    fn emit(&mut self, a: LlmAction) {
        self.meter.add(&a, &self.bank, &approx_tokens);
    }
    fn tool_result(&mut self, facts: &[&Fact]) {
        for f in facts {
            bank_add_fact(&mut self.bank, f);
        }
    }
}

fn row(name: &str, n: usize, m: &CostMeter) {
    let r = |a: usize, b: usize| a as f64 / b.max(1) as f64;
    let n = n.max(1) as f64;
    println!(
        "{name:<18}| {:>4} | {:>6.1} | {:>6.1} | {:>5.2}x | {:>6.1} | {:>6.1} | {:>5.2}x",
        n as usize,
        m.json as f64 / n,
        m.isa as f64 / n,
        r(m.json, m.isa),
        m.json_ctl as f64 / n,
        m.isa_ctl as f64 / n,
        r(m.json_ctl, m.isa_ctl),
    );
}

fn main() {
    let n = 80;
    let facts = catalog(n);
    let subjects: std::collections::HashSet<&str> =
        facts.iter().map(|f| f.subject.as_str()).collect();
    let by_subject = |s: &str| facts.iter().filter(|f| f.subject == s).collect::<Vec<_>>();

    let mut sc: [(&str, usize, CostMeter); 4] = [
        ("direct_lookup", 0, CostMeter::default()),
        ("search_discovery", 0, CostMeter::default()),
        ("math_exact", 0, CostMeter::default()),
        ("graph_traversal", 0, CostMeter::default()),
    ];

    for (idx, f) in facts.iter().enumerate() {
        // 1. direct_lookup
        let mut t = Traj::new(&format!("Qual o(a) {} de {}?", f.relation, f.subject));
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
        sc[0].1 += 1;
        sc[0].2.merge(&t.meter);

        // 2. search_discovery: o prompt só traz as formas com espaço
        let mut t = Traj::new(&format!(
            "Pesquise na base qual e o dado de {} relacionado a {}.",
            f.relation.replace('_', " "),
            f.subject.replace('_', " ")
        ));
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
        sc[1].1 += 1;
        sc[1].2.merge(&t.meter);

        // 4. math_exact (só fatos numéricos)
        if f.object.parse::<f64>().is_ok() {
            let m = 2 + idx % 11;
            let mut t = Traj::new(&format!(
                "Calcule {m} vezes {} de {} com precisao exata.",
                f.relation, f.subject
            ));
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
            sc[2].1 += 1;
            sc[2].2.merge(&t.meter);
        }

        // 5. graph_traversal
        if subjects.contains(f.object.as_str()) && f.object != f.subject {
            let mut reached = by_subject(&f.subject);
            reached.extend(by_subject(&f.object));
            let mut t = Traj::new(&format!(
                "Percorra o grafo a partir de {} em 2 saltos e resuma as relacoes.",
                f.subject
            ));
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
            sc[3].1 += 1;
            sc[3].2.merge(&t.meter);
        }
    }

    println!("Tokens por trajetória (proxy; JSON = serde_json real do LlmAction)");
    println!(
        "{:<18}| {:>4} | {:^23} | {:^23}",
        "cenário", "traj", "TOTAL json / isa / razão", "CONTROLE (sem texto livre)"
    );
    println!("{}", "-".repeat(85));
    let (mut tn, mut tm) = (0, CostMeter::default());
    for (name, c, m) in &sc {
        row(name, *c, m);
        tn += c;
        tm.merge(m);
    }
    println!("{}", "-".repeat(85));
    row("TOTAL", tn, &tm);
}
