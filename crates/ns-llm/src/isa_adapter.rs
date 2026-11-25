//! Ponte entre `ns_core::action::LlmAction` e o codec ISA, mais utilidades para
//! montar o [`Bank`] (prompt + resultados de TOOL) e medir custo JSON vs ISA.

use ns_core::action::LlmAction;
use ns_core::fact::{Fact, FactId};

use crate::isa::{Action, Bank, encode, isa_cost};

impl From<&LlmAction> for Action {
    fn from(a: &LlmAction) -> Self {
        match a {
            LlmAction::QueryFact { entity, relation } => Action::QueryFact {
                entity: entity.clone(),
                relation: relation.clone(),
            },
            LlmAction::SearchFacts { query, limit } => Action::SearchFacts {
                query: query.clone(),
                limit: *limit,
            },
            LlmAction::SearchContext { query, limit } => Action::SearchContext {
                query: query.clone(),
                limit: *limit,
            },
            LlmAction::ReadSpans { span_ids } => Action::ReadSpans {
                span_ids: span_ids.clone(),
            },
            LlmAction::TraverseGraph {
                start_entity,
                max_hops,
            } => Action::TraverseGraph {
                start_entity: start_entity.clone(),
                max_hops: *max_hops,
            },
            LlmAction::EvalMath { expression } => Action::EvalMath {
                expression: expression.clone(),
            },
            LlmAction::CheckLogic { smtlib2_script } => Action::CheckLogic {
                smtlib2_script: smtlib2_script.clone(),
            },
            LlmAction::RunWasmCheck {
                wasm_wat_source,
                expected_stdout,
            } => Action::RunWasmCheck {
                wasm_wat_source: wasm_wat_source.clone(),
                expected_stdout: expected_stdout.clone(),
            },
            LlmAction::ProposeAnswer { text, cited_facts } => Action::ProposeAnswer {
                text: text.clone(),
                cited_facts: cited_facts
                    .iter()
                    .map(|id| id.as_str().to_owned())
                    .collect(),
            },
        }
    }
}

impl From<Action> for LlmAction {
    fn from(a: Action) -> Self {
        match a {
            Action::QueryFact { entity, relation } => LlmAction::QueryFact { entity, relation },
            Action::SearchFacts { query, limit } => LlmAction::SearchFacts { query, limit },
            Action::SearchContext { query, limit } => LlmAction::SearchContext { query, limit },
            Action::ReadSpans { span_ids } => LlmAction::ReadSpans { span_ids },
            Action::TraverseGraph {
                start_entity,
                max_hops,
            } => LlmAction::TraverseGraph {
                start_entity,
                max_hops,
            },
            Action::EvalMath { expression } => LlmAction::EvalMath { expression },
            Action::CheckLogic { smtlib2_script } => LlmAction::CheckLogic { smtlib2_script },
            Action::RunWasmCheck {
                wasm_wat_source,
                expected_stdout,
            } => LlmAction::RunWasmCheck {
                wasm_wat_source,
                expected_stdout,
            },
            Action::ProposeAnswer { text, cited_facts } => LlmAction::ProposeAnswer {
                text,
                cited_facts: cited_facts.into_iter().map(FactId::from).collect(),
            },
        }
    }
}

/// Entidades que o modelo pode "apontar" no prompt: cada palavra sem pontuação nas pontas.
pub fn bank_add_text(bank: &mut Bank, text: &str) {
    for w in text.split_whitespace() {
        let w = w.trim_matches(|c: char| {
            matches!(
                c,
                '.' | ',' | '?' | '!' | ';' | ':' | '(' | ')' | '"' | '\''
            )
        });
        if !w.is_empty() {
            bank.intern(w);
        }
    }
}

/// Entidades vindas de um resultado de TOOL (fato). Indexa também as formas com
/// `_` trocado por espaço, que é como os prompts em linguagem natural as escrevem.
pub fn bank_add_fact(bank: &mut Bank, f: &Fact) {
    for s in [
        f.id.as_str(),
        f.subject.as_str(),
        f.relation.as_str(),
        f.object.as_str(),
    ] {
        bank.intern(s);
    }
    bank.intern(&f.subject.replace('_', " "));
    bank.intern(&f.relation.replace('_', " "));
}

/// Acumulador de custo para pendurar no `dataset_gen` (uma chamada por ação emitida).
#[derive(Debug, Default, Clone, Copy)]
pub struct CostMeter {
    pub steps: usize,
    pub json: usize,
    pub isa: usize,
    /// Mesmas medidas com o texto livre de `ProposeAnswer` zerado (só o "plano de controle").
    pub json_ctl: usize,
    pub isa_ctl: usize,
}

impl CostMeter {
    pub fn add(&mut self, action: &LlmAction, bank: &Bank, count: &dyn Fn(&str) -> usize) {
        let a = Action::from(action);
        self.steps += 1;
        self.json += count(&serde_json::to_string(action).unwrap_or_default());
        self.isa += isa_cost(&encode(&a, bank), count);

        let stripped = match action {
            LlmAction::ProposeAnswer { cited_facts, .. } => LlmAction::ProposeAnswer {
                text: String::new(),
                cited_facts: cited_facts.clone(),
            },
            other => other.clone(),
        };
        self.json_ctl += count(&serde_json::to_string(&stripped).unwrap_or_default());
        self.isa_ctl += isa_cost(&encode(&Action::from(&stripped), bank), count);
    }

    pub fn merge(&mut self, o: &CostMeter) {
        self.steps += o.steps;
        self.json += o.json;
        self.isa += o.isa;
        self.json_ctl += o.json_ctl;
        self.isa_ctl += o.isa_ctl;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::{approx_tokens, canonical_json, decode_one};

    fn samples() -> Vec<LlmAction> {
        let weird = "a \"q\" \\ \n ação 日本".to_owned();
        vec![
            LlmAction::QueryFact {
                entity: "plano_001".into(),
                relation: weird.clone(),
            },
            LlmAction::SearchFacts {
                query: "x".into(),
                limit: 10,
            },
            LlmAction::SearchContext {
                query: weird.clone(),
                limit: 3,
            },
            LlmAction::ReadSpans {
                span_ids: vec!["cl_02".into(), "cl_47".into()],
            },
            LlmAction::ReadSpans { span_ids: vec![] },
            LlmAction::TraverseGraph {
                start_entity: "p".into(),
                max_hops: 2,
            },
            LlmAction::EvalMath {
                expression: "(1.5 * 2) / 3".into(),
            },
            LlmAction::CheckLogic {
                smtlib2_script: "(assert (= 1 1))\n".into(),
            },
            LlmAction::RunWasmCheck {
                wasm_wat_source: "(module)".into(),
                expected_stdout: weird.clone(),
            },
            LlmAction::ProposeAnswer {
                text: weird,
                cited_facts: vec![FactId::from("f1"), FactId::from("f2")],
            },
        ]
    }

    #[test]
    fn isa_roundtrip_preserves_every_llm_action() {
        let mut bank = Bank::new();
        for s in ["plano_001", "cl_02", "cl_47", "f1"] {
            bank.intern(s);
        }
        for a in samples() {
            let isa: Action = (&a).into();
            let syms = encode(&isa, &bank);
            let (back, used) = decode_one(&syms, &bank).expect("decode");
            assert_eq!(used, syms.len());
            assert_eq!(LlmAction::from(back), a);
        }
    }

    #[test]
    fn canonical_json_matches_serde_json_exactly() {
        for a in samples() {
            let mine = canonical_json(&Action::from(&a));
            let real = serde_json::to_string(&a).expect("serde");
            assert_eq!(mine, real);
            assert_eq!(serde_json::from_str::<LlmAction>(&mine).expect("parse"), a);
        }
    }

    #[test]
    fn meter_counts_fewer_isa_tokens_with_a_bank() {
        let a = LlmAction::ReadSpans {
            span_ids: vec!["cl_02".into(), "cl_47".into()],
        };
        let mut bank = Bank::new();
        bank.intern("cl_02");
        bank.intern("cl_47");
        let mut m = CostMeter::default();
        m.add(&a, &bank, &approx_tokens);
        assert!(m.isa < m.json);
        assert_eq!(m.isa, 4);
    }
}
