use std::sync::atomic::{AtomicUsize, Ordering};

use ns_core::action::LlmAction;
use ns_core::error::LlmPortError;
use ns_core::fact::{Fact, FactId, Message};
use ns_core::ports::ConstrainedLlm;
use ns_engine::{AuditOutcome, Orchestrator};
use ns_memory::RedbMemory;
use ns_verifier::DefaultSymbolicVerifier;

struct ScriptedMock {
    actions: Vec<LlmAction>,
    cursor: AtomicUsize,
}

impl ScriptedMock {
    fn new(actions: Vec<LlmAction>) -> Self {
        Self {
            actions,
            cursor: AtomicUsize::new(0),
        }
    }
}

impl ConstrainedLlm for ScriptedMock {
    async fn next_action(&self, _history: &[Message]) -> Result<LlmAction, LlmPortError> {
        let idx = self.cursor.fetch_add(1, Ordering::Relaxed);
        self.actions
            .get(idx)
            .cloned()
            .ok_or_else(|| LlmPortError::Backend("scripted mock exhausted".to_owned()))
    }
}

#[tokio::test]
async fn blocks_degenerate_o_and_nonsense_answers_then_accepts_grounded_repair()
-> Result<(), Box<dyn std::error::Error>> {
    let mem = RedbMemory::open_in_memory()?;
    mem.upsert_fact(&Fact {
        id: FactId::new("fact-inedito-9842"),
        subject: "plano_enterprise".to_owned(),
        relation: "preco_mensal".to_owned(),
        object: "749.50".to_owned(),
        source_ref: "test#grounding".to_owned(),
    })?;

    let llm = ScriptedMock::new(vec![
        LlmAction::QueryFact {
            entity: "plano_enterprise".to_owned(),
            relation: "preco_mensal".to_owned(),
        },
        // 1) Furo antigo: texto "O" com FactId válido -> deve ser bloqueado!
        LlmAction::ProposeAnswer {
            text: "O".to_owned(),
            cited_facts: vec![FactId::new("fact-inedito-9842")],
        },
        // 2) Texto sem sentido sem o valor real -> deve ser bloqueado!
        LlmAction::ProposeAnswer {
            text: "O, entre as definições.".to_owned(),
            cited_facts: vec![FactId::new("fact-inedito-9842")],
        },
        // 3) Reparo usando "@slot" (zero head 65k) -> expande e aprova!
        LlmAction::ProposeAnswer {
            text: "@slot".to_owned(),
            cited_facts: vec![FactId::new("fact-inedito-9842")],
        },
    ]);

    let orch = Orchestrator::new(llm, mem, DefaultSymbolicVerifier::new(), 6)
        .with_require_citations(true);

    let outcome = orch
        .answer("Qual o(a) preco_mensal de plano_enterprise?")
        .await?;

    assert_eq!(
        outcome.answer().text(),
        "O valor de preco_mensal de plano_enterprise é 749.50."
    );
    assert_eq!(outcome.audit().repair_count(), 2);
    assert!(matches!(
        outcome.audit().steps()[1].outcome,
        AuditOutcome::AnswerContentRejected { .. }
    ));
    assert!(matches!(
        outcome.audit().steps()[2].outcome,
        AuditOutcome::AnswerContentRejected { .. }
    ));
    Ok(())
}

#[tokio::test]
async fn slot_template_renders_exact_eval_math_and_rejects_wrong_number()
-> Result<(), Box<dyn std::error::Error>> {
    let mem = RedbMemory::open_in_memory()?;
    mem.upsert_fact(&Fact {
        id: FactId::new("fact-inedito-9842"),
        subject: "plano_enterprise".to_owned(),
        relation: "preco_mensal".to_owned(),
        object: "749.50".to_owned(),
        source_ref: "test#math".to_owned(),
    })?;

    let llm = ScriptedMock::new(vec![
        LlmAction::QueryFact {
            entity: "plano_enterprise".to_owned(),
            relation: "preco_mensal".to_owned(),
        },
        LlmAction::EvalMath {
            expression: "749.50 * 4".to_owned(),
        },
        // Número errado mesmo citando o fato -> bloqueado!
        LlmAction::ProposeAnswer {
            text: "O total é 999.00.".to_owned(),
            cited_facts: vec![FactId::new("fact-inedito-9842")],
        },
        // Template com slots explícitos -> preenchido deterministicamente!
        LlmAction::ProposeAnswer {
            text: "Total para 4x {{relation}} ({{subject}}): {{math}}.".to_owned(),
            cited_facts: vec![FactId::new("fact-inedito-9842")],
        },
    ]);

    let orch = Orchestrator::new(llm, mem, DefaultSymbolicVerifier::new(), 6)
        .with_require_citations(true);

    let outcome = orch
        .answer("Calcule 4 vezes preco_mensal de plano_enterprise.")
        .await?;

    assert_eq!(
        outcome.answer().text(),
        "Total para 4x preco_mensal (plano_enterprise): 2998.00."
    );
    assert_eq!(outcome.audit().repair_count(), 1);
    Ok(())
}
