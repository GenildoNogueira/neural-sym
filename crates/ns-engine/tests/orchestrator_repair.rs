use ns_core::prelude::*;
use ns_engine::{AuditOutcome, EngineError, Orchestrator};
use ns_llm::MockLlm;
use rust_decimal::Decimal;

/// Memória determinística em RAM exclusiva para testes do orquestrador.
struct InMemoryFacts {
    facts: Vec<Fact>,
}

impl InMemoryFacts {
    fn new(facts: Vec<Fact>) -> Self {
        Self { facts }
    }
}

impl DeterministicMemory for InMemoryFacts {
    async fn lookup_fact(
        &self,
        entity: &str,
        relation: &str,
    ) -> Result<Option<Fact>, MemoryPortError> {
        Ok(self
            .facts
            .iter()
            .find(|f| {
                f.subject.eq_ignore_ascii_case(entity) && f.relation.eq_ignore_ascii_case(relation)
            })
            .cloned())
    }

    async fn search_facts(&self, query: &str, limit: u8) -> Result<Vec<Fact>, MemoryPortError> {
        let needle = query.trim().to_lowercase();
        if needle.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        Ok(self
            .facts
            .iter()
            .filter(|f| {
                f.subject.to_lowercase().contains(&needle)
                    || f.relation.to_lowercase().contains(&needle)
                    || f.object.to_lowercase().contains(&needle)
            })
            .take(usize::from(limit))
            .cloned()
            .collect())
    }

    async fn traverse(
        &self,
        start_entity: &str,
        max_hops: u8,
    ) -> Result<Vec<Fact>, MemoryPortError> {
        if max_hops == 0 {
            return Ok(Vec::new());
        }
        Ok(self
            .facts
            .iter()
            .filter(|f| f.subject.eq_ignore_ascii_case(start_entity))
            .cloned()
            .collect())
    }

    async fn verify_fact_exists(&self, id: &FactId) -> Result<bool, MemoryPortError> {
        Ok(self.facts.iter().any(|f| &f.id == id))
    }
}

/// Validador simbólico simples para testes unitários/integração do loop de reparo.
struct StubVerifier;

impl SymbolicVerifier for StubVerifier {
    fn eval_decimal(&self, expression: &str) -> Result<Decimal, VerifierPortError> {
        match expression.trim() {
            "10 / 0" => Err(VerifierPortError::Arithmetic),
            "10 / 2" => Ok(Decimal::from(5)),
            other => Err(VerifierPortError::Syntax(format!(
                "unsupported test expression: {other}"
            ))),
        }
    }

    fn check_smt(&self, _smtlib2_script: &str) -> Result<LogicProof, VerifierPortError> {
        Ok(LogicProof::Satisfiable {
            model_summary: "x = 1".to_owned(),
        })
    }

    fn run_sandbox(
        &self,
        _wat_source: &str,
        expected_stdout: &str,
    ) -> Result<SandboxOutput, VerifierPortError> {
        Ok(SandboxOutput {
            stdout: expected_stdout.to_owned(),
            fuel_consumed: 42,
            passed: true,
        })
    }
}

fn sample_memory() -> InMemoryFacts {
    InMemoryFacts::new(vec![Fact {
        id: FactId::from("fact-rust-edition"),
        subject: "rust".to_owned(),
        relation: "edition".to_owned(),
        object: "2024".to_owned(),
        source_ref: "cargo.toml#workspace".to_owned(),
    }])
}

#[tokio::test]
async fn blocks_hallucinated_fact_and_repairs_after_memory_lookup()
-> Result<(), Box<dyn std::error::Error>> {
    let llm = MockLlm::from_actions([
        // 1. Tentativa de alucinação: cita um FactId inventado sem consultar a memória
        LlmAction::ProposeAnswer {
            text: "A edição é 2018.".to_owned(),
            cited_facts: vec![FactId::from("fact-inventado-999")],
        },
        // 2. Após feedback de reparo, consulta a memória determinística
        LlmAction::QueryFact {
            entity: "rust".to_owned(),
            relation: "edition".to_owned(),
        },
        // 3. Propõe a resposta real citando o FactId efetivamente recuperado
        LlmAction::ProposeAnswer {
            text: "A edição configurada no workspace é 2024.".to_owned(),
            cited_facts: vec![FactId::from("fact-rust-edition")],
        },
    ]);

    let orchestrator = Orchestrator::new(llm, sample_memory(), StubVerifier, 5);
    let outcome = orchestrator
        .answer("Qual a edição do Rust no workspace?")
        .await?;

    assert_eq!(
        outcome.answer().text(),
        "A edição configurada no workspace é 2024."
    );
    assert_eq!(
        outcome.answer().cited_facts(),
        &[FactId::from("fact-rust-edition")]
    );
    assert_eq!(outcome.audit().repair_count(), 1);
    assert_eq!(outcome.audit().steps().len(), 3);
    assert_eq!(
        outcome.audit().steps()[0].outcome,
        AuditOutcome::CitationRejected {
            missing: vec![FactId::from("fact-inventado-999")],
        }
    );
    assert_eq!(
        outcome.audit().steps()[1].outcome,
        AuditOutcome::FactFound(FactId::from("fact-rust-edition"))
    );
    assert_eq!(
        outcome.audit().steps()[2].outcome,
        AuditOutcome::AnswerVerified
    );

    Ok(())
}

#[tokio::test]
async fn delivers_repair_feedback_in_history_and_fails_if_llm_keeps_hallucinating() {
    let llm = MockLlm::from_actions([
        LlmAction::ProposeAnswer {
            text: "Chute 1".to_owned(),
            cited_facts: vec![FactId::from("fact-fake-1")],
        },
        LlmAction::ProposeAnswer {
            text: "Chute 2".to_owned(),
            cited_facts: vec![FactId::from("fact-fake-2")],
        },
        LlmAction::ProposeAnswer {
            text: "Chute 3".to_owned(),
            cited_facts: vec![FactId::from("fact-fake-3")],
        },
    ]);

    // Envolve em referência compartilhada para inspecionar o histórico após o erro
    let llm_ref = &llm;
    let orchestrator = Orchestrator::new(RefLlm(llm_ref), sample_memory(), StubVerifier, 3);
    let result = orchestrator.answer("Qual a edição?").await;

    assert!(matches!(
        result,
        Err(EngineError::MaxStepsExceeded {
            max_steps: 3,
            repair_attempts: 3,
        })
    ));

    let histories = llm.recorded_histories();
    assert_eq!(histories.len(), 3);
    // Na segunda chamada, o histórico já precisa conter o aviso de reparo sobre fact-fake-1
    let second_call_messages = &histories[1];
    assert!(second_call_messages.iter().any(|msg| {
        msg.role == Role::System
            && msg.content.contains("REPAIR REQUIRED")
            && msg.content.contains("fact-fake-1")
    }));
}

#[tokio::test]
async fn blocks_uncited_answer_when_require_citations_is_enabled()
-> Result<(), Box<dyn std::error::Error>> {
    let llm = MockLlm::from_actions([
        // Tenta responder "de cabeça" sem nenhuma citação
        LlmAction::ProposeAnswer {
            text: "Acho que é 2024.".to_owned(),
            cited_facts: Vec::new(),
        },
        LlmAction::QueryFact {
            entity: "rust".to_owned(),
            relation: "edition".to_owned(),
        },
        LlmAction::ProposeAnswer {
            text: "Confirmado na memória: edição 2024.".to_owned(),
            cited_facts: vec![FactId::from("fact-rust-edition")],
        },
    ]);

    let orchestrator =
        Orchestrator::new(llm, sample_memory(), StubVerifier, 4).with_require_citations(true);
    let outcome = orchestrator.answer("Qual a edição?").await?;

    assert_eq!(outcome.audit().repair_count(), 1);
    assert_eq!(
        outcome.audit().steps()[0].outcome,
        AuditOutcome::MissingRequiredCitations
    );
    assert_eq!(
        outcome.audit().steps()[2].outcome,
        AuditOutcome::AnswerVerified
    );

    Ok(())
}

#[tokio::test]
async fn repairs_invalid_math_expression_using_symbolic_verifier_feedback()
-> Result<(), Box<dyn std::error::Error>> {
    let llm = MockLlm::from_actions([
        LlmAction::EvalMath {
            expression: "10 / 0".to_owned(),
        },
        LlmAction::EvalMath {
            expression: "10 / 2".to_owned(),
        },
        LlmAction::ProposeAnswer {
            text: "O resultado exato é 5.".to_owned(),
            cited_facts: Vec::new(),
        },
    ]);

    let orchestrator = Orchestrator::new(llm, sample_memory(), StubVerifier, 5);
    let outcome = orchestrator.answer("Quanto é 10 dividido por 2?").await?;

    assert_eq!(outcome.answer().text(), "O resultado exato é 5.");
    assert_eq!(outcome.audit().repair_count(), 1);
    assert!(matches!(
        outcome.audit().steps()[0].outcome,
        AuditOutcome::VerifierRejected { .. }
    ));
    assert_eq!(
        outcome.audit().steps()[1].outcome,
        AuditOutcome::MathEvaluated {
            value: "5".to_owned(),
        }
    );

    Ok(())
}

/// Pequeno wrapper para emprestar `&MockLlm` sem mover a posse no teste de inspeção.
struct RefLlm<'a>(&'a MockLlm);

impl ConstrainedLlm for RefLlm<'_> {
    async fn next_action(&self, history: &[Message]) -> Result<LlmAction, LlmPortError> {
        self.0.next_action(history).await
    }
}
