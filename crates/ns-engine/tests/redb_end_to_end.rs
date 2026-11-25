use ns_core::prelude::*;
use ns_engine::{AuditOutcome, Orchestrator};
use ns_llm::MockLlm;
use ns_memory::RedbMemory;
use rust_decimal::Decimal;

struct NoopVerifier;

impl SymbolicVerifier for NoopVerifier {
    fn eval_decimal(&self, expression: &str) -> Result<Decimal, VerifierPortError> {
        Err(VerifierPortError::Syntax(format!(
            "not implemented in noop verifier: {expression}"
        )))
    }

    fn check_smt(&self, _smtlib2_script: &str) -> Result<LogicProof, VerifierPortError> {
        Ok(LogicProof::Unknown {
            reason: "noop verifier".to_owned(),
        })
    }

    fn run_sandbox(
        &self,
        _wat_source: &str,
        _expected_stdout: &str,
    ) -> Result<SandboxOutput, VerifierPortError> {
        Err(VerifierPortError::ResourceLimit)
    }
}

#[tokio::test]
async fn orchestrator_blocks_hallucination_and_verifies_multihop_redb_graph()
-> Result<(), Box<dyn std::error::Error>> {
    // Popula redb em memória e aquece o cache em RAM
    let memory = RedbMemory::open_in_memory()?;
    memory.upsert_fact(&Fact {
        id: FactId::from("fact-engine-storage"),
        subject: "ns-memory".to_owned(),
        relation: "supports".to_owned(),
        object: "redb".to_owned(),
        source_ref: "crates/ns-memory/Cargo.toml".to_owned(),
    })?;
    memory.upsert_fact(&Fact {
        id: FactId::from("fact-redb-structure"),
        subject: "redb".to_owned(),
        relation: "data_structure".to_owned(),
        object: "copy-on-write B+tree".to_owned(),
        source_ref: "docs.rs/redb".to_owned(),
    })?;
    assert_eq!(memory.warm_up()?, 2);

    // LLM tenta responder sem consultar o banco, é bloqueado,
    //    executa TraverseGraph em snapshot MVCC no redb e só então passa no gate.
    let llm = MockLlm::from_actions([
        LlmAction::ProposeAnswer {
            text: "Usa LSM-tree.".to_owned(),
            cited_facts: vec![FactId::from("fact-redb-structure")],
        },
        LlmAction::TraverseGraph {
            start_entity: "ns-memory".to_owned(),
            max_hops: 2,
        },
        LlmAction::ProposeAnswer {
            text: "ns-memory suporta redb, baseado em copy-on-write B+tree.".to_owned(),
            cited_facts: vec![
                FactId::from("fact-engine-storage"),
                FactId::from("fact-redb-structure"),
            ],
        },
    ]);

    let orchestrator = Orchestrator::new(llm, memory, NoopVerifier, 5)
        .with_require_citations(true);

    let outcome = orchestrator
        .answer("Qual estrutura de dados usa o backend alternativo do ns-memory?")
        .await?;

    assert_eq!(
        outcome.answer().text(),
        "ns-memory suporta redb, baseado em copy-on-write B+tree."
    );
    assert_eq!(outcome.audit().repair_count(), 1);
    assert_eq!(
        outcome.audit().steps()[0].outcome,
        AuditOutcome::CitationRejected {
            missing: vec![FactId::from("fact-redb-structure")],
        }
    );
    assert_eq!(
        outcome.audit().steps()[1].outcome,
        AuditOutcome::GraphTraversed { count: 2 }
    );
    assert_eq!(
        outcome.audit().steps()[2].outcome,
        AuditOutcome::AnswerVerified
    );

    Ok(())
}
