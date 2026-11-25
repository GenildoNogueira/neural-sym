#![cfg(feature = "z3-solver")]

use ns_core::prelude::*;
use ns_engine::{AuditOutcome, Orchestrator};
use ns_llm::MockLlm;
use ns_memory::RedbMemory;
use ns_verifier::DefaultSymbolicVerifier;

#[tokio::test]
async fn orchestrator_repairs_broken_smt_and_proves_policy_violation_with_z3()
-> Result<(), Box<dyn std::error::Error>> {
    let memory = RedbMemory::open_in_memory()?;
    memory.upsert_fact(&Fact {
        id: FactId::from("fact-max-desconto"),
        subject: "politica_comercial".to_owned(),
        relation: "desconto_maximo".to_owned(),
        object: "0.30".to_owned(),
        source_ref: "regimento#art-12".to_owned(),
    })?;

    let llm = MockLlm::from_actions([
        // 1. Consulta a regra exata no redb
        LlmAction::QueryFact {
            entity: "politica_comercial".to_owned(),
            relation: "desconto_maximo".to_owned(),
        },
        // 2. Emite SMT-LIB2 quebrado (parêntese faltando) -> barrado sem crash + REPAIR REQUIRED
        LlmAction::CheckLogic {
            smtlib2_script: "(declare-const d Real) (assert (= d 0.45)".to_owned(),
        },
        // 3. Corrige o script SMT-LIB2 -> Z3 prova formalmente UNSAT com unsat_core nomeado
        LlmAction::CheckLogic {
            smtlib2_script: r#"
                (declare-const desconto Real)
                (assert (! (<= desconto 0.30) :named limite_politica))
                (assert (! (= desconto 0.45) :named desconto_solicitado))
            "#
            .to_owned(),
        },
        // 4. Emite resposta verificada apoiada no fato e na prova UNSAT
        LlmAction::ProposeAnswer {
            text: "Desconto de 45% viola a política comercial (máximo de 30%), provado como UNSAT pelo Z3."
                .to_owned(),
            cited_facts: vec![FactId::from("fact-max-desconto")],
        },
    ]);

    let orchestrator = Orchestrator::new(llm, memory, DefaultSymbolicVerifier::new(), 6)
        .with_require_citations(true);

    let outcome = orchestrator
        .answer("Posso aplicar 45% de desconto segundo a política comercial?")
        .await?;

    assert_eq!(outcome.audit().repair_count(), 1);
    assert!(matches!(
        outcome.audit().steps()[1].outcome,
        AuditOutcome::VerifierRejected { .. }
    ));
    assert!(matches!(
        &outcome.audit().steps()[2].outcome,
        AuditOutcome::LogicChecked {
            proof: LogicProof::Unsatisfiable { unsat_core }
        } if unsat_core.contains("limite_politica") && unsat_core.contains("desconto_solicitado")
    ));
    assert_eq!(
        outcome.audit().steps()[3].outcome,
        AuditOutcome::AnswerVerified
    );

    Ok(())
}
