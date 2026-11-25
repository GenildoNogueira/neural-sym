use ns_core::prelude::*;
use ns_engine::{AuditOutcome, Orchestrator};
use ns_llm::MockLlm;
use ns_memory::{RedbMemory, SessionDocumentStore};
use ns_verifier::DefaultSymbolicVerifier;

#[tokio::test]
async fn compares_distant_clauses_side_by_side_and_verifies_span_citations()
-> Result<(), Box<dyn std::error::Error>> {
    let mut contract = String::new();
    contract.push_str("Cláusula 1 - Objeto\nFornecimento de controladores embarcados.\n\n");
    contract.push_str(
        "Cláusula 2 - Prazo Máximo\nO prazo máximo contratual é de 15 dias corridos (vide Cláusula 47).\n\n",
    );
    for i in 3..=46 {
        contract.push_str(&format!(
            "Cláusula {i} - Termo Técnico {i}\nEspecificação auxiliar de hardware versão {i}.\n\n"
        ));
    }
    contract.push_str(
        "Cláusula 47 - Entrega Especial\nPara lotes importados, o prazo de entrega será de 30 dias corridos.\n",
    );

    let doc_store = SessionDocumentStore::from_text(&contract)?;
    let memory = RedbMemory::open_in_memory()?;
    let verifier = DefaultSymbolicVerifier::new();

    let llm = MockLlm::from_actions([
        // 1. Tenta citar cl_47 sem tê-la lido: deve ser bloqueado pelo typestate Verified
        LlmAction::ProposeAnswer {
            text: "Chute sem abrir as cláusulas.".to_owned(),
            cited_facts: vec![FactId::from("cl_47")],
        },
        // 2. Busca no documento longo em RAM
        LlmAction::SearchContext {
            query: "prazo entrega".to_owned(),
            limit: 4,
        },
        // 3. Lê Cláusula 2 e Cláusula 47 lado a lado em uma única janela compacta
        LlmAction::ReadSpans {
            span_ids: vec!["cl_02".to_owned(), "cl_47".to_owned()],
        },
        // 4. Calcula a diferença exata via rust_decimal
        LlmAction::EvalMath {
            expression: "30 - 15".to_owned(),
        },
        // 5. Responde citando ambas as cláusulas efetivamente lidas
        LlmAction::ProposeAnswer {
            text: "A Cláusula 47 excede em 15 dias o prazo máximo de 15 dias fixado na Cláusula 2."
                .to_owned(),
            cited_facts: vec![FactId::from("cl_02"), FactId::from("cl_47")],
        },
    ]);

    let orchestrator =
        Orchestrator::new(llm, memory, verifier, 8).with_require_citations(true);
    let outcome = orchestrator
        .answer_with_context(
            "Existe conflito entre o prazo da cláusula 2 e a cláusula 47?",
            &doc_store,
        )
        .await?;

    assert_eq!(outcome.audit().repair_count(), 1);
    assert_eq!(
        outcome.answer().cited_facts(),
        &[FactId::from("cl_02"), FactId::from("cl_47")]
    );
    assert!(matches!(
        outcome.audit().steps()[0].outcome,
        AuditOutcome::CitationRejected { .. }
    ));
    assert!(matches!(
        outcome.audit().steps()[2].outcome,
        AuditOutcome::SpansRead { .. }
    ));
    assert_eq!(
        outcome.audit().steps()[4].outcome,
        AuditOutcome::AnswerVerified
    );

    Ok(())
}
