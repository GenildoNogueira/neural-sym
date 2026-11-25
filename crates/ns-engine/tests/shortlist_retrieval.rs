use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use ns_core::action::LlmAction;
use ns_core::error::LlmPortError;
use ns_core::fact::{Fact, FactId, Message, Role};
use ns_core::ports::ConstrainedLlm;
use ns_engine::Orchestrator;
use ns_memory::IndexedRedbMemory;
use ns_verifier::DefaultSymbolicVerifier;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(prefix: &str) -> Result<Self, std::io::Error> {
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("neurosym-{prefix}-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Mock determinístico que simula o decoder ISA com Peça D:
/// se `CANDIDATES:` já trouxer `fact-preco-000`, responde em 1 passo;
/// caso contrário, executa o fluxo clássico de 3 passos (`QueryFact -> SearchFacts -> ProposeAnswer`).
struct ShortlistAwareMock;

impl ConstrainedLlm for ShortlistAwareMock {
    async fn next_action(&self, history: &[Message]) -> Result<LlmAction, LlmPortError> {
        let has_candidate = history
            .iter()
            .any(|m| m.content.contains("CANDIDATES:") && m.content.contains("fact-preco-000"));
        if has_candidate {
            return Ok(LlmAction::ProposeAnswer {
                text: "O preco_mensal de plano_000 e 49.00.".to_owned(),
                cited_facts: vec![FactId::new("fact-preco-000")],
            });
        }

        let tool_turns = history
            .iter()
            .filter(|m| matches!(m.role, Role::Tool))
            .count();
        match tool_turns {
            0 => Ok(LlmAction::QueryFact {
                entity: "plano 000".to_owned(),
                relation: "valor_desconhecido".to_owned(),
            }),
            1 => Ok(LlmAction::SearchFacts {
                query: "plano_000".to_owned(),
                limit: 10,
            }),
            _ => Ok(LlmAction::ProposeAnswer {
                text: "Encontrado via busca: preco_mensal de plano_000 e 49.00.".to_owned(),
                cited_facts: vec![FactId::new("fact-preco-000")],
            }),
        }
    }
}

#[tokio::test]
async fn shortlist_collapses_search_discovery_from_three_turns_to_one()
-> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDirGuard::new("shortlist")?;
    let db_path = temp.0.join("db.redb");
    let idx_path = temp.0.join("db.idx");

    {
        let mem = IndexedRedbMemory::open(&db_path, &idx_path)?;
        let _ = mem.upsert_fact(&Fact {
            id: FactId::new("fact-preco-000"),
            subject: "plano_000".to_owned(),
            relation: "preco_mensal".to_owned(),
            object: "49.00".to_owned(),
            source_ref: "test#shortlist".to_owned(),
        })?;
    }

    let prompt = "Pesquise na base qual e o dado de preco mensal relacionado a plano_000.";

    // Sem Peça D (shortlist_limit = 0): fluxo clássico de 3 passos
    {
        let mem = IndexedRedbMemory::open(&db_path, &idx_path)?;
        let orch_no_shortlist =
            Orchestrator::new(ShortlistAwareMock, mem, DefaultSymbolicVerifier::new(), 6)
                .with_require_citations(true)
                .with_shortlist_limit(0);

        let baseline = orch_no_shortlist.answer(prompt).await?;
        assert_eq!(baseline.audit().steps().len(), 3);
        assert_eq!(baseline.audit().repair_count(), 0);
        assert_eq!(baseline.audit().shortlist_count(), 0);
    }

    // Com Peça D (shortlist_limit = 12): colapsa para 1 passo com citação verificada
    {
        let mem = IndexedRedbMemory::open(&db_path, &idx_path)?;
        let orch_shortlist =
            Orchestrator::new(ShortlistAwareMock, mem, DefaultSymbolicVerifier::new(), 6)
                .with_require_citations(true)
                .with_shortlist_limit(12);

        let with_shortlist = orch_shortlist.answer(prompt).await?;
        assert_eq!(with_shortlist.audit().steps().len(), 1);
        assert_eq!(with_shortlist.audit().repair_count(), 0);
        assert_eq!(with_shortlist.audit().shortlist_count(), 1);
        assert_eq!(
            with_shortlist.audit().shortlist_facts(),
            &[FactId::new("fact-preco-000")]
        );
        assert_eq!(
            with_shortlist.answer().cited_facts(),
            &[FactId::new("fact-preco-000")]
        );
    }

    Ok(())
}
