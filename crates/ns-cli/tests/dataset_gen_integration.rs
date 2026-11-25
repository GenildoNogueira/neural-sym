#![cfg(feature = "tantivy-search")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use clap::Parser;
use ns_cli::{Cli, run_with_writer};
use ns_core::action::LlmAction;
use ns_engine::TrainingSample;

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

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn cli_dataset_generate_exports_only_verified_multiturn_and_repair_recovery_samples()
-> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDirGuard::new("dataset-gen")?;
    let db_path = temp.path("selfplay.redb");
    let idx_path = temp.path("selfplay.tantivy");
    let jsonl_path = temp.path("verified.jsonl");

    let db_str = db_path.to_string_lossy().into_owned();
    let idx_str = idx_path.to_string_lossy().into_owned();
    let out_str = jsonl_path.to_string_lossy().into_owned();

    let cli = Cli::try_parse_from([
        "neurosym",
        "--backend",
        "indexed-redb",
        "--db-path",
        &db_str,
        "--index-dir",
        &idx_str,
        "dataset",
        "generate",
        "--output",
        &out_str,
        "--seed-synthetic",
        "4",
    ])?;

    let mut stdout = Vec::new();
    run_with_writer(cli, &mut stdout).await?;
    let report = String::from_utf8(stdout)?;
    assert!(report.contains("[indexed-redb] generated"));

    let content = std::fs::read_to_string(&jsonl_path)?;
    let samples: Vec<TrainingSample> = content
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<Vec<_>, _>>()?;

    assert!(!samples.is_empty());

    // 1. Deve conter amostras reais de recuperação após REPAIR REQUIRED
    let recoveries: Vec<&TrainingSample> =
        samples.iter().filter(|s| s.is_repair_recovery).collect();
    assert!(!recoveries.is_empty());
    assert!(
        recoveries
            .iter()
            .all(|s| s.prompt.contains("SYSTEM: REPAIR REQUIRED"))
    );

    // 2. Nenhum alvo SFT pode ensinar citação falsa ou divisão por zero
    for sample in &samples {
        assert!(!sample.cited_facts.is_empty());
        match &sample.action {
            LlmAction::ProposeAnswer { cited_facts, .. } => {
                assert!(
                    !cited_facts
                        .iter()
                        .any(|id| id.as_str() == "fact-alucinado-000")
                );
            }
            LlmAction::EvalMath { expression } => {
                assert!(!expression.contains("/ (1 - 1)"));
            }
            _ => {}
        }
    }

    Ok(())
}
