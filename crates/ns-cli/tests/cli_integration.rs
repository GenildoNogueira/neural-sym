use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use clap::Parser;
use ns_cli::{Cli, run_with_writer};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(prefix: &str) -> Result<Self, std::io::Error> {
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "neurosym-{prefix}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

async fn exec_cli(args: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    let cli = Cli::try_parse_from(args)?;
    let mut buf = Vec::new();
    run_with_writer(cli, &mut buf).await?;
    Ok(String::from_utf8(buf)?)
}

async fn verify_full_cli_flow(
    backend_name: &str,
    db_file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDirGuard::new(backend_name)?;
    let db_path = temp.file(db_file);
    let db_str = db_path.to_string_lossy().into_owned();

    // 1. Cadastra fato via CLI
    let add_out = exec_cli(&[
        "neurosym",
        "--backend",
        backend_name,
        "--db-path",
        &db_str,
        "fact",
        "add",
        "--id",
        "fact-preco",
        "--subject",
        "plano_pro",
        "--relation",
        "preco_mensal",
        "--object",
        "199.90",
        "--source",
        "tabela#2026",
    ])
    .await?;
    assert!(add_out.contains("saved fact-preco"));

    // 2. Consulta fato via CLI
    let get_out = exec_cli(&[
        "neurosym",
        "--backend",
        backend_name,
        "--db-path",
        &db_str,
        "fact",
        "get",
        "--entity",
        "PLANO_PRO",
        "--relation",
        "PRECO_MENSAL",
    ])
    .await?;
    assert!(get_out.contains("199.90"));

    // 3. Roda o Orchestrator via CLI com tentativa de alucinação + reparo + conta exata
    let actions = r#"[
        {"action":"propose_answer","text":"Chute sem consultar.","cited_facts":["fact-inventado"]},
        {"action":"query_fact","entity":"plano_pro","relation":"preco_mensal"},
        {"action":"eval_math","expression":"199.90 * 3"},
        {"action":"propose_answer","text":"Trimestre custa R$ 599.7.","cited_facts":["fact-preco"]}
    ]"#;

    let ask_out = exec_cli(&[
        "neurosym",
        "--backend",
        backend_name,
        "--db-path",
        &db_str,
        "ask",
        "--prompt",
        "Quanto custa um trimestre do plano_pro?",
        "--actions-json",
        actions,
        "--require-citations",
        "--show-audit",
    ])
    .await?;

    assert!(ask_out.contains("Answer: Trimestre custa R$ 599.7."));
    assert!(ask_out.contains("Citations: fact-preco"));
    assert!(ask_out.contains("Repairs: 1"));
    assert!(ask_out.contains("CitationRejected"));
    assert!(ask_out.contains("MathEvaluated { value: \"599.7\" }"));

    Ok(())
}

#[tokio::test]
async fn cli_flow_works_with_redb_backend() -> Result<(), Box<dyn std::error::Error>> {
    verify_full_cli_flow("redb", "cli_test.redb").await
}

#[cfg(feature = "tantivy-search")]
#[tokio::test]
async fn cli_flow_works_with_indexed_redb_backend() -> Result<(), Box<dyn std::error::Error>> {
    verify_full_cli_flow("indexed-redb", "cli_indexed.redb").await
}

#[tokio::test]
async fn cli_eval_computes_exact_decimal_expression() -> Result<(), Box<dyn std::error::Error>> {
    let out = exec_cli(&["neurosym", "eval", "0.1 + 0.2"]).await?;
    assert_eq!(out.trim(), "0.3");
    Ok(())
}

#[cfg(not(feature = "custom-rwkv"))]
#[tokio::test]
async fn cli_rwkv_dir_reports_missing_feature_cleanly() {
    let result = exec_cli(&[
        "neurosym",
        "ask",
        "--prompt",
        "Teste",
        "--rwkv-dir",
        "artifacts/rwkv_copy_8k",
    ])
    .await;

    let err_msg = match result {
        Ok(_) => String::new(),
        Err(err) => err.to_string(),
    };
    assert!(err_msg.contains("custom-rwkv"));
}
