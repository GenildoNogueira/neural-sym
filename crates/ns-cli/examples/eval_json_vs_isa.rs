use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use ns_cli::backend::MemoryBackend;
use ns_cli::config::{AppConfig, StorageBackend};
use ns_engine::Orchestrator;
use ns_llm::Rwkv7Llm;
use ns_llm::rwkv7::DecodeMode;
use ns_verifier::DefaultSymbolicVerifier;

struct EvalTask {
    name: &'static str,
    prompt: &'static str,
    expected_fact_id: &'static str,
}

struct ModeReport {
    label: String,
    total: usize,
    verified: usize,
    citation_hits: usize,
    total_steps: usize,
    total_repairs: usize,
    total_shortlist: usize,
    latencies_ms: Vec<f64>,
}

impl ModeReport {
    fn print(&mut self) {
        self.latencies_ms.sort_by(f64::total_cmp);
        let mean = if self.latencies_ms.is_empty() {
            0.0
        } else {
            self.latencies_ms.iter().sum::<f64>() / (self.latencies_ms.len() as f64)
        };
        let p50 = percentile(&self.latencies_ms, 50);
        let p99 = percentile(&self.latencies_ms, 99);
        let avg_steps = (self.total_steps as f64) / (self.verified.max(1) as f64);

        println!(
            "{:<22} | {:>2}/{:<2} ({:>5.1}%) | {:>2}/{:<2} | {:>5.2} | {:>4} | {:>4} | {:>7.1} ms | {:>7.1} ms | {:>7.1} ms",
            self.label,
            self.verified,
            self.total,
            100.0 * (self.verified as f64) / (self.total.max(1) as f64),
            self.citation_hits,
            self.total,
            avg_steps,
            self.total_repairs,
            self.total_shortlist,
            mean,
            p50,
            p99,
        );
    }
}

fn percentile(sorted: &[f64], pct: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = sorted.len().saturating_sub(1).saturating_mul(pct) / 100;
    sorted[idx]
}

async fn run_mode(
    label: &str,
    model_dir: &Path,
    mode: DecodeMode,
    shortlist_limit: u8,
    app_cfg: &AppConfig,
    tasks: &[EvalTask],
) -> Result<ModeReport> {
    let llm = Rwkv7Llm::load_dir(model_dir)
        .with_context(|| format!("failed to load model from '{}'", model_dir.display()))?
        .with_decode_mode(mode)
        .with_max_new_tokens(128);

    let memory = MemoryBackend::open(app_cfg.backend, &app_cfg.db_path, &app_cfg.index_dir).await?;
    let _ = memory.warm_up().await?;
    let verifier = DefaultSymbolicVerifier::new();

    let orchestrator = Orchestrator::new(llm, memory, verifier, 6)
        .with_require_citations(true)
        .with_shortlist_limit(shortlist_limit);

    let mut report = ModeReport {
        label: label.to_owned(),
        total: tasks.len(),
        verified: 0,
        citation_hits: 0,
        total_steps: 0,
        total_repairs: 0,
        total_shortlist: 0,
        latencies_ms: Vec::with_capacity(tasks.len()),
    };

    for task in tasks {
        let t0 = Instant::now();
        match orchestrator.answer(task.prompt).await {
            Ok(outcome) => {
                let ms = t0.elapsed().as_secs_f64() * 1_000.0;
                report.verified += 1;
                report.total_steps += outcome.audit().steps().len();
                report.total_repairs += outcome.audit().repair_count();
                report.total_shortlist += outcome.audit().shortlist_count();
                report.latencies_ms.push(ms);
                if outcome
                    .answer()
                    .cited_facts()
                    .iter()
                    .any(|id| id.as_str() == task.expected_fact_id)
                {
                    report.citation_hits += 1;
                }
            }
            Err(err) => {
                eprintln!("[{label}] task '{}' falhou: {err:#}", task.name);
            }
        }
    }

    Ok(report)
}

#[tokio::main]
async fn main() -> Result<()> {
    let db_path = std::env::args().nth(1).map_or_else(
        || PathBuf::from("artifacts/eval_inedito.redb"),
        PathBuf::from,
    );
    let json_dir = std::env::args().nth(2).map_or_else(
        || PathBuf::from("artifacts/rwkv7_g1_01b_json"),
        PathBuf::from,
    );
    let isa_dir = std::env::args().nth(3).map_or_else(
        || PathBuf::from("artifacts/rwkv7_g1_01b_isa"),
        PathBuf::from,
    );

    let app_cfg = AppConfig::resolve(StorageBackend::IndexedRedb, Some(db_path), None);

    let tasks = [
        EvalTask {
            name: "direct_inedito",
            prompt: "Qual o(a) preco_mensal de plano_enterprise?",
            expected_fact_id: "fact-inedito-9842",
        },
        EvalTask {
            name: "math_inedito",
            prompt: "Calcule 4 vezes preco_mensal de plano_enterprise com precisao exata.",
            expected_fact_id: "fact-inedito-9842",
        },
        EvalTask {
            name: "search_discovery_inedito",
            prompt: "Pesquise na base qual e o dado de preco mensal relacionado a plano enterprise.",
            expected_fact_id: "fact-inedito-9842",
        },
    ];

    println!();
    println!(
        "Comparativo ponta a ponta | DB: {} | Indice: {}",
        app_cfg.db_path.display(),
        app_cfg.index_dir.display()
    );
    println!(
        "{:<22} | {:<13} | {:<5} | {:<5} | {:<4} | {:<4} | {:>10} | {:>10} | {:>10}",
        "Modo", "Verificadas", "Cit.", "Steps", "Rep.", "SL", "Média", "p50", "p99"
    );
    println!("{}", "-".repeat(105));

    let mut r_json = run_mode(
        "JSON (llguidance)",
        &json_dir,
        DecodeMode::Json,
        0,
        &app_cfg,
        &tasks,
    )
    .await?;
    let mut r_isa = run_mode(
        "ISA (A+B+C)",
        &isa_dir,
        DecodeMode::Isa,
        0,
        &app_cfg,
        &tasks,
    )
    .await?;
    let mut r_isa_d = run_mode(
        "ISA + Shortlist (D)",
        &isa_dir,
        DecodeMode::Isa,
        12,
        &app_cfg,
        &tasks,
    )
    .await?;

    r_json.print();
    r_isa.print();
    r_isa_d.print();

    Ok(())
}
