pub mod backend;
pub mod config;
pub mod dataset_gen;

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Args as ClapArgs, Parser, Subcommand};
use ns_core::prelude::*;
use ns_engine::{Orchestrator, VerifiedOutcome};
use ns_llm::{MockLlm, action_schema_json};
use ns_verifier::DefaultSymbolicVerifier;

use crate::backend::{LlmBackend, MemoryBackend};
use crate::config::{AppConfig, StorageBackend};

#[derive(Parser, Debug, Clone)]
#[command(
    name = "neurosym",
    version,
    about = "Orquestrador neuro-simbolico com memoria deterministica (redb ou indexed-redb)"
)]
pub struct Cli {
    /// Engine de armazenamento deterministico (redb ou indexed-redb)
    #[arg(long, value_enum, default_value_t = StorageBackend::IndexedRedb, global = true)]
    pub backend: StorageBackend,

    /// Caminho do arquivo do banco (padrao: neurosym.redb)
    #[arg(long, global = true)]
    pub db_path: Option<PathBuf>,

    /// Diretorio do indice Tantivy (usado com --backend indexed-redb; padrao: <db_path>.tantivy)
    #[arg(long, global = true)]
    pub index_dir: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Gerencia fatos, buscas e travessias na memoria deterministica
    Fact {
        #[command(subcommand)]
        action: FactCommand,
    },
    Dataset {
        #[command(subcommand)]
        action: DatasetCommand,
    },
    /// Avalia uma expressao matematica usando aritmetica decimal exata
    Eval {
        /// Expressao aritmetica (ex: "(199.90 * 3) * (1 + 0.15)")
        expression: String,
    },
    /// Valida regras logicas em SMT-LIB2 usando o solver Z3 (requer --features z3-solver)
    Logic {
        /// Script SMT-LIB2 inline
        #[arg(long, conflicts_with = "file", required_unless_present = "file")]
        script: Option<String>,
        /// Caminho para arquivo .smt2
        #[arg(long, conflicts_with = "script", required_unless_present = "script")]
        file: Option<PathBuf>,
    },
    /// Imprime o JSON Schema de LlmAction usado no constrained decoding
    Schema,
    /// Executa o Orchestrator (memoria + validador simbolico + loop de reparo)
    Ask(AskArgs),
    /// Inicia conversa interativa continua mantendo o estado recorrente O(1) do RWKV-7 vivo em RAM
    Chat(ChatArgs),
}

#[derive(Subcommand, Debug, Clone)]
pub enum FactCommand {
    /// Cadastra ou atualiza um fato (subject, relation -> object)
    Add {
        #[arg(long)]
        id: String,
        #[arg(long)]
        subject: String,
        #[arg(long)]
        relation: String,
        #[arg(long)]
        object: String,
        #[arg(long, default_value = "cli#manual")]
        source: String,
    },
    /// Consulta um fato exato por entidade e relacao
    Get {
        #[arg(long)]
        entity: String,
        #[arg(long)]
        relation: String,
    },
    /// Busca fatos por termo parcial (BM25 + FST no indexed-redb; substring no redb)
    Search {
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: u8,
    },
    /// Percorre o grafo de fatos em largura (BFS) a partir de uma entidade
    Traverse {
        #[arg(long)]
        start: String,
        #[arg(long, default_value_t = 2)]
        max_hops: u8,
    },
    /// Reconstroi o indice Tantivy a partir do banco redb existente (apenas indexed-redb)
    RebuildIndex,
}

#[derive(Subcommand, Debug, Clone)]
pub enum DatasetCommand {
    /// Executa trajetorias multi-turno no Orchestrator e exporta apenas as Verified em JSONL
    Generate {
        /// Arquivo de saida .jsonl
        #[arg(short, long, default_value = "artifacts/verified_dataset.jsonl")]
        output: PathBuf,

        /// Opcional: popula N entidades sinteticas (3 fatos cada) antes de gerar
        #[arg(long, default_value_t = 0)]
        seed_synthetic: usize,

        /// Desativa cenarios de erro + recuperacao REPAIR REQUIRED
        #[arg(long, default_value_t = false)]
        no_repairs: bool,
    },
}

#[derive(ClapArgs, Debug, Clone)]
pub struct AskArgs {
    /// Pergunta enviada ao orquestrador
    #[arg(short, long)]
    pub prompt: String,

    /// Diretorio do modelo RWKV-7 ISA (config.json + vocab.bin + model.safetensors)
    #[arg(
        long,
        conflicts_with_all = ["actions_file", "actions_json"],
        required_unless_present_any = ["actions_file", "actions_json"]
    )]
    pub rwkv_dir: Option<PathBuf>,

    /// Quantidade de fatos candidatos pre-injetados antes do passo 0 (0 = desativado; Peca D)
    #[arg(long, default_value_t = 0)]
    pub shortlist_k: u8,

    /// Limite de tokens gerados por acao (usado com --rwkv-dir)
    #[arg(long, default_value_t = 512)]
    pub max_tokens: u32,

    /// Arquivo JSON contendo um array de LlmAction para simular o LLM
    #[arg(
        long,
        conflicts_with_all = ["rwkv_dir", "actions_json"],
        required_unless_present_any = ["rwkv_dir", "actions_json"]
    )]
    pub actions_file: Option<PathBuf>,

    /// Array JSON inline de LlmAction para testes rapidos sem arquivo
    #[arg(
        long,
        conflicts_with_all = ["rwkv_dir", "actions_file"],
        required_unless_present_any = ["rwkv_dir", "actions_file"]
    )]
    pub actions_json: Option<String>,

    /// Limite maximo de passos do loop Generate -> Verify -> Repair
    #[arg(long, default_value_t = 6)]
    pub max_steps: u8,

    /// Bloqueia respostas finais sem ao menos um FactId consultado na sessao
    #[arg(long, default_value_t = false)]
    pub require_citations: bool,

    /// Exibe todos os passos da trilha de auditoria no terminal
    #[arg(long, default_value_t = false)]
    pub show_audit: bool,

    /// Arquivo de texto longo (ex: contrato/manual) indexado em RAM para consulta via ReadSpans
    #[arg(long)]
    pub context_file: Option<PathBuf>,
}

#[derive(ClapArgs, Debug, Clone)]
pub struct ChatArgs {
    /// Diretorio do modelo RWKV-7 (config.json + vocab.bin + model.safetensors)
    #[arg(long)]
    pub rwkv_dir: PathBuf,

    /// Instrucao inicial opcional absorvida uma unica vez no estado recorrente
    #[arg(long)]
    pub system: Option<String>,

    /// Temperatura de amostragem (0.0 = greedy deterministico; padrao: 0.7)
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,

    /// Nucleus sampling top-p (0.05..=1.0; padrao: 0.85)
    #[arg(long, default_value_t = 0.85)]
    pub top_p: f32,

    /// Penalidade de presenca para evitar repeticoes (padrao: 0.3)
    #[arg(long, default_value_t = 0.3)]
    pub presence_penalty: f32,

    /// Limite maximo de tokens gerados por turno
    #[arg(long, default_value_t = 256)]
    pub max_tokens: u32,

    /// Exibe estatisticas de velocidade, RAM e esparsidade ReLU² ao final de cada turno
    #[arg(long, default_value_t = true)]
    pub stats: bool,

    /// Pula o raciocinio do RWKV-7 G1 (fecha um bloco <think></think> vazio apos "Assistant:")
    #[arg(long, default_value_t = false)]
    pub no_think: bool,
}

/// Executa o comando CLI escrevendo em `stdout`.
///
/// # Errors
///
/// Propaga erros contextualizados de banco, parsing JSON, verificação ou orquestração.
pub async fn run(cli: Cli) -> Result<()> {
    let mut stdout = io::stdout();
    run_with_writer(cli, &mut stdout).await
}

/// Executa o comando CLI escrevendo em um buffer configurável (facilita testes de integração).
///
/// # Errors
///
/// Propaga erros contextualizados de banco, parsing JSON, verificação ou orquestração.
pub async fn run_with_writer<W: Write>(cli: Cli, out: &mut W) -> Result<()> {
    let cfg = AppConfig::resolve(cli.backend, cli.db_path, cli.index_dir);

    match cli.command {
        Commands::Fact { action } => handle_fact(&cfg, action, out).await,
        Commands::Dataset { action } => handle_dataset(&cfg, action, out).await,
        Commands::Eval { expression } => handle_eval(&expression, out),
        Commands::Logic { script, file } => handle_logic(script.as_deref(), file.as_deref(), out),
        Commands::Schema => handle_schema(out),
        Commands::Ask(args) => handle_ask(&cfg, &args, out).await,
        Commands::Chat(args) => {
            let stdin = io::stdin();
            let mut reader = stdin.lock();
            run_chat_with_io(&args, &mut reader, out)
        }
    }
}

async fn handle_fact<W: Write>(cfg: &AppConfig, action: FactCommand, out: &mut W) -> Result<()> {
    let memory = MemoryBackend::open(cfg.backend, &cfg.db_path, &cfg.index_dir).await?;

    match action {
        FactCommand::Add {
            id,
            subject,
            relation,
            object,
            source,
        } => {
            let fact = Fact {
                id: FactId::new(id),
                subject,
                relation,
                object,
                source_ref: source,
            };
            let stored = memory.upsert_fact(&fact).await?;
            writeln!(
                out,
                "[{}] saved {} :: ({}, {}) -> {} [{}]",
                cfg.backend.as_str(),
                stored.id,
                stored.subject,
                stored.relation,
                stored.object,
                stored.source_ref
            )
            .context("failed to write output")?;
        }
        FactCommand::Get { entity, relation } => {
            let found = memory
                .lookup_fact(&entity, &relation)
                .await
                .with_context(|| format!("failed to lookup fact ({entity}, {relation})"))?;

            match found {
                Some(fact) => {
                    writeln!(
                        out,
                        "{} :: ({}, {}) -> {} [{}]",
                        fact.id, fact.subject, fact.relation, fact.object, fact.source_ref
                    )
                    .context("failed to write output")?;
                }
                None => {
                    writeln!(out, "not found: ({entity}, {relation})")
                        .context("failed to write output")?;
                }
            }
        }
        FactCommand::Search { query, limit } => {
            let facts = memory
                .search_facts(&query, limit)
                .await
                .with_context(|| format!("failed to search facts for '{query}'"))?;

            if facts.is_empty() {
                writeln!(out, "no facts matched '{query}'").context("failed to write output")?;
            } else {
                for fact in &facts {
                    writeln!(
                        out,
                        "{} :: ({}, {}) -> {} [{}]",
                        fact.id, fact.subject, fact.relation, fact.object, fact.source_ref
                    )
                    .context("failed to write output")?;
                }
            }
        }
        FactCommand::Traverse { start, max_hops } => {
            let facts = memory
                .traverse(&start, max_hops)
                .await
                .with_context(|| format!("failed to traverse graph from '{start}'"))?;

            if facts.is_empty() {
                writeln!(
                    out,
                    "no facts reached from '{start}' within {max_hops} hops"
                )
                .context("failed to write output")?;
            } else {
                for fact in &facts {
                    writeln!(
                        out,
                        "{} :: ({}, {}) -> {} [{}]",
                        fact.id, fact.subject, fact.relation, fact.object, fact.source_ref
                    )
                    .context("failed to write output")?;
                }
            }
        }
        FactCommand::RebuildIndex => match memory.rebuild_index()? {
            Some(count) => {
                writeln!(
                    out,
                    "[{}] rebuilt tantivy index at '{}' with {count} facts",
                    cfg.backend.as_str(),
                    cfg.index_dir.display()
                )
                .context("failed to write rebuild summary")?;
            }
            None => {
                bail!(
                    "rebuild-index is only supported with '--backend indexed-redb' (active: '{}')",
                    cfg.backend.as_str()
                );
            }
        },
    }

    Ok(())
}

async fn handle_dataset<W: Write>(
    cfg: &AppConfig,
    action: DatasetCommand,
    out: &mut W,
) -> Result<()> {
    let memory = MemoryBackend::open(cfg.backend, &cfg.db_path, &cfg.index_dir).await?;

    match action {
        DatasetCommand::Generate {
            output,
            seed_synthetic,
            no_repairs,
        } => {
            let report =
                dataset_gen::generate_verified_jsonl(&memory, &output, seed_synthetic, !no_repairs)
                    .await?;

            writeln!(
                out,
                "[{}] generated {} verified samples ({} repair recoveries) from {}/{} trajectories ({} facts) -> {}",
                cfg.backend.as_str(),
                report.samples_written,
                report.repair_recovery_samples,
                report.trajectories_verified,
                report.trajectories_attempted,
                report.facts_used,
                output.display()
            )
            .context("failed to write dataset report")?;
        }
    }

    Ok(())
}

fn handle_eval<W: Write>(expression: &str, out: &mut W) -> Result<()> {
    let verifier = DefaultSymbolicVerifier::new();
    let value = verifier
        .eval_decimal(expression)
        .with_context(|| format!("failed to evaluate decimal expression '{expression}'"))?;

    writeln!(out, "{value}").context("failed to write output")?;
    Ok(())
}

fn handle_logic<W: Write>(
    inline_script: Option<&str>,
    file: Option<&Path>,
    out: &mut W,
) -> Result<()> {
    let script = match (inline_script, file) {
        (Some(text), None) => text.to_owned(),
        (None, Some(path)) => std::fs::read_to_string(path)
            .with_context(|| format!("failed to read smt-lib2 file '{}'", path.display()))?,
        _ => bail!("provide either --script or --file"),
    };

    let verifier = DefaultSymbolicVerifier::new();
    let proof = verifier
        .check_smt(&script)
        .context("failed to verify smt-lib2 logic script")?;

    match proof {
        LogicProof::Satisfiable { model_summary } => {
            writeln!(out, "SAT\n{model_summary}").context("failed to write logic output")?;
        }
        LogicProof::Unsatisfiable { unsat_core } => {
            writeln!(out, "UNSAT\nCore: {unsat_core}").context("failed to write logic output")?;
        }
        LogicProof::Unknown { reason } => {
            writeln!(out, "UNKNOWN\nReason: {reason}").context("failed to write logic output")?;
        }
    }

    Ok(())
}

fn handle_schema<W: Write>(out: &mut W) -> Result<()> {
    let schema = action_schema_json().context("failed to generate LlmAction JSON schema")?;
    writeln!(out, "{schema}").context("failed to write output")?;
    Ok(())
}

async fn handle_ask<W: Write>(cfg: &AppConfig, args: &AskArgs, out: &mut W) -> Result<()> {
    let llm = build_llm_backend(args)?;
    let memory = MemoryBackend::open(cfg.backend, &cfg.db_path, &cfg.index_dir).await?;
    let cached = memory.warm_up().await?;

    tracing::info!(
        backend = cfg.backend.as_str(),
        db = %cfg.db_path.display(),
        index = %cfg.index_dir.display(),
        cached_facts = cached,
        shortlist_k = args.shortlist_k,
        "deterministic memory ready"
    );

    let verifier = DefaultSymbolicVerifier::new();
    let orchestrator = Orchestrator::new(llm, memory, verifier, args.max_steps)
        .with_require_citations(args.require_citations)
        .with_shortlist_limit(args.shortlist_k);

    let ask_start = Instant::now();
    let outcome = if let Some(ctx_path) = &args.context_file {
        let raw_doc = std::fs::read_to_string(ctx_path)
            .with_context(|| format!("failed to read context document '{}'", ctx_path.display()))?;
        let doc_store = ns_memory::SessionDocumentStore::from_text(&raw_doc)
            .context("failed to index context document in RAM")?;

        tracing::info!(
            path = %ctx_path.display(),
            spans = doc_store.len(),
            "ephemeral session document store ready"
        );

        orchestrator
            .answer_with_context(&args.prompt, &doc_store)
            .await
            .context("orchestrator failed to produce a verified answer")?
    } else {
        orchestrator
            .answer(&args.prompt)
            .await
            .context("orchestrator failed to produce a verified answer")?
    };
    let elapsed_ms = ask_start.elapsed().as_secs_f64() * 1_000.0;

    render_verified_outcome(&outcome, cfg.backend, args.show_audit, elapsed_ms, out)
}

fn build_llm_backend(args: &AskArgs) -> Result<LlmBackend> {
    if let Some(rwkv_dir) = &args.rwkv_dir {
        return build_rwkv_backend(rwkv_dir, args);
    }

    let actions = load_actions(args)?;
    Ok(LlmBackend::Mock(MockLlm::from_actions(actions)))
}

#[cfg(feature = "custom-rwkv")]
fn build_rwkv_backend(rwkv_dir: &Path, args: &AskArgs) -> Result<LlmBackend> {
    let max_new_tokens = usize::try_from(args.max_tokens).unwrap_or(256);
    let llm = ns_llm::Rwkv7Llm::load_dir(rwkv_dir)
        .with_context(|| {
            format!(
                "failed to load native rwkv-7 model from '{}'",
                rwkv_dir.display()
            )
        })?
        .with_max_new_tokens(max_new_tokens);
    Ok(LlmBackend::Rwkv7(llm))
}

#[cfg(not(feature = "custom-rwkv"))]
fn build_rwkv_backend(rwkv_dir: &Path, _args: &AskArgs) -> Result<LlmBackend> {
    bail!(
        "cannot load '{}': ns-cli was compiled without the 'custom-rwkv' feature. Rebuild with '--features custom-rwkv'",
        rwkv_dir.display()
    );
}

fn load_actions(args: &AskArgs) -> Result<Vec<LlmAction>> {
    let raw_json = match (&args.actions_file, &args.actions_json) {
        (Some(path), None) => std::fs::read_to_string(path)
            .with_context(|| format!("failed to read actions file '{}'", path.display()))?,
        (None, Some(inline)) => inline.clone(),
        _ => bail!("provide --rwkv-dir, --actions-file, or --actions-json"),
    };

    let actions: Vec<LlmAction> = serde_json::from_str(&raw_json)
        .context("failed to parse LLM action sequence as JSON array of LlmAction")?;

    if actions.is_empty() {
        bail!("action sequence must contain at least one LlmAction");
    }

    Ok(actions)
}

fn render_verified_outcome<W: Write>(
    outcome: &VerifiedOutcome,
    backend: StorageBackend,
    show_audit: bool,
    elapsed_ms: f64,
    out: &mut W,
) -> Result<()> {
    writeln!(out, "Answer: {}", outcome.answer().text()).context("failed to write answer")?;

    let citations: Vec<&str> = outcome
        .answer()
        .cited_facts()
        .iter()
        .map(FactId::as_str)
        .collect();
    writeln!(
        out,
        "Citations: {}",
        if citations.is_empty() {
            "(none)".to_owned()
        } else {
            citations.join(", ")
        }
    )
    .context("failed to write citations")?;

    writeln!(
        out,
        "Backend: {} | Steps: {} | Repairs: {} | Shortlist: {} | Latency: {:.1} ms",
        backend.as_str(),
        outcome.audit().steps().len(),
        outcome.audit().repair_count(),
        outcome.audit().shortlist_count(),
        elapsed_ms
    )
    .context("failed to write audit summary")?;

    if show_audit {
        if !outcome.audit().shortlist_facts().is_empty() {
            let sl: Vec<&str> = outcome
                .audit()
                .shortlist_facts()
                .iter()
                .map(FactId::as_str)
                .collect();
            writeln!(out, "Shortlist: {}", sl.join(", "))
                .context("failed to write shortlist summary")?;
        }
        writeln!(out, "Audit Trail:").context("failed to write audit header")?;
        for step in outcome.audit().steps() {
            writeln!(
                out,
                "  #{} {:?} => {:?}",
                step.step_index, step.action, step.outcome
            )
            .context("failed to write audit step")?;
        }
    }

    Ok(())
}

/// Executa o loop interativo `chat` lendo de `reader` e escrevendo em `out` (facilita testes E2E).
///
/// # Errors
///
/// Retorna erro se o carregamento do RWKV-7, a inicialização do estado ou a escrita falharem.
#[cfg(feature = "custom-rwkv")]
pub fn run_chat_with_io<R: BufRead, W: Write>(
    args: &ChatArgs,
    reader: &mut R,
    out: &mut W,
) -> Result<()> {
    let model = ns_llm::Rwkv7Llm::load_dir(&args.rwkv_dir).with_context(|| {
        format!(
            "failed to load native rwkv-7 model from '{}'",
            args.rwkv_dir.display()
        )
    })?;
    let mut session = model
        .start_chat(args.system.as_deref())
        .context("failed to initialize rwkv-7 chat state")?;
    session.set_no_think(args.no_think);

    let sampling = ns_llm::ChatSamplingConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        presence_penalty: args.presence_penalty,
        max_new_tokens: usize::try_from(args.max_tokens).unwrap_or(256).max(1),
    };

    writeln!(
        out,
        "NeuroSym Chat | RWKV-7 Goose (estado matricial O(1)) | Estado fixo: {:.1} KiB",
        (session.state_bytes() as f64) / 1024.0
    )
    .context("failed to write chat banner")?;
    writeln!(
        out,
        "Comandos: /reset (nova conversa), /stats (estado), /exit (sair)\n"
    )
    .context("failed to write chat help")?;

    let mut line = String::new();
    loop {
        write!(out, "Você> ").context("failed to write prompt")?;
        out.flush().context("failed to flush prompt")?;

        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .context("failed to read user input")?;
        if bytes_read == 0 {
            writeln!(out).context("failed to write newline on eof")?;
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match trimmed {
            "/exit" | "/quit" => break,
            "/reset" => {
                session
                    .reset(args.system.as_deref())
                    .context("failed to reset rwkv-7 chat state")?;
                writeln!(out, "[estado recorrente zerado]\n")
                    .context("failed to write reset confirmation")?;
                continue;
            }
            "/stats" => {
                writeln!(
                    out,
                    "[tokens acumulados: {} | estado RAM: {:.1} KiB | FFN ReLU² pulados: {:.1}%]\n",
                    session.total_tokens_seen(),
                    (session.state_bytes() as f64) / 1024.0,
                    session.ffn_sparsity_ratio() * 100.0
                )
                .context("failed to write chat stats")?;
                continue;
            }
            _ => {}
        }

        write!(out, "NeuroSym> ").context("failed to write assistant prefix")?;
        out.flush().context("failed to flush assistant prefix")?;

        let stats = session
            .send_message_streaming(trimmed, sampling, |chunk| {
                let _ = write!(out, "{chunk}");
                let _ = out.flush();
            })
            .context("rwkv-7 chat generation failed")?;

        writeln!(out).context("failed to write end of turn")?;
        if args.stats {
            let sparsity_str = stats.ffn_sparsity_ratio.map_or_else(String::new, |r| {
                format!(" | FFN zeros pulados: {:.1}%", r * 100.0)
            });
            writeln!(
                out,
                "  [{} tok em {:.0} ms | {:.1} tok/s | estado: {:.1} KiB{}]\n",
                stats.generated_tokens,
                stats.elapsed_ms,
                stats.tokens_per_sec(),
                (stats.state_bytes as f64) / 1024.0,
                sparsity_str
            )
            .context("failed to write turn stats")?;
        }
    }

    Ok(())
}

#[cfg(not(feature = "custom-rwkv"))]
pub fn run_chat_with_io<R: BufRead, W: Write>(
    args: &ChatArgs,
    _reader: &mut R,
    _out: &mut W,
) -> Result<()> {
    bail!(
        "cannot start chat with '{}': ns-cli was compiled without the 'custom-rwkv' feature. Rebuild with '--features custom-rwkv'",
        args.rwkv_dir.display()
    );
}
