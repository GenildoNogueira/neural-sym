#![cfg(all(feature = "tantivy-search", feature = "custom-rwkv"))]

use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use clap::Parser;
use ns_cli::{ChatArgs, Cli, run_chat_with_io, run_with_writer};
use ns_llm::Rwkv7Config;
use safetensors::Dtype;
use safetensors::tensor::TensorView;

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

async fn exec_cli(args: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    let cli = Cli::try_parse_from(args)?;
    let mut buf = Vec::new();
    run_with_writer(cli, &mut buf).await?;
    Ok(String::from_utf8(buf)?)
}

fn build_minimal_vocab_bin(vocab_size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(vocab_size * 4);
    out.extend_from_slice(&(vocab_size as u32).to_le_bytes());
    out.extend_from_slice(&0_u16.to_le_bytes());

    let mut next_id = 1_usize;
    for b in 0_u8..=255_u8 {
        if next_id >= vocab_size {
            break;
        }
        out.extend_from_slice(&1_u16.to_le_bytes());
        out.push(b);
        next_id += 1;
    }

    let mut idx = 0_usize;
    while next_id < vocab_size {
        let tok = format!("t{idx}");
        let bytes = tok.as_bytes();
        out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(bytes);
        next_id += 1;
        idx += 1;
    }
    out
}

fn write_dummy_rwkv7_dir(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    let cfg = Rwkv7Config {
        arch: "rwkv7".to_owned(),
        vocab_size: 260,
        d_model: 8,
        n_layers: 2,
        head_dim: 4,
        ffn_dim: 16,
        decay_lora_dim: 4,
        a_lora_dim: 4,
        v_lora_dim: 4,
        gate_lora_dim: 4,
        bos_token_id: 0,
        eos_token_id: 0,
        layer_norm_eps: 1e-5,
        group_norm_eps: 64e-5,
        vocab_file: "vocab.bin".to_owned(),
        copy_dim: 4,
        isa_ptr_dim: 0,
        isa_num_vocab: 64,
        copy_max_len: 64,
    };

    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&cfg)?)?;
    std::fs::write(
        dir.join(&cfg.vocab_file),
        build_minimal_vocab_bin(cfg.vocab_size),
    )?;

    let mut bufs: Vec<(String, Vec<usize>, Dtype, Vec<u8>)> = Vec::new();

    // Embeddings quantizados em I8 (inseridos antes de criar a closure `push_f32`)
    let emb_q = vec![10_i8.to_ne_bytes()[0]; cfg.vocab_size * cfg.d_model];
    let mut emb_s = Vec::with_capacity(cfg.vocab_size * 4);
    for _ in 0..cfg.vocab_size {
        emb_s.extend_from_slice(&0.01_f32.to_le_bytes());
    }
    bufs.push((
        "emb.weight.q".to_owned(),
        vec![cfg.vocab_size, cfg.d_model],
        Dtype::I8,
        emb_q,
    ));
    bufs.push((
        "emb.weight.scale".to_owned(),
        vec![cfg.vocab_size],
        Dtype::F32,
        emb_s,
    ));

    let mut push_f32 = |name: &str, shape: Vec<usize>, val: f32| {
        let count: usize = shape.iter().product();
        let mut bytes = Vec::with_capacity(count * 4);
        for _ in 0..count {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bufs.push((name.to_owned(), shape, Dtype::F32, bytes));
    };

    push_f32("ln_in.weight", vec![cfg.d_model], 1.0);
    push_f32("ln_in.bias", vec![cfg.d_model], 0.0);

    for i in 0..cfg.n_layers {
        let bp = format!("blocks.{i}");
        push_f32(&format!("{bp}.ln1.weight"), vec![cfg.d_model], 1.0);
        push_f32(&format!("{bp}.ln1.bias"), vec![cfg.d_model], 0.0);

        for shift in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"] {
            push_f32(&format!("{bp}.tm.{shift}"), vec![cfg.d_model], 0.5);
        }
        for proj in ["receptance", "key", "value", "output"] {
            push_f32(
                &format!("{bp}.tm.{proj}.weight"),
                vec![cfg.d_model, cfg.d_model],
                0.1,
            );
        }

        push_f32(&format!("{bp}.tm.w0"), vec![cfg.d_model], 0.0);
        push_f32(
            &format!("{bp}.tm.w1.weight"),
            vec![cfg.decay_lora_dim, cfg.d_model],
            0.05,
        );
        push_f32(
            &format!("{bp}.tm.w2.weight"),
            vec![cfg.d_model, cfg.decay_lora_dim],
            0.05,
        );

        push_f32(&format!("{bp}.tm.a0"), vec![cfg.d_model], 0.0);
        push_f32(
            &format!("{bp}.tm.a1.weight"),
            vec![cfg.a_lora_dim, cfg.d_model],
            0.05,
        );
        push_f32(
            &format!("{bp}.tm.a2.weight"),
            vec![cfg.d_model, cfg.a_lora_dim],
            0.05,
        );

        if i > 0 {
            push_f32(&format!("{bp}.tm.v0"), vec![cfg.d_model], 0.0);
            push_f32(
                &format!("{bp}.tm.v1.weight"),
                vec![cfg.v_lora_dim, cfg.d_model],
                0.05,
            );
            push_f32(
                &format!("{bp}.tm.v2.weight"),
                vec![cfg.d_model, cfg.v_lora_dim],
                0.05,
            );
        }

        push_f32(
            &format!("{bp}.tm.g1.weight"),
            vec![cfg.gate_lora_dim, cfg.d_model],
            0.05,
        );
        push_f32(
            &format!("{bp}.tm.g2.weight"),
            vec![cfg.d_model, cfg.gate_lora_dim],
            0.05,
        );

        push_f32(&format!("{bp}.tm.k_k"), vec![cfg.d_model], 0.85);
        push_f32(&format!("{bp}.tm.k_a"), vec![cfg.d_model], 1.0);
        push_f32(&format!("{bp}.tm.r_k"), vec![cfg.d_model], 0.1);
        push_f32(&format!("{bp}.tm.ln_x.weight"), vec![cfg.d_model], 1.0);
        push_f32(&format!("{bp}.tm.ln_x.bias"), vec![cfg.d_model], 0.0);

        push_f32(&format!("{bp}.ln2.weight"), vec![cfg.d_model], 1.0);
        push_f32(&format!("{bp}.ln2.bias"), vec![cfg.d_model], 0.0);
        push_f32(&format!("{bp}.cm.x_k"), vec![cfg.d_model], 0.5);
        push_f32(
            &format!("{bp}.cm.key.weight"),
            vec![cfg.ffn_dim, cfg.d_model],
            0.1,
        );
        push_f32(
            &format!("{bp}.cm.value.weight"),
            vec![cfg.d_model, cfg.ffn_dim],
            0.1,
        );
    }

    push_f32("ln_out.weight", vec![cfg.d_model], 1.0);
    push_f32("ln_out.bias", vec![cfg.d_model], 0.0);
    push_f32("head.weight", vec![cfg.vocab_size, cfg.d_model], 0.1);

    push_f32("copy_q.weight", vec![cfg.copy_dim, cfg.d_model], 0.1);
    push_f32("copy_k.weight", vec![cfg.copy_dim, cfg.d_model], 0.1);
    push_f32("copy_gate.weight", vec![1, cfg.d_model + cfg.copy_dim], 0.0);
    push_f32("copy_gate.bias", vec![1], 0.0);

    let mut views = HashMap::with_capacity(bufs.len());
    for (name, shape, dtype, bytes) in &bufs {
        views.insert(
            name.as_str(),
            TensorView::new(*dtype, shape.clone(), bytes)?,
        );
    }

    let serialized = safetensors::serialize(views, &None)?;
    std::fs::write(dir.join("model.safetensors"), serialized)?;
    Ok(())
}

#[tokio::test]
async fn cli_indexed_redb_runs_search_rebuild_and_verified_orchestrator()
-> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDirGuard::new("indexed-redb")?;
    let db_path = temp.path("base.redb");
    let idx_path = temp.path("base.idx");
    let db_str = db_path.to_string_lossy().into_owned();
    let idx_str = idx_path.to_string_lossy().into_owned();

    let out1 = exec_cli(&[
        "neurosym",
        "--backend",
        "indexed-redb",
        "--db-path",
        &db_str,
        "--index-dir",
        &idx_str,
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
    ])
    .await?;
    assert!(out1.contains("[indexed-redb] saved fact-preco"));

    let _ = exec_cli(&[
        "neurosym",
        "--backend",
        "indexed-redb",
        "--db-path",
        &db_str,
        "--index-dir",
        &idx_str,
        "fact",
        "add",
        "--id",
        "fact-desc",
        "--subject",
        "plano_pro",
        "--relation",
        "desconto_anual",
        "--object",
        "15%",
    ])
    .await?;

    let search_out = exec_cli(&[
        "neurosym",
        "--backend",
        "indexed-redb",
        "--db-path",
        &db_str,
        "--index-dir",
        &idx_str,
        "fact",
        "search",
        "--query",
        "15%",
    ])
    .await?;
    assert!(search_out.contains("fact-desc"));

    let rebuild_out = exec_cli(&[
        "neurosym",
        "--backend",
        "indexed-redb",
        "--db-path",
        &db_str,
        "--index-dir",
        &idx_str,
        "fact",
        "rebuild-index",
    ])
    .await?;
    assert!(rebuild_out.contains("with 2 facts"));

    let actions = r#"[
        {"action":"search_facts","query":"plano_","limit":5},
        {"action":"eval_math","expression":"199.90 * 12"},
        {"action":"propose_answer","text":"Anual do plano_pro: R$ 2398.8 (desconto 15%).","cited_facts":["fact-preco","fact-desc"]}
    ]"#;

    let ask_out = exec_cli(&[
        "neurosym",
        "--backend",
        "indexed-redb",
        "--db-path",
        &db_str,
        "--index-dir",
        &idx_str,
        "ask",
        "--prompt",
        "Quanto custa o plano anual?",
        "--actions-json",
        actions,
        "--require-citations",
        "--show-audit",
    ])
    .await?;

    assert!(ask_out.contains("Backend: indexed-redb | Steps: 3 | Repairs: 0"));
    assert!(ask_out.contains("FactsSearched"));
    assert!(ask_out.contains("MathEvaluated { value: \"2398.8\" }"));
    assert!(ask_out.contains("Citations: fact-preco, fact-desc"));
    Ok(())
}

#[tokio::test]
async fn cli_rwkv7_dir_loads_weights_and_validates_conflicts()
-> Result<(), Box<dyn std::error::Error>> {
    let conflict = Cli::try_parse_from([
        "neurosym",
        "ask",
        "--prompt",
        "oi",
        "--rwkv-dir",
        "artifacts/rwkv7_g1_01b_ns",
        "--actions-json",
        "[]",
    ]);
    assert!(conflict.is_err());

    let temp = TempDirGuard::new("rwkv7-cli")?;
    let db_path = temp.path("rwkv7.redb");
    let model_dir = temp.path("rwkv7_model");
    write_dummy_rwkv7_dir(&model_dir)?;

    let db_str = db_path.to_string_lossy().into_owned();
    let model_str = model_dir.to_string_lossy().into_owned();

    let result = exec_cli(&[
        "neurosym",
        "--backend",
        "indexed-redb",
        "--db-path",
        &db_str,
        "ask",
        "--prompt",
        "Qual o preco?",
        "--rwkv-dir",
        &model_str,
        "--max-steps",
        "1",
        "--max-tokens",
        "8",
    ])
    .await;

    let err_chain = match result {
        Ok(_) => String::new(),
        Err(err) => format!("{err:#}"),
    };
    assert!(
        err_chain.contains("orchestrator failed to produce a verified answer"),
        "unexpected error chain: {err_chain}"
    );
    Ok(())
}

#[test]
fn cli_chat_rwkv7_runs_multiturn_with_stats_reset_and_exit()
-> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDirGuard::new("rwkv7-chat")?;
    let model_dir = temp.path("rwkv7_chat_model");
    write_dummy_rwkv7_dir(&model_dir)?;

    let args = ChatArgs {
        rwkv_dir: model_dir,
        system: Some("Assistente embarcado.".to_owned()),
        temperature: 0.0,
        top_p: 1.0,
        presence_penalty: 0.2,
        max_tokens: 4,
        stats: true,
        no_think: false,
    };

    let scripted_input = "Ola NeuroSym\n/stats\n/reset\nSegunda mensagem\n/exit\n";
    let mut reader = Cursor::new(scripted_input.as_bytes());
    let mut output = Vec::new();

    run_chat_with_io(&args, &mut reader, &mut output)?;
    let rendered = String::from_utf8(output)?;

    assert!(rendered.contains("RWKV-7 Goose (estado matricial O(1))"));
    assert!(rendered.contains("FFN ReLU² pulados:"));
    assert!(rendered.contains("[estado recorrente zerado]"));
    assert!(rendered.contains("NeuroSym> "));
    Ok(())
}

#[tokio::test]
async fn cli_ask_with_context_file_audits_distant_contract_clauses()
-> Result<(), Box<dyn std::error::Error>> {
    let temp = TempDirGuard::new("ctx-file")?;
    let db_path = temp.path("ctx.redb");
    let doc_path = temp.path("contrato.txt");

    let mut contract = String::new();
    contract.push_str("Cláusula 1 - Objeto\nFornecimento de sensores.\n\n");
    contract.push_str(
        "Cláusula 2 - Prazo Máximo\nPrazo máximo de 15 dias corridos (vide Cláusula 47).\n\n",
    );
    for i in 3..=46 {
        contract.push_str(&format!(
            "Cláusula {i} - Regra {i}\nTexto auxiliar {i}.\n\n"
        ));
    }
    contract.push_str("Cláusula 47 - Exceção\nEntrega especial em 30 dias corridos.\n");
    std::fs::write(&doc_path, contract)?;

    let db_str = db_path.to_string_lossy().into_owned();
    let doc_str = doc_path.to_string_lossy().into_owned();

    let actions = r#"[
        {"action":"search_context","query":"entrega especial","limit":3},
        {"action":"read_spans","span_ids":["cl_02","cl_47"]},
        {"action":"eval_math","expression":"30 - 15"},
        {"action":"propose_answer","text":"A Clausula 47 excede em 15 dias a Clausula 2.","cited_facts":["cl_02","cl_47"]}
    ]"#;

    let ask_out = exec_cli(&[
        "neurosym",
        "--backend",
        "indexed-redb",
        "--db-path",
        &db_str,
        "ask",
        "--prompt",
        "Ha conflito entre a clausula 2 e a clausula 47?",
        "--context-file",
        &doc_str,
        "--actions-json",
        actions,
        "--require-citations",
        "--show-audit",
    ])
    .await?;

    assert!(ask_out.contains("Citations: cl_02, cl_47"));
    assert!(ask_out.contains("SpansRead"));
    assert!(ask_out.contains("MathEvaluated { value: \"15\" }"));
    assert!(ask_out.contains("Repairs: 0"));
    Ok(())
}
