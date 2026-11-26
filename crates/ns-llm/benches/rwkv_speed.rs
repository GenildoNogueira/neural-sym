use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use ns_llm::quant::active_q4_backend;
use ns_llm::{Rwkv7Config, Rwkv7Llm, Rwkv7State};
use safetensors::Dtype;
use safetensors::tensor::TensorView;

const BENCH_VOCAB_SIZE: usize = 8_192;
const BENCH_D_MODEL: usize = 256;
const BENCH_N_LAYERS: usize = 6;
const BENCH_HEAD_DIM: usize = 64;
const BENCH_FFN_DIM: usize = 768;
const BENCH_LORA_DIM: usize = 32;

const PREFILL_TOKENS: usize = 128;
const PREFILL_RUNS: usize = 20;
const HIDDEN_STEPS: usize = 300;
const HEAD_ITERS: usize = 300;
const DECODE_STEPS: usize = 200;
const E2E_RUNS: usize = 10;

#[derive(Debug, Clone, Copy)]
struct StepStats {
    mean_ms: f64,
    p50_ms: f64,
    p99_ms: f64,
    tok_per_sec: f64,
}

impl StepStats {
    fn from_nanos(mut samples_ns: Vec<u128>, total_elapsed: Duration, total_tokens: usize) -> Self {
        samples_ns.sort_unstable();
        let count = samples_ns.len().max(1);
        let sum_ns: u128 = samples_ns.iter().copied().sum();

        let p50_ns = percentile(&samples_ns, 50);
        let p99_ns = percentile(&samples_ns, 99);

        Self {
            mean_ms: (sum_ns as f64) / (count as f64) / 1_000_000.0,
            p50_ms: (p50_ns as f64) / 1_000_000.0,
            p99_ms: (p99_ns as f64) / 1_000_000.0,
            tok_per_sec: (total_tokens as f64) / total_elapsed.as_secs_f64().max(1e-9),
        }
    }
}

struct ModelBenchReport {
    label: String,
    weight_bytes: usize,
    state_bytes: usize,
    prefill_seq: StepStats,
    prefill_chunk4: StepStats,
    hidden: StepStats,
    hidden_sparsity_pct: f64,
    head_full: StepStats,
    head_256: StepStats,
    head_32: StepStats,
    decode_full: StepStats,
    decode_sparse_32: StepStats,
    e2e_isa: Option<(f64, usize)>,
}

fn percentile(sorted: &[u128], pct: usize) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = sorted.len().saturating_sub(1).saturating_mul(pct) / 100;
    sorted[idx]
}

struct XorShift64(u64);

impl XorShift64 {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_signed_f32(&mut self, scale: f32) -> f32 {
        let bits = self.next_u64() >> 40;
        let unit = (bits as f32) / ((1_u32 << 24) as f32);
        (unit * 2.0 - 1.0) * scale
    }
}

fn build_synthetic_vocab_bin(vocab_size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(vocab_size * 8);
    out.extend_from_slice(&(vocab_size as u32).to_le_bytes());

    // Token 0 = EOS/BOS (padrão RWKV World)
    out.extend_from_slice(&0_u16.to_le_bytes());

    // Tokens 1..=256 = todos os 256 bytes individuais
    let mut next_id = 1_usize;
    for b in 0_u8..=255_u8 {
        if next_id >= vocab_size {
            break;
        }
        out.extend_from_slice(&1_u16.to_le_bytes());
        out.push(b);
        next_id += 1;
    }

    let anchors = [
        "plano_pro",
        "plano_enterprise",
        "preco_mensal",
        "cl_02",
        "cl_47",
        "fact-",
        "USER:",
        "TOOL:",
        "SYSTEM:",
        "Assistant:",
    ];

    for anchor in anchors {
        if next_id >= vocab_size {
            break;
        }
        let bytes = anchor.as_bytes();
        out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(bytes);
        next_id += 1;
    }

    let mut filler_idx = 0_usize;
    while next_id < vocab_size {
        let candidate = format!("tk{filler_idx:04}");
        filler_idx += 1;
        let bytes = candidate.as_bytes();
        out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(bytes);
        next_id += 1;
    }

    out
}

fn build_synthetic_rwkv7_model() -> Result<Rwkv7Llm, Box<dyn std::error::Error>> {
    let cfg = Rwkv7Config {
        arch: "rwkv7".to_owned(),
        vocab_size: BENCH_VOCAB_SIZE,
        d_model: BENCH_D_MODEL,
        n_layers: BENCH_N_LAYERS,
        head_dim: BENCH_HEAD_DIM,
        ffn_dim: BENCH_FFN_DIM,
        decay_lora_dim: BENCH_LORA_DIM,
        a_lora_dim: BENCH_LORA_DIM,
        v_lora_dim: BENCH_LORA_DIM,
        gate_lora_dim: BENCH_LORA_DIM,
        bos_token_id: 0,
        eos_token_id: 0,
        layer_norm_eps: 1e-5,
        group_norm_eps: 64e-5,
        vocab_file: "vocab.bin".to_owned(),
        copy_dim: 0,
        isa_ptr_dim: 0,
        isa_num_vocab: 64,
        copy_max_len: 0,
        moe_num_experts: 0,
        moe_top_k: 2,
        moe_neuron_top_k: 0,
    };

    let vocab_bin = build_synthetic_vocab_bin(cfg.vocab_size);
    let mut rng = XorShift64::new(0x5EED_2026_1124_0007);
    let mut buffers: Vec<(String, Vec<usize>, Dtype, Vec<u8>)> = Vec::with_capacity(220);

    let push_f32 = |bufs: &mut Vec<(String, Vec<usize>, Dtype, Vec<u8>)>,
                    rng: &mut XorShift64,
                    name: &str,
                    shape: Vec<usize>,
                    scale: f32,
                    bias_offset: f32| {
        let count: usize = shape.iter().product();
        let mut bytes = Vec::with_capacity(count * 4);
        for _ in 0..count {
            let val = bias_offset + rng.next_signed_f32(scale);
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bufs.push((name.to_owned(), shape, Dtype::F32, bytes));
    };

    let push_q8_matrix = |bufs: &mut Vec<(String, Vec<usize>, Dtype, Vec<u8>)>,
                          rng: &mut XorShift64,
                          name: &str,
                          out_dim: usize,
                          in_dim: usize| {
        let total = out_dim * in_dim;
        let mut q_bytes = Vec::with_capacity(total);
        for _ in 0..total {
            let raw = (rng.next_u64() % 255) as u8;
            let val_i8 = i8::try_from(i16::from(raw) - 127).unwrap_or(0);
            q_bytes.push(val_i8.to_ne_bytes()[0]);
        }

        let mut s_bytes = Vec::with_capacity(out_dim * 4);
        let base_scale = 0.05_f32 / (in_dim as f32).sqrt();
        for _ in 0..out_dim {
            let scale = base_scale * (0.8 + (rng.next_u64() % 40) as f32 / 100.0);
            s_bytes.extend_from_slice(&scale.to_le_bytes());
        }

        bufs.push((
            format!("{name}.q"),
            vec![out_dim, in_dim],
            Dtype::I8,
            q_bytes,
        ));
        bufs.push((format!("{name}.scale"), vec![out_dim], Dtype::F32, s_bytes));
    };

    push_q8_matrix(
        &mut buffers,
        &mut rng,
        "emb.weight",
        cfg.vocab_size,
        cfg.d_model,
    );
    push_f32(
        &mut buffers,
        &mut rng,
        "ln_in.weight",
        vec![cfg.d_model],
        0.05,
        1.0,
    );
    push_f32(
        &mut buffers,
        &mut rng,
        "ln_in.bias",
        vec![cfg.d_model],
        0.01,
        0.0,
    );

    for i in 0..cfg.n_layers {
        let p = format!("blocks.{i}");
        push_f32(
            &mut buffers,
            &mut rng,
            &format!("{p}.ln1.weight"),
            vec![cfg.d_model],
            0.05,
            1.0,
        );
        push_f32(
            &mut buffers,
            &mut rng,
            &format!("{p}.ln1.bias"),
            vec![cfg.d_model],
            0.01,
            0.0,
        );

        for shift in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"] {
            push_f32(
                &mut buffers,
                &mut rng,
                &format!("{p}.tm.{shift}"),
                vec![cfg.d_model],
                0.2,
                0.5,
            );
        }
        for proj in ["receptance", "key", "value", "output"] {
            push_q8_matrix(
                &mut buffers,
                &mut rng,
                &format!("{p}.tm.{proj}.weight"),
                cfg.d_model,
                cfg.d_model,
            );
        }

        push_f32(
            &mut buffers,
            &mut rng,
            &format!("{p}.tm.w0"),
            vec![cfg.d_model],
            1.0,
            -1.0,
        );
        push_q8_matrix(
            &mut buffers,
            &mut rng,
            &format!("{p}.tm.w1.weight"),
            cfg.decay_lora_dim,
            cfg.d_model,
        );
        push_q8_matrix(
            &mut buffers,
            &mut rng,
            &format!("{p}.tm.w2.weight"),
            cfg.d_model,
            cfg.decay_lora_dim,
        );

        push_f32(
            &mut buffers,
            &mut rng,
            &format!("{p}.tm.a0"),
            vec![cfg.d_model],
            0.5,
            0.0,
        );
        push_q8_matrix(
            &mut buffers,
            &mut rng,
            &format!("{p}.tm.a1.weight"),
            cfg.a_lora_dim,
            cfg.d_model,
        );
        push_q8_matrix(
            &mut buffers,
            &mut rng,
            &format!("{p}.tm.a2.weight"),
            cfg.d_model,
            cfg.a_lora_dim,
        );

        if i > 0 {
            push_f32(
                &mut buffers,
                &mut rng,
                &format!("{p}.tm.v0"),
                vec![cfg.d_model],
                0.5,
                0.0,
            );
            push_q8_matrix(
                &mut buffers,
                &mut rng,
                &format!("{p}.tm.v1.weight"),
                cfg.v_lora_dim,
                cfg.d_model,
            );
            push_q8_matrix(
                &mut buffers,
                &mut rng,
                &format!("{p}.tm.v2.weight"),
                cfg.d_model,
                cfg.v_lora_dim,
            );
        }

        push_q8_matrix(
            &mut buffers,
            &mut rng,
            &format!("{p}.tm.g1.weight"),
            cfg.gate_lora_dim,
            cfg.d_model,
        );
        push_q8_matrix(
            &mut buffers,
            &mut rng,
            &format!("{p}.tm.g2.weight"),
            cfg.d_model,
            cfg.gate_lora_dim,
        );

        push_f32(
            &mut buffers,
            &mut rng,
            &format!("{p}.tm.k_k"),
            vec![cfg.d_model],
            0.1,
            0.85,
        );
        push_f32(
            &mut buffers,
            &mut rng,
            &format!("{p}.tm.k_a"),
            vec![cfg.d_model],
            0.1,
            1.0,
        );
        push_f32(
            &mut buffers,
            &mut rng,
            &format!("{p}.tm.r_k"),
            vec![cfg.d_model],
            0.1,
            0.0,
        );
        push_f32(
            &mut buffers,
            &mut rng,
            &format!("{p}.tm.ln_x.weight"),
            vec![cfg.d_model],
            0.05,
            1.0,
        );
        push_f32(
            &mut buffers,
            &mut rng,
            &format!("{p}.tm.ln_x.bias"),
            vec![cfg.d_model],
            0.01,
            0.0,
        );

        push_f32(
            &mut buffers,
            &mut rng,
            &format!("{p}.ln2.weight"),
            vec![cfg.d_model],
            0.05,
            1.0,
        );
        push_f32(
            &mut buffers,
            &mut rng,
            &format!("{p}.ln2.bias"),
            vec![cfg.d_model],
            0.01,
            0.0,
        );
        push_f32(
            &mut buffers,
            &mut rng,
            &format!("{p}.cm.x_k"),
            vec![cfg.d_model],
            0.2,
            0.5,
        );
        push_q8_matrix(
            &mut buffers,
            &mut rng,
            &format!("{p}.cm.key.weight"),
            cfg.ffn_dim,
            cfg.d_model,
        );
        push_q8_matrix(
            &mut buffers,
            &mut rng,
            &format!("{p}.cm.value.weight"),
            cfg.d_model,
            cfg.ffn_dim,
        );
    }

    push_f32(
        &mut buffers,
        &mut rng,
        "ln_out.weight",
        vec![cfg.d_model],
        0.05,
        1.0,
    );
    push_f32(
        &mut buffers,
        &mut rng,
        "ln_out.bias",
        vec![cfg.d_model],
        0.01,
        0.0,
    );
    push_q8_matrix(
        &mut buffers,
        &mut rng,
        "head.weight",
        cfg.vocab_size,
        cfg.d_model,
    );

    let mut views = HashMap::with_capacity(buffers.len());
    for (name, shape, dtype, bytes) in &buffers {
        views.insert(
            name.as_str(),
            TensorView::new(*dtype, shape.clone(), bytes)?,
        );
    }

    let serialized = safetensors::serialize(views, &None)?;
    Ok(Rwkv7Llm::from_parts(cfg, &vocab_bin, &serialized)?.with_max_new_tokens(64))
}

fn make_prompt_tokens(vocab: usize, count: usize) -> Vec<usize> {
    let span = vocab.saturating_sub(2).max(1);
    (0..count).map(|i| 1 + ((i * 131 + 17) % span)).collect()
}

fn prefill_seq(
    model: &Rwkv7Llm,
    prompt_ids: &[usize],
    state: &mut Rwkv7State,
) -> Result<(), Box<dyn std::error::Error>> {
    model.forward_hidden(model.config().bos_token_id, state)?;
    for &tok in prompt_ids {
        model.forward_hidden(tok, state)?;
    }
    Ok(())
}

fn prefill_chunk4(
    model: &Rwkv7Llm,
    prompt_ids: &[usize],
    state: &mut Rwkv7State,
    x_outs_buf: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let d = model.config().d_model;
    model.forward_hidden(model.config().bos_token_id, state)?;

    let chunks = prompt_ids.chunks_exact(4);
    let rem = chunks.remainder();
    for chunk in chunks {
        let (s0, rest) = x_outs_buf.split_at_mut(d);
        let (s1, rest) = rest.split_at_mut(d);
        let (s2, s3) = rest.split_at_mut(d);
        model.forward_hidden_chunk4(
            [chunk[0], chunk[1], chunk[2], chunk[3]],
            state,
            [s0, s1, s2, s3],
        )?;
    }
    for &tok in rem {
        model.forward_hidden(tok, state)?;
    }
    Ok(())
}

fn argmax_slice(logits: &[f32]) -> usize {
    let mut best_idx = 0_usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx
}

fn bench_prefill_seq(model: &Rwkv7Llm) -> Result<StepStats, Box<dyn std::error::Error>> {
    let prompt_ids = make_prompt_tokens(model.config().vocab_size, PREFILL_TOKENS);
    let mut state = Rwkv7State::new(model.config());
    prefill_seq(model, &prompt_ids[..16], &mut state)?;

    let mut per_run_per_token_ns = Vec::with_capacity(PREFILL_RUNS);
    let total_start = Instant::now();

    for _ in 0..PREFILL_RUNS {
        state.reset_recurrent_zero();
        let t0 = Instant::now();
        prefill_seq(model, &prompt_ids, &mut state)?;
        let elapsed_ns = t0.elapsed().as_nanos();
        black_box(&state);
        per_run_per_token_ns.push(elapsed_ns / (PREFILL_TOKENS as u128));
    }

    Ok(StepStats::from_nanos(
        per_run_per_token_ns,
        total_start.elapsed(),
        PREFILL_RUNS * PREFILL_TOKENS,
    ))
}

fn bench_prefill_chunk4(model: &Rwkv7Llm) -> Result<StepStats, Box<dyn std::error::Error>> {
    let prompt_ids = make_prompt_tokens(model.config().vocab_size, PREFILL_TOKENS);
    let mut state = Rwkv7State::new(model.config());
    let mut x_outs = vec![0.0_f32; 4 * model.config().d_model];
    prefill_chunk4(model, &prompt_ids[..16], &mut state, &mut x_outs)?;

    let mut per_run_per_token_ns = Vec::with_capacity(PREFILL_RUNS);
    let total_start = Instant::now();

    for _ in 0..PREFILL_RUNS {
        state.reset_recurrent_zero();
        let t0 = Instant::now();
        prefill_chunk4(model, &prompt_ids, &mut state, &mut x_outs)?;
        let elapsed_ns = t0.elapsed().as_nanos();
        black_box((&state, &x_outs));
        per_run_per_token_ns.push(elapsed_ns / (PREFILL_TOKENS as u128));
    }

    Ok(StepStats::from_nanos(
        per_run_per_token_ns,
        total_start.elapsed(),
        PREFILL_RUNS * PREFILL_TOKENS,
    ))
}

fn bench_forward_hidden(model: &Rwkv7Llm) -> Result<(StepStats, f64), Box<dyn std::error::Error>> {
    let vocab = model.config().vocab_size;
    let prompt_ids = make_prompt_tokens(vocab, PREFILL_TOKENS);
    let mut state = Rwkv7State::new(model.config());
    let mut x_outs = vec![0.0_f32; 4 * model.config().d_model];
    prefill_chunk4(model, &prompt_ids, &mut state, &mut x_outs)?;

    let mut samples_ns = Vec::with_capacity(HIDDEN_STEPS);
    let total_start = Instant::now();

    for step in 0..HIDDEN_STEPS {
        let tok = 1 + ((step * 73 + 5) % (vocab - 1));
        let t0 = Instant::now();
        model.forward_hidden(tok, &mut state)?;
        samples_ns.push(t0.elapsed().as_nanos());
    }
    let sparsity_pct = state.ffn_sparsity_ratio() * 100.0;
    black_box(&state);

    Ok((
        StepStats::from_nanos(samples_ns, total_start.elapsed(), HIDDEN_STEPS),
        sparsity_pct,
    ))
}

fn bench_head_full(model: &Rwkv7Llm) -> Result<StepStats, Box<dyn std::error::Error>> {
    let mut state = Rwkv7State::new(model.config());
    model.forward_hidden(model.config().bos_token_id, &mut state)?;
    let mut logits = vec![0.0_f32; model.config().vocab_size];

    for _ in 0..16 {
        model.project_head_into(&mut state, &mut logits)?;
        black_box(argmax_slice(&logits));
    }

    let mut samples_ns = Vec::with_capacity(HEAD_ITERS);
    let total_start = Instant::now();

    for _ in 0..HEAD_ITERS {
        let t0 = Instant::now();
        model.project_head_into(&mut state, &mut logits)?;
        let winner = argmax_slice(&logits);
        samples_ns.push(t0.elapsed().as_nanos());
        black_box(winner);
    }

    Ok(StepStats::from_nanos(
        samples_ns,
        total_start.elapsed(),
        HEAD_ITERS,
    ))
}

fn bench_head_sparse_subset(
    model: &Rwkv7Llm,
    allowed_count: usize,
) -> Result<StepStats, Box<dyn std::error::Error>> {
    let vocab = model.config().vocab_size;
    let effective = allowed_count.min(vocab).max(1);
    let step = (vocab / effective).max(1);
    let rows: Vec<usize> = (0..effective).map(|i| (i * step) % vocab).collect();

    let mut state = Rwkv7State::new(model.config());
    model.forward_hidden(model.config().bos_token_id, &mut state)?;

    for _ in 0..16 {
        black_box(model.sample_subset_from_hidden(&mut state, &rows));
    }

    let mut samples_ns = Vec::with_capacity(HEAD_ITERS);
    let total_start = Instant::now();

    for _ in 0..HEAD_ITERS {
        let t0 = Instant::now();
        let winner = model.sample_subset_from_hidden(&mut state, &rows);
        samples_ns.push(t0.elapsed().as_nanos());
        black_box(winner);
    }

    Ok(StepStats::from_nanos(
        samples_ns,
        total_start.elapsed(),
        HEAD_ITERS,
    ))
}

fn bench_decode_full(model: &Rwkv7Llm) -> Result<StepStats, Box<dyn std::error::Error>> {
    let prompt_ids = make_prompt_tokens(model.config().vocab_size, PREFILL_TOKENS);
    let mut state = Rwkv7State::new(model.config());
    let mut x_outs = vec![0.0_f32; 4 * model.config().d_model];
    prefill_chunk4(model, &prompt_ids, &mut state, &mut x_outs)?;

    let mut logits = vec![0.0_f32; model.config().vocab_size];
    let mut token = 1_usize;

    let mut samples_ns = Vec::with_capacity(DECODE_STEPS);
    let total_start = Instant::now();

    for _ in 0..DECODE_STEPS {
        let t0 = Instant::now();
        model.forward_hidden(token, &mut state)?;
        model.project_head_into(&mut state, &mut logits)?;
        token = argmax_slice(&logits);
        samples_ns.push(t0.elapsed().as_nanos());
        black_box(token);
    }

    Ok(StepStats::from_nanos(
        samples_ns,
        total_start.elapsed(),
        DECODE_STEPS,
    ))
}

fn bench_decode_sparse(
    model: &Rwkv7Llm,
    allowed_count: usize,
) -> Result<StepStats, Box<dyn std::error::Error>> {
    let vocab = model.config().vocab_size;
    let effective = allowed_count.min(vocab).max(1);
    let step = (vocab / effective).max(1);
    let rows: Vec<usize> = (0..effective).map(|i| (i * step) % vocab).collect();

    let prompt_ids = make_prompt_tokens(vocab, PREFILL_TOKENS);
    let mut state = Rwkv7State::new(model.config());
    let mut x_outs = vec![0.0_f32; 4 * model.config().d_model];
    prefill_chunk4(model, &prompt_ids, &mut state, &mut x_outs)?;

    let mut token = 1_usize;
    let mut samples_ns = Vec::with_capacity(DECODE_STEPS);
    let total_start = Instant::now();

    for _ in 0..DECODE_STEPS {
        let t0 = Instant::now();
        model.forward_hidden(token, &mut state)?;
        let winner = model.sample_subset_from_hidden(&mut state, &rows);
        token = winner.and_then(|id| usize::try_from(id).ok()).unwrap_or(1);
        samples_ns.push(t0.elapsed().as_nanos());
        black_box(token);
    }

    Ok(StepStats::from_nanos(
        samples_ns,
        total_start.elapsed(),
        DECODE_STEPS,
    ))
}

fn bench_e2e_isa(model: &Rwkv7Llm) -> Option<(f64, usize)> {
    if !model.has_isa_head() {
        return None;
    }

    let prompt = concat!(
        "USER: Qual o(a) preco_mensal de plano_enterprise?\n",
        "TOOL: {\"status\":\"found\",\"fact\":{\"id\":\"fact-inedito-9842\",",
        "\"subject\":\"plano_enterprise\",\"relation\":\"preco_mensal\",",
        "\"object\":\"749.50\",\"source_ref\":\"bench\"}}\n"
    );

    let mut ok_count = 0_usize;
    let t0 = Instant::now();
    for _ in 0..E2E_RUNS {
        model.clear_prefix_cache();
        let action = model.generate_action_isa(prompt, true);
        if action.is_ok() {
            ok_count += 1;
        }
        let _ = black_box(action);
    }

    let avg_ms = t0.elapsed().as_secs_f64() * 1_000.0 / (E2E_RUNS as f64);
    Some((avg_ms, ok_count))
}

fn run_suite(
    label: String,
    model: &Rwkv7Llm,
) -> Result<ModelBenchReport, Box<dyn std::error::Error>> {
    let state = Rwkv7State::new(model.config());
    let prefill_seq = bench_prefill_seq(model)?;
    let prefill_chunk4 = bench_prefill_chunk4(model)?;
    let (hidden, hidden_sparsity_pct) = bench_forward_hidden(model)?;
    let head_full = bench_head_full(model)?;
    let head_256 = bench_head_sparse_subset(model, 256)?;
    let head_32 = bench_head_sparse_subset(model, 32)?;
    let decode_full = bench_decode_full(model)?;
    let decode_sparse_32 = bench_decode_sparse(model, 32)?;
    let e2e_isa = bench_e2e_isa(model);

    Ok(ModelBenchReport {
        label,
        weight_bytes: model.total_weight_bytes(),
        state_bytes: state.state_bytes(),
        prefill_seq,
        prefill_chunk4,
        hidden,
        hidden_sparsity_pct,
        head_full,
        head_256,
        head_32,
        decode_full,
        decode_sparse_32,
        e2e_isa,
    })
}

fn print_row(label: &str, stats: StepStats) {
    println!(
        "{:<44} | {:>8.3} ms | {:>8.3} ms | {:>8.3} ms | {:>10.1} tok/s",
        label, stats.mean_ms, stats.p50_ms, stats.p99_ms, stats.tok_per_sec
    );
}

fn print_report(r: &ModelBenchReport, cfg: &Rwkv7Config) {
    let chunk4_speedup = r.prefill_seq.mean_ms / r.prefill_chunk4.mean_ms.max(1e-9);

    println!();
    println!(
        "================================================================================================="
    );
    println!(
        "  NeuroSym RWKV-7 Goose CPU Speed Benchmark | {} (Q4: {})",
        r.label,
        active_q4_backend()
    );
    println!(
        "  Perfil: d={} | L={} | H={}x{} | ffn={} | V={} | isa={} | pesos: {:.2} MiB | estado O(1): {:.1} KiB",
        cfg.d_model,
        cfg.n_layers,
        cfg.d_model / cfg.head_dim.max(1),
        cfg.head_dim,
        cfg.ffn_dim,
        cfg.vocab_size,
        cfg.isa_ptr_dim > 0,
        (r.weight_bytes as f64) / (1024.0 * 1024.0),
        (r.state_bytes as f64) / 1024.0,
    );
    println!(
        "================================================================================================="
    );
    println!(
        "{:<44} | {:>11} | {:>11} | {:>11} | {:>16}",
        "Etapa medida", "Média", "p50", "p99", "Throughput"
    );
    println!(
        "---------------------------------------------+-------------+-------------+-------------+-----------------"
    );
    print_row("Prefill sequencial (128 tok, 1 a 1)", r.prefill_seq);
    print_row("Prefill chunk4 (128 tok, matmul4_into)", r.prefill_chunk4);
    print_row("forward_hidden RWKV-7 (estado matricial)", r.hidden);
    print_row("head completa (V linhas + argmax)", r.head_full);
    print_row("head esparsa (256 linhas)", r.head_256);
    print_row("head esparsa (32 linhas)", r.head_32);
    println!(
        "---------------------------------------------+-------------+-------------+-------------+-----------------"
    );
    print_row("Decode denso (hidden + head completa)", r.decode_full);
    print_row(
        "Decode esparso (hidden + head 32 linhas)",
        r.decode_sparse_32,
    );
    println!(
        "================================================================================================="
    );
    println!(
        "  Ganho chunk4 vs sequencial: {:.2}x | FFN ReLU² esparsa: {:.1}% colunas puladas",
        chunk4_speedup, r.hidden_sparsity_pct
    );
    if let Some((avg_ms, ok_count)) = r.e2e_isa {
        println!(
            "  E2E generate_action_isa (frio): média = {:.2} ms | ações válidas = {ok_count}/{E2E_RUNS}",
            avg_ms
        );
    }
    println!(
        "================================================================================================="
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(dir) = std::env::var("NS_RWKV_BENCH_DIR") {
        let model = Rwkv7Llm::load_dir(&dir)?;
        let report = run_suite(format!("modelo real ({dir})"), &model)?;
        print_report(&report, model.config());
        return Ok(());
    }

    let model = build_synthetic_rwkv7_model()?;
    let report = run_suite("RWKV-7 Goose sintético (ISA/chunk4)".to_owned(), &model)?;
    print_report(&report, model.config());
    Ok(())
}
