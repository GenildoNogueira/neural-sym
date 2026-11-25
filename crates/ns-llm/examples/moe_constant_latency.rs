use std::time::Instant;

use ns_llm::quant::QuantizedMatrix;
use ns_llm::rwkv7::SparseColQuantizedMatrix;
use ns_llm::sparse_moe::{SparseExpert, SparseMoeFfn};

const D_MODEL: usize = 768;
const N_LAYERS: usize = 8;
const EXPERT_FFN_DIM: usize = 1536;
const TOP_K: usize = 2;
const WARMUP_STEPS: usize = 8;
const MEASURE_STEPS: usize = 60;

fn make_token_inputs(count: usize, d_model: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|t| {
            (0..d_model)
                .map(|c| {
                    let phase = ((t * 17 + c * 13) % 97) as f32 * 0.065;
                    phase.sin()
                })
                .collect()
        })
        .collect()
}

fn build_dense_layers(ffn_dim: usize) -> Vec<(QuantizedMatrix, SparseColQuantizedMatrix)> {
    let mut layers = Vec::with_capacity(N_LAYERS);
    let mut key_buf = vec![0.0_f32; ffn_dim * D_MODEL];
    let mut val_buf = vec![0.0_f32; D_MODEL * ffn_dim];

    for layer in 0..N_LAYERS {
        for r in 0..ffn_dim {
            // ~80% negativos -> replica a esparsidade ReLU² real do RWKV-7 G1
            let sign = if (r + layer) % 5 == 0 { 1.0 } else { -1.0 };
            let base = r * D_MODEL;
            for c in 0..D_MODEL {
                key_buf[base + c] = (((r + c + layer) % 13) as f32 * 0.01) * sign;
            }
        }
        for r in 0..D_MODEL {
            let base = r * ffn_dim;
            for c in 0..ffn_dim {
                val_buf[base + c] = (((r * 3 + c + layer) % 11) as f32 - 5.0) * 0.01;
            }
        }
        let key = QuantizedMatrix::from_f32(&key_buf, ffn_dim, D_MODEL)
            .expect("valid dense key dimensions");
        let val = SparseColQuantizedMatrix::from_row_major_f32(&val_buf, D_MODEL, ffn_dim)
            .expect("valid dense value dimensions");
        layers.push((key, val));
    }
    layers
}

fn build_moe_layers(num_experts: usize, neuron_top_k: usize) -> Vec<SparseMoeFfn> {
    let mut layers = Vec::with_capacity(N_LAYERS);
    let mut key_buf = vec![0.0_f32; EXPERT_FFN_DIM * D_MODEL];
    let mut val_buf = vec![0.0_f32; D_MODEL * EXPERT_FFN_DIM];

    for layer in 0..N_LAYERS {
        let mut router_w = vec![0.0_f32; num_experts * D_MODEL];
        for e in 0..num_experts {
            router_w[e * D_MODEL + ((e * 11 + layer * 7) % D_MODEL)] = 1.0;
        }
        let router = QuantizedMatrix::from_f32(&router_w, num_experts, D_MODEL)
            .expect("valid router dimensions");

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            for r in 0..EXPERT_FFN_DIM {
                let sign = if (r + e + layer) % 5 == 0 { 1.0 } else { -1.0 };
                let base = r * D_MODEL;
                for c in 0..D_MODEL {
                    key_buf[base + c] = (((r + c + e) % 13) as f32 * 0.01) * sign;
                }
            }
            for r in 0..D_MODEL {
                let base = r * EXPERT_FFN_DIM;
                for c in 0..EXPERT_FFN_DIM {
                    val_buf[base + c] = (((r * 3 + c + e) % 11) as f32 - 5.0) * 0.01;
                }
            }
            let exp = SparseExpert::from_f32(&key_buf, &val_buf, D_MODEL, EXPERT_FFN_DIM)
                .expect("valid expert dimensions");
            experts.push(exp);
        }

        let moe = SparseMoeFfn::new(
            D_MODEL,
            EXPERT_FFN_DIM,
            TOP_K,
            neuron_top_k,
            router,
            experts,
        )
        .expect("valid moe config");
        layers.push(moe);
    }
    layers
}

fn bench_dense(scale: usize, inputs: &[Vec<f32>]) {
    let ffn_dim = TOP_K * EXPERT_FFN_DIM * scale;
    let layers = build_dense_layers(ffn_dim);
    let total_bytes: usize = layers
        .iter()
        .map(|(k, v)| k.weight_bytes() + v.weight_bytes())
        .sum();

    let mut q_in = vec![0_i8; D_MODEL];
    let mut hidden = vec![0.0_f32; ffn_dim];
    let mut out = vec![0.0_f32; D_MODEL];

    for step in 0..WARMUP_STEPS {
        let x = &inputs[step % inputs.len()];
        out.copy_from_slice(x);
        for (k, v) in &layers {
            k.matvec_into(x, &mut q_in, &mut hidden);
            for h in &mut hidden {
                let r = h.max(0.0);
                *h = r * r;
            }
            let _ = v.accumulate_sparse(&hidden, &mut out);
        }
    }

    let mut bytes_read_acc = 0_u64;
    let t0 = Instant::now();
    for step in 0..MEASURE_STEPS {
        let x = &inputs[step % inputs.len()];
        out.copy_from_slice(x);
        for (k, v) in &layers {
            k.matvec_into(x, &mut q_in, &mut hidden);
            for h in &mut hidden {
                let r = h.max(0.0);
                *h = r * r;
            }
            let active = v.accumulate_sparse(&hidden, &mut out);
            bytes_read_acc += (k.weight_bytes() + active * v.col_stride_bytes()) as u64;
        }
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;
    let ms_per_tok = elapsed_ms / (MEASURE_STEPS as f64);
    let tok_s = 1_000.0 / ms_per_tok;
    let avg_read_mib = (bytes_read_acc as f64 / MEASURE_STEPS as f64) / (1024.0 * 1024.0);
    let total_mib = (total_bytes as f64) / (1024.0 * 1024.0);
    let skip_pct = 100.0 * (1.0 - avg_read_mib / total_mib);

    std::hint::black_box(&out);
    println!(
        "{:<24} | {:>4}x | {:>7.1} MiB | {:>7.2} MiB | {:>6.1}% | {:>7.2} ms | {:>8.1} tok/s",
        format!("Densa (ffn={ffn_dim})"),
        scale,
        total_mib,
        avg_read_mib,
        skip_pct,
        ms_per_tok,
        tok_s,
    );
}

fn bench_moe(label: &str, scale: usize, neuron_top_k: usize, inputs: &[Vec<f32>]) {
    let num_experts = TOP_K * scale;
    let layers = build_moe_layers(num_experts, neuron_top_k);
    let total_bytes: usize = layers.iter().map(SparseMoeFfn::weight_bytes).sum();

    let mut q_in = vec![0_i8; D_MODEL];
    let mut scratch = layers[0].alloc_scratch();
    let mut out = vec![0.0_f32; D_MODEL];

    for step in 0..WARMUP_STEPS {
        let x = &inputs[step % inputs.len()];
        out.copy_from_slice(x);
        for moe in &layers {
            let _ = moe.forward_into(x, &mut q_in, &mut scratch, &mut out);
        }
    }

    let mut bytes_read_acc = 0_u64;
    let t0 = Instant::now();
    for step in 0..MEASURE_STEPS {
        let x = &inputs[step % inputs.len()];
        out.copy_from_slice(x);
        for moe in &layers {
            let st = moe.forward_into(x, &mut q_in, &mut scratch, &mut out);
            bytes_read_acc += st.bytes_read as u64;
        }
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;
    let ms_per_tok = elapsed_ms / (MEASURE_STEPS as f64);
    let tok_s = 1_000.0 / ms_per_tok;
    let avg_read_mib = (bytes_read_acc as f64 / MEASURE_STEPS as f64) / (1024.0 * 1024.0);
    let total_mib = (total_bytes as f64) / (1024.0 * 1024.0);
    let skip_pct = 100.0 * (1.0 - avg_read_mib / total_mib);

    std::hint::black_box(&out);
    println!(
        "{:<24} | {:>4}x | {:>7.1} MiB | {:>7.2} MiB | {:>6.1}% | {:>7.2} ms | {:>8.1} tok/s",
        label, scale, total_mib, avg_read_mib, skip_pct, ms_per_tok, tok_s,
    );
}

fn main() {
    let inputs = make_token_inputs(MEASURE_STEPS, D_MODEL);

    println!();
    println!(
        "Escala de Capacidade FFN na CPU (L={N_LAYERS}, d={D_MODEL}, expert_ffn={EXPERT_FFN_DIM}, top_k={TOP_K})"
    );
    println!(
        "{:<24} | {:>5} | {:>11} | {:>11} | {:>7} | {:>10} | {:>14}",
        "Arquitetura", "Cap.", "Total RAM", "Lido/tok", "Pulado", "ms/tok", "Throughput"
    );
    println!("{}", "-".repeat(100));

    for scale in [1_usize, 4, 8, 16] {
        bench_dense(scale, &inputs);
    }
    println!("{}", "-".repeat(100));
    for scale in [1_usize, 4, 8, 16] {
        let e = TOP_K * scale;
        bench_moe(&format!("MoE E={e} (ReLU²)"), scale, 0, &inputs);
    }
    println!("{}", "-".repeat(100));
    for scale in [1_usize, 4, 8, 16] {
        let e = TOP_K * scale;
        bench_moe(&format!("MoE E={e} + TopM=192"), scale, 192, &inputs);
    }
}
