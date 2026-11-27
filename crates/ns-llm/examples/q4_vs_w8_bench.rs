//! cargo run --release -p ns-llm --features custom-rwkv --example q4_vs_w8_bench
//!
//! Compara `QuantizedMatrix` W8A8 × Q4A8 em matvec (M x 1) para as formas do RWKV-7 0.1B.
//!
//! Variáveis úteis: `RAYON_NUM_THREADS`, `NS_BENCH_SAMPLES` (padrão 9) e `NS_BENCH_POOL_MIB`
//! (padrão 96, ~3x o L3). Cada forma é medida em dois modos:
//! - `quente`: a mesma matriz repetida (mede o kernel com pesos no cache);
//! - `stream`: gira por um pool de cópias maior que o L3 (mede com os pesos vindo da RAM, como
//!   no decode real, que lê ~150 MiB por token).
//!
//! Qualidade: o cosseno é contra o matvec em `f64` sobre os pesos originais (não W8 contra Q4).

use std::process::ExitCode;

use ns_llm::quant::{QuantFormat, QuantizedMatrix, active_q4_backend};

#[path = "common/mod.rs"]
mod common;

fn build_pool(
    weights: &[f32],
    out_dim: usize,
    in_dim: usize,
    format: QuantFormat,
) -> Result<Vec<QuantizedMatrix>, Box<dyn std::error::Error>> {
    let first = QuantizedMatrix::from_f32_with_format(weights, out_dim, in_dim, format)?;
    let n = common::pool_len(first.weight_bytes());
    let mut pool = Vec::with_capacity(n);
    for _ in 1..n {
        pool.push(first.clone());
    }
    pool.push(first);
    Ok(pool)
}

fn bench_shape(
    label: &str,
    out_dim: usize,
    in_dim: usize,
    iters: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let weights = common::fill_weights(out_dim, in_dim, 0x5eed_0001 ^ (out_dim as u64));
    let x = common::fill_input(in_dim, 0x5eed_0002);
    let reference = common::matvec_ref(&weights, out_dim, in_dim, &x);

    let w8 = build_pool(&weights, out_dim, in_dim, QuantFormat::W8)?;
    let q4 = build_pool(&weights, out_dim, in_dim, QuantFormat::Q4)?;
    drop(weights);

    let mut q_in_w8 = vec![0_i8; in_dim];
    let mut q_in_q4 = vec![0_i8; in_dim];
    let mut out_w8 = vec![0.0_f32; out_dim];
    let mut out_q4 = vec![0.0_f32; out_dim];

    w8[0].matvec_into(&x, &mut q_in_w8, &mut out_w8);
    q4[0].matvec_into(&x, &mut q_in_q4, &mut out_q4);
    let cos_w8 = common::cosine(&reference, &out_w8);
    let cos_q4 = common::cosine(&reference, &out_q4);

    let w8_bytes = w8[0].weight_bytes();
    let q4_bytes = q4[0].weight_bytes();

    for (mode, hot) in [("quente", true), ("stream", false)] {
        let (w8_ms, q4_ms) = common::time_ab_ms(
            iters,
            |i| {
                let m = if hot { &w8[0] } else { &w8[i % w8.len()] };
                m.matvec_into(
                    std::hint::black_box(&x),
                    &mut q_in_w8,
                    std::hint::black_box(&mut out_w8),
                );
            },
            |i| {
                let m = if hot { &q4[0] } else { &q4[i % q4.len()] };
                m.matvec_into(
                    std::hint::black_box(&x),
                    &mut q_in_q4,
                    std::hint::black_box(&mut out_q4),
                );
            },
        );
        println!(
            "{:<20} | {:<6} | {:>7.2} | {:>7.2} | {:>8.3} | {:>8.3} | {:>5.2}x | {:>6.1} | {:>6.1} | {:.5} | {:.5}",
            label,
            mode,
            common::mib(w8_bytes),
            common::mib(q4_bytes),
            w8_ms,
            q4_ms,
            w8_ms / q4_ms,
            common::gib_per_s(w8_bytes, w8_ms),
            common::gib_per_s(q4_bytes, q4_ms),
            cos_w8,
            cos_q4,
        );
    }

    Ok(())
}

fn main() -> ExitCode {
    println!();
    println!("Comparativo QuantizedMatrix: W8A8 vs Q4A8 (bloco 32), matvec M x 1");
    common::print_env(active_q4_backend());
    println!(
        "{:<20} | {:<6} | {:>7} | {:>7} | {:>8} | {:>8} | {:>6} | {:>6} | {:>6} | {:<7} | {:<7}",
        "Matriz (0.1B)",
        "modo",
        "MiB W8",
        "MiB Q4",
        "W8 ms",
        "Q4 ms",
        "Ganho",
        "W8 GiB/s",
        "Q4 GiB/s",
        "cos W8",
        "cos Q4"
    );
    println!("{}", "-".repeat(122));

    let shapes: [(&str, usize, usize, usize); 3] = [
        ("tm.proj (768x768)", 768, 768, 400),
        ("cm.key (3072x768)", 3072, 768, 200),
        ("head (65536x768)", 65536, 768, 20),
    ];
    for (label, out_dim, in_dim, iters) in shapes {
        if let Err(err) = bench_shape(label, out_dim, in_dim, iters) {
            eprintln!("erro em {label}: {err}");
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}
