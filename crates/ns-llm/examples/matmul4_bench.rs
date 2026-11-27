//! cargo run --release -p ns-llm --features custom-rwkv --example matmul4_bench
//!
//! Micro-matmul `M x 4` (`matmul4_into`, pesos lidos 1x) contra 4x matvec `M x 1`.
//!
//! - Verifica a **equivalência numérica** (matmul4 × 4 matvec) e sai com erro se divergir além de
//!   `NS_BENCH_TOL` (padrão 1e-3, relativo ao maior módulo da saída).
//! - Mede A/B intercalado, mediana de `NS_BENCH_SAMPLES` amostras, nos modos `quente` (mesma
//!   matriz, cache) e `stream` (pool de cópias > L3, como no decode real).
//! - `NS_BENCH_SKIP_15B=1` pula as formas do 1.5B (a head 65536x2048 exige ~0,6 GiB de RAM).

use std::process::ExitCode;

use ns_llm::quant::{QuantFormat, QuantizedMatrix, active_q4_backend};

#[path = "common/mod.rs"]
mod common;

#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    out_dim: usize,
    in_dim: usize,
    format: QuantFormat,
    iters: usize,
}

fn fmt_name(format: QuantFormat) -> &'static str {
    match format {
        QuantFormat::W8 => "W8",
        QuantFormat::Q4 => "Q4",
    }
}

/// Devolve `true` se a equivalência numérica passou.
fn bench_pair(shape: &Shape, tol: f64) -> Result<bool, Box<dyn std::error::Error>> {
    let Shape {
        label,
        out_dim,
        in_dim,
        format,
        iters,
    } = *shape;

    let weights = common::fill_weights(out_dim, in_dim, 0x5eed_1000 ^ (out_dim as u64));
    let xs: Vec<Vec<f32>> = (0..4)
        .map(|k| common::fill_input(in_dim, 0x5eed_2000 + k))
        .collect();

    let first = QuantizedMatrix::from_f32_with_format(&weights, out_dim, in_dim, format)?;
    drop(weights);
    let bytes = first.weight_bytes();
    let n = common::pool_len(bytes);
    let mut pool: Vec<QuantizedMatrix> = (1..n).map(|_| first.clone()).collect();
    pool.push(first);

    let mut q = [
        vec![0_i8; in_dim],
        vec![0_i8; in_dim],
        vec![0_i8; in_dim],
        vec![0_i8; in_dim],
    ];
    let mut o = [
        vec![0.0_f32; out_dim],
        vec![0.0_f32; out_dim],
        vec![0.0_f32; out_dim],
        vec![0.0_f32; out_dim],
    ];
    let mut q_seq = vec![0_i8; in_dim];
    let mut o_seq = [
        vec![0.0_f32; out_dim],
        vec![0.0_f32; out_dim],
        vec![0.0_f32; out_dim],
        vec![0.0_f32; out_dim],
    ];

    // Equivalência: 4x matvec vs 1x matmul4 (mesmas entradas).
    for k in 0..4 {
        pool[0].matvec_into(&xs[k], &mut q_seq, &mut o_seq[k]);
    }
    {
        let [q0, q1, q2, q3] = &mut q;
        let [o0, o1, o2, o3] = &mut o;
        pool[0].matmul4_into(
            [&xs[0], &xs[1], &xs[2], &xs[3]],
            [q0, q1, q2, q3],
            [o0, o1, o2, o3],
        );
    }
    let diff = (0..4)
        .map(|k| common::max_rel_diff(&o_seq[k], &o[k]))
        .fold(0.0_f64, f64::max);
    let ok = diff <= tol;

    for (mode, hot) in [("quente", true), ("stream", false)] {
        let (four_ms, m4_ms) = common::time_ab_ms(
            iters,
            |i| {
                let m = if hot { &pool[0] } else { &pool[i % pool.len()] };
                for k in 0..4 {
                    m.matvec_into(
                        std::hint::black_box(&xs[k]),
                        &mut q_seq,
                        std::hint::black_box(&mut o_seq[k]),
                    );
                }
            },
            |i| {
                let m = if hot { &pool[0] } else { &pool[i % pool.len()] };
                let [q0, q1, q2, q3] = &mut q;
                let [o0, o1, o2, o3] = &mut o;
                m.matmul4_into(
                    [&xs[0], &xs[1], &xs[2], &xs[3]],
                    [q0, q1, q2, q3],
                    [o0, o1, o2, o3],
                );
            },
        );
        println!(
            "{:<24} | {:<2} | {:<6} | {:>7.2} | {:>8.3} | {:>8.3} | {:>7.3} | {:>5.2}x | {:>9.2e}{}",
            label,
            fmt_name(format),
            mode,
            common::mib(bytes),
            four_ms,
            m4_ms,
            m4_ms / 4.0,
            four_ms / m4_ms,
            diff,
            if ok { "" } else { "  <-- DIVERGE" },
        );
    }

    std::hint::black_box((&o, &o_seq));
    Ok(ok)
}

fn main() -> ExitCode {
    let tol = std::env::var("NS_BENCH_TOL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1e-3);

    println!();
    println!("Micro-Matmul (M x 4) vs 4x Matvec (M x 1) | equivalência: tol {tol:e}");
    common::print_env(active_q4_backend());
    println!(
        "{:<24} | Q  | {:<6} | {:>7} | {:>8} | {:>8} | {:>7} | {:>6} | {:>9}",
        "Matriz", "modo", "MiB", "4x matvec", "1x mm4", "ms/ramo", "Ganho", "dif. rel."
    );
    println!("{}", "-".repeat(104));

    let mut shapes = vec![
        Shape {
            label: "0.1B cm.key (3072x768)",
            out_dim: 3072,
            in_dim: 768,
            format: QuantFormat::Q4,
            iters: 100,
        },
        Shape {
            label: "0.1B head (65536x768)",
            out_dim: 65536,
            in_dim: 768,
            format: QuantFormat::W8,
            iters: 12,
        },
        Shape {
            label: "0.1B head (65536x768)",
            out_dim: 65536,
            in_dim: 768,
            format: QuantFormat::Q4,
            iters: 12,
        },
    ];
    if !common::env_flag("NS_BENCH_SKIP_15B") {
        shapes.extend([
            Shape {
                label: "1.5B tm.proj (2048x2048)",
                out_dim: 2048,
                in_dim: 2048,
                format: QuantFormat::Q4,
                iters: 80,
            },
            Shape {
                label: "1.5B cm.key (7168x2048)",
                out_dim: 7168,
                in_dim: 2048,
                format: QuantFormat::Q4,
                iters: 40,
            },
            Shape {
                label: "1.5B head (65536x2048)",
                out_dim: 65536,
                in_dim: 2048,
                format: QuantFormat::W8,
                iters: 6,
            },
            Shape {
                label: "1.5B head (65536x2048)",
                out_dim: 65536,
                in_dim: 2048,
                format: QuantFormat::Q4,
                iters: 6,
            },
        ]);
    }

    let mut all_ok = true;
    for (idx, shape) in shapes.iter().enumerate() {
        if idx == 3 {
            println!("{}", "-".repeat(104));
        }
        match bench_pair(shape, tol) {
            Ok(ok) => all_ok &= ok,
            Err(err) => {
                eprintln!("erro em {}: {err}", shape.label);
                return ExitCode::FAILURE;
            }
        }
    }

    if all_ok {
        ExitCode::SUCCESS
    } else {
        eprintln!("matmul4 divergiu de 4x matvec acima da tolerância");
        ExitCode::FAILURE
    }
}
