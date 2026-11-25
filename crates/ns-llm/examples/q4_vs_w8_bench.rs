use std::time::Instant;

use ns_llm::quant::{QuantFormat, QuantizedMatrix};

fn bench_shape(label: &str, out_dim: usize, in_dim: usize, iters: usize) {
    let mut weights = vec![0.0_f32; out_dim * in_dim];
    for r in 0..out_dim {
        let base = r * in_dim;
        for c in 0..in_dim {
            weights[base + c] = (((r * 19 + c * 7) % 61) as f32 - 30.0) * 0.015;
        }
    }
    let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.17).sin()).collect();

    let w8 = QuantizedMatrix::from_f32_with_format(&weights, out_dim, in_dim, QuantFormat::W8)
        .expect("valid W8");
    let q4 = QuantizedMatrix::from_f32_with_format(&weights, out_dim, in_dim, QuantFormat::Q4)
        .expect("valid Q4");

    let mut q_in = vec![0_i8; in_dim];
    let mut out_w8 = vec![0.0_f32; out_dim];
    let mut out_q4 = vec![0.0_f32; out_dim];

    // Warmup
    for _ in 0..5 {
        w8.matvec_into(&x, &mut q_in, &mut out_w8);
        q4.matvec_into(&x, &mut q_in, &mut out_q4);
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        w8.matvec_into(&x, &mut q_in, &mut out_w8);
    }
    let w8_ms = (t0.elapsed().as_secs_f64() * 1_000.0) / (iters as f64);

    let t1 = Instant::now();
    for _ in 0..iters {
        q4.matvec_into(&x, &mut q_in, &mut out_q4);
    }
    let q4_ms = (t1.elapsed().as_secs_f64() * 1_000.0) / (iters as f64);

    let dot: f32 = out_w8.iter().zip(&out_q4).map(|(a, b)| a * b).sum();
    let n8: f32 = out_w8.iter().map(|v| v * v).sum::<f32>().sqrt();
    let n4: f32 = out_q4.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cos = dot / (n8 * n4).max(1e-12);

    let w8_mib = (w8.weight_bytes() as f64) / (1024.0 * 1024.0);
    let q4_mib = (q4.weight_bytes() as f64) / (1024.0 * 1024.0);
    let speedup = w8_ms / q4_ms;

    std::hint::black_box((&out_w8, &out_q4));
    println!(
        "{:<22} | {:>6.2} MiB | {:>6.2} MiB | {:>6.3} ms | {:>6.3} ms | {:>5.2}x | {:.5}",
        label, w8_mib, q4_mib, w8_ms, q4_ms, speedup, cos
    );
}

fn main() {
    println!();
    println!("Comparativo QuantizedMatrix: W8A8 vs Q4A8 (bloco 32, AVX2+FMA+F16C)");
    println!(
        "{:<22} | {:>10} | {:>10} | {:>9} | {:>9} | {:>6} | CosSim",
        "Matriz (RWKV-7 0.1B)", "RAM W8", "RAM Q4", "W8/cham.", "Q4/cham.", "Ganho"
    );
    println!("{}", "-".repeat(88));

    bench_shape("tm.proj (768x768)", 768, 768, 400);
    bench_shape("cm.key (3072x768)", 3072, 768, 200);
    bench_shape("head (65536x768)", 65536, 768, 40);
}
