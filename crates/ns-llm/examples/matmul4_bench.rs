use std::time::Instant;

use ns_llm::quant::{QuantFormat, QuantizedMatrix, active_q4_backend};

fn bench_pair(
    label: &str,
    out_dim: usize,
    in_dim: usize,
    format: QuantFormat,
    iters: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut weights = vec![0.0_f32; out_dim * in_dim];
    for r in 0..out_dim {
        let base = r * in_dim;
        for c in 0..in_dim {
            weights[base + c] = (((r * 19 + c * 7) % 61) as f32 - 30.0) * 0.015;
        }
    }
    let xs: Vec<Vec<f32>> = (0..4)
        .map(|k| {
            (0..in_dim)
                .map(|i| ((i as f32) * (0.17 + k as f32 * 0.03)).sin())
                .collect()
        })
        .collect();

    let mat = QuantizedMatrix::from_f32_with_format(&weights, out_dim, in_dim, format)?;

    let mut q0 = vec![0_i8; in_dim];
    let mut q1 = vec![0_i8; in_dim];
    let mut q2 = vec![0_i8; in_dim];
    let mut q3 = vec![0_i8; in_dim];
    let mut o0 = vec![0.0_f32; out_dim];
    let mut o1 = vec![0.0_f32; out_dim];
    let mut o2 = vec![0.0_f32; out_dim];
    let mut o3 = vec![0.0_f32; out_dim];

    // Warmup
    for _ in 0..3 {
        mat.matvec_into(&xs[0], &mut q0, &mut o0);
        mat.matmul4_into(
            [&xs[0], &xs[1], &xs[2], &xs[3]],
            [&mut q0, &mut q1, &mut q2, &mut q3],
            [&mut o0, &mut o1, &mut o2, &mut o3],
        );
    }

    // 4 chamadas independentes de matvec (lê pesos 4x da RAM)
    let t0 = Instant::now();
    for _ in 0..iters {
        mat.matvec_into(&xs[0], &mut q0, &mut o0);
        mat.matvec_into(&xs[1], &mut q1, &mut o1);
        mat.matvec_into(&xs[2], &mut q2, &mut o2);
        mat.matvec_into(&xs[3], &mut q3, &mut o3);
    }
    let four_matvec_ms = (t0.elapsed().as_secs_f64() * 1_000.0) / (iters as f64);

    // 1 chamada micro-matmul M x 4 (lê pesos 1x da RAM)
    let t1 = Instant::now();
    for _ in 0..iters {
        mat.matmul4_into(
            [&xs[0], &xs[1], &xs[2], &xs[3]],
            [&mut q0, &mut q1, &mut q2, &mut q3],
            [&mut o0, &mut o1, &mut o2, &mut o3],
        );
    }
    let matmul4_ms = (t1.elapsed().as_secs_f64() * 1_000.0) / (iters as f64);

    let ram_mib = (mat.weight_bytes() as f64) / (1024.0 * 1024.0);
    let speedup = four_matvec_ms / matmul4_ms;
    let per_branch_ms = matmul4_ms / 4.0;

    std::hint::black_box((&o0, &o1, &o2, &o3));
    println!(
        "{:<26} | {:<2} | {:>6.2} MiB | {:>7.3} ms | {:>7.3} ms | {:>6.3} ms | {:>5.2}x",
        label,
        match format {
            QuantFormat::W8 => "W8",
            QuantFormat::Q4 => "Q4",
        },
        ram_mib,
        four_matvec_ms,
        matmul4_ms,
        per_branch_ms,
        speedup
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!(
        "Micro-Matmul (M x 4) vs 4x Matvec (M x 1) | backend Q4: {}",
        active_q4_backend()
    );
    println!(
        "{:<26} | Q  | {:>10} | {:>10} | {:>10} | {:>9} | Ganho",
        "Matriz (0.1B e 1.5B)", "RAM", "4x matvec", "1x matmul4", "ms/ramo"
    );
    println!("{}", "-".repeat(92));

    // Perfis RWKV-7 0.1B (d=768)
    bench_pair("0.1B cm.key (3072x768)", 3072, 768, QuantFormat::Q4, 100)?;
    bench_pair("0.1B head (65536x768)", 65536, 768, QuantFormat::W8, 20)?;
    bench_pair("0.1B head (65536x768)", 65536, 768, QuantFormat::Q4, 20)?;
    println!("{}", "-".repeat(92));

    // Perfis RWKV-7 1.5B (d=2048, ffn=7168)
    bench_pair("1.5B tm.proj (2048x2048)", 2048, 2048, QuantFormat::Q4, 80)?;
    bench_pair("1.5B cm.key (7168x2048)", 7168, 2048, QuantFormat::Q4, 40)?;
    bench_pair("1.5B head (65536x2048)", 65536, 2048, QuantFormat::W8, 10)?;
    bench_pair("1.5B head (65536x2048)", 65536, 2048, QuantFormat::Q4, 10)?;

    Ok(())
}
