#![allow(dead_code)]

use std::time::Instant;

const DEFAULT_SAMPLES: usize = 9;
const DEFAULT_POOL_MIB: usize = 96;

#[must_use]
pub fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn bench_samples() -> usize {
    std::env::var("NS_BENCH_SAMPLES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_SAMPLES)
}

fn bench_pool_mib() -> usize {
    std::env::var("NS_BENCH_POOL_MIB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_POOL_MIB)
}

#[must_use]
pub fn pool_len(matrix_bytes: usize) -> usize {
    let target_bytes = bench_pool_mib().saturating_mul(1024 * 1024);
    target_bytes.div_ceil(matrix_bytes.max(1)).max(1)
}

#[must_use]
pub fn mib(bytes: usize) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0)
}

#[must_use]
pub fn gib_per_s(bytes: usize, ms: f64) -> f64 {
    let gib = (bytes as f64) / (1024.0 * 1024.0 * 1024.0);
    let secs = (ms / 1_000.0).max(1e-12);
    gib / secs
}

pub fn print_env(q4_backend: &str) {
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| {
        std::thread::available_parallelism()
            .map(|n| n.get().to_string())
            .unwrap_or_else(|_| "auto".to_owned())
    });
    println!(
        "backend Q4: {q4_backend} | threads: {threads} | amostras: {} | pool stream: {} MiB",
        bench_samples(),
        bench_pool_mib()
    );
}

struct SplitMix64(u64);

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_unit_signed(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        let u = (bits as f32) / ((1_u32 << 24) as f32);
        u * 2.0 - 1.0
    }
}

#[must_use]
pub fn fill_weights(out_dim: usize, in_dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let scale = 1.0 / (in_dim.max(1) as f32).sqrt();
    let total = out_dim.saturating_mul(in_dim);
    let mut out = Vec::with_capacity(total);
    for _ in 0..total {
        out.push(rng.next_unit_signed() * scale);
    }
    out
}

#[must_use]
pub fn fill_input(in_dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(in_dim);
    for _ in 0..in_dim {
        out.push(rng.next_unit_signed());
    }
    out
}

#[must_use]
pub fn matvec_ref(weights: &[f32], out_dim: usize, in_dim: usize, x: &[f32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(out_dim);
    for row in 0..out_dim {
        let base = row * in_dim;
        let mut acc = 0.0_f64;
        for col in 0..in_dim {
            acc += f64::from(weights[base + col]) * f64::from(x[col]);
        }
        out.push(acc);
    }
    out
}

#[must_use]
pub fn cosine(reference: &[f64], approx: &[f32]) -> f64 {
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (&r, &v) in reference.iter().zip(approx.iter()) {
        let vf = f64::from(v);
        dot += r * vf;
        na += r * r;
        nb += vf * vf;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 { 1.0 } else { dot / denom }
}

#[must_use]
pub fn max_rel_diff(a: &[f32], b: &[f32]) -> f64 {
    let mut max_abs = 0.0_f64;
    let mut max_err = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = f64::from(x);
        let yf = f64::from(y);
        max_abs = max_abs.max(xf.abs()).max(yf.abs());
        max_err = max_err.max((xf - yf).abs());
    }
    max_err / max_abs.max(1e-12)
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        0.5 * (samples[mid - 1] + samples[mid])
    } else {
        samples[mid]
    }
}

/// Mede A e B de forma intercalada (invertendo a ordem a cada amostra) e retorna a mediana em ms/iter.
pub fn time_ab_ms<FA, FB>(iters: usize, mut fa: FA, mut fb: FB) -> (f64, f64)
where
    FA: FnMut(usize),
    FB: FnMut(usize),
{
    let iters = iters.max(1);
    let n_samples = bench_samples();

    // Warmup curto
    for i in 0..iters.min(4) {
        fa(i);
        fb(i);
    }

    let mut a_ms = Vec::with_capacity(n_samples);
    let mut b_ms = Vec::with_capacity(n_samples);

    for s in 0..n_samples {
        let base = s * iters;
        if s % 2 == 0 {
            let t0 = Instant::now();
            for i in 0..iters {
                fa(base + i);
            }
            a_ms.push(t0.elapsed().as_secs_f64() * 1_000.0 / (iters as f64));

            let t1 = Instant::now();
            for i in 0..iters {
                fb(base + i);
            }
            b_ms.push(t1.elapsed().as_secs_f64() * 1_000.0 / (iters as f64));
        } else {
            let t1 = Instant::now();
            for i in 0..iters {
                fb(base + i);
            }
            b_ms.push(t1.elapsed().as_secs_f64() * 1_000.0 / (iters as f64));

            let t0 = Instant::now();
            for i in 0..iters {
                fa(base + i);
            }
            a_ms.push(t0.elapsed().as_secs_f64() * 1_000.0 / (iters as f64));
        }
    }

    (median(&mut a_ms), median(&mut b_ms))
}
