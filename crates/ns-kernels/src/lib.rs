#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

/// Quantidade de pesos por bloco na quantização simétrica 4-bit (`Q4_0`: 18 bytes / 32 pesos = 0,5625 B/peso).
pub const Q4_BLOCK_SIZE: usize = 32;

/// Bloco compacto de 32 pesos 4-bit (`[-8, +7]` armazenados com offset `+8` em `[0, 15]`).
///
/// Layout *split-half*:
/// - nibble baixo de `qs[j]` = peso `j` (`0..16`)
/// - nibble alto  de `qs[j]` = peso `j + 16` (`16..32`)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Q4Block {
    pub scale_f16: u16,
    pub qs: [u8; Q4_BLOCK_SIZE / 2],
}

const _: () = assert!(std::mem::size_of::<Q4Block>() == 18);

/// Retorna o nome do backend SIMD efetivamente ativo na CPU atual.
#[must_use]
pub fn active_q4_backend() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            return "x86_64 avx2+fma+f16c";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "aarch64 neon";
    }
    #[cfg(not(target_arch = "aarch64"))]
    "scalar"
}

/// Conversão IEEE-754 `f32 -> binary16` sem dependências externas (satura em `65504.0`).
#[inline]
#[must_use]
pub fn f32_to_f16(value: f32) -> u16 {
    if !value.is_finite() || value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    #[allow(clippy::cast_possible_truncation)]
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (((bits >> 23) & 0xFF) as i32) - 127 + 15;
    let mant = bits & 0x007F_FFFF;

    if exp >= 31 {
        return sign | 0x7BFF;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let full_mant = mant | 0x0080_0000;
        #[allow(clippy::cast_sign_loss)]
        let shift = (14 - exp) as u32;
        #[allow(clippy::cast_possible_truncation)]
        let half_mant = ((full_mant + (1_u32 << (shift - 1))) >> shift) as u16;
        return sign | half_mant;
    }

    let rounded = mant + 0x0000_1000;
    if rounded >= 0x0080_0000 {
        let next_exp = exp + 1;
        if next_exp >= 31 {
            return sign | 0x7BFF;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        return sign | ((next_exp as u16) << 10);
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let exp_u16 = exp as u16;
    #[allow(clippy::cast_possible_truncation)]
    let mant_u16 = (rounded >> 13) as u16;
    sign | (exp_u16 << 10) | mant_u16
}

/// Conversão IEEE-754 `binary16 -> f32` (usa `vcvtph2ps` quando `f16c` está habilitado).
#[inline]
#[must_use]
pub fn f16_to_f32(bits: u16) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "f16c"))]
    {
        use std::arch::x86_64::{_mm_cvtph_ps, _mm_cvtsi32_si128, _mm_cvtss_f32};
        // SAFETY: `f16c` garantido pelo atributo `#[cfg(..., target_feature = "f16c")]`.
        unsafe { _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(bits)))) }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "f16c")))]
    {
        f16_to_f32_scalar(bits)
    }
}

#[inline]
fn f16_to_f32_scalar(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x03FF);

    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let shift = mant.leading_zeros() - 21;
            let norm_mant = (mant << shift) & 0x03FF;
            let f32_exp = 127 - 14 - shift;
            sign | (f32_exp << 23) | (norm_mant << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + (127 - 15)) << 23) | (mant << 13)
    };

    f32::from_bits(f32_bits)
}

/// Produto escalar Q4 × Q8 de uma única linha, despachando automaticamente para o melhor SIMD.
#[inline]
#[must_use]
pub fn dot_q4_row(
    row_blocks: &[Q4Block],
    qx: &[i8],
    x_scales: &[f32],
    x_bias: &[f32],
    in_dim: usize,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            // SAFETY: AVX2, FMA e F16C verificados em runtime acima.
            return unsafe { dot_q4_row_avx2(row_blocks, qx, x_scales, x_bias, in_dim) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON é obrigatório na arquitetura base aarch64.
        return unsafe { dot_q4_row_neon(row_blocks, qx, x_scales, x_bias, in_dim) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    dot_q4_row_scalar(row_blocks, qx, x_scales, x_bias, in_dim)
}

/// Calcula uma faixa contígua de linhas Q4 (`out_slice[i] = row(row0 + i) · x`),
/// checando CPUID uma única vez por lote e aplicando `PREFETCHNTA` quando `use_nta = true`.
#[allow(clippy::too_many_arguments)]
pub fn compute_q4_row_range(
    blocks: &[Q4Block],
    blocks_per_row: usize,
    row0: usize,
    out_slice: &mut [f32],
    qx: &[i8],
    x_scales: &[f32],
    x_bias: &[f32],
    in_dim: usize,
    use_nta: bool,
) {
    assert!(blocks_per_row >= in_dim.div_ceil(Q4_BLOCK_SIZE));
    assert!(qx.len() >= in_dim);
    assert!(x_scales.len() >= blocks_per_row);
    assert!(x_bias.len() >= blocks_per_row);
    assert!(
        row0.saturating_add(out_slice.len())
            .saturating_mul(blocks_per_row)
            <= blocks.len()
    );

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            // SAFETY: AVX2, FMA e F16C verificados em runtime e limites validados acima.
            unsafe {
                compute_q4_row_range_avx2(
                    blocks,
                    blocks_per_row,
                    row0,
                    out_slice,
                    qx,
                    x_scales,
                    x_bias,
                    in_dim,
                    use_nta,
                );
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let _ = use_nta;
        for (i, dst) in out_slice.iter_mut().enumerate() {
            let row = row0 + i;
            let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
            // SAFETY: NEON é padrão em aarch64 e limites foram validados acima.
            *dst = unsafe { dot_q4_row_neon(row_blocks, qx, x_scales, x_bias, in_dim) };
        }
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = use_nta;
        for (i, dst) in out_slice.iter_mut().enumerate() {
            let row = row0 + i;
            let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
            *dst = dot_q4_row_scalar(row_blocks, qx, x_scales, x_bias, in_dim);
        }
    }
}

/// Produto escalar Q4 × Q8 simultâneo para **4 vetores de entrada** lendo a linha de pesos
/// (e descompactando nibbles + escala `f16`) **uma única vez**.
#[inline]
#[must_use]
pub fn dot_q4_row_x4(
    row_blocks: &[Q4Block],
    qx: [&[i8]; 4],
    x_scales: [&[f32]; 4],
    x_bias: [&[f32]; 4],
    in_dim: usize,
) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            // SAFETY: AVX2, FMA e F16C verificados em runtime acima.
            return unsafe { dot_q4_row_x4_avx2(row_blocks, qx, x_scales, x_bias, in_dim) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON é obrigatório na arquitetura base aarch64.
        return unsafe { dot_q4_row_x4_neon(row_blocks, qx, x_scales, x_bias, in_dim) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    dot_q4_row_x4_scalar(row_blocks, qx, x_scales, x_bias, in_dim)
}

/// Calcula uma faixa contígua de linhas Q4 contra **4 vetores em paralelo** (`M × 4`),
/// lendo os pesos da RAM uma única vez por faixa e aplicando `PREFETCHNTA` quando `use_nta = true`.
#[allow(clippy::too_many_arguments)]
pub fn compute_q4_row_range_x4(
    blocks: &[Q4Block],
    blocks_per_row: usize,
    row0: usize,
    out_slices: [&mut [f32]; 4],
    qx: [&[i8]; 4],
    x_scales: [&[f32]; 4],
    x_bias: [&[f32]; 4],
    in_dim: usize,
    use_nta: bool,
) {
    let count = out_slices[0].len();
    assert!(out_slices.iter().all(|s| s.len() == count));
    assert!(blocks_per_row >= in_dim.div_ceil(Q4_BLOCK_SIZE));
    assert!(qx.iter().all(|q| q.len() >= in_dim));
    assert!(x_scales.iter().all(|s| s.len() >= blocks_per_row));
    assert!(x_bias.iter().all(|b| b.len() >= blocks_per_row));
    assert!(
        row0.saturating_add(count).saturating_mul(blocks_per_row) <= blocks.len()
    );

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            // SAFETY: AVX2, FMA e F16C verificados em runtime e limites validados acima.
            unsafe {
                compute_q4_row_range_x4_avx2(
                    blocks,
                    blocks_per_row,
                    row0,
                    out_slices,
                    qx,
                    x_scales,
                    x_bias,
                    in_dim,
                    use_nta,
                );
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let _ = use_nta;
        let [o0, o1, o2, o3] = out_slices;
        for i in 0..count {
            let row = row0 + i;
            let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
            // SAFETY: NEON é padrão em aarch64 e limites foram validados acima.
            let vals = unsafe { dot_q4_row_x4_neon(row_blocks, qx, x_scales, x_bias, in_dim) };
            o0[i] = vals[0];
            o1[i] = vals[1];
            o2[i] = vals[2];
            o3[i] = vals[3];
        }
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = use_nta;
        let [o0, o1, o2, o3] = out_slices;
        for i in 0..count {
            let row = row0 + i;
            let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
            let vals = dot_q4_row_x4_scalar(row_blocks, qx, x_scales, x_bias, in_dim);
            o0[i] = vals[0];
            o1[i] = vals[1];
            o2[i] = vals[2];
            o3[i] = vals[3];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn compute_q4_row_range_x4_avx2(
    blocks: &[Q4Block],
    blocks_per_row: usize,
    row0: usize,
    out_slices: [&mut [f32]; 4],
    qx: [&[i8]; 4],
    x_scales: [&[f32]; 4],
    x_bias: [&[f32]; 4],
    in_dim: usize,
    use_nta: bool,
) {
    use std::arch::x86_64::{_MM_HINT_NTA, _mm_prefetch};

    let [o0, o1, o2, o3] = out_slices;
    let count = o0.len();
    for i in 0..count {
        let row = row0 + i;
        if use_nta && i + 1 < count {
            let next_row = &blocks[(row + 1) * blocks_per_row..(row + 2) * blocks_per_row];
            let p0 = next_row.as_ptr().cast::<i8>();
            // SAFETY: `p0` aponta para dentro de `blocks`; `_mm_prefetch` é apenas hint de cache.
            unsafe {
                _mm_prefetch::<_MM_HINT_NTA>(p0);
                if blocks_per_row >= 8 {
                    _mm_prefetch::<_MM_HINT_NTA>(p0.add(256));
                }
            }
        }
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        // SAFETY: `#[target_feature]` ativo e limites validados pelo chamador.
        let vals = unsafe { dot_q4_row_x4_avx2(row_blocks, qx, x_scales, x_bias, in_dim) };
        o0[i] = vals[0];
        o1[i] = vals[1];
        o2[i] = vals[2];
        o3[i] = vals[3];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::{
        _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehdup_ps, _mm_movehl_ps,
        _mm256_castps256_ps128, _mm256_extractf128_ps,
    };
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// Kernel AVX2 + FMA + F16C para `M × 4`:
/// descompacta cada `Q4Block` 1 única vez e satura 4 acumuladores YMM independentes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q4_row_x4_avx2(
    blocks: &[Q4Block],
    qx: [&[i8]; 4],
    x_scales: [&[f32]; 4],
    x_bias: [&[f32]; 4],
    in_dim: usize,
) -> [f32; 4] {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm_and_si128, _mm_cvtph_ps, _mm_cvtsi32_si128, _mm_cvtss_f32,
        _mm_loadu_si128, _mm_set1_epi8, _mm_srli_epi16, _mm256_cvtepi32_ps, _mm256_fmadd_ps,
        _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set_m128i,
        _mm256_set1_epi16, _mm256_set1_ps, _mm256_setzero_ps,
    };

    let full_blocks = in_dim / Q4_BLOCK_SIZE;
    let rem = in_dim % Q4_BLOCK_SIZE;

    // SAFETY: AVX2/FMA/F16C ativos pelo `#[target_feature]`; limites garantidos pelo chamador.
    unsafe {
        let mask_0f = _mm_set1_epi8(0x0F);
        let ones_16 = _mm256_set1_epi16(1);

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut b0 = 0.0_f32;
        let mut b1 = 0.0_f32;
        let mut b2 = 0.0_f32;
        let mut b3 = 0.0_f32;

        let p0 = qx[0].as_ptr();
        let p1 = qx[1].as_ptr();
        let p2 = qx[2].as_ptr();
        let p3 = qx[3].as_ptr();

        for b in 0..full_blocks {
            let blk = blocks.get_unchecked(b);
            let ws = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(blk.scale_f16))));

            // Descompactação 1x compartilhada pelos 4 vetores de entrada
            let raw = _mm_loadu_si128(blk.qs.as_ptr().cast::<__m128i>());
            let w_u4 = _mm256_set_m128i(
                _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f),
                _mm_and_si128(raw, mask_0f),
            );

            let off = b * Q4_BLOCK_SIZE;

            let q0 = _mm256_loadu_si256(p0.add(off).cast::<__m256i>());
            let prod0 =
                _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(w_u4, q0), ones_16));
            acc0 = _mm256_fmadd_ps(prod0, _mm256_set1_ps(ws * *x_scales[0].get_unchecked(b)), acc0);
            b0 = ws.mul_add(*x_bias[0].get_unchecked(b), b0);

            let q1 = _mm256_loadu_si256(p1.add(off).cast::<__m256i>());
            let prod1 =
                _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(w_u4, q1), ones_16));
            acc1 = _mm256_fmadd_ps(prod1, _mm256_set1_ps(ws * *x_scales[1].get_unchecked(b)), acc1);
            b1 = ws.mul_add(*x_bias[1].get_unchecked(b), b1);

            let q2 = _mm256_loadu_si256(p2.add(off).cast::<__m256i>());
            let prod2 =
                _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(w_u4, q2), ones_16));
            acc2 = _mm256_fmadd_ps(prod2, _mm256_set1_ps(ws * *x_scales[2].get_unchecked(b)), acc2);
            b2 = ws.mul_add(*x_bias[2].get_unchecked(b), b2);

            let q3 = _mm256_loadu_si256(p3.add(off).cast::<__m256i>());
            let prod3 =
                _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(w_u4, q3), ones_16));
            acc3 = _mm256_fmadd_ps(prod3, _mm256_set1_ps(ws * *x_scales[3].get_unchecked(b)), acc3);
            b3 = ws.mul_add(*x_bias[3].get_unchecked(b), b3);
        }

        if rem > 0 {
            let b_tail = full_blocks;
            let blk = blocks.get_unchecked(b_tail);
            let ws = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(blk.scale_f16))));
            let raw = _mm_loadu_si128(blk.qs.as_ptr().cast::<__m128i>());
            let w_u4 = _mm256_set_m128i(
                _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f),
                _mm_and_si128(raw, mask_0f),
            );

            let mut t0 = [0_i8; Q4_BLOCK_SIZE];
            let mut t1 = [0_i8; Q4_BLOCK_SIZE];
            let mut t2 = [0_i8; Q4_BLOCK_SIZE];
            let mut t3 = [0_i8; Q4_BLOCK_SIZE];
            let base = b_tail * Q4_BLOCK_SIZE;
            t0[..rem].copy_from_slice(&qx[0][base..in_dim]);
            t1[..rem].copy_from_slice(&qx[1][base..in_dim]);
            t2[..rem].copy_from_slice(&qx[2][base..in_dim]);
            t3[..rem].copy_from_slice(&qx[3][base..in_dim]);

            let prod0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(
                _mm256_maddubs_epi16(w_u4, _mm256_loadu_si256(t0.as_ptr().cast::<__m256i>())),
                ones_16,
            ));
            acc0 = _mm256_fmadd_ps(prod0, _mm256_set1_ps(ws * *x_scales[0].get_unchecked(b_tail)), acc0);
            b0 = ws.mul_add(*x_bias[0].get_unchecked(b_tail), b0);

            let prod1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(
                _mm256_maddubs_epi16(w_u4, _mm256_loadu_si256(t1.as_ptr().cast::<__m256i>())),
                ones_16,
            ));
            acc1 = _mm256_fmadd_ps(prod1, _mm256_set1_ps(ws * *x_scales[1].get_unchecked(b_tail)), acc1);
            b1 = ws.mul_add(*x_bias[1].get_unchecked(b_tail), b1);

            let prod2 = _mm256_cvtepi32_ps(_mm256_madd_epi16(
                _mm256_maddubs_epi16(w_u4, _mm256_loadu_si256(t2.as_ptr().cast::<__m256i>())),
                ones_16,
            ));
            acc2 = _mm256_fmadd_ps(prod2, _mm256_set1_ps(ws * *x_scales[2].get_unchecked(b_tail)), acc2);
            b2 = ws.mul_add(*x_bias[2].get_unchecked(b_tail), b2);

            let prod3 = _mm256_cvtepi32_ps(_mm256_madd_epi16(
                _mm256_maddubs_epi16(w_u4, _mm256_loadu_si256(t3.as_ptr().cast::<__m256i>())),
                ones_16,
            ));
            acc3 = _mm256_fmadd_ps(prod3, _mm256_set1_ps(ws * *x_scales[3].get_unchecked(b_tail)), acc3);
            b3 = ws.mul_add(*x_bias[3].get_unchecked(b_tail), b3);
        }

        [
            hsum256_ps(acc0) + b0,
            hsum256_ps(acc1) + b1,
            hsum256_ps(acc2) + b2,
            hsum256_ps(acc3) + b3,
        ]
    }
}

/// Kernel ARM64 NEON para `M × 4` (Raspberry Pi 4 / Pi 5): descompacta cada `Q4Block` 1×.
#[cfg(target_arch = "aarch64")]
unsafe fn dot_q4_row_x4_neon(
    blocks: &[Q4Block],
    qx: [&[i8]; 4],
    x_scales: [&[f32]; 4],
    x_bias: [&[f32]; 4],
    in_dim: usize,
) -> [f32; 4] {
    use core::arch::aarch64::{
        vaddlvq_s16, vandq_u8, vdupq_n_u8, vget_high_s8, vget_low_s8, vld1q_s8, vld1q_u8, vmlal_s8,
        vmull_s8, vreinterpretq_s8_u8, vshrq_n_u8,
    };

    let full_blocks = in_dim / Q4_BLOCK_SIZE;
    let rem = in_dim % Q4_BLOCK_SIZE;
    let mut totals = [0.0_f32; 4];

    // SAFETY: NEON é baseline em aarch64 e os offsets são limitados por `full_blocks` e `rem`.
    unsafe {
        let mask_0f = vdupq_n_u8(0x0F);

        for b in 0..full_blocks {
            let blk = blocks.get_unchecked(b);
            let w_scale = f16_to_f32_scalar(blk.scale_f16);
            if w_scale == 0.0 {
                continue;
            }
            let raw = vld1q_u8(blk.qs.as_ptr());
            let lo_s8 = vreinterpretq_s8_u8(vandq_u8(raw, mask_0f));
            let hi_s8 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(raw));
            let lo_l = vget_low_s8(lo_s8);
            let lo_h = vget_high_s8(lo_s8);
            let hi_l = vget_low_s8(hi_s8);
            let hi_h = vget_high_s8(hi_s8);

            let off = b * Q4_BLOCK_SIZE;
            for k in 0..4 {
                let xs = *x_scales[k].get_unchecked(b);
                if xs == 0.0 {
                    continue;
                }
                let qptr = qx[k].as_ptr().add(off);
                let qlo = vld1q_s8(qptr);
                let qhi = vld1q_s8(qptr.add(Q4_BLOCK_SIZE / 2));

                let mut acc16 = vmull_s8(lo_l, vget_low_s8(qlo));
                acc16 = vmlal_s8(acc16, lo_h, vget_high_s8(qlo));
                acc16 = vmlal_s8(acc16, hi_l, vget_low_s8(qhi));
                acc16 = vmlal_s8(acc16, hi_h, vget_high_s8(qhi));

                let acc = vaddlvq_s16(acc16);
                totals[k] = (acc as f32).mul_add(
                    w_scale * xs,
                    w_scale.mul_add(*x_bias[k].get_unchecked(b), totals[k]),
                );
            }
        }

        if rem > 0 {
            let b = full_blocks;
            let blk = blocks.get_unchecked(b);
            let w_scale = f16_to_f32_scalar(blk.scale_f16);
            if w_scale != 0.0 {
                let raw = vld1q_u8(blk.qs.as_ptr());
                let lo_s8 = vreinterpretq_s8_u8(vandq_u8(raw, mask_0f));
                let hi_s8 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(raw));
                let lo_l = vget_low_s8(lo_s8);
                let lo_h = vget_high_s8(lo_s8);
                let hi_l = vget_low_s8(hi_s8);
                let hi_h = vget_high_s8(hi_s8);
                let base = b * Q4_BLOCK_SIZE;

                for k in 0..4 {
                    let xs = *x_scales[k].get_unchecked(b);
                    if xs == 0.0 {
                        continue;
                    }
                    let mut tail = [0_i8; Q4_BLOCK_SIZE];
                    tail[..rem].copy_from_slice(&qx[k][base..in_dim]);
                    let qlo = vld1q_s8(tail.as_ptr());
                    let qhi = vld1q_s8(tail.as_ptr().add(Q4_BLOCK_SIZE / 2));

                    let mut acc16 = vmull_s8(lo_l, vget_low_s8(qlo));
                    acc16 = vmlal_s8(acc16, lo_h, vget_high_s8(qlo));
                    acc16 = vmlal_s8(acc16, hi_l, vget_low_s8(qhi));
                    acc16 = vmlal_s8(acc16, hi_h, vget_high_s8(qhi));

                    let acc = vaddlvq_s16(acc16);
                    totals[k] = (acc as f32).mul_add(
                        w_scale * xs,
                        w_scale.mul_add(*x_bias[k].get_unchecked(b), totals[k]),
                    );
                }
            }
        }
    }

    totals
}

/// Referência escalar `M × 4` (lê cada bloco uma única vez para os 4 vetores).
#[must_use]
pub fn dot_q4_row_x4_scalar(
    blocks: &[Q4Block],
    qx: [&[i8]; 4],
    x_scales: [&[f32]; 4],
    x_bias: [&[f32]; 4],
    in_dim: usize,
) -> [f32; 4] {
    let half = Q4_BLOCK_SIZE / 2;
    let mut totals = [0.0_f32; 4];

    for (b, blk) in blocks.iter().enumerate() {
        let w_scale = f16_to_f32_scalar(blk.scale_f16);
        if w_scale == 0.0 {
            continue;
        }
        let base = b * Q4_BLOCK_SIZE;
        let end = (base + Q4_BLOCK_SIZE).min(in_dim);
        let mut accs = [0_i32; 4];

        for (j, &packed) in blk.qs.iter().enumerate() {
            let lo = i32::from(packed & 0x0F);
            let hi = i32::from(packed >> 4);
            let idx0 = base + j;
            let idx1 = base + j + half;
            if idx0 < end {
                for k in 0..4 {
                    accs[k] += lo * i32::from(qx[k][idx0]);
                }
            }
            if idx1 < end {
                for k in 0..4 {
                    accs[k] += hi * i32::from(qx[k][idx1]);
                }
            }
        }

        for k in 0..4 {
            let combined = w_scale * x_scales[k][b];
            if combined != 0.0 {
                totals[k] = (accs[k] as f32).mul_add(
                    combined,
                    w_scale.mul_add(x_bias[k][b], totals[k]),
                );
            }
        }
    }

    totals
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn compute_q4_row_range_avx2(
    blocks: &[Q4Block],
    blocks_per_row: usize,
    row0: usize,
    out_slice: &mut [f32],
    qx: &[i8],
    x_scales: &[f32],
    x_bias: &[f32],
    in_dim: usize,
    use_nta: bool,
) {
    use std::arch::x86_64::{_mm_prefetch, _MM_HINT_NTA};

    let count = out_slice.len();
    for (i, dst) in out_slice.iter_mut().enumerate() {
        let row = row0 + i;
        if use_nta && i + 1 < count {
            let next_row = &blocks[(row + 1) * blocks_per_row..(row + 2) * blocks_per_row];
            let p0 = next_row.as_ptr().cast::<i8>();
            // SAFETY: `p0` aponta para dentro de `blocks`; `_mm_prefetch` é apenas hint de cache.
            unsafe {
                _mm_prefetch::<_MM_HINT_NTA>(p0);
                if blocks_per_row >= 8 {
                    _mm_prefetch::<_MM_HINT_NTA>(p0.add(256));
                }
            }
        }
        let row_blocks = &blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        // SAFETY: `#[target_feature]` ativo e limites validados pelo chamador.
        *dst = unsafe { dot_q4_row_avx2(row_blocks, qx, x_scales, x_bias, in_dim) };
    }
}

/// Kernel x86-64 AVX2 + FMA + F16C com 2 acumuladores independentes (desenrolado 2× por bloco).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dot_q4_row_avx2(
    blocks: &[Q4Block],
    qx: &[i8],
    x_scales: &[f32],
    x_bias: &[f32],
    in_dim: usize,
) -> f32 {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm256_add_ps, _mm256_castps256_ps128, _mm256_cvtepi32_ps,
        _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_set1_ps, _mm256_set_m128i,
        _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_and_si128, _mm_cvtph_ps, _mm_cvtsi32_si128,
        _mm_cvtss_f32, _mm_loadu_si128, _mm_movehdup_ps, _mm_movehl_ps, _mm_set1_epi8,
        _mm_srli_epi16,
    };

    let full_blocks = in_dim / Q4_BLOCK_SIZE;
    let rem = in_dim % Q4_BLOCK_SIZE;
    debug_assert!(blocks.len() >= in_dim.div_ceil(Q4_BLOCK_SIZE));
    debug_assert!(qx.len() >= in_dim);

    // SAFETY: AVX2/FMA/F16C ativos pelo `#[target_feature]`; índices limitados por `full_blocks` e `rem`.
    unsafe {
        let mask_0f = _mm_set1_epi8(0x0F);
        let ones_16 = _mm256_set1_epi16(1);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut bias0 = 0.0_f32;
        let mut bias1 = 0.0_f32;

        let mut b = 0_usize;
        while b + 1 < full_blocks {
            let blk0 = blocks.get_unchecked(b);
            let blk1 = blocks.get_unchecked(b + 1);

            let ws0 = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(blk0.scale_f16))));
            let ws1 = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(blk1.scale_f16))));

            let c0 = ws0 * *x_scales.get_unchecked(b);
            let c1 = ws1 * *x_scales.get_unchecked(b + 1);
            bias0 = ws0.mul_add(*x_bias.get_unchecked(b), bias0);
            bias1 = ws1.mul_add(*x_bias.get_unchecked(b + 1), bias1);

            let raw0 = _mm_loadu_si128(blk0.qs.as_ptr().cast::<__m128i>());
            let raw1 = _mm_loadu_si128(blk1.qs.as_ptr().cast::<__m128i>());

            let w0 = _mm256_set_m128i(
                _mm_and_si128(_mm_srli_epi16(raw0, 4), mask_0f),
                _mm_and_si128(raw0, mask_0f),
            );
            let w1 = _mm256_set_m128i(
                _mm_and_si128(_mm_srli_epi16(raw1, 4), mask_0f),
                _mm_and_si128(raw1, mask_0f),
            );

            let qx0 = _mm256_loadu_si256(qx.as_ptr().add(b * Q4_BLOCK_SIZE).cast::<__m256i>());
            let qx1 =
                _mm256_loadu_si256(qx.as_ptr().add((b + 1) * Q4_BLOCK_SIZE).cast::<__m256i>());

            let p0 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(w0, qx0), ones_16));
            let p1 = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(w1, qx1), ones_16));

            acc0 = _mm256_fmadd_ps(p0, _mm256_set1_ps(c0), acc0);
            acc1 = _mm256_fmadd_ps(p1, _mm256_set1_ps(c1), acc1);
            b += 2;
        }

        if b < full_blocks {
            let blk = blocks.get_unchecked(b);
            let ws = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(blk.scale_f16))));
            let combined = ws * *x_scales.get_unchecked(b);
            bias0 = ws.mul_add(*x_bias.get_unchecked(b), bias0);

            let raw = _mm_loadu_si128(blk.qs.as_ptr().cast::<__m128i>());
            let w = _mm256_set_m128i(
                _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f),
                _mm_and_si128(raw, mask_0f),
            );
            let qx_vec = _mm256_loadu_si256(qx.as_ptr().add(b * Q4_BLOCK_SIZE).cast::<__m256i>());
            let prod =
                _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(w, qx_vec), ones_16));
            acc0 = _mm256_fmadd_ps(prod, _mm256_set1_ps(combined), acc0);
        }

        if rem > 0 {
            let b_tail = full_blocks;
            let blk = blocks.get_unchecked(b_tail);
            let ws = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(blk.scale_f16))));
            let combined = ws * *x_scales.get_unchecked(b_tail);
            bias1 = ws.mul_add(*x_bias.get_unchecked(b_tail), bias1);

            let mut tail_qx = [0_i8; Q4_BLOCK_SIZE];
            tail_qx[..rem].copy_from_slice(&qx[b_tail * Q4_BLOCK_SIZE..in_dim]);

            let raw = _mm_loadu_si128(blk.qs.as_ptr().cast::<__m128i>());
            let w = _mm256_set_m128i(
                _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f),
                _mm_and_si128(raw, mask_0f),
            );
            let qx_vec = _mm256_loadu_si256(tail_qx.as_ptr().cast::<__m256i>());
            let prod =
                _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(w, qx_vec), ones_16));
            acc1 = _mm256_fmadd_ps(prod, _mm256_set1_ps(combined), acc1);
        }

        let acc_f32 = _mm256_add_ps(acc0, acc1);
        let hi128 = _mm256_extractf128_ps(acc_f32, 1);
        let lo128 = _mm256_castps256_ps128(acc_f32);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2)) + (bias0 + bias1)
    }
}

/// Kernel ARM64 NEON compatível com Raspberry Pi 4 (Cortex-A72) e Raspberry Pi 5 (Cortex-A76).
#[cfg(target_arch = "aarch64")]
unsafe fn dot_q4_row_neon(
    blocks: &[Q4Block],
    qx: &[i8],
    x_scales: &[f32],
    x_bias: &[f32],
    in_dim: usize,
) -> f32 {
    use core::arch::aarch64::{
        vaddlvq_s16, vandq_u8, vdupq_n_u8, vget_high_s8, vget_low_s8, vld1q_s8, vld1q_u8, vmlal_s8,
        vmull_s8, vreinterpretq_s8_u8, vshrq_n_u8,
    };

    let full_blocks = in_dim / Q4_BLOCK_SIZE;
    let rem = in_dim % Q4_BLOCK_SIZE;
    let mut total = 0.0_f32;

    // SAFETY: NEON é baseline em aarch64 e os offsets são limitados por `full_blocks` e `rem`.
    unsafe {
        let mask_0f = vdupq_n_u8(0x0F);

        for b in 0..full_blocks {
            let blk = blocks.get_unchecked(b);
            let w_scale = f16_to_f32_scalar(blk.scale_f16);
            let combined = w_scale * *x_scales.get_unchecked(b);
            if combined == 0.0 {
                continue;
            }

            let raw = vld1q_u8(blk.qs.as_ptr());
            let lo_s8 = vreinterpretq_s8_u8(vandq_u8(raw, mask_0f));
            let hi_s8 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(raw));

            let qx_ptr = qx.as_ptr().add(b * Q4_BLOCK_SIZE);
            let qx_lo = vld1q_s8(qx_ptr);
            let qx_hi = vld1q_s8(qx_ptr.add(Q4_BLOCK_SIZE / 2));

            let mut acc16 = vmull_s8(vget_low_s8(lo_s8), vget_low_s8(qx_lo));
            acc16 = vmlal_s8(acc16, vget_high_s8(lo_s8), vget_high_s8(qx_lo));
            acc16 = vmlal_s8(acc16, vget_low_s8(hi_s8), vget_low_s8(qx_hi));
            acc16 = vmlal_s8(acc16, vget_high_s8(hi_s8), vget_high_s8(qx_hi));

            let acc = vaddlvq_s16(acc16);
            total =
                (acc as f32).mul_add(combined, w_scale.mul_add(*x_bias.get_unchecked(b), total));
        }

        if rem > 0 {
            let b = full_blocks;
            let blk = blocks.get_unchecked(b);
            let w_scale = f16_to_f32_scalar(blk.scale_f16);
            let combined = w_scale * *x_scales.get_unchecked(b);
            if combined != 0.0 {
                let mut tail_qx = [0_i8; Q4_BLOCK_SIZE];
                tail_qx[..rem].copy_from_slice(&qx[b * Q4_BLOCK_SIZE..in_dim]);

                let raw = vld1q_u8(blk.qs.as_ptr());
                let lo_s8 = vreinterpretq_s8_u8(vandq_u8(raw, mask_0f));
                let hi_s8 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(raw));

                let qx_lo = vld1q_s8(tail_qx.as_ptr());
                let qx_hi = vld1q_s8(tail_qx.as_ptr().add(Q4_BLOCK_SIZE / 2));

                let mut acc16 = vmull_s8(vget_low_s8(lo_s8), vget_low_s8(qx_lo));
                acc16 = vmlal_s8(acc16, vget_high_s8(lo_s8), vget_high_s8(qx_lo));
                acc16 = vmlal_s8(acc16, vget_low_s8(hi_s8), vget_low_s8(qx_hi));
                acc16 = vmlal_s8(acc16, vget_high_s8(hi_s8), vget_high_s8(qx_hi));

                let acc = vaddlvq_s16(acc16);
                total = (acc as f32)
                    .mul_add(combined, w_scale.mul_add(*x_bias.get_unchecked(b), total));
            }
        }
    }

    total
}

/// Referência escalar portável (usada em CPUs sem AVX2/NEON e para validação de paridade).
#[must_use]
pub fn dot_q4_row_scalar(
    blocks: &[Q4Block],
    qx: &[i8],
    x_scales: &[f32],
    x_bias: &[f32],
    in_dim: usize,
) -> f32 {
    let half = Q4_BLOCK_SIZE / 2;
    let mut total = 0.0_f32;

    for (b, blk) in blocks.iter().enumerate() {
        let w_scale = f16_to_f32_scalar(blk.scale_f16);
        let combined = w_scale * x_scales[b];
        if combined == 0.0 {
            continue;
        }
        let base = b * Q4_BLOCK_SIZE;
        let end = (base + Q4_BLOCK_SIZE).min(in_dim);
        let qx_blk = &qx[base..end];

        let mut acc = 0_i32;
        for (j, &packed) in blk.qs.iter().enumerate() {
            if let Some(&q0) = qx_blk.get(j) {
                acc += i32::from(packed & 0x0F) * i32::from(q0);
            }
            if let Some(&q1) = qx_blk.get(j + half) {
                acc += i32::from(packed >> 4) * i32::from(q1);
            }
        }
        total = (acc as f32).mul_add(combined, w_scale.mul_add(x_bias[b], total));
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_handles_normal_subnormal_and_saturation() {
        for &val in &[0.0_f32, 1.0, 0.5, 0.125, 0.001_5, 12.75, 65504.0] {
            let back = f16_to_f32(f32_to_f16(val));
            let tol = (val.abs() * 2e-3).max(1e-6);
            assert!((back - val).abs() <= tol, "val={val}, got={back}");
        }
        let sub = 2.0_f32.powi(-20);
        assert!((f16_to_f32(f32_to_f16(sub)) - sub).abs() < 1e-7);
        assert_eq!(f16_to_f32(f32_to_f16(100_000.0)), 65504.0);
    }

    #[test]
    fn simd_q4_matches_scalar_across_all_odd_even_and_tail_dimensions() {
        for in_dim in [1_usize, 7, 15, 16, 31, 32, 33, 64, 96, 100, 768] {
            let n_blk = in_dim.div_ceil(Q4_BLOCK_SIZE);
            let rows = 6;
            let mut blocks = Vec::with_capacity(rows * n_blk);
            for r in 0..rows {
                for b in 0..n_blk {
                    let mut qs = [0_u8; 16];
                    for (j, slot) in qs.iter_mut().enumerate() {
                        let lo = ((r * 7 + b * 3 + j) % 16) as u8;
                        let hi = ((r * 5 + b * 11 + j * 3) % 16) as u8;
                        *slot = lo | (hi << 4);
                    }
                    blocks.push(Q4Block {
                        scale_f16: f32_to_f16(0.02 + ((r + b) % 7) as f32 * 0.01),
                        qs,
                    });
                }
            }

            let qx: Vec<i8> = (0..in_dim)
                .map(|i| (((i * 37 + 13) % 255) as i32 - 127) as i8)
                .collect();
            let x_scales: Vec<f32> = (0..n_blk).map(|b| 0.01 + (b % 5) as f32 * 0.005).collect();
            let x_bias: Vec<f32> = (0..n_blk)
                .map(|b| {
                    let s = b * Q4_BLOCK_SIZE;
                    let e = (s + Q4_BLOCK_SIZE).min(in_dim);
                    let sum: i32 = qx[s..e].iter().map(|&v| i32::from(v)).sum();
                    -8.0 * x_scales[b] * (sum as f32)
                })
                .collect();

            let mut out = vec![0.0_f32; rows];
            compute_q4_row_range(
                &blocks, n_blk, 0, &mut out, &qx, &x_scales, &x_bias, in_dim, true,
            );

            for r in 0..rows {
                let row_blocks = &blocks[r * n_blk..(r + 1) * n_blk];
                let expected = dot_q4_row_scalar(row_blocks, &qx, &x_scales, &x_bias, in_dim);
                let single = dot_q4_row(row_blocks, &qx, &x_scales, &x_bias, in_dim);
                assert!(
                    (out[r] - expected).abs() < 1e-3,
                    "range in_dim={in_dim} r={r}: got={} vs expected={expected}",
                    out[r]
                );
                assert!(
                    (single - expected).abs() < 1e-3,
                    "single in_dim={in_dim} r={r}: got={single} vs expected={expected}"
                );
            }
        }
    }

    #[test]
    fn simd_q4_x4_matches_single_row_and_scalar() {
        for in_dim in [1_usize, 15, 32, 33, 96, 768] {
            let n_blk = in_dim.div_ceil(Q4_BLOCK_SIZE);
            let rows = 5;
            let mut blocks = Vec::with_capacity(rows * n_blk);
            for r in 0..rows {
                for b in 0..n_blk {
                    let mut qs = [0_u8; 16];
                    for (j, slot) in qs.iter_mut().enumerate() {
                        let lo = ((r * 7 + b * 3 + j) % 16) as u8;
                        let hi = ((r * 5 + b * 11 + j * 3) % 16) as u8;
                        *slot = lo | (hi << 4);
                    }
                    blocks.push(Q4Block {
                        scale_f16: f32_to_f16(0.02 + ((r + b) % 7) as f32 * 0.01),
                        qs,
                    });
                }
            }

            let qx_vecs: Vec<Vec<i8>> = (0..4)
                .map(|k| {
                    (0..in_dim)
                        .map(|i| (((i * (31 + k * 7) + k * 19 + 5) % 255) as i32 - 127) as i8)
                        .collect()
                })
                .collect();
            let xs_vecs: Vec<Vec<f32>> = (0..4)
                .map(|k| {
                    (0..n_blk)
                        .map(|b| 0.01 + ((b + k) % 5) as f32 * 0.004)
                        .collect()
                })
                .collect();
            let xb_vecs: Vec<Vec<f32>> = (0..4)
                .map(|k| {
                    (0..n_blk)
                        .map(|b| {
                            let s = b * Q4_BLOCK_SIZE;
                            let e = (s + Q4_BLOCK_SIZE).min(in_dim);
                            let sum: i32 = qx_vecs[k][s..e].iter().map(|&v| i32::from(v)).sum();
                            -8.0 * xs_vecs[k][b] * (sum as f32)
                        })
                        .collect()
                })
                .collect();

            let qx = [&qx_vecs[0][..], &qx_vecs[1][..], &qx_vecs[2][..], &qx_vecs[3][..]];
            let xs = [&xs_vecs[0][..], &xs_vecs[1][..], &xs_vecs[2][..], &xs_vecs[3][..]];
            let xb = [&xb_vecs[0][..], &xb_vecs[1][..], &xb_vecs[2][..], &xb_vecs[3][..]];

            let mut o0 = vec![0.0_f32; rows];
            let mut o1 = vec![0.0_f32; rows];
            let mut o2 = vec![0.0_f32; rows];
            let mut o3 = vec![0.0_f32; rows];
            compute_q4_row_range_x4(
                &blocks,
                n_blk,
                0,
                [&mut o0, &mut o1, &mut o2, &mut o3],
                qx,
                xs,
                xb,
                in_dim,
                true,
            );

            for r in 0..rows {
                let row_blocks = &blocks[r * n_blk..(r + 1) * n_blk];
                let expected = dot_q4_row_x4_scalar(row_blocks, qx, xs, xb, in_dim);
                let got = [o0[r], o1[r], o2[r], o3[r]];
                for k in 0..4 {
                    assert!(
                        (got[k] - expected[k]).abs() < 1e-3,
                        "in_dim={in_dim} r={r} k={k}: got={} vs expected={}",
                        got[k],
                        expected[k]
                    );
                }
            }
        }
    }
}
