use llguidance::toktrie::{SimpleVob, TokenId};
pub use ns_kernels::{Q4_BLOCK_SIZE, active_q4_backend};
use ns_kernels::{
    Q4Block, compute_q4_row_range, compute_q4_row_range_x4, dot_q4_row, f16_to_f32, f32_to_f16,
};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};

use crate::error::LlmError;

/// Limite seguro para garantir ausência matemática de overflow em acumuladores `i32`:
/// `127 * 127 * 131_072 = 2_114_060_288 <= i32::MAX (2_147_483_647)`.
pub const MAX_QUANT_IN_DIM: usize = 131_072;

/// Máximo de blocos de entrada mantidos na stack sem alocação (`512 * 32 = 16.384` dims: cobre 0.1B, 1.5B e 2.9B+).
const MAX_STACK_Q4_BLOCKS: usize = 512;

/// Volume mínimo de multiplicações (`out_dim * in_dim`) para ativar o Rayon em `matvec_into`.
const PARALLEL_MATVEC_MIN_MACS: usize = 512 * 1024;
/// Linhas por tarefa do Rayon: agrupar reduz o overhead de agendamento por linha.
const PARALLEL_ROW_CHUNK: usize = 64;
const PARALLEL_SPARSE_THRESHOLD: usize = 1_024;
/// Acima de 2 MiB, ativa `PREFETCHNTA` no kernel SIMD para proteger o `Rwkv7State` na L3.
const NTA_PREFETCH_MIN_BYTES: usize = 2 * 1024 * 1024;

const Q4_MIXED_MIN_ELEMENTS: usize = 512 * 1024;
const Q4_LARGE_MIN_ELEMENTS: usize = 1024 * 1024;

/// Formato de quantização dos pesos lineares em RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantFormat {
    /// 8-bit simétrico por linha (`i8` + 1 `f32` por linha).
    #[default]
    W8,
    /// 4-bit simétrico por bloco de 32 pesos (18 bytes por bloco: `f16` scale + 16 bytes empacotados).
    Q4,
}

impl QuantFormat {
    /// Lê a variável de ambiente `NS_WEIGHT_QUANT` (`w8` desativa Q4; padrão usa Q4 em matrizes grandes).
    #[must_use]
    pub fn from_env() -> Self {
        match std::env::var("NS_WEIGHT_QUANT") {
            Ok(val) if val.eq_ignore_ascii_case("w8") => Self::W8,
            _ => Self::Q4,
        }
    }

    /// Precisão mista por tamanho de matriz:
    /// - padrão / `q4_large`: Q4 em matrizes `>= 1M` pesos (`cm.key`, `head`, `tm.proj` 1.5B+);
    ///   Time-Mixing 0.1B e LoRAs pequenos permanecem em W8.
    /// - `q4`: Q4 em matrizes `>= 512k` pesos.
    /// - `q4_all`: força Q4 em todas as matrizes.
    /// - `w8`: força W8 em todas as matrizes.
    #[must_use]
    pub fn from_env_for_shape(out_dim: usize, in_dim: usize) -> Self {
        let elements = out_dim.saturating_mul(in_dim);
        match std::env::var("NS_WEIGHT_QUANT") {
            Ok(val) if val.eq_ignore_ascii_case("w8") => Self::W8,
            Ok(val) if val.eq_ignore_ascii_case("q4_all") => Self::Q4,
            Ok(val) if val.eq_ignore_ascii_case("q4") => {
                if elements >= Q4_MIXED_MIN_ELEMENTS {
                    Self::Q4
                } else {
                    Self::W8
                }
            }
            _ => {
                if elements >= Q4_LARGE_MIN_ELEMENTS {
                    Self::Q4
                } else {
                    Self::W8
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
enum QuantizedStorage {
    W8 {
        data: Box<[i8]>,
        scales: Box<[f32]>,
    },
    Q4 {
        blocks: Box<[Q4Block]>,
        blocks_per_row: usize,
    },
}

/// Matriz linear quantizada (`W8A8` ou `Q4A8` por bloco de 32 com kernel SIMD em `ns-kernels`).
#[derive(Debug, Clone)]
pub struct QuantizedMatrix {
    out_dim: usize,
    in_dim: usize,
    storage: QuantizedStorage,
}

impl QuantizedMatrix {
    /// Carrega uma matriz do `.safetensors` usando o formato configurado em `NS_WEIGHT_QUANT`
    /// (`W8` por padrão; `Q4` quando `NS_WEIGHT_QUANT=q4`).
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::ModelLoad`] se o tensor estiver ausente, com shape/dtype inválido
    /// ou exceder [`MAX_QUANT_IN_DIM`].
    pub fn load_from_safetensors(
        st: &SafeTensors<'_>,
        name: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self, LlmError> {
        let format = QuantFormat::from_env_for_shape(out_dim, in_dim);
        Self::load_from_safetensors_with_format(st, name, out_dim, in_dim, format)
    }

    /// Carrega uma matriz do `.safetensors` no formato explícito (`W8` ou `Q4`).
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::ModelLoad`] se o tensor estiver ausente, com shape/dtype inválido
    /// ou exceder [`MAX_QUANT_IN_DIM`].
    pub fn load_from_safetensors_with_format(
        st: &SafeTensors<'_>,
        name: &str,
        out_dim: usize,
        in_dim: usize,
        format: QuantFormat,
    ) -> Result<Self, LlmError> {
        if in_dim == 0 || out_dim == 0 || in_dim > MAX_QUANT_IN_DIM {
            return Err(LlmError::ModelLoad(format!(
                "invalid matrix dimensions [{out_dim}, {in_dim}] for '{name}'"
            )));
        }

        let q_name = format!("{name}.q");
        let s_name = format!("{name}.scale");

        if let (Ok(q_view), Ok(s_view)) = (st.tensor(&q_name), st.tensor(&s_name)) {
            if q_view.dtype() != Dtype::I8 || q_view.shape() != [out_dim, in_dim] {
                return Err(LlmError::ModelLoad(format!(
                    "tensor '{q_name}' must be I8 with shape [{out_dim}, {in_dim}]"
                )));
            }
            if s_view.dtype() != Dtype::F32 || s_view.shape() != [out_dim] {
                return Err(LlmError::ModelLoad(format!(
                    "tensor '{s_name}' must be F32 with shape [{out_dim}]"
                )));
            }

            let mut scales = Vec::with_capacity(out_dim);
            for chunk in s_view.data().chunks_exact(4) {
                scales.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }

            match format {
                QuantFormat::W8 => {
                    let data: Box<[i8]> = q_view
                        .data()
                        .iter()
                        .map(|&b| i8::from_ne_bytes([b]))
                        .collect::<Vec<i8>>()
                        .into_boxed_slice();
                    return Ok(Self {
                        out_dim,
                        in_dim,
                        storage: QuantizedStorage::W8 {
                            data,
                            scales: scales.into_boxed_slice(),
                        },
                    });
                }
                QuantFormat::Q4 => {
                    let blocks_per_row = in_dim.div_ceil(Q4_BLOCK_SIZE);
                    let total_blocks = out_dim.checked_mul(blocks_per_row).ok_or_else(|| {
                        LlmError::ModelLoad(format!("shape overflow for '{name}'"))
                    })?;
                    let mut blocks = Vec::with_capacity(total_blocks);
                    let q_bytes = q_view.data();
                    let mut row_f32 = vec![0.0_f32; in_dim];

                    for (row, &row_scale) in scales.iter().enumerate() {
                        let base = row * in_dim;
                        for (col, dst) in row_f32.iter_mut().enumerate() {
                            let q = i8::from_ne_bytes([q_bytes[base + col]]);
                            *dst = f32::from(q) * row_scale;
                        }
                        pack_row_q4(&row_f32, &mut blocks);
                    }

                    return Ok(Self {
                        out_dim,
                        in_dim,
                        storage: QuantizedStorage::Q4 {
                            blocks: blocks.into_boxed_slice(),
                            blocks_per_row,
                        },
                    });
                }
            }
        }

        let view = st
            .tensor(name)
            .map_err(|err| LlmError::ModelLoad(format!("missing tensor '{name}': {err}")))?;
        if view.dtype() != Dtype::F32 || view.shape() != [out_dim, in_dim] {
            return Err(LlmError::ModelLoad(format!(
                "tensor '{name}' must be F32 with shape [{out_dim}, {in_dim}], got {:?} {:?}",
                view.dtype(),
                view.shape()
            )));
        }

        let total = out_dim
            .checked_mul(in_dim)
            .ok_or_else(|| LlmError::ModelLoad(format!("shape overflow for '{name}'")))?;

        let mut weights_f32 = Vec::with_capacity(total);
        for chunk in view.data().chunks_exact(4) {
            weights_f32.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        Self::from_f32_with_format(&weights_f32, out_dim, in_dim, format)
    }

    /// Quantiza uma matriz `f32` row-major em `W8` (`i8` + escala por linha).
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::ModelLoad`] se as dimensões não baterem com o slice ou excederem o limite.
    pub fn from_f32(weights: &[f32], out_dim: usize, in_dim: usize) -> Result<Self, LlmError> {
        Self::from_f32_with_format(weights, out_dim, in_dim, QuantFormat::W8)
    }

    /// Quantiza uma matriz `f32` row-major em `Q4` (4-bit por bloco de 32 com escala `f16`).
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::ModelLoad`] se as dimensões não baterem com o slice ou excederem o limite.
    pub fn from_f32_q4(weights: &[f32], out_dim: usize, in_dim: usize) -> Result<Self, LlmError> {
        Self::from_f32_with_format(weights, out_dim, in_dim, QuantFormat::Q4)
    }

    /// Quantiza uma matriz `f32` row-major no formato solicitado (`W8` ou `Q4`).
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::ModelLoad`] se as dimensões não baterem com o slice ou excederem o limite.
    pub fn from_f32_with_format(
        weights: &[f32],
        out_dim: usize,
        in_dim: usize,
        format: QuantFormat,
    ) -> Result<Self, LlmError> {
        let expected = out_dim
            .checked_mul(in_dim)
            .ok_or_else(|| LlmError::ModelLoad("matrix element count overflow".to_owned()))?;
        if out_dim == 0 || in_dim == 0 || in_dim > MAX_QUANT_IN_DIM || weights.len() != expected {
            return Err(LlmError::ModelLoad(format!(
                "invalid f32 buffer length {} for [{out_dim}, {in_dim}]",
                weights.len()
            )));
        }

        let storage = match format {
            QuantFormat::W8 => {
                let mut data = vec![0_i8; expected];
                let mut scales = vec![0.0_f32; out_dim];

                for (row, scale_slot) in scales.iter_mut().enumerate() {
                    let start = row * in_dim;
                    let src = &weights[start..start + in_dim];
                    let dst = &mut data[start..start + in_dim];
                    *scale_slot = quantize_slice_symmetric(src, dst);
                }

                QuantizedStorage::W8 {
                    data: data.into_boxed_slice(),
                    scales: scales.into_boxed_slice(),
                }
            }
            QuantFormat::Q4 => {
                let blocks_per_row = in_dim.div_ceil(Q4_BLOCK_SIZE);
                let total_blocks = out_dim
                    .checked_mul(blocks_per_row)
                    .ok_or_else(|| LlmError::ModelLoad("q4 block count overflow".to_owned()))?;
                let mut blocks = Vec::with_capacity(total_blocks);

                for row in 0..out_dim {
                    let start = row * in_dim;
                    pack_row_q4(&weights[start..start + in_dim], &mut blocks);
                }

                QuantizedStorage::Q4 {
                    blocks: blocks.into_boxed_slice(),
                    blocks_per_row,
                }
            }
        };

        Ok(Self {
            out_dim,
            in_dim,
            storage,
        })
    }

    #[must_use]
    pub const fn format(&self) -> QuantFormat {
        match self.storage {
            QuantizedStorage::W8 { .. } => QuantFormat::W8,
            QuantizedStorage::Q4 { .. } => QuantFormat::Q4,
        }
    }

    #[must_use]
    pub const fn weight_bytes(&self) -> usize {
        match &self.storage {
            QuantizedStorage::W8 { data, scales } => {
                data.len() + scales.len() * std::mem::size_of::<f32>()
            }
            QuantizedStorage::Q4 { blocks, .. } => blocks.len() * std::mem::size_of::<Q4Block>(),
        }
    }

    /// Calcula `out = W * x` sem alocar no heap (para `in_dim <= 8192`), reutilizando `q_input_scratch`.
    pub fn matvec_into(&self, x: &[f32], q_input_scratch: &mut [i8], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_dim);
        debug_assert_eq!(out.len(), self.out_dim);
        debug_assert!(q_input_scratch.len() >= self.in_dim);

        match &self.storage {
            QuantizedStorage::W8 { data, scales } => {
                let qx = &mut q_input_scratch[..self.in_dim];
                let scale_x = quantize_slice_symmetric(x, qx);
                if scale_x == 0.0 {
                    out.fill(0.0);
                    return;
                }

                let qx_ref: &[i8] = qx;
                let in_dim = self.in_dim;

                if self.out_dim.saturating_mul(in_dim) >= PARALLEL_MATVEC_MIN_MACS {
                    out.par_chunks_mut(PARALLEL_ROW_CHUNK).enumerate().for_each(
                        |(chunk_idx, chunk)| {
                            let row0 = chunk_idx * PARALLEL_ROW_CHUNK;
                            for (i, dst) in chunk.iter_mut().enumerate() {
                                let row = row0 + i;
                                let start = row * in_dim;
                                let acc = dot_i8_unrolled(&data[start..start + in_dim], qx_ref);
                                *dst = (acc as f32) * (scales[row] * scale_x);
                            }
                        },
                    );
                } else {
                    for (row, dst) in out.iter_mut().enumerate() {
                        let start = row * in_dim;
                        let acc = dot_i8_unrolled(&data[start..start + in_dim], qx_ref);
                        *dst = (acc as f32) * (scales[row] * scale_x);
                    }
                }
            }
            QuantizedStorage::Q4 {
                blocks,
                blocks_per_row,
            } => {
                let qx = &mut q_input_scratch[..self.in_dim];
                let n_blk = *blocks_per_row;
                let in_dim = self.in_dim;
                let use_nta = self.weight_bytes() >= NTA_PREFETCH_MIN_BYTES;

                with_quantized_q8_blocks(x, qx, n_blk, |qx_ref, x_scales, x_bias, all_zero| {
                    if all_zero {
                        out.fill(0.0);
                        return;
                    }

                    if self.out_dim.saturating_mul(in_dim) >= PARALLEL_MATVEC_MIN_MACS {
                        out.par_chunks_mut(PARALLEL_ROW_CHUNK).enumerate().for_each(
                            |(chunk_idx, chunk)| {
                                let row0 = chunk_idx * PARALLEL_ROW_CHUNK;
                                compute_q4_row_range(
                                    blocks, n_blk, row0, chunk, qx_ref, x_scales, x_bias, in_dim,
                                    use_nta,
                                );
                            },
                        );
                    } else {
                        compute_q4_row_range(
                            blocks, n_blk, 0, out, qx_ref, x_scales, x_bias, in_dim, use_nta,
                        );
                    }
                });
            }
        }
    }

    /// Calcula `outs[k] = W * xs[k]` para **4 trajetórias simultâneas (`M × 4`)**
    /// lendo os pesos da matriz **uma única vez da RAM** (base do Best-of-N sem custo extra de banda).
    pub fn matmul4_into(&self, xs: [&[f32]; 4], q_inputs: [&mut [i8]; 4], outs: [&mut [f32]; 4]) {
        let in_dim = self.in_dim;
        debug_assert!(xs.iter().all(|x| x.len() == in_dim));
        debug_assert!(outs.iter().all(|o| o.len() == self.out_dim));
        debug_assert!(q_inputs.iter().all(|q| q.len() >= in_dim));

        let [q0, q1, q2, q3] = q_inputs;
        let q0 = &mut q0[..in_dim];
        let q1 = &mut q1[..in_dim];
        let q2 = &mut q2[..in_dim];
        let q3 = &mut q3[..in_dim];

        match &self.storage {
            QuantizedStorage::W8 { data, scales } => {
                let s0 = quantize_slice_symmetric(xs[0], q0);
                let s1 = quantize_slice_symmetric(xs[1], q1);
                let s2 = quantize_slice_symmetric(xs[2], q2);
                let s3 = quantize_slice_symmetric(xs[3], q3);

                if s0 == 0.0 && s1 == 0.0 && s2 == 0.0 && s3 == 0.0 {
                    for out in outs {
                        out.fill(0.0);
                    }
                    return;
                }

                let qx0: &[i8] = q0;
                let qx1: &[i8] = q1;
                let qx2: &[i8] = q2;
                let qx3: &[i8] = q3;
                let [out0, out1, out2, out3] = outs;

                if self.out_dim.saturating_mul(in_dim).saturating_mul(4) >= PARALLEL_MATVEC_MIN_MACS
                {
                    (
                        out0.par_chunks_mut(PARALLEL_ROW_CHUNK),
                        out1.par_chunks_mut(PARALLEL_ROW_CHUNK),
                        out2.par_chunks_mut(PARALLEL_ROW_CHUNK),
                        out3.par_chunks_mut(PARALLEL_ROW_CHUNK),
                    )
                        .into_par_iter()
                        .enumerate()
                        .for_each(|(chunk_idx, (c0, c1, c2, c3))| {
                            let row0 = chunk_idx * PARALLEL_ROW_CHUNK;
                            for i in 0..c0.len() {
                                let row = row0 + i;
                                let w_row = &data[row * in_dim..(row + 1) * in_dim];
                                let sw = scales[row];
                                // Cada linha fica quente em L1 durante os 4 dots (1 única leitura de RAM).
                                c0[i] = (dot_i8_unrolled(w_row, qx0) as f32) * (sw * s0);
                                c1[i] = (dot_i8_unrolled(w_row, qx1) as f32) * (sw * s1);
                                c2[i] = (dot_i8_unrolled(w_row, qx2) as f32) * (sw * s2);
                                c3[i] = (dot_i8_unrolled(w_row, qx3) as f32) * (sw * s3);
                            }
                        });
                } else {
                    for row in 0..self.out_dim {
                        let w_row = &data[row * in_dim..(row + 1) * in_dim];
                        let sw = scales[row];
                        out0[row] = (dot_i8_unrolled(w_row, qx0) as f32) * (sw * s0);
                        out1[row] = (dot_i8_unrolled(w_row, qx1) as f32) * (sw * s1);
                        out2[row] = (dot_i8_unrolled(w_row, qx2) as f32) * (sw * s2);
                        out3[row] = (dot_i8_unrolled(w_row, qx3) as f32) * (sw * s3);
                    }
                }
            }
            QuantizedStorage::Q4 {
                blocks,
                blocks_per_row,
            } => {
                let n_blk = *blocks_per_row;
                let use_nta = self.weight_bytes() >= NTA_PREFETCH_MIN_BYTES;

                with_quantized_q8_blocks_x4(
                    xs,
                    [q0, q1, q2, q3],
                    n_blk,
                    |qx_refs, x_scales, x_bias, all_zero| {
                        if all_zero.iter().all(|&z| z) {
                            for out in outs {
                                out.fill(0.0);
                            }
                            return;
                        }

                        let [out0, out1, out2, out3] = outs;
                        if self.out_dim.saturating_mul(in_dim).saturating_mul(4)
                            >= PARALLEL_MATVEC_MIN_MACS
                        {
                            (
                                out0.par_chunks_mut(PARALLEL_ROW_CHUNK),
                                out1.par_chunks_mut(PARALLEL_ROW_CHUNK),
                                out2.par_chunks_mut(PARALLEL_ROW_CHUNK),
                                out3.par_chunks_mut(PARALLEL_ROW_CHUNK),
                            )
                                .into_par_iter()
                                .enumerate()
                                .for_each(
                                    |(chunk_idx, (c0, c1, c2, c3))| {
                                        let row0 = chunk_idx * PARALLEL_ROW_CHUNK;
                                        compute_q4_row_range_x4(
                                            blocks,
                                            n_blk,
                                            row0,
                                            [c0, c1, c2, c3],
                                            qx_refs,
                                            x_scales,
                                            x_bias,
                                            in_dim,
                                            use_nta,
                                        );
                                    },
                                );
                        } else {
                            compute_q4_row_range_x4(
                                blocks,
                                n_blk,
                                0,
                                [out0, out1, out2, out3],
                                qx_refs,
                                x_scales,
                                x_bias,
                                in_dim,
                                use_nta,
                            );
                        }
                    },
                );
            }
        }
    }

    /// Avalia o produto escalar apenas sobre os índices de linha em `rows`
    /// e retorna o `TokenId` de maior score com desempate determinístico.
    pub fn argmax_rows(
        &self,
        x: &[f32],
        rows: &[usize],
        q_input_scratch: &mut [i8],
    ) -> Option<TokenId> {
        debug_assert_eq!(x.len(), self.in_dim);
        debug_assert!(q_input_scratch.len() >= self.in_dim);

        if rows.is_empty() {
            return None;
        }

        let best = match &self.storage {
            QuantizedStorage::W8 { data, scales } => {
                let qx = &mut q_input_scratch[..self.in_dim];
                let scale_x = quantize_slice_symmetric(x, qx);
                if scale_x == 0.0 {
                    return TokenId::try_from(rows[0]).ok();
                }

                let qx_ref: &[i8] = qx;
                let in_dim = self.in_dim;

                if rows.len() >= PARALLEL_SPARSE_THRESHOLD {
                    rows.par_iter()
                        .filter(|&&row| row < self.out_dim)
                        .map(|&row| {
                            let start = row * in_dim;
                            let acc = dot_i8_unrolled(&data[start..start + in_dim], qx_ref);
                            let score = (acc as f32) * (scales[row] * scale_x);
                            (row, score)
                        })
                        .reduce_with(pick_better)
                } else {
                    let mut best_pair: Option<(usize, f32)> = None;
                    for &row in rows {
                        if row >= self.out_dim {
                            continue;
                        }
                        let start = row * in_dim;
                        let acc = dot_i8_unrolled(&data[start..start + in_dim], qx_ref);
                        let score = (acc as f32) * (scales[row] * scale_x);
                        best_pair = Some(match best_pair {
                            Some(curr) => pick_better(curr, (row, score)),
                            None => (row, score),
                        });
                    }
                    best_pair
                }
            }
            QuantizedStorage::Q4 {
                blocks,
                blocks_per_row,
            } => {
                let qx = &mut q_input_scratch[..self.in_dim];
                let n_blk = *blocks_per_row;
                let in_dim = self.in_dim;
                let out_dim = self.out_dim;

                with_quantized_q8_blocks(x, qx, n_blk, |qx_ref, x_scales, x_bias, all_zero| {
                    if all_zero {
                        return Some((rows[0], 0.0_f32));
                    }
                    if rows.len() >= PARALLEL_SPARSE_THRESHOLD {
                        rows.par_iter()
                            .filter(|&&row| row < out_dim)
                            .map(|&row| {
                                let row_blocks = &blocks[row * n_blk..(row + 1) * n_blk];
                                let score =
                                    dot_q4_row(row_blocks, qx_ref, x_scales, x_bias, in_dim);
                                (row, score)
                            })
                            .reduce_with(pick_better)
                    } else {
                        let mut best_pair: Option<(usize, f32)> = None;
                        for &row in rows {
                            if row >= out_dim {
                                continue;
                            }
                            let row_blocks = &blocks[row * n_blk..(row + 1) * n_blk];
                            let score = dot_q4_row(row_blocks, qx_ref, x_scales, x_bias, in_dim);
                            best_pair = Some(match best_pair {
                                Some(curr) => pick_better(curr, (row, score)),
                                None => (row, score),
                            });
                        }
                        best_pair
                    }
                })
            }
        };

        best.and_then(|(row, _)| TokenId::try_from(row).ok())
    }

    /// Calcula os logits **apenas** para as linhas em `rows`, gravando o resultado
    /// em `out_scores` (zero realocação após o warm-up).
    pub fn scores_for_rows(
        &self,
        x: &[f32],
        rows: &[usize],
        q_input_scratch: &mut [i8],
        out_scores: &mut Vec<f32>,
    ) {
        debug_assert_eq!(x.len(), self.in_dim);
        debug_assert!(q_input_scratch.len() >= self.in_dim);

        out_scores.clear();
        if rows.is_empty() {
            return;
        }

        match &self.storage {
            QuantizedStorage::W8 { data, scales } => {
                let qx = &mut q_input_scratch[..self.in_dim];
                let scale_x = quantize_slice_symmetric(x, qx);
                if scale_x == 0.0 {
                    out_scores.resize(rows.len(), 0.0);
                    return;
                }

                let qx_ref: &[i8] = qx;
                let in_dim = self.in_dim;

                if rows.len() >= PARALLEL_SPARSE_THRESHOLD {
                    out_scores.resize(rows.len(), 0.0);
                    out_scores
                        .par_iter_mut()
                        .zip(rows.par_iter())
                        .for_each(|(dst, &row)| {
                            if row < self.out_dim {
                                let start = row * in_dim;
                                let acc = dot_i8_unrolled(&data[start..start + in_dim], qx_ref);
                                *dst = (acc as f32) * (scales[row] * scale_x);
                            } else {
                                *dst = f32::NEG_INFINITY;
                            }
                        });
                } else {
                    out_scores.reserve(rows.len());
                    for &row in rows {
                        if row < self.out_dim {
                            let start = row * in_dim;
                            let acc = dot_i8_unrolled(&data[start..start + in_dim], qx_ref);
                            out_scores.push((acc as f32) * (scales[row] * scale_x));
                        } else {
                            out_scores.push(f32::NEG_INFINITY);
                        }
                    }
                }
            }
            QuantizedStorage::Q4 {
                blocks,
                blocks_per_row,
            } => {
                let qx = &mut q_input_scratch[..self.in_dim];
                let n_blk = *blocks_per_row;
                let in_dim = self.in_dim;
                let out_dim = self.out_dim;

                with_quantized_q8_blocks(x, qx, n_blk, |qx_ref, x_scales, x_bias, all_zero| {
                    if all_zero {
                        out_scores.resize(rows.len(), 0.0);
                        return;
                    }
                    if rows.len() >= PARALLEL_SPARSE_THRESHOLD {
                        out_scores.resize(rows.len(), 0.0);
                        out_scores
                            .par_iter_mut()
                            .zip(rows.par_iter())
                            .for_each(|(dst, &row)| {
                                if row < out_dim {
                                    let row_blocks = &blocks[row * n_blk..(row + 1) * n_blk];
                                    *dst = dot_q4_row(row_blocks, qx_ref, x_scales, x_bias, in_dim);
                                } else {
                                    *dst = f32::NEG_INFINITY;
                                }
                            });
                    } else {
                        out_scores.reserve(rows.len());
                        for &row in rows {
                            if row < out_dim {
                                let row_blocks = &blocks[row * n_blk..(row + 1) * n_blk];
                                out_scores
                                    .push(dot_q4_row(row_blocks, qx_ref, x_scales, x_bias, in_dim));
                            } else {
                                out_scores.push(f32::NEG_INFINITY);
                            }
                        }
                    }
                });
            }
        }
    }
}

fn pack_row_q4(row: &[f32], out_blocks: &mut Vec<Q4Block>) {
    let half = Q4_BLOCK_SIZE / 2;
    for chunk in row.chunks(Q4_BLOCK_SIZE) {
        let mut max_abs = 0.0_f32;
        for &v in chunk {
            let abs = v.abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }

        if max_abs == 0.0 || !max_abs.is_finite() {
            out_blocks.push(Q4Block {
                scale_f16: 0,
                qs: [0x88; Q4_BLOCK_SIZE / 2],
            });
            continue;
        }

        let scale = max_abs / 7.0;
        let scale_f16 = f32_to_f16(scale);
        let eff_scale = f16_to_f32(scale_f16);
        let inv_scale = if eff_scale > 0.0 {
            1.0 / eff_scale
        } else {
            0.0
        };

        let mut qs = [0x88_u8; Q4_BLOCK_SIZE / 2];
        for (j, byte_slot) in qs.iter_mut().enumerate() {
            let lo = chunk
                .get(j)
                .map_or(8_u8, |&v| quantize_nibble(v, inv_scale));
            let hi = chunk
                .get(j + half)
                .map_or(8_u8, |&v| quantize_nibble(v, inv_scale));
            *byte_slot = lo | (hi << 4);
        }

        out_blocks.push(Q4Block { scale_f16, qs });
    }
}

#[inline]
fn quantize_nibble(val: f32, inv_scale: f32) -> u8 {
    let scaled = (val * inv_scale).round();
    if !scaled.is_finite() {
        return 8;
    }
    let clamped = scaled.clamp(-8.0, 7.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let u = (clamped + 8.0) as u8;
    u
}

fn with_quantized_q8_blocks<R, F>(x: &[f32], qx: &mut [i8], n_blocks: usize, f: F) -> R
where
    F: FnOnce(&[i8], &[f32], &[f32], bool) -> R,
{
    if n_blocks <= MAX_STACK_Q4_BLOCKS {
        let mut x_scales = [0.0_f32; MAX_STACK_Q4_BLOCKS];
        let mut x_bias = [0.0_f32; MAX_STACK_Q4_BLOCKS];
        let all_zero =
            quantize_q8_blocks_into(x, qx, &mut x_scales[..n_blocks], &mut x_bias[..n_blocks]);
        f(qx, &x_scales[..n_blocks], &x_bias[..n_blocks], all_zero)
    } else {
        let mut x_scales = vec![0.0_f32; n_blocks];
        let mut x_bias = vec![0.0_f32; n_blocks];
        let all_zero = quantize_q8_blocks_into(x, qx, &mut x_scales, &mut x_bias);
        f(qx, &x_scales, &x_bias, all_zero)
    }
}

fn with_quantized_q8_blocks_x4<R, F>(
    xs: [&[f32]; 4],
    qxs: [&mut [i8]; 4],
    n_blocks: usize,
    f: F,
) -> R
where
    F: FnOnce([&[i8]; 4], [&[f32]; 4], [&[f32]; 4], [bool; 4]) -> R,
{
    let [q0, q1, q2, q3] = qxs;
    if n_blocks <= MAX_STACK_Q4_BLOCKS {
        let mut s0 = [0.0_f32; MAX_STACK_Q4_BLOCKS];
        let mut s1 = [0.0_f32; MAX_STACK_Q4_BLOCKS];
        let mut s2 = [0.0_f32; MAX_STACK_Q4_BLOCKS];
        let mut s3 = [0.0_f32; MAX_STACK_Q4_BLOCKS];
        let mut b0 = [0.0_f32; MAX_STACK_Q4_BLOCKS];
        let mut b1 = [0.0_f32; MAX_STACK_Q4_BLOCKS];
        let mut b2 = [0.0_f32; MAX_STACK_Q4_BLOCKS];
        let mut b3 = [0.0_f32; MAX_STACK_Q4_BLOCKS];

        let z0 = quantize_q8_blocks_into(xs[0], q0, &mut s0[..n_blocks], &mut b0[..n_blocks]);
        let z1 = quantize_q8_blocks_into(xs[1], q1, &mut s1[..n_blocks], &mut b1[..n_blocks]);
        let z2 = quantize_q8_blocks_into(xs[2], q2, &mut s2[..n_blocks], &mut b2[..n_blocks]);
        let z3 = quantize_q8_blocks_into(xs[3], q3, &mut s3[..n_blocks], &mut b3[..n_blocks]);

        f(
            [q0, q1, q2, q3],
            [
                &s0[..n_blocks],
                &s1[..n_blocks],
                &s2[..n_blocks],
                &s3[..n_blocks],
            ],
            [
                &b0[..n_blocks],
                &b1[..n_blocks],
                &b2[..n_blocks],
                &b3[..n_blocks],
            ],
            [z0, z1, z2, z3],
        )
    } else {
        let mut s0 = vec![0.0_f32; n_blocks];
        let mut s1 = vec![0.0_f32; n_blocks];
        let mut s2 = vec![0.0_f32; n_blocks];
        let mut s3 = vec![0.0_f32; n_blocks];
        let mut b0 = vec![0.0_f32; n_blocks];
        let mut b1 = vec![0.0_f32; n_blocks];
        let mut b2 = vec![0.0_f32; n_blocks];
        let mut b3 = vec![0.0_f32; n_blocks];

        let z0 = quantize_q8_blocks_into(xs[0], q0, &mut s0, &mut b0);
        let z1 = quantize_q8_blocks_into(xs[1], q1, &mut s1, &mut b1);
        let z2 = quantize_q8_blocks_into(xs[2], q2, &mut s2, &mut b2);
        let z3 = quantize_q8_blocks_into(xs[3], q3, &mut s3, &mut b3);

        f(
            [q0, q1, q2, q3],
            [&s0, &s1, &s2, &s3],
            [&b0, &b1, &b2, &b3],
            [z0, z1, z2, z3],
        )
    }
}

fn quantize_q8_blocks_into(
    x: &[f32],
    qx: &mut [i8],
    x_scales: &mut [f32],
    x_bias: &mut [f32],
) -> bool {
    let mut all_zero = true;
    for (b, (src, dst)) in x
        .chunks(Q4_BLOCK_SIZE)
        .zip(qx.chunks_mut(Q4_BLOCK_SIZE))
        .enumerate()
    {
        let mut max_abs = 0.0_f32;
        for &v in src {
            let abs = v.abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }
        if max_abs == 0.0 || !max_abs.is_finite() {
            dst.fill(0);
            x_scales[b] = 0.0;
            x_bias[b] = 0.0;
            continue;
        }
        all_zero = false;
        let scale = max_abs / 127.0;
        let inv_scale = 127.0 / max_abs;
        let mut q_sum = 0_i32;
        for (out_q, &val) in dst.iter_mut().zip(src.iter()) {
            let q = quantize_scalar(val, inv_scale);
            *out_q = q;
            q_sum += i32::from(q);
        }
        x_scales[b] = scale;
        x_bias[b] = -8.0 * scale * (q_sum as f32);
    }
    all_zero
}

#[inline]
fn pick_better(a: (usize, f32), b: (usize, f32)) -> (usize, f32) {
    if a.1 > b.1 || (a.1 == b.1 && a.0 <= b.0) {
        a
    } else {
        b
    }
}

/// Quantiza um vetor `f32` em `i8` simétrico `[-127, 127]` retornando a escala `max(|x|) / 127.0`.
#[inline]
pub fn quantize_slice_symmetric(src: &[f32], dst: &mut [i8]) -> f32 {
    let mut max_abs = 0.0_f32;
    for &v in src {
        let abs = v.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }

    if max_abs == 0.0 || !max_abs.is_finite() {
        dst.fill(0);
        return 0.0;
    }

    let scale = max_abs / 127.0;
    let inv_scale = 127.0 / max_abs;
    for (out, &val) in dst.iter_mut().zip(src.iter()) {
        *out = quantize_scalar(val, inv_scale);
    }
    scale
}

#[inline]
fn quantize_scalar(val: f32, inv_scale: f32) -> i8 {
    let scaled = (val * inv_scale).round();
    if !scaled.is_finite() {
        return 0;
    }
    let clamped = scaled.clamp(-127.0, 127.0);
    #[allow(clippy::cast_possible_truncation)]
    let q = clamped as i8;
    q
}

#[inline]
fn dot_i8_unrolled(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

#[cfg(test)]
mod tests {
    use ns_kernels::dot_q4_row_scalar;

    use super::*;

    #[test]
    fn quantized_matvec_matches_f32_within_small_tolerance()
    -> Result<(), Box<dyn std::error::Error>> {
        let out_dim = 16;
        let in_dim = 64;
        let mut w = Vec::with_capacity(out_dim * in_dim);
        for r in 0..out_dim {
            for c in 0..in_dim {
                let angle = ((r * 13 + c * 7) as f32) * 0.17;
                w.push(angle.sin() * 1.5);
            }
        }
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.23).cos()).collect();

        let qmat = QuantizedMatrix::from_f32(&w, out_dim, in_dim)?;
        let mut q_scratch = vec![0_i8; in_dim];
        let mut got = vec![0.0_f32; out_dim];
        qmat.matvec_into(&x, &mut q_scratch, &mut got);

        for row in 0..out_dim {
            let expected: f32 = w[row * in_dim..(row + 1) * in_dim]
                .iter()
                .zip(x.iter())
                .map(|(a, b)| a * b)
                .sum();
            let diff = (got[row] - expected).abs();
            assert!(
                diff < 0.15,
                "row {row}: expected {expected:.4}, got {:.4}",
                got[row]
            );
        }

        assert_eq!(qmat.weight_bytes(), out_dim * (in_dim + 4));
        Ok(())
    }

    #[test]
    fn parallel_matvec_is_bit_exact_with_sequential_reference()
    -> Result<(), Box<dyn std::error::Error>> {
        let out_dim = 768;
        let in_dim = 768;
        assert!(out_dim * in_dim >= PARALLEL_MATVEC_MIN_MACS);

        let mut w = Vec::with_capacity(out_dim * in_dim);
        for r in 0..out_dim {
            for c in 0..in_dim {
                w.push((((r * 31 + c * 17) % 251) as f32 - 125.0) * 0.01);
            }
        }
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.37).sin()).collect();

        let qmat = QuantizedMatrix::from_f32(&w, out_dim, in_dim)?;
        let mut q_scratch = vec![0_i8; in_dim];
        let mut got = vec![0.0_f32; out_dim];
        qmat.matvec_into(&x, &mut q_scratch, &mut got);

        let QuantizedStorage::W8 { data, scales } = &qmat.storage else {
            return Err("expected W8 storage".into());
        };

        let mut qx = vec![0_i8; in_dim];
        let scale_x = quantize_slice_symmetric(&x, &mut qx);
        for row in 0..out_dim {
            let start = row * in_dim;
            let acc = dot_i8_unrolled(&data[start..start + in_dim], &qx);
            let expected = (acc as f32) * (scales[row] * scale_x);
            assert_eq!(got[row].to_bits(), expected.to_bits(), "row {row}");
        }
        Ok(())
    }

    #[test]
    fn dot_i8_matches_i64_reference_for_all_tail_lengths_and_extremes() {
        for len in 0..=200_usize {
            let a: Vec<i8> = (0..len)
                .map(|i| (((i * 37 + 11) % 255) as i32 - 127) as i8)
                .collect();
            let b: Vec<i8> = (0..len)
                .map(|i| (((i * 91 + 5) % 255) as i32 - 127) as i8)
                .collect();
            let want: i64 = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| i64::from(x) * i64::from(y))
                .sum();
            assert_eq!(i64::from(dot_i8_unrolled(&a, &b)), want, "len {len}");
        }

        let worst = vec![-127_i8; MAX_QUANT_IN_DIM];
        assert_eq!(
            i64::from(dot_i8_unrolled(&worst, &worst)),
            127 * 127 * MAX_QUANT_IN_DIM as i64
        );
    }

    #[test]
    fn q4_matvec_matches_f32_and_shrinks_ram_by_nearly_half()
    -> Result<(), Box<dyn std::error::Error>> {
        let out_dim = 64;
        let in_dim = 768;
        let mut w = Vec::with_capacity(out_dim * in_dim);
        for r in 0..out_dim {
            for c in 0..in_dim {
                w.push((((r * 19 + c * 7) % 61) as f32 - 30.0) * 0.015);
            }
        }
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.17).sin()).collect();

        let w8 = QuantizedMatrix::from_f32(&w, out_dim, in_dim)?;
        let q4 = QuantizedMatrix::from_f32_q4(&w, out_dim, in_dim)?;

        assert_eq!(q4.format(), QuantFormat::Q4);
        assert_eq!(q4.weight_bytes(), out_dim * (in_dim / Q4_BLOCK_SIZE) * 18);
        assert!(q4.weight_bytes() * 100 < w8.weight_bytes() * 57);

        let mut q_scratch = vec![0_i8; in_dim];
        let mut got_q4 = vec![0.0_f32; out_dim];
        q4.matvec_into(&x, &mut q_scratch, &mut got_q4);

        let mut exact = vec![0.0_f32; out_dim];
        for row in 0..out_dim {
            exact[row] = w[row * in_dim..(row + 1) * in_dim]
                .iter()
                .zip(x.iter())
                .map(|(a, b)| a * b)
                .sum();
        }
        let dot: f32 = exact.iter().zip(&got_q4).map(|(a, b)| a * b).sum();
        let norm_e: f32 = exact.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_q: f32 = got_q4.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos_sim = dot / (norm_e * norm_q);
        assert!(cos_sim > 0.995, "expected cosine > 0.995, got {cos_sim:.6}");
        Ok(())
    }

    #[test]
    fn q4_simd_matches_scalar_for_aligned_and_unaligned_tail_dimensions()
    -> Result<(), Box<dyn std::error::Error>> {
        for in_dim in [1_usize, 7, 15, 16, 31, 32, 33, 63, 64, 100, 768] {
            let out_dim = 8;
            let mut w = Vec::with_capacity(out_dim * in_dim);
            for r in 0..out_dim {
                for c in 0..in_dim {
                    w.push((((r * 23 + c * 13) % 41) as f32 - 20.0) * 0.05);
                }
            }
            let x: Vec<f32> = (0..in_dim)
                .map(|i| (((i * 19 + 3) % 29) as f32 - 14.0) * 0.1)
                .collect();

            let q4 = QuantizedMatrix::from_f32_q4(&w, out_dim, in_dim)?;
            let QuantizedStorage::Q4 {
                blocks,
                blocks_per_row,
            } = &q4.storage
            else {
                return Err("expected Q4 storage".into());
            };

            let mut q_scratch = vec![0_i8; in_dim];
            let mut got = vec![0.0_f32; out_dim];
            q4.matvec_into(&x, &mut q_scratch, &mut got);

            let mut qx_ref = vec![0_i8; in_dim];
            let mut x_scales = vec![0.0_f32; *blocks_per_row];
            let mut x_bias = vec![0.0_f32; *blocks_per_row];
            let _ = quantize_q8_blocks_into(&x, &mut qx_ref, &mut x_scales, &mut x_bias);

            for row in 0..out_dim {
                let row_blocks = &blocks[row * *blocks_per_row..(row + 1) * *blocks_per_row];
                let expected = dot_q4_row_scalar(row_blocks, &qx_ref, &x_scales, &x_bias, in_dim);
                let diff = (got[row] - expected).abs();
                assert!(
                    diff < 1e-3,
                    "in_dim={in_dim}, row={row}: got={} vs expected={} (diff={diff})",
                    got[row],
                    expected
                );
            }
        }
        Ok(())
    }

    #[test]
    fn q4_sparse_argmax_and_scores_match_full_matvec() -> Result<(), Box<dyn std::error::Error>> {
        let out_dim = 128;
        let in_dim = 96;
        let mut w = Vec::with_capacity(out_dim * in_dim);
        for r in 0..out_dim {
            for c in 0..in_dim {
                w.push((((r * 29 + c * 7) % 101) as f32 - 50.0) * 0.02);
            }
        }
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.31).sin()).collect();

        let q4 = QuantizedMatrix::from_f32_q4(&w, out_dim, in_dim)?;
        let mut q_scratch = vec![0_i8; in_dim];
        let mut full = vec![0.0_f32; out_dim];
        q4.matvec_into(&x, &mut q_scratch, &mut full);

        let subset = [3_usize, 17, 42, 64, 99, 127];
        let mut sparse_scores = Vec::new();
        q4.scores_for_rows(&x, &subset, &mut q_scratch, &mut sparse_scores);
        for (&row, &score) in subset.iter().zip(&sparse_scores) {
            assert!((score - full[row]).abs() < 1e-4);
        }

        let winner = q4
            .argmax_rows(&x, &subset, &mut q_scratch)
            .ok_or("expected winner")?;
        let expected_winner = subset
            .iter()
            .copied()
            .max_by(|&a, &b| full[a].total_cmp(&full[b]).then_with(|| b.cmp(&a)))
            .ok_or("non-empty subset")?;
        assert_eq!(usize::try_from(winner)?, expected_winner);
        Ok(())
    }

    #[test]
    fn matmul4_matches_four_matvecs_for_w8_and_q4() -> Result<(), Box<dyn std::error::Error>> {
        let out_dim = 512;
        let in_dim = 768;
        let mut w = Vec::with_capacity(out_dim * in_dim);
        for r in 0..out_dim {
            for c in 0..in_dim {
                w.push((((r * 23 + c * 11) % 79) as f32 - 39.0) * 0.012);
            }
        }
        let xs: Vec<Vec<f32>> = (0..4)
            .map(|k| {
                (0..in_dim)
                    .map(|i| (((i * (17 + k * 5) + k * 13) % 53) as f32 - 26.0) * 0.04)
                    .collect()
            })
            .collect();

        for format in [QuantFormat::W8, QuantFormat::Q4] {
            let mat = QuantizedMatrix::from_f32_with_format(&w, out_dim, in_dim, format)?;

            let mut ref_outs = vec![vec![0.0_f32; out_dim]; 4];
            let mut q_single = vec![0_i8; in_dim];
            for k in 0..4 {
                mat.matvec_into(&xs[k], &mut q_single, &mut ref_outs[k]);
            }

            let mut q0 = vec![0_i8; in_dim];
            let mut q1 = vec![0_i8; in_dim];
            let mut q2 = vec![0_i8; in_dim];
            let mut q3 = vec![0_i8; in_dim];
            let mut o0 = vec![0.0_f32; out_dim];
            let mut o1 = vec![0.0_f32; out_dim];
            let mut o2 = vec![0.0_f32; out_dim];
            let mut o3 = vec![0.0_f32; out_dim];

            mat.matmul4_into(
                [&xs[0], &xs[1], &xs[2], &xs[3]],
                [&mut q0, &mut q1, &mut q2, &mut q3],
                [&mut o0, &mut o1, &mut o2, &mut o3],
            );

            for (k, got) in [&o0, &o1, &o2, &o3].iter().enumerate() {
                for row in 0..out_dim {
                    let diff = (got[row] - ref_outs[k][row]).abs();
                    assert!(
                        diff < 1e-4,
                        "{format:?} k={k} row={row}: got={} vs ref={} (diff={diff})",
                        got[row],
                        ref_outs[k][row]
                    );
                }
            }
        }
        Ok(())
    }
}
