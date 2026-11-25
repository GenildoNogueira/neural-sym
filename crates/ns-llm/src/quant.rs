use llguidance::toktrie::{SimpleVob, TokenId};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};

use crate::error::LlmError;

/// Limite seguro para garantir ausência matemática de overflow em acumuladores `i32`:
/// `127 * 127 * 131_072 = 2_114_060_288 <= i32::MAX (2_147_483_647)`.
pub const MAX_QUANT_IN_DIM: usize = 131_072;

/// Volume mínimo de multiplicações (`out_dim * in_dim`) para ativar o Rayon em `matvec_into`.
/// Decidir por volume (e não por nº de linhas) paraleliza as projeções 768×768 do RWKV-7
/// e mantém sequenciais as matrizes LoRA pequenas.
const PARALLEL_MATVEC_MIN_MACS: usize = 512 * 1024;
/// Linhas por tarefa do Rayon: agrupar reduz o overhead de agendamento por linha.
const PARALLEL_ROW_CHUNK: usize = 64;
const PARALLEL_SPARSE_THRESHOLD: usize = 1_024;

/// Matriz linear quantizada simetricamente por linha (`W8A8`: pesos `i8` + escala `f32` por linha).
#[derive(Debug, Clone)]
pub struct QuantizedMatrix {
    out_dim: usize,
    in_dim: usize,
    /// Layout row-major fixo: `out_dim * in_dim` bytes.
    data: Box<[i8]>,
    /// Fator de escala `max(|row|) / 127.0` para cada uma das `out_dim` linhas.
    scales: Box<[f32]>,
}

impl QuantizedMatrix {
    /// Carrega uma matriz do `.safetensors`:
    /// - Se existirem `<name>.q` (`I8`) e `<name>.scale` (`F32`), carrega direto sem re-quantizar.
    /// - Caso contrário, lê `<name>` (`F32`) e quantiza por linha no carregamento.
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

            let data: Box<[i8]> = q_view
                .data()
                .iter()
                .map(|&b| i8::from_ne_bytes([b]))
                .collect::<Vec<i8>>()
                .into_boxed_slice();

            let mut scales = Vec::with_capacity(out_dim);
            for chunk in s_view.data().chunks_exact(4) {
                scales.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }

            return Ok(Self {
                out_dim,
                in_dim,
                data,
                scales: scales.into_boxed_slice(),
            });
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

        Self::from_f32(&weights_f32, out_dim, in_dim)
    }

    /// Quantiza uma matriz `f32` row-major em W8 (`i8` + escala por linha).
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::ModelLoad`] se as dimensões não baterem com o slice ou excederem o limite.
    pub fn from_f32(weights: &[f32], out_dim: usize, in_dim: usize) -> Result<Self, LlmError> {
        let expected = out_dim
            .checked_mul(in_dim)
            .ok_or_else(|| LlmError::ModelLoad("matrix element count overflow".to_owned()))?;
        if in_dim == 0 || in_dim > MAX_QUANT_IN_DIM || weights.len() != expected {
            return Err(LlmError::ModelLoad(format!(
                "invalid f32 buffer length {} for [{out_dim}, {in_dim}]",
                weights.len()
            )));
        }

        let mut data = vec![0_i8; expected];
        let mut scales = vec![0.0_f32; out_dim];

        for row in 0..out_dim {
            let start = row * in_dim;
            let src = &weights[start..start + in_dim];
            let dst = &mut data[start..start + in_dim];
            scales[row] = quantize_slice_symmetric(src, dst);
        }

        Ok(Self {
            out_dim,
            in_dim,
            data: data.into_boxed_slice(),
            scales: scales.into_boxed_slice(),
        })
    }

    #[must_use]
    pub const fn weight_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * std::mem::size_of::<f32>()
    }

    /// Calcula `out = W * x` em W8A8 sem alocar no heap, reutilizando `q_input_scratch`.
    pub fn matvec_into(&self, x: &[f32], q_input_scratch: &mut [i8], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_dim);
        debug_assert_eq!(out.len(), self.out_dim);
        debug_assert!(q_input_scratch.len() >= self.in_dim);

        let qx = &mut q_input_scratch[..self.in_dim];
        let scale_x = quantize_slice_symmetric(x, qx);
        if scale_x == 0.0 {
            out.fill(0.0);
            return;
        }

        let qx_ref: &[i8] = qx;
        let in_dim = self.in_dim;
        let data = &self.data;
        let scales = &self.scales;

        if self.out_dim.saturating_mul(in_dim) >= PARALLEL_MATVEC_MIN_MACS {
            out.par_chunks_mut(PARALLEL_ROW_CHUNK)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let row0 = chunk_idx * PARALLEL_ROW_CHUNK;
                    for (i, dst) in chunk.iter_mut().enumerate() {
                        let row = row0 + i;
                        let start = row * in_dim;
                        let acc = dot_i8_unrolled(&data[start..start + in_dim], qx_ref);
                        *dst = (acc as f32) * (scales[row] * scale_x);
                    }
                });
        } else {
            for (row, dst) in out.iter_mut().enumerate() {
                let start = row * in_dim;
                let acc = dot_i8_unrolled(&data[start..start + in_dim], qx_ref);
                *dst = (acc as f32) * (scales[row] * scale_x);
            }
        }
    }

    /// Avalia o produto escalar W8A8 apenas sobre os índices de linha em `rows`
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

        let qx = &mut q_input_scratch[..self.in_dim];
        let scale_x = quantize_slice_symmetric(x, qx);
        if scale_x == 0.0 {
            return TokenId::try_from(rows[0]).ok();
        }

        let qx_ref: &[i8] = qx;
        let in_dim = self.in_dim;
        let data = &self.data;
        let scales = &self.scales;

        let best = if rows.len() >= PARALLEL_SPARSE_THRESHOLD {
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
        };

        best.and_then(|(row, _)| TokenId::try_from(row).ok())
    }

    /// Projeção esparsa mascarada para a camada `head` usando o bitset `SimpleVob` do `llguidance`.
    pub fn argmax_masked(
        &self,
        x: &[f32],
        mask: &SimpleVob,
        q_input_scratch: &mut [i8],
        allowed_scratch: &mut Vec<usize>,
    ) -> Option<TokenId> {
        allowed_scratch.clear();
        for row in 0..self.out_dim {
            let Ok(tok_id) = TokenId::try_from(row) else {
                break;
            };
            if mask.is_allowed(tok_id) {
                allowed_scratch.push(row);
            }
        }
        self.argmax_rows(x, allowed_scratch, q_input_scratch)
    }

    /// Calcula os logits W8A8 **apenas** para as linhas em `rows`, gravando o resultado
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

        let qx = &mut q_input_scratch[..self.in_dim];
        let scale_x = quantize_slice_symmetric(x, qx);
        if scale_x == 0.0 {
            out_scores.resize(rows.len(), 0.0);
            return;
        }

        let qx_ref: &[i8] = qx;
        let in_dim = self.in_dim;
        let data = &self.data;
        let scales = &self.scales;

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
}

/// Desempate determinístico independente da ordem de threads do `rayon`.
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
    // Faixa garantida em [-127.0, 127.0] e finita antes da conversão inteira.
    #[allow(clippy::cast_possible_truncation)]
    let q = clamped as i8;
    q
}

/// Produto escalar `i8 × i8 → i32`.
///
/// Mantido como iterador simples de propósito: é a forma que o LLVM vetoriza bem
/// (`vpmovsxbw` + `vpmaddwd` em AVX2). A versão anterior, com 4 acumuladores escalares sobre
/// `chunks_exact(8)`, compilava para código **escalar** (~4x mais lento em microbenchmark).
/// Sem overflow em `i32`: ver [`MAX_QUANT_IN_DIM`].
#[inline]
fn dot_i8_unrolled(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

#[cfg(test)]
mod tests {
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
                "row {row}: expected {expected:.4}, got {:.4} (diff={diff:.4})",
                got[row]
            );
        }

        // 4x mais leve que F32 (in_dim bytes + 4 bytes de escala por linha)
        assert_eq!(qmat.weight_bytes(), out_dim * (in_dim + 4));
        Ok(())
    }

    #[test]
    fn parallel_matvec_is_bit_exact_with_sequential_reference()
    -> Result<(), Box<dyn std::error::Error>> {
        // 768 x 768 = 589_824 MACs >= PARALLEL_MATVEC_MIN_MACS: exercita o caminho paralelo.
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

        // Referência sequencial: mesma quantização de x e mesmo produto inteiro por linha.
        let mut qx = vec![0_i8; in_dim];
        let scale_x = quantize_slice_symmetric(&x, &mut qx);
        for row in 0..out_dim {
            let start = row * in_dim;
            let acc = dot_i8_unrolled(&qmat.data[start..start + in_dim], &qx);
            let expected = (acc as f32) * (qmat.scales[row] * scale_x);
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

        // Pior caso de magnitude: nenhum overflow em MAX_QUANT_IN_DIM.
        let worst = vec![-127_i8; MAX_QUANT_IN_DIM];
        assert_eq!(
            i64::from(dot_i8_unrolled(&worst, &worst)),
            127 * 127 * MAX_QUANT_IN_DIM as i64
        );
    }
}
