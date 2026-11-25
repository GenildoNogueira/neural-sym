use safetensors::SafeTensors;

use crate::error::LlmError;
use crate::quant::QuantizedMatrix;
use crate::rwkv7::SparseColQuantizedMatrix;

/// Estatísticas exatas de tráfego físico de memória em 1 passo de [`SparseMoeFfn`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MoeStepStats {
    pub experts_total: usize,
    pub experts_active: usize,
    pub cols_total: usize,
    pub cols_active: usize,
    pub total_weight_bytes: usize,
    pub bytes_read: usize,
}

impl MoeStepStats {
    /// Fração de bytes da FFN que **não saíram da RAM** neste token (`0.0 .. 1.0`).
    #[must_use]
    pub fn ram_skip_ratio(self) -> f64 {
        if self.total_weight_bytes == 0 {
            return 0.0;
        }
        let skipped = self.total_weight_bytes.saturating_sub(self.bytes_read);
        (skipped as f64) / (self.total_weight_bytes as f64)
    }

    /// Fração total de colunas FFN puladas (especialistas inativos + colunas zeradas/podadas).
    #[must_use]
    pub fn col_skip_ratio(self) -> f64 {
        if self.cols_total == 0 {
            return 0.0;
        }
        let skipped = self.cols_total.saturating_sub(self.cols_active);
        (skipped as f64) / (self.cols_total as f64)
    }
}

/// Buffers pré-alocados para executar [`SparseMoeFfn::forward_into`] com zero alocações no heap.
#[derive(Debug, Clone)]
pub struct SparseMoeScratch {
    router_logits: Box<[f32]>,
    router_ranked: Vec<(usize, f32)>,
    expert_hidden: Box<[f32]>,
    active_cols: Vec<(usize, f32)>,
}

impl SparseMoeScratch {
    #[must_use]
    pub fn new(num_experts: usize, expert_ffn_dim: usize) -> Self {
        Self {
            router_logits: vec![0.0_f32; num_experts].into_boxed_slice(),
            router_ranked: Vec::with_capacity(num_experts),
            expert_hidden: vec![0.0_f32; expert_ffn_dim].into_boxed_slice(),
            active_cols: Vec::with_capacity(expert_ffn_dim),
        }
    }
}

/// Um especialista da FFN: projeção de entrada `key` (`[expert_ffn_dim, d_model]`)
/// acoplada a uma matriz `value` column-major esparsa (`[d_model, expert_ffn_dim]`).
#[derive(Debug, Clone)]
pub struct SparseExpert {
    key: QuantizedMatrix,
    value: SparseColQuantizedMatrix,
}

impl SparseExpert {
    /// Constrói um especialista W8 a partir de matrizes densas `f32` row-major.
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError`] se as dimensões não baterem com `d_model` e `expert_ffn_dim`.
    pub fn from_f32(
        key_f32: &[f32],
        value_f32: &[f32],
        d_model: usize,
        expert_ffn_dim: usize,
    ) -> Result<Self, LlmError> {
        let key = QuantizedMatrix::from_f32(key_f32, expert_ffn_dim, d_model)?;
        let value = SparseColQuantizedMatrix::from_row_major_f32(value_f32, d_model, expert_ffn_dim)?;
        Ok(Self { key, value })
    }

    /// Carrega `<prefix>.key.weight` e `<prefix>.value.weight` de um `.safetensors`.
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::ModelLoad`] se algum tensor estiver ausente ou com shape inválido.
    pub fn load_from_safetensors(
        st: &SafeTensors<'_>,
        prefix: &str,
        d_model: usize,
        expert_ffn_dim: usize,
    ) -> Result<Self, LlmError> {
        let key = QuantizedMatrix::load_from_safetensors(
            st,
            &format!("{prefix}.key.weight"),
            expert_ffn_dim,
            d_model,
        )?;
        let value = SparseColQuantizedMatrix::load_from_safetensors(
            st,
            &format!("{prefix}.value.weight"),
            d_model,
            expert_ffn_dim,
        )?;
        Ok(Self { key, value })
    }

    #[must_use]
    pub const fn weight_bytes(&self) -> usize {
        self.key.weight_bytes() + self.value.weight_bytes()
    }
}

/// FFN Mixture-of-Experts com dupla esparsidade (Top-K especialistas × ReLU²/Top-M neurônios).
///
/// Desacopla a capacidade total em RAM (`num_experts * expert_ffn_dim`) do orçamento fixo
/// de bytes lidos por token (`top_k * active_budget`), mantendo a latência por token
/// quase constante mesmo multiplicando o total de parâmetros por 8x–32x.
#[derive(Debug, Clone)]
pub struct SparseMoeFfn {
    d_model: usize,
    expert_ffn_dim: usize,
    top_k: usize,
    /// `0` = sem teto rígido (apenas esparsidade natural do `ReLU²`); `> 0` = mantém no máximo M colunas/especialista.
    neuron_top_k: usize,
    router: QuantizedMatrix,
    experts: Box<[SparseExpert]>,
    total_weight_bytes: usize,
}

impl SparseMoeFfn {
    /// Constrói uma camada [`SparseMoeFfn`] validando todas as invariantes dimensionais.
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::InvalidConfig`] se `d_model`, `expert_ffn_dim`, `experts` ou `top_k`
    /// forem inválidos.
    pub fn new(
        d_model: usize,
        expert_ffn_dim: usize,
        top_k: usize,
        neuron_top_k: usize,
        router: QuantizedMatrix,
        experts: Vec<SparseExpert>,
    ) -> Result<Self, LlmError> {
        if d_model == 0 || expert_ffn_dim == 0 {
            return Err(LlmError::InvalidConfig(
                "d_model and expert_ffn_dim must be positive",
            ));
        }
        if experts.is_empty() {
            return Err(LlmError::InvalidConfig(
                "SparseMoeFfn requires at least one expert",
            ));
        }
        if top_k == 0 || top_k > experts.len() {
            return Err(LlmError::InvalidConfig(
                "moe top_k must be in 1..=num_experts",
            ));
        }
        for exp in &experts {
            if exp.value.in_dim() != expert_ffn_dim || exp.value.out_dim() != d_model {
                return Err(LlmError::InvalidConfig(
                    "expert dimensions do not match SparseMoeFfn config",
                ));
            }
        }

        let experts_bytes: usize = experts.iter().map(SparseExpert::weight_bytes).sum();
        let total_weight_bytes = router.weight_bytes().saturating_add(experts_bytes);

        Ok(Self {
            d_model,
            expert_ffn_dim,
            top_k,
            neuron_top_k,
            router,
            experts: experts.into_boxed_slice(),
            total_weight_bytes,
        })
    }

    /// Carrega roteador e todos os especialistas (`<prefix>.router.weight`,
    /// `<prefix>.experts.{e}.key.weight`, `<prefix>.experts.{e}.value.weight`) de um `.safetensors`.
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError`] se tensores estiverem ausentes ou se a configuração for inválida.
    pub fn load_from_safetensors(
        st: &SafeTensors<'_>,
        prefix: &str,
        d_model: usize,
        expert_ffn_dim: usize,
        num_experts: usize,
        top_k: usize,
        neuron_top_k: usize,
    ) -> Result<Self, LlmError> {
        let router = QuantizedMatrix::load_from_safetensors(
            st,
            &format!("{prefix}.router.weight"),
            num_experts,
            d_model,
        )?;
        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let exp_prefix = format!("{prefix}.experts.{e}");
            experts.push(SparseExpert::load_from_safetensors(
                st,
                &exp_prefix,
                d_model,
                expert_ffn_dim,
            )?);
        }
        Self::new(d_model, expert_ffn_dim, top_k, neuron_top_k, router, experts)
    }

    #[must_use]
    pub const fn num_experts(&self) -> usize {
        self.experts.len()
    }

    #[must_use]
    pub const fn top_k(&self) -> usize {
        self.top_k
    }

    #[must_use]
    pub const fn expert_ffn_dim(&self) -> usize {
        self.expert_ffn_dim
    }

    #[must_use]
    pub const fn total_cols(&self) -> usize {
        self.experts.len().saturating_mul(self.expert_ffn_dim)
    }

    #[must_use]
    pub const fn weight_bytes(&self) -> usize {
        self.total_weight_bytes
    }

    #[must_use]
    pub fn alloc_scratch(&self) -> SparseMoeScratch {
        SparseMoeScratch::new(self.experts.len(), self.expert_ffn_dim)
    }

    /// Executa 1 passo da FFN MoE acumulando o resultado em `out` (`out += Moe(x_shift)`),
    /// sem alocar no heap e tocando na RAM **apenas** o roteador, os `top_k` especialistas
    /// vencedores e as colunas ativas sobreviventes ao `ReLU²` / `neuron_top_k`.
    pub fn forward_into(
        &self,
        x_shift: &[f32],
        q_input_scratch: &mut [i8],
        scratch: &mut SparseMoeScratch,
        out: &mut [f32],
    ) -> MoeStepStats {
        debug_assert_eq!(x_shift.len(), self.d_model);
        debug_assert_eq!(out.len(), self.d_model);
        debug_assert!(q_input_scratch.len() >= self.d_model);
        debug_assert_eq!(scratch.router_logits.len(), self.experts.len());
        debug_assert_eq!(scratch.expert_hidden.len(), self.expert_ffn_dim);

        // 1. Roteador minúsculo decide quais `top_k` especialistas acordar
        self.router
            .matvec_into(x_shift, q_input_scratch, &mut scratch.router_logits);

        scratch.router_ranked.clear();
        for (idx, &score) in scratch.router_logits.iter().enumerate() {
            scratch.router_ranked.push((idx, score));
        }

        let k = self.top_k;
        if k < scratch.router_ranked.len() {
            scratch.router_ranked.select_nth_unstable_by(k - 1, |a, b| {
                b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
            });
        }
        scratch.router_ranked[..k]
            .sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // Softmax normalizado apenas sobre os `top_k` vencedores
        let max_logit = scratch.router_ranked[0].1;
        let mut weight_sum = 0.0_f32;
        for &(_, logit) in &scratch.router_ranked[..k] {
            weight_sum += (logit - max_logit).exp();
        }
        let inv_sum = if weight_sum > 0.0 && weight_sum.is_finite() {
            1.0 / weight_sum
        } else {
            1.0 / (k as f32)
        };

        let mut bytes_read = self.router.weight_bytes();
        let mut cols_active = 0_usize;

        // 2. Avalia exclusivamente os `top_k` especialistas vencedores (os demais nunca saem da RAM)
        for slot in 0..k {
            let (expert_idx, logit) = scratch.router_ranked[slot];
            let gate_weight = (logit - max_logit).exp() * inv_sum;
            let expert = &self.experts[expert_idx];

            expert
                .key
                .matvec_into(x_shift, q_input_scratch, &mut scratch.expert_hidden);
            bytes_read = bytes_read.saturating_add(expert.key.weight_bytes());

            // Ativação ReLU² do RWKV-7
            for val in scratch.expert_hidden.iter_mut() {
                let r = val.max(0.0);
                *val = r * r;
            }

            // Teto rígido Top-M de neurônios por especialista (garante teto de leitura na `value`)
            if self.neuron_top_k > 0 && self.neuron_top_k < self.expert_ffn_dim {
                scratch.active_cols.clear();
                for (col, &val) in scratch.expert_hidden.iter().enumerate() {
                    if val > 0.0 {
                        scratch.active_cols.push((col, val));
                    }
                }
                if scratch.active_cols.len() > self.neuron_top_k {
                    let keep = self.neuron_top_k;
                    scratch.active_cols.select_nth_unstable_by(keep - 1, |a, b| {
                        b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
                    });
                    for &(col, _) in &scratch.active_cols[keep..] {
                        scratch.expert_hidden[col] = 0.0;
                    }
                }
            }

            let active = expert.value.accumulate_sparse_scaled(
                &scratch.expert_hidden,
                gate_weight,
                out,
            );
            cols_active = cols_active.saturating_add(active);
            bytes_read =
                bytes_read.saturating_add(active.saturating_mul(expert.value.col_stride_bytes()));
        }

        MoeStepStats {
            experts_total: self.experts.len(),
            experts_active: k,
            cols_total: self.total_cols(),
            cols_active,
            total_weight_bytes: self.total_weight_bytes,
            bytes_read,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_deterministic_moe(
        d_model: usize,
        expert_ffn_dim: usize,
        num_experts: usize,
        top_k: usize,
        neuron_top_k: usize,
    ) -> Result<SparseMoeFfn, LlmError> {
        let mut router_w = vec![0.0_f32; num_experts * d_model];
        for e in 0..num_experts {
            router_w[e * d_model + (e % d_model)] = 1.0 + (e as f32) * 0.01;
        }
        let router = QuantizedMatrix::from_f32(&router_w, num_experts, d_model)?;

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let mut key_w = vec![0.0_f32; expert_ffn_dim * d_model];
            for r in 0..expert_ffn_dim {
                // Alterna sinais positivos/negativos para simular esparsidade natural ReLU²
                let sign = if (r + e) % 3 == 0 { 1.0 } else { -1.0 };
                for c in 0..d_model {
                    let v = (((r * 13 + c * 7 + e * 5) % 19) as f32 + 1.0) * 0.02 * sign;
                    key_w[r * d_model + c] = v;
                }
            }
            let mut val_w = vec![0.0_f32; d_model * expert_ffn_dim];
            for r in 0..d_model {
                for c in 0..expert_ffn_dim {
                    val_w[r * expert_ffn_dim + c] =
                        (((r * 11 + c * 3 + e) % 17) as f32 - 8.0) * 0.03;
                }
            }
            experts.push(SparseExpert::from_f32(&key_w, &val_w, d_model, expert_ffn_dim)?);
        }

        SparseMoeFfn::new(d_model, expert_ffn_dim, top_k, neuron_top_k, router, experts)
    }

    #[test]
    fn inactive_experts_are_never_executed_and_exact_top_k_output_matches()
    -> Result<(), Box<dyn std::error::Error>> {
        let d = 8;
        let ffn = 4;
        let num_experts = 6;

        // Roteador força especialistas 1 e 4 como únicos vencedores para x = [1, 1, ..., 1]
        let mut router_w = vec![0.0_f32; num_experts * d];
        router_w[d] = 10.0; // logit alto -> vencedor #1
        router_w[4 * d] = 10.0; // logit empatado -> vencedor #2 (softmax 0.5 / 0.5)
        for inactive in [0_usize, 2, 3, 5] {
            router_w[inactive * d] = -10.0;
        }
        let router = QuantizedMatrix::from_f32(&router_w, num_experts, d)?;

        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let (k_val, v_val) = match e {
                // Especialista 1: ativa só a coluna 0 com valor +2.0 em out[0]
                1 => (0.5_f32, 1.0_f32),
                // Especialista 4: ativa só a coluna 0 com valor +3.0 em out[0]
                4 => (0.5_f32, 2.0_f32),
                // Especialistas inativos: valores gigantes que contaminariam `out` se fossem tocados
                _ => (10.0_f32, 50.0_f32),
            };
            let mut key_w = vec![0.0_f32; ffn * d];
            let mut val_w = vec![0.0_f32; d * ffn];
            key_w[0] = k_val;
            val_w[0] = v_val;
            experts.push(SparseExpert::from_f32(&key_w, &val_w, d, ffn)?);
        }

        let moe = SparseMoeFfn::new(d, ffn, 2, 0, router, experts)?;
        let mut q_in = vec![0_i8; d];
        let mut scratch = moe.alloc_scratch();
        let x = [2.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut out = vec![0.0_f32; d];

        let stats = moe.forward_into(&x, &mut q_in, &mut scratch, &mut out);

        // Para x[0]=2.0: key_out = 2.0 * 0.5 = 1.0 -> ReLU²(1.0) = 1.0
        // Expert 1 contribui 0.5 * 1.0 = 0.5; Expert 4 contribui 0.5 * 2.0 = 1.0 -> soma = 1.5
        assert!(
            (out[0] - 1.5).abs() < 0.08,
            "expected ~1.5 (only experts 1 and 4), got {}",
            out[0]
        );
        assert!(out[1..].iter().all(|&v| v.abs() < 1e-5));
        assert_eq!(stats.experts_active, 2);
        assert_eq!(stats.experts_total, 6);
        assert_eq!(stats.cols_active, 2); // apenas 1 coluna ativa em cada um dos 2 especialistas
        assert!(stats.ram_skip_ratio() > 0.70);
        Ok(())
    }

    #[test]
    fn scaling_capacity_32x_keeps_active_expert_ram_budget_bounded()
    -> Result<(), Box<dyn std::error::Error>> {
        let d = 64;
        let expert_ffn = 128;
        let top_k = 2;
        let neuron_top_k = 16;

        let x: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.31).cos() + 0.5).collect();
        let mut q_in = vec![0_i8; d];

        let moe_small = build_deterministic_moe(d, expert_ffn, 4, top_k, neuron_top_k)?;
        let moe_large = build_deterministic_moe(d, expert_ffn, 128, top_k, neuron_top_k)?;

        let mut sc_small = moe_small.alloc_scratch();
        let mut sc_large = moe_large.alloc_scratch();
        let mut out_small = vec![0.0_f32; d];
        let mut out_large = vec![0.0_f32; d];

        let st_small = moe_small.forward_into(&x, &mut q_in, &mut sc_small, &mut out_small);
        let st_large = moe_large.forward_into(&x, &mut q_in, &mut sc_large, &mut out_large);

        // Capacidade total cresceu ~32x
        let capacity_ratio =
            (st_large.total_weight_bytes as f64) / (st_small.total_weight_bytes as f64);
        assert!(
            capacity_ratio > 30.0,
            "expected >30x capacity scaling, got {capacity_ratio:.2}x"
        );

        // Teto rígido Top-M garante no máximo top_k * neuron_top_k colunas lidas na `value`
        assert!(st_small.cols_active <= top_k * neuron_top_k);
        assert!(st_large.cols_active <= top_k * neuron_top_k);

        // Mesmo com 32x mais parâmetros, o modelo grande pula >97% da RAM da FFN!
        assert!(
            st_large.ram_skip_ratio() > 0.97,
            "expected >97% RAM skipped on 128-expert MoE, got {:.2}%",
            st_large.ram_skip_ratio() * 100.0
        );

        // Excluindo o roteador (que cresce linearmente e é minúsculo),
        // o tráfego de especialistas lido por token é estritamente limitado pelo mesmo teto!
        let max_expert_bytes_per_step = top_k
            * (moe_large.experts[0].key.weight_bytes()
                + neuron_top_k * moe_large.experts[0].value.col_stride_bytes());
        let expert_bytes_large = st_large
            .bytes_read
            .saturating_sub(moe_large.router.weight_bytes());
        assert!(expert_bytes_large <= max_expert_bytes_per_step);
        Ok(())
    }

    #[test]
    fn neuron_top_k_keeps_strictly_highest_relu_squared_activations()
    -> Result<(), Box<dyn std::error::Error>> {
        let d = 4;
        let ffn = 6;
        let router = QuantizedMatrix::from_f32(&[1.0, 0.0, 0.0, 0.0], 1, d)?;

        // Para x = [1, 0, 0, 0], ativações pré-ReLU² são [0.2, 0.9, 0.4, 1.5, 0.1, 1.2]
        // Top-2 neurônios devem ser exclusivamente colunas 3 (1.5² = 2.25) e 5 (1.2² = 1.44)
        let mut key_w = vec![0.0_f32; ffn * d];
        let activations = [0.2_f32, 0.9, 0.4, 1.5, 0.1, 1.2];
        for (col, &a) in activations.iter().enumerate() {
            key_w[col * d] = a;
        }

        // value identidade por coluna em out[0]: cada coluna j soma 1.0 * act[j]² em out[0]
        let mut val_w = vec![0.0_f32; d * ffn];
        for col in 0..ffn {
            val_w[col] = 1.0;
        }

        let expert = SparseExpert::from_f32(&key_w, &val_w, d, ffn)?;
        let moe = SparseMoeFfn::new(d, ffn, 1, 2, router, vec![expert])?;

        let mut q_in = vec![0_i8; d];
        let mut scratch = moe.alloc_scratch();
        let mut out = vec![0.0_f32; d];

        let stats = moe.forward_into(&[1.0, 0.0, 0.0, 0.0], &mut q_in, &mut scratch, &mut out);
        let expected = 1.5_f32.powi(2) + 1.2_f32.powi(2); // 3.69 (somente cols 3 e 5)
        assert_eq!(stats.cols_active, 2);
        assert!(
            (out[0] - expected).abs() < 0.12,
            "expected ~{expected:.3}, got {:.3}",
            out[0]
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_moe_configuration() -> Result<(), Box<dyn std::error::Error>> {
        let router = QuantizedMatrix::from_f32(&[1.0; 8], 2, 4)?;
        let exp = SparseExpert::from_f32(&[0.1; 16], &[0.1; 16], 4, 4)?;

        // top_k = 0 deve falhar
        assert!(matches!(
            SparseMoeFfn::new(4, 4, 0, 0, router.clone(), vec![exp.clone()]),
            Err(LlmError::InvalidConfig(_))
        ));
        // top_k > num_experts deve falhar
        assert!(matches!(
            SparseMoeFfn::new(4, 4, 3, 0, router, vec![exp]),
            Err(LlmError::InvalidConfig(_))
        ));
        Ok(())
    }
}
