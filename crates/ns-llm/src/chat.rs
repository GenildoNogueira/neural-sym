use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

/// Parâmetros de amostragem para o modo `chat` contínuo (sem máscara JSON).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChatSamplingConfig {
    /// `0.0` força decodificação greedy determinística; ex.: `0.7` para conversa natural.
    pub temperature: f32,
    /// Nucleus sampling (`0.0..=1.0`); ex.: `0.85`.
    pub top_p: f32,
    /// Penalidade leve de presença sobre os últimos tokens gerados para evitar loops (ex.: `0.3`).
    pub presence_penalty: f32,
    /// Limite máximo de tokens emitidos por resposta.
    pub max_new_tokens: usize,
}

impl Default for ChatSamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.85,
            presence_penalty: 0.3,
            max_new_tokens: 256,
        }
    }
}

/// Telemetria devolvida ao final de cada turno do `chat`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChatTurnStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub elapsed_ms: f64,
    pub state_bytes: usize,
    /// Fração de colunas da FFN fisicamente puladas por `ReLU² == 0` (disponível no RWKV-7).
    pub ffn_sparsity_ratio: Option<f64>,
}

impl ChatTurnStats {
    #[must_use]
    pub fn tokens_per_sec(&self) -> f64 {
        if self.elapsed_ms <= 0.0 {
            return 0.0;
        }
        (self.generated_tokens as f64) / (self.elapsed_ms / 1_000.0)
    }
}

/// Gerador pseudo-aleatório leve (XorShift64) para nucleus sampling sem dependência externa.
#[derive(Debug, Clone)]
pub(crate) struct ChatRng(u64);

impl ChatRng {
    pub(crate) fn from_clock() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0x9E37_79B9_7F4A_7C15, |d| d.as_nanos() as u64);
        Self(if seed == 0 {
            0xCAFE_BABE_1234_5678
        } else {
            seed
        })
    }

    pub(crate) fn next_unit_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let bits = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
        (bits as f32) / ((1_u32 << 24) as f32)
    }
}

/// Decodificador incremental UTF-8: retém bytes incompletos de acentos (ex.: `ç`, `ã`, `é`)
/// até fechar um codepoint válido, impedindo `` no streaming do terminal.
#[derive(Debug, Default)]
pub(crate) struct Utf8StreamBuffer {
    pending: Vec<u8>,
}

impl Utf8StreamBuffer {
    pub(crate) fn push_bytes(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(valid) => {
                let out = valid.to_owned();
                self.pending.clear();
                out
            }
            Err(err) => {
                let valid_up_to = err.valid_up_to();
                let out = String::from_utf8_lossy(&self.pending[..valid_up_to]).into_owned();
                if let Some(invalid_len) = err.error_len() {
                    self.pending.drain(..valid_up_to + invalid_len);
                } else {
                    self.pending.drain(..valid_up_to);
                }
                out
            }
        }
    }

    pub(crate) fn flush_lossy(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

/// Amostra o próximo token aplicando penalidade de presença + temperatura + nucleus (`top_p`).
pub(crate) fn sample_logits(
    logits: &mut [f32],
    recent_tokens: &VecDeque<usize>,
    cfg: ChatSamplingConfig,
    rng: &mut ChatRng,
) -> usize {
    if logits.is_empty() {
        return 0;
    }

    if cfg.presence_penalty > 0.0 {
        // Penalidade de *presença*: aplicada uma única vez por token distinto da janela
        // (sem acumular por repetição, o que a transformaria em penalidade de frequência).
        let mut seen: Vec<usize> = recent_tokens.iter().copied().collect();
        seen.sort_unstable();
        seen.dedup();
        for tok in seen {
            if let Some(slot) = logits.get_mut(tok) {
                *slot -= cfg.presence_penalty;
            }
        }
    }

    if cfg.temperature <= 1e-5 {
        return argmax_logits(logits);
    }

    let inv_temp = 1.0 / cfg.temperature;
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Mantém apenas os candidatos relevantes (top-64) para nucleus sampling em O(V) sem ordenar V inteiro
    const TOP_K_CAP: usize = 64;
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(TOP_K_CAP + 1);
    let mut min_top_val = f32::NEG_INFINITY;

    for (idx, &val) in logits.iter().enumerate() {
        if top.len() < TOP_K_CAP || val > min_top_val {
            top.push((idx, val));
            if top.len() > TOP_K_CAP {
                let mut worst_pos = 0_usize;
                for i in 1..top.len() {
                    if top[i].1 < top[worst_pos].1 {
                        worst_pos = i;
                    }
                }
                top.swap_remove(worst_pos);
                min_top_val = top.iter().map(|&(_, v)| v).fold(f32::INFINITY, f32::min);
            }
        }
    }

    top.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut probs: Vec<(usize, f32)> = Vec::with_capacity(top.len());
    let mut prob_sum = 0.0_f32;
    for &(idx, val) in &top {
        let e = ((val - max_logit) * inv_temp).exp();
        probs.push((idx, e));
        prob_sum += e;
    }

    if prob_sum <= 0.0 || !prob_sum.is_finite() {
        return top.first().map_or(0, |&(idx, _)| idx);
    }

    let top_p = cfg.top_p.clamp(0.05, 1.0);
    let mut cum = 0.0_f32;
    let mut cutoff = probs.len();
    for (i, (_, p)) in probs.iter_mut().enumerate() {
        *p /= prob_sum;
        cum += *p;
        if cum >= top_p {
            cutoff = i + 1;
            break;
        }
    }
    probs.truncate(cutoff.max(1));

    let renorm: f32 = probs.iter().map(|&(_, p)| p).sum();
    let target = rng.next_unit_f32() * renorm;
    let mut acc = 0.0_f32;
    for &(idx, p) in &probs {
        acc += p;
        if acc >= target {
            return idx;
        }
    }
    probs.first().map_or(0, |&(idx, _)| idx)
}

fn argmax_logits(logits: &[f32]) -> usize {
    let mut best_idx = 0_usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx
}
