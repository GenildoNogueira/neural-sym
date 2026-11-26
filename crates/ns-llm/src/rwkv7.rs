use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, SimpleVob, TokEnv, TokRxInfo, TokTrie, TokenId};
use llguidance::{Matcher, ParserFactory};
use ns_core::action::LlmAction;
use ns_core::error::LlmPortError;
use ns_core::fact::{Message, Role};
use ns_core::ports::ConstrainedLlm;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};

use crate::chat::{ChatRng, ChatSamplingConfig, ChatTurnStats, Utf8StreamBuffer, sample_logits};
use crate::error::LlmError;
use crate::grammar::action_schema_value;
use crate::isa::{Expect, Op, Sym, decode_one, expect};
use crate::isa_bank::{
    NeuralEntityBank, SlotFilter, extract_prompt_entities, find_entity_token_span,
};
use crate::quant::QuantizedMatrix;
use crate::{SparseMoeFfn, SparseMoeScratch};

const NEG_EXP_NEG_HALF: f32 = -0.606_530_66;
const ISA_OP_COUNT: usize = 9;
const ISA_CTL_COUNT: usize = 3;
const CTL_END: usize = 0;
const CTL_LIT_OPEN: usize = 1;
const CTL_LIT_CLOSE: usize = 2;
const MAX_ISA_REFLIST_ITEMS: usize = 16;
/// Prefixo oficial do RWKV-7 G1 para pular o raciocínio (`<think></think`, sem fechar o `>`).
/// Confira em `RWKV7-G1x-templates.txt` do model card se trocar de checkpoint.
const NO_THINK_SUFFIX: &str = " <think></think";

fn default_arch() -> String {
    "rwkv7".to_owned()
}

fn default_vocab_file() -> String {
    "vocab.bin".to_owned()
}

const fn default_copy_dim() -> usize {
    0
}

const fn default_copy_max_len() -> usize {
    512
}

const fn default_group_norm_eps() -> f32 {
    64e-5
}

const fn default_isa_ptr_dim() -> usize {
    0
}

const fn default_isa_num_vocab() -> usize {
    64
}

const fn default_moe_top_k() -> usize {
    2
}

/// Modo de decodificação usado por [`Rwkv7Llm`] em [`ConstrainedLlm::next_action`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeMode {
    /// Usa ISA automaticamente quando o checkpoint tiver `isa_ptr_dim > 0`; caso contrário JSON.
    #[default]
    Auto,
    /// Força decodificação JSON via `llguidance`.
    Json,
    /// Força decodificação via Dual-Head ISA + `NeuralEntityBank` (Peças A + B + C).
    Isa,
}

/// Configuração do motor RWKV-v7 ("Goose") compatível 1:1 com `training/finetune_rwkv7_g1.py`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rwkv7Config {
    #[serde(default = "default_arch")]
    pub arch: String,
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub decay_lora_dim: usize,
    pub a_lora_dim: usize,
    pub v_lora_dim: usize,
    pub gate_lora_dim: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub layer_norm_eps: f32,
    #[serde(default = "default_group_norm_eps")]
    pub group_norm_eps: f32,
    #[serde(default = "default_vocab_file")]
    pub vocab_file: String,
    #[serde(default = "default_copy_dim")]
    pub copy_dim: usize,
    #[serde(default = "default_copy_max_len")]
    pub copy_max_len: usize,
    /// `0` mantém modo JSON + `llguidance` (padrão); ex.: `64` ativa a Dual-Head ISA + Entity Pointer.
    #[serde(default = "default_isa_ptr_dim")]
    pub isa_ptr_dim: usize,
    /// Quantidade de inteiros `0..isa_num_vocab` representados em 1 passo na cabeça numérica da ISA.
    #[serde(default = "default_isa_num_vocab")]
    pub isa_num_vocab: usize,
    /// `0` mantém FFN densa tradicional; `> 0` ativa `SparseMoeFfn` com N especialistas por camada.
    #[serde(default)]
    pub moe_num_experts: usize,
    /// Quantidade de especialistas ativos por token quando `moe_num_experts > 0`.
    #[serde(default = "default_moe_top_k")]
    pub moe_top_k: usize,
    /// Teto rígido de colunas ativas por especialista (`0` = apenas esparsidade natural `ReLU²`).
    #[serde(default)]
    pub moe_neuron_top_k: usize,
}

/// Sessão de conversa contínua para o RWKV-v7 mantendo o estado matricial O(1) vivo entre turnos.
#[derive(Debug)]
pub struct Rwkv7ChatSession<'a> {
    model: &'a Rwkv7Llm,
    state: Rwkv7State,
    logits_buf: Box<[f32]>,
    rng: ChatRng,
    turn_count: usize,
    total_tokens_seen: usize,
    /// Quando `true`, fecha um bloco `<think></think` vazio logo após `Assistant:`.
    no_think: bool,
    /// `true` se o último token consumido terminou o turno com `\n\n` (já está no estado).
    ends_with_blank_line: bool,
}

impl Rwkv7ChatSession<'_> {
    /// Liga/desliga o modo sem raciocínio (`<think></think` vazio após `Assistant:`).
    pub fn set_no_think(&mut self, no_think: bool) {
        self.no_think = no_think;
    }

    /// Zera o estado recorrente O(1), iniciando uma conversa nova sem recarregar o modelo.
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::Inference`] se o re-ancoramento inicial falhar.
    pub fn reset(&mut self, system_prompt: Option<&str>) -> Result<(), LlmError> {
        let fresh = self.model.start_chat(system_prompt)?;
        self.state = fresh.state;
        self.turn_count = 0;
        self.ends_with_blank_line = false;
        self.total_tokens_seen = fresh.total_tokens_seen;
        Ok(())
    }

    #[must_use]
    pub fn state_bytes(&self) -> usize {
        self.state.state_bytes()
    }

    #[must_use]
    pub fn total_tokens_seen(&self) -> usize {
        self.total_tokens_seen
    }

    #[must_use]
    pub fn ffn_sparsity_ratio(&self) -> f64 {
        self.state.ffn_sparsity_ratio()
    }

    /// Alimenta apenas o turno atual do usuário no estado vivo e faz streaming da resposta via `on_chunk`.
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::Inference`] se algum token estiver fora do vocabulário.
    pub fn send_message_streaming<F>(
        &mut self,
        user_message: &str,
        sampling: ChatSamplingConfig,
        mut on_chunk: F,
    ) -> Result<ChatTurnStats, LlmError>
    where
        F: FnMut(&str),
    {
        let cleaned = user_message.trim().replace("\r\n", "\n");
        let think_suffix = if self.no_think { NO_THINK_SUFFIX } else { "" };
        // O `\n\n` que encerrou o turno anterior já foi alimentado no estado: não repetir.
        let turn_prefix = if self.turn_count == 0 || self.ends_with_blank_line {
            format!("User: {cleaned}\n\nAssistant:")
        } else {
            format!("\n\nUser: {cleaned}\n\nAssistant:{think_suffix}")
        };
        self.ends_with_blank_line = false;

        let prompt_ids = self.model.tok_trie.greedy_tokenize(turn_prefix.as_bytes());
        let t0 = Instant::now();

        for &tok in &prompt_ids {
            let id = usize::try_from(tok)
                .map_err(|_| LlmError::Inference("prompt token id overflow".to_owned()))?;
            self.model.forward_hidden(id, &mut self.state)?;
        }
        self.total_tokens_seen = self.total_tokens_seen.saturating_add(prompt_ids.len());

        let mut recent: VecDeque<usize> = VecDeque::with_capacity(64);
        let mut utf8_buf = Utf8StreamBuffer::default();
        let mut reply_accum = String::with_capacity(256);
        let mut generated_tokens = 0_usize;

        for _ in 0..sampling.max_new_tokens.max(1) {
            self.model.head.matvec_into(
                &self.state.scratch.x_out,
                &mut self.state.scratch.q_input,
                &mut self.logits_buf,
            );

            let next_id = sample_logits(&mut self.logits_buf, &recent, sampling, &mut self.rng);
            if next_id == self.model.cfg.eos_token_id {
                break;
            }

            if recent.len() >= 64 {
                recent.pop_front();
            }
            recent.push_back(next_id);

            let tok_u32 = TokenId::try_from(next_id)
                .map_err(|_| LlmError::Inference("generated token id exceeds u32".to_owned()))?;
            let piece = utf8_buf.push_bytes(self.model.tok_trie.token(tok_u32));

            // Avança o estado com o token gerado antes de verificar parada para manter o estado coerente
            self.model.forward_hidden(next_id, &mut self.state)?;
            generated_tokens = generated_tokens.saturating_add(1);
            self.total_tokens_seen = self.total_tokens_seen.saturating_add(1);

            if !piece.is_empty() {
                let prev_len = reply_accum.len();
                reply_accum.push_str(&piece);

                // Delimitador oficial de fim de turno do RWKV-7 G1: "\n\n" após texto gerado
                if let Some(stop_pos) = reply_accum.find("\n\n") {
                    if stop_pos > prev_len {
                        on_chunk(&reply_accum[prev_len..stop_pos]);
                    }
                    self.ends_with_blank_line = true;
                    break;
                }
                on_chunk(&piece);
            }
        }

        let tail = utf8_buf.flush_lossy();
        if !tail.is_empty() && !reply_accum.contains("\n\n") {
            on_chunk(&tail);
        }

        self.turn_count = self.turn_count.saturating_add(1);

        Ok(ChatTurnStats {
            prompt_tokens: prompt_ids.len(),
            generated_tokens,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1_000.0,
            state_bytes: self.state.state_bytes(),
            ffn_sparsity_ratio: Some(self.state.ffn_sparsity_ratio()),
        })
    }
}

#[derive(Debug, Clone)]
struct IsaHeadWeights {
    d_model: usize,
    ptr_dim: usize,
    num_vocab: usize,
    /// `[9, d_model]` — embeddings de entrada para `Op::ALL[0..9]`.
    op_emb: Box<[f32]>,
    /// `[9, d_model]` — cabeça linear de classificação de operação.
    op_head: Box<[f32]>,
    /// `[num_vocab, d_model]` — embeddings de entrada para `%0..%num_vocab`.
    num_emb: Box<[f32]>,
    /// `[num_vocab, d_model]` — cabeça linear para argumentos numéricos (`limit`, `max_hops`).
    num_head: Box<[f32]>,
    /// `[3, d_model]` — embeddings de controle (`END`, `LIT_OPEN`, `LIT_CLOSE`).
    ctl_emb: Box<[f32]>,
    /// `[3, d_model]` — cabeça linear de controle (`END`, `LIT_OPEN`, `LIT_CLOSE`).
    ctl_head: Box<[f32]>,
    /// `[ptr_dim, d_model]` — projeção Query de ponteiro de entidade.
    ptr_q: Box<[f32]>,
    /// `[ptr_dim, d_model]` — projeção Key sobre `e_s` pooled de cada entidade.
    ptr_k: Box<[f32]>,
    /// `[d_model, ptr_dim]` — projeta `K_s` de volta para `d_model` na realimentação do estado.
    ptr_in: Box<[f32]>,
    /// `[d_model]` — vetor de tag somado a `ptr_in * K_s` antes de `forward_hidden_vec`.
    ptr_tag: Box<[f32]>,
}

impl IsaHeadWeights {
    fn load(
        st: &SafeTensors<'_>,
        d_model: usize,
        ptr_dim: usize,
        num_vocab: usize,
    ) -> Result<Self, LlmError> {
        Ok(Self {
            d_model,
            ptr_dim,
            num_vocab,
            op_emb: read_f32_boxed(st, "isa.op_emb.weight", &[ISA_OP_COUNT, d_model])?,
            op_head: read_f32_boxed(st, "isa.op_head.weight", &[ISA_OP_COUNT, d_model])?,
            num_emb: read_f32_boxed(st, "isa.num_emb.weight", &[num_vocab, d_model])?,
            num_head: read_f32_boxed(st, "isa.num_head.weight", &[num_vocab, d_model])?,
            ctl_emb: read_f32_boxed(st, "isa.ctl_emb.weight", &[ISA_CTL_COUNT, d_model])?,
            ctl_head: read_f32_boxed(st, "isa.ctl_head.weight", &[ISA_CTL_COUNT, d_model])?,
            ptr_q: read_f32_boxed(st, "isa.ptr_q.weight", &[ptr_dim, d_model])?,
            ptr_k: read_f32_boxed(st, "isa.ptr_k.weight", &[ptr_dim, d_model])?,
            ptr_in: read_f32_boxed(st, "isa.ptr_in.weight", &[d_model, ptr_dim])?,
            ptr_tag: read_f32_boxed(st, "isa.ptr_tag", &[d_model])?,
        })
    }

    const fn weight_bytes(&self) -> usize {
        (self.op_emb.len()
            + self.op_head.len()
            + self.num_emb.len()
            + self.num_head.len()
            + self.ctl_emb.len()
            + self.ctl_head.len()
            + self.ptr_q.len()
            + self.ptr_k.len()
            + self.ptr_in.len()
            + self.ptr_tag.len())
            * std::mem::size_of::<f32>()
    }

    #[inline]
    fn dot_row(mat: &[f32], row: usize, cols: usize, x: &[f32]) -> f32 {
        let slice = &mat[row * cols..(row + 1) * cols];
        let mut acc = 0.0_f32;
        for (&w, &v) in slice.iter().zip(x.iter()) {
            acc += w * v;
        }
        acc
    }

    fn argmax_rows(mat: &[f32], rows: usize, cols: usize, x: &[f32]) -> usize {
        let mut best_idx = 0_usize;
        let mut best_val = f32::NEG_INFINITY;
        for r in 0..rows {
            let s = Self::dot_row(mat, r, cols, x);
            if s > best_val {
                best_val = s;
                best_idx = r;
            }
        }
        best_idx
    }

    fn project_ptr_k(&self, pooled_hidden: &[f32]) -> Box<[f32]> {
        let mut k = vec![0.0_f32; self.ptr_dim];
        for r in 0..self.ptr_dim {
            k[r] = Self::dot_row(&self.ptr_k, r, self.d_model, pooled_hidden);
        }
        k.into_boxed_slice()
    }

    fn project_ptr_q(&self, h: &[f32], out: &mut [f32]) {
        for r in 0..self.ptr_dim {
            out[r] = Self::dot_row(&self.ptr_q, r, self.d_model, h);
        }
    }

    fn build_ptr_feedback(&self, ptr_key: &[f32], out_emb: &mut [f32]) {
        for r in 0..self.d_model {
            out_emb[r] = self.ptr_tag[r] + Self::dot_row(&self.ptr_in, r, self.ptr_dim, ptr_key);
        }
    }
}

#[derive(Debug, Clone)]
struct LayerNormWeights {
    weight: Box<[f32]>,
    bias: Box<[f32]>,
}

/// Tabela de embeddings com suporte a F32 ou I8 por linha (economiza ~150 MiB no RWKV-7 0.1B).
#[derive(Debug, Clone)]
enum EmbeddingTable {
    F32 {
        data: Box<[f32]>,
        d_model: usize,
    },
    I8 {
        data: Box<[i8]>,
        scales: Box<[f32]>,
        d_model: usize,
    },
}

impl EmbeddingTable {
    fn load(st: &SafeTensors<'_>, vocab_size: usize, d_model: usize) -> Result<Self, LlmError> {
        let q_name = "emb.weight.q";
        let s_name = "emb.weight.scale";
        if let (Ok(q_view), Ok(s_view)) = (st.tensor(q_name), st.tensor(s_name)) {
            if q_view.dtype() != Dtype::I8 || q_view.shape() != [vocab_size, d_model] {
                return Err(LlmError::ModelLoad(format!(
                    "invalid '{q_name}' dtype {:?} or shape {:?}",
                    q_view.dtype(),
                    q_view.shape()
                )));
            }
            if s_view.dtype() != Dtype::F32 || s_view.shape() != [vocab_size] {
                return Err(LlmError::ModelLoad(format!(
                    "invalid '{s_name}' dtype {:?} or shape {:?}",
                    s_view.dtype(),
                    s_view.shape()
                )));
            }
            let data: Box<[i8]> = q_view
                .data()
                .iter()
                .map(|&b| i8::from_ne_bytes([b]))
                .collect::<Vec<i8>>()
                .into_boxed_slice();
            let mut scales = Vec::with_capacity(vocab_size);
            for chunk in s_view.data().chunks_exact(4) {
                scales.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            return Ok(Self::I8 {
                data,
                scales: scales.into_boxed_slice(),
                d_model,
            });
        }

        let data = read_f32_boxed(st, "emb.weight", &[vocab_size, d_model])?;
        Ok(Self::F32 { data, d_model })
    }

    fn lookup_into(&self, token_id: usize, out: &mut [f32]) -> Result<(), LlmError> {
        match self {
            Self::F32 { data, d_model } => {
                let start = token_id
                    .checked_mul(*d_model)
                    .ok_or_else(|| LlmError::Inference("embedding offset overflow".to_owned()))?;
                let end = start
                    .checked_add(*d_model)
                    .ok_or_else(|| LlmError::Inference("embedding end overflow".to_owned()))?;
                let slice = data.get(start..end).ok_or_else(|| {
                    LlmError::Inference(format!("token id {token_id} out of embedding range"))
                })?;
                out.copy_from_slice(slice);
                Ok(())
            }
            Self::I8 {
                data,
                scales,
                d_model,
            } => {
                let scale = *scales.get(token_id).ok_or_else(|| {
                    LlmError::Inference(format!("token id {token_id} out of embedding range"))
                })?;
                let start = token_id
                    .checked_mul(*d_model)
                    .ok_or_else(|| LlmError::Inference("embedding offset overflow".to_owned()))?;
                let end = start
                    .checked_add(*d_model)
                    .ok_or_else(|| LlmError::Inference("embedding end overflow".to_owned()))?;
                let slice = data.get(start..end).ok_or_else(|| {
                    LlmError::Inference(format!("token id {token_id} out of embedding range"))
                })?;
                for (dst, &q) in out.iter_mut().zip(slice.iter()) {
                    *dst = f32::from(q) * scale;
                }
                Ok(())
            }
        }
    }

    const fn bytes(&self) -> usize {
        match self {
            Self::F32 { data, .. } => data.len() * std::mem::size_of::<f32>(),
            Self::I8 { data, scales, .. } => data.len() + scales.len() * std::mem::size_of::<f32>(),
        }
    }
}

/// Matriz W8 armazenada por **colunas** (`[in_dim, out_dim]`) para pular fisicamente
/// toda coluna zerada pelo `ReLU²` no Channel-Mixing do RWKV-7 (sem tocar na RAM!).
#[derive(Debug, Clone)]
pub struct SparseColQuantizedMatrix {
    in_dim: usize,
    out_dim: usize,
    cols_q: Box<[i8]>,
    col_scales: Box<[f32]>,
}

impl SparseColQuantizedMatrix {
    /// Constrói o layout column-major W8 a partir de um tensor row-major F32 `[out_dim, in_dim]`.
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::InvalidConfig`] se o tamanho do buffer não corresponder a `out_dim * in_dim`.
    pub fn from_row_major_f32(
        row_major: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self, LlmError> {
        let expected = out_dim
            .checked_mul(in_dim)
            .ok_or(LlmError::InvalidConfig("matrix dimensions overflow"))?;
        if row_major.len() != expected {
            return Err(LlmError::InvalidConfig(
                "row-major matrix buffer length mismatch",
            ));
        }

        let mut cols_q = vec![0_i8; expected];
        let mut col_scales = vec![0.0_f32; in_dim];

        for col in 0..in_dim {
            let mut max_abs = 0.0_f32;
            for row in 0..out_dim {
                max_abs = max_abs.max(row_major[row * in_dim + col].abs());
            }
            let scale = max_abs / 127.0;
            col_scales[col] = scale;
            if scale > 0.0 {
                let inv = 1.0 / scale;
                let dst = &mut cols_q[col * out_dim..(col + 1) * out_dim];
                for row in 0..out_dim {
                    dst[row] = (row_major[row * in_dim + col] * inv)
                        .round()
                        .clamp(-127.0, 127.0) as i8;
                }
            }
        }

        Ok(Self {
            in_dim,
            out_dim,
            cols_q: cols_q.into_boxed_slice(),
            col_scales: col_scales.into_boxed_slice(),
        })
    }

    pub(crate) fn load_from_safetensors(
        st: &SafeTensors<'_>,
        name: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self, LlmError> {
        let q_name = format!("{name}.q");
        let s_name = format!("{name}.scale");
        if let (Ok(q_view), Ok(s_view)) = (st.tensor(&q_name), st.tensor(&s_name)) {
            if q_view.dtype() != Dtype::I8 || q_view.shape() != [out_dim, in_dim] {
                return Err(LlmError::ModelLoad(format!(
                    "invalid '{q_name}' dtype {:?} or shape {:?}",
                    q_view.dtype(),
                    q_view.shape()
                )));
            }
            if s_view.dtype() != Dtype::F32 || s_view.shape() != [out_dim] {
                return Err(LlmError::ModelLoad(format!(
                    "invalid '{s_name}' dtype {:?} or shape {:?}",
                    s_view.dtype(),
                    s_view.shape()
                )));
            }
            let mut row_scales = Vec::with_capacity(out_dim);
            for chunk in s_view.data().chunks_exact(4) {
                row_scales.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let q_bytes = q_view.data();
            let mut dense = vec![0.0_f32; out_dim * in_dim];
            for row in 0..out_dim {
                let sc = row_scales[row];
                let base = row * in_dim;
                for col in 0..in_dim {
                    let q = i8::from_ne_bytes([q_bytes[base + col]]);
                    dense[base + col] = f32::from(q) * sc;
                }
            }
            return Self::from_row_major_f32(&dense, out_dim, in_dim);
        }

        let dense = read_f32_boxed(st, name, &[out_dim, in_dim])?;
        Self::from_row_major_f32(&dense, out_dim, in_dim)
    }

    /// Acumula `out += W * act` lendo da RAM **apenas** colunas com `act[col] > 0.0`.
    /// Retorna quantas colunas ativas foram lidas.
    pub fn accumulate_sparse(&self, act: &[f32], out: &mut [f32]) -> usize {
        self.accumulate_sparse_scaled(act, 1.0, out)
    }

    /// Acumula `out += (scale * W) * act` fundindo o peso do roteador MoE no fator de coluna,
    /// sem tocar na RAM para colunas zeradas (`act[col] <= 0.0`).
    pub fn accumulate_sparse_scaled(&self, act: &[f32], scale: f32, out: &mut [f32]) -> usize {
        debug_assert_eq!(act.len(), self.in_dim);
        debug_assert_eq!(out.len(), self.out_dim);

        if scale == 0.0 || !scale.is_finite() {
            return 0;
        }

        let mut active_cols: usize = 0;
        let out_dim = self.out_dim;

        for (col, &a) in act.iter().enumerate() {
            if a <= 0.0 {
                continue;
            }
            let factor = a * scale * self.col_scales[col];
            if factor == 0.0 {
                continue;
            }
            active_cols += 1;
            let col_slice = &self.cols_q[col * out_dim..(col + 1) * out_dim];
            for (dst, &q) in out.iter_mut().zip(col_slice.iter()) {
                *dst += f32::from(q) * factor;
            }
        }

        active_cols
    }

    #[must_use]
    pub const fn in_dim(&self) -> usize {
        self.in_dim
    }

    #[must_use]
    pub const fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Bytes físicos lidos da RAM por coluna ativa (`out_dim` bytes W8 + 1 escala `f32`).
    #[must_use]
    pub const fn col_stride_bytes(&self) -> usize {
        self.out_dim + std::mem::size_of::<f32>()
    }

    #[must_use]
    pub const fn weight_bytes(&self) -> usize {
        self.cols_q.len() + self.col_scales.len() * std::mem::size_of::<f32>()
    }
}

#[derive(Debug, Clone)]
struct ValueResidualWeights {
    v0: Box<[f32]>,
    v1: QuantizedMatrix,
    v2: QuantizedMatrix,
    lora_dim: usize,
}

#[derive(Debug, Clone)]
struct Rwkv7TimeMixingWeights {
    x_r: Box<[f32]>,
    x_w: Box<[f32]>,
    x_k: Box<[f32]>,
    x_v: Box<[f32]>,
    x_a: Box<[f32]>,
    x_g: Box<[f32]>,
    receptance: QuantizedMatrix,
    key: QuantizedMatrix,
    value: QuantizedMatrix,
    output: QuantizedMatrix,
    w0: Box<[f32]>,
    w1: QuantizedMatrix,
    w2: QuantizedMatrix,
    a0: Box<[f32]>,
    a1: QuantizedMatrix,
    a2: QuantizedMatrix,
    v_res: Option<ValueResidualWeights>,
    g1: QuantizedMatrix,
    g2: QuantizedMatrix,
    k_k: Box<[f32]>,
    k_a: Box<[f32]>,
    r_k: Box<[f32]>,
    ln_x: LayerNormWeights,
}

impl Rwkv7TimeMixingWeights {
    fn weight_bytes(&self) -> usize {
        let v_bytes = self
            .v_res
            .as_ref()
            .map_or(0, |vr| vr.v1.weight_bytes() + vr.v2.weight_bytes());
        self.receptance.weight_bytes()
            + self.key.weight_bytes()
            + self.value.weight_bytes()
            + self.output.weight_bytes()
            + self.w1.weight_bytes()
            + self.w2.weight_bytes()
            + self.a1.weight_bytes()
            + self.a2.weight_bytes()
            + self.g1.weight_bytes()
            + self.g2.weight_bytes()
            + v_bytes
    }
}

#[derive(Debug, Clone)]
enum ChannelMixingFfn {
    Dense {
        key: QuantizedMatrix,
        value: SparseColQuantizedMatrix,
    },
    SparseMoe(SparseMoeFfn),
}

#[derive(Debug, Clone)]
struct Rwkv7ChannelMixingWeights {
    x_k: Box<[f32]>,
    ffn: ChannelMixingFfn,
}

impl Rwkv7ChannelMixingWeights {
    const fn weight_bytes(&self) -> usize {
        match &self.ffn {
            ChannelMixingFfn::Dense { key, value } => key.weight_bytes() + value.weight_bytes(),
            ChannelMixingFfn::SparseMoe(moe) => moe.weight_bytes(),
        }
    }
}

#[derive(Debug, Clone)]
struct Rwkv7BlockWeights {
    ln1: LayerNormWeights,
    tm: Rwkv7TimeMixingWeights,
    ln2: LayerNormWeights,
    cm: Rwkv7ChannelMixingWeights,
}

impl Rwkv7BlockWeights {
    fn weight_bytes(&self) -> usize {
        self.tm.weight_bytes() + self.cm.weight_bytes()
    }
}

#[derive(Debug, Clone)]
struct CopyGateWeights {
    copy_dim: usize,
    inv_sqrt_dim: f32,
    query: QuantizedMatrix,
    key: QuantizedMatrix,
    gate_weight: Box<[f32]>,
    gate_bias: f32,
}

impl CopyGateWeights {
    fn load(st: &SafeTensors<'_>, d_model: usize, copy_dim: usize) -> Result<Self, LlmError> {
        let query = QuantizedMatrix::load_from_safetensors(st, "copy_q.weight", copy_dim, d_model)?;
        let key = QuantizedMatrix::load_from_safetensors(st, "copy_k.weight", copy_dim, d_model)?;
        let gate_in = d_model
            .checked_add(copy_dim)
            .ok_or_else(|| LlmError::ModelLoad("copy_gate input dimension overflow".to_owned()))?;
        let gate_weight = read_f32_boxed(st, "copy_gate.weight", &[1, gate_in])?;
        let gate_bias_slice = read_f32_boxed(st, "copy_gate.bias", &[1])?;
        let gate_bias = gate_bias_slice.first().copied().unwrap_or(0.0);

        Ok(Self {
            copy_dim,
            inv_sqrt_dim: 1.0 / (copy_dim as f32).sqrt(),
            query,
            key,
            gate_weight,
            gate_bias,
        })
    }

    const fn weight_bytes(&self) -> usize {
        self.query.weight_bytes()
            + self.key.weight_bytes()
            + (self.gate_weight.len() + 1) * std::mem::size_of::<f32>()
    }
}

#[derive(Debug, Clone)]
struct Rwkv7LayerState {
    prev_x_tm: Box<[f32]>,
    prev_x_cm: Box<[f32]>,
    state_mat: Box<[f32]>,
}

impl Rwkv7LayerState {
    fn new(d_model: usize, n_heads: usize, head_dim: usize) -> Self {
        Self {
            prev_x_tm: vec![0.0; d_model].into_boxed_slice(),
            prev_x_cm: vec![0.0; d_model].into_boxed_slice(),
            state_mat: vec![0.0; n_heads * head_dim * head_dim].into_boxed_slice(),
        }
    }
}

#[derive(Debug, Clone)]
struct Rwkv7Scratch {
    x_emb: Box<[f32]>,
    x: Box<[f32]>,
    x_ln: Box<[f32]>,
    x_out: Box<[f32]>,
    v_first: Box<[f32]>,
    dx: Box<[f32]>,
    x_shift: Box<[f32]>,
    r: Box<[f32]>,
    k: Box<[f32]>,
    v: Box<[f32]>,
    w_logit: Box<[f32]>,
    a_vec: Box<[f32]>,
    g_vec: Box<[f32]>,
    lora_buf: Box<[f32]>,
    kk_head: Box<[f32]>,
    k_hat_head: Box<[f32]>,
    w_decay_head: Box<[f32]>,
    ak_head: Box<[f32]>,
    y_head: Box<[f32]>,
    rwkv: Box<[f32]>,
    tm_out: Box<[f32]>,
    cm_k: Box<[f32]>,
    q_input: Box<[i8]>,
    allowed_tokens: Vec<usize>,
    // Buffers do Pointer/Copy Gate
    copy_q: Box<[f32]>,
    copy_ctx: Box<[f32]>,
    copy_keys: Vec<f32>,
    copy_tokens: Vec<TokenId>,
    copy_attn: Vec<f32>,
    vocab_scores: Vec<f32>,
    copy_vocab_acc: Box<[f32]>,
    moe: Option<SparseMoeScratch>,
    chunk4: Option<Box<Rwkv7Chunk4Scratch>>,
}

impl Rwkv7Scratch {
    fn new(cfg: &Rwkv7Config) -> Self {
        let d = cfg.d_model;
        let max_lora = cfg
            .decay_lora_dim
            .max(cfg.a_lora_dim)
            .max(cfg.v_lora_dim)
            .max(cfg.gate_lora_dim)
            .max(1);
        let max_in = d.max(cfg.ffn_dim).max(max_lora);
        let hd = cfg.head_dim;

        let copy_dim = cfg.copy_dim;
        let copy_cap = if copy_dim > 0 { cfg.copy_max_len } else { 0 };
        let copy_acc_len = if copy_dim > 0 { cfg.vocab_size } else { 0 };

        let moe = if cfg.moe_num_experts > 0 {
            Some(SparseMoeScratch::new(cfg.moe_num_experts, cfg.ffn_dim))
        } else {
            None
        };

        Self {
            x_emb: vec![0.0; d].into_boxed_slice(),
            x: vec![0.0; d].into_boxed_slice(),
            x_ln: vec![0.0; d].into_boxed_slice(),
            x_out: vec![0.0; d].into_boxed_slice(),
            v_first: vec![0.0; d].into_boxed_slice(),
            dx: vec![0.0; d].into_boxed_slice(),
            x_shift: vec![0.0; d].into_boxed_slice(),
            r: vec![0.0; d].into_boxed_slice(),
            k: vec![0.0; d].into_boxed_slice(),
            v: vec![0.0; d].into_boxed_slice(),
            w_logit: vec![0.0; d].into_boxed_slice(),
            a_vec: vec![0.0; d].into_boxed_slice(),
            g_vec: vec![0.0; d].into_boxed_slice(),
            lora_buf: vec![0.0; max_lora].into_boxed_slice(),
            kk_head: vec![0.0; hd].into_boxed_slice(),
            k_hat_head: vec![0.0; hd].into_boxed_slice(),
            w_decay_head: vec![0.0; hd].into_boxed_slice(),
            ak_head: vec![0.0; hd].into_boxed_slice(),
            y_head: vec![0.0; hd].into_boxed_slice(),
            rwkv: vec![0.0; d].into_boxed_slice(),
            tm_out: vec![0.0; d].into_boxed_slice(),
            cm_k: vec![0.0; cfg.ffn_dim].into_boxed_slice(),
            q_input: vec![0_i8; max_in].into_boxed_slice(),
            allowed_tokens: Vec::with_capacity(cfg.vocab_size.min(512)),
            copy_q: vec![0.0; copy_dim].into_boxed_slice(),
            copy_ctx: vec![0.0; copy_dim].into_boxed_slice(),
            copy_keys: Vec::with_capacity(copy_cap.saturating_mul(copy_dim)),
            copy_tokens: Vec::with_capacity(copy_cap),
            copy_attn: Vec::with_capacity(copy_cap),
            vocab_scores: Vec::with_capacity(cfg.vocab_size.min(512)),
            copy_vocab_acc: vec![0.0; copy_acc_len].into_boxed_slice(),
            moe,
            chunk4: None,
        }
    }
}

/// Buffers pré-alocados sob demanda para processar 4 tokens simultâneos (`matmul4_into`)
/// com zero alocações no heap por bloco de 4 tokens.
#[derive(Debug, Clone)]
struct Rwkv7Chunk4Scratch {
    x_emb: [Box<[f32]>; 4],
    x: [Box<[f32]>; 4],
    x_ln: [Box<[f32]>; 4],
    v_first: [Box<[f32]>; 4],
    dx: [Box<[f32]>; 4],
    x_shift: [Box<[f32]>; 4],
    r: [Box<[f32]>; 4],
    k: [Box<[f32]>; 4],
    v: [Box<[f32]>; 4],
    w_logit: [Box<[f32]>; 4],
    a_vec: [Box<[f32]>; 4],
    g_vec: [Box<[f32]>; 4],
    lora_buf: [Box<[f32]>; 4],
    rwkv: [Box<[f32]>; 4],
    tm_out: [Box<[f32]>; 4],
    cm_k: [Box<[f32]>; 4],
    q_input: [Box<[i8]>; 4],
}

impl Rwkv7Chunk4Scratch {
    fn new(cfg: &Rwkv7Config) -> Self {
        let d = cfg.d_model;
        let max_lora = cfg
            .decay_lora_dim
            .max(cfg.a_lora_dim)
            .max(cfg.v_lora_dim)
            .max(cfg.gate_lora_dim)
            .max(1);
        let max_in = d.max(cfg.ffn_dim).max(max_lora);

        let f32_x4 = |len: usize| std::array::from_fn(|_| vec![0.0_f32; len].into_boxed_slice());
        let i8_x4 = |len: usize| std::array::from_fn(|_| vec![0_i8; len].into_boxed_slice());

        Self {
            x_emb: f32_x4(d),
            x: f32_x4(d),
            x_ln: f32_x4(d),
            v_first: f32_x4(d),
            dx: f32_x4(d),
            x_shift: f32_x4(d),
            r: f32_x4(d),
            k: f32_x4(d),
            v: f32_x4(d),
            w_logit: f32_x4(d),
            a_vec: f32_x4(d),
            g_vec: f32_x4(d),
            lora_buf: f32_x4(max_lora),
            rwkv: f32_x4(d),
            tm_out: f32_x4(d),
            cm_k: f32_x4(cfg.ffn_dim),
            q_input: i8_x4(max_in),
        }
    }
}

/// Métricas observáveis do reuso de estado recorrente O(1) entre turnos do [`Rwkv7Llm`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub tokens_reused: u64,
    pub tokens_prefilled: u64,
}

#[derive(Debug, Default)]
struct PrefixStateCache {
    tokens: Vec<TokenId>,
    layers_snapshot: Option<Box<[Rwkv7LayerState]>>,
    h_prefill: Vec<f32>,
    token_byte_ends: Vec<usize>,
    checkpoint_x_out: Vec<f32>,
    copy_start_idx: usize,
    copy_keys: Vec<f32>,
    copy_tokens: Vec<TokenId>,
    working_state: Option<Rwkv7State>,
    stats: PrefixCacheStats,
}

fn copy_layers(dst: &mut [Rwkv7LayerState], src: &[Rwkv7LayerState]) {
    for (d_layer, s_layer) in dst.iter_mut().zip(src.iter()) {
        d_layer.prev_x_tm.copy_from_slice(&s_layer.prev_x_tm);
        d_layer.prev_x_cm.copy_from_slice(&s_layer.prev_x_cm);
        d_layer.state_mat.copy_from_slice(&s_layer.state_mat);
    }
}

fn common_prefix_len(a: &[TokenId], b: &[TokenId]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Estado recorrente estritamente O(1) do RWKV-v7 (zero KV-Cache, zero alocações no decode).
#[derive(Debug, Clone)]
pub struct Rwkv7State {
    layers: Box<[Rwkv7LayerState]>,
    scratch: Rwkv7Scratch,
    pub total_ffn_cols: u64,
    pub active_ffn_cols: u64,
    pub total_ffn_bytes: u64,
    pub read_ffn_bytes: u64,
}

impl Rwkv7State {
    #[must_use]
    pub fn new(cfg: &Rwkv7Config) -> Self {
        let n_heads = cfg.d_model / cfg.head_dim.max(1);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            layers.push(Rwkv7LayerState::new(cfg.d_model, n_heads, cfg.head_dim));
        }
        Self {
            layers: layers.into_boxed_slice(),
            scratch: Rwkv7Scratch::new(cfg),
            total_ffn_cols: 0,
            active_ffn_cols: 0,
            total_ffn_bytes: 0,
            read_ffn_bytes: 0,
        }
    }

    /// Bytes fixos ocupados pelos estados recorrentes das camadas em RAM.
    #[must_use]
    pub fn state_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                (l.prev_x_tm.len() + l.prev_x_cm.len() + l.state_mat.len())
                    .saturating_mul(std::mem::size_of::<f32>())
            })
            .sum()
    }

    /// Fração de colunas da FFN fisicamente puladas por estarem zeradas pelo `ReLU²` (0.0 .. 1.0).
    #[must_use]
    pub fn ffn_sparsity_ratio(&self) -> f64 {
        if self.total_ffn_cols == 0 {
            return 0.0;
        }
        1.0 - (self.active_ffn_cols as f64) / (self.total_ffn_cols as f64)
    }

    /// Quantidade de entradas registradas atualmente no banco de cópia do prompt.
    #[must_use]
    pub fn copy_bank_len(&self) -> usize {
        self.scratch.copy_tokens.len()
    }

    /// Fração física de bytes da FFN que não precisaram ser lidos da RAM (0.0 .. 1.0).
    #[must_use]
    pub fn ffn_ram_skip_ratio(&self) -> f64 {
        if self.total_ffn_bytes == 0 {
            return 0.0;
        }
        1.0 - (self.read_ffn_bytes as f64) / (self.total_ffn_bytes as f64)
    }

    /// Zera apenas os estados recorrentes e contadores, preservando as alocações de `layers` e `scratch`.
    pub fn reset_recurrent_zero(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.prev_x_tm.fill(0.0);
            layer.prev_x_cm.fill(0.0);
            layer.state_mat.fill(0.0);
        }
        self.scratch.copy_keys.clear();
        self.scratch.copy_tokens.clear();
        self.total_ffn_cols = 0;
        self.active_ffn_cols = 0;
        self.total_ffn_bytes = 0;
        self.read_ffn_bytes = 0;
    }
}

/// Motor nativo RWKV-v7 ("Goose") em Rust puro com W8A8, pulo de zeros `ReLU²`,
/// Pointer/Copy Gate e constrained decoding via `llguidance` + `TokTrie` (RWKV World 65k).
pub struct Rwkv7Llm {
    cfg: Rwkv7Config,
    n_heads: usize,
    tok_trie: TokTrie,
    parser_factory: ParserFactory,
    grammar: TopLevelGrammar,
    emb: EmbeddingTable,
    ln_in: LayerNormWeights,
    blocks: Box<[Rwkv7BlockWeights]>,
    ln_out: LayerNormWeights,
    head: QuantizedMatrix,
    copy_gate: Option<CopyGateWeights>,
    decode_mode: DecodeMode,
    isa_head: Option<IsaHeadWeights>,
    max_new_tokens: usize,
    prefix_cache: Mutex<PrefixStateCache>,
}

impl std::fmt::Debug for Rwkv7Llm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rwkv7Llm")
            .field("cfg", &self.cfg)
            .field("copy_gate_enabled", &self.copy_gate.is_some())
            .field("total_weight_bytes", &self.total_weight_bytes())
            .finish_non_exhaustive()
    }
}

impl Rwkv7Llm {
    /// Carrega `config.json`, `vocab.bin` e `model.safetensors` gerados por `finetune_rwkv7_g1.py`.
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError`] se algum arquivo estiver ausente, inválido ou se a gramática falhar.
    pub fn load_dir(dir: impl AsRef<Path>) -> Result<Self, LlmError> {
        let dir_ref = dir.as_ref();
        let cfg_text = std::fs::read_to_string(dir_ref.join("config.json"))
            .map_err(|err| LlmError::ModelLoad(format!("failed to read config.json: {err}")))?;
        let cfg: Rwkv7Config = serde_json::from_str(&cfg_text)?;

        let vocab_bytes = std::fs::read(dir_ref.join(&cfg.vocab_file)).map_err(|err| {
            LlmError::ModelLoad(format!("failed to read '{}': {err}", cfg.vocab_file))
        })?;
        let weights_bytes = std::fs::read(dir_ref.join("model.safetensors")).map_err(|err| {
            LlmError::ModelLoad(format!("failed to read model.safetensors: {err}"))
        })?;

        Self::from_parts(cfg, &vocab_bytes, &weights_bytes)
    }

    /// Constrói o motor completo em RAM a partir de `config`, `vocab.bin` e `model.safetensors`.
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError`] se dimensões, vocabulário, tensores ou gramática `llguidance` forem inválidos.
    pub fn from_parts(
        cfg: Rwkv7Config,
        vocab_bin: &[u8],
        safetensors_bytes: &[u8],
    ) -> Result<Self, LlmError> {
        if cfg.head_dim == 0 || !cfg.d_model.is_multiple_of(cfg.head_dim) {
            return Err(LlmError::InvalidConfig(
                "d_model must be divisible by positive head_dim",
            ));
        }
        let n_heads = cfg.d_model / cfg.head_dim;

        let words = parse_vocab_bin(vocab_bin, cfg.vocab_size)?;
        let vocab_u32 = u32::try_from(cfg.vocab_size)
            .map_err(|_| LlmError::InvalidConfig("vocab_size exceeds u32"))?;
        let eos_u32 = TokenId::try_from(cfg.eos_token_id)
            .map_err(|_| LlmError::InvalidConfig("eos_token_id exceeds u32"))?;

        let rx_info = TokRxInfo::new(vocab_u32, eos_u32);
        let tok_trie = TokTrie::from(&rx_info, &words);
        let tok_env: TokEnv = Arc::new(ApproximateTokEnv::new(tok_trie.clone()));

        let parser_factory = ParserFactory::new_simple(&tok_env)
            .map_err(|err| LlmError::GrammarInit(err.to_string()))?;
        let schema_value = action_schema_value()?;
        let grammar = TopLevelGrammar::from_json_schema(schema_value);

        let probe = Matcher::new(parser_factory.create_parser(grammar.clone()));
        if let Some(err) = probe.get_error() {
            return Err(LlmError::GrammarInit(err));
        }

        let st = SafeTensors::deserialize(safetensors_bytes)
            .map_err(|err| LlmError::ModelLoad(format!("invalid safetensors buffer: {err}")))?;

        let d = cfg.d_model;
        let ffn = cfg.ffn_dim;
        let v = cfg.vocab_size;

        let emb = EmbeddingTable::load(&st, v, d)?;
        let ln_in = read_ln(&st, "ln_in", d)?;

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let bp = format!("blocks.{i}");
            let ln1 = read_ln(&st, &format!("{bp}.ln1"), d)?;

            let v_res = if i > 0 {
                Some(ValueResidualWeights {
                    v0: read_f32_boxed(&st, &format!("{bp}.tm.v0"), &[d])?,
                    v1: QuantizedMatrix::load_from_safetensors(
                        &st,
                        &format!("{bp}.tm.v1.weight"),
                        cfg.v_lora_dim,
                        d,
                    )?,
                    v2: QuantizedMatrix::load_from_safetensors(
                        &st,
                        &format!("{bp}.tm.v2.weight"),
                        d,
                        cfg.v_lora_dim,
                    )?,
                    lora_dim: cfg.v_lora_dim,
                })
            } else {
                None
            };

            let tm = Rwkv7TimeMixingWeights {
                x_r: read_f32_boxed(&st, &format!("{bp}.tm.x_r"), &[d])?,
                x_w: read_f32_boxed(&st, &format!("{bp}.tm.x_w"), &[d])?,
                x_k: read_f32_boxed(&st, &format!("{bp}.tm.x_k"), &[d])?,
                x_v: read_f32_boxed(&st, &format!("{bp}.tm.x_v"), &[d])?,
                x_a: read_f32_boxed(&st, &format!("{bp}.tm.x_a"), &[d])?,
                x_g: read_f32_boxed(&st, &format!("{bp}.tm.x_g"), &[d])?,
                receptance: QuantizedMatrix::load_from_safetensors(
                    &st,
                    &format!("{bp}.tm.receptance.weight"),
                    d,
                    d,
                )?,
                key: QuantizedMatrix::load_from_safetensors(
                    &st,
                    &format!("{bp}.tm.key.weight"),
                    d,
                    d,
                )?,
                value: QuantizedMatrix::load_from_safetensors(
                    &st,
                    &format!("{bp}.tm.value.weight"),
                    d,
                    d,
                )?,
                output: QuantizedMatrix::load_from_safetensors(
                    &st,
                    &format!("{bp}.tm.output.weight"),
                    d,
                    d,
                )?,
                w0: read_f32_boxed(&st, &format!("{bp}.tm.w0"), &[d])?,
                w1: QuantizedMatrix::load_from_safetensors(
                    &st,
                    &format!("{bp}.tm.w1.weight"),
                    cfg.decay_lora_dim,
                    d,
                )?,
                w2: QuantizedMatrix::load_from_safetensors(
                    &st,
                    &format!("{bp}.tm.w2.weight"),
                    d,
                    cfg.decay_lora_dim,
                )?,
                a0: read_f32_boxed(&st, &format!("{bp}.tm.a0"), &[d])?,
                a1: QuantizedMatrix::load_from_safetensors(
                    &st,
                    &format!("{bp}.tm.a1.weight"),
                    cfg.a_lora_dim,
                    d,
                )?,
                a2: QuantizedMatrix::load_from_safetensors(
                    &st,
                    &format!("{bp}.tm.a2.weight"),
                    d,
                    cfg.a_lora_dim,
                )?,
                v_res,
                g1: QuantizedMatrix::load_from_safetensors(
                    &st,
                    &format!("{bp}.tm.g1.weight"),
                    cfg.gate_lora_dim,
                    d,
                )?,
                g2: QuantizedMatrix::load_from_safetensors(
                    &st,
                    &format!("{bp}.tm.g2.weight"),
                    d,
                    cfg.gate_lora_dim,
                )?,
                k_k: read_f32_boxed(&st, &format!("{bp}.tm.k_k"), &[d])?,
                k_a: read_f32_boxed(&st, &format!("{bp}.tm.k_a"), &[d])?,
                r_k: read_f32_boxed(&st, &format!("{bp}.tm.r_k"), &[d])?,
                ln_x: read_ln(&st, &format!("{bp}.tm.ln_x"), d)?,
            };

            let ln2 = read_ln(&st, &format!("{bp}.ln2"), d)?;
            let ffn_backend = if cfg.moe_num_experts > 0 {
                ChannelMixingFfn::SparseMoe(SparseMoeFfn::load_from_safetensors(
                    &st,
                    &format!("{bp}.cm"),
                    d,
                    ffn,
                    cfg.moe_num_experts,
                    cfg.moe_top_k,
                    cfg.moe_neuron_top_k,
                )?)
            } else {
                ChannelMixingFfn::Dense {
                    key: QuantizedMatrix::load_from_safetensors(
                        &st,
                        &format!("{bp}.cm.key.weight"),
                        ffn,
                        d,
                    )?,
                    value: SparseColQuantizedMatrix::load_from_safetensors(
                        &st,
                        &format!("{bp}.cm.value.weight"),
                        d,
                        ffn,
                    )?,
                }
            };
            let cm = Rwkv7ChannelMixingWeights {
                x_k: read_f32_boxed(&st, &format!("{bp}.cm.x_k"), &[d])?,
                ffn: ffn_backend,
            };

            blocks.push(Rwkv7BlockWeights { ln1, tm, ln2, cm });
        }

        let ln_out = read_ln(&st, "ln_out", d)?;
        let head = QuantizedMatrix::load_from_safetensors(&st, "head.weight", v, d)?;
        let copy_gate = if cfg.copy_dim > 0 {
            Some(CopyGateWeights::load(&st, d, cfg.copy_dim)?)
        } else {
            None
        };
        let isa_head = if cfg.isa_ptr_dim > 0 {
            Some(IsaHeadWeights::load(
                &st,
                d,
                cfg.isa_ptr_dim,
                cfg.isa_num_vocab.max(1),
            )?)
        } else {
            None
        };

        let model = Self {
            cfg,
            n_heads,
            tok_trie,
            parser_factory,
            grammar,
            emb,
            ln_in,
            blocks: blocks.into_boxed_slice(),
            ln_out,
            head,
            copy_gate,
            decode_mode: DecodeMode::Auto,
            isa_head,
            max_new_tokens: 256,
            prefix_cache: Mutex::new(PrefixStateCache::default()),
        };

        tracing::info!(
            vocab = model.cfg.vocab_size,
            d_model = model.cfg.d_model,
            n_layers = model.cfg.n_layers,
            copy_dim = model.cfg.copy_dim,
            total_mib = (model.total_weight_bytes() as f64) / (1024.0 * 1024.0),
            "native rwkv-7 goose model loaded"
        );

        Ok(model)
    }

    #[must_use]
    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens.max(1);
        self
    }

    #[must_use]
    pub fn config(&self) -> &Rwkv7Config {
        &self.cfg
    }

    /// Define explicitamente o modo de decodificação (`Auto`, `Json` ou `Isa`).
    #[must_use]
    pub const fn with_decode_mode(mut self, mode: DecodeMode) -> Self {
        self.decode_mode = mode;
        self
    }

    #[must_use]
    pub fn total_weight_bytes(&self) -> usize {
        let blocks_bytes: usize = self
            .blocks
            .iter()
            .map(Rwkv7BlockWeights::weight_bytes)
            .sum();
        let copy_bytes = self
            .copy_gate
            .as_ref()
            .map_or(0, CopyGateWeights::weight_bytes);
        let isa_bytes = self
            .isa_head
            .as_ref()
            .map_or(0, IsaHeadWeights::weight_bytes);

        self.emb
            .bytes()
            .saturating_add(blocks_bytes)
            .saturating_add(self.head.weight_bytes())
            .saturating_add(copy_bytes)
            .saturating_add(isa_bytes)
    }

    /// Avança 1 token do vocabulário atualizando o estado matricial O(1) e `state.scratch.x_out`
    /// com zero alocações no heap e pulo de colunas zeradas na FFN (`ReLU²`).
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::Inference`] se `token_id` estiver fora do vocabulário.
    fn forward_hidden(&self, token_id: usize, state: &mut Rwkv7State) -> Result<(), LlmError> {
        self.emb.lookup_into(token_id, &mut state.scratch.x_emb)?;
        self.forward_from_x_emb(state)
    }

    /// Avança 1 passo a partir de um vetor de embedding contínuo arbitrário (`len == d_model`).
    /// Usado na decodificação ISA para injetar `op_emb`, `num_emb`, `ctl_emb` e a realimentação
    /// semântica do ponteiro de entidade (`ptr_tag + W_ptr_in * K_s`) direto no estado O(1).
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::InvalidConfig`] se `input_emb.len() != config().d_model`.
    pub fn forward_hidden_vec(
        &self,
        input_emb: &[f32],
        state: &mut Rwkv7State,
    ) -> Result<(), LlmError> {
        if input_emb.len() != self.cfg.d_model {
            return Err(LlmError::InvalidConfig(
                "input_emb length must match d_model",
            ));
        }
        state.scratch.x_emb.copy_from_slice(input_emb);
        self.forward_from_x_emb(state)
    }

    /// Núcleo compartilhado das `n_layers` camadas RWKV-7: lê `state.scratch.x_emb` e grava `state.scratch.x_out`.
    fn forward_from_x_emb(&self, state: &mut Rwkv7State) -> Result<(), LlmError> {
        let d = self.cfg.d_model;
        let eps = self.cfg.layer_norm_eps;
        let gn_eps = self.cfg.group_norm_eps;
        let hd = self.cfg.head_dim;

        let scratch = &mut state.scratch;
        layer_norm_into(&scratch.x_emb, &self.ln_in, eps, &mut scratch.x);

        for (layer_idx, (block, layer_state)) in
            self.blocks.iter().zip(state.layers.iter_mut()).enumerate()
        {
            // 1. Time-Mixing RWKV-7 (Regra Delta Generalizada)
            layer_norm_into(&scratch.x, &block.ln1, eps, &mut scratch.x_ln);
            for c in 0..d {
                scratch.dx[c] = layer_state.prev_x_tm[c] - scratch.x_ln[c];
            }
            layer_state.prev_x_tm.copy_from_slice(&scratch.x_ln);

            let tm = &block.tm;
            shift_into(&scratch.x_ln, &scratch.dx, &tm.x_r, &mut scratch.x_shift);
            tm.receptance
                .matvec_into(&scratch.x_shift, &mut scratch.q_input, &mut scratch.r);

            shift_into(&scratch.x_ln, &scratch.dx, &tm.x_k, &mut scratch.x_shift);
            tm.key
                .matvec_into(&scratch.x_shift, &mut scratch.q_input, &mut scratch.k);

            shift_into(&scratch.x_ln, &scratch.dx, &tm.x_v, &mut scratch.x_shift);
            tm.value
                .matvec_into(&scratch.x_shift, &mut scratch.q_input, &mut scratch.v);

            if layer_idx == 0 {
                scratch.v_first.copy_from_slice(&scratch.v);
            } else if let Some(ref vr) = tm.v_res {
                let ldim = vr.lora_dim;
                vr.v1.matvec_into(
                    &scratch.x_shift,
                    &mut scratch.q_input,
                    &mut scratch.lora_buf[..ldim],
                );
                vr.v2.matvec_into(
                    &scratch.lora_buf[..ldim],
                    &mut scratch.q_input,
                    &mut scratch.tm_out,
                );
                for c in 0..d {
                    let gate = sigmoid(vr.v0[c] + scratch.tm_out[c]);
                    scratch.v[c] += (scratch.v_first[c] - scratch.v[c]) * gate;
                }
            }

            shift_into(&scratch.x_ln, &scratch.dx, &tm.x_w, &mut scratch.x_shift);
            let w_ldim = self.cfg.decay_lora_dim;
            tm.w1.matvec_into(
                &scratch.x_shift,
                &mut scratch.q_input,
                &mut scratch.lora_buf[..w_ldim],
            );
            for val in &mut scratch.lora_buf[..w_ldim] {
                *val = val.tanh();
            }
            tm.w2.matvec_into(
                &scratch.lora_buf[..w_ldim],
                &mut scratch.q_input,
                &mut scratch.w_logit,
            );
            for c in 0..d {
                scratch.w_logit[c] += tm.w0[c];
            }

            shift_into(&scratch.x_ln, &scratch.dx, &tm.x_a, &mut scratch.x_shift);
            let a_ldim = self.cfg.a_lora_dim;
            tm.a1.matvec_into(
                &scratch.x_shift,
                &mut scratch.q_input,
                &mut scratch.lora_buf[..a_ldim],
            );
            tm.a2.matvec_into(
                &scratch.lora_buf[..a_ldim],
                &mut scratch.q_input,
                &mut scratch.a_vec,
            );
            for c in 0..d {
                scratch.a_vec[c] = sigmoid(tm.a0[c] + scratch.a_vec[c]);
            }

            shift_into(&scratch.x_ln, &scratch.dx, &tm.x_g, &mut scratch.x_shift);
            let g_ldim = self.cfg.gate_lora_dim;
            tm.g1.matvec_into(
                &scratch.x_shift,
                &mut scratch.q_input,
                &mut scratch.lora_buf[..g_ldim],
            );
            for val in &mut scratch.lora_buf[..g_ldim] {
                *val = sigmoid(*val);
            }
            tm.g2.matvec_into(
                &scratch.lora_buf[..g_ldim],
                &mut scratch.q_input,
                &mut scratch.g_vec,
            );

            for head in 0..self.n_heads {
                let off = head * hd;
                let mat_off = head * hd * hd;

                prepare_head_vectors(
                    &scratch.k[off..off + hd],
                    &tm.k_k[off..off + hd],
                    &tm.k_a[off..off + hd],
                    &scratch.a_vec[off..off + hd],
                    &scratch.w_logit[off..off + hd],
                    &mut scratch.kk_head,
                    &mut scratch.k_hat_head,
                    &mut scratch.w_decay_head,
                );

                wkv7_head_step(
                    &mut layer_state.state_mat[mat_off..mat_off + hd * hd],
                    &scratch.r[off..off + hd],
                    &scratch.w_decay_head,
                    &scratch.k_hat_head,
                    &scratch.v[off..off + hd],
                    &scratch.kk_head,
                    &scratch.a_vec[off..off + hd],
                    &mut scratch.ak_head,
                    &mut scratch.y_head,
                );

                group_norm_and_bonus(
                    &scratch.y_head,
                    &scratch.r[off..off + hd],
                    &scratch.k_hat_head,
                    &scratch.v[off..off + hd],
                    &tm.r_k[off..off + hd],
                    &tm.ln_x.weight[off..off + hd],
                    &tm.ln_x.bias[off..off + hd],
                    gn_eps,
                    &mut scratch.rwkv[off..off + hd],
                );
            }

            for c in 0..d {
                scratch.rwkv[c] *= scratch.g_vec[c];
            }
            tm.output
                .matvec_into(&scratch.rwkv, &mut scratch.q_input, &mut scratch.tm_out);
            for c in 0..d {
                scratch.x[c] += scratch.tm_out[c];
            }

            // 2. Channel-Mixing RWKV-7 (ReLU² com pulo físico de colunas zeradas)
            layer_norm_into(&scratch.x, &block.ln2, eps, &mut scratch.x_ln);
            for c in 0..d {
                scratch.x_shift[c] = scratch.x_ln[c]
                    + block.cm.x_k[c] * (layer_state.prev_x_cm[c] - scratch.x_ln[c]);
            }
            layer_state.prev_x_cm.copy_from_slice(&scratch.x_ln);

            match &block.cm.ffn {
                ChannelMixingFfn::Dense { key, value } => {
                    key.matvec_into(&scratch.x_shift, &mut scratch.q_input, &mut scratch.cm_k);
                    for val in scratch.cm_k.iter_mut() {
                        let r = val.max(0.0);
                        *val = r * r;
                    }
                    let active = value.accumulate_sparse(&scratch.cm_k, &mut scratch.x);
                    let total_bytes = key.weight_bytes() + value.weight_bytes();
                    let read_bytes = key.weight_bytes() + active * value.col_stride_bytes();
                    state.total_ffn_cols =
                        state.total_ffn_cols.saturating_add(self.cfg.ffn_dim as u64);
                    state.active_ffn_cols = state.active_ffn_cols.saturating_add(active as u64);
                    state.total_ffn_bytes =
                        state.total_ffn_bytes.saturating_add(total_bytes as u64);
                    state.read_ffn_bytes = state.read_ffn_bytes.saturating_add(read_bytes as u64);
                }
                ChannelMixingFfn::SparseMoe(moe) => {
                    if let Some(ref mut moe_sc) = scratch.moe {
                        let stats = moe.forward_into(
                            &scratch.x_shift,
                            &mut scratch.q_input,
                            moe_sc,
                            &mut scratch.x,
                        );
                        state.total_ffn_cols =
                            state.total_ffn_cols.saturating_add(stats.cols_total as u64);
                        state.active_ffn_cols = state
                            .active_ffn_cols
                            .saturating_add(stats.cols_active as u64);
                        state.total_ffn_bytes = state
                            .total_ffn_bytes
                            .saturating_add(stats.total_weight_bytes as u64);
                        state.read_ffn_bytes =
                            state.read_ffn_bytes.saturating_add(stats.bytes_read as u64);
                    }
                }
            }
        }

        layer_norm_into(&scratch.x, &self.ln_out, eps, &mut scratch.x_out);
        Ok(())
    }

    /// Processa 4 tokens consecutivos lendo cada matriz de peso **1 única vez** (`matmul4_into`),
    /// gravando os 4 vetores `x_out` em `out_x_outs` e deixando `state.scratch.x_out` igual ao 4º token.
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError`] se algum `token_id` estiver fora do vocabulário.
    pub fn forward_hidden_chunk4(
        &self,
        token_ids: [usize; 4],
        state: &mut Rwkv7State,
        out_x_outs: [&mut [f32]; 4],
    ) -> Result<(), LlmError> {
        let mut c4_box = state
            .scratch
            .chunk4
            .take()
            .unwrap_or_else(|| Box::new(Rwkv7Chunk4Scratch::new(&self.cfg)));

        let load_res = (|| -> Result<(), LlmError> {
            for (tok_id, dst) in token_ids.into_iter().zip(c4_box.x_emb.iter_mut()) {
                self.emb.lookup_into(tok_id, dst)?;
            }
            Ok(())
        })();

        if let Err(err) = load_res {
            state.scratch.chunk4 = Some(c4_box);
            return Err(err);
        }

        let res = self.forward_from_x_emb_chunk4(&mut c4_box, state, out_x_outs);
        state.scratch.chunk4 = Some(c4_box);
        res
    }

    /// Núcleo em lote de 4 tokens das `n_layers` camadas RWKV-7:
    /// - Todas as projeções lineares pesadas rodam via `matmul4_into` (1 leitura da RAM / 4 tokens).
    /// - O loop WKV7 itera `head` por fora e `t = 0..4` por dentro, mantendo cada matriz `64×64`
    ///   (16 KiB) travada na L1 durante os 4 passos consecutivos.
    fn forward_from_x_emb_chunk4(
        &self,
        c4: &mut Rwkv7Chunk4Scratch,
        state: &mut Rwkv7State,
        out_x_outs: [&mut [f32]; 4],
    ) -> Result<(), LlmError> {
        let d = self.cfg.d_model;
        let eps = self.cfg.layer_norm_eps;
        let gn_eps = self.cfg.group_norm_eps;
        let hd = self.cfg.head_dim;

        let scratch = &mut state.scratch;
        for (emb, x_dst) in c4.x_emb.iter().zip(c4.x.iter_mut()) {
            layer_norm_into(emb, &self.ln_in, eps, x_dst);
        }

        for (layer_idx, (block, layer_state)) in
            self.blocks.iter().zip(state.layers.iter_mut()).enumerate()
        {
            // 1. Time-Mixing RWKV-7 (LayerNorm + token-shift dos 4 tokens)
            for (x_src, ln_dst) in c4.x.iter().zip(c4.x_ln.iter_mut()) {
                layer_norm_into(x_src, &block.ln1, eps, ln_dst);
            }
            for (c, dst) in c4.dx[0].iter_mut().enumerate() {
                *dst = layer_state.prev_x_tm[c] - c4.x_ln[0][c];
            }
            for t in 1..4 {
                let prev_ln = &c4.x_ln[t - 1];
                let curr_ln = &c4.x_ln[t];
                for (c, dst) in c4.dx[t].iter_mut().enumerate() {
                    *dst = prev_ln[c] - curr_ln[c];
                }
            }
            layer_state.prev_x_tm.copy_from_slice(&c4.x_ln[3]);

            let tm = &block.tm;

            // Receptance (r)
            for t in 0..4 {
                shift_into(&c4.x_ln[t], &c4.dx[t], &tm.x_r, &mut c4.x_shift[t]);
            }
            {
                let [q0, q1, q2, q3] = &mut c4.q_input;
                let [r0, r1, r2, r3] = &mut c4.r;
                tm.receptance.matmul4_into(
                    [
                        &c4.x_shift[0][..],
                        &c4.x_shift[1][..],
                        &c4.x_shift[2][..],
                        &c4.x_shift[3][..],
                    ],
                    [&mut q0[..], &mut q1[..], &mut q2[..], &mut q3[..]],
                    [&mut r0[..], &mut r1[..], &mut r2[..], &mut r3[..]],
                );
            }

            // Key (k)
            for t in 0..4 {
                shift_into(&c4.x_ln[t], &c4.dx[t], &tm.x_k, &mut c4.x_shift[t]);
            }
            {
                let [q0, q1, q2, q3] = &mut c4.q_input;
                let [k0, k1, k2, k3] = &mut c4.k;
                tm.key.matmul4_into(
                    [
                        &c4.x_shift[0][..],
                        &c4.x_shift[1][..],
                        &c4.x_shift[2][..],
                        &c4.x_shift[3][..],
                    ],
                    [&mut q0[..], &mut q1[..], &mut q2[..], &mut q3[..]],
                    [&mut k0[..], &mut k1[..], &mut k2[..], &mut k3[..]],
                );
            }

            // Value (v) + residual v_first
            for t in 0..4 {
                shift_into(&c4.x_ln[t], &c4.dx[t], &tm.x_v, &mut c4.x_shift[t]);
            }
            {
                let [q0, q1, q2, q3] = &mut c4.q_input;
                let [v0, v1, v2, v3] = &mut c4.v;
                tm.value.matmul4_into(
                    [
                        &c4.x_shift[0][..],
                        &c4.x_shift[1][..],
                        &c4.x_shift[2][..],
                        &c4.x_shift[3][..],
                    ],
                    [&mut q0[..], &mut q1[..], &mut q2[..], &mut q3[..]],
                    [&mut v0[..], &mut v1[..], &mut v2[..], &mut v3[..]],
                );
            }

            if layer_idx == 0 {
                for (vf, v_src) in c4.v_first.iter_mut().zip(c4.v.iter()) {
                    vf.copy_from_slice(v_src);
                }
            } else if let Some(ref vr) = tm.v_res {
                let ldim = vr.lora_dim;
                {
                    let [q0, q1, q2, q3] = &mut c4.q_input;
                    let [l0, l1, l2, l3] = &mut c4.lora_buf;
                    vr.v1.matmul4_into(
                        [
                            &c4.x_shift[0][..],
                            &c4.x_shift[1][..],
                            &c4.x_shift[2][..],
                            &c4.x_shift[3][..],
                        ],
                        [&mut q0[..], &mut q1[..], &mut q2[..], &mut q3[..]],
                        [
                            &mut l0[..ldim],
                            &mut l1[..ldim],
                            &mut l2[..ldim],
                            &mut l3[..ldim],
                        ],
                    );
                }
                {
                    let [q0, q1, q2, q3] = &mut c4.q_input;
                    let [o0, o1, o2, o3] = &mut c4.tm_out;
                    vr.v2.matmul4_into(
                        [
                            &c4.lora_buf[0][..ldim],
                            &c4.lora_buf[1][..ldim],
                            &c4.lora_buf[2][..ldim],
                            &c4.lora_buf[3][..ldim],
                        ],
                        [&mut q0[..], &mut q1[..], &mut q2[..], &mut q3[..]],
                        [&mut o0[..], &mut o1[..], &mut o2[..], &mut o3[..]],
                    );
                }
                for t in 0..4 {
                    for c in 0..d {
                        let gate = sigmoid(vr.v0[c] + c4.tm_out[t][c]);
                        c4.v[t][c] += (c4.v_first[t][c] - c4.v[t][c]) * gate;
                    }
                }
            }

            // Decay (w_logit)
            for t in 0..4 {
                shift_into(&c4.x_ln[t], &c4.dx[t], &tm.x_w, &mut c4.x_shift[t]);
            }
            let w_ldim = self.cfg.decay_lora_dim;
            {
                let [q0, q1, q2, q3] = &mut c4.q_input;
                let [l0, l1, l2, l3] = &mut c4.lora_buf;
                tm.w1.matmul4_into(
                    [
                        &c4.x_shift[0][..],
                        &c4.x_shift[1][..],
                        &c4.x_shift[2][..],
                        &c4.x_shift[3][..],
                    ],
                    [&mut q0[..], &mut q1[..], &mut q2[..], &mut q3[..]],
                    [
                        &mut l0[..w_ldim],
                        &mut l1[..w_ldim],
                        &mut l2[..w_ldim],
                        &mut l3[..w_ldim],
                    ],
                );
            }
            for buf in &mut c4.lora_buf {
                for val in &mut buf[..w_ldim] {
                    *val = val.tanh();
                }
            }
            {
                let [q0, q1, q2, q3] = &mut c4.q_input;
                let [w0, w1, w2, w3] = &mut c4.w_logit;
                tm.w2.matmul4_into(
                    [
                        &c4.lora_buf[0][..w_ldim],
                        &c4.lora_buf[1][..w_ldim],
                        &c4.lora_buf[2][..w_ldim],
                        &c4.lora_buf[3][..w_ldim],
                    ],
                    [&mut q0[..], &mut q1[..], &mut q2[..], &mut q3[..]],
                    [&mut w0[..], &mut w1[..], &mut w2[..], &mut w3[..]],
                );
            }
            for w_vec in &mut c4.w_logit {
                for (dst, &bias) in w_vec.iter_mut().zip(tm.w0.iter()) {
                    *dst += bias;
                }
            }

            // In-context learning rate (a_vec)
            for t in 0..4 {
                shift_into(&c4.x_ln[t], &c4.dx[t], &tm.x_a, &mut c4.x_shift[t]);
            }
            let a_ldim = self.cfg.a_lora_dim;
            {
                let [q0, q1, q2, q3] = &mut c4.q_input;
                let [l0, l1, l2, l3] = &mut c4.lora_buf;
                tm.a1.matmul4_into(
                    [
                        &c4.x_shift[0][..],
                        &c4.x_shift[1][..],
                        &c4.x_shift[2][..],
                        &c4.x_shift[3][..],
                    ],
                    [&mut q0[..], &mut q1[..], &mut q2[..], &mut q3[..]],
                    [
                        &mut l0[..a_ldim],
                        &mut l1[..a_ldim],
                        &mut l2[..a_ldim],
                        &mut l3[..a_ldim],
                    ],
                );
            }
            {
                let [q0, q1, q2, q3] = &mut c4.q_input;
                let [a0, a1, a2, a3] = &mut c4.a_vec;
                tm.a2.matmul4_into(
                    [
                        &c4.lora_buf[0][..a_ldim],
                        &c4.lora_buf[1][..a_ldim],
                        &c4.lora_buf[2][..a_ldim],
                        &c4.lora_buf[3][..a_ldim],
                    ],
                    [&mut q0[..], &mut q1[..], &mut q2[..], &mut q3[..]],
                    [&mut a0[..], &mut a1[..], &mut a2[..], &mut a3[..]],
                );
            }
            for a_out in &mut c4.a_vec {
                for (dst, &bias) in a_out.iter_mut().zip(tm.a0.iter()) {
                    *dst = sigmoid(bias + *dst);
                }
            }

            // Gate (g_vec)
            for t in 0..4 {
                shift_into(&c4.x_ln[t], &c4.dx[t], &tm.x_g, &mut c4.x_shift[t]);
            }
            let g_ldim = self.cfg.gate_lora_dim;
            {
                let [q0, q1, q2, q3] = &mut c4.q_input;
                let [l0, l1, l2, l3] = &mut c4.lora_buf;
                tm.g1.matmul4_into(
                    [
                        &c4.x_shift[0][..],
                        &c4.x_shift[1][..],
                        &c4.x_shift[2][..],
                        &c4.x_shift[3][..],
                    ],
                    [&mut q0[..], &mut q1[..], &mut q2[..], &mut q3[..]],
                    [
                        &mut l0[..g_ldim],
                        &mut l1[..g_ldim],
                        &mut l2[..g_ldim],
                        &mut l3[..g_ldim],
                    ],
                );
            }
            for buf in &mut c4.lora_buf {
                for val in &mut buf[..g_ldim] {
                    *val = sigmoid(*val);
                }
            }
            {
                let [q0, q1, q2, q3] = &mut c4.q_input;
                let [g0, g1, g2, g3] = &mut c4.g_vec;
                tm.g2.matmul4_into(
                    [
                        &c4.lora_buf[0][..g_ldim],
                        &c4.lora_buf[1][..g_ldim],
                        &c4.lora_buf[2][..g_ldim],
                        &c4.lora_buf[3][..g_ldim],
                    ],
                    [&mut q0[..], &mut q1[..], &mut q2[..], &mut q3[..]],
                    [&mut g0[..], &mut g1[..], &mut g2[..], &mut g3[..]],
                );
            }

            // Regra Delta WKV7: `head` por fora mantém cada bloco 64×64 (16 KiB) quente na L1 por 4 passos
            for head in 0..self.n_heads {
                let off = head * hd;
                let mat_off = head * hd * hd;
                let s_mat = &mut layer_state.state_mat[mat_off..mat_off + hd * hd];

                for t in 0..4 {
                    prepare_head_vectors(
                        &c4.k[t][off..off + hd],
                        &tm.k_k[off..off + hd],
                        &tm.k_a[off..off + hd],
                        &c4.a_vec[t][off..off + hd],
                        &c4.w_logit[t][off..off + hd],
                        &mut scratch.kk_head,
                        &mut scratch.k_hat_head,
                        &mut scratch.w_decay_head,
                    );

                    wkv7_head_step(
                        s_mat,
                        &c4.r[t][off..off + hd],
                        &scratch.w_decay_head,
                        &scratch.k_hat_head,
                        &c4.v[t][off..off + hd],
                        &scratch.kk_head,
                        &c4.a_vec[t][off..off + hd],
                        &mut scratch.ak_head,
                        &mut scratch.y_head,
                    );

                    group_norm_and_bonus(
                        &scratch.y_head,
                        &c4.r[t][off..off + hd],
                        &scratch.k_hat_head,
                        &c4.v[t][off..off + hd],
                        &tm.r_k[off..off + hd],
                        &tm.ln_x.weight[off..off + hd],
                        &tm.ln_x.bias[off..off + hd],
                        gn_eps,
                        &mut c4.rwkv[t][off..off + hd],
                    );
                }
            }

            for (rwkv_vec, g_vec) in c4.rwkv.iter_mut().zip(c4.g_vec.iter()) {
                for (dst, &g) in rwkv_vec.iter_mut().zip(g_vec.iter()) {
                    *dst *= g;
                }
            }
            {
                let [q0, q1, q2, q3] = &mut c4.q_input;
                let [o0, o1, o2, o3] = &mut c4.tm_out;
                tm.output.matmul4_into(
                    [
                        &c4.rwkv[0][..],
                        &c4.rwkv[1][..],
                        &c4.rwkv[2][..],
                        &c4.rwkv[3][..],
                    ],
                    [&mut q0[..], &mut q1[..], &mut q2[..], &mut q3[..]],
                    [&mut o0[..], &mut o1[..], &mut o2[..], &mut o3[..]],
                );
            }
            for (x_vec, out_vec) in c4.x.iter_mut().zip(c4.tm_out.iter()) {
                for (dst, &delta) in x_vec.iter_mut().zip(out_vec.iter()) {
                    *dst += delta;
                }
            }

            // 2. Channel-Mixing RWKV-7 (FFN em lote de 4 no `key` + acumulação esparsa no `value`)
            for (x_src, ln_dst) in c4.x.iter().zip(c4.x_ln.iter_mut()) {
                layer_norm_into(x_src, &block.ln2, eps, ln_dst);
            }
            for (c, dst) in c4.x_shift[0].iter_mut().enumerate() {
                *dst = c4.x_ln[0][c] + block.cm.x_k[c] * (layer_state.prev_x_cm[c] - c4.x_ln[0][c]);
            }
            for t in 1..4 {
                let prev_ln = &c4.x_ln[t - 1];
                let curr_ln = &c4.x_ln[t];
                for (c, dst) in c4.x_shift[t].iter_mut().enumerate() {
                    *dst = curr_ln[c] + block.cm.x_k[c] * (prev_ln[c] - curr_ln[c]);
                }
            }
            layer_state.prev_x_cm.copy_from_slice(&c4.x_ln[3]);

            match &block.cm.ffn {
                ChannelMixingFfn::Dense { key, value } => {
                    {
                        let [q0, q1, q2, q3] = &mut c4.q_input;
                        let [c0, c1, c2, c3] = &mut c4.cm_k;
                        key.matmul4_into(
                            [
                                &c4.x_shift[0][..],
                                &c4.x_shift[1][..],
                                &c4.x_shift[2][..],
                                &c4.x_shift[3][..],
                            ],
                            [&mut q0[..], &mut q1[..], &mut q2[..], &mut q3[..]],
                            [&mut c0[..], &mut c1[..], &mut c2[..], &mut c3[..]],
                        );
                    }
                    let mut active_sum = 0_usize;
                    for (cm_vec, x_vec) in c4.cm_k.iter_mut().zip(c4.x.iter_mut()) {
                        for val in cm_vec.iter_mut() {
                            let r = val.max(0.0);
                            *val = r * r;
                        }
                        active_sum =
                            active_sum.saturating_add(value.accumulate_sparse(cm_vec, x_vec));
                    }
                    let total_bytes = (key.weight_bytes() + value.weight_bytes()).saturating_mul(4);
                    // `key` foi lida 1x da RAM para os 4 tokens; `value` leu apenas as colunas ativas.
                    let read_bytes = key
                        .weight_bytes()
                        .saturating_add(active_sum.saturating_mul(value.col_stride_bytes()));
                    state.total_ffn_cols = state
                        .total_ffn_cols
                        .saturating_add((self.cfg.ffn_dim as u64).saturating_mul(4));
                    state.active_ffn_cols = state.active_ffn_cols.saturating_add(active_sum as u64);
                    state.total_ffn_bytes =
                        state.total_ffn_bytes.saturating_add(total_bytes as u64);
                    state.read_ffn_bytes = state.read_ffn_bytes.saturating_add(read_bytes as u64);
                }
                ChannelMixingFfn::SparseMoe(moe) => {
                    if let Some(ref mut moe_sc) = scratch.moe {
                        for t in 0..4 {
                            let stats = moe.forward_into(
                                &c4.x_shift[t],
                                &mut c4.q_input[t],
                                moe_sc,
                                &mut c4.x[t],
                            );
                            state.total_ffn_cols =
                                state.total_ffn_cols.saturating_add(stats.cols_total as u64);
                            state.active_ffn_cols = state
                                .active_ffn_cols
                                .saturating_add(stats.cols_active as u64);
                            state.total_ffn_bytes = state
                                .total_ffn_bytes
                                .saturating_add(stats.total_weight_bytes as u64);
                            state.read_ffn_bytes =
                                state.read_ffn_bytes.saturating_add(stats.bytes_read as u64);
                        }
                    }
                }
            }
        }

        for (x_src, out_dst) in c4.x.iter().zip(out_x_outs.into_iter()) {
            layer_norm_into(x_src, &self.ln_out, eps, out_dst);
        }
        // Mantém `state.scratch.x_out` sincronizado com o 4º token (paridade com `forward_hidden`)
        layer_norm_into(&c4.x[3], &self.ln_out, eps, &mut scratch.x_out);
        Ok(())
    }

    /// Avança 1 token e projeta a `head` completa (útil para inspeção e testes).
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::Inference`] se `token_id` estiver fora do vocabulário.
    pub fn forward_step(
        &self,
        token_id: usize,
        state: &mut Rwkv7State,
    ) -> Result<Vec<f32>, LlmError> {
        self.forward_hidden(token_id, state)?;
        let mut logits = vec![0.0_f32; self.cfg.vocab_size];
        self.head.matvec_into(
            &state.scratch.x_out,
            &mut state.scratch.q_input,
            &mut logits,
        );
        Ok(logits)
    }

    /// Registra o estado atual no Copy Bank apontando para `next_prompt_tok`.
    pub fn record_copy_entry(&self, state: &mut Rwkv7State, next_prompt_tok: TokenId) {
        let Some(ref cg) = self.copy_gate else {
            return;
        };
        if state.scratch.copy_tokens.len() >= self.cfg.copy_max_len {
            return;
        }
        let scratch = &mut state.scratch;
        let base = scratch.copy_keys.len();
        scratch
            .copy_keys
            .resize(base.saturating_add(cg.copy_dim), 0.0);
        cg.key.matvec_into(
            &scratch.x_out,
            &mut scratch.q_input,
            &mut scratch.copy_keys[base..base + cg.copy_dim],
        );
        scratch.copy_tokens.push(next_prompt_tok);
    }

    /// Amostra o próximo token respeitando `SimpleVob` do `llguidance` (+ Copy Gate quando ativo).
    #[must_use]
    pub fn sample_with_copy_gate(
        &self,
        state: &mut Rwkv7State,
        mask: &SimpleVob,
    ) -> Option<(TokenId, f32)> {
        let Some(ref cg) = self.copy_gate else {
            return self
                .head
                .argmax_masked(
                    &state.scratch.x_out,
                    mask,
                    &mut state.scratch.q_input,
                    &mut state.scratch.allowed_tokens,
                )
                .map(|tok| (tok, 0.0));
        };

        let num_entries = state.scratch.copy_tokens.len();
        if num_entries == 0 {
            return self
                .head
                .argmax_masked(
                    &state.scratch.x_out,
                    mask,
                    &mut state.scratch.q_input,
                    &mut state.scratch.allowed_tokens,
                )
                .map(|tok| (tok, 0.0));
        }

        let scratch = &mut state.scratch;
        scratch.allowed_tokens.clear();
        for row in 0..self.cfg.vocab_size {
            let Ok(tok_id) = TokenId::try_from(row) else {
                break;
            };
            if mask.is_allowed(tok_id) {
                scratch.allowed_tokens.push(row);
            }
        }
        if scratch.allowed_tokens.is_empty() {
            return None;
        }

        let d_c = cg.copy_dim;
        cg.query
            .matvec_into(&scratch.x_out, &mut scratch.q_input, &mut scratch.copy_q);

        scratch.copy_attn.clear();
        scratch.copy_attn.reserve(num_entries);
        let mut max_score = f32::NEG_INFINITY;

        for slot in 0..num_entries {
            let k_slice = &scratch.copy_keys[slot * d_c..(slot + 1) * d_c];
            let mut dot = 0.0_f32;
            for (qv, kv) in scratch.copy_q.iter().zip(k_slice.iter()) {
                dot += qv * kv;
            }
            let scaled = dot * cg.inv_sqrt_dim;
            if scaled > max_score {
                max_score = scaled;
            }
            scratch.copy_attn.push(scaled);
        }

        let mut attn_sum = 0.0_f32;
        for val in &mut scratch.copy_attn {
            let e = (*val - max_score).exp();
            *val = e;
            attn_sum += e;
        }
        let inv_attn_sum = if attn_sum > 0.0 { 1.0 / attn_sum } else { 0.0 };

        scratch.copy_ctx.fill(0.0);
        let mut any_allowed_copy = false;
        for slot in 0..num_entries {
            let weight = scratch.copy_attn[slot] * inv_attn_sum;
            let k_slice = &scratch.copy_keys[slot * d_c..(slot + 1) * d_c];
            for (dst, &kv) in scratch.copy_ctx.iter_mut().zip(k_slice.iter()) {
                *dst += weight * kv;
            }
            let tok = scratch.copy_tokens[slot];
            if mask.is_allowed(tok)
                && let Ok(idx) = usize::try_from(tok)
                && let Some(acc) = scratch.copy_vocab_acc.get_mut(idx)
            {
                *acc += weight;
                any_allowed_copy = true;
            }
        }

        let d_model = self.cfg.d_model;
        let mut gate_logit = cg.gate_bias;
        for (w, v) in cg.gate_weight[..d_model].iter().zip(scratch.x_out.iter()) {
            gate_logit += w * v;
        }
        for (w, v) in cg.gate_weight[d_model..]
            .iter()
            .zip(scratch.copy_ctx.iter())
        {
            gate_logit += w * v;
        }
        let p_copy = sigmoid(gate_logit).clamp(1e-6, 1.0 - 1e-6);

        self.head.scores_for_rows(
            &scratch.x_out,
            &scratch.allowed_tokens,
            &mut scratch.q_input,
            &mut scratch.vocab_scores,
        );

        let max_logit = scratch
            .vocab_scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut vocab_sum = 0.0_f32;
        for s in &mut scratch.vocab_scores {
            let e = (*s - max_logit).exp();
            *s = e;
            vocab_sum += e;
        }
        let inv_vocab_sum = if vocab_sum > 0.0 {
            1.0 / vocab_sum
        } else {
            0.0
        };

        let eff_copy = if any_allowed_copy { p_copy } else { 0.0 };
        let eff_gen = 1.0 - eff_copy;

        let mut best_row: Option<usize> = None;
        let mut best_prob = f32::NEG_INFINITY;
        for (&row, &exp_logit) in scratch
            .allowed_tokens
            .iter()
            .zip(scratch.vocab_scores.iter())
        {
            let p_vocab = exp_logit * inv_vocab_sum;
            let p_c = scratch.copy_vocab_acc.get(row).copied().unwrap_or(0.0);
            let combined = eff_gen * p_vocab + eff_copy * p_c;
            if best_row.is_none()
                || combined > best_prob
                || (combined == best_prob && best_row.is_some_and(|prev| row < prev))
            {
                best_prob = combined;
                best_row = Some(row);
            }
        }

        for &tok in &scratch.copy_tokens {
            if let Ok(idx) = usize::try_from(tok)
                && let Some(acc) = scratch.copy_vocab_acc.get_mut(idx)
            {
                *acc = 0.0;
            }
        }

        best_row
            .and_then(|row| TokenId::try_from(row).ok())
            .map(|tok| (tok, eff_copy))
    }

    /// Gera uma [`LlmAction`] 100% aderente ao JSON Schema usando o RWKV-7 G1 + `llguidance`,
    /// reutilizando o estado recorrente O(1) do prefixo entre turnos do `Orchestrator`.
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError`] se a inferência, a gramática ou a desserialização JSON falharem.
    pub fn generate_action_from_prompt(&self, prompt: &str) -> Result<LlmAction, LlmError> {
        let trimmed = prompt.trim();
        let formatted = format!("{trimmed}\nAssistant:");
        let prompt_ids = self.tok_trie.greedy_tokenize(formatted.as_bytes());
        let base_ids = self.tok_trie.greedy_tokenize(trimmed.as_bytes());
        let checkpoint_len = common_prefix_len(&base_ids, &prompt_ids);
        let record_from = prompt_ids.len().saturating_sub(self.cfg.copy_max_len);

        let mut cache = self
            .prefix_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut state = cache
            .working_state
            .take()
            .unwrap_or_else(|| Rwkv7State::new(&self.cfg));

        let copy_ok = self.copy_gate.is_none()
            || (record_from >= cache.copy_start_idx
                && cache.copy_tokens.len()
                    == cache.tokens.len().saturating_sub(cache.copy_start_idx)
                && cache.checkpoint_x_out.len() == self.cfg.d_model);
        let cache_hit = !cache.tokens.is_empty()
            && cache.layers_snapshot.is_some()
            && copy_ok
            && prompt_ids.starts_with(&cache.tokens);

        let reuse_len = if cache_hit {
            let n = cache.tokens.len();
            if let Some(ref snap) = cache.layers_snapshot {
                copy_layers(&mut state.layers, snap);
            }
            state.total_ffn_cols = 0;
            state.active_ffn_cols = 0;
            state.total_ffn_bytes = 0;
            state.read_ffn_bytes = 0;
            state.scratch.copy_keys.clear();
            state.scratch.copy_tokens.clear();
            if let Some(ref cg) = self.copy_gate {
                if record_from < n {
                    let skip = record_from.saturating_sub(cache.copy_start_idx);
                    state
                        .scratch
                        .copy_keys
                        .extend_from_slice(&cache.copy_keys[skip * cg.copy_dim..]);
                    state
                        .scratch
                        .copy_tokens
                        .extend_from_slice(&cache.copy_tokens[skip..]);
                }
                if n >= record_from
                    && let Some(&next_tok) = prompt_ids.get(n)
                {
                    state.scratch.x_out.copy_from_slice(&cache.checkpoint_x_out);
                    self.record_copy_entry(&mut state, next_tok);
                }
            }
            cache.stats.hits = cache.stats.hits.saturating_add(1);
            cache.stats.tokens_reused = cache.stats.tokens_reused.saturating_add(n as u64);
            n
        } else {
            state.reset_recurrent_zero();
            if let Err(err) = self.forward_hidden(self.cfg.bos_token_id, &mut state) {
                cache.working_state = Some(state);
                return Err(err);
            }
            if record_from == 0
                && let Some(&first_tok) = prompt_ids.first()
            {
                self.record_copy_entry(&mut state, first_tok);
            }
            cache.stats.misses = cache.stats.misses.saturating_add(1);
            0
        };

        let delta_len = prompt_ids.len().saturating_sub(reuse_len);
        for (offset, &tok_id) in prompt_ids[reuse_len..].iter().enumerate() {
            let idx = reuse_len + offset;
            let Ok(id) = usize::try_from(tok_id) else {
                cache.working_state = Some(state);
                return Err(LlmError::Inference(
                    "token id conversion overflow".to_owned(),
                ));
            };
            if let Err(err) = self.forward_hidden(id, &mut state) {
                cache.working_state = Some(state);
                return Err(err);
            }

            if idx + 1 == checkpoint_len && checkpoint_len > reuse_len {
                cache.tokens.clear();
                cache
                    .tokens
                    .extend_from_slice(&prompt_ids[..checkpoint_len]);
                match cache.layers_snapshot {
                    Some(ref mut snap) => copy_layers(snap, &state.layers),
                    None => cache.layers_snapshot = Some(state.layers.clone()),
                }
                cache.h_prefill.clear();
                cache.token_byte_ends.clear();
                cache.checkpoint_x_out.clear();
                cache.copy_start_idx = 0;
                cache.copy_keys.clear();
                cache.copy_tokens.clear();
                if self.copy_gate.is_some() {
                    cache.copy_start_idx = record_from.min(checkpoint_len);
                    cache
                        .checkpoint_x_out
                        .extend_from_slice(&state.scratch.x_out);
                    cache.copy_keys.extend_from_slice(&state.scratch.copy_keys);
                    cache
                        .copy_tokens
                        .extend_from_slice(&state.scratch.copy_tokens);
                }
            }

            let next_idx = idx.saturating_add(1);
            if next_idx >= record_from
                && let Some(&next_tok) = prompt_ids.get(next_idx)
            {
                self.record_copy_entry(&mut state, next_tok);
            }
        }
        cache.stats.tokens_prefilled = cache
            .stats
            .tokens_prefilled
            .saturating_add(delta_len as u64);

        let result = self.decode_json_after_prefill(&mut state, cache_hit, reuse_len, delta_len);
        cache.working_state = Some(state);
        result
    }

    fn decode_json_after_prefill(
        &self,
        state: &mut Rwkv7State,
        cache_hit: bool,
        reused_tokens: usize,
        prefilled_tokens: usize,
    ) -> Result<LlmAction, LlmError> {
        let mut matcher = Matcher::new(self.parser_factory.create_parser(self.grammar.clone()));
        if let Some(err) = matcher.get_error() {
            return Err(LlmError::GrammarInit(err));
        }

        let eos_u32 = TokenId::try_from(self.cfg.eos_token_id)
            .map_err(|_| LlmError::Inference("eos_token_id exceeds u32".to_owned()))?;
        let mut generated_ids: Vec<TokenId> = Vec::with_capacity(64);

        while generated_ids.len() < self.max_new_tokens && !matcher.is_stopped() {
            let mask = matcher
                .compute_mask()
                .map_err(|err| LlmError::GrammarInit(err.to_string()))?;

            let (next_tok, p_copy) = self.sample_with_copy_gate(state, &mask).ok_or_else(|| {
                LlmError::GrammarInit("llguidance mask allowed zero tokens".to_owned())
            })?;

            tracing::trace!(next_tok, p_copy, "rwkv7 sampled token");

            matcher
                .consume_token(next_tok)
                .map_err(|err| LlmError::GrammarInit(err.to_string()))?;

            if next_tok == eos_u32 {
                break;
            }

            generated_ids.push(next_tok);
            let next_usize = usize::try_from(next_tok).map_err(|_| {
                LlmError::Inference("generated token conversion overflow".to_owned())
            })?;
            self.forward_hidden(next_usize, state)?;

            for ff_tok in matcher.consume_ff_tokens() {
                if ff_tok == eos_u32 || generated_ids.len() >= self.max_new_tokens {
                    break;
                }
                generated_ids.push(ff_tok);
                let ff_usize = usize::try_from(ff_tok).map_err(|_| {
                    LlmError::Inference("fast-forward token conversion overflow".to_owned())
                })?;
                self.forward_hidden(ff_usize, state)?;
            }
        }

        let raw_json = self.tok_trie.decode_str(&generated_ids);
        tracing::debug!(
            cache_hit,
            reused_tokens,
            prefilled_tokens,
            tokens = generated_ids.len(),
            ffn_sparsity_pct = state.ffn_sparsity_ratio() * 100.0,
            raw_json = %raw_json,
            "rwkv-7 goose emitted constrained action json"
        );

        serde_json::from_str::<LlmAction>(raw_json.trim()).map_err(|err| {
            LlmError::Inference(format!(
                "constrained rwkv7 json failed LlmAction deserialization: {err}; raw={raw_json}"
            ))
        })
    }

    /// Indica se o checkpoint carregado possui a Dual-Head ISA + Entity Pointer ativa.
    #[must_use]
    pub const fn has_isa_head(&self) -> bool {
        self.isa_head.is_some()
    }

    /// Gera uma [`LlmAction`] usando a ISA compacta (Peças A + B + C):
    /// 1) Prefill único do prompt + pooling `e_s` das entidades -> [`NeuralEntityBank`].
    /// 2) Loop guiado pela máquina de estados `isa::expect`, usando `forward_hidden_vec`
    ///    para realimentar a semântica real de cada entidade apontada.
    /// 3) Com `strict_citations = true`, `FactId` inventado ou cláusula não lida não tem logit.
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError`] se a Dual-Head ISA não estiver carregada ou se a decodificação falhar.
    pub fn generate_action_isa(
        &self,
        prompt: &str,
        strict_citations: bool,
    ) -> Result<LlmAction, LlmError> {
        let isa = self.isa_head.as_ref().ok_or(LlmError::InvalidConfig(
            "checkpoint does not contain isa head weights (isa_ptr_dim == 0)",
        ))?;

        let d = self.cfg.d_model;
        let trimmed = prompt.trim();
        let formatted = format!("{trimmed}\nAssistant:");
        let prompt_ids = self.tok_trie.greedy_tokenize(formatted.as_bytes());
        let base_ids = self.tok_trie.greedy_tokenize(trimmed.as_bytes());
        let checkpoint_len = common_prefix_len(&base_ids, &prompt_ids);

        let mut cache = self
            .prefix_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut state = cache
            .working_state
            .take()
            .unwrap_or_else(|| Rwkv7State::new(&self.cfg));

        let cache_hit = !cache.tokens.is_empty()
            && cache.layers_snapshot.is_some()
            && cache.h_prefill.len() == cache.tokens.len().saturating_mul(d)
            && cache.token_byte_ends.len() == cache.tokens.len()
            && prompt_ids.starts_with(&cache.tokens);

        let mut h_prefill = vec![0.0_f32; prompt_ids.len().saturating_mul(d)];
        let mut token_byte_ends = Vec::with_capacity(prompt_ids.len());

        let (reuse_len, mut byte_cursor) = if cache_hit {
            let n = cache.tokens.len();
            if let Some(ref snap) = cache.layers_snapshot {
                copy_layers(&mut state.layers, snap);
            }
            state.total_ffn_cols = 0;
            state.active_ffn_cols = 0;
            state.total_ffn_bytes = 0;
            state.read_ffn_bytes = 0;
            h_prefill[..n * d].copy_from_slice(&cache.h_prefill);
            token_byte_ends.extend_from_slice(&cache.token_byte_ends);
            let cursor = token_byte_ends.last().copied().unwrap_or(0);
            cache.stats.hits = cache.stats.hits.saturating_add(1);
            cache.stats.tokens_reused = cache.stats.tokens_reused.saturating_add(n as u64);
            (n, cursor)
        } else {
            state.reset_recurrent_zero();
            if let Err(err) = self.forward_hidden(self.cfg.bos_token_id, &mut state) {
                cache.working_state = Some(state);
                return Err(err);
            }
            cache.stats.misses = cache.stats.misses.saturating_add(1);
            (0, 0)
        };

        let delta_len = prompt_ids.len().saturating_sub(reuse_len);

        // Segmento 1: até o checkpoint compartilhável entre turnos (em blocos de 4)
        if checkpoint_len > reuse_len {
            if let Err(err) = self.prefill_isa_range(
                &prompt_ids,
                reuse_len,
                checkpoint_len,
                &mut state,
                &mut h_prefill,
                &mut token_byte_ends,
                &mut byte_cursor,
            ) {
                cache.working_state = Some(state);
                return Err(err);
            }

            cache.tokens.clear();
            cache
                .tokens
                .extend_from_slice(&prompt_ids[..checkpoint_len]);
            match cache.layers_snapshot {
                Some(ref mut snap) => copy_layers(snap, &state.layers),
                None => cache.layers_snapshot = Some(state.layers.clone()),
            }
            cache.h_prefill.clear();
            cache
                .h_prefill
                .extend_from_slice(&h_prefill[..checkpoint_len * d]);
            cache.token_byte_ends.clear();
            cache
                .token_byte_ends
                .extend_from_slice(&token_byte_ends[..checkpoint_len]);
            cache.checkpoint_x_out.clear();
            cache.copy_start_idx = 0;
            cache.copy_keys.clear();
            cache.copy_tokens.clear();
        }

        // Segmento 2: sufixo "\nAssistant:" (também em bloco de 4 quando >= 4 tokens)
        let tail_start = checkpoint_len.max(reuse_len);
        if tail_start < prompt_ids.len()
            && let Err(err) = self.prefill_isa_range(
                &prompt_ids,
                tail_start,
                prompt_ids.len(),
                &mut state,
                &mut h_prefill,
                &mut token_byte_ends,
                &mut byte_cursor,
            )
        {
            cache.working_state = Some(state);
            return Err(err);
        }

        cache.stats.tokens_prefilled = cache
            .stats
            .tokens_prefilled
            .saturating_add(delta_len as u64);

        let result = self.decode_isa_after_prefill(
            isa,
            &formatted,
            &h_prefill,
            &token_byte_ends,
            strict_citations,
            &mut state,
            cache_hit,
            reuse_len,
            delta_len,
        );
        cache.working_state = Some(state);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_isa_range(
        &self,
        prompt_ids: &[TokenId],
        start: usize,
        end: usize,
        state: &mut Rwkv7State,
        h_prefill: &mut [f32],
        token_byte_ends: &mut Vec<usize>,
        byte_cursor: &mut usize,
    ) -> Result<(), LlmError> {
        let d = self.cfg.d_model;
        let mut idx = start;

        while idx + 4 <= end {
            let t0 = usize::try_from(prompt_ids[idx])
                .map_err(|_| LlmError::Inference("prompt token id overflow".to_owned()))?;
            let t1 = usize::try_from(prompt_ids[idx + 1])
                .map_err(|_| LlmError::Inference("prompt token id overflow".to_owned()))?;
            let t2 = usize::try_from(prompt_ids[idx + 2])
                .map_err(|_| LlmError::Inference("prompt token id overflow".to_owned()))?;
            let t3 = usize::try_from(prompt_ids[idx + 3])
                .map_err(|_| LlmError::Inference("prompt token id overflow".to_owned()))?;

            let (s0, rest) = h_prefill[idx * d..(idx + 4) * d].split_at_mut(d);
            let (s1, rest) = rest.split_at_mut(d);
            let (s2, s3) = rest.split_at_mut(d);

            self.forward_hidden_chunk4([t0, t1, t2, t3], state, [s0, s1, s2, s3])?;

            for &tok_id in &prompt_ids[idx..idx + 4] {
                *byte_cursor = byte_cursor.saturating_add(self.tok_trie.token(tok_id).len());
                token_byte_ends.push(*byte_cursor);
            }
            idx += 4;
        }

        while idx < end {
            let tok_id = prompt_ids[idx];
            let id = usize::try_from(tok_id)
                .map_err(|_| LlmError::Inference("prompt token id overflow".to_owned()))?;
            self.forward_hidden(id, state)?;
            h_prefill[idx * d..(idx + 1) * d].copy_from_slice(&state.scratch.x_out);
            *byte_cursor = byte_cursor.saturating_add(self.tok_trie.token(tok_id).len());
            token_byte_ends.push(*byte_cursor);
            idx += 1;
        }

        Ok(())
    }

    fn prefill_plain_range(
        &self,
        prompt_ids: &[TokenId],
        start: usize,
        end: usize,
        state: &mut Rwkv7State,
    ) -> Result<(), LlmError> {
        let d = self.cfg.d_model;
        let mut idx = start;

        if end.saturating_sub(start) >= 4 {
            let mut tmp = vec![0.0_f32; 4 * d];
            while idx + 4 <= end {
                let t0 = usize::try_from(prompt_ids[idx])
                    .map_err(|_| LlmError::Inference("token id conversion overflow".to_owned()))?;
                let t1 = usize::try_from(prompt_ids[idx + 1])
                    .map_err(|_| LlmError::Inference("token id conversion overflow".to_owned()))?;
                let t2 = usize::try_from(prompt_ids[idx + 2])
                    .map_err(|_| LlmError::Inference("token id conversion overflow".to_owned()))?;
                let t3 = usize::try_from(prompt_ids[idx + 3])
                    .map_err(|_| LlmError::Inference("token id conversion overflow".to_owned()))?;

                let (s0, rest) = tmp.split_at_mut(d);
                let (s1, rest) = rest.split_at_mut(d);
                let (s2, s3) = rest.split_at_mut(d);
                self.forward_hidden_chunk4([t0, t1, t2, t3], state, [s0, s1, s2, s3])?;
                idx += 4;
            }
        }

        while idx < end {
            let id = usize::try_from(prompt_ids[idx])
                .map_err(|_| LlmError::Inference("token id conversion overflow".to_owned()))?;
            self.forward_hidden(id, state)?;
            idx += 1;
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_isa_after_prefill(
        &self,
        isa: &IsaHeadWeights,
        formatted: &str,
        h_prefill: &[f32],
        token_byte_ends: &[usize],
        strict_citations: bool,
        state: &mut Rwkv7State,
        cache_hit: bool,
        reused_tokens: usize,
        prefilled_tokens: usize,
    ) -> Result<LlmAction, LlmError> {
        let d = self.cfg.d_model;

        // Monta o NeuralEntityBank (Peças B + C)
        let mut bank = NeuralEntityBank::new(isa.ptr_dim);
        let mut pooled_buf = vec![0.0_f32; d];
        for (entity_text, kind) in extract_prompt_entities(formatted) {
            if let Some((t0, t1)) = find_entity_token_span(formatted, &entity_text, token_byte_ends)
            {
                pooled_buf.fill(0.0);
                let inv_len = 1.0 / ((t1 - t0).max(1) as f32);
                for t in t0..t1 {
                    let h_slice = &h_prefill[t * d..(t + 1) * d];
                    for c in 0..d {
                        pooled_buf[c] += h_slice[c] * inv_len;
                    }
                }
                let ptr_key = isa.project_ptr_k(&pooled_buf);
                let _ = bank.upsert(&entity_text, kind, ptr_key);
            }
        }

        let mut syms: Vec<Sym> = Vec::with_capacity(8);
        let mut ptr_q_buf = vec![0.0_f32; isa.ptr_dim];
        let mut fb_vec = vec![0.0_f32; d];
        let mut logits_buf = vec![0.0_f32; self.cfg.vocab_size];

        // 1. Escolhe a operação (`Expect::Op` -> 9 logits)
        let op_idx =
            IsaHeadWeights::argmax_rows(&isa.op_head, ISA_OP_COUNT, d, &state.scratch.x_out);
        let op = Op::ALL[op_idx];
        syms.push(Sym::Op(op));
        self.forward_hidden_vec(&isa.op_emb[op_idx * d..(op_idx + 1) * d], state)?;

        // 2. Decodifica argumentos até `expect(&syms)` voltar para `Expect::Op`
        let mut reflist_items = 0_usize;
        loop {
            let next_exp = expect(&syms)
                .map_err(|err| LlmError::Inference(format!("isa grammar error: {err}")))?;
            match next_exp {
                Expect::Op => break,

                Expect::Num => {
                    let n_idx = IsaHeadWeights::argmax_rows(
                        &isa.num_head,
                        isa.num_vocab,
                        d,
                        &state.scratch.x_out,
                    );
                    let n_u8 = u8::try_from(n_idx).unwrap_or(u8::MAX);
                    syms.push(Sym::Num(n_u8));
                    self.forward_hidden_vec(&isa.num_emb[n_idx * d..(n_idx + 1) * d], state)?;
                }

                Expect::Lit => {
                    let text = if op == Op::ProposeAnswer {
                        // Placeholder: o texto é renderizado depois, a partir do FactId citado.
                        // Mantém o estado recorrente "fechando" o literal, sem tocar na head 65k.
                        self.forward_hidden_vec(
                            &isa.ctl_emb[CTL_LIT_CLOSE * d..(CTL_LIT_CLOSE + 1) * d],
                            state,
                        )?;
                        String::new()
                    } else {
                        self.decode_isa_literal(isa, state, &mut logits_buf)?
                    };
                    syms.push(Sym::Lit(text));
                }

                Expect::Ref | Expect::RefOrEnd => {
                    let filter = SlotFilter::for_op(op, next_exp);
                    let allow_end = next_exp == Expect::RefOrEnd;
                    let allow_lit = filter.allows_literal_fallback(strict_citations);

                    if allow_end && reflist_items >= MAX_ISA_REFLIST_ITEMS {
                        syms.push(Sym::End);
                        self.forward_hidden_vec(
                            &isa.ctl_emb[CTL_END * d..(CTL_END + 1) * d],
                            state,
                        )?;
                        break;
                    }

                    isa.project_ptr_q(&state.scratch.x_out, &mut ptr_q_buf);
                    let best_ptr = bank.argmax_pointer(&ptr_q_buf, filter);
                    let end_score = if allow_end {
                        IsaHeadWeights::dot_row(&isa.ctl_head, CTL_END, d, &state.scratch.x_out)
                    } else {
                        f32::NEG_INFINITY
                    };
                    let lit_score = if allow_lit {
                        IsaHeadWeights::dot_row(
                            &isa.ctl_head,
                            CTL_LIT_OPEN,
                            d,
                            &state.scratch.x_out,
                        )
                    } else {
                        f32::NEG_INFINITY
                    };

                    let ptr_score = best_ptr.map_or(f32::NEG_INFINITY, |(_, s)| s);

                    if allow_end && end_score >= ptr_score && end_score >= lit_score {
                        syms.push(Sym::End);
                        self.forward_hidden_vec(
                            &isa.ctl_emb[CTL_END * d..(CTL_END + 1) * d],
                            state,
                        )?;
                    } else if let Some((slot_id, _)) = best_ptr
                        && ptr_score >= lit_score
                    {
                        syms.push(Sym::Ptr(slot_id));
                        if allow_end {
                            reflist_items = reflist_items.saturating_add(1);
                        }
                        if let Some(entry) = bank.get(slot_id) {
                            isa.build_ptr_feedback(&entry.ptr_key, &mut fb_vec);
                            self.forward_hidden_vec(&fb_vec, state)?;
                        }
                    } else if allow_lit {
                        self.forward_hidden_vec(
                            &isa.ctl_emb[CTL_LIT_OPEN * d..(CTL_LIT_OPEN + 1) * d],
                            state,
                        )?;
                        let text = self.decode_isa_literal(isa, state, &mut logits_buf)?;
                        syms.push(Sym::Lit(text));
                        if allow_end {
                            reflist_items = reflist_items.saturating_add(1);
                        }
                    } else {
                        return Err(LlmError::Inference(
                            "no eligible entity slot available for strict isa pointer".to_owned(),
                        ));
                    }
                }
            }
        }

        let text_bank = bank.to_text_bank();
        let (action, _) = decode_one(&syms, &text_bank)
            .map_err(|err| LlmError::Inference(format!("isa decode failed: {err}")))?;

        tracing::debug!(
            cache_hit,
            reused_tokens,
            prefilled_tokens,
            isa_symbols = syms.len(),
            bank_slots = bank.len(),
            ffn_sparsity_pct = state.ffn_sparsity_ratio() * 100.0,
            isa_text = %crate::isa::to_text(&syms),
            "rwkv-7 goose emitted isa action"
        );

        Ok(LlmAction::from(action))
    }

    fn decode_isa_literal(
        &self,
        isa: &IsaHeadWeights,
        state: &mut Rwkv7State,
        logits_buf: &mut [f32],
    ) -> Result<String, LlmError> {
        let d = self.cfg.d_model;
        let mut lit_tokens: Vec<TokenId> = Vec::with_capacity(32);

        for _ in 0..self.max_new_tokens {
            self.head
                .matvec_into(&state.scratch.x_out, &mut state.scratch.q_input, logits_buf);
            let close_score =
                IsaHeadWeights::dot_row(&isa.ctl_head, CTL_LIT_CLOSE, d, &state.scratch.x_out);

            let mut best_id = self.cfg.eos_token_id;
            let mut best_score = f32::NEG_INFINITY;
            for (idx, &val) in logits_buf.iter().enumerate() {
                if val > best_score {
                    best_score = val;
                    best_id = idx;
                }
            }

            if close_score >= best_score || best_id == self.cfg.eos_token_id {
                break;
            }

            let tok_u32 = TokenId::try_from(best_id)
                .map_err(|_| LlmError::Inference("literal token id exceeds u32".to_owned()))?;
            lit_tokens.push(tok_u32);
            self.forward_hidden(best_id, state)?;
        }

        // Fecha o literal no estado recorrente para o próximo símbolo ISA (ex.: cited_facts)
        self.forward_hidden_vec(
            &isa.ctl_emb[CTL_LIT_CLOSE * d..(CTL_LIT_CLOSE + 1) * d],
            state,
        )?;

        Ok(self.tok_trie.decode_str(&lit_tokens))
    }

    /// Inicia uma sessão de chat contínua com estado recorrente O(1) ancorado em `bos_token_id`.
    /// Se `system_prompt` for fornecido, é absorvido uma única vez no estado inicial.
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::Inference`] se o prefill inicial falhar.
    pub fn start_chat(
        &self,
        system_prompt: Option<&str>,
    ) -> Result<Rwkv7ChatSession<'_>, LlmError> {
        let mut state = Rwkv7State::new(&self.cfg);
        self.forward_hidden(self.cfg.bos_token_id, &mut state)?;
        let mut total_tokens_seen = 1_usize;

        if let Some(sys) = system_prompt.map(str::trim).filter(|s| !s.is_empty()) {
            let formatted = format!("System: {sys}\n\n");
            let ids = self.tok_trie.greedy_tokenize(formatted.as_bytes());
            for tok in ids {
                let id = usize::try_from(tok)
                    .map_err(|_| LlmError::Inference("system token id overflow".to_owned()))?;
                self.forward_hidden(id, &mut state)?;
                total_tokens_seen = total_tokens_seen.saturating_add(1);
            }
        }

        Ok(Rwkv7ChatSession {
            model: self,
            state,
            logits_buf: vec![0.0_f32; self.cfg.vocab_size].into_boxed_slice(),
            rng: ChatRng::from_clock(),
            turn_count: 0,
            total_tokens_seen,
            no_think: false,
            ends_with_blank_line: false,
        })
    }

    /// Projeta o estado oculto atual (`state.scratch.x_out`) na `head` completa sem alocar no heap.
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::InvalidConfig`] se `out_logits.len() != config().vocab_size`.
    pub fn project_head_into(
        &self,
        state: &mut Rwkv7State,
        out_logits: &mut [f32],
    ) -> Result<(), LlmError> {
        if out_logits.len() != self.cfg.vocab_size {
            return Err(LlmError::InvalidConfig(
                "out_logits length must match vocab_size",
            ));
        }
        self.head
            .matvec_into(&state.scratch.x_out, &mut state.scratch.q_input, out_logits);
        Ok(())
    }

    /// Projeta a `head` apenas nas linhas autorizadas pelo bitset `SimpleVob` do `llguidance`.
    #[must_use]
    pub fn sample_masked_from_hidden(
        &self,
        state: &mut Rwkv7State,
        mask: &SimpleVob,
    ) -> Option<TokenId> {
        self.head.argmax_masked(
            &state.scratch.x_out,
            mask,
            &mut state.scratch.q_input,
            &mut state.scratch.allowed_tokens,
        )
    }

    /// Projeta a `head` apenas no subconjunto explícito de linhas `rows` (para benchmarks de esparsidade).
    #[must_use]
    pub fn sample_subset_from_hidden(
        &self,
        state: &mut Rwkv7State,
        rows: &[usize],
    ) -> Option<TokenId> {
        self.head
            .argmax_rows(&state.scratch.x_out, rows, &mut state.scratch.q_input)
    }

    /// Cria um [`Matcher`] novo do `llguidance` vinculado ao JSON Schema de [`LlmAction`].
    ///
    /// # Errors
    ///
    /// Retorna [`LlmError::GrammarInit`] se o parser reportar erro de compilação.
    pub fn new_grammar_matcher(&self) -> Result<Matcher, LlmError> {
        let matcher = Matcher::new(self.parser_factory.create_parser(self.grammar.clone()));
        if let Some(err) = matcher.get_error() {
            return Err(LlmError::GrammarInit(err));
        }
        Ok(matcher)
    }

    /// Estatísticas acumuladas de hits/misses e tokens economizados pelo estado vivo entre turnos.
    #[must_use]
    pub fn prefix_cache_stats(&self) -> PrefixCacheStats {
        self.prefix_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stats
    }

    /// Limpa o snapshot de prefixo mantido em RAM (forçando prefill frio na próxima chamada).
    pub fn clear_prefix_cache(&self) {
        let mut cache = self
            .prefix_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cache.tokens.clear();
        cache.layers_snapshot = None;
        cache.h_prefill.clear();
        cache.token_byte_ends.clear();
        cache.checkpoint_x_out.clear();
        cache.copy_start_idx = 0;
        cache.copy_keys.clear();
        cache.copy_tokens.clear();
    }
}

impl ConstrainedLlm for Rwkv7Llm {
    async fn next_action(&self, history: &[Message]) -> Result<LlmAction, LlmPortError> {
        let mut prompt = String::with_capacity(256);
        for msg in history {
            let prefix = match msg.role {
                Role::System => "SYSTEM: ",
                Role::User => "USER: ",
                Role::Assistant => "ASSISTANT: ",
                Role::Tool => "TOOL: ",
            };
            prompt.push_str(prefix);
            prompt.push_str(&msg.content);
            prompt.push('\n');
        }
        let action = match self.decode_mode {
            DecodeMode::Auto => {
                if self.has_isa_head() {
                    self.generate_action_isa(&prompt, true)?
                } else {
                    self.generate_action_from_prompt(&prompt)?
                }
            }
            DecodeMode::Json => self.generate_action_from_prompt(&prompt)?,
            DecodeMode::Isa => self.generate_action_isa(&prompt, true)?,
        };
        Ok(action)
    }
}

fn parse_vocab_bin(bytes: &[u8], expected_vocab: usize) -> Result<Vec<Vec<u8>>, LlmError> {
    if bytes.len() < 4 {
        return Err(LlmError::ModelLoad("vocab.bin is truncated".to_owned()));
    }
    let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if count != expected_vocab {
        return Err(LlmError::ModelLoad(format!(
            "vocab.bin count ({count}) does not match config.vocab_size ({expected_vocab})"
        )));
    }

    let mut cursor = 4_usize;
    let mut words = Vec::with_capacity(count);
    for _ in 0..count {
        let len_end = cursor
            .checked_add(2)
            .ok_or_else(|| LlmError::ModelLoad("vocab.bin cursor overflow".to_owned()))?;
        let len_slice = bytes
            .get(cursor..len_end)
            .ok_or_else(|| LlmError::ModelLoad("unexpected eof in vocab.bin header".to_owned()))?;
        let tok_len = usize::from(u16::from_le_bytes([len_slice[0], len_slice[1]]));
        cursor = len_end;

        let tok_end = cursor
            .checked_add(tok_len)
            .ok_or_else(|| LlmError::ModelLoad("vocab.bin token length overflow".to_owned()))?;
        let tok_slice = bytes
            .get(cursor..tok_end)
            .ok_or_else(|| LlmError::ModelLoad("unexpected eof in vocab.bin body".to_owned()))?;
        words.push(tok_slice.to_vec());
        cursor = tok_end;
    }

    Ok(words)
}

#[inline]
fn shift_into(x: &[f32], dx: &[f32], factor: &[f32], out: &mut [f32]) {
    for i in 0..x.len() {
        out[i] = x[i] + dx[i] * factor[i];
    }
}

#[inline]
fn prepare_head_vectors(
    k: &[f32],
    k_k: &[f32],
    k_a: &[f32],
    a: &[f32],
    w_logit: &[f32],
    out_kk: &mut [f32],
    out_k_hat: &mut [f32],
    out_w_decay: &mut [f32],
) {
    let n = k.len();
    let mut norm_sq = 0.0_f32;
    for j in 0..n {
        let kk_raw = k[j] * k_k[j];
        out_kk[j] = kk_raw;
        norm_sq += kk_raw * kk_raw;
        out_k_hat[j] = k[j] * (1.0 + (a[j] - 1.0) * k_a[j]);
        out_w_decay[j] = (NEG_EXP_NEG_HALF * sigmoid(w_logit[j])).exp();
    }
    let inv_norm = 1.0 / norm_sq.sqrt().max(1e-12);
    for val in out_kk.iter_mut() {
        *val *= inv_norm;
    }
}

#[inline]
fn wkv7_head_step(
    s_mat: &mut [f32],
    r: &[f32],
    w_decay: &[f32],
    k_hat: &[f32],
    v: &[f32],
    kk: &[f32],
    a: &[f32],
    ak_scratch: &mut [f32],
    out_y: &mut [f32],
) {
    let n = r.len();
    if n == 0 {
        return;
    }
    for ((ak, &a_j), &kk_j) in ak_scratch.iter_mut().zip(a).zip(kk) {
        *ak = a_j * kk_j;
    }

    for (i, (row, y_out)) in s_mat.chunks_exact_mut(n).zip(out_y.iter_mut()).enumerate() {
        let sa_i = dot_f32_lanes(row, kk);
        let v_i = v[i];
        for (((s, &w), &ak), &kh) in row
            .iter_mut()
            .zip(w_decay)
            .zip(ak_scratch.iter())
            .zip(k_hat)
        {
            *s = *s * w - sa_i * ak + v_i * kh;
        }
        *y_out = dot_f32_lanes(row, r);
    }
}

/// Produto escalar `f32` com 8 acumuladores independentes: permite ao LLVM vetorizar a redução
/// (uma soma sequencial única não é vetorizável sem `-ffast-math`). A ordem de soma difere da
/// versão escalar, então o resultado muda apenas por arredondamento de `f32`.
#[inline]
fn dot_f32_lanes(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut acc = [0.0_f32; LANES];
    let a_chunks = a.chunks_exact(LANES);
    let b_chunks = b.chunks_exact(LANES);
    let a_rem = a_chunks.remainder();
    let b_rem = b_chunks.remainder();
    for (ac, bc) in a_chunks.zip(b_chunks) {
        for ((s, &x), &y) in acc.iter_mut().zip(ac).zip(bc) {
            *s += x * y;
        }
    }
    let mut total: f32 = acc.iter().sum();
    for (&x, &y) in a_rem.iter().zip(b_rem) {
        total += x * y;
    }
    total
}

/// Carrega apenas o [`TokTrie`] do vocabulário RWKV World (`vocab.bin`) sem carregar os pesos do modelo.
///
/// # Errors
///
/// Retorna [`LlmError::ModelLoad`] se o arquivo estiver ausente ou corrompido, ou
/// [`LlmError::InvalidConfig`] se `vocab_size` ou `eos_token_id` excederem `u32`.
pub fn load_vocab_trie(
    vocab_path: impl AsRef<Path>,
    eos_token_id: usize,
) -> Result<TokTrie, LlmError> {
    let bytes = std::fs::read(vocab_path.as_ref()).map_err(|err| {
        LlmError::ModelLoad(format!(
            "failed to read '{}': {err}",
            vocab_path.as_ref().display()
        ))
    })?;
    if bytes.len() < 4 {
        return Err(LlmError::ModelLoad("vocab.bin is truncated".to_owned()));
    }
    let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let words = parse_vocab_bin(&bytes, count)?;
    let vocab_u32 =
        u32::try_from(count).map_err(|_| LlmError::InvalidConfig("vocab_size exceeds u32"))?;
    let eos_u32 = TokenId::try_from(eos_token_id)
        .map_err(|_| LlmError::InvalidConfig("eos_token_id exceeds u32"))?;
    Ok(TokTrie::from(&TokRxInfo::new(vocab_u32, eos_u32), &words))
}

#[inline]
fn group_norm_and_bonus(
    y_head: &[f32],
    r_head: &[f32],
    k_hat_head: &[f32],
    v_head: &[f32],
    r_k_head: &[f32],
    ln_w_head: &[f32],
    ln_b_head: &[f32],
    eps: f32,
    out_head: &mut [f32],
) {
    let n = y_head.len().max(1) as f32;
    let mean = y_head.iter().copied().sum::<f32>() / n;
    let var = y_head
        .iter()
        .map(|&v| {
            let diff = v - mean;
            diff * diff
        })
        .sum::<f32>()
        / n;
    let inv_std = 1.0 / (var + eps).sqrt();

    let mut bonus_scalar = 0.0_f32;
    for j in 0..y_head.len() {
        bonus_scalar += r_head[j] * k_hat_head[j] * r_k_head[j];
    }

    for j in 0..y_head.len() {
        let normed = ((y_head[j] - mean) * inv_std) * ln_w_head[j] + ln_b_head[j];
        out_head[j] = normed + bonus_scalar * v_head[j];
    }
}

fn read_ln(st: &SafeTensors<'_>, prefix: &str, d: usize) -> Result<LayerNormWeights, LlmError> {
    Ok(LayerNormWeights {
        weight: read_f32_boxed(st, &format!("{prefix}.weight"), &[d])?,
        bias: read_f32_boxed(st, &format!("{prefix}.bias"), &[d])?,
    })
}

fn read_f32_boxed(
    st: &SafeTensors<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Box<[f32]>, LlmError> {
    let view = st
        .tensor(name)
        .map_err(|err| LlmError::ModelLoad(format!("missing tensor '{name}': {err}")))?;
    if view.dtype() != Dtype::F32 {
        return Err(LlmError::ModelLoad(format!(
            "tensor '{name}' has dtype {:?}, expected F32",
            view.dtype()
        )));
    }
    if view.shape() != expected_shape {
        return Err(LlmError::ModelLoad(format!(
            "tensor '{name}' has shape {:?}, expected {:?}",
            view.shape(),
            expected_shape
        )));
    }
    let expected_len: usize = expected_shape.iter().product();
    let mut values = Vec::with_capacity(expected_len);
    for chunk in view.data().chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values.into_boxed_slice())
}

#[inline]
fn layer_norm_into(x: &[f32], ln: &LayerNormWeights, eps: f32, out: &mut [f32]) {
    let n = x.len().max(1) as f32;
    let mean = x.iter().copied().sum::<f32>() / n;
    let var = x
        .iter()
        .map(|&v| {
            let diff = v - mean;
            diff * diff
        })
        .sum::<f32>()
        / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..x.len() {
        out[i] = ((x[i] - mean) * inv_std) * ln.weight[i] + ln.bias[i];
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use safetensors::tensor::TensorView;

    use super::*;

    fn to_le_bytes(data: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &val in data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    fn build_minimal_vocab_bin(vocab_size: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(vocab_size * 4);
        out.extend_from_slice(&(vocab_size as u32).to_le_bytes());

        // Token 0 = EOS/BOS (vazio, padrão RWKV World)
        out.extend_from_slice(&0_u16.to_le_bytes());

        // Tokens 1..=256 = todos os 256 bytes individuais (cobertura UTF-8 100%)
        let mut next_id = 1_usize;
        for b in 0_u8..=255_u8 {
            if next_id >= vocab_size {
                break;
            }
            out.extend_from_slice(&1_u16.to_le_bytes());
            out.push(b);
            next_id += 1;
        }

        // Tokens extras
        let mut idx = 0_usize;
        while next_id < vocab_size {
            let s = format!("t{idx}");
            let b = s.as_bytes();
            out.extend_from_slice(&(b.len() as u16).to_le_bytes());
            out.extend_from_slice(b);
            next_id += 1;
            idx += 1;
        }
        out
    }

    fn build_test_rwkv7_model(
        copy_dim: usize,
        quantize_emb: bool,
        zero_head: bool,
    ) -> Result<Rwkv7Llm, Box<dyn std::error::Error>> {
        fn push_f32(
            bufs: &mut Vec<(String, Vec<usize>, Dtype, Vec<u8>)>,
            name: &str,
            shape: Vec<usize>,
            val: f32,
        ) {
            let count: usize = shape.iter().product();
            bufs.push((
                name.to_owned(),
                shape,
                Dtype::F32,
                to_le_bytes(&vec![val; count]),
            ));
        }

        let cfg = Rwkv7Config {
            arch: "rwkv7".to_owned(),
            vocab_size: 260,
            d_model: 8,
            n_layers: 2,
            head_dim: 4,
            ffn_dim: 16,
            decay_lora_dim: 4,
            a_lora_dim: 4,
            v_lora_dim: 4,
            gate_lora_dim: 4,
            bos_token_id: 0,
            eos_token_id: 0,
            layer_norm_eps: 1e-5,
            group_norm_eps: 64e-5,
            vocab_file: "vocab.bin".to_owned(),
            copy_dim,
            isa_ptr_dim: 0,
            isa_num_vocab: 64,
            copy_max_len: 64,
            moe_num_experts: 0,
            moe_top_k: 2,
            moe_neuron_top_k: 0,
        };

        let mut bufs: Vec<(String, Vec<usize>, Dtype, Vec<u8>)> = Vec::new();

        if quantize_emb {
            let q = vec![10_i8.to_ne_bytes()[0]; cfg.vocab_size * cfg.d_model];
            let s = to_le_bytes(&vec![0.01_f32; cfg.vocab_size]);
            bufs.push((
                "emb.weight.q".to_owned(),
                vec![cfg.vocab_size, cfg.d_model],
                Dtype::I8,
                q,
            ));
            bufs.push((
                "emb.weight.scale".to_owned(),
                vec![cfg.vocab_size],
                Dtype::F32,
                s,
            ));
        } else {
            push_f32(
                &mut bufs,
                "emb.weight",
                vec![cfg.vocab_size, cfg.d_model],
                0.1,
            );
        }

        push_f32(&mut bufs, "ln_in.weight", vec![cfg.d_model], 1.0);
        push_f32(&mut bufs, "ln_in.bias", vec![cfg.d_model], 0.0);

        for i in 0..cfg.n_layers {
            let bp = format!("blocks.{i}");
            push_f32(
                &mut bufs,
                &format!("{bp}.ln1.weight"),
                vec![cfg.d_model],
                1.0,
            );
            push_f32(&mut bufs, &format!("{bp}.ln1.bias"), vec![cfg.d_model], 0.0);

            for name in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"] {
                push_f32(
                    &mut bufs,
                    &format!("{bp}.tm.{name}"),
                    vec![cfg.d_model],
                    0.5,
                );
            }
            for name in ["receptance", "key", "value", "output"] {
                push_f32(
                    &mut bufs,
                    &format!("{bp}.tm.{name}.weight"),
                    vec![cfg.d_model, cfg.d_model],
                    0.1,
                );
            }

            push_f32(&mut bufs, &format!("{bp}.tm.w0"), vec![cfg.d_model], 0.0);
            push_f32(
                &mut bufs,
                &format!("{bp}.tm.w1.weight"),
                vec![cfg.decay_lora_dim, cfg.d_model],
                0.05,
            );
            push_f32(
                &mut bufs,
                &format!("{bp}.tm.w2.weight"),
                vec![cfg.d_model, cfg.decay_lora_dim],
                0.05,
            );

            push_f32(&mut bufs, &format!("{bp}.tm.a0"), vec![cfg.d_model], 0.0);
            push_f32(
                &mut bufs,
                &format!("{bp}.tm.a1.weight"),
                vec![cfg.a_lora_dim, cfg.d_model],
                0.05,
            );
            push_f32(
                &mut bufs,
                &format!("{bp}.tm.a2.weight"),
                vec![cfg.d_model, cfg.a_lora_dim],
                0.05,
            );

            if i > 0 {
                push_f32(&mut bufs, &format!("{bp}.tm.v0"), vec![cfg.d_model], 0.0);
                push_f32(
                    &mut bufs,
                    &format!("{bp}.tm.v1.weight"),
                    vec![cfg.v_lora_dim, cfg.d_model],
                    0.05,
                );
                push_f32(
                    &mut bufs,
                    &format!("{bp}.tm.v2.weight"),
                    vec![cfg.d_model, cfg.v_lora_dim],
                    0.05,
                );
            }

            push_f32(
                &mut bufs,
                &format!("{bp}.tm.g1.weight"),
                vec![cfg.gate_lora_dim, cfg.d_model],
                0.05,
            );
            push_f32(
                &mut bufs,
                &format!("{bp}.tm.g2.weight"),
                vec![cfg.d_model, cfg.gate_lora_dim],
                0.05,
            );

            push_f32(&mut bufs, &format!("{bp}.tm.k_k"), vec![cfg.d_model], 0.85);
            push_f32(&mut bufs, &format!("{bp}.tm.k_a"), vec![cfg.d_model], 1.0);
            push_f32(&mut bufs, &format!("{bp}.tm.r_k"), vec![cfg.d_model], 0.1);
            push_f32(
                &mut bufs,
                &format!("{bp}.tm.ln_x.weight"),
                vec![cfg.d_model],
                1.0,
            );
            push_f32(
                &mut bufs,
                &format!("{bp}.tm.ln_x.bias"),
                vec![cfg.d_model],
                0.0,
            );

            push_f32(
                &mut bufs,
                &format!("{bp}.ln2.weight"),
                vec![cfg.d_model],
                1.0,
            );
            push_f32(&mut bufs, &format!("{bp}.ln2.bias"), vec![cfg.d_model], 0.0);
            push_f32(&mut bufs, &format!("{bp}.cm.x_k"), vec![cfg.d_model], 0.5);

            // Alterna sinais positivos/negativos em cm.key para exercitar o pulo de zeros do ReLU²
            let mut cm_key_vals = vec![0.0_f32; cfg.ffn_dim * cfg.d_model];
            for row in 0..cfg.ffn_dim {
                let sign = if row % 2 == 0 { 0.1 } else { -0.1 };
                for col in 0..cfg.d_model {
                    cm_key_vals[row * cfg.d_model + col] = sign;
                }
            }
            bufs.push((
                format!("{bp}.cm.key.weight"),
                vec![cfg.ffn_dim, cfg.d_model],
                Dtype::F32,
                to_le_bytes(&cm_key_vals),
            ));
            push_f32(
                &mut bufs,
                &format!("{bp}.cm.value.weight"),
                vec![cfg.d_model, cfg.ffn_dim],
                0.1,
            );
        }

        push_f32(&mut bufs, "ln_out.weight", vec![cfg.d_model], 1.0);
        push_f32(&mut bufs, "ln_out.bias", vec![cfg.d_model], 0.0);
        push_f32(
            &mut bufs,
            "head.weight",
            vec![cfg.vocab_size, cfg.d_model],
            if zero_head { 0.0 } else { 0.1 },
        );

        if copy_dim > 0 {
            push_f32(
                &mut bufs,
                "copy_gate.weight",
                vec![1, cfg.d_model + copy_dim],
                0.0,
            );
            push_f32(&mut bufs, "copy_gate.bias", vec![1], 10.0);

            let mut proj = vec![0.0_f32; copy_dim * cfg.d_model];
            for r in 0..copy_dim.min(cfg.d_model) {
                proj[r * cfg.d_model + r] = 1.0;
            }
            bufs.push((
                "copy_q.weight".to_owned(),
                vec![copy_dim, cfg.d_model],
                Dtype::F32,
                to_le_bytes(&proj),
            ));
            bufs.push((
                "copy_k.weight".to_owned(),
                vec![copy_dim, cfg.d_model],
                Dtype::F32,
                to_le_bytes(&proj),
            ));
        }

        let mut views = HashMap::with_capacity(bufs.len());
        for (name, shape, dtype, bytes) in &bufs {
            views.insert(
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes)?,
            );
        }

        let serialized = safetensors::serialize(views, &None)?;
        let vocab_bin = build_minimal_vocab_bin(cfg.vocab_size);
        Ok(Rwkv7Llm::from_parts(cfg, &vocab_bin, &serialized)?)
    }

    #[test]
    fn wkv7_delta_rule_overwrites_existing_key_and_preserves_orthogonal_keys() {
        let mut s_mat = [0.0_f32; 4 * 4];
        let mut ak = [0.0_f32; 4];
        let mut y = [0.0_f32; 4];

        let k1 = [1.0, 0.0, 0.0, 0.0];
        let k2 = [0.0, 1.0, 0.0, 0.0];
        let w_keep = [1.0, 1.0, 1.0, 1.0];
        let a_full = [1.0, 1.0, 1.0, 1.0];

        // Grava k1 -> v1_old e k2 -> v2
        let v1_old = [10.0, 20.0, 30.0, 40.0];
        let v2 = [-3.0, 7.0, 11.0, 13.0];
        wkv7_head_step(
            &mut s_mat, &k1, &w_keep, &k1, &v1_old, &k1, &a_full, &mut ak, &mut y,
        );
        wkv7_head_step(
            &mut s_mat, &k2, &w_keep, &k2, &v2, &k2, &a_full, &mut ak, &mut y,
        );

        // Sobrescreve k1 -> v1_new e consulta k1
        let v1_new = [99.0, -5.0, 1.0, 2.0];
        wkv7_head_step(
            &mut s_mat, &k1, &w_keep, &k1, &v1_new, &k1, &a_full, &mut ak, &mut y,
        );
        for (got, want) in y.iter().zip(v1_new.iter()) {
            assert!(
                (got - want).abs() < 1e-4,
                "k1 overwrite mismatch: {got} vs {want}"
            );
        }

        // Consulta k2 (sem gravar nada novo: a=0, k_hat=0) -> deve preservar v2 intacto!
        let zero = [0.0; 4];
        wkv7_head_step(
            &mut s_mat, &k2, &w_keep, &zero, &zero, &zero, &zero, &mut ak, &mut y,
        );
        for (got, want) in y.iter().zip(v2.iter()) {
            assert!((got - want).abs() < 1e-4, "k2 corrupted: {got} vs {want}");
        }
    }

    #[test]
    fn wkv7_head_step_64_matches_scalar_reference() {
        // n = 64 exercita o caminho de 8 lanes (o teste acima usa n = 4, só o resto escalar).
        let n = 64_usize;
        let wave = |seed: f32, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32) * 0.37 + seed).sin() * scale)
                .collect()
        };
        let r = wave(0.1, 1.0);
        let w_decay: Vec<f32> = wave(0.2, 0.2).iter().map(|v| 0.8 + v).collect();
        let k_hat = wave(0.3, 0.5);
        let v = wave(0.4, 1.0);
        let kk = wave(0.5, 0.125);
        let a: Vec<f32> = wave(0.6, 0.4).iter().map(|v| 0.5 + v).collect();

        let init: Vec<f32> = (0..n * n)
            .map(|i| ((i as f32) * 0.011).cos() * 0.3)
            .collect();
        let mut s_fast = init.clone();
        let mut s_ref = init;
        let mut ak = vec![0.0_f32; n];
        let mut y_fast = vec![0.0_f32; n];
        wkv7_head_step(
            &mut s_fast,
            &r,
            &w_decay,
            &k_hat,
            &v,
            &kk,
            &a,
            &mut ak,
            &mut y_fast,
        );

        // Referência escalar ingênua (ordem de soma sequencial).
        let mut y_ref = vec![0.0_f32; n];
        for i in 0..n {
            let mut sa = 0.0_f32;
            for j in 0..n {
                sa += s_ref[i * n + j] * kk[j];
            }
            let mut y_i = 0.0_f32;
            for j in 0..n {
                let up = s_ref[i * n + j] * w_decay[j] - sa * (a[j] * kk[j]) + v[i] * k_hat[j];
                s_ref[i * n + j] = up;
                y_i += up * r[j];
            }
            y_ref[i] = y_i;
        }

        for (got, want) in y_fast.iter().zip(&y_ref) {
            assert!((got - want).abs() < 1e-3, "y: {got} vs {want}");
        }
        for (got, want) in s_fast.iter().zip(&s_ref) {
            assert!((got - want).abs() < 1e-4, "state: {got} vs {want}");
        }
    }

    #[test]
    fn sparse_col_matrix_skips_zero_activations_and_matches_dense()
    -> Result<(), Box<dyn std::error::Error>> {
        let dense = [1.0_f32, 2.0, 3.0, 4.0, -1.0, 0.5, -2.0, 3.0];
        let sparse = SparseColQuantizedMatrix::from_row_major_f32(&dense, 2, 4)?;
        let act = [0.0_f32, 2.0, 0.0, 0.5];
        let mut out = [0.0_f32; 2];

        let active = sparse.accumulate_sparse(&act, &mut out);
        assert_eq!(active, 2);
        assert!((out[0] - 6.0).abs() < 0.15, "got {}", out[0]);
        assert!((out[1] - 2.5).abs() < 0.15, "got {}", out[1]);
        Ok(())
    }

    #[test]
    fn rwkv7_runs_constant_o1_state_with_both_f32_and_i8_embeddings()
    -> Result<(), Box<dyn std::error::Error>> {
        for quantize_emb in [false, true] {
            let model = build_test_rwkv7_model(0, quantize_emb, false)?;
            let mut state = Rwkv7State::new(model.config());
            let initial_bytes = state.state_bytes();

            for step in 0..50_usize {
                let tok = 1 + (step % 250);
                let logits = model.forward_step(tok, &mut state)?;
                assert_eq!(logits.len(), model.config().vocab_size);
                assert!(logits.iter().all(|v| v.is_finite()));
            }

            assert_eq!(state.state_bytes(), initial_bytes);
            assert!(state.total_ffn_cols > 0);
            // Metade das linhas de cm.key foi inicializada negativa -> esparsidade > 0%
            assert!(state.ffn_sparsity_ratio() > 0.0);
        }
        Ok(())
    }

    #[test]
    fn rwkv7_copy_gate_copies_exact_prompt_token_with_zero_vocab_head()
    -> Result<(), Box<dyn std::error::Error>> {
        let model = build_test_rwkv7_model(4, true, true)?;
        let mut state = Rwkv7State::new(model.config());

        state
            .scratch
            .x_out
            .copy_from_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        model.record_copy_entry(&mut state, 12);

        state
            .scratch
            .x_out
            .copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        model.record_copy_entry(&mut state, 42);
        assert_eq!(state.copy_bank_len(), 2);

        state
            .scratch
            .x_out
            .copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let mut mask = SimpleVob::alloc(model.config().vocab_size);
        mask.allow_token(12);
        mask.allow_token(42);

        let (winner, p_copy) = model
            .sample_with_copy_gate(&mut state, &mask)
            .ok_or("expected sampled token")?;
        assert_eq!(winner, 42);
        assert!(p_copy > 0.999, "expected open copy gate, got {p_copy}");
        assert!(state.scratch.copy_vocab_acc.iter().all(|&v| v == 0.0));
        Ok(())
    }

    #[test]
    fn rwkv7_chat_session_keeps_constant_state_across_turns_and_resets()
    -> Result<(), Box<dyn std::error::Error>> {
        let model = build_test_rwkv7_model(0, true, false)?;
        let mut session = model.start_chat(Some("Voce e um assistente tecnico."))?;
        let state_bytes_initial = session.state_bytes();
        let tokens_after_sys = session.total_tokens_seen();
        assert!(tokens_after_sys > 1);

        let sampling = ChatSamplingConfig {
            temperature: 0.0,
            top_p: 1.0,
            presence_penalty: 0.2,
            max_new_tokens: 6,
        };

        let mut streamed = String::new();
        let stats1 = session.send_message_streaming("Olá mundo!", sampling, |chunk| {
            streamed.push_str(chunk);
        })?;
        assert_eq!(stats1.state_bytes, state_bytes_initial);
        assert!(stats1.prompt_tokens > 0);
        assert!(session.total_tokens_seen() > tokens_after_sys);

        let stats2 = session.send_message_streaming("Segunda pergunta.", sampling, |_| {})?;
        assert_eq!(stats2.state_bytes, state_bytes_initial);

        session.reset(Some("Novo sistema"))?;
        assert_eq!(session.state_bytes(), state_bytes_initial);
        assert!(
            session.total_tokens_seen()
                < tokens_after_sys + stats1.prompt_tokens + stats2.prompt_tokens
        );
        Ok(())
    }

    #[test]
    fn rwkv7_rejects_invalid_vocab_and_out_of_range_token() -> Result<(), Box<dyn std::error::Error>>
    {
        let model = build_test_rwkv7_model(0, false, false)?;
        let mut state = Rwkv7State::new(model.config());

        let err = model.forward_hidden(999_999, &mut state);
        assert!(matches!(err, Err(LlmError::Inference(_))));

        let bad_vocab = parse_vocab_bin(&[1, 0], 260);
        assert!(matches!(bad_vocab, Err(LlmError::ModelLoad(_))));

        let mut wrong_logits = vec![0.0_f32; 4];
        let head_err = model.project_head_into(&mut state, &mut wrong_logits);
        assert!(matches!(head_err, Err(LlmError::InvalidConfig(_))));
        Ok(())
    }

    #[test]
    fn prefix_cache_reuses_multiturn_state_bit_exactly_and_resets_on_new_prompt()
    -> Result<(), Box<dyn std::error::Error>> {
        let model = build_test_rwkv7_model(4, true, false)?.with_max_new_tokens(8);
        let turn0 = "USER: Qual o preco_mensal de plano_enterprise?\n";
        let turn1 = "USER: Qual o preco_mensal de plano_enterprise?\nTOOL: {\"status\":\"found\",\"id\":\"fact-1\"}\n";
        let other = "USER: Outra conversa completamente diferente.\n";

        // 1. Turno 0: miss + grava snapshot do prefixo
        let _ = model.generate_action_from_prompt(turn0);
        let st0 = model.prefix_cache_stats();
        assert_eq!(st0.misses, 1);
        assert_eq!(st0.hits, 0);
        assert!(st0.tokens_prefilled > 0);

        // 2. Turno 1 (extensão do turno 0): hit reutilizando todos os tokens do turno 0
        let warm_res = model.generate_action_from_prompt(turn1);
        let st1 = model.prefix_cache_stats();
        assert_eq!(st1.hits, 1);
        assert!(
            st1.tokens_reused >= 20,
            "expected reused tokens, got {}",
            st1.tokens_reused
        );

        // 3. Paridade com prefill frio do turno 1
        model.clear_prefix_cache();
        let cold_res = model.generate_action_from_prompt(turn1);
        assert_eq!(format!("{warm_res:?}"), format!("{cold_res:?}"));

        // 4. Novo prompt independente: invalida automaticamente sem vazar estado
        let _ = model.generate_action_from_prompt(other);
        let st2 = model.prefix_cache_stats();
        assert_eq!(st2.misses, 3);
        Ok(())
    }

    #[test]
    fn forward_hidden_chunk4_matches_sequential_forward_hidden_and_state()
    -> Result<(), Box<dyn std::error::Error>> {
        let model = build_test_rwkv7_model(4, false, false)?;
        let d = model.cfg.d_model;
        let tokens = [3_usize, 11, 27, 5, 42, 19, 8, 31];

        // 1. Referência sequencial (1 token por vez)
        let mut state_seq = Rwkv7State::new(&model.cfg);
        model.forward_hidden(model.cfg.bos_token_id, &mut state_seq)?;
        let mut seq_outs = Vec::with_capacity(tokens.len());
        for &tok in &tokens {
            model.forward_hidden(tok, &mut state_seq)?;
            seq_outs.push(state_seq.scratch.x_out.to_vec());
        }

        // 2. Prefill em blocos de 4 (`forward_hidden_chunk4`)
        let mut state_c4 = Rwkv7State::new(&model.cfg);
        model.forward_hidden(model.cfg.bos_token_id, &mut state_c4)?;
        let mut c4_flat = vec![0.0_f32; tokens.len() * d];
        for chunk_idx in 0..2 {
            let base = chunk_idx * 4;
            let ids = [
                tokens[base],
                tokens[base + 1],
                tokens[base + 2],
                tokens[base + 3],
            ];
            let (s0, rest) = c4_flat[base * d..(base + 4) * d].split_at_mut(d);
            let (s1, rest) = rest.split_at_mut(d);
            let (s2, s3) = rest.split_at_mut(d);
            model.forward_hidden_chunk4(ids, &mut state_c4, [s0, s1, s2, s3])?;
        }

        // Todos os 8 `x_out` devem bater
        for (idx, expected) in seq_outs.iter().enumerate() {
            let got = &c4_flat[idx * d..(idx + 1) * d];
            for (c, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (g - e).abs() < 1e-4,
                    "token {idx} dim {c}: chunk4={g} vs seq={e}"
                );
            }
        }

        // `scratch.x_out` final deve bater com o último token
        for (c, (&g, &e)) in state_c4
            .scratch
            .x_out
            .iter()
            .zip(state_seq.scratch.x_out.iter())
            .enumerate()
        {
            assert!((g - e).abs() < 1e-4, "final x_out dim {c}: {g} vs {e}");
        }

        // Todos os estados recorrentes das camadas devem bater
        for (l, (lc4, lseq)) in state_c4
            .layers
            .iter()
            .zip(state_seq.layers.iter())
            .enumerate()
        {
            for (i, (&g, &e)) in lc4.state_mat.iter().zip(lseq.state_mat.iter()).enumerate() {
                assert!((g - e).abs() < 1e-4, "layer {l} state_mat[{i}]: {g} vs {e}");
            }
            for (i, (&g, &e)) in lc4.prev_x_tm.iter().zip(lseq.prev_x_tm.iter()).enumerate() {
                assert!((g - e).abs() < 1e-4, "layer {l} prev_x_tm[{i}]: {g} vs {e}");
            }
            for (i, (&g, &e)) in lc4.prev_x_cm.iter().zip(lseq.prev_x_cm.iter()).enumerate() {
                assert!((g - e).abs() < 1e-4, "layer {l} prev_x_cm[{i}]: {g} vs {e}");
            }
        }
        Ok(())
    }
}
