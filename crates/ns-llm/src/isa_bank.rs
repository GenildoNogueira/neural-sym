use std::collections::HashMap;

use serde_json::Value;

use crate::isa::{Bank, Expect, Op, Slot};

/// Origem de cada entidade no banco — torna IDs inventados irrepresentáveis (Peça C).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlotKind {
    PromptEntity,
    DocumentToc,
    ReadSpan,
    VerifiedFact,
    CandidateFact,
}

impl SlotKind {
    #[must_use]
    pub const fn is_citable(self) -> bool {
        matches!(
            self,
            Self::ReadSpan | Self::VerifiedFact | Self::CandidateFact
        )
    }

    #[must_use]
    pub const fn is_readable_span(self) -> bool {
        matches!(self, Self::DocumentToc | Self::ReadSpan)
    }
}

/// Filtro de admissibilidade de ponteiros por posição da gramática ISA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotFilter {
    Any,
    DocumentSpan,
    CitableProof,
}

impl SlotFilter {
    #[must_use]
    pub const fn for_op(op: Op, expect: Expect) -> Self {
        match (op, expect) {
            (Op::ProposeAnswer, Expect::RefOrEnd) => Self::CitableProof,
            (Op::ReadSpans, Expect::RefOrEnd) => Self::DocumentSpan,
            _ => Self::Any,
        }
    }

    #[must_use]
    pub const fn allows_literal_fallback(self, strict_citations: bool) -> bool {
        match self {
            Self::Any => true,
            Self::DocumentSpan | Self::CitableProof => !strict_citations,
        }
    }

    #[must_use]
    pub const fn allows_kind(self, kind: SlotKind) -> bool {
        match self {
            Self::Any => true,
            Self::DocumentSpan => kind.is_readable_span(),
            Self::CitableProof => kind.is_citable(),
        }
    }
}

/// Entrada neural de 1 entidade: texto canônico, origem e chave projetada `K_s` (`len == ptr_dim`).
#[derive(Debug, Clone, PartialEq)]
pub struct NeuralEntitySlot {
    pub text: String,
    pub kind: SlotKind,
    pub ptr_key: Box<[f32]>,
}

/// Banco neural tipado montado durante o prefill para decodificação ISA com ponteiros de entidade.
#[derive(Debug, Clone)]
pub struct NeuralEntityBank {
    ptr_dim: usize,
    slots: Vec<NeuralEntitySlot>,
    by_text: HashMap<String, Slot>,
}

impl NeuralEntityBank {
    #[must_use]
    pub fn new(ptr_dim: usize) -> Self {
        Self {
            ptr_dim,
            slots: Vec::with_capacity(64),
            by_text: HashMap::with_capacity(64),
        }
    }

    pub fn upsert(&mut self, text: &str, kind: SlotKind, ptr_key: Box<[f32]>) -> Option<Slot> {
        debug_assert_eq!(ptr_key.len(), self.ptr_dim);
        if let Some(&existing) = self.by_text.get(text) {
            if let Some(slot) = self.slots.get_mut(usize::from(existing)) {
                if kind.is_citable() || !slot.kind.is_citable() {
                    slot.kind = kind;
                }
                slot.ptr_key = ptr_key;
            }
            return Some(existing);
        }

        let slot_id = Slot::try_from(self.slots.len()).ok()?;
        let owned = text.to_owned();
        self.slots.push(NeuralEntitySlot {
            text: owned.clone(),
            kind,
            ptr_key,
        });
        self.by_text.insert(owned, slot_id);
        Some(slot_id)
    }

    #[must_use]
    pub fn get(&self, slot: Slot) -> Option<&NeuralEntitySlot> {
        self.slots.get(usize::from(slot))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Converte para o [`Bank`] textual do `isa.rs` preservando exatamente os índices `0..len`.
    #[must_use]
    pub fn to_text_bank(&self) -> Bank {
        let mut bank = Bank::new();
        for slot in &self.slots {
            let _ = bank.intern(&slot.text);
        }
        bank
    }

    /// Retorna o melhor `(Slot, score)` entre os slots permitidos por `filter`.
    #[must_use]
    pub fn argmax_pointer(&self, query_ptr: &[f32], filter: SlotFilter) -> Option<(Slot, f32)> {
        debug_assert_eq!(query_ptr.len(), self.ptr_dim);
        let inv_sqrt = 1.0 / (self.ptr_dim.max(1) as f32).sqrt();

        let mut best: Option<(Slot, f32)> = None;
        for (idx, entry) in self.slots.iter().enumerate() {
            if !filter.allows_kind(entry.kind) {
                continue;
            }
            let Ok(slot_id) = Slot::try_from(idx) else {
                break;
            };
            let mut dot = 0.0_f32;
            for (&q, &k) in query_ptr.iter().zip(entry.ptr_key.iter()) {
                dot += q * k;
            }
            let score = dot * inv_sqrt;
            if best.is_none_or(|(_, prev)| score > prev) {
                best = Some((slot_id, score));
            }
        }
        best
    }
}

/// Extrai todas as entidades endereçáveis do prompt com sua origem [`SlotKind`],
/// mantendo ordem determinística 1:1 com `extract_prompt_entities` no Python.
#[must_use]
pub fn extract_prompt_entities(prompt: &str) -> Vec<(String, SlotKind)> {
    let mut order: Vec<String> = Vec::with_capacity(32);
    let mut kinds: HashMap<String, SlotKind> = HashMap::with_capacity(32);

    let mut register = |raw: &str, kind: SlotKind| {
        let s = raw.trim();
        if s.is_empty() {
            return;
        }
        if let Some(prev) = kinds.get_mut(s) {
            if kind.is_citable() || !prev.is_citable() {
                *prev = kind;
            }
        } else {
            let owned = s.to_owned();
            kinds.insert(owned.clone(), kind);
            order.push(owned);
        }
    };

    for line in prompt.lines() {
        let trimmed = line.trim();
        let maybe_candidates = trimmed
            .strip_prefix("SYSTEM:")
            .map(str::trim)
            .unwrap_or(trimmed)
            .strip_prefix("CANDIDATES:");

        if let Some(json_part) = maybe_candidates {
            if let Ok(Value::Array(facts)) = serde_json::from_str::<Value>(json_part.trim()) {
                for fact in &facts {
                    register_fact_object(fact, SlotKind::CandidateFact, &mut register);
                }
            }
        } else if let Some(json_part) = trimmed.strip_prefix("TOOL:") {
            if let Ok(val) = serde_json::from_str::<Value>(json_part.trim()) {
                collect_tool_entities(&val, &mut register);
            }
        } else if let Some(user_part) = trimmed.strip_prefix("USER:") {
            for raw_word in user_part.split_whitespace() {
                let w = raw_word.trim_matches(|c: char| {
                    matches!(
                        c,
                        '.' | ',' | '?' | '!' | ';' | ':' | '(' | ')' | '"' | '\''
                    )
                });
                if w.len() >= 2 {
                    register(w, SlotKind::PromptEntity);
                }
            }
        } else if let Some(rest) = trimmed.strip_prefix('[') {
            if let Some(end_bracket) = rest.find(']') {
                let span_id = &rest[..end_bracket];
                if !span_id.is_empty() && !span_id.contains(char::is_whitespace) {
                    register(span_id, SlotKind::DocumentToc);
                }
            }
        }
    }

    order
        .into_iter()
        .filter_map(|text| kinds.get(&text).copied().map(|kind| (text, kind)))
        .collect()
}

fn collect_tool_entities(val: &Value, register: &mut impl FnMut(&str, SlotKind)) {
    let Some(obj) = val.as_object() else {
        return;
    };

    if let Some(fact) = obj.get("fact") {
        register_fact_object(fact, SlotKind::VerifiedFact, register);
    }
    if let Some(facts) = obj.get("facts").and_then(Value::as_array) {
        for fact in facts {
            register_fact_object(fact, SlotKind::VerifiedFact, register);
        }
    }
    if let Some(spans) = obj.get("spans").and_then(Value::as_array) {
        for span in spans {
            if let Some(id) = span.get("id").and_then(Value::as_str) {
                register(id, SlotKind::ReadSpan);
            }
        }
    }
    if let Some(matches) = obj.get("matches").and_then(Value::as_array) {
        for item in matches {
            if let Some(id) = item.get("id").and_then(Value::as_str) {
                register(id, SlotKind::DocumentToc);
            }
        }
    }
}

fn register_fact_object(
    fact: &Value,
    id_kind: SlotKind,
    register: &mut impl FnMut(&str, SlotKind),
) {
    let Some(map) = fact.as_object() else {
        return;
    };
    if let Some(id) = map.get("id").and_then(Value::as_str) {
        register(id, id_kind);
    }
    for key in ["subject", "relation", "object"] {
        if let Some(v) = map.get(key).and_then(Value::as_str) {
            register(v, SlotKind::PromptEntity);
        }
    }
}

/// Mapeia o último intervalo de bytes de `needle` em `haystack` para os índices de token `[t0, t1)`
/// usando os offsets finais cumulativos `token_byte_ends`.
#[must_use]
pub fn find_entity_token_span(
    haystack: &str,
    needle: &str,
    token_byte_ends: &[usize],
) -> Option<(usize, usize)> {
    let byte_start = haystack.rfind(needle)?;
    let byte_end = byte_start.saturating_add(needle.len());
    let mut t0: Option<usize> = None;
    let mut t1: usize = 0;

    let mut prev_end = 0_usize;
    for (idx, &end) in token_byte_ends.iter().enumerate() {
        if end > byte_start && prev_end < byte_end {
            if t0.is_none() {
                t0 = Some(idx);
            }
            t1 = idx.saturating_add(1);
        }
        prev_end = end;
    }

    t0.map(|start| (start, t1.max(start + 1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_filter_blocks_unverified_prompt_entities_and_unread_toc() {
        let mut bank = NeuralEntityBank::new(2);
        let _ = bank.upsert("plano_pro", SlotKind::PromptEntity, Box::new([10.0, 0.0]));
        let _ = bank.upsert("cl_02", SlotKind::DocumentToc, Box::new([9.0, 0.0]));
        let _ = bank.upsert("fact-real-01", SlotKind::VerifiedFact, Box::new([1.0, 0.0]));

        // Query apontando fortemente para [1, 0]: em CitableProof, só fact-real-01 é visível!
        let (slot, _) = bank
            .argmax_pointer(&[1.0, 0.0], SlotFilter::CitableProof)
            .unwrap_or((u16::MAX, 0.0));
        assert_eq!(slot, 2);
        assert_eq!(
            bank.get(slot).map(|s| s.text.as_str()),
            Some("fact-real-01")
        );

        // Em DocumentSpan, cl_02 vence e plano_pro / fact-real-01 ficam invisíveis
        let (span_slot, _) = bank
            .argmax_pointer(&[1.0, 0.0], SlotFilter::DocumentSpan)
            .unwrap_or((u16::MAX, 0.0));
        assert_eq!(span_slot, 1);
    }

    #[test]
    fn extract_prompt_entities_classifies_tool_facts_and_spans() {
        let prompt = concat!(
            "SYSTEM: CONTEXTO INDEXADO\n",
            "[cl_02] Clausula 2\n",
            "USER: Qual o preco_mensal de plano_pro?\n",
            "TOOL: {\"status\":\"found\",\"fact\":{\"id\":\"fact-preco-001\",\"subject\":\"plano_pro\",\"relation\":\"preco_mensal\",\"object\":\"199.90\"}}\n",
            "TOOL: {\"status\":\"ok\",\"spans\":[{\"id\":\"cl_02\",\"text\":\"Prazo 15 dias\"}]}\n"
        );
        let entities = extract_prompt_entities(prompt);
        let map: HashMap<String, SlotKind> = entities.into_iter().collect();

        assert_eq!(map.get("fact-preco-001"), Some(&SlotKind::VerifiedFact));
        assert_eq!(map.get("cl_02"), Some(&SlotKind::ReadSpan)); // promovido de DocumentToc -> ReadSpan
        assert_eq!(map.get("plano_pro"), Some(&SlotKind::PromptEntity));
    }
}
