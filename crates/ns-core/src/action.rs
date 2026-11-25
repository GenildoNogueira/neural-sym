use schemars::{JsonSchema, schema::RootSchema, schema_for};
use serde::{Deserialize, Serialize};

use crate::fact::FactId;

const fn default_search_limit() -> u8 {
    10
}

/// Únicas ações que o LLM na VRAM tem permissão gramatical para emitir.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum LlmAction {
    QueryFact {
        entity: String,
        relation: String,
    },
    /// Busca fatos por termo parcial (case-insensitive) em subject, relation ou object.
    SearchFacts {
        query: String,
        #[serde(default = "default_search_limit")]
        limit: u8,
    },
    /// Busca seções ou cláusulas relevantes no documento anexado à sessão (BM25 + fallback literal).
    SearchContext {
        query: String,
        limit: u8,
    },
    /// Lê uma ou mais seções/cláusulas completas lado a lado para comparação direta.
    ReadSpans {
        span_ids: Vec<String>,
    },
    TraverseGraph {
        start_entity: String,
        max_hops: u8,
    },
    EvalMath {
        expression: String,
    },
    CheckLogic {
        smtlib2_script: String,
    },
    RunWasmCheck {
        wasm_wat_source: String,
        expected_stdout: String,
    },
    ProposeAnswer {
        text: String,
        cited_facts: Vec<FactId>,
    },
}

impl LlmAction {
    /// Gera o JSON Schema que alimenta o constrained decoding (ex: llguidance).
    pub fn json_schema() -> RootSchema {
        schema_for!(Self)
    }
}
