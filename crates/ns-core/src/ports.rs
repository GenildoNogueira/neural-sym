use std::future::Future;

use rust_decimal::Decimal;

use crate::action::LlmAction;
use crate::error::{LlmPortError, MemoryPortError, VerifierPortError};
use crate::fact::{DocumentSpan, Fact, FactId, Message};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogicProof {
    Satisfiable { model_summary: String },
    Unsatisfiable { unsat_core: String },
    Unknown { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxOutput {
    pub stdout: String,
    pub fuel_consumed: u64,
    pub passed: bool,
}

/// Contrato da camada de inferência na VRAM com constrained decoding.
pub trait ConstrainedLlm: Send + Sync {
    fn next_action(
        &self,
        history: &[Message],
    ) -> impl Future<Output = Result<LlmAction, LlmPortError>> + Send;
}

/// Contrato da Memória Determinística (SQLite, redb, Tantivy + cache em RAM).
pub trait DeterministicMemory: Send + Sync {
    fn lookup_fact(
        &self,
        entity: &str,
        relation: &str,
    ) -> impl Future<Output = Result<Option<Fact>, MemoryPortError>> + Send;

    fn search_facts(
        &self,
        query: &str,
        limit: u8,
    ) -> impl Future<Output = Result<Vec<Fact>, MemoryPortError>> + Send;

    fn traverse(
        &self,
        start_entity: &str,
        max_hops: u8,
    ) -> impl Future<Output = Result<Vec<Fact>, MemoryPortError>> + Send;

    fn verify_fact_exists(
        &self,
        id: &FactId,
    ) -> impl Future<Output = Result<bool, MemoryPortError>> + Send;
}

/// Contrato do Motor Simbólico (matemática exata, SMT solver e sandbox).
pub trait SymbolicVerifier: Send + Sync {
    fn eval_decimal(&self, expression: &str) -> Result<Decimal, VerifierPortError>;

    fn check_smt(&self, smtlib2_script: &str) -> Result<LogicProof, VerifierPortError>;

    fn run_sandbox(
        &self,
        wat_source: &str,
        expected_stdout: &str,
    ) -> Result<SandboxOutput, VerifierPortError>;
}

/// Contrato da memória efêmera em RAM para documentos longos anexados à sessão.
pub trait DocumentContext: Send + Sync {
    /// Indica se não há nenhum documento indexado na sessão.
    #[must_use]
    fn is_empty(&self) -> bool;

    /// Sumário compacto (IDs, títulos e remissões) enviado ao LLM no lugar do texto integral.
    #[must_use]
    fn table_of_contents(&self) -> String;

    /// Busca seções relevantes por termo ou consulta natural.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryPortError`] se a consulta ao índice em RAM falhar.
    fn search_spans(
        &self,
        query: &str,
        limit: u8,
    ) -> Result<Vec<DocumentSpan>, MemoryPortError>;

    /// Retorna as seções solicitadas na ordem pedida para inspeção lado a lado.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryPortError`] se a leitura dos trechos falhar.
    fn read_spans(&self, span_ids: &[String]) -> Result<Vec<DocumentSpan>, MemoryPortError>;
}

/// Implementação nula (zero custo) usada quando não há documento longo anexado.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoDocumentContext;

impl DocumentContext for NoDocumentContext {
    fn is_empty(&self) -> bool {
        true
    }

    fn table_of_contents(&self) -> String {
        String::new()
    }

    fn search_spans(
        &self,
        _query: &str,
        _limit: u8,
    ) -> Result<Vec<DocumentSpan>, MemoryPortError> {
        Ok(Vec::new())
    }

    fn read_spans(&self, _span_ids: &[String]) -> Result<Vec<DocumentSpan>, MemoryPortError> {
        Ok(Vec::new())
    }
}
