use thiserror::Error;

use crate::fact::FactId;

#[derive(Debug, Error)]
pub enum VerificationError {
    #[error("answer cited facts not retrieved from deterministic memory: {missing:?}")]
    UncitedOrHallucinatedFacts { missing: Vec<FactId> },
}

#[derive(Debug, Error)]
pub enum MemoryPortError {
    #[error("deterministic storage query failed: {0}")]
    Storage(String),
}

#[derive(Debug, Error)]
pub enum LlmPortError {
    #[error("constrained decoding failed: {0}")]
    Decoding(String),
}

#[derive(Debug, Error)]
pub enum VerifierPortError {
    #[error("invalid expression syntax: {0}")]
    Syntax(String),

    #[error("arithmetic overflow or division by zero")]
    Arithmetic,

    #[error("sandbox execution exceeded resource limits")]
    ResourceLimit,
}
