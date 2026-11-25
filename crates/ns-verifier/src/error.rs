use ns_core::error::VerifierPortError;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierError {
    #[error("expression must not be empty")]
    EmptyExpression,

    #[error("expression exceeds maximum length of {max} bytes")]
    ExpressionTooLong { max: usize },

    #[error("expression exceeds maximum nesting depth of {max}")]
    MaxDepthExceeded { max: usize },

    #[error("unexpected character '{ch}' at byte {position}")]
    UnexpectedCharacter { ch: char, position: usize },

    #[error("invalid decimal literal '{literal}' at byte {position}")]
    InvalidNumber { literal: String, position: usize },

    #[error("unexpected token '{token}' at byte {position}")]
    UnexpectedToken { token: &'static str, position: usize },

    #[error("unexpected end of expression")]
    UnexpectedEndOfInput,

    #[error("unclosed parenthesis opened at byte {position}")]
    UnclosedParenthesis { position: usize },

    #[error("division or modulo by zero")]
    DivisionByZero,

    #[error("decimal arithmetic overflow")]
    Overflow,

    #[error("exponent must be an exact integer")]
    NonIntegerExponent,

    #[error("exponent magnitude exceeds limit of {max}")]
    ExponentOutOfRange { max: u64 },

    #[error("invalid smt-lib2 script: {message}")]
    SmtSyntax { message: String },

    #[error("smt solver timed out after {timeout_ms}ms")]
    SmtTimeout { timeout_ms: u64 },

    #[error("failed to compile wat/wasm module: {message}")]
    SandboxCompile { message: String },

    #[error("wasm sandbox exhausted fuel limit of {limit} instructions")]
    SandboxOutOfFuel { limit: u64 },

    #[error("wasm sandbox exceeded memory limit of {max_bytes} bytes")]
    SandboxMemoryLimit { max_bytes: usize },

    #[error("wasm sandbox exceeded stdout limit of {max_bytes} bytes")]
    SandboxStdoutLimit { max_bytes: usize },

    #[error("wasm module must export a '_start' or 'run' function with signature () -> ()")]
    SandboxMissingEntrypoint,

    #[error("wasm sandbox trapped during execution: {message}")]
    SandboxTrap { message: String },

    #[error("optional verifier feature '{feature}' is not enabled")]
    FeatureNotEnabled { feature: &'static str },
}

impl From<VerifierError> for VerifierPortError {
    fn from(err: VerifierError) -> Self {
        match err {
            VerifierError::DivisionByZero
            | VerifierError::Overflow
            | VerifierError::NonIntegerExponent => Self::Arithmetic,

            VerifierError::ExpressionTooLong { .. }
            | VerifierError::MaxDepthExceeded { .. }
            | VerifierError::ExponentOutOfRange { .. }
            | VerifierError::SmtTimeout { .. }
            | VerifierError::SandboxOutOfFuel { .. }
            | VerifierError::SandboxMemoryLimit { .. }
            | VerifierError::SandboxStdoutLimit { .. }
            | VerifierError::FeatureNotEnabled { .. } => Self::ResourceLimit,

            VerifierError::EmptyExpression
            | VerifierError::UnexpectedCharacter { .. }
            | VerifierError::InvalidNumber { .. }
            | VerifierError::UnexpectedToken { .. }
            | VerifierError::UnexpectedEndOfInput
            | VerifierError::UnclosedParenthesis { .. }
            | VerifierError::SmtSyntax { .. }
            | VerifierError::SandboxCompile { .. }
            | VerifierError::SandboxMissingEntrypoint
            | VerifierError::SandboxTrap { .. } => Self::Syntax(err.to_string()),
        }
    }
}
