use ns_core::error::LlmPortError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("failed to serialize action json schema")]
    SchemaSerialization(#[from] serde_json::Error),

    #[error("invalid llama configuration: {0}")]
    InvalidConfig(&'static str),

    #[error("failed to initialize llama backend: {0}")]
    BackendInit(String),

    #[error("failed to load rwkv-7 model: {0}")]
    ModelLoad(String),

    #[error("failed to create llama context: {0}")]
    ContextInit(String),

    #[error("failed to compile llguidance grammar: {0}")]
    GrammarInit(String),

    #[error("prompt requires {tokens} tokens, exceeding context window of {limit}")]
    ContextExceeded { tokens: usize, limit: usize },

    #[error("llama inference failed: {0}")]
    Inference(String),

    #[error("llama worker thread terminated unexpectedly")]
    WorkerClosed,

    #[error("llm port operation failed")]
    Port(#[from] LlmPortError),
}

impl From<LlmError> for LlmPortError {
    fn from(err: LlmError) -> Self {
        Self::Decoding(err.to_string())
    }
}
