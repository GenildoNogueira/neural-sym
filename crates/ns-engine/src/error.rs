use ns_core::error::{LlmPortError, MemoryPortError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("user prompt must not be empty")]
    EmptyPrompt,

    #[error("llm action generation failed")]
    Llm(#[from] LlmPortError),

    #[error("deterministic memory lookup failed")]
    Memory(#[from] MemoryPortError),

    #[error("failed to serialize tool response payload")]
    Serialization(#[from] serde_json::Error),

    #[error("orchestrator exceeded maximum of {max_steps} steps after {repair_attempts} repair attempts")]
    MaxStepsExceeded { max_steps: u8, repair_attempts: u8 },
}
