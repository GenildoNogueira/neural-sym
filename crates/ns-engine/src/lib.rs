pub mod audit;
pub mod dataset;
pub mod error;
pub mod orchestrator;

pub use audit::{AuditOutcome, AuditStep, AuditTrail, VerifiedOutcome};
pub use dataset::{
    TrainingSample, extract_verified_samples, format_history_prompt, write_samples_jsonl,
};
pub use error::EngineError;
pub use orchestrator::Orchestrator;
