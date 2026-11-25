use ns_core::action::LlmAction;
use serde_json::Value;

use crate::error::LlmError;

/// Retorna o JSON Schema de [`LlmAction`] como [`serde_json::Value`].
///
/// # Errors
///
/// Retorna [`LlmError::SchemaSerialization`] se a conversão do schema falhar.
pub fn action_schema_value() -> Result<Value, LlmError> {
    let schema = LlmAction::json_schema();
    Ok(serde_json::to_value(&schema)?)
}

/// Serializa o JSON Schema de [`LlmAction`] em string formatada.
///
/// # Errors
///
/// Retorna [`LlmError::SchemaSerialization`] se a serialização falhar.
pub fn action_schema_json() -> Result<String, LlmError> {
    let schema = LlmAction::json_schema();
    Ok(serde_json::to_string_pretty(&schema)?)
}
