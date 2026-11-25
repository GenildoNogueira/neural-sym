use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Identificador único e auditável de um fato armazenado na memória determinística.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
pub struct FactId(String);

impl FactId {
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for FactId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl From<String> for FactId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for FactId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl fmt::Display for FactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Tripla determinística `(subject, relation) -> object` com rastreabilidade de origem.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Fact {
    pub id: FactId,
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub source_ref: String,
}

/// Papel de cada mensagem trocada no histórico enviado ao LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Mensagem estruturada do histórico mantido pelo orquestrador.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
        }
    }
}

/// Trecho estruturado (cláusula, artigo ou seção) extraído de um documento longo na sessão.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentSpan {
    pub id: FactId,
    pub title: String,
    pub text: String,
    /// IDs de outras seções citadas explicitamente neste trecho (ex.: `"cl_02"`).
    pub references: Vec<FactId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact_id_preserves_value_and_display() {
        let id = FactId::from("fact-preco-01");
        assert_eq!(id.as_str(), "fact-preco-01");
        assert_eq!(id.as_ref(), "fact-preco-01");
        assert_eq!(id.to_string(), "fact-preco-01");
    }

    #[test]
    fn message_constructors_assign_expected_roles() {
        assert_eq!(Message::system("regra").role, Role::System);
        assert_eq!(Message::user("pergunta").role, Role::User);
        assert_eq!(Message::assistant("acao").role, Role::Assistant);
        assert_eq!(Message::tool("retorno").role, Role::Tool);
    }
}
