use ns_core::error::MemoryPortError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[cfg(feature = "redb")]
    #[error("redb operation failed")]
    Redb(#[source] Box<redb::Error>),

    #[cfg(feature = "tantivy-search")]
    #[error("tantivy search index operation failed")]
    Search(#[source] Box<tantivy::TantivyError>),

    #[error("search writer lock poisoned")]
    LockPoisoned,

    #[error("fact id already assigned to a different (subject, relation): {0}")]
    DuplicateFactId(String),

    #[error("invalid fact field: {field} must not be empty")]
    InvalidField { field: &'static str },
}

#[cfg(feature = "redb")]
impl From<redb::Error> for MemoryError {
    fn from(err: redb::Error) -> Self {
        Self::Redb(Box::new(err))
    }
}

#[cfg(feature = "tantivy-search")]
impl From<tantivy::TantivyError> for MemoryError {
    fn from(err: tantivy::TantivyError) -> Self {
        Self::Search(Box::new(err))
    }
}

impl From<MemoryError> for MemoryPortError {
    fn from(err: MemoryError) -> Self {
        Self::Storage(err.to_string())
    }
}
