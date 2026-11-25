pub mod cache;

pub use crate::facts::cache::FactCache;

/// Limite máximo de fatos retornados por chamada de `search_facts`.
pub const MAX_SEARCH_LIMIT: u8 = 50;

/// Normaliza chaves de entidade e relação para buscas determinísticas case-insensitive.
#[must_use]
pub fn normalize_key(value: &str) -> String {
    value.trim().to_lowercase()
}
