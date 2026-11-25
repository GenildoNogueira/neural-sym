pub mod error;
pub mod facts;
#[cfg(feature = "tantivy-search")]
pub mod search;
pub mod store;
pub mod session_doc;

pub use error::MemoryError;
pub use session_doc::SessionDocumentStore;
#[cfg(feature = "redb")]
pub use store::RedbMemory;

#[cfg(feature = "tantivy-search")]
pub use search::{IndexedRedbMemory, TantivyFactIndex};
