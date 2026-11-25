use std::path::{Path, PathBuf};

use clap::ValueEnum;

/// Motores de persistência determinística suportados pelo CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum StorageBackend {
    Redb,
    /// `redb` (verdade determinística ACID) + índice invertido Tantivy (BM25 + FST)
    IndexedRedb,
}

impl StorageBackend {
    #[must_use]
    pub const fn default_filename(self) -> &'static str {
        match self {
            Self::Redb | Self::IndexedRedb => "neurosym.redb",
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Redb => "redb",
            Self::IndexedRedb => "indexed-redb",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppConfig {
    pub backend: StorageBackend,
    pub db_path: PathBuf,
    pub index_dir: PathBuf,
}

impl AppConfig {
    #[must_use]
    pub fn resolve(
        backend: StorageBackend,
        db_path: Option<PathBuf>,
        index_dir: Option<PathBuf>,
    ) -> Self {
        let resolved_db = db_path.unwrap_or_else(|| PathBuf::from(backend.default_filename()));
        let resolved_index = index_dir.unwrap_or_else(|| default_index_dir(&resolved_db));
        Self {
            backend,
            db_path: resolved_db,
            index_dir: resolved_index,
        }
    }
}

fn default_index_dir(db_path: &Path) -> PathBuf {
    let mut os = db_path.as_os_str().to_os_string();
    os.push(".tantivy");
    PathBuf::from(os)
}
