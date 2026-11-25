use std::path::Path;

use anyhow::{Context, Result};
use ns_core::action::LlmAction;
use ns_core::error::{LlmPortError, MemoryPortError};
use ns_core::fact::{Fact, FactId, Message};
use ns_core::ports::{ConstrainedLlm, DeterministicMemory};
use ns_llm::MockLlm;
#[cfg(feature = "custom-rwkv")]
use ns_llm::Rwkv7Llm;
#[cfg(feature = "tantivy-search")]
use ns_memory::IndexedRedbMemory;
use ns_memory::RedbMemory;

use crate::config::StorageBackend;

/// Despachante estático entre `redb` puro e `redb` + índice invertido Tantivy.
#[derive(Debug)]
pub enum MemoryBackend {
    Redb(RedbMemory),
    #[cfg(feature = "tantivy-search")]
    IndexedRedb(IndexedRedbMemory),
}

impl MemoryBackend {
    /// Abre o banco escolhido, criando diretórios pais ausentes.
    ///
    /// # Errors
    ///
    /// Retorna erro contextualizado caso a criação de diretório, abertura do banco
    /// ou inicialização do índice Tantivy falhe.
    pub async fn open(backend: StorageBackend, db_path: &Path, index_dir: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create database directory '{}'", parent.display())
            })?;
        }

        match backend {
            StorageBackend::Redb => {
                let memory = RedbMemory::open_file(db_path).with_context(|| {
                    format!("failed to open redb database at '{}'", db_path.display())
                })?;
                Ok(Self::Redb(memory))
            }
            StorageBackend::IndexedRedb => open_indexed_redb(db_path, index_dir),
        }
    }

    /// Grava um lote de fatos de forma atômica no backend ativo.
    ///
    /// # Errors
    ///
    /// Retorna erro contextualizado caso a gravação em lote ou indexação falhe.
    pub async fn upsert_batch(&self, facts: &[Fact]) -> Result<usize> {
        match self {
            Self::Redb(memory) => memory
                .upsert_batch(facts)
                .context("failed to upsert fact batch in redb"),
            #[cfg(feature = "tantivy-search")]
            Self::IndexedRedb(memory) => memory
                .upsert_batch(facts)
                .context("failed to upsert fact batch in indexed-redb"),
        }
    }

    /// Carrega todos os fatos do backend ativo para geração de trajetórias.
    ///
    /// # Errors
    ///
    /// Retorna erro contextualizado caso a leitura falhe.
    pub async fn load_all(&self) -> Result<Vec<Fact>> {
        match self {
            Self::Redb(memory) => memory
                .load_all()
                .context("failed to load all facts from redb"),
            #[cfg(feature = "tantivy-search")]
            Self::IndexedRedb(memory) => memory
                .load_all()
                .context("failed to load all facts from indexed-redb"),
        }
    }

    /// Grava ou atualiza um fato no backend ativo e reflete imediatamente no cache em RAM.
    ///
    /// # Errors
    ///
    /// Retorna erro se o fato for inválido, a gravação falhar ou a indexação falhar.
    pub async fn upsert_fact(&self, fact: &Fact) -> Result<Fact> {
        match self {
            Self::Redb(memory) => memory
                .upsert_fact(fact)
                .with_context(|| format!("failed to upsert fact '{}' in redb", fact.id)),
            #[cfg(feature = "tantivy-search")]
            Self::IndexedRedb(memory) => memory
                .upsert_fact(fact)
                .with_context(|| format!("failed to upsert fact '{}' in indexed-redb", fact.id)),
        }
    }

    /// Pré-aquece o cache em RAM com os fatos persistidos no backend ativo.
    ///
    /// # Errors
    ///
    /// Retorna erro se a varredura inicial do banco falhar.
    pub async fn warm_up(&self) -> Result<usize> {
        match self {
            Self::Redb(memory) => memory
                .warm_up()
                .context("failed to warm up redb cache in RAM"),
            #[cfg(feature = "tantivy-search")]
            Self::IndexedRedb(memory) => memory
                .warm_up()
                .context("failed to warm up indexed-redb cache in RAM"),
        }
    }

    /// Reconstrói o índice Tantivy a partir do conteúdo autoritativo do `redb`.
    /// Retorna `Ok(None)` para backends sem índice externo.
    ///
    /// # Errors
    ///
    /// Retorna erro se a leitura do `redb` ou o commit do Tantivy falhar.
    pub fn rebuild_index(&self) -> Result<Option<usize>> {
        match self {
            Self::Redb(_) => Ok(None),
            #[cfg(feature = "tantivy-search")]
            Self::IndexedRedb(memory) => memory
                .rebuild_index()
                .context("failed to rebuild tantivy index from redb")
                .map(Some),
        }
    }
}

#[cfg(feature = "tantivy-search")]
fn open_indexed_redb(db_path: &Path, index_dir: &Path) -> Result<MemoryBackend> {
    let memory = IndexedRedbMemory::open(db_path, index_dir).with_context(|| {
        format!(
            "failed to open indexed-redb (db='{}', index='{}')",
            db_path.display(),
            index_dir.display()
        )
    })?;
    Ok(MemoryBackend::IndexedRedb(memory))
}

#[cfg(not(feature = "tantivy-search"))]
fn open_indexed_redb(_db_path: &Path, _index_dir: &Path) -> Result<MemoryBackend> {
    anyhow::bail!(
        "cannot use '--backend indexed-redb': ns-cli was compiled without the 'tantivy-search' feature"
    );
}

impl DeterministicMemory for MemoryBackend {
    async fn lookup_fact(
        &self,
        entity: &str,
        relation: &str,
    ) -> Result<Option<Fact>, MemoryPortError> {
        match self {
            Self::Redb(memory) => memory.lookup_fact(entity, relation).await,
            #[cfg(feature = "tantivy-search")]
            Self::IndexedRedb(memory) => memory.lookup_fact(entity, relation).await,
        }
    }

    async fn search_facts(&self, query: &str, limit: u8) -> Result<Vec<Fact>, MemoryPortError> {
        match self {
            Self::Redb(memory) => memory.search_facts(query, limit).await,
            #[cfg(feature = "tantivy-search")]
            Self::IndexedRedb(memory) => memory.search_facts(query, limit).await,
        }
    }

    async fn traverse(
        &self,
        start_entity: &str,
        max_hops: u8,
    ) -> Result<Vec<Fact>, MemoryPortError> {
        match self {
            Self::Redb(memory) => memory.traverse(start_entity, max_hops).await,
            #[cfg(feature = "tantivy-search")]
            Self::IndexedRedb(memory) => memory.traverse(start_entity, max_hops).await,
        }
    }

    async fn verify_fact_exists(&self, id: &FactId) -> Result<bool, MemoryPortError> {
        match self {
            Self::Redb(memory) => memory.verify_fact_exists(id).await,
            #[cfg(feature = "tantivy-search")]
            Self::IndexedRedb(memory) => memory.verify_fact_exists(id).await,
        }
    }
}

/// Despachante estático entre `MockLlm` e `Rwkv7Llm` (RWKV-v7 Goose nativo em Rust puro).
#[derive(Debug)]
pub enum LlmBackend {
    Mock(MockLlm),
    #[cfg(feature = "custom-rwkv")]
    Rwkv7(Rwkv7Llm),
}

impl ConstrainedLlm for LlmBackend {
    async fn next_action(&self, history: &[Message]) -> Result<LlmAction, LlmPortError> {
        match self {
            Self::Mock(mock) => mock.next_action(history).await,
            #[cfg(feature = "custom-rwkv")]
            Self::Rwkv7(rwkv7) => rwkv7.next_action(history).await,
        }
    }
}
