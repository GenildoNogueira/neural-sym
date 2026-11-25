use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::RwLock;

use ns_core::error::MemoryPortError;
use ns_core::fact::{Fact, FactId};
use ns_core::ports::DeterministicMemory;
use redb::backends::InMemoryBackend;
use redb::{Builder, Database, ReadableDatabase, ReadableTable, TableDefinition};

use crate::error::MemoryError;
use crate::facts::{FactCache, normalize_key};

/// Tabela primária: `(subject, relation) -> (id, object, source_ref)`.
/// Como a B+Tree ordena lexicograficamente por `subject` e depois `relation`,
/// tanto `lookup_fact` quanto o range scan do BFS leem a mesma árvore sem índice extra.
const FACTS_BY_PAIR: TableDefinition<
    'static,
    (&'static str, &'static str),
    (&'static str, &'static str, &'static str),
> = TableDefinition::new("facts_by_pair");

/// Índice reverso enxuto: `fact_id -> (subject, relation)` para garantir unicidade
/// e validar `verify_fact_exists` em O(log N).
const FACT_IDS: TableDefinition<'static, &'static str, (&'static str, &'static str)> =
    TableDefinition::new("fact_ids");

const DEFAULT_CACHE_BYTES: usize = 64 * 1024 * 1024;

enum SingleUpsertOutcome {
    Stored { replaced_id: Option<FactId> },
    DuplicateId(String),
}

enum BatchUpsertOutcome {
    Stored { replaced_ids: Vec<FactId> },
    DuplicateId(String),
}

/// Relatório interno de gravação em lote contendo os fatos sobreviventes e IDs substituídos.
#[cfg(feature = "tantivy-search")]
pub(crate) struct BatchUpsertReport {
    pub(crate) count: usize,
    pub(crate) surviving: Vec<Fact>,
    pub(crate) replaced_ids: Vec<FactId>,
}

/// Memória determinística apoiada em `redb` (B+Tree Copy-on-Write em Rust puro) + cache quente em RAM.
#[derive(Debug)]
pub struct RedbMemory {
    db: Database,
    cache: FactCache,
    id_cache: RwLock<HashMap<FactId, Fact>>,
}

impl RedbMemory {
    /// Abre (ou cria) um arquivo `.redb` em disco com 64 MiB de cache de páginas.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Redb`] se não for possível abrir o arquivo ou inicializar as tabelas.
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self, MemoryError> {
        let db = Self::open_file_inner(path.as_ref())?;
        Ok(Self {
            db,
            cache: FactCache::default(),
            id_cache: RwLock::new(HashMap::new()),
        })
    }

    /// Abre um banco `redb` exclusivamente em RAM via [`InMemoryBackend`] (ideal para testes).
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Redb`] se a inicialização do backend em memória falhar.
    pub fn open_in_memory() -> Result<Self, MemoryError> {
        let db = Self::open_in_memory_inner()?;
        Ok(Self {
            db,
            cache: FactCache::default(),
            id_cache: RwLock::new(HashMap::new()),
        })
    }

    fn open_file_inner(path: &Path) -> Result<Database, redb::Error> {
        let db = Builder::new()
            .set_cache_size(DEFAULT_CACHE_BYTES)
            .create(path)?;
        Self::init_tables(&db)?;
        Ok(db)
    }

    fn open_in_memory_inner() -> Result<Database, redb::Error> {
        let db = Builder::new().create_with_backend(InMemoryBackend::new())?;
        Self::init_tables(&db)?;
        Ok(db)
    }

    /// Garante que ambas as tabelas existam antes da primeira transação de leitura.
    fn init_tables(db: &Database) -> Result<(), redb::Error> {
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(FACTS_BY_PAIR)?;
            let _ = write_txn.open_table(FACT_IDS)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn cache_insert_fact(&self, fact: &Fact) {
        self.cache.insert(fact.clone());
        if let Ok(mut by_id) = self.id_cache.write() {
            by_id.insert(fact.id.clone(), fact.clone());
        }
    }

    /// Grava ou atualiza um fato de forma atômica no `redb` e reflete imediatamente no cache em RAM.
    ///
    /// # Errors
    ///
    /// Retorna:
    /// - [`MemoryError::InvalidField`] se `id`, `subject` ou `relation` estiverem vazios.
    /// - [`MemoryError::DuplicateFactId`] se `id` já pertencer a outro par `(subject, relation)`.
    /// - [`MemoryError::Redb`] em caso de falha de transação ou armazenamento.
    pub fn upsert_fact(&self, fact: &Fact) -> Result<Fact, MemoryError> {
        let (stored, _) = self.upsert_fact_with_replaced(fact)?;
        Ok(stored)
    }

    /// Variante interna que também devolve o `FactId` substituído no mesmo par `(subject, relation)`,
    /// permitindo que índices secundários (ex.: Tantivy) removam a entrada antiga sem consulta extra.
    pub(crate) fn upsert_fact_with_replaced(
        &self,
        fact: &Fact,
    ) -> Result<(Fact, Option<FactId>), MemoryError> {
        let normalized = validate_and_normalize(fact)?;

        match self.upsert_tx(&normalized)? {
            SingleUpsertOutcome::Stored { replaced_id } => {
                self.cache.insert(normalized.clone());
                if let Ok(mut by_id) = self.id_cache.write() {
                    if let Some(ref old_id) = replaced_id {
                        by_id.remove(old_id);
                    }
                    by_id.insert(normalized.id.clone(), normalized.clone());
                }
                Ok((normalized, replaced_id))
            }
            SingleUpsertOutcome::DuplicateId(id) => Err(MemoryError::DuplicateFactId(id)),
        }
    }

    fn upsert_tx(&self, normalized: &Fact) -> Result<SingleUpsertOutcome, redb::Error> {
        let mut replaced_id = None;
        let write_txn = self.db.begin_write()?;
        {
            let mut facts_table = write_txn.open_table(FACTS_BY_PAIR)?;
            let mut ids_table = write_txn.open_table(FACT_IDS)?;

            let pair = (normalized.subject.as_str(), normalized.relation.as_str());

            let id_conflict = ids_table
                .get(normalized.id.as_str())?
                .is_some_and(|guard| guard.value() != pair);

            if id_conflict {
                return Ok(SingleUpsertOutcome::DuplicateId(normalized.id.to_string()));
            }

            let previous_id = facts_table
                .insert(
                    pair,
                    (
                        normalized.id.as_str(),
                        normalized.object.as_str(),
                        normalized.source_ref.as_str(),
                    ),
                )?
                .map(|guard| guard.value().0.to_owned());

            if let Some(old_id) = previous_id.filter(|old| old != normalized.id.as_str()) {
                ids_table.remove(old_id.as_str())?;
                replaced_id = Some(FactId::new(old_id));
            }

            ids_table.insert(normalized.id.as_str(), pair)?;
        }
        write_txn.commit()?;
        Ok(SingleUpsertOutcome::Stored { replaced_id })
    }

    /// Grava um lote inteiro em uma única transação `redb` e atualiza o cache em RAM.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::InvalidField`], [`MemoryError::DuplicateFactId`] ou [`MemoryError::Redb`].
    pub fn upsert_batch(&self, facts: &[Fact]) -> Result<usize, MemoryError> {
        let (count, _, _) = self.upsert_batch_internal(facts)?;
        Ok(count)
    }

    #[cfg(feature = "tantivy-search")]
    pub(crate) fn upsert_batch_with_replaced(
        &self,
        facts: &[Fact],
    ) -> Result<BatchUpsertReport, MemoryError> {
        let (count, surviving, replaced_ids) = self.upsert_batch_internal(facts)?;
        Ok(BatchUpsertReport {
            count,
            surviving,
            replaced_ids,
        })
    }

    fn upsert_batch_internal(
        &self,
        facts: &[Fact],
    ) -> Result<(usize, Vec<Fact>, Vec<FactId>), MemoryError> {
        let mut normalized_batch = Vec::with_capacity(facts.len());
        for fact in facts {
            normalized_batch.push(validate_and_normalize(fact)?);
        }

        match self.upsert_batch_tx(&normalized_batch)? {
            BatchUpsertOutcome::Stored { replaced_ids } => {
                let count = normalized_batch.len();
                let replaced_set: HashSet<&FactId> = replaced_ids.iter().collect();

                for fact in &normalized_batch {
                    self.cache.insert(fact.clone());
                }

                let mut surviving = Vec::with_capacity(count);
                if let Ok(mut by_id) = self.id_cache.write() {
                    for old_id in &replaced_ids {
                        by_id.remove(old_id);
                    }
                    for fact in normalized_batch {
                        if !replaced_set.contains(&fact.id) {
                            by_id.insert(fact.id.clone(), fact.clone());
                            surviving.push(fact);
                        }
                    }
                } else {
                    for fact in normalized_batch {
                        if !replaced_set.contains(&fact.id) {
                            surviving.push(fact);
                        }
                    }
                }

                Ok((count, surviving, replaced_ids))
            }
            BatchUpsertOutcome::DuplicateId(id) => Err(MemoryError::DuplicateFactId(id)),
        }
    }

    fn upsert_batch_tx(&self, batch: &[Fact]) -> Result<BatchUpsertOutcome, redb::Error> {
        let mut replaced: HashSet<FactId> = HashSet::new();
        let write_txn = self.db.begin_write()?;
        {
            let mut facts_table = write_txn.open_table(FACTS_BY_PAIR)?;
            let mut ids_table = write_txn.open_table(FACT_IDS)?;

            for normalized in batch {
                let pair = (normalized.subject.as_str(), normalized.relation.as_str());

                let id_conflict = ids_table
                    .get(normalized.id.as_str())?
                    .is_some_and(|guard| guard.value() != pair);

                if id_conflict {
                    return Ok(BatchUpsertOutcome::DuplicateId(normalized.id.to_string()));
                }

                let previous_id = facts_table
                    .insert(
                        pair,
                        (
                            normalized.id.as_str(),
                            normalized.object.as_str(),
                            normalized.source_ref.as_str(),
                        ),
                    )?
                    .map(|guard| guard.value().0.to_owned());

                if let Some(old_id) = previous_id.filter(|old| old != normalized.id.as_str()) {
                    ids_table.remove(old_id.as_str())?;
                    replaced.insert(FactId::new(old_id));
                }

                replaced.remove(&normalized.id);
                ids_table.insert(normalized.id.as_str(), pair)?;
            }
        }
        write_txn.commit()?;
        Ok(BatchUpsertOutcome::Stored {
            replaced_ids: replaced.into_iter().collect(),
        })
    }

    /// Busca um fato exato pelo [`FactId`], priorizando o cache em RAM antes de abrir transação no `redb`.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Redb`] se a leitura da transação falhar.
    pub fn lookup_by_id(&self, id: &FactId) -> Result<Option<Fact>, MemoryError> {
        if let Ok(by_id) = self.id_cache.read()
            && let Some(hit) = by_id.get(id)
        {
            tracing::trace!(fact_id = %id, "redb lookup_by_id RAM hit");
            return Ok(Some(hit.clone()));
        }

        let found = self.lookup_by_id_db(id)?;
        if let Some(ref fact) = found {
            self.cache_insert_fact(fact);
        }
        Ok(found)
    }

    /// Resolve vários [`FactId`]s preservando a ordem recebida (ex.: ranking BM25 do Tantivy),
    /// servindo hits direto da RAM e abrindo no máximo uma única transação de leitura para os *misses*.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Redb`] se a leitura em lote no `redb` falhar.
    pub fn lookup_many_by_ids(&self, ids: &[FactId]) -> Result<Vec<Fact>, MemoryError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut slots: Vec<Option<Fact>> = vec![None; ids.len()];
        let mut missing_indices: Vec<usize> = Vec::new();

        if let Ok(by_id) = self.id_cache.read() {
            for (idx, id) in ids.iter().enumerate() {
                if let Some(hit) = by_id.get(id) {
                    slots[idx] = Some(hit.clone());
                } else {
                    missing_indices.push(idx);
                }
            }
        } else {
            missing_indices.extend(0..ids.len());
        }

        if !missing_indices.is_empty() {
            let fetched = self.lookup_many_by_ids_db(ids, &missing_indices)?;
            let mut newly_cached: Vec<Fact> = Vec::with_capacity(fetched.len());

            for (slot_idx, fact) in fetched {
                self.cache.insert(fact.clone());
                newly_cached.push(fact.clone());
                slots[slot_idx] = Some(fact);
            }

            if !newly_cached.is_empty()
                && let Ok(mut by_id) = self.id_cache.write()
            {
                for fact in newly_cached {
                    by_id.insert(fact.id.clone(), fact);
                }
            }
        }

        Ok(slots.into_iter().flatten().collect())
    }

    fn lookup_by_id_db(&self, id: &FactId) -> Result<Option<Fact>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let ids_table = read_txn.open_table(FACT_IDS)?;
        let Some(pair_guard) = ids_table.get(id.as_str())? else {
            return Ok(None);
        };
        let (subject, relation) = pair_guard.value();

        let facts_table = read_txn.open_table(FACTS_BY_PAIR)?;
        let Some(val_guard) = facts_table.get((subject, relation))? else {
            return Ok(None);
        };
        let (stored_id, object, source_ref) = val_guard.value();

        Ok(Some(Fact {
            id: FactId::new(stored_id),
            subject: subject.to_owned(),
            relation: relation.to_owned(),
            object: object.to_owned(),
            source_ref: source_ref.to_owned(),
        }))
    }

    fn lookup_many_by_ids_db(
        &self,
        ids: &[FactId],
        missing_indices: &[usize],
    ) -> Result<Vec<(usize, Fact)>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let ids_table = read_txn.open_table(FACT_IDS)?;
        let facts_table = read_txn.open_table(FACTS_BY_PAIR)?;

        let mut fetched = Vec::with_capacity(missing_indices.len());
        for &idx in missing_indices {
            let id = &ids[idx];
            let Some(pair_guard) = ids_table.get(id.as_str())? else {
                continue;
            };
            let (subject, relation) = pair_guard.value();
            let Some(val_guard) = facts_table.get((subject, relation))? else {
                continue;
            };
            let (stored_id, object, source_ref) = val_guard.value();
            fetched.push((
                idx,
                Fact {
                    id: FactId::new(stored_id),
                    subject: subject.to_owned(),
                    relation: relation.to_owned(),
                    object: object.to_owned(),
                    source_ref: source_ref.to_owned(),
                },
            ));
        }

        Ok(fetched)
    }

    /// Carrega todos os fatos do `redb` ordenados por `(subject, relation)` (útil para reindexação).
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Redb`] se a leitura falhar.
    pub fn load_all(&self) -> Result<Vec<Fact>, MemoryError> {
        Ok(self.load_all_db()?)
    }

    fn load_all_db(&self) -> Result<Vec<Fact>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(FACTS_BY_PAIR)?;
        let mut facts = Vec::new();
        for entry in table.iter()? {
            let (key_guard, val_guard) = entry?;
            let (subject, relation) = key_guard.value();
            let (id, object, source_ref) = val_guard.value();
            facts.push(Fact {
                id: FactId::new(id),
                subject: subject.to_owned(),
                relation: relation.to_owned(),
                object: object.to_owned(),
                source_ref: source_ref.to_owned(),
            });
        }
        Ok(facts)
    }

    fn search_db(&self, query: &str, limit: u8) -> Result<Vec<Fact>, redb::Error> {
        let needle = normalize_key(query);
        if needle.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        let cap = usize::from(limit.min(crate::facts::MAX_SEARCH_LIMIT));
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(FACTS_BY_PAIR)?;

        let mut matches = Vec::with_capacity(cap.min(16));
        for entry in table.iter()? {
            let (key_guard, val_guard) = entry?;
            let (subject, relation) = key_guard.value();
            let (id, object, source_ref) = val_guard.value();

            let matched = subject.contains(&needle)
                || relation.contains(&needle)
                || object.to_lowercase().contains(&needle);

            if matched {
                let fact = Fact {
                    id: FactId::new(id),
                    subject: subject.to_owned(),
                    relation: relation.to_owned(),
                    object: object.to_owned(),
                    source_ref: source_ref.to_owned(),
                };
                self.cache_insert_fact(&fact);
                matches.push(fact);

                if matches.len() >= cap {
                    break;
                }
            }
        }

        Ok(matches)
    }

    /// Compacta páginas livres da B+Tree no arquivo `.redb`.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Redb`] se a compactação falhar.
    pub fn compact(&mut self) -> Result<bool, MemoryError> {
        self.db
            .compact(redb::CompactionMode::Balanced)
            .map_err(redb::Error::from)
            .map_err(MemoryError::from)
    }

    /// Pré-carrega toda a B+Tree no cache em RAM (incluindo o índice por `FactId`).
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Redb`] se a leitura da tabela falhar.
    pub fn warm_up(&self) -> Result<usize, MemoryError> {
        let count = self.warm_up_inner()?;
        tracing::info!(
            cached_facts = count,
            "redb deterministic memory warmed up in RAM"
        );
        Ok(count)
    }

    fn warm_up_inner(&self) -> Result<usize, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(FACTS_BY_PAIR)?;

        let mut loaded = Vec::new();
        for entry in table.iter()? {
            let (key_guard, val_guard) = entry?;
            let (subject, relation) = key_guard.value();
            let (id, object, source_ref) = val_guard.value();

            let fact = Fact {
                id: FactId::new(id),
                subject: subject.to_owned(),
                relation: relation.to_owned(),
                object: object.to_owned(),
                source_ref: source_ref.to_owned(),
            };
            self.cache.insert(fact.clone());
            loaded.push(fact);
        }

        let count = loaded.len();
        if let Ok(mut by_id) = self.id_cache.write() {
            by_id.reserve(count);
            for fact in loaded {
                by_id.insert(fact.id.clone(), fact);
            }
        }
        Ok(count)
    }

    #[must_use]
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }

    fn lookup_db(&self, entity: &str, relation: &str) -> Result<Option<Fact>, redb::Error> {
        let subject = normalize_key(entity);
        let rel = normalize_key(relation);

        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(FACTS_BY_PAIR)?;

        let found = table.get((subject.as_str(), rel.as_str()))?.map(|guard| {
            let (id, object, source_ref) = guard.value();
            Fact {
                id: FactId::new(id),
                subject,
                relation: rel,
                object: object.to_owned(),
                source_ref: source_ref.to_owned(),
            }
        });
        Ok(found)
    }

    fn traverse_bfs_db(&self, start_entity: &str, max_hops: u8) -> Result<Vec<Fact>, redb::Error> {
        if max_hops == 0 {
            return Ok(Vec::new());
        }

        let start = normalize_key(start_entity);
        if start.is_empty() {
            return Ok(Vec::new());
        }

        // Uma única transação MVCC mantém snapshot consistente em todos os saltos do grafo.
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(FACTS_BY_PAIR)?;

        let mut queue = VecDeque::with_capacity(16);
        let mut visited_entities = HashSet::with_capacity(16);
        let mut seen_fact_ids: HashSet<FactId> = HashSet::with_capacity(32);
        let mut collected = Vec::with_capacity(32);

        visited_entities.insert(start.clone());
        queue.push_back((start, 0_u8));

        while let Some((current_entity, depth)) = queue.pop_front() {
            if depth >= max_hops {
                continue;
            }

            for entry in table.range((current_entity.as_str(), "")..)? {
                let (key_guard, val_guard) = entry?;
                let (subject, relation) = key_guard.value();
                if subject != current_entity.as_str() {
                    break;
                }

                let (id, object, source_ref) = val_guard.value();
                let fact = Fact {
                    id: FactId::new(id),
                    subject: subject.to_owned(),
                    relation: relation.to_owned(),
                    object: object.to_owned(),
                    source_ref: source_ref.to_owned(),
                };

                let next_entity = normalize_key(&fact.object);
                if !next_entity.is_empty() && visited_entities.insert(next_entity.clone()) {
                    queue.push_back((next_entity, depth.saturating_add(1)));
                }

                if seen_fact_ids.insert(fact.id.clone()) {
                    self.cache_insert_fact(&fact);
                    collected.push(fact);
                }
            }
        }

        Ok(collected)
    }

    fn exists_id_db(&self, id: &FactId) -> Result<bool, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(FACT_IDS)?;
        Ok(table.get(id.as_str())?.is_some())
    }
}

fn validate_and_normalize(fact: &Fact) -> Result<Fact, MemoryError> {
    let id_str = fact.id.as_str().trim();
    if id_str.is_empty() {
        return Err(MemoryError::InvalidField { field: "id" });
    }
    let subject = normalize_key(&fact.subject);
    if subject.is_empty() {
        return Err(MemoryError::InvalidField { field: "subject" });
    }
    let relation = normalize_key(&fact.relation);
    if relation.is_empty() {
        return Err(MemoryError::InvalidField { field: "relation" });
    }

    Ok(Fact {
        id: FactId::new(id_str),
        subject,
        relation,
        object: fact.object.trim().to_owned(),
        source_ref: fact.source_ref.trim().to_owned(),
    })
}

impl DeterministicMemory for RedbMemory {
    async fn lookup_fact(
        &self,
        entity: &str,
        relation: &str,
    ) -> Result<Option<Fact>, MemoryPortError> {
        if let Some(hit) = self.cache.get_pair(entity, relation) {
            tracing::trace!(entity, relation, "redb deterministic memory RAM hit");
            return Ok(Some(hit));
        }

        let found = self
            .lookup_db(entity, relation)
            .map_err(MemoryError::from)?;
        if let Some(ref fact) = found {
            self.cache_insert_fact(fact);
        }
        Ok(found)
    }

    async fn traverse(
        &self,
        start_entity: &str,
        max_hops: u8,
    ) -> Result<Vec<Fact>, MemoryPortError> {
        let facts = self
            .traverse_bfs_db(start_entity, max_hops)
            .map_err(MemoryError::from)?;
        Ok(facts)
    }

    async fn verify_fact_exists(&self, id: &FactId) -> Result<bool, MemoryPortError> {
        if self.cache.contains_id(id) {
            return Ok(true);
        }
        let exists = self.exists_id_db(id).map_err(MemoryError::from)?;
        Ok(exists)
    }

    async fn search_facts(&self, query: &str, limit: u8) -> Result<Vec<Fact>, MemoryPortError> {
        let facts = self.search_db(query, limit).map_err(MemoryError::from)?;
        Ok(facts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fact(id: &str, subject: &str, relation: &str, object: &str) -> Fact {
        Fact {
            id: FactId::from(id),
            subject: subject.to_owned(),
            relation: relation.to_owned(),
            object: object.to_owned(),
            source_ref: "test#redb".to_owned(),
        }
    }

    #[tokio::test]
    async fn lookup_is_case_insensitive_and_enforces_unique_fact_id()
    -> Result<(), Box<dyn std::error::Error>> {
        let memory = RedbMemory::open_in_memory()?;
        memory.upsert_fact(&sample_fact("f-1", "NeuroSym", "engine", "redb"))?;

        // Colisão de FactId em outro par (subject, relation) precisa falhar
        let duplicate = memory.upsert_fact(&sample_fact("f-1", "other", "engine", "sqlite"));
        assert!(matches!(duplicate, Err(MemoryError::DuplicateFactId(_))));

        // Substituir o mesmo par por um novo FactId deve remover o ID antigo do índice e do cache
        memory.upsert_fact(&sample_fact("f-1-v2", "NeuroSym", "engine", "redb-5"))?;
        assert!(!memory.verify_fact_exists(&FactId::from("f-1")).await?);
        assert!(memory.lookup_by_id(&FactId::from("f-1"))?.is_none());
        assert!(memory.verify_fact_exists(&FactId::from("f-1-v2")).await?);

        let found = memory.lookup_fact("NEUROSYM", "ENGINE").await?;
        assert_eq!(found.as_ref().map(|f| f.object.as_str()), Some("redb-5"));
        assert_eq!(memory.cached_count(), 1);

        Ok(())
    }

    #[test]
    fn lookup_many_by_ids_preserves_order_and_uses_ram_cache()
    -> Result<(), Box<dyn std::error::Error>> {
        let memory = RedbMemory::open_in_memory()?;
        memory.upsert_batch(&[
            sample_fact("f-1", "a", "rel", "1"),
            sample_fact("f-2", "b", "rel", "2"),
            sample_fact("f-3", "c", "rel", "3"),
        ])?;

        let ordered = memory.lookup_many_by_ids(&[
            FactId::from("f-3"),
            FactId::from("f-inexistente"),
            FactId::from("f-1"),
        ])?;
        let ids: Vec<&str> = ordered.iter().map(|f| f.id.as_str()).collect();
        assert_eq!(ids, vec!["f-3", "f-1"]);
        Ok(())
    }

    #[tokio::test]
    async fn traverse_bfs_respects_hops_and_avoids_cycles() -> Result<(), Box<dyn std::error::Error>>
    {
        let memory = RedbMemory::open_in_memory()?;
        memory.upsert_fact(&sample_fact("f-1", "ns-engine", "depends_on", "ns-core"))?;
        memory.upsert_fact(&sample_fact("f-2", "ns-core", "uses", "thiserror"))?;
        memory.upsert_fact(&sample_fact(
            "f-3",
            "thiserror",
            "referenced_by",
            "ns-engine",
        ))?;

        let one_hop = memory.traverse("ns-engine", 1).await?;
        assert_eq!(one_hop.len(), 1);
        assert_eq!(one_hop[0].id, FactId::from("f-1"));

        let two_hops = memory.traverse("ns-engine", 2).await?;
        assert_eq!(two_hops.len(), 2);
        assert_eq!(two_hops[1].id, FactId::from("f-2"));

        let ten_hops = memory.traverse("ns-engine", 10).await?;
        assert_eq!(ten_hops.len(), 3);

        Ok(())
    }
}
