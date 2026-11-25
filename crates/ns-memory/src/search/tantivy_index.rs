use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

use ns_core::error::MemoryPortError;
use ns_core::fact::{Fact, FactId};
use ns_core::ports::DeterministicMemory;
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, QueryParser, RegexQuery};
use tantivy::schema::{Field, STORED, STRING, Schema, TEXT, Value};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term, doc};

use crate::error::MemoryError;
use crate::facts::{MAX_SEARCH_LIMIT, normalize_key};
use crate::store::RedbMemory;

const WRITER_HEAP_BYTES: usize = 15_000_000;

#[derive(Debug, Clone, Copy)]
struct FactFields {
    id: Field,
    subject_text: Field,
    relation_text: Field,
    object_text: Field,
    subject_raw: Field,
    relation_raw: Field,
    object_raw: Field,
}

fn build_schema() -> (Schema, FactFields) {
    let mut builder = Schema::builder();
    let fields = FactFields {
        id: builder.add_text_field("id", STRING | STORED),
        subject_text: builder.add_text_field("subject_text", TEXT),
        relation_text: builder.add_text_field("relation_text", TEXT),
        object_text: builder.add_text_field("object_text", TEXT),
        subject_raw: builder.add_text_field("subject_raw", STRING),
        relation_raw: builder.add_text_field("relation_raw", STRING),
        object_raw: builder.add_text_field("object_raw", STRING),
    };
    (builder.build(), fields)
}

/// Índice invertido BM25 + FST em Rust puro (`tantivy`) para descoberta rápida de `FactId`s.
pub struct TantivyFactIndex {
    index: Index,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,
    fields: FactFields,
}

impl std::fmt::Debug for TantivyFactIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TantivyFactIndex").finish_non_exhaustive()
    }
}

impl TantivyFactIndex {
    /// Abre um índice Tantivy existente em `dir` ou cria um novo se ainda não existir.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Search`] se a abertura ou inicialização do índice falhar.
    pub fn open_in_dir(dir: impl AsRef<Path>) -> Result<Self, MemoryError> {
        let dir_ref = dir.as_ref();
        let (schema, fields) = build_schema();

        let index = if dir_ref.join("meta.json").exists() {
            Index::open_in_dir(dir_ref)?
        } else {
            if let Err(io_err) = std::fs::create_dir_all(dir_ref) {
                return Err(MemoryError::Search(Box::new(tantivy::TantivyError::from(
                    io_err,
                ))));
            }
            Index::create_in_dir(dir_ref, schema)?
        };

        Self::from_index(index, fields)
    }

    /// Cria um índice Tantivy em RAM (ideal para testes).
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Search`] se a criação do índice falhar.
    pub fn open_in_memory() -> Result<Self, MemoryError> {
        let (schema, fields) = build_schema();
        let index = Index::create_in_ram(schema);
        Self::from_index(index, fields)
    }

    fn from_index(index: Index, fields: FactFields) -> Result<Self, MemoryError> {
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let writer = index.writer(WRITER_HEAP_BYTES)?;

        Ok(Self {
            index,
            reader,
            writer: Mutex::new(writer),
            fields,
        })
    }

    /// Indexa ou substitui um único fato e realiza commit + reload imediato.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Search`] ou [`MemoryError::LockPoisoned`].
    pub fn upsert_fact(
        &self,
        fact: &Fact,
        replaced_id: Option<&FactId>,
    ) -> Result<(), MemoryError> {
        let mut writer = self.writer.lock().map_err(|_| MemoryError::LockPoisoned)?;

        if let Some(old_id) = replaced_id {
            writer.delete_term(Term::from_field_text(self.fields.id, old_id.as_str()));
        }
        writer.delete_term(Term::from_field_text(self.fields.id, fact.id.as_str()));
        writer.add_document(self.to_document(fact))?;
        writer.commit()?;
        drop(writer);

        self.reader.reload()?;
        Ok(())
    }

    /// Indexa um lote inteiro em um único commit no Tantivy.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Search`] ou [`MemoryError::LockPoisoned`].
    pub fn upsert_batch(&self, facts: &[Fact], replaced_ids: &[FactId]) -> Result<(), MemoryError> {
        let mut writer = self.writer.lock().map_err(|_| MemoryError::LockPoisoned)?;

        for old_id in replaced_ids {
            writer.delete_term(Term::from_field_text(self.fields.id, old_id.as_str()));
        }
        for fact in facts {
            writer.delete_term(Term::from_field_text(self.fields.id, fact.id.as_str()));
            writer.add_document(self.to_document(fact))?;
        }
        writer.commit()?;
        drop(writer);

        self.reader.reload()?;
        Ok(())
    }

    /// Limpa o índice e reindexa todos os fatos fornecidos em um único commit.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Search`] ou [`MemoryError::LockPoisoned`].
    pub fn rebuild_from(&self, facts: &[Fact]) -> Result<usize, MemoryError> {
        let mut writer = self.writer.lock().map_err(|_| MemoryError::LockPoisoned)?;
        writer.delete_all_documents()?;
        for fact in facts {
            writer.add_document(self.to_document(fact))?;
        }
        writer.commit()?;
        drop(writer);

        self.reader.reload()?;
        Ok(facts.len())
    }

    /// Retorna até `limit` `FactId`s ranqueados por BM25, usando busca literal direta na FST
    /// quando o termo contém símbolos (ex.: `"plano_"` ou `"15%"`) para evitar que o
    /// `SimpleTokenizer` descarte a pontuação e gere falsos positivos.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Search`] se a execução da consulta falhar.
    pub fn search_ids(&self, query: &str, limit: u8) -> Result<Vec<FactId>, MemoryError> {
        let needle = normalize_key(query);
        if needle.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        let cap = usize::from(limit.min(MAX_SEARCH_LIMIT));
        let searcher = self.reader.searcher();

        let mut seen = HashSet::with_capacity(cap);
        let mut ids = Vec::with_capacity(cap);

        let has_whitespace = needle.chars().any(char::is_whitespace);
        let has_symbol = needle
            .chars()
            .any(|c| !c.is_alphanumeric() && !c.is_whitespace());
        let literal_symbol_query = !has_whitespace && has_symbol;

        // 1. Fast-path BM25 nos campos tokenizados (consultas em linguagem natural)
        if !literal_symbol_query {
            let parser = QueryParser::for_index(
                &self.index,
                vec![
                    self.fields.subject_text,
                    self.fields.relation_text,
                    self.fields.object_text,
                ],
            );
            let (bm25_query, _parse_errors) = parser.parse_query_lenient(&needle);
            let bm25_hits = searcher.search(&bm25_query, &TopDocs::with_limit(cap))?;

            for (_score, addr) in bm25_hits {
                let doc: TantivyDocument = searcher.doc(addr)?;
                if let Some(raw_id) = doc.get_first(self.fields.id).and_then(|v| v.as_str()) {
                    let fact_id = FactId::new(raw_id);
                    if seen.insert(fact_id.clone()) {
                        ids.push(fact_id);
                    }
                }
            }
        }

        // 2. Busca literal na FST para substrings exatas (obrigatória para símbolos; fallback para demais)
        if ids.len() < cap {
            let pattern = format!(".*{}.*", escape_regex(&needle));
            let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::with_capacity(3);

            for field in [
                self.fields.subject_raw,
                self.fields.relation_raw,
                self.fields.object_raw,
            ] {
                if let Ok(regex_query) = RegexQuery::from_pattern(&pattern, field) {
                    clauses.push((Occur::Should, Box::new(regex_query)));
                }
            }

            if !clauses.is_empty() {
                let union = BooleanQuery::new(clauses);
                let fst_hits = searcher.search(&union, &TopDocs::with_limit(cap))?;
                for (_score, addr) in fst_hits {
                    if ids.len() >= cap {
                        break;
                    }
                    let doc: TantivyDocument = searcher.doc(addr)?;
                    if let Some(raw_id) = doc.get_first(self.fields.id).and_then(|v| v.as_str()) {
                        let fact_id = FactId::new(raw_id);
                        if seen.insert(fact_id.clone()) {
                            ids.push(fact_id);
                        }
                    }
                }
            }
        }

        Ok(ids)
    }

    fn to_document(&self, fact: &Fact) -> TantivyDocument {
        let subject_norm = normalize_key(&fact.subject);
        let relation_norm = normalize_key(&fact.relation);
        let object_norm = normalize_key(&fact.object);

        doc!(
            self.fields.id => fact.id.as_str(),
            self.fields.subject_text => subject_norm.as_str(),
            self.fields.relation_text => relation_norm.as_str(),
            self.fields.object_text => object_norm.as_str(),
            self.fields.subject_raw => subject_norm.as_str(),
            self.fields.relation_raw => relation_norm.as_str(),
            self.fields.object_raw => object_norm.as_str(),
        )
    }
}

/// Combina `RedbMemory` (verdade determinística ACID) com `TantivyFactIndex` (busca escalável BM25 + FST).
#[derive(Debug)]
pub struct IndexedRedbMemory {
    store: RedbMemory,
    index: TantivyFactIndex,
}

impl IndexedRedbMemory {
    /// Abre o arquivo `.redb` e o diretório de índice Tantivy sincronizados.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError`] se a abertura do `redb` ou do índice Tantivy falhar.
    pub fn open(
        redb_path: impl AsRef<Path>,
        index_dir: impl AsRef<Path>,
    ) -> Result<Self, MemoryError> {
        let store = RedbMemory::open_file(redb_path)?;
        let index = TantivyFactIndex::open_in_dir(index_dir)?;
        Ok(Self { store, index })
    }

    /// Abre ambos os motores em memória para testes rápidos.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError`] se a inicialização falhar.
    pub fn open_in_memory() -> Result<Self, MemoryError> {
        let store = RedbMemory::open_in_memory()?;
        let index = TantivyFactIndex::open_in_memory()?;
        Ok(Self { store, index })
    }

    /// Grava primeiro no `redb` (fonte da verdade) e atualiza o índice Tantivy em seguida.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError`] se a gravação ou indexação falhar.
    pub fn upsert_fact(&self, fact: &Fact) -> Result<Fact, MemoryError> {
        let previous_id = self
            .store
            .lookup_by_id(&fact.id)?
            .map(|existing| existing.id);
        let stored = self.store.upsert_fact(fact)?;
        self.index.upsert_fact(&stored, previous_id.as_ref())?;
        Ok(stored)
    }

    /// Grava um lote inteiro em uma única transação `redb` e um único commit Tantivy.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError`] se a gravação em lote ou indexação falhar.
    pub fn upsert_batch(&self, facts: &[Fact]) -> Result<usize, MemoryError> {
        let count = self.store.upsert_batch(facts)?;
        self.index.upsert_batch(facts, &[])?;
        Ok(count)
    }

    /// Reconstrói completamente o índice Tantivy a partir do conteúdo autoritativo do `redb`.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError`] se a leitura do `redb` ou o commit do índice falhar.
    pub fn rebuild_index(&self) -> Result<usize, MemoryError> {
        let all_facts = self.store.load_all()?;
        self.index.rebuild_from(&all_facts)
    }

    /// Pré-aquece o cache em RAM do `RedbMemory`.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Redb`] se o aquecimento falhar.
    pub fn warm_up(&self) -> Result<usize, MemoryError> {
        self.store.warm_up()
    }

    /// Carrega todos os fatos diretamente do `redb` subjacente.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Redb`] se a leitura falhar.
    pub fn load_all(&self) -> Result<Vec<Fact>, MemoryError> {
        self.store.load_all()
    }
}

impl DeterministicMemory for IndexedRedbMemory {
    async fn lookup_fact(
        &self,
        entity: &str,
        relation: &str,
    ) -> Result<Option<Fact>, MemoryPortError> {
        self.store.lookup_fact(entity, relation).await
    }

    async fn search_facts(&self, query: &str, limit: u8) -> Result<Vec<Fact>, MemoryPortError> {
        let candidate_ids = self.index.search_ids(query, limit)?;
        let mut resolved = Vec::with_capacity(candidate_ids.len());

        for id in &candidate_ids {
            if let Some(fact) = self.store.lookup_by_id(id)? {
                resolved.push(fact);
            }
        }

        Ok(resolved)
    }

    async fn traverse(
        &self,
        start_entity: &str,
        max_hops: u8,
    ) -> Result<Vec<Fact>, MemoryPortError> {
        self.store.traverse(start_entity, max_hops).await
    }

    async fn verify_fact_exists(&self, id: &FactId) -> Result<bool, MemoryPortError> {
        self.store.verify_fact_exists(id).await
    }
}

fn escape_regex(input: &str) -> String {
    let mut out = String::with_capacity(input.len().saturating_add(4));
    for ch in input.chars() {
        if matches!(
            ch,
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$'
        ) {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn indexed_redb_resolves_bm25_and_fst_substrings_from_redb()
    -> Result<(), Box<dyn std::error::Error>> {
        let mem = IndexedRedbMemory::open_in_memory()?;
        mem.upsert_batch(&[
            Fact {
                id: FactId::from("f-pro-preco"),
                subject: "plano_pro".to_owned(),
                relation: "preco_mensal".to_owned(),
                object: "199.90".to_owned(),
                source_ref: "catalogo#2026".to_owned(),
            },
            Fact {
                id: FactId::from("f-pro-desc"),
                subject: "plano_pro".to_owned(),
                relation: "desconto_anual".to_owned(),
                object: "15%".to_owned(),
                source_ref: "catalogo#2026".to_owned(),
            },
            Fact {
                id: FactId::from("f-basico"),
                subject: "plano-basico".to_owned(),
                relation: "preco_mensal".to_owned(),
                object: "99.90".to_owned(),
                source_ref: "catalogo#2026".to_owned(),
            },
        ])?;

        // Consulta BM25 natural
        let bm25 = mem.search_facts("mensal plano", 10).await?;
        assert!(!bm25.is_empty());

        // Substring literal com '_' e '%' via FST
        let under = mem.search_facts("plano_", 10).await?;
        assert_eq!(under.len(), 2);
        assert!(under.iter().all(|f| f.subject == "plano_pro"));

        let pct = mem.search_facts("15%", 10).await?;
        assert_eq!(pct.len(), 1);
        assert_eq!(pct[0].id, FactId::from("f-pro-desc"));

        // Rebuild completo do índice a partir do redb
        assert_eq!(mem.rebuild_index()?, 3);
        let exact = mem.lookup_fact("plano_pro", "preco_mensal").await?;
        assert_eq!(exact.as_ref().map(|f| f.object.as_str()), Some("199.90"));

        Ok(())
    }
}
