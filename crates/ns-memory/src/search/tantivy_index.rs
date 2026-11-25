use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;

use ns_core::error::MemoryPortError;
use ns_core::fact::{Fact, FactId};
use ns_core::ports::DeterministicMemory;
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, QueryParser, RegexQuery, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, STORED, STRING, Schema, TEXT, Value};
use tantivy::{
    Index, IndexReader, IndexWriter, ReloadPolicy, Searcher, TantivyDocument, Term, doc,
};

use crate::error::MemoryError;
use crate::facts::{MAX_SEARCH_LIMIT, normalize_key};
use crate::store::RedbMemory;

const WRITER_HEAP_BYTES: usize = 15_000_000;
const MAX_FST_FALLBACK_TERMS: usize = 6;

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
            writer.delete_term(Term::from_field_text(
                self.fields.id,
                old_id.as_str().trim(),
            ));
        }
        writer.delete_term(Term::from_field_text(
            self.fields.id,
            fact.id.as_str().trim(),
        ));
        writer.add_document(self.to_document(fact))?;
        writer.commit()?;
        drop(writer);

        self.reader.reload()?;
        Ok(())
    }

    /// Indexa um lote inteiro em um único commit no Tantivy, removendo quaisquer `replaced_ids`.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError::Search`] ou [`MemoryError::LockPoisoned`].
    pub fn upsert_batch(&self, facts: &[Fact], replaced_ids: &[FactId]) -> Result<(), MemoryError> {
        let mut writer = self.writer.lock().map_err(|_| MemoryError::LockPoisoned)?;

        let replaced_set: HashSet<&str> =
            replaced_ids.iter().map(|id| id.as_str().trim()).collect();
        for old_id in replaced_ids {
            writer.delete_term(Term::from_field_text(
                self.fields.id,
                old_id.as_str().trim(),
            ));
        }
        for fact in facts {
            let id_str = fact.id.as_str().trim();
            writer.delete_term(Term::from_field_text(self.fields.id, id_str));
            if !replaced_set.contains(id_str) {
                writer.add_document(self.to_document(fact))?;
            }
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

    /// Retorna até `limit` `FactId`s ranqueados por BM25 + boost exato em `*_raw`.
    ///
    /// - Consultas de termo único contendo símbolos (ex.: `"plano_"` ou `"15%"`) vão direto na FST
    ///   para evitar falsos positivos causados pela remoção de pontuação do `SimpleTokenizer`.
    /// - Perguntas em linguagem natural (ex.: `"Qual o(a) preco_mensal de plano_enterprise?"`)
    ///   são sanitizadas contra operadores de sintaxe e stopwords antes da busca BM25.
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

        // 1. Fast-path BM25 + boost exato nos campos *_raw (linguagem natural e termos limpos)
        if !literal_symbol_query {
            let terms = extract_query_terms(&needle);
            if !terms.is_empty() {
                self.search_bm25_and_raw(&searcher, &terms, cap, &mut seen, &mut ids)?;
            }
        }

        // 2. FST literal: obrigatória para símbolos isolados; fallback para substrings/prefixos
        if literal_symbol_query || (!has_whitespace && ids.len() < cap) {
            self.search_fst_terms(&searcher, &[needle.as_str()], cap, &mut seen, &mut ids)?;
        } else if has_whitespace && ids.is_empty() {
            let mut fallback_terms: Vec<String> = extract_query_terms(&needle)
                .into_iter()
                .filter(|t| t.chars().count() >= 3)
                .collect();
            fallback_terms.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
            fallback_terms.truncate(MAX_FST_FALLBACK_TERMS);

            let refs: Vec<&str> = fallback_terms.iter().map(String::as_str).collect();
            if !refs.is_empty() {
                self.search_fst_terms(&searcher, &refs, cap, &mut seen, &mut ids)?;
            }
        }

        Ok(ids)
    }

    fn search_bm25_and_raw(
        &self,
        searcher: &Searcher,
        terms: &[String],
        cap: usize,
        seen: &mut HashSet<FactId>,
        ids: &mut Vec<FactId>,
    ) -> Result<(), MemoryError> {
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::with_capacity(terms.len() * 3 + 1);

        // Separa '_' e '-' em subtokens limpos para casar tanto "plano_enterprise" quanto "plano enterprise"
        let bm25_text = terms
            .iter()
            .flat_map(|t| t.split(['_', '-']))
            .filter(|sub| !sub.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        if !bm25_text.is_empty() {
            let parser = QueryParser::for_index(
                &self.index,
                vec![
                    self.fields.subject_text,
                    self.fields.relation_text,
                    self.fields.object_text,
                ],
            );
            let (bm25_query, _parse_errors) = parser.parse_query_lenient(&bm25_text);
            clauses.push((Occur::Should, bm25_query));
        }

        // Boost O(1) para identificadores exatos em *_raw (ex.: "plano_enterprise", "preco_mensal")
        for tok in terms {
            for field in [
                self.fields.subject_raw,
                self.fields.relation_raw,
                self.fields.object_raw,
            ] {
                let term = Term::from_field_text(field, tok);
                clauses.push((
                    Occur::Should,
                    Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
                ));
            }
        }

        if clauses.is_empty() {
            return Ok(());
        }

        let combined = BooleanQuery::new(clauses);
        let hits = searcher.search(&combined, &TopDocs::with_limit(cap))?;
        self.collect_hit_ids(searcher, hits, cap, seen, ids)
    }

    fn search_fst_terms(
        &self,
        searcher: &Searcher,
        terms: &[&str],
        cap: usize,
        seen: &mut HashSet<FactId>,
        ids: &mut Vec<FactId>,
    ) -> Result<(), MemoryError> {
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::with_capacity(terms.len() * 3);

        for &term in terms {
            let pattern = format!(".*{}.*", escape_regex(term));
            for field in [
                self.fields.subject_raw,
                self.fields.relation_raw,
                self.fields.object_raw,
            ] {
                if let Ok(regex_query) = RegexQuery::from_pattern(&pattern, field) {
                    clauses.push((Occur::Should, Box::new(regex_query)));
                }
            }
        }

        if clauses.is_empty() {
            return Ok(());
        }

        let union = BooleanQuery::new(clauses);
        let hits = searcher.search(&union, &TopDocs::with_limit(cap))?;
        self.collect_hit_ids(searcher, hits, cap, seen, ids)
    }

    fn collect_hit_ids(
        &self,
        searcher: &Searcher,
        hits: Vec<(f32, tantivy::DocAddress)>,
        cap: usize,
        seen: &mut HashSet<FactId>,
        ids: &mut Vec<FactId>,
    ) -> Result<(), MemoryError> {
        for (_score, addr) in hits {
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
        Ok(())
    }

    fn to_document(&self, fact: &Fact) -> TantivyDocument {
        let id_norm = fact.id.as_str().trim();
        let subject_norm = normalize_key(&fact.subject);
        let relation_norm = normalize_key(&fact.relation);
        let object_norm = normalize_key(&fact.object);

        doc!(
            self.fields.id => id_norm,
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

    /// Grava primeiro no `redb` (fonte da verdade) e sincroniza o índice Tantivy removendo
    /// eventual `FactId` antigo substituído no mesmo par `(subject, relation)`.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError`] se a gravação ou indexação falhar.
    pub fn upsert_fact(&self, fact: &Fact) -> Result<Fact, MemoryError> {
        let (stored, replaced_id) = self.store.upsert_fact_with_replaced(fact)?;
        self.index.upsert_fact(&stored, replaced_id.as_ref())?;
        Ok(stored)
    }

    /// Grava um lote inteiro em uma única transação `redb` e um único commit Tantivy,
    /// removendo do índice todos os `FactId`s substituídos pelo lote.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError`] se a gravação em lote ou indexação falhar.
    pub fn upsert_batch(&self, facts: &[Fact]) -> Result<usize, MemoryError> {
        let report = self.store.upsert_batch_with_replaced(facts)?;
        self.index
            .upsert_batch(&report.surviving, &report.replaced_ids)?;
        Ok(report.count)
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
        let resolved = self.store.lookup_many_by_ids(&candidate_ids)?;
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

fn extract_query_terms(needle: &str) -> Vec<String> {
    let mut all_tokens: Vec<String> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();

    for raw in needle.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-') {
        let tok = raw.trim_matches(['_', '-']);
        if tok.is_empty() || !seen.insert(tok) {
            continue;
        }
        all_tokens.push(tok.to_owned());
    }

    let filtered: Vec<String> = all_tokens
        .iter()
        .filter(|tok| tok.chars().count() >= 2 && !is_stopword(tok))
        .cloned()
        .collect();

    if filtered.is_empty() {
        all_tokens
    } else {
        filtered
    }
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "de" | "do"
            | "da"
            | "dos"
            | "das"
            | "em"
            | "no"
            | "na"
            | "nos"
            | "nas"
            | "um"
            | "uma"
            | "uns"
            | "umas"
            | "os"
            | "as"
            | "ao"
            | "aos"
            | "ou"
            | "se"
            | "qual"
            | "quais"
            | "que"
            | "quem"
            | "onde"
            | "quando"
            | "como"
            | "para"
            | "por"
            | "pelo"
            | "pela"
            | "com"
            | "sem"
            | "the"
            | "of"
            | "in"
            | "to"
            | "for"
            | "is"
            | "are"
            | "what"
    )
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

    #[tokio::test]
    async fn natural_language_shortlist_query_ranks_target_fact_first()
    -> Result<(), Box<dyn std::error::Error>> {
        let mem = IndexedRedbMemory::open_in_memory()?;
        mem.upsert_batch(&[
            Fact {
                id: FactId::from("fact-preco-001"),
                subject: "plano_001".to_owned(),
                relation: "preco_mensal".to_owned(),
                object: "49.00".to_owned(),
                source_ref: "test".to_owned(),
            },
            Fact {
                id: FactId::from("fact-inedito-9842"),
                subject: "plano_enterprise".to_owned(),
                relation: "preco_mensal".to_owned(),
                object: "749.50".to_owned(),
                source_ref: "test".to_owned(),
            },
        ])?;

        let hits = mem
            .search_facts("Qual o(a) preco_mensal de plano_enterprise?", 12)
            .await?;
        assert!(!hits.is_empty());
        assert_eq!(hits[0].id, FactId::from("fact-inedito-9842"));

        // Substring parcial dentro de pergunta natural também cai no fallback FST por token
        let partial = mem.search_facts("Qual o plano enterpr?", 5).await?;
        assert!(
            partial
                .iter()
                .any(|f| f.id == FactId::from("fact-inedito-9842"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn replacing_fact_id_removes_stale_entry_from_tantivy()
    -> Result<(), Box<dyn std::error::Error>> {
        let mem = IndexedRedbMemory::open_in_memory()?;
        mem.upsert_fact(&Fact {
            id: FactId::from("fact-old"),
            subject: "plano_enterprise".to_owned(),
            relation: "preco_mensal".to_owned(),
            object: "500.00".to_owned(),
            source_ref: "v1".to_owned(),
        })?;

        // Atualiza o mesmo par (subject, relation) com novo FactId
        mem.upsert_fact(&Fact {
            id: FactId::from("fact-new"),
            subject: "plano_enterprise".to_owned(),
            relation: "preco_mensal".to_owned(),
            object: "749.50".to_owned(),
            source_ref: "v2".to_owned(),
        })?;

        let indexed_ids = mem.index.search_ids("plano_enterprise", 10)?;
        assert_eq!(indexed_ids, vec![FactId::from("fact-new")]);

        // Também limpa IDs antigos quando a substituição acontece via upsert_batch
        mem.upsert_batch(&[Fact {
            id: FactId::from("fact-v3"),
            subject: "plano_enterprise".to_owned(),
            relation: "preco_mensal".to_owned(),
            object: "899.00".to_owned(),
            source_ref: "v3".to_owned(),
        }])?;

        let indexed_after_batch = mem.index.search_ids("plano_enterprise", 10)?;
        assert_eq!(indexed_after_batch, vec![FactId::from("fact-v3")]);
        Ok(())
    }
}
