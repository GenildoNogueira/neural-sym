use std::collections::{HashMap, HashSet};

use ns_core::error::MemoryPortError;
use ns_core::fact::{DocumentSpan, FactId};
use ns_core::ports::DocumentContext;
#[cfg(feature = "tantivy-search")]
use tantivy::collector::TopDocs;
#[cfg(feature = "tantivy-search")]
use tantivy::query::QueryParser;
#[cfg(feature = "tantivy-search")]
use tantivy::schema::{Field, STORED, STRING, Schema, TEXT, Value};
#[cfg(feature = "tantivy-search")]
use tantivy::{Index, IndexReader, ReloadPolicy, TantivyDocument, doc};

use crate::error::MemoryError;

const FALLBACK_CHUNK_MAX_CHARS: usize = 700;
const MAX_SEARCH_SPANS: u8 = 16;
const MAX_READ_SPANS: usize = 6;

#[cfg(feature = "tantivy-search")]
struct RamTantivyIndex {
    index: Index,
    reader: IndexReader,
    id_field: Field,
    title_field: Field,
    text_field: Field,
}

/// Memória efêmera de sessão para documentos longos (contratos, manuais, logs):
/// fatia em cláusulas/seções, constrói o grafo de remissões e indexa em RAM via Tantivy.
pub struct SessionDocumentStore {
    spans: Vec<DocumentSpan>,
    by_id: HashMap<String, usize>,
    #[cfg(feature = "tantivy-search")]
    ram_index: Option<RamTantivyIndex>,
}

impl std::fmt::Debug for SessionDocumentStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionDocumentStore")
            .field("span_count", &self.spans.len())
            .finish_non_exhaustive()
    }
}

impl SessionDocumentStore {
    /// Segmenta `raw_text` em seções/cláusulas, extrai remissões cruzadas e indexa em RAM.
    ///
    /// # Errors
    ///
    /// Retorna [`MemoryError`] se a criação ou commit do índice Tantivy em RAM falhar.
    pub fn from_text(raw_text: &str) -> Result<Self, MemoryError> {
        let mut spans = split_into_spans(raw_text);
        let mut by_id = HashMap::with_capacity(spans.len());
        for (idx, span) in spans.iter().enumerate() {
            by_id.insert(span.id.as_str().to_lowercase(), idx);
        }

        // Segunda passagem: monta o grafo de remissões entre seções existentes
        for idx in 0..spans.len() {
            let own_key = spans[idx].id.as_str().to_lowercase();
            let refs = extract_cross_references(&spans[idx].text, &own_key, &by_id);
            spans[idx].references = refs;
        }

        #[cfg(feature = "tantivy-search")]
        let ram_index = if spans.is_empty() {
            None
        } else {
            Some(build_ram_tantivy_index(&spans)?)
        };

        Ok(Self {
            spans,
            by_id,
            #[cfg(feature = "tantivy-search")]
            ram_index,
        })
    }

    /// Retorna todas as seções indexadas na ordem original do documento.
    #[must_use]
    pub fn spans(&self) -> &[DocumentSpan] {
        &self.spans
    }

    /// Quantidade de seções indexadas.
    #[must_use]
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Indica se o documento não produziu nenhuma seção.
    #[must_use]
    pub fn is_empty_store(&self) -> bool {
        self.spans.is_empty()
    }

    /// Busca por ID exato (case-insensitive).
    #[must_use]
    pub fn get_span(&self, id: &str) -> Option<&DocumentSpan> {
        let key = id.trim().to_lowercase();
        self.by_id.get(&key).and_then(|&idx| self.spans.get(idx))
    }

    fn search_spans_inner(&self, query: &str, limit: u8) -> Result<Vec<DocumentSpan>, MemoryError> {
        let needle = query.trim().to_lowercase();
        if needle.is_empty() || limit == 0 || self.spans.is_empty() {
            return Ok(Vec::new());
        }

        let cap = usize::from(limit.min(MAX_SEARCH_SPANS));
        let mut results = Vec::with_capacity(cap);
        let mut seen = HashSet::with_capacity(cap);

        // 1. Fast-path BM25 em RAM via Tantivy (quando compilado com tantivy-search)
        #[cfg(feature = "tantivy-search")]
        if let Some(ref idx) = self.ram_index {
            let searcher = idx.reader.searcher();
            let parser = QueryParser::for_index(&idx.index, vec![idx.title_field, idx.text_field]);
            let (parsed, _errors) = parser.parse_query_lenient(&needle);
            let hits = searcher.search(&parsed, &TopDocs::with_limit(cap))?;

            for (_score, addr) in hits {
                let doc: TantivyDocument = searcher.doc(addr)?;
                if let Some(raw_id) = doc.get_first(idx.id_field).and_then(|v| v.as_str()) {
                    let key = raw_id.to_lowercase();
                    if seen.insert(key.clone())
                        && let Some(&pos) = self.by_id.get(&key) {
                            results.push(self.spans[pos].clone());
                        }
                }
            }
        }

        // 2. Fallback léxico leve (cobre substrings literais e builds sem tantivy-search)
        if results.len() < cap {
            let terms: Vec<&str> = needle.split_whitespace().collect();
            let mut scored: Vec<(usize, usize)> = Vec::new();

            for (pos, span) in self.spans.iter().enumerate() {
                let key = span.id.as_str().to_lowercase();
                if seen.contains(&key) {
                    continue;
                }
                let hay_id = key;
                let hay_title = span.title.to_lowercase();
                let hay_text = span.text.to_lowercase();

                let mut score: usize = 0;
                if hay_id.contains(&needle) || hay_title.contains(&needle) {
                    score = score.saturating_add(4);
                }
                if hay_text.contains(&needle) {
                    score = score.saturating_add(3);
                }
                for term in &terms {
                    if term.len() >= 2
                        && (hay_title.contains(term)
                            || hay_text.contains(term)
                            || hay_id.contains(term))
                    {
                        score = score.saturating_add(1);
                    }
                }
                if score > 0 {
                    scored.push((score, pos));
                }
            }

            scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            for (_score, pos) in scored {
                if results.len() >= cap {
                    break;
                }
                results.push(self.spans[pos].clone());
            }
        }

        Ok(results)
    }
}

impl DocumentContext for SessionDocumentStore {
    fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    fn table_of_contents(&self) -> String {
        let mut out = String::with_capacity(self.spans.len().saturating_mul(48));
        for span in &self.spans {
            out.push('[');
            out.push_str(span.id.as_str());
            out.push_str("] ");
            out.push_str(&span.title);
            if !span.references.is_empty() {
                out.push_str(" (refs: ");
                for (i, r) in span.references.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(r.as_str());
                }
                out.push(')');
            }
            out.push('\n');
        }
        out
    }

    fn search_spans(
        &self,
        query: &str,
        limit: u8,
    ) -> Result<Vec<DocumentSpan>, MemoryPortError> {
        let found = self.search_spans_inner(query, limit)?;
        Ok(found)
    }

    fn read_spans(&self, span_ids: &[String]) -> Result<Vec<DocumentSpan>, MemoryPortError> {
        let mut out = Vec::with_capacity(span_ids.len().min(MAX_READ_SPANS));
        let mut seen = HashSet::with_capacity(span_ids.len().min(MAX_READ_SPANS));

        for raw_id in span_ids.iter().take(MAX_READ_SPANS) {
            let key = raw_id.trim().to_lowercase();
            if seen.insert(key.clone())
                && let Some(&pos) = self.by_id.get(&key) {
                    out.push(self.spans[pos].clone());
                }
        }

        Ok(out)
    }
}

#[cfg(feature = "tantivy-search")]
fn build_ram_tantivy_index(spans: &[DocumentSpan]) -> Result<RamTantivyIndex, MemoryError> {
    let mut builder = Schema::builder();
    let id_field = builder.add_text_field("id", STRING | STORED);
    let title_field = builder.add_text_field("title", TEXT);
    let text_field = builder.add_text_field("text", TEXT);
    let schema = builder.build();

    let index = Index::create_in_ram(schema);
    {
        // Writer é descartado logo após o commit para liberar o buffer de indexação em embarcados
        let mut writer = index.writer::<TantivyDocument>(15_000_000)?;
        for span in spans {
            writer.add_document(doc!(
                id_field => span.id.as_str(),
                title_field => span.title.as_str(),
                text_field => span.text.as_str(),
            ))?;
        }
        writer.commit()?;
    }

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()?;
    reader.reload()?;

    Ok(RamTantivyIndex {
        index,
        reader,
        id_field,
        title_field,
        text_field,
    })
}

fn split_into_spans(raw_text: &str) -> Vec<DocumentSpan> {
    let mut spans: Vec<DocumentSpan> = Vec::new();
    let mut used_ids: HashSet<String> = HashSet::new();

    let mut current_id: Option<String> = None;
    let mut current_title = String::new();
    let mut current_body = String::new();
    let mut heading_count: usize = 0;

    let flush_current = |spans: &mut Vec<DocumentSpan>,
                         used_ids: &mut HashSet<String>,
                         cur_id: &mut Option<String>,
                         cur_title: &mut String,
                         cur_body: &mut String| {
        if let Some(base_id) = cur_id.take() {
            let trimmed_body = cur_body.trim();
            if !trimmed_body.is_empty() || !cur_title.is_empty() {
                let unique_id = ensure_unique_id(&base_id, used_ids);
                let text = if trimmed_body.is_empty() {
                    cur_title.clone()
                } else {
                    format!("{}\n{trimmed_body}", cur_title.trim())
                };
                spans.push(DocumentSpan {
                    id: FactId::new(unique_id),
                    title: truncate_title(cur_title.trim(), 72),
                    text,
                    references: Vec::new(),
                });
            }
        }
        cur_title.clear();
        cur_body.clear();
    };

    for line in raw_text.lines() {
        let trimmed = line.trim();
        if let Some((parsed_id, parsed_title)) = parse_heading_line(trimmed, heading_count + 1) {
            flush_current(
                &mut spans,
                &mut used_ids,
                &mut current_id,
                &mut current_title,
                &mut current_body,
            );
            heading_count = heading_count.saturating_add(1);
            current_id = Some(parsed_id);
            current_title = parsed_title;
        } else if current_id.is_some() {
            if !current_body.is_empty() {
                current_body.push('\n');
            }
            current_body.push_str(trimmed);
        } else if !trimmed.is_empty() {
            heading_count = heading_count.saturating_add(1);
            current_id = Some(format!("sec_{heading_count:02}"));
            current_title = truncate_title(trimmed, 72);
            current_body.push_str(trimmed);
        }
    }

    flush_current(
        &mut spans,
        &mut used_ids,
        &mut current_id,
        &mut current_title,
        &mut current_body,
    );

    // Se o documento não tinha cabeçalhos estruturais e virou 1 bloco gigante, fatia por parágrafos
    if spans.len() == 1 && spans[0].text.chars().count() > FALLBACK_CHUNK_MAX_CHARS {
        return split_by_paragraph_budget(raw_text);
    }

    spans
}

fn split_by_paragraph_budget(raw_text: &str) -> Vec<DocumentSpan> {
    let mut spans = Vec::new();
    let mut chunk = String::new();
    let mut seq: usize = 0;

    let push_chunk = |spans: &mut Vec<DocumentSpan>, seq: &mut usize, buf: &mut String| {
        let trimmed = buf.trim();
        if trimmed.is_empty() {
            buf.clear();
            return;
        }
        *seq = seq.saturating_add(1);
        let first_line = trimmed.lines().next().unwrap_or(trimmed);
        spans.push(DocumentSpan {
            id: FactId::new(format!("sec_{:02}", *seq)),
            title: truncate_title(first_line, 72),
            text: trimmed.to_owned(),
            references: Vec::new(),
        });
        buf.clear();
    };

    for para in raw_text.split("\n\n") {
        let p = para.trim();
        if p.is_empty() {
            continue;
        }
        if !chunk.is_empty()
            && chunk.chars().count().saturating_add(p.chars().count()) > FALLBACK_CHUNK_MAX_CHARS
        {
            push_chunk(&mut spans, &mut seq, &mut chunk);
        }
        if !chunk.is_empty() {
            chunk.push_str("\n\n");
        }
        chunk.push_str(p);
    }
    push_chunk(&mut spans, &mut seq, &mut chunk);
    spans
}

fn parse_heading_line(line: &str, fallback_seq: usize) -> Option<(String, String)> {
    if line.is_empty() {
        return None;
    }

    // Cabeçalhos Markdown (#, ##, ###)
    if let Some(rest) = line
        .strip_prefix("### ")
        .or_else(|| line.strip_prefix("## "))
        .or_else(|| line.strip_prefix("# "))
    {
        let title = rest.trim();
        if !title.is_empty() {
            let id = extract_leading_structured_id(title)
                .unwrap_or_else(|| format!("sec_{fallback_seq:02}"));
            return Some((id, title.to_owned()));
        }
    }

    // Cabeçalhos jurídicos/técnicos (Cláusula 2, Art. 15, Seção 4)
    if let Some(id) = extract_leading_structured_id(line) {
        return Some((id, line.to_owned()));
    }

    None
}

fn extract_leading_structured_id(text: &str) -> Option<String> {
    let lower = text.trim_start().to_lowercase();
    let prefixes = [
        ("cláusula", "cl"),
        ("clausula", "cl"),
        ("artigo", "art"),
        ("art.", "art"),
        ("seção", "sec"),
        ("secao", "sec"),
        ("capítulo", "cap"),
        ("capitulo", "cap"),
    ];

    for (kw, tag) in prefixes {
        if let Some(rest) = lower.strip_prefix(kw)
            && let Some(num) = read_first_number(rest) {
                return Some(format!("{tag}_{num:02}"));
            }
    }
    None
}

fn extract_cross_references(
    text: &str,
    own_key: &str,
    by_id: &HashMap<String, usize>,
) -> Vec<FactId> {
    let lower = text.to_lowercase();
    let keywords = [
        ("cláusula", "cl"),
        ("clausula", "cl"),
        ("artigo", "art"),
        ("art.", "art"),
        ("seção", "sec"),
        ("secao", "sec"),
    ];

    let mut refs = Vec::new();
    let mut seen = HashSet::new();

    for (kw, tag) in keywords {
        let mut cursor: usize = 0;
        while let Some(rel_pos) = lower[cursor..].find(kw) {
            let after_kw = cursor + rel_pos + kw.len();
            if let Some(num) = read_first_number(&lower[after_kw..]) {
                let candidate = format!("{tag}_{num:02}");
                if candidate != own_key
                    && by_id.contains_key(&candidate)
                    && seen.insert(candidate.clone())
                {
                    refs.push(FactId::new(candidate));
                }
            }
            cursor = after_kw;
        }
    }

    refs
}

fn read_first_number(slice: &str) -> Option<usize> {
    let trimmed = slice.trim_start_matches(|c: char| {
        c.is_whitespace() || matches!(c, '.' | ':' | '-' | 'º' | '°' | 'ª' | 'n' | 'N')
    });
    let digits: String = trimmed.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<usize>().ok()
}

fn ensure_unique_id(base: &str, used: &mut HashSet<String>) -> String {
    let key = base.to_lowercase();
    if used.insert(key.clone()) {
        return key;
    }
    let mut suffix: usize = 2;
    loop {
        let candidate = format!("{key}_{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        suffix = suffix.saturating_add(1);
    }
}

fn truncate_title(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let taken: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{taken}...")
    } else {
        taken
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_contract_extracts_cross_references_and_searches_spans()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut doc = String::new();
        doc.push_str("Cláusula 1 - Objeto\nPrestação de serviços de manutenção embarcada.\n\n");
        doc.push_str(
            "Cláusula 2 - Prazo Máximo\nO prazo máximo de entrega é de 15 dias corridos, salvo disposto na Cláusula 47.\n\n",
        );
        for i in 3..=46 {
            doc.push_str(&format!(
                "Cláusula {i} - Disposição Operacional {i}\nRegras padrão de operação do módulo {i}.\n\n"
            ));
        }
        doc.push_str(
            "Cláusula 47 - Exceção de Logística\nConforme Cláusula 2, para peças importadas a entrega ocorrerá em 30 dias.\n",
        );

        let store = SessionDocumentStore::from_text(&doc)?;
        assert_eq!(store.len(), 47);

        let cl2 = store.get_span("cl_02").ok_or("missing cl_02")?;
        assert_eq!(cl2.references, vec![FactId::from("cl_47")]);

        let cl47 = store.get_span("cl_47").ok_or("missing cl_47")?;
        assert_eq!(cl47.references, vec![FactId::from("cl_02")]);

        let hits = store.search_spans("peças importadas", 3)?;
        assert!(hits.iter().any(|s| s.id == FactId::from("cl_47")));

        let side_by_side = store.read_spans(&["cl_02".to_owned(), "cl_47".to_owned()])?;
        assert_eq!(side_by_side.len(), 2);
        assert_eq!(side_by_side[0].id, FactId::from("cl_02"));
        assert_eq!(side_by_side[1].id, FactId::from("cl_47"));

        Ok(())
    }
}
