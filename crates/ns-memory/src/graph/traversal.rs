use std::collections::{HashSet, VecDeque};

use ns_core::fact::{Fact, FactId};

use crate::error::MemoryError;
use crate::facts::{FactCache, FactRepository, normalize_key};

/// Percorre o grafo de fatos em largura (BFS) até `max_hops`, evitando ciclos
/// e alimentando o cache em RAM com cada fato descoberto.
///
/// # Errors
///
/// Retorna [`MemoryError::Sqlite`] se alguma consulta de vizinhança falhar.
pub async fn traverse_bfs(
    repo: &FactRepository,
    cache: &FactCache,
    start_entity: &str,
    max_hops: u8,
) -> Result<Vec<Fact>, MemoryError> {
    if max_hops == 0 {
        return Ok(Vec::new());
    }

    let start = normalize_key(start_entity);
    if start.is_empty() {
        return Ok(Vec::new());
    }

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

        let outgoing = repo.find_by_subject(&current_entity).await?;
        for fact in outgoing {
            let next_entity = normalize_key(&fact.object);
            if !next_entity.is_empty() && visited_entities.insert(next_entity.clone()) {
                queue.push_back((next_entity, depth.saturating_add(1)));
            }

            if seen_fact_ids.insert(fact.id.clone()) {
                cache.insert(fact.clone());
                collected.push(fact);
            }
        }
    }

    Ok(collected)
}
