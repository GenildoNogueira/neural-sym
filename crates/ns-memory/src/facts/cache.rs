use dashmap::{DashMap, DashSet};
use ns_core::fact::{Fact, FactId};

use crate::facts::normalize_key;

/// Cache concorrente em RAM para respostas em sub-microssegundo (O(1)).
#[derive(Debug, Default)]
pub struct FactCache {
    by_pair: DashMap<(String, String), Fact>,
    known_ids: DashSet<FactId>,
}

impl FactCache {
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            by_pair: DashMap::with_capacity(capacity),
            known_ids: DashSet::with_capacity(capacity),
        }
    }

    #[must_use]
    pub fn get_pair(&self, entity: &str, relation: &str) -> Option<Fact> {
        let key = (normalize_key(entity), normalize_key(relation));
        self.by_pair.get(&key).map(|entry| entry.value().clone())
    }

    #[must_use]
    pub fn contains_id(&self, id: &FactId) -> bool {
        self.known_ids.contains(id)
    }

    pub fn insert(&self, fact: Fact) {
        let key = (normalize_key(&fact.subject), normalize_key(&fact.relation));
        let new_id = fact.id.clone();

        if let Some(previous) = self.by_pair.insert(key, fact)
            && previous.id != new_id
        {
            self.known_ids.remove(&previous.id);
        }

        self.known_ids.insert(new_id);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.by_pair.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_pair.is_empty()
    }
}
