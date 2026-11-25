use std::collections::HashSet;
use std::marker::PhantomData;

use crate::error::VerificationError;
use crate::fact::FactId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unverified;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verified;

/// Typestate que garante em tempo de compilação que só respostas verificadas saiam do motor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateAnswer<State> {
    text: String,
    cited_facts: Vec<FactId>,
    _state: PhantomData<State>,
}

impl CandidateAnswer<Unverified> {
    #[must_use]
    pub fn draft(text: String, cited_facts: Vec<FactId>) -> Self {
        Self {
            text,
            cited_facts,
            _state: PhantomData,
        }
    }

    /// Transição de estado: só promove para `Verified` se cada citação veio da memória nesta sessão.
    ///
    /// # Errors
    ///
    /// Retorna [`VerificationError::UncitedOrHallucinatedFacts`] se algum `FactId` citado
    /// não estiver presente em `retrieved_in_session`.
    pub fn verify_citations(
        self,
        retrieved_in_session: &HashSet<FactId>,
    ) -> Result<CandidateAnswer<Verified>, VerificationError> {
        let missing: Vec<FactId> = self
            .cited_facts
            .iter()
            .filter(|id| !retrieved_in_session.contains(*id))
            .cloned()
            .collect();

        if !missing.is_empty() {
            return Err(VerificationError::UncitedOrHallucinatedFacts { missing });
        }

        Ok(CandidateAnswer {
            text: self.text,
            cited_facts: self.cited_facts,
            _state: PhantomData,
        })
    }
}

impl CandidateAnswer<Verified> {
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    #[must_use]
    pub fn cited_facts(&self) -> &[FactId] {
        &self.cited_facts
    }
}
