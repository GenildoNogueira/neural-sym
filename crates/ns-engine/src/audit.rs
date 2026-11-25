use ns_core::action::LlmAction;
use ns_core::fact::{FactId, Message};
use ns_core::ports::LogicProof;
use ns_core::verification::{CandidateAnswer, Verified};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditOutcome {
    FactFound(FactId),
    FactNotFound { entity: String, relation: String },
    FactsSearched { query: String, count: usize },
    GraphTraversed { count: usize },
    MathEvaluated { value: String },
    LogicChecked { proof: LogicProof },
    SandboxExecuted { passed: bool, fuel_consumed: u64 },
    VerifierRejected { reason: String },
    MissingRequiredCitations,
    CitationRejected { missing: Vec<FactId> },
    AnswerVerified,
    ContextSearched { query: String, count: usize },
    SpansRead { span_ids: Vec<FactId> },
    SpanNotFound { missing: Vec<String> },
}

impl AuditOutcome {
    /// Indica se a ação foi barrada pelo orquestrador ou validador e exigiu reparo.
    #[must_use]
    pub const fn is_rejection(&self) -> bool {
        matches!(
            self,
            Self::CitationRejected { .. }
                | Self::MissingRequiredCitations
                | Self::VerifierRejected { .. }
                | Self::SpanNotFound { .. }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditStep {
    pub step_index: u8,
    pub history_snapshot: Vec<Message>,
    pub action: LlmAction,
    pub outcome: AuditOutcome,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AuditTrail {
    shortlist_facts: Vec<FactId>,
    steps: Vec<AuditStep>,
}

impl AuditTrail {
    pub(crate) fn record_shortlist(&mut self, fact_ids: Vec<FactId>) {
        self.shortlist_facts = fact_ids;
    }

    pub(crate) fn record(
        &mut self,
        step_index: u8,
        history_snapshot: Vec<Message>,
        action: LlmAction,
        outcome: AuditOutcome,
    ) {
        self.steps.push(AuditStep {
            step_index,
            history_snapshot,
            action,
            outcome,
        });
    }

    /// IDs dos fatos pré-injetados antes do passo 0 via shortlist (Peça D).
    #[must_use]
    pub fn shortlist_facts(&self) -> &[FactId] {
        &self.shortlist_facts
    }

    /// Quantidade de fatos pré-injetados antes do passo 0 via shortlist.
    #[must_use]
    pub fn shortlist_count(&self) -> usize {
        self.shortlist_facts.len()
    }

    #[must_use]
    pub fn steps(&self) -> &[AuditStep] {
        &self.steps
    }

    #[must_use]
    pub fn repair_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|step| step.outcome.is_rejection())
            .count()
    }
}

/// Resultado final contendo exclusivamente uma resposta no typestate [`Verified`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedOutcome {
    answer: CandidateAnswer<Verified>,
    audit: AuditTrail,
}

impl VerifiedOutcome {
    pub(crate) fn new(answer: CandidateAnswer<Verified>, audit: AuditTrail) -> Self {
        Self { answer, audit }
    }

    #[must_use]
    pub fn answer(&self) -> &CandidateAnswer<Verified> {
        &self.answer
    }

    #[must_use]
    pub fn audit(&self) -> &AuditTrail {
        &self.audit
    }
}
