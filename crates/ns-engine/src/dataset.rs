use std::io::Write;

use ns_core::action::LlmAction;
use ns_core::fact::{FactId, Message, Role};
use serde::{Deserialize, Serialize};

use crate::audit::VerifiedOutcome;
use crate::error::EngineError;

/// Um turno supervisionado extraído de uma trajetória que cruzou o typestate `Verified`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingSample {
    pub trajectory_id: String,
    pub scenario: String,
    pub step_index: u8,
    pub is_repair_recovery: bool,
    /// Prompt formatado byte a byte no mesmo padrão usado por `ns-llm::RwkvLlm`.
    pub prompt: String,
    pub action: LlmAction,
    pub cited_facts: Vec<FactId>,
}

/// Formata o histórico no mesmo layout de `ns-llm::RwkvLlm` (zero train/serve skew).
#[must_use]
pub fn format_history_prompt(history: &[Message]) -> String {
    let mut out = String::with_capacity(256);
    for msg in history {
        let prefix = match msg.role {
            Role::System => "SYSTEM: ",
            Role::User => "USER: ",
            Role::Assistant => "ASSISTANT: ",
            Role::Tool => "TOOL: ",
        };
        out.push_str(prefix);
        out.push_str(&msg.content);
        out.push('\n');
    }
    out
}

/// Extrai amostras SFT válidas de um [`VerifiedOutcome`].
///
/// Regras de qualidade:
/// - Só aceita [`VerifiedOutcome`], garantindo por tipo que a trajetória final foi verificada.
/// - Exclui como alvo os turnos rejeitados (`CitationRejected`, `VerifierRejected`, etc.) para não
///   ensinar o modelo a emitir erros.
/// - Mantém os turnos seguintes contendo `REPAIR REQUIRED` no histórico com `is_repair_recovery = true`,
///   ensinando o modelo a se autocorrigir.
#[must_use]
pub fn extract_verified_samples(
    trajectory_id: impl Into<String>,
    scenario: impl Into<String>,
    outcome: &VerifiedOutcome,
) -> Vec<TrainingSample> {
    let trajectory_id = trajectory_id.into();
    let scenario = scenario.into();
    let cited_facts = outcome.answer().cited_facts().to_vec();
    let steps = outcome.audit().steps();

    let mut samples = Vec::with_capacity(steps.len());

    for (idx, step) in steps.iter().enumerate() {
        if step.outcome.is_rejection() {
            continue;
        }

        let is_repair_recovery = idx > 0
            && steps
                .get(idx.saturating_sub(1))
                .is_some_and(|prev| prev.outcome.is_rejection());

        samples.push(TrainingSample {
            trajectory_id: trajectory_id.clone(),
            scenario: scenario.clone(),
            step_index: step.step_index,
            is_repair_recovery,
            prompt: format_history_prompt(&step.history_snapshot),
            action: step.action.clone(),
            cited_facts: cited_facts.clone(),
        });
    }

    samples
}

/// Escreve uma coleção de [`TrainingSample`] no formato JSON Lines (1 objeto JSON por linha).
///
/// # Errors
///
/// Retorna [`EngineError::Serialization`] se algum exemplo falhar na serialização, ou erro de I/O.
pub fn write_samples_jsonl<W: Write>(
    writer: &mut W,
    samples: &[TrainingSample],
) -> Result<usize, std::io::Error> {
    for sample in samples {
        let line = serde_json::to_string(sample)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
        writeln!(writer, "{line}")?;
    }
    Ok(samples.len())
}

/// Mantém referência explícita a [`EngineError`] para conveniência de chamadores.
#[must_use]
pub const fn is_recoverable_for_dataset(err: &EngineError) -> bool {
    matches!(err, EngineError::MaxStepsExceeded { .. })
}
