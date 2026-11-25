use std::collections::VecDeque;
use std::sync::Mutex;

use ns_core::action::LlmAction;
use ns_core::error::LlmPortError;
use ns_core::fact::Message;
use ns_core::ports::ConstrainedLlm;

/// Implementação determinística de [`ConstrainedLlm`] para testes sem VRAM.
#[derive(Debug)]
pub struct MockLlm {
    queue: Mutex<VecDeque<Result<LlmAction, LlmPortError>>>,
    histories: Mutex<Vec<Vec<Message>>>,
}

impl MockLlm {
    /// Cria um mock a partir de uma sequência de ações válidas.
    #[must_use]
    pub fn from_actions(actions: impl IntoIterator<Item = LlmAction>) -> Self {
        let queue = actions.into_iter().map(Ok).collect();
        Self {
            queue: Mutex::new(queue),
            histories: Mutex::new(Vec::new()),
        }
    }

    /// Cria um mock permitindo intercalar ações válidas e erros simulados de decodificação.
    #[must_use]
    pub fn from_responses(
        responses: impl IntoIterator<Item = Result<LlmAction, LlmPortError>>,
    ) -> Self {
        Self {
            queue: Mutex::new(responses.into_iter().collect()),
            histories: Mutex::new(Vec::new()),
        }
    }

    /// Retorna todos os históricos recebidos em ordem cronológica de chamada.
    #[must_use]
    pub fn recorded_histories(&self) -> Vec<Vec<Message>> {
        self.histories
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Quantidade total de chamadas realizadas a [`ConstrainedLlm::next_action`].
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.histories
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

impl ConstrainedLlm for MockLlm {
    async fn next_action(&self, history: &[Message]) -> Result<LlmAction, LlmPortError> {
        {
            let mut recorded = self.histories.lock().map_err(|_| {
                LlmPortError::Decoding("mock llm history lock poisoned".to_owned())
            })?;
            recorded.push(history.to_vec());
        }

        let next = {
            let mut queue = self
                .queue
                .lock()
                .map_err(|_| LlmPortError::Decoding("mock llm queue lock poisoned".to_owned()))?;
            queue.pop_front()
        };

        let result = next.unwrap_or_else(|| {
            Err(LlmPortError::Decoding(
                "mock llm action queue exhausted".to_owned(),
            ))
        });

        tracing::debug!(?result, history_len = history.len(), "mock llm emitted action");
        result
    }
}
