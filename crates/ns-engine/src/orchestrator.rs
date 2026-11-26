use std::collections::{HashMap, HashSet};

use ns_core::action::LlmAction;
use ns_core::error::VerificationError;
use ns_core::fact::{Fact, FactId, Message};
use ns_core::ports::{
    ConstrainedLlm, DeterministicMemory, DocumentContext, LogicProof, NoDocumentContext,
    SymbolicVerifier,
};
use ns_core::verification::CandidateAnswer;
use serde_json::json;

use crate::audit::{AuditOutcome, AuditTrail, VerifiedOutcome};
use crate::error::EngineError;

/// Orquestra o ciclo determinístico Generate → Verify → Repair.
#[derive(Debug)]
pub struct Orchestrator<L, M, V> {
    llm: L,
    memory: M,
    verifier: V,
    max_steps: u8,
    require_citations: bool,
    shortlist_limit: u8,
    verify_answer_content: bool,
}

impl<L, M, V> Orchestrator<L, M, V>
where
    L: ConstrainedLlm,
    M: DeterministicMemory,
    V: SymbolicVerifier,
{
    #[must_use]
    pub fn new(llm: L, memory: M, verifier: V, max_steps: u8) -> Self {
        Self {
            llm,
            memory,
            verifier,
            max_steps,
            require_citations: false,
            shortlist_limit: 0,
            verify_answer_content: true,
        }
    }

    /// Quando `true`, impede respostas finais sem ao menos um `FactId` consultado na sessão.
    #[must_use]
    pub fn with_require_citations(mut self, required: bool) -> Self {
        self.require_citations = required;
        self
    }

    /// Define quantos fatos candidatos da memória determinística são pré-injetados
    /// antes do passo 0 como `CANDIDATES:` (Peça D: Retrieval-as-Decoding). `0` desativa.
    #[must_use]
    pub fn with_shortlist_limit(mut self, limit: u8) -> Self {
        self.shortlist_limit = limit;
        self
    }

    /// Quando `true` (padrão), exige que a resposta final contenha o valor real (`object`)
    /// dos fatos citados ou o resultado do `EvalMath`, bloqueando respostas degeneradas (ex.: `"O"`).
    #[must_use]
    pub fn with_verify_answer_content(mut self, enabled: bool) -> Self {
        self.verify_answer_content = enabled;
        self
    }

    /// Executa o loop neuro-simbólico e só retorna após a resposta cruzar o typestate `Verified`.
    ///
    /// # Errors
    ///
    /// Retorna:
    /// - [`EngineError::EmptyPrompt`] se o prompt estiver vazio ou só contiver espaços.
    /// - [`EngineError::Llm`] se o backend do LLM falhar na decodificação restrita.
    /// - [`EngineError::Memory`] se a consulta à memória determinística falhar.
    /// - [`EngineError::Serialization`] se falhar ao serializar o retorno de ferramenta.
    /// - [`EngineError::MaxStepsExceeded`] se o modelo não produzir uma resposta verificada dentro do limite de passos.
    pub async fn answer(&self, prompt: &str) -> Result<VerifiedOutcome, EngineError> {
        self.answer_with_context(prompt, &NoDocumentContext).await
    }

    /// Executa o loop neuro-simbólico com memória efêmera de documentos anexada à sessão.
    ///
    /// # Errors
    ///
    /// Retorna os mesmos erros de [`Self::answer`].
    pub async fn answer_with_context<D: DocumentContext>(
        &self,
        prompt: &str,
        doc: &D,
    ) -> Result<VerifiedOutcome, EngineError> {
        if prompt.trim().is_empty() {
            return Err(EngineError::EmptyPrompt);
        }

        let mut history = Vec::with_capacity(8);
        let mut session_facts: HashSet<FactId> = HashSet::new();
        let mut session_fact_records: HashMap<FactId, Fact> = HashMap::new();
        let mut last_math_value: Option<String> = None;
        let mut audit = AuditTrail::default();

        if !doc.is_empty() {
            history.push(Message::system(format!(
                "CONTEXTO INDEXADO NA SESSAO (use search_context ou read_spans para comparar trechos lado a lado):\n{}",
                doc.table_of_contents()
            )));
        }

        // Peça D (Retrieval-as-Decoding): pré-injeta top-K fatos do índice ACID antes do passo 0.
        if self.shortlist_limit > 0 {
            let candidates = self
                .memory
                .search_facts(prompt, self.shortlist_limit)
                .await?;
            if !candidates.is_empty() {
                let mut candidate_ids = Vec::with_capacity(candidates.len());
                let mut payload = Vec::with_capacity(candidates.len());

                for fact in &candidates {
                    session_facts.insert(fact.id.clone());
                    session_fact_records.insert(fact.id.clone(), fact.clone());
                    candidate_ids.push(fact.id.clone());
                    payload.push(json!({
                        "id": fact.id.as_str(),
                        "subject": fact.subject,
                        "relation": fact.relation,
                        "object": fact.object,
                    }));
                }

                let serialized = serde_json::to_string(&payload)?;
                tracing::debug!(
                    shortlist_count = candidate_ids.len(),
                    "orchestrator injected candidate shortlist before step 0"
                );
                audit.record_shortlist(candidate_ids);
                history.push(Message::system(format!("CANDIDATES: {serialized}")));
            }
        }

        history.push(Message::user(prompt));

        let mut repair_attempts: u8 = 0;

        for step_index in 0..self.max_steps {
            let action = self.llm.next_action(&history).await?;
            let snapshot = history.clone();
            tracing::debug!(step_index, ?action, "orchestrator received llm action");

            match action.clone() {
                LlmAction::QueryFact { entity, relation } => {
                    let fact = self.memory.lookup_fact(&entity, &relation).await?;
                    let payload = match &fact {
                        Some(found) => {
                            session_facts.insert(found.id.clone());
                            session_fact_records.insert(found.id.clone(), found.clone());
                            audit.record(
                                step_index,
                                snapshot,
                                action,
                                AuditOutcome::FactFound(found.id.clone()),
                            );
                            serde_json::to_string(&json!({
                                "status": "found",
                                "fact": found,
                            }))?
                        }
                        None => {
                            audit.record(
                                step_index,
                                snapshot,
                                action,
                                AuditOutcome::FactNotFound {
                                    entity: entity.clone(),
                                    relation: relation.clone(),
                                },
                            );
                            serde_json::to_string(&json!({
                                "status": "not_found",
                                "entity": entity,
                                "relation": relation,
                            }))?
                        }
                    };
                    history.push(Message::tool(payload));
                }

                LlmAction::TraverseGraph {
                    start_entity,
                    max_hops,
                } => {
                    let facts = self.memory.traverse(&start_entity, max_hops).await?;
                    for fact in &facts {
                        session_facts.insert(fact.id.clone());
                        session_fact_records.insert(fact.id.clone(), fact.clone());
                    }
                    audit.record(
                        step_index,
                        snapshot,
                        action,
                        AuditOutcome::GraphTraversed { count: facts.len() },
                    );
                    let payload = serde_json::to_string(&json!({
                        "status": "ok",
                        "facts": facts,
                    }))?;
                    history.push(Message::tool(payload));
                }

                LlmAction::EvalMath { expression } => match self.verifier.eval_decimal(&expression)
                {
                    Ok(value) => {
                        let value_str = value.to_string();
                        last_math_value = Some(value_str.clone());
                        audit.record(
                            step_index,
                            snapshot,
                            action,
                            AuditOutcome::MathEvaluated {
                                value: value_str.clone(),
                            },
                        );
                        let payload = serde_json::to_string(&json!({
                            "status": "ok",
                            "value": value_str,
                        }))?;
                        history.push(Message::tool(payload));
                    }
                    Err(err) => {
                        repair_attempts = repair_attempts.saturating_add(1);
                        let reason = err.to_string();
                        tracing::warn!(step_index, %reason, "math verifier rejected expression");
                        audit.record(
                            step_index,
                            snapshot,
                            action,
                            AuditOutcome::VerifierRejected {
                                reason: reason.clone(),
                            },
                        );
                        history.push(Message::system(format!(
                            "REPAIR REQUIRED (math): {reason}. Corrija a expressão e reenvie EvalMath."
                        )));
                    }
                },

                LlmAction::CheckLogic { smtlib2_script } => {
                    match self.verifier.check_smt(&smtlib2_script) {
                        Ok(proof) => {
                            let payload = serialize_logic_proof(&proof)?;
                            audit.record(
                                step_index,
                                snapshot,
                                action,
                                AuditOutcome::LogicChecked { proof },
                            );
                            history.push(Message::tool(payload));
                        }
                        Err(err) => {
                            repair_attempts = repair_attempts.saturating_add(1);
                            let reason = err.to_string();
                            tracing::warn!(step_index, %reason, "logic verifier rejected script");
                            audit.record(
                                step_index,
                                snapshot,
                                action,
                                AuditOutcome::VerifierRejected {
                                    reason: reason.clone(),
                                },
                            );
                            history.push(Message::system(format!(
                                "REPAIR REQUIRED (logic): {reason}. Corrija o script SMT-LIB2."
                            )));
                        }
                    }
                }

                LlmAction::RunWasmCheck {
                    wasm_wat_source,
                    expected_stdout,
                } => match self
                    .verifier
                    .run_sandbox(&wasm_wat_source, &expected_stdout)
                {
                    Ok(output) => {
                        audit.record(
                            step_index,
                            snapshot,
                            action,
                            AuditOutcome::SandboxExecuted {
                                passed: output.passed,
                                fuel_consumed: output.fuel_consumed,
                            },
                        );
                        let payload = serde_json::to_string(&json!({
                            "status": "ok",
                            "passed": output.passed,
                            "stdout": output.stdout,
                            "fuel_consumed": output.fuel_consumed,
                        }))?;
                        history.push(Message::tool(payload));
                    }
                    Err(err) => {
                        repair_attempts = repair_attempts.saturating_add(1);
                        let reason = err.to_string();
                        tracing::warn!(step_index, %reason, "sandbox verifier rejected code");
                        audit.record(
                            step_index,
                            snapshot,
                            action,
                            AuditOutcome::VerifierRejected {
                                reason: reason.clone(),
                            },
                        );
                        history.push(Message::system(format!(
                            "REPAIR REQUIRED (sandbox): {reason}. Corrija o módulo WASM."
                        )));
                    }
                },

                LlmAction::ProposeAnswer { text, cited_facts } => {
                    if self.require_citations && cited_facts.is_empty() {
                        repair_attempts = repair_attempts.saturating_add(1);
                        tracing::warn!(step_index, "blocked uncited answer in grounded mode");
                        audit.record(
                            step_index,
                            snapshot,
                            action,
                            AuditOutcome::MissingRequiredCitations,
                        );
                        history.push(Message::system(
                            "REPAIR REQUIRED (citations): cite ao menos um FactId consultado na memória determinística antes de propor a resposta.",
                        ));
                        continue;
                    }

                    // 1. Verifica primeiro se todos os FactIds citados foram realmente recuperados.
                    let citation_check =
                        CandidateAnswer::draft(text.clone(), cited_facts.clone())
                            .verify_citations(&session_facts);
                    if let Err(VerificationError::UncitedOrHallucinatedFacts { missing }) =
                        citation_check
                    {
                        repair_attempts = repair_attempts.saturating_add(1);
                        let missing_list: Vec<&str> =
                            missing.iter().map(FactId::as_str).collect();
                        tracing::warn!(
                            step_index,
                            ?missing_list,
                            "blocked answer citing unretrieved or hallucinated facts"
                        );
                        audit.record(
                            step_index,
                            snapshot,
                            action,
                            AuditOutcome::CitationRejected {
                                missing: missing.clone(),
                            },
                        );
                        history.push(Message::system(format!(
                            "REPAIR REQUIRED (hallucinated facts): os IDs {:?} não foram recuperados da memória determinística nesta sessão. Consulte via QueryFact ou TraverseGraph antes de responder.",
                            missing_list
                        )));
                        continue;
                    }

                    // 2. Expande slots/templates determinísticos (permite <ANSWER> sem chamar a head 65k).
                    let rendered_text = render_slot_template(
                        &text,
                        &cited_facts,
                        &session_fact_records,
                        last_math_value.as_deref(),
                    );

                    // 3. Fecha o buraco de conteúdo: garante valor do fato citado / EvalMath no texto.
                    if self.verify_answer_content
                        && let Err(reason) = verify_answer_grounding(
                            &rendered_text,
                            &cited_facts,
                            &session_fact_records,
                            last_math_value.as_deref(),
                        )
                    {
                        repair_attempts = repair_attempts.saturating_add(1);
                        tracing::warn!(
                            step_index,
                            %reason,
                            "blocked answer missing grounded fact object or math value"
                        );
                        audit.record(
                            step_index,
                            snapshot,
                            action,
                            AuditOutcome::AnswerContentRejected {
                                reason: reason.clone(),
                            },
                        );
                        history.push(Message::system(format!(
                            "REPAIR REQUIRED (content): {reason}. Inclua o valor exato retornado ou use \"@slot\" com os FactIds citados."
                        )));
                        continue;
                    }

                    let verified = CandidateAnswer::draft(rendered_text, cited_facts)
                        .verify_citations(&session_facts)
                        .map_err(|_| EngineError::EmptyPrompt)?;
                    audit.record(
                        step_index,
                        snapshot,
                        action,
                        AuditOutcome::AnswerVerified,
                    );
                    return Ok(VerifiedOutcome::new(verified, audit));
                }

                LlmAction::SearchFacts { query, limit } => {
                    let facts = self.memory.search_facts(&query, limit).await?;
                    for fact in &facts {
                        session_facts.insert(fact.id.clone());
                        session_fact_records.insert(fact.id.clone(), fact.clone());
                    }
                    audit.record(
                        step_index,
                        snapshot,
                        action,
                        AuditOutcome::FactsSearched {
                            query: query.clone(),
                            count: facts.len(),
                        },
                    );
                    let payload = serde_json::to_string(&json!({
                        "status": "ok",
                        "query": query,
                        "count": facts.len(),
                        "facts": facts,
                    }))?;
                    history.push(Message::tool(payload));
                }

                LlmAction::SearchContext { query, limit } => {
                    let spans = doc.search_spans(&query, limit)?;
                    let summaries: Vec<serde_json::Value> = spans
                        .iter()
                        .map(|s| {
                            json!({
                                "id": s.id,
                                "title": s.title,
                                "references": s.references,
                            })
                        })
                        .collect();
                    audit.record(
                        step_index,
                        snapshot,
                        action,
                        AuditOutcome::ContextSearched {
                            query: query.clone(),
                            count: spans.len(),
                        },
                    );
                    let payload = serde_json::to_string(&json!({
                        "status": "ok",
                        "query": query,
                        "count": spans.len(),
                        "matches": summaries,
                    }))?;
                    history.push(Message::tool(payload));
                }

                LlmAction::ReadSpans { span_ids } => {
                    let spans = doc.read_spans(&span_ids)?;
                    let found_set: HashSet<String> =
                        spans.iter().map(|s| s.id.as_str().to_lowercase()).collect();
                    let missing: Vec<String> = span_ids
                        .iter()
                        .filter(|id| !found_set.contains(&id.trim().to_lowercase()))
                        .cloned()
                        .collect();

                    if spans.is_empty() {
                        repair_attempts = repair_attempts.saturating_add(1);
                        tracing::warn!(step_index, ?missing, "requested unknown document spans");
                        audit.record(
                            step_index,
                            snapshot,
                            action,
                            AuditOutcome::SpanNotFound {
                                missing: missing.clone(),
                            },
                        );
                        history.push(Message::system(format!(
                            "REPAIR REQUIRED (spans): os IDs {:?} não existem no índice do documento. Use apenas IDs listados no sumário ou via search_context.",
                            missing
                        )));
                        continue;
                    }

                    let found_ids: Vec<FactId> = spans.iter().map(|s| s.id.clone()).collect();
                    for id in &found_ids {
                        session_facts.insert(id.clone());
                    }
                    audit.record(
                        step_index,
                        snapshot,
                        action,
                        AuditOutcome::SpansRead {
                            span_ids: found_ids,
                        },
                    );
                    let payload = serde_json::to_string(&json!({
                        "status": "ok",
                        "spans": spans,
                        "missing": missing,
                    }))?;
                    history.push(Message::tool(payload));
                }
            }
        }

        Err(EngineError::MaxStepsExceeded {
            max_steps: self.max_steps,
            repair_attempts,
        })
    }
}

/// Expande templates determinísticos sem precisar gerar texto livre na `head` de 65k:
/// - `""` ou `"@slot"` -> materializa frase exata a partir dos fatos citados e/ou `EvalMath`.
/// - Placeholders `{{subject}}`, `{{relation}}`, `{{object}}`, `{{id}}`, `{{math}}` -> substituídos.
fn render_slot_template(
    raw_text: &str,
    cited_facts: &[FactId],
    records: &HashMap<FactId, Fact>,
    last_math_value: Option<&str>,
) -> String {
    let trimmed = raw_text.trim();
    let cited_records: Vec<&Fact> = cited_facts
        .iter()
        .filter_map(|id| records.get(id))
        .collect();

    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("@slot") {
        return match (last_math_value, cited_records.as_slice()) {
            (Some(math_val), [fact]) => format!(
                "O resultado exato é {math_val} (baseado em {} de {} = {}).",
                fact.relation, fact.subject, fact.object
            ),
            (Some(math_val), _) => format!("O resultado exato é {math_val}."),
            (None, [fact]) => format!(
                "O valor de {} de {} é {}.",
                fact.relation, fact.subject, fact.object
            ),
            (None, many) if !many.is_empty() => {
                let items: Vec<String> = many
                    .iter()
                    .map(|f| format!("{} ({}) = {}", f.subject, f.relation, f.object))
                    .collect();
                format!("Fatos verificados: {}.", items.join("; "))
            }
            (None, _) => trimmed.to_owned(),
        };
    }

    if !raw_text.contains("{{") {
        return raw_text.to_owned();
    }

    let mut out = raw_text.to_owned();
    if let Some(math_val) = last_math_value {
        out = out.replace("{{math}}", math_val);
    }
    if let Some(first) = cited_records.first() {
        out = out
            .replace("{{id}}", first.id.as_str())
            .replace("{{subject}}", &first.subject)
            .replace("{{relation}}", &first.relation)
            .replace("{{object}}", &first.object);
    }
    out
}

/// Garante que a resposta não é vazia/degenerada e contém os valores verificados da sessão.
fn verify_answer_grounding(
    text: &str,
    cited_facts: &[FactId],
    records: &HashMap<FactId, Fact>,
    last_math_value: Option<&str>,
) -> Result<(), String> {
    let trimmed = text.trim();
    if trimmed.chars().count() < 2 {
        return Err(format!(
            "resposta vazia ou curta demais ('{trimmed}')"
        ));
    }

    if let Some(math_val) = last_math_value
        && !contains_value_normalized(trimmed, math_val)
    {
        return Err(format!(
            "a resposta não contém o resultado exato '{math_val}' calculado pelo EvalMath"
        ));
    }

    let cited_objects: Vec<&str> = cited_facts
        .iter()
        .filter_map(|id| records.get(id))
        .map(|fact| fact.object.trim())
        .filter(|obj| !obj.is_empty())
        .collect();

    // Quando não houve EvalMath, ao menos um valor `object` dos fatos citados precisa estar presente.
    if last_math_value.is_none()
        && !cited_objects.is_empty()
        && !cited_objects
            .iter()
            .any(|obj| contains_value_normalized(trimmed, obj))
    {
        return Err(format!(
            "a resposta não menciona o valor verificado ({:?}) dos fatos citados",
            cited_objects
        ));
    }

    Ok(())
}

fn contains_value_normalized(text: &str, expected: &str) -> bool {
    let expected_trimmed = expected.trim();
    if expected_trimmed.is_empty() {
        return true;
    }

    let text_lower = text.to_lowercase();
    let exp_lower = expected_trimmed.to_lowercase();

    if text_lower.contains(&exp_lower) {
        return true;
    }

    let exp_comma = exp_lower.replace('.', ",");
    if text_lower.contains(&exp_comma) {
        return true;
    }

    if let Some(compact_num) = strip_trailing_decimal_zeros(&exp_lower)
        && (text_lower.contains(&compact_num)
            || text_lower.contains(&compact_num.replace('.', ",")))
    {
        return true;
    }

    if exp_lower.contains('_') && text_lower.contains(&exp_lower.replace('_', " ")) {
        return true;
    }

    false
}

fn strip_trailing_decimal_zeros(num: &str) -> Option<String> {
    let (int_part, frac_part) = num.split_once('.')?;
    let int_digits = int_part.strip_prefix('-').unwrap_or(int_part);
    if int_digits.is_empty()
        || !int_digits.chars().all(|c| c.is_ascii_digit())
        || !frac_part.chars().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let trimmed_frac = frac_part.trim_end_matches('0');
    if trimmed_frac.is_empty() {
        Some(int_part.to_owned())
    } else if trimmed_frac.len() < frac_part.len() {
        Some(format!("{int_part}.{trimmed_frac}"))
    } else {
        None
    }
}

fn serialize_logic_proof(proof: &LogicProof) -> Result<String, serde_json::Error> {
    let value = match proof {
        LogicProof::Satisfiable { model_summary } => json!({
            "status": "sat",
            "model_summary": model_summary,
        }),
        LogicProof::Unsatisfiable { unsat_core } => json!({
            "status": "unsat",
            "unsat_core": unsat_core,
        }),
        LogicProof::Unknown { reason } => json!({
            "status": "unknown",
            "reason": reason,
        }),
    };
    serde_json::to_string(&value)
}
