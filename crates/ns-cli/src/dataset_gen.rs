use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result, bail};
use ns_core::prelude::*;
use ns_engine::{Orchestrator, TrainingSample, extract_verified_samples, write_samples_jsonl};
use ns_llm::MockLlm;
use ns_verifier::DefaultSymbolicVerifier;
use rust_decimal::Decimal;

use crate::backend::MemoryBackend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationReport {
    pub facts_used: usize,
    pub trajectories_attempted: usize,
    pub trajectories_verified: usize,
    pub samples_written: usize,
    pub repair_recovery_samples: usize,
}

/// Popula um conjunto sintético realista (preços, taxas e relações em grafo) se o banco estiver vazio
/// ou se `--seed-synthetic` for solicitado.
#[must_use]
pub fn build_synthetic_catalog(entity_count: usize) -> Vec<Fact> {
    let mut facts = Vec::with_capacity(entity_count.saturating_mul(3));
    for idx in 0..entity_count {
        let subject = format!("plano_{idx:03}");
        let next_subject = format!("plano_{:03}", (idx + 1) % entity_count.max(1));
        let price_reais = 49_u64.saturating_add((idx as u64).saturating_mul(15) % 450);
        let price_cents = (idx as u64).saturating_mul(10) % 100;

        facts.push(Fact {
            id: FactId::new(format!("fact-preco-{idx:03}")),
            subject: subject.clone(),
            relation: "preco_mensal".to_owned(),
            object: format!("{price_reais}.{price_cents:02}"),
            source_ref: "synthetic#catalogo".to_owned(),
        });

        facts.push(Fact {
            id: FactId::new(format!("fact-taxa-{idx:03}")),
            subject: subject.clone(),
            relation: "taxa_servico".to_owned(),
            object: format!("0.{:02}", 5 + (idx % 20)),
            source_ref: "synthetic#catalogo".to_owned(),
        });

        facts.push(Fact {
            id: FactId::new(format!("fact-upgrade-{idx:03}")),
            subject,
            relation: "upgrade_para".to_owned(),
            object: next_subject,
            source_ref: "synthetic#grafo".to_owned(),
        });
    }
    facts
}

/// Executa o self-play determinístico pelo `Orchestrator` real e grava apenas amostras
/// provenientes de trajetórias que cruzaram o typestate `Verified` em formato JSONL.
///
/// # Errors
///
/// Retorna erro se o banco não tiver fatos, se falhar ao criar o arquivo JSONL ou ao gravar.
pub async fn generate_verified_jsonl(
    memory: &MemoryBackend,
    output_path: &Path,
    seed_synthetic: usize,
    include_repairs: bool,
) -> Result<GenerationReport> {
    if seed_synthetic > 0 {
        let synthetic = build_synthetic_catalog(seed_synthetic);
        let _ = memory.upsert_batch(&synthetic).await?;
    }

    let facts = memory.load_all().await?;
    if facts.is_empty() {
        bail!(
            "database has no facts to generate trajectories; add facts first or pass --seed-synthetic <N>"
        );
    }

    let _ = memory.warm_up().await?;

    let known_subjects: HashSet<String> = facts
        .iter()
        .map(|f| f.subject.trim().to_lowercase())
        .collect();

    let mut all_samples: Vec<TrainingSample> = Vec::with_capacity(facts.len().saturating_mul(6));
    let mut trajectories_attempted: usize = 0;
    let mut trajectories_verified: usize = 0;
    let mut traj_seq: usize = 0;

    let verifier = DefaultSymbolicVerifier::new();

    for (idx, fact) in facts.iter().enumerate() {
        // 1. Consulta direta: USER -> QueryFact -> TOOL -> ProposeAnswer
        traj_seq = traj_seq.saturating_add(1);
        trajectories_attempted = trajectories_attempted.saturating_add(1);
        let direct_prompt = direct_prompt_variant(idx, fact);
        let direct_actions = vec![
            LlmAction::QueryFact {
                entity: fact.subject.clone(),
                relation: fact.relation.clone(),
            },
            LlmAction::ProposeAnswer {
                text: format!(
                    "O valor de {} para {} e {}.",
                    fact.relation, fact.subject, fact.object
                ),
                cited_facts: vec![fact.id.clone()],
            },
        ];
        if let Some(samples) = run_and_extract(
            memory,
            &verifier,
            &format!("traj-{traj_seq:05}"),
            "direct_lookup",
            &direct_prompt,
            direct_actions,
        )
        .await
        {
            trajectories_verified = trajectories_verified.saturating_add(1);
            all_samples.extend(samples);
        }

        // 2. Descoberta de chaves: USER -> QueryFact (not_found) -> SearchFacts -> ProposeAnswer
        traj_seq = traj_seq.saturating_add(1);
        trajectories_attempted = trajectories_attempted.saturating_add(1);
        let search_prompt = format!(
            "Pesquise na base qual e o dado de {} relacionado a {}.",
            fact.relation.replace('_', " "),
            fact.subject.replace('_', " ")
        );
        let search_actions = vec![
            LlmAction::QueryFact {
                entity: fact.subject.replace('_', " "),
                relation: "valor_desconhecido".to_owned(),
            },
            LlmAction::SearchFacts {
                query: fact.subject.clone(),
                limit: 10,
            },
            LlmAction::ProposeAnswer {
                text: format!(
                    "Encontrado via busca: {} de {} e {}.",
                    fact.relation, fact.subject, fact.object
                ),
                cited_facts: vec![fact.id.clone()],
            },
        ];
        if let Some(samples) = run_and_extract(
            memory,
            &verifier,
            &format!("traj-{traj_seq:05}"),
            "search_discovery",
            &search_prompt,
            search_actions,
        )
        .await
        {
            trajectories_verified = trajectories_verified.saturating_add(1);
            all_samples.extend(samples);
        }

        // 3. Reparo de citação inventada: Hallucination -> REPAIR REQUIRED -> QueryFact -> Verified
        if include_repairs {
            traj_seq = traj_seq.saturating_add(1);
            trajectories_attempted = trajectories_attempted.saturating_add(1);
            let repair_prompt = format!(
                "Informe {} de {} citando a fonte oficial.",
                fact.relation, fact.subject
            );
            let repair_actions = vec![
                LlmAction::ProposeAnswer {
                    text: "Valor estimado sem consulta.".to_owned(),
                    cited_facts: vec![FactId::from("fact-alucinado-000")],
                },
                LlmAction::QueryFact {
                    entity: fact.subject.clone(),
                    relation: fact.relation.clone(),
                },
                LlmAction::ProposeAnswer {
                    text: format!(
                        "Confirmado na memoria: {} de {} e {}.",
                        fact.relation, fact.subject, fact.object
                    ),
                    cited_facts: vec![fact.id.clone()],
                },
            ];
            if let Some(samples) = run_and_extract(
                memory,
                &verifier,
                &format!("traj-{traj_seq:05}"),
                "citation_repair",
                &repair_prompt,
                repair_actions,
            )
            .await
            {
                trajectories_verified = trajectories_verified.saturating_add(1);
                all_samples.extend(samples);
            }
        }

        // 4. Matemática decimal exata (+ reparo de divisão por zero) para fatos numéricos
        if Decimal::from_str_exact(fact.object.trim()).is_ok() {
            let multiplier = 2_usize.saturating_add(idx % 11);
            let good_expr = format!("{} * {multiplier}", fact.object.trim());
            if let Ok(exact_total) = verifier.eval_decimal(&good_expr) {
                traj_seq = traj_seq.saturating_add(1);
                trajectories_attempted = trajectories_attempted.saturating_add(1);

                let math_prompt = format!(
                    "Calcule {multiplier} vezes {} de {} com precisao exata.",
                    fact.relation, fact.subject
                );

                let mut math_actions = Vec::with_capacity(4);
                math_actions.push(LlmAction::QueryFact {
                    entity: fact.subject.clone(),
                    relation: fact.relation.clone(),
                });
                if include_repairs {
                    math_actions.push(LlmAction::EvalMath {
                        expression: format!("({good_expr}) / (1 - 1)"),
                    });
                }
                math_actions.push(LlmAction::EvalMath {
                    expression: good_expr,
                });
                math_actions.push(LlmAction::ProposeAnswer {
                    text: format!(
                        "O total exato para {multiplier} unidades de {} ({}) e {exact_total}.",
                        fact.subject, fact.relation
                    ),
                    cited_facts: vec![fact.id.clone()],
                });

                let scenario = if include_repairs {
                    "math_with_repair"
                } else {
                    "math_exact"
                };

                if let Some(samples) = run_and_extract(
                    memory,
                    &verifier,
                    &format!("traj-{traj_seq:05}"),
                    scenario,
                    &math_prompt,
                    math_actions,
                )
                .await
                {
                    trajectories_verified = trajectories_verified.saturating_add(1);
                    all_samples.extend(samples);
                }
            }
        }

        // 5. Travessia de grafo multi-hop quando o objeto aponta para outra entidade existente
        let obj_key = fact.object.trim().to_lowercase();
        if known_subjects.contains(&obj_key) && obj_key != fact.subject
            && let Ok(reached) = memory.traverse(&fact.subject, 2).await
                && reached.len() >= 2
            {
                traj_seq = traj_seq.saturating_add(1);
                trajectories_attempted = trajectories_attempted.saturating_add(1);

                let cited_ids: Vec<FactId> = reached.iter().map(|f| f.id.clone()).collect();
                let graph_prompt = format!(
                    "Percorra o grafo a partir de {} em 2 saltos e resuma as relacoes.",
                    fact.subject
                );
                let graph_actions = vec![
                    LlmAction::TraverseGraph {
                        start_entity: fact.subject.clone(),
                        max_hops: 2,
                    },
                    LlmAction::ProposeAnswer {
                        text: format!(
                            "A partir de {}, o grafo alcanca {} fatos em 2 saltos incluindo {} -> {}.",
                            fact.subject,
                            reached.len(),
                            fact.relation,
                            fact.object
                        ),
                        cited_facts: cited_ids,
                    },
                ];

                if let Some(samples) = run_and_extract(
                    memory,
                    &verifier,
                    &format!("traj-{traj_seq:05}"),
                    "graph_traversal",
                    &graph_prompt,
                    graph_actions,
                )
                .await
                {
                    trajectories_verified = trajectories_verified.saturating_add(1);
                    all_samples.extend(samples);
                }
            }
    }

    if let Some(parent) = output_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory '{}'", parent.display()))?;
    }

    let file = File::create(output_path)
        .with_context(|| format!("failed to create jsonl file '{}'", output_path.display()))?;
    let mut writer = BufWriter::new(file);
    let samples_written = write_samples_jsonl(&mut writer, &all_samples)
        .with_context(|| format!("failed to write samples to '{}'", output_path.display()))?;

    let repair_recovery_samples = all_samples
        .iter()
        .filter(|sample| sample.is_repair_recovery)
        .count();

    Ok(GenerationReport {
        facts_used: facts.len(),
        trajectories_attempted,
        trajectories_verified,
        samples_written,
        repair_recovery_samples,
    })
}

async fn run_and_extract(
    memory: &MemoryBackend,
    verifier: &DefaultSymbolicVerifier,
    trajectory_id: &str,
    scenario: &str,
    prompt: &str,
    actions: Vec<LlmAction>,
) -> Option<Vec<TrainingSample>> {
    let llm = MockLlm::from_actions(actions);
    let orchestrator =
        Orchestrator::new(llm, BorrowedMemory(memory), *verifier, 8).with_require_citations(true);

    match orchestrator.answer(prompt).await {
        Ok(verified) => Some(extract_verified_samples(trajectory_id, scenario, &verified)),
        Err(err) => {
            tracing::warn!(trajectory_id, scenario, %err, "discarded unverified trajectory");
            None
        }
    }
}

fn direct_prompt_variant(idx: usize, fact: &Fact) -> String {
    match idx % 3 {
        0 => format!("Qual o(a) {} de {}?", fact.relation, fact.subject),
        1 => format!(
            "Consulte {} para a entidade {}.",
            fact.relation, fact.subject
        ),
        _ => format!("Me diga qual e {} de {}.", fact.relation, fact.subject),
    }
}

/// Wrapper leve para emprestar `&MemoryBackend` para o `Orchestrator` sem reabrir o banco.
struct BorrowedMemory<'a>(&'a MemoryBackend);

impl DeterministicMemory for BorrowedMemory<'_> {
    async fn lookup_fact(
        &self,
        entity: &str,
        relation: &str,
    ) -> Result<Option<Fact>, MemoryPortError> {
        self.0.lookup_fact(entity, relation).await
    }

    async fn search_facts(&self, query: &str, limit: u8) -> Result<Vec<Fact>, MemoryPortError> {
        self.0.search_facts(query, limit).await
    }

    async fn traverse(
        &self,
        start_entity: &str,
        max_hops: u8,
    ) -> Result<Vec<Fact>, MemoryPortError> {
        self.0.traverse(start_entity, max_hops).await
    }

    async fn verify_fact_exists(&self, id: &FactId) -> Result<bool, MemoryPortError> {
        self.0.verify_fact_exists(id).await
    }
}
