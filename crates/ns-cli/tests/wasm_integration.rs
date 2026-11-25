#![cfg(feature = "wasm-sandbox")]

use ns_core::prelude::*;
use ns_engine::{AuditOutcome, Orchestrator};
use ns_llm::MockLlm;
use ns_memory::RedbMemory;
use ns_verifier::DefaultSymbolicVerifier;

#[tokio::test]
async fn orchestrator_kills_infinite_loop_and_verifies_repaired_wasm_factorial()
-> Result<(), Box<dyn std::error::Error>> {
    let memory = RedbMemory::open_in_memory()?;
    memory.upsert_fact(&Fact {
        id: FactId::from("fact-fatorial-n"),
        subject: "desafio_algo".to_owned(),
        relation: "fatorial_alvo".to_owned(),
        object: "5".to_owned(),
        source_ref: "spec#exercicio-1".to_owned(),
    })?;

    // 1. WAT defeituoso: loop infinito -> cortado deterministicamente pelo fuel metering
    let infinite_loop_wat = r#"
        (module
          (func (export "run")
            (loop $forever
              (br $forever)
            )
          )
        )
    "#;

    // 2. WAT reparado: calcula 5! = 120 em loop finito e imprime no stdout capturado
    let repaired_factorial_wat = r#"
        (module
          (import "env" "write_i64" (func $write_i64 (param i64)))
          (func (export "run")
            (local $n i64)
            (local $acc i64)
            (local.set $n (i64.const 5))
            (local.set $acc (i64.const 1))
            (block $done
              (loop $step
                (br_if $done (i64.le_s (local.get $n) (i64.const 1)))
                (local.set $acc (i64.mul (local.get $acc) (local.get $n)))
                (local.set $n (i64.sub (local.get $n) (i64.const 1)))
                (br $step)
              )
            )
            (call $write_i64 (local.get $acc))
          )
        )
    "#;

    let llm = MockLlm::from_actions([
        LlmAction::QueryFact {
            entity: "desafio_algo".to_owned(),
            relation: "fatorial_alvo".to_owned(),
        },
        LlmAction::RunWasmCheck {
            wasm_wat_source: infinite_loop_wat.to_owned(),
            expected_stdout: "120".to_owned(),
        },
        LlmAction::RunWasmCheck {
            wasm_wat_source: repaired_factorial_wat.to_owned(),
            expected_stdout: "120".to_owned(),
        },
        LlmAction::ProposeAnswer {
            text: "O fatorial de 5 verificado na sandbox WebAssembly é 120.".to_owned(),
            cited_facts: vec![FactId::from("fact-fatorial-n")],
        },
    ]);

    let orchestrator = Orchestrator::new(llm, memory, DefaultSymbolicVerifier::new(), 6)
        .with_require_citations(true);

    let outcome = orchestrator
        .answer("Calcule e valide em WebAssembly o fatorial definido em desafio_algo.")
        .await?;

    assert_eq!(
        outcome.answer().text(),
        "O fatorial de 5 verificado na sandbox WebAssembly é 120."
    );
    assert_eq!(outcome.audit().repair_count(), 1);
    assert!(matches!(
        outcome.audit().steps()[1].outcome,
        AuditOutcome::VerifierRejected { .. }
    ));
    assert!(matches!(
        outcome.audit().steps()[2].outcome,
        AuditOutcome::SandboxExecuted {
            passed: true,
            fuel_consumed,
        } if fuel_consumed > 0
    ));
    assert_eq!(
        outcome.audit().steps()[3].outcome,
        AuditOutcome::AnswerVerified
    );

    Ok(())
}
