use ns_core::ports::LogicProof;

use crate::error::VerifierError;

pub const MAX_SMT_SCRIPT_LEN: usize = 32_768;
pub const DEFAULT_SMT_TIMEOUT_MS: u64 = 5_000;

/// Valida e normaliza um script SMT-LIB2 em Rust puro antes de enviá-lo ao solver.
///
/// Remove comandos finais de controle (`check-sat`, `get-model`, `get-unsat-core`, `exit`)
/// pois o orquestrador invoca a checagem e a extração de provas explicitamente.
///
/// # Errors
///
/// Retorna [`VerifierError`] se o script estiver vazio, exceder [`MAX_SMT_SCRIPT_LEN`],
/// possuir parênteses/strings desbalanceados ou contiver comandos proibidos (`reset`).
pub fn prepare_smtlib2_script(input: &str) -> Result<String, VerifierError> {
    if input.len() > MAX_SMT_SCRIPT_LEN {
        return Err(VerifierError::ExpressionTooLong {
            max: MAX_SMT_SCRIPT_LEN,
        });
    }

    let bytes = input.as_bytes();
    let mut i = 0usize;
    let mut sanitized = String::with_capacity(input.len());

    while i < bytes.len() {
        skip_ws_and_comments(bytes, &mut i);
        if i >= bytes.len() {
            break;
        }

        if bytes[i] != b'(' {
            let ch = input[i..].chars().next().unwrap_or('?');
            return Err(VerifierError::SmtSyntax {
                message: format!("expected top-level '(' at byte {i}, found '{ch}'"),
            });
        }

        let start = i;
        let end = scan_balanced_sexpr(bytes, start)?;
        let form = &input[start..end];
        let command = extract_head_command(form);

        match command {
            "check-sat" | "get-model" | "get-unsat-core" | "exit" => {}
            "reset" | "reset-assertions" => {
                return Err(VerifierError::SmtSyntax {
                    message: format!("forbidden command '{command}' in verification script"),
                });
            }
            "" => {
                return Err(VerifierError::SmtSyntax {
                    message: format!("empty s-expression at byte {start}"),
                });
            }
            _ => {
                sanitized.push_str(form);
                sanitized.push('\n');
            }
        }

        i = end;
    }

    if sanitized.trim().is_empty() {
        return Err(VerifierError::EmptyExpression);
    }

    Ok(sanitized)
}

/// Avalia um script SMT-LIB2 com timeout padrão ([`DEFAULT_SMT_TIMEOUT_MS`]).
///
/// # Errors
///
/// Retorna [`VerifierError`] em caso de erro sintático, timeout ou feature `z3-solver` desativada.
pub fn check_smtlib2(script: &str) -> Result<LogicProof, VerifierError> {
    check_smtlib2_with_timeout(script, DEFAULT_SMT_TIMEOUT_MS)
}

/// Avalia um script SMT-LIB2 usando o solver Z3 com limite de tempo explícito em milissegundos.
///
/// # Errors
///
/// Retorna:
/// - [`VerifierError::EmptyExpression`] ou [`VerifierError::SmtSyntax`] se o script for inválido;
/// - [`VerifierError::SmtTimeout`] se o Z3 exceder `timeout_ms`;
/// - [`VerifierError::FeatureNotEnabled`] se o crate foi compilado sem `--features z3-solver`.
pub fn check_smtlib2_with_timeout(
    script: &str,
    timeout_ms: u64,
) -> Result<LogicProof, VerifierError> {
    let prepared = prepare_smtlib2_script(script)?;
    execute_z3(&prepared, timeout_ms)
}

#[cfg(feature = "z3-solver")]
fn execute_z3(prepared_script: &str, timeout_ms: u64) -> Result<LogicProof, VerifierError> {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use z3::{Config, Context, Params, SatResult, Solver};

    let run = catch_unwind(AssertUnwindSafe(|| {
        let mut cfg = Config::new();
        cfg.set_timeout_msec(timeout_ms);

        let ctx = Context::new(&cfg);
        let solver = Solver::new(&ctx);

        let mut params = Params::new(&ctx);
        params.set_u32("timeout", u32::try_from(timeout_ms).unwrap_or(u32::MAX));
        params.set_bool("unsat_core", true);
        solver.set_params(&params);

        solver.from_string(prepared_script);

        match solver.check() {
            SatResult::Sat => {
                let model_summary = solver.get_model().map_or_else(
                    || "sat (empty model)".to_owned(),
                    |model| model.to_string(),
                );
                Ok(LogicProof::Satisfiable { model_summary })
            }
            SatResult::Unsat => {
                let core_items: Vec<String> = solver
                    .get_unsat_core()
                    .into_iter()
                    .map(|term| term.to_string())
                    .collect();

                let unsat_core = if core_items.is_empty() {
                    "unsatisfiable (add :named labels to assertions to isolate the unsat core)"
                        .to_owned()
                } else {
                    core_items.join(", ")
                };

                Ok(LogicProof::Unsatisfiable { unsat_core })
            }
            SatResult::Unknown => {
                let reason = solver
                    .get_reason_unknown()
                    .unwrap_or_else(|| "solver returned unknown".to_owned());

                let lower = reason.to_ascii_lowercase();
                if lower.contains("timeout") || lower.contains("canceled") {
                    Err(VerifierError::SmtTimeout { timeout_ms })
                } else {
                    Ok(LogicProof::Unknown { reason })
                }
            }
        }
    }));

    match run {
        Ok(result) => result,
        Err(payload) => {
            let detail = extract_panic_message(payload.as_ref());
            tracing::warn!(%detail, "z3 rejected malformed smt-lib2 script");
            Err(VerifierError::SmtSyntax { message: detail })
        }
    }
}

#[cfg(feature = "z3-solver")]
fn extract_panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        (*msg).to_owned()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "z3 failed to parse smt-lib2 input".to_owned()
    }
}

#[cfg(not(feature = "z3-solver"))]
fn execute_z3(_prepared_script: &str, _timeout_ms: u64) -> Result<LogicProof, VerifierError> {
    Err(VerifierError::FeatureNotEnabled {
        feature: "z3-solver",
    })
}

fn skip_ws_and_comments(bytes: &[u8], i: &mut usize) {
    while *i < bytes.len() {
        let b = bytes[*i];
        if b.is_ascii_whitespace() {
            *i = i.saturating_add(1);
        } else if b == b';' {
            while *i < bytes.len() && bytes[*i] != b'\n' {
                *i = i.saturating_add(1);
            }
        } else {
            break;
        }
    }
}

fn scan_balanced_sexpr(bytes: &[u8], start: usize) -> Result<usize, VerifierError> {
    let mut depth = 0usize;
    let mut i = start;
    let mut in_string = false;
    let mut in_quoted_symbol = false;

    while i < bytes.len() {
        let b = bytes[i];

        if in_string {
            if b == b'"' {
                // Em SMT-LIB2, aspas duplas escapam com "".
                if i.checked_add(1).and_then(|next| bytes.get(next)) == Some(&b'"') {
                    i = i.saturating_add(2);
                    continue;
                }
                in_string = false;
            }
            i = i.saturating_add(1);
            continue;
        }

        if in_quoted_symbol {
            if b == b'|' {
                in_quoted_symbol = false;
            }
            i = i.saturating_add(1);
            continue;
        }

        match b {
            b';' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i = i.saturating_add(1);
                }
            }
            b'"' => {
                in_string = true;
                i = i.saturating_add(1);
            }
            b'|' => {
                in_quoted_symbol = true;
                i = i.saturating_add(1);
            }
            b'(' => {
                depth = depth.saturating_add(1);
                i = i.saturating_add(1);
            }
            b')' => {
                depth = depth.saturating_sub(1);
                i = i.saturating_add(1);
                if depth == 0 {
                    return Ok(i);
                }
            }
            _ => {
                i = i.saturating_add(1);
            }
        }
    }

    if in_string || in_quoted_symbol {
        return Err(VerifierError::SmtSyntax {
            message: format!("unterminated literal starting inside form at byte {start}"),
        });
    }

    Err(VerifierError::UnclosedParenthesis { position: start })
}

fn extract_head_command(form: &str) -> &str {
    let inner = form.trim_start_matches('(').trim_start();
    inner
        .split(|c: char| c.is_ascii_whitespace() || c == '(' || c == ')')
        .next()
        .unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepares_script_and_strips_runner_commands() -> Result<(), Box<dyn std::error::Error>> {
        let raw = r#"
            ; Comentário com parêntese ) que deve ser ignorado
            (declare-const x Int)
            (assert (> x 10))
            (check-sat)
            (get-model)
        "#;

        let cleaned = prepare_smtlib2_script(raw)?;
        assert!(cleaned.contains("(declare-const x Int)"));
        assert!(cleaned.contains("(assert (> x 10))"));
        assert!(!cleaned.contains("check-sat"));
        assert!(!cleaned.contains("get-model"));
        Ok(())
    }

    #[test]
    fn rejects_unbalanced_or_forbidden_smt_scripts() {
        assert_eq!(
            prepare_smtlib2_script("   ; somente comentario"),
            Err(VerifierError::EmptyExpression)
        );
        assert!(matches!(
            prepare_smtlib2_script("(declare-const x Int"),
            Err(VerifierError::UnclosedParenthesis { position: 0 })
        ));
        assert!(matches!(
            prepare_smtlib2_script("(reset)"),
            Err(VerifierError::SmtSyntax { .. })
        ));
    }

    #[cfg(feature = "z3-solver")]
    #[test]
    fn z3_proves_satisfiable_constraints_with_model() -> Result<(), Box<dyn std::error::Error>> {
        let script = r#"
            (declare-const x Int)
            (declare-const y Int)
            (assert (= (+ x y) 10))
            (assert (> x 7))
            (assert (> y 0))
            (check-sat)
        "#;

        let proof = check_smtlib2(script)?;
        match proof {
            LogicProof::Satisfiable { model_summary } => {
                assert!(model_summary.contains("x ->"));
                assert!(model_summary.contains("y ->"));
            }
            other => return Err(format!("expected Satisfiable, got {other:?}").into()),
        }
        Ok(())
    }

    #[cfg(feature = "z3-solver")]
    #[test]
    fn z3_proves_unsatisfiable_and_extracts_named_unsat_core()
    -> Result<(), Box<dyn std::error::Error>> {
        let script = r#"
            (declare-const desconto Real)
            (assert (! (<= desconto 0.30) :named limite_politica))
            (assert (! (= desconto 0.50) :named desconto_proposto))
        "#;

        let proof = check_smtlib2(script)?;
        match proof {
            LogicProof::Unsatisfiable { unsat_core } => {
                assert!(unsat_core.contains("limite_politica"));
                assert!(unsat_core.contains("desconto_proposto"));
            }
            other => return Err(format!("expected Unsatisfiable, got {other:?}").into()),
        }
        Ok(())
    }

    #[cfg(feature = "z3-solver")]
    #[test]
    fn z3_converts_undeclared_symbol_into_syntax_error_without_panicking() {
        let broken = "(assert (> variavel_inexistente 0))";
        let result = check_smtlib2(broken);
        assert!(matches!(result, Err(VerifierError::SmtSyntax { .. })));
    }
}
