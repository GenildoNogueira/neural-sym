pub mod error;
pub mod logic;
pub mod math;
pub mod sandbox;

use ns_core::error::VerifierPortError;
use ns_core::ports::{LogicProof, SandboxOutput, SymbolicVerifier};
use rust_decimal::Decimal;

pub use error::VerifierError;

/// Implementação padrão de [`SymbolicVerifier`] com aritmética decimal exata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DefaultSymbolicVerifier;

impl DefaultSymbolicVerifier {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SymbolicVerifier for DefaultSymbolicVerifier {
    fn eval_decimal(&self, expression: &str) -> Result<Decimal, VerifierPortError> {
        let result = math::evaluate_decimal(expression)?;
        tracing::debug!(expression, %result, "evaluated exact decimal expression");
        Ok(result)
    }

    fn check_smt(&self, smtlib2_script: &str) -> Result<LogicProof, VerifierPortError> {
        Ok(logic::check_smtlib2(smtlib2_script)?)
    }

    fn run_sandbox(
        &self,
        wat_source: &str,
        expected_stdout: &str,
    ) -> Result<SandboxOutput, VerifierPortError> {
        Ok(sandbox::run_wat_sandbox(wat_source, expected_stdout)?)
    }
}
