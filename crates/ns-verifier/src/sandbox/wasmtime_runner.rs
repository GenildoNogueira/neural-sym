use ns_core::ports::SandboxOutput;

use crate::error::VerifierError;

/// Limite padrão de instruções executadas na sandbox (interrompe loops infinitos rapidamente).
pub const DEFAULT_MAX_FUEL: u64 = 500_000;
/// Limite padrão de memória linear WebAssembly (1 MiB = 16 páginas de 64 KiB).
pub const DEFAULT_MAX_MEMORY_BYTES: usize = 1024 * 1024;
/// Limite padrão de bytes acumulados em `stdout`.
pub const DEFAULT_MAX_STDOUT_BYTES: usize = 16 * 1024;
/// Limite padrão de tamanho do código-fonte WAT recebido.
pub const DEFAULT_MAX_SOURCE_BYTES: usize = 64 * 1024;

/// Parâmetros de isolamento determinístico da sandbox WebAssembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SandboxConfig {
    pub max_fuel: u64,
    pub max_memory_bytes: usize,
    pub max_stdout_bytes: usize,
    pub max_source_bytes: usize,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            max_fuel: DEFAULT_MAX_FUEL,
            max_memory_bytes: DEFAULT_MAX_MEMORY_BYTES,
            max_stdout_bytes: DEFAULT_MAX_STDOUT_BYTES,
            max_source_bytes: DEFAULT_MAX_SOURCE_BYTES,
        }
    }
}

/// Executa um módulo WAT em sandbox isolada com limites padrão de CPU, memória e `stdout`.
///
/// # Errors
///
/// Retorna [`VerifierError`] se o módulo for inválido, exceder limites de recursos,
/// sofrer *trap* em execução ou se a feature `wasm-sandbox` não estiver habilitada.
pub fn run_wat_sandbox(
    wat_source: &str,
    expected_stdout: &str,
) -> Result<SandboxOutput, VerifierError> {
    run_wat_sandbox_with_config(wat_source, expected_stdout, &SandboxConfig::default())
}

/// Executa um módulo WAT em sandbox isolada usando uma [`SandboxConfig`] customizada.
///
/// Funções de host disponíveis ao módulo:
/// - `(import "env" "write" (func (param i32 i32)))` — escreve `[ptr..ptr+len]` da `memory` exportada no `stdout`;
/// - `(import "env" "write_i64" (func (param i64)))` — escreve um inteiro decimal no `stdout`;
/// - `(import "wasi_snapshot_preview1" "fd_write" (func (param i32 i32 i32 i32) (result i32)))` — escrita WASI em `fd = 1`.
///
/// O módulo deve exportar `_start` ou `run` com assinatura `() -> ()`.
///
/// # Errors
///
/// Retorna:
/// - [`VerifierError::EmptyExpression`] ou [`VerifierError::ExpressionTooLong`] na validação de entrada;
/// - [`VerifierError::SandboxCompile`] se o código WAT for sintaticamente inválido;
/// - [`VerifierError::SandboxMemoryLimit`], [`VerifierError::SandboxOutOfFuel`] ou
///   [`VerifierError::SandboxStdoutLimit`] se qualquer cota de recurso for violada;
/// - [`VerifierError::SandboxMissingEntrypoint`] ou [`VerifierError::SandboxTrap`] em falha de execução;
/// - [`VerifierError::FeatureNotEnabled`] se a feature `wasm-sandbox` não estiver ativa.
pub fn run_wat_sandbox_with_config(
    wat_source: &str,
    expected_stdout: &str,
    config: &SandboxConfig,
) -> Result<SandboxOutput, VerifierError> {
    if wat_source.trim().is_empty() {
        return Err(VerifierError::EmptyExpression);
    }
    if wat_source.len() > config.max_source_bytes {
        return Err(VerifierError::ExpressionTooLong {
            max: config.max_source_bytes,
        });
    }

    execute_wasmtime(wat_source, expected_stdout, config)
}

#[cfg(feature = "wasm-sandbox")]
fn execute_wasmtime(
    wat_source: &str,
    expected_stdout: &str,
    config: &SandboxConfig,
) -> Result<SandboxOutput, VerifierError> {
    use wasmtime::{
        Caller, Config, Engine, Extern, Linker, Module, Store, StoreLimits, StoreLimitsBuilder,
        Trap,
    };

    const MAX_WASM_STACK_BYTES: usize = 128 * 1024;
    const WASI_ESUCCESS: u32 = 0;
    const WASI_EBADF: u32 = 8;

    struct SandboxState {
        limits: StoreLimits,
        stdout: Vec<u8>,
        max_stdout_bytes: usize,
        stdout_overflowed: bool,
    }

    impl SandboxState {
        fn push_stdout(&mut self, bytes: &[u8]) -> wasmtime::Result<()> {
            let new_len = self
                .stdout
                .len()
                .checked_add(bytes.len())
                .ok_or_else(|| wasmtime::Error::msg("stdout length overflow"))?;

            if new_len > self.max_stdout_bytes {
                self.stdout_overflowed = true;
                return Err(wasmtime::Error::msg("stdout byte limit exceeded"));
            }

            self.stdout.extend_from_slice(bytes);
            Ok(())
        }
    }

    let mut engine_cfg = Config::new();
    engine_cfg.consume_fuel(true);
    engine_cfg.max_wasm_stack(MAX_WASM_STACK_BYTES);

    let engine = Engine::new(&engine_cfg).map_err(|err| VerifierError::SandboxCompile {
        message: err.to_string(),
    })?;

    let module = Module::new(&engine, wat_source).map_err(|err| VerifierError::SandboxCompile {
        message: err.to_string(),
    })?;

    let limits = StoreLimitsBuilder::new()
        .memory_size(config.max_memory_bytes)
        .table_elements(1_024)
        .instances(1)
        .memories(1)
        .tables(1)
        .build();

    let state = SandboxState {
        limits,
        stdout: Vec::with_capacity(256.min(config.max_stdout_bytes)),
        max_stdout_bytes: config.max_stdout_bytes,
        stdout_overflowed: false,
    };

    let mut store = Store::new(&engine, state);
    store.limiter(|data| &mut data.limits);
    store
        .set_fuel(config.max_fuel)
        .map_err(|err| VerifierError::SandboxTrap {
            message: err.to_string(),
        })?;

    let mut linker: Linker<SandboxState> = Linker::new(&engine);

    // 1. Host minimalista: env::write(ptr, len) -> grava fatia UTF-8 de "memory" em stdout.
    linker
        .func_wrap(
            "env",
            "write",
            |mut caller: Caller<'_, SandboxState>, ptr: u32, len: u32| -> wasmtime::Result<()> {
                let Some(Extern::Memory(memory)) = caller.get_export("memory") else {
                    return Err(wasmtime::Error::msg("module must export 'memory'"));
                };

                let start = usize::try_from(ptr)
                    .map_err(|_| wasmtime::Error::msg("pointer conversion overflow"))?;
                let count = usize::try_from(len)
                    .map_err(|_| wasmtime::Error::msg("length conversion overflow"))?;
                let end = start
                    .checked_add(count)
                    .ok_or_else(|| wasmtime::Error::msg("memory slice overflow"))?;

                let (mem, data) = memory.data_and_store_mut(&mut caller);
                let slice = mem
                    .get(start..end)
                    .ok_or_else(|| wasmtime::Error::msg("out-of-bounds memory read in env::write"))?;

                data.push_stdout(slice)
            },
        )
        .map_err(|err| VerifierError::SandboxCompile {
            message: err.to_string(),
        })?;

    // 2. Host ergonômico para LLMs: env::write_i64(value) -> imprime inteiro decimal em stdout.
    linker
        .func_wrap(
            "env",
            "write_i64",
            |mut caller: Caller<'_, SandboxState>, value: i64| -> wasmtime::Result<()> {
                let formatted = value.to_string();
                caller.data_mut().push_stdout(formatted.as_bytes())
            },
        )
        .map_err(|err| VerifierError::SandboxCompile {
            message: err.to_string(),
        })?;

    // 3. Compatibilidade WASI preview1: fd_write(fd, iovs_ptr, iovs_len, nwritten_ptr) -> errno.
    linker
        .func_wrap(
            "wasi_snapshot_preview1",
            "fd_write",
            |mut caller: Caller<'_, SandboxState>,
             fd: u32,
             iovs_ptr: u32,
             iovs_len: u32,
             nwritten_ptr: u32|
             -> wasmtime::Result<u32> {
                if fd != 1 && fd != 2 {
                    return Ok(WASI_EBADF);
                }

                let Some(Extern::Memory(memory)) = caller.get_export("memory") else {
                    return Err(wasmtime::Error::msg("module must export 'memory'"));
                };

                let iovs_base = usize::try_from(iovs_ptr)
                    .map_err(|_| wasmtime::Error::msg("iovs_ptr overflow"))?;
                let iovs_count = usize::try_from(iovs_len)
                    .map_err(|_| wasmtime::Error::msg("iovs_len overflow"))?;
                let nwritten_base = usize::try_from(nwritten_ptr)
                    .map_err(|_| wasmtime::Error::msg("nwritten_ptr overflow"))?;

                let (mem, data) = memory.data_and_store_mut(&mut caller);
                let mut total_written: u32 = 0;

                for idx in 0..iovs_count {
                    let offset = idx
                        .checked_mul(8)
                        .and_then(|step| iovs_base.checked_add(step))
                        .ok_or_else(|| wasmtime::Error::msg("iovec offset overflow"))?;
                    let header_end = offset
                        .checked_add(8)
                        .ok_or_else(|| wasmtime::Error::msg("iovec header overflow"))?;
                    let header = mem
                        .get(offset..header_end)
                        .ok_or_else(|| wasmtime::Error::msg("out-of-bounds iovec header"))?;

                    let buf_ptr = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
                    let buf_len = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);

                    let buf_start = usize::try_from(buf_ptr)
                        .map_err(|_| wasmtime::Error::msg("buf_ptr overflow"))?;
                    let buf_count = usize::try_from(buf_len)
                        .map_err(|_| wasmtime::Error::msg("buf_len overflow"))?;
                    let buf_end = buf_start
                        .checked_add(buf_count)
                        .ok_or_else(|| wasmtime::Error::msg("buf slice overflow"))?;

                    let chunk = mem
                        .get(buf_start..buf_end)
                        .ok_or_else(|| wasmtime::Error::msg("out-of-bounds iovec buffer"))?;

                    if fd == 1 {
                        data.push_stdout(chunk)?;
                    }

                    total_written = total_written
                        .checked_add(buf_len)
                        .ok_or_else(|| wasmtime::Error::msg("total_written overflow"))?;
                }

                let nw_end = nwritten_base
                    .checked_add(4)
                    .ok_or_else(|| wasmtime::Error::msg("nwritten_ptr slice overflow"))?;
                let target = mem
                    .get_mut(nwritten_base..nw_end)
                    .ok_or_else(|| wasmtime::Error::msg("out-of-bounds nwritten_ptr"))?;
                target.copy_from_slice(&total_written.to_le_bytes());

                Ok(WASI_ESUCCESS)
            },
        )
        .map_err(|err| VerifierError::SandboxCompile {
            message: err.to_string(),
        })?;

    let instance = linker.instantiate(&mut store, &module).map_err(|err| {
        let message = err.to_string();
        let lower = message.to_ascii_lowercase();
        if lower.contains("memory")
            && (lower.contains("limit") || lower.contains("minimum") || lower.contains("exceed"))
        {
            VerifierError::SandboxMemoryLimit {
                max_bytes: config.max_memory_bytes,
            }
        } else {
            VerifierError::SandboxTrap { message }
        }
    })?;

    let entrypoint = instance
        .get_typed_func::<(), ()>(&mut store, "_start")
        .or_else(|_| instance.get_typed_func::<(), ()>(&mut store, "run"))
        .map_err(|_| VerifierError::SandboxMissingEntrypoint)?;

    if let Err(err) = entrypoint.call(&mut store, ()) {
        if store.data().stdout_overflowed {
            return Err(VerifierError::SandboxStdoutLimit {
                max_bytes: config.max_stdout_bytes,
            });
        }
        if err.downcast_ref::<Trap>() == Some(&Trap::OutOfFuel) {
            return Err(VerifierError::SandboxOutOfFuel {
                limit: config.max_fuel,
            });
        }
        return Err(VerifierError::SandboxTrap {
            message: err.to_string(),
        });
    }

    let remaining_fuel = store.get_fuel().unwrap_or(0);
    let fuel_consumed = config.max_fuel.saturating_sub(remaining_fuel);

    let stdout_bytes = std::mem::take(&mut store.data_mut().stdout);
    let stdout = String::from_utf8(stdout_bytes).map_err(|err| VerifierError::SandboxTrap {
        message: format!("stdout is not valid utf-8: {err}"),
    })?;

    let passed = normalize_line_endings(&stdout) == normalize_line_endings(expected_stdout);

    tracing::debug!(
        fuel_consumed,
        passed,
        stdout_len = stdout.len(),
        "wasm sandbox finished execution"
    );

    Ok(SandboxOutput {
        stdout,
        fuel_consumed,
        passed,
    })
}

#[cfg(feature = "wasm-sandbox")]
fn normalize_line_endings(text: &str) -> &str {
    text.trim_end_matches(['\r', '\n'])
}

#[cfg(not(feature = "wasm-sandbox"))]
fn execute_wasmtime(
    _wat_source: &str,
    _expected_stdout: &str,
    _config: &SandboxConfig,
) -> Result<SandboxOutput, VerifierError> {
    Err(VerifierError::FeatureNotEnabled {
        feature: "wasm-sandbox",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_or_oversized_wat_source() {
        assert_eq!(
            run_wat_sandbox("   ", ""),
            Err(VerifierError::EmptyExpression)
        );

        let tiny_cfg = SandboxConfig {
            max_source_bytes: 8,
            ..SandboxConfig::default()
        };
        assert_eq!(
            run_wat_sandbox_with_config("(module (func))", "", &tiny_cfg),
            Err(VerifierError::ExpressionTooLong { max: 8 })
        );
    }

    #[cfg(feature = "wasm-sandbox")]
    #[test]
    fn executes_wat_and_captures_stdout_via_env_imports() -> Result<(), Box<dyn std::error::Error>>
    {
        let wat = r#"
            (module
              (import "env" "write" (func $write (param i32 i32)))
              (import "env" "write_i64" (func $write_i64 (param i64)))
              (memory (export "memory") 1)
              (data (i32.const 0) "5! = ")
              (func (export "run")
                (call $write (i32.const 0) (i32.const 5))
                (call $write_i64 (i64.const 120))
              )
            )
        "#;

        let output = run_wat_sandbox(wat, "5! = 120")?;
        assert!(output.passed);
        assert_eq!(output.stdout, "5! = 120");
        assert!(output.fuel_consumed > 0);
        Ok(())
    }

    #[cfg(feature = "wasm-sandbox")]
    #[test]
    fn supports_standard_wasi_fd_write_stdout() -> Result<(), Box<dyn std::error::Error>> {
        let wat = r#"
            (module
              (import "wasi_snapshot_preview1" "fd_write"
                (func $fd_write (param i32 i32 i32 i32) (result i32)))
              (memory (export "memory") 1)
              ;; iovec em offset 0: ptr = 16, len = 9
              (data (i32.const 0) "\10\00\00\00\09\00\00\00")
              (data (i32.const 16) "ok-wasi\n!")
              (func (export "_start")
                (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 64)))
              )
            )
        "#;

        let output = run_wat_sandbox(wat, "ok-wasi\n!")?;
        assert!(output.passed);
        assert_eq!(output.stdout, "ok-wasi\n!");
        Ok(())
    }

    #[cfg(feature = "wasm-sandbox")]
    #[test]
    fn stops_infinite_loop_when_fuel_is_exhausted() {
        let wat = r#"
            (module
              (func (export "run")
                (loop $inf
                  (br $inf)
                )
              )
            )
        "#;

        let cfg = SandboxConfig {
            max_fuel: 1_000,
            ..SandboxConfig::default()
        };
        assert_eq!(
            run_wat_sandbox_with_config(wat, "", &cfg),
            Err(VerifierError::SandboxOutOfFuel { limit: 1_000 })
        );
    }

    #[cfg(feature = "wasm-sandbox")]
    #[test]
    fn enforces_memory_limit_on_initial_allocation_and_growth()
    -> Result<(), Box<dyn std::error::Error>> {
        // 32 páginas = 2 MiB (excede o padrão de 1 MiB)
        let too_big_initial = r#"
            (module
              (memory (export "memory") 32)
              (func (export "run"))
            )
        "#;
        assert_eq!(
            run_wat_sandbox(too_big_initial, ""),
            Err(VerifierError::SandboxMemoryLimit {
                max_bytes: DEFAULT_MAX_MEMORY_BYTES,
            })
        );

        // Começa com 1 página (64 KiB) e tenta crescer +20 páginas -> memory.grow deve retornar -1
        let grow_beyond_limit = r#"
            (module
              (import "env" "write_i64" (func $write_i64 (param i64)))
              (memory (export "memory") 1)
              (func (export "run")
                (call $write_i64 (i64.extend_i32_s (memory.grow (i32.const 20))))
              )
            )
        "#;
        let output = run_wat_sandbox(grow_beyond_limit, "-1")?;
        assert!(output.passed);
        assert_eq!(output.stdout, "-1");
        Ok(())
    }

    #[cfg(feature = "wasm-sandbox")]
    #[test]
    fn blocks_stdout_flood_beyond_byte_cap() {
        let wat = r#"
            (module
              (import "env" "write" (func $write (param i32 i32)))
              (memory (export "memory") 1)
              (data (i32.const 0) "0123456789ABCDEF")
              (func (export "run")
                (call $write (i32.const 0) (i32.const 16))
                (call $write (i32.const 0) (i32.const 16))
              )
            )
        "#;

        let cfg = SandboxConfig {
            max_stdout_bytes: 20,
            ..SandboxConfig::default()
        };
        assert_eq!(
            run_wat_sandbox_with_config(wat, "", &cfg),
            Err(VerifierError::SandboxStdoutLimit { max_bytes: 20 })
        );
    }
}
