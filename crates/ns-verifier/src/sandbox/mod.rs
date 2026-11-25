mod wasmtime_runner;

pub use wasmtime_runner::{
    DEFAULT_MAX_FUEL, DEFAULT_MAX_MEMORY_BYTES, DEFAULT_MAX_SOURCE_BYTES, DEFAULT_MAX_STDOUT_BYTES,
    SandboxConfig, run_wat_sandbox, run_wat_sandbox_with_config,
};
