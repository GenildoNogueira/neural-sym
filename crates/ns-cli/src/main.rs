use std::process::ExitCode;

use clap::Parser;
use ns_cli::{Cli, run};

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    if let Err(err) = run(cli).await {
        eprintln!("error: {err:#}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}
