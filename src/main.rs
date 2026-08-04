use llm_proxy::{config, log, metrics, server};

use arc_swap::ArcSwap;
use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tracing::info;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(name = "llm-proxy", version = VERSION, about = "Thin local proxy for LLM APIs with automatic retry, protocol transform, and model-level routing")]
struct Args {
    /// Path to config file
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// Listen address
    #[arg(short, long, default_value = "127.0.0.1:8888")]
    addr: String,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Load config (strict at startup)
    let config = match config::Config::load(&args.config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "error: failed to load config from {}: {}",
                args.config.display(),
                e
            );
            std::process::exit(1);
        }
    };

    // Init tracing (log level comes from config, hot-reloadable)
    let log_handle = log::init_tracing(&config.log_level);

    let addr: SocketAddr = match args.addr.parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: invalid listen address '{}': {}", args.addr, e);
            std::process::exit(1);
        }
    };

    let route_names = config.route_names();
    info!(
        "llm-proxy v{} starting, listening on http://{}, routes: {}",
        VERSION,
        addr,
        route_names.join(", ")
    );

    // Create reqwest client
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(config.defaults.connect_timeout_secs))
        .redirect(reqwest::redirect::Policy::none())
        .pool_idle_timeout(None)
        .build()
        .expect("failed to build reqwest client");

    // Wrap config in ArcSwap for hot reload
    let config = Arc::new(ArcSwap::from_pointee(config));

    // Start config watcher (must keep the returned watcher alive)
    let _config_watcher =
        match config::ConfigWatcher::start(config.clone(), args.config.clone(), log_handle) {
            Ok(w) => w,
            Err(e) => {
                tracing::warn!("config hot reload disabled: {}", e);
                return;
            }
        };

    // Create metrics
    let metrics = metrics::Metrics::new();

    // Shutdown channel
    let (shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);

    // Signal handler
    let shutdown_tx_clone = shutdown_tx.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("received SIGINT, initiating shutdown");
        let _ = shutdown_tx_clone.send(());
    });

    #[cfg(unix)]
    {
        let shutdown_tx_clone = shutdown_tx.clone();
        tokio::spawn(async move {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm =
                signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");
            sigterm.recv().await;
            tracing::info!("received SIGTERM, initiating shutdown");
            let _ = shutdown_tx_clone.send(());
        });
    }

    // Run server
    if let Err(e) = server::run(config, metrics, client, addr, VERSION, shutdown_rx).await {
        tracing::error!("server error: {}", e);
        std::process::exit(1);
    }

    // Grace period
    tracing::info!("graceful shutdown: waiting 5s for in-flight requests");
    tokio::time::sleep(Duration::from_secs(5)).await;
    tracing::info!("shutdown complete");
}
