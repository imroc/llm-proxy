use serde_json::Value;
use tracing_subscriber::layer::{Layer, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{reload, EnvFilter, Registry};

type FilterHandle = reload::Handle<EnvFilter, Registry>;

/// Handle for dynamically updating the log level at runtime.
#[derive(Clone)]
pub struct LogLevelHandle {
    handle: FilterHandle,
}

impl LogLevelHandle {
    /// Update the log filter at runtime, e.g. "info", "debug", "trace".
    ///
    /// The level is scoped to the `llm_proxy` crate only — dependency crates
    /// (h2, hyper, reqwest, etc.) stay at `warn` to avoid noisy frame-level logs.
    pub fn set_level(&self, level: &str) {
        let scoped = Self::scope_level(level);
        match EnvFilter::try_new(&scoped) {
            Ok(filter) => {
                if self.handle.modify(|f| *f = filter).is_ok() {
                    tracing::info!("log level updated: {}", level);
                }
            }
            Err(e) => {
                tracing::warn!("invalid log level '{}', keeping current: {}", level, e);
            }
        }
    }

    /// Scope a bare level string to the llm_proxy crate.
    ///
    /// "debug" → "warn,llm_proxy=debug"
    /// "warn,llm_proxy=debug" → passed through as-is (already scoped)
    fn scope_level(level: &str) -> String {
        // If the user already provided a complex filter, use it as-is
        if level.contains(',') || level.contains('=') {
            level.to_string()
        } else {
            format!("warn,llm_proxy={}", level)
        }
    }
}

/// Initialize tracing subscriber with the given log level.
/// Returns a handle that can update the filter at runtime (config hot reload).
///
/// Supports `RUST_LOG` env var override (takes precedence over `level` at startup;
/// later runtime updates via the returned handle still apply).
/// If `RUST_LOG_FORMAT=json`, use JSON output.
pub fn init_tracing(level: &str) -> LogLevelHandle {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(LogLevelHandle::scope_level(level)));
    let (filter, handle) = reload::Layer::new(filter);

    let format = std::env::var("RUST_LOG_FORMAT").unwrap_or_default();
    if format == "json" {
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_target(false)
            .json()
            .with_filter(filter);
        Registry::default().with(fmt_layer).init();
    } else {
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_target(false)
            .with_filter(filter);
        Registry::default().with(fmt_layer).init();
    }

    LogLevelHandle { handle }
}

/// Best-effort extraction of the `model` field from a request body.
///
/// Works with OpenAI chat completions format (`{"model": "glm-latest", ...}`)
/// and Anthropic Messages format (`{"model": "claude-3", ...}`).
/// Returns `None` on any parse error.
pub fn extract_model(body: &[u8]) -> Option<String> {
    if body.is_empty() {
        return None;
    }
    let value: Value = serde_json::from_slice(body).ok()?;
    value.get("model")?.as_str().map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_model_openai() {
        let body = br#"{"model": "glm-latest", "messages": []}"#;
        assert_eq!(extract_model(body), Some("glm-latest".into()));
    }

    #[test]
    fn test_extract_model_anthropic() {
        let body = br#"{"model": "claude-3", "max_tokens": 1024}"#;
        assert_eq!(extract_model(body), Some("claude-3".into()));
    }

    #[test]
    fn test_extract_model_empty() {
        assert_eq!(extract_model(b""), None);
    }

    #[test]
    fn test_extract_model_no_model_field() {
        let body = br#"{"foo": "bar"}"#;
        assert_eq!(extract_model(body), None);
    }

    #[test]
    fn test_extract_model_invalid_json() {
        assert_eq!(extract_model(b"not json"), None);
    }
}
