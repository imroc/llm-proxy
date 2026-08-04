use crate::format::UpstreamFormats;
use arc_swap::ArcSwap;
use notify::{event::EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

/// Special route name for the default route.
/// Requests without a route name in the URL path hit this route.
pub const DEFAULT_ROUTE: &str = "default";

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Log level (error, warn, info, debug, trace). Hot-reloadable.
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default)]
    pub defaults: Defaults,
    #[serde(default)]
    pub routes: HashMap<String, Route>,
}

fn default_log_level() -> String {
    "info".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct Defaults {
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default = "default_base_delay_ms")]
    pub base_delay_ms: u64,
    #[serde(default = "default_max_delay_ms")]
    pub max_delay_ms: u64,
    #[serde(default)]
    pub max_total_wait_ms: u64,
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,
    #[serde(default = "default_retry_status_codes")]
    pub retry_status_codes: Vec<u16>,
}

impl Default for Defaults {
    fn default() -> Self {
        Self {
            max_retries: default_max_retries(),
            base_delay_ms: default_base_delay_ms(),
            max_delay_ms: default_max_delay_ms(),
            max_total_wait_ms: 0,
            connect_timeout_secs: default_connect_timeout_secs(),
            retry_status_codes: default_retry_status_codes(),
        }
    }
}

/// Model-level override configuration.
///
/// All fields are optional — only specified fields override the route-level config.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub target: Option<String>,
    /// Upstream-supported protocol formats, ordered by preference.
    /// Empty (unset) means "accept anything, passthrough".
    #[serde(default)]
    pub upstream_formats: UpstreamFormats,
    /// API key for the upstream (supports `${ENV_VAR}` expansion).
    #[serde(default)]
    pub api_key: Option<String>,
    /// Rewrite the `model` field in the request body before forwarding upstream.
    #[serde(default)]
    pub upstream_model: Option<String>,
    #[serde(default)]
    pub max_retries: Option<u32>,
    #[serde(default)]
    pub base_delay_ms: Option<u64>,
    #[serde(default)]
    pub max_delay_ms: Option<u64>,
    #[serde(default)]
    pub max_total_wait_ms: Option<u64>,
    #[serde(default)]
    pub connect_timeout_secs: Option<u64>,
    #[serde(default)]
    pub retry_status_codes: Option<Vec<u16>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Route {
    /// Upstream base URL. Optional for default route (each model has its own target).
    #[serde(default)]
    pub target: Option<String>,
    /// Upstream-supported protocol formats, ordered by preference.
    #[serde(default)]
    pub upstream_formats: UpstreamFormats,
    /// API key for the upstream (supports `${ENV_VAR}` expansion).
    #[serde(default)]
    pub api_key: Option<String>,
    /// Rewrite the `model` field in the request body before forwarding upstream.
    #[serde(default)]
    pub upstream_model: Option<String>,
    /// Model-level overrides: keyed by the model name extracted from the request body.
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,
    #[serde(default)]
    pub max_retries: Option<u32>,
    #[serde(default)]
    pub base_delay_ms: Option<u64>,
    #[serde(default)]
    pub max_delay_ms: Option<u64>,
    #[serde(default)]
    pub max_total_wait_ms: Option<u64>,
    #[serde(default)]
    pub connect_timeout_secs: Option<u64>,
    #[serde(default)]
    pub retry_status_codes: Option<Vec<u16>>,
}

/// Resolved route config: route-level overrides merged with defaults.
/// Call `resolve_model()` to further apply model-level overrides.
#[derive(Debug, Clone)]
pub struct ResolvedRouteConfig {
    pub target: Option<String>,
    pub upstream_formats: UpstreamFormats,
    pub api_key: Option<String>,
    pub upstream_model: Option<String>,
    pub models: HashMap<String, ModelConfig>,
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub max_total_wait_ms: u64,
    pub connect_timeout_secs: u64,
    pub retry_status_codes: Vec<u16>,
}

fn default_max_retries() -> u32 {
    9999
}
fn default_base_delay_ms() -> u64 {
    1000
}
fn default_max_delay_ms() -> u64 {
    60000
}
fn default_connect_timeout_secs() -> u64 {
    30
}
fn default_retry_status_codes() -> Vec<u16> {
    vec![429, 500, 502, 503, 504, 408, 529]
}

/// Expand `${ENV_VAR}` references in a string value.
/// If the env var is not set, leaves the reference as-is.
pub fn expand_env_vars(s: &str) -> String {
    let mut result = s.to_string();
    while let Some(start) = result.find("${") {
        if let Some(end) = result[start..].find('}') {
            let var_name = &result[start + 2..start + end];
            if let Ok(val) = std::env::var(var_name) {
                result.replace_range(start..start + end + 1, &val);
            } else {
                // Env var not set — skip to avoid infinite loop
                break;
            }
        } else {
            break;
        }
    }
    result
}

#[derive(Debug)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            ConfigError(format!(
                "failed to read config file {}: {}",
                path.display(),
                e
            ))
        })?;
        let config: Config = toml::from_str(&content)
            .map_err(|e| ConfigError(format!("failed to parse TOML: {}", e)))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.routes.is_empty() {
            return Err(ConfigError("no routes defined".into()));
        }
        for (name, route) in &self.routes {
            if name.contains('/') {
                return Err(ConfigError(format!(
                    "route name '{}' must not contain '/'",
                    name
                )));
            }
            // Named routes (non-default) must have a target.
            // Default route may omit target (each model has its own).
            if name != DEFAULT_ROUTE && route.target.is_none() {
                return Err(ConfigError(format!(
                    "route '{}' (non-default) must have a target",
                    name
                )));
            }
            // Validate URLs
            if let Some(ref target) = route.target {
                if target.is_empty() {
                    return Err(ConfigError(format!("route '{}' has empty target", name)));
                }
                if let Err(e) = target.parse::<http::Uri>() {
                    return Err(ConfigError(format!(
                        "route '{}' target '{}' is not a valid URL: {}",
                        name, target, e
                    )));
                }
            }
            // Validate model-level configs
            for (model_name, mc) in &route.models {
                if model_name.is_empty() {
                    return Err(ConfigError(format!(
                        "route '{}' has a model entry with empty name",
                        name
                    )));
                }
                if let Some(ref target) = mc.target {
                    if target.is_empty() {
                        return Err(ConfigError(format!(
                            "route '{}' model '{}' has empty target",
                            name, model_name
                        )));
                    }
                    if let Err(e) = target.parse::<http::Uri>() {
                        return Err(ConfigError(format!(
                            "route '{}' model '{}' target '{}' is not a valid URL: {}",
                            name, model_name, target, e
                        )));
                    }
                }
            }
        }
        if self.defaults.max_retries == 0 {
            return Err(ConfigError("defaults.max_retries must be > 0".into()));
        }
        if self.defaults.base_delay_ms == 0 {
            return Err(ConfigError("defaults.base_delay_ms must be > 0".into()));
        }
        if self.defaults.max_delay_ms == 0 {
            return Err(ConfigError("defaults.max_delay_ms must be > 0".into()));
        }
        Ok(())
    }

    /// Resolve a named route by name, merging route-level config with defaults.
    pub fn resolve_route(&self, name: &str) -> Option<ResolvedRouteConfig> {
        let route = self.routes.get(name)?;
        let d = &self.defaults;
        Some(ResolvedRouteConfig {
            target: route.target.clone(),
            upstream_formats: route.upstream_formats.clone(),
            api_key: route.api_key.as_ref().map(|s| expand_env_vars(s)),
            upstream_model: route.upstream_model.clone(),
            models: route.models.clone(),
            max_retries: route.max_retries.unwrap_or(d.max_retries),
            base_delay_ms: route.base_delay_ms.unwrap_or(d.base_delay_ms),
            max_delay_ms: route.max_delay_ms.unwrap_or(d.max_delay_ms),
            max_total_wait_ms: route.max_total_wait_ms.unwrap_or(d.max_total_wait_ms),
            connect_timeout_secs: route.connect_timeout_secs.unwrap_or(d.connect_timeout_secs),
            retry_status_codes: route
                .retry_status_codes
                .clone()
                .unwrap_or_else(|| d.retry_status_codes.clone()),
        })
    }

    /// Get the default route config, if configured.
    pub fn resolve_default_route(&self) -> Option<ResolvedRouteConfig> {
        self.resolve_route(DEFAULT_ROUTE)
    }

    pub fn route_names(&self) -> Vec<&str> {
        self.routes.keys().map(|s| s.as_str()).collect()
    }

    /// Collect all model names from all routes (for GET /v1/models).
    pub fn all_model_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for route in self.routes.values() {
            for model_name in route.models.keys() {
                names.push(model_name.clone());
            }
        }
        names.sort();
        names.dedup();
        names
    }
}

impl ResolvedRouteConfig {
    /// Apply model-level overrides on top of the resolved route config.
    ///
    /// If the model is not found in the models map, returns a clone of self unchanged.
    pub fn resolve_model(&self, model: &str) -> ResolvedRouteConfig {
        let Some(mc) = self.models.get(model) else {
            return self.clone();
        };
        let mut result = self.clone();
        if let Some(t) = &mc.target {
            result.target = Some(t.clone());
        }
        if !mc.upstream_formats.is_empty() {
            result.upstream_formats = mc.upstream_formats.clone();
        }
        if let Some(v) = &mc.api_key {
            result.api_key = Some(expand_env_vars(v));
        }
        if let Some(v) = &mc.upstream_model {
            result.upstream_model = Some(v.clone());
        }
        if let Some(v) = mc.max_retries {
            result.max_retries = v;
        }
        if let Some(v) = mc.base_delay_ms {
            result.base_delay_ms = v;
        }
        if let Some(v) = mc.max_delay_ms {
            result.max_delay_ms = v;
        }
        if let Some(v) = mc.max_total_wait_ms {
            result.max_total_wait_ms = v;
        }
        if let Some(v) = mc.connect_timeout_secs {
            result.connect_timeout_secs = v;
        }
        if let Some(v) = &mc.retry_status_codes {
            result.retry_status_codes = v.clone();
        }
        result
    }

    /// List model names configured for this route.
    pub fn model_names(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }
}

/// Watches the config file for changes and hot-reloads into ArcSwap.
pub struct ConfigWatcher {
    _watcher: RecommendedWatcher,
}

impl ConfigWatcher {
    pub fn start(
        config: Arc<ArcSwap<Config>>,
        path: PathBuf,
        log_handle: crate::log::LogLevelHandle,
    ) -> Result<Self, ConfigError> {
        let watch_path = path.clone();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if let Ok(event) = res {
                if !matches!(
                    event.kind,
                    EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
                ) {
                    return;
                }
                // Debounce: check mtime, wait 100ms, reload
                let path = watch_path.clone();
                let config = config.clone();
                let log_handle = log_handle.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    match Config::load(&path) {
                        Ok(new_config) => {
                            let route_names = new_config.route_names();
                            info!("config reloaded: routes = {}", route_names.join(", "));
                            log_handle.set_level(&new_config.log_level);
                            config.store(Arc::new(new_config));
                        }
                        Err(e) => {
                            warn!("config reload failed, keeping old config: {}", e);
                        }
                    }
                });
            }
        })
        .map_err(|e| ConfigError(format!("failed to create file watcher: {}", e)))?;

        watcher
            .watch(&path, RecursiveMode::NonRecursive)
            .map_err(|e| ConfigError(format!("failed to watch config file: {}", e)))?;

        Ok(Self { _watcher: watcher })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_env_vars() {
        std::env::set_var("TEST_API_KEY", "secret123");
        assert_eq!(expand_env_vars("${TEST_API_KEY}"), "secret123");
        assert_eq!(
            expand_env_vars("Bearer ${TEST_API_KEY}"),
            "Bearer secret123"
        );
        std::env::remove_var("TEST_API_KEY");
    }

    #[test]
    fn test_expand_env_vars_unset() {
        assert_eq!(expand_env_vars("${NONEXISTENT_VAR}"), "${NONEXISTENT_VAR}");
    }

    #[test]
    fn test_config_default_route_no_target() {
        let toml_str = r#"
[defaults]
max_retries = 9999
base_delay_ms = 1000
max_delay_ms = 60000

[routes.default]

[routes.default.models."test-model"]
target = "http://localhost:8080"
upstream_formats = ["chat"]
api_key = "test-key"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!(config.validate().is_ok());
        let rc = config.resolve_default_route().unwrap();
        assert!(rc.target.is_none()); // default route has no route-level target
    }

    #[test]
    fn test_config_named_route_must_have_target() {
        let toml_str = r#"
[defaults]
max_retries = 9999
base_delay_ms = 1000
max_delay_ms = 60000

[routes.myroute]
# no target — should fail for non-default route
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!(config.validate().is_err());
    }
}
