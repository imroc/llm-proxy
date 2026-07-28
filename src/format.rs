//! Protocol format detection and conversion dispatch.
//!
//! Three supported protocol formats:
//! - `Responses` — OpenAI Responses API (`/v1/responses`)
//! - `Chat` — OpenAI Chat Completions (`/v1/chat/completions`)
//! - `Anthropic` — Anthropic Messages (`/v1/messages`)

use serde::{Deserialize, Serialize};
use std::fmt;

/// The three LLM API protocol formats supported by llm-proxy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    /// OpenAI Responses API (`POST /v1/responses`).
    Responses,
    /// OpenAI Chat Completions (`POST /v1/chat/completions`).
    Chat,
    /// Anthropic Messages (`POST /v1/messages`).
    Anthropic,
}

impl fmt::Display for Protocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Protocol::Responses => write!(f, "responses"),
            Protocol::Chat => write!(f, "chat"),
            Protocol::Anthropic => write!(f, "anthropic"),
        }
    }
}

impl Protocol {
    /// Parse a protocol from a string (used by serde config parsing).
    pub fn parse_from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "responses" | "response" => Some(Protocol::Responses),
            "chat" | "chat_completions" | "chat-completions" => Some(Protocol::Chat),
            "anthropic" | "messages" => Some(Protocol::Anthropic),
            _ => None,
        }
    }

    /// The standard URL path for this protocol format.
    ///
    /// - Responses → `/v1/responses`
    /// - Chat → `/v1/chat/completions`
    /// - Anthropic → `/v1/messages`
    pub fn api_path(&self) -> &'static str {
        match self {
            Protocol::Responses => "/v1/responses",
            Protocol::Chat => "/v1/chat/completions",
            Protocol::Anthropic => "/v1/messages",
        }
    }

    /// Detect the inbound protocol from the request URL path.
    ///
    /// Returns `None` if the path doesn't match any known API path.
    pub fn from_path(path: &str) -> Option<Self> {
        // Normalize: strip leading route segment if present (handled by caller).
        // We look for the API-specific suffix.
        if path.ends_with("/v1/responses") || path.ends_with("/responses") {
            Some(Protocol::Responses)
        } else if path.ends_with("/v1/chat/completions") || path.ends_with("/chat/completions") {
            Some(Protocol::Chat)
        } else if path.ends_with("/v1/messages") || path.ends_with("/messages") {
            Some(Protocol::Anthropic)
        } else {
            None
        }
    }
}

/// A flexible list of upstream-supported formats, ordered by preference.
///
/// Parsed from config like `upstream_formats = ["chat", "anthropic"]`.
/// The order matters: when conversion is needed, the first format the
/// inbound protocol is not already is selected.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(transparent)]
pub struct UpstreamFormats(pub Vec<Protocol>);

impl UpstreamFormats {
    /// Empty (no formats declared) — means "accept anything, passthrough".
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Check if the inbound protocol is directly supported by the upstream.
    pub fn supports(&self, inbound: Protocol) -> bool {
        self.0.is_empty() || self.0.contains(&inbound)
    }

    /// Select the target format for forwarding to the upstream.
    ///
    /// - If `inbound` is in the list (or list is empty) → return `inbound` (passthrough).
    /// - Otherwise → return the first format in the list (convert to it).
    pub fn select_target(&self, inbound: Protocol) -> Protocol {
        if self.supports(inbound) {
            inbound
        } else {
            self.0[0]
        }
    }
}

/// Decide whether a conversion is needed, and if so, in which direction.
///
/// Returns `None` if passthrough (no conversion).
/// Returns `Some((from, to))` if conversion is needed.
pub fn conversion_direction(
    inbound: Protocol,
    upstream_formats: &UpstreamFormats,
) -> Option<(Protocol, Protocol)> {
    let target = upstream_formats.select_target(inbound);
    if target == inbound {
        None
    } else {
        Some((inbound, target))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_path() {
        assert_eq!(
            Protocol::from_path("/v1/responses"),
            Some(Protocol::Responses)
        );
        assert_eq!(
            Protocol::from_path("/v1/chat/completions"),
            Some(Protocol::Chat)
        );
        assert_eq!(
            Protocol::from_path("/v1/messages"),
            Some(Protocol::Anthropic)
        );
        // With route prefix
        assert_eq!(
            Protocol::from_path("/tkehub/v1/responses"),
            Some(Protocol::Responses)
        );
        assert_eq!(
            Protocol::from_path("/default/v1/chat/completions"),
            Some(Protocol::Chat)
        );
        // Unknown
        assert_eq!(Protocol::from_path("/v1/models"), None);
        assert_eq!(Protocol::from_path("/healthz"), None);
    }

    #[test]
    fn test_select_target_passthrough() {
        let uf = UpstreamFormats(vec![Protocol::Responses]);
        assert_eq!(uf.select_target(Protocol::Responses), Protocol::Responses);
    }

    #[test]
    fn test_select_target_convert() {
        let uf = UpstreamFormats(vec![Protocol::Chat, Protocol::Anthropic]);
        // Inbound responses not in list → pick first (chat)
        assert_eq!(uf.select_target(Protocol::Responses), Protocol::Chat);
    }

    #[test]
    fn test_select_target_match_not_first() {
        let uf = UpstreamFormats(vec![Protocol::Chat, Protocol::Anthropic]);
        // Inbound anthropic is in list → passthrough (match priority)
        assert_eq!(uf.select_target(Protocol::Anthropic), Protocol::Anthropic);
    }

    #[test]
    fn test_select_target_empty_passthrough() {
        let uf = UpstreamFormats(vec![]);
        // Empty list → passthrough anything
        assert_eq!(uf.select_target(Protocol::Responses), Protocol::Responses);
        assert_eq!(uf.select_target(Protocol::Chat), Protocol::Chat);
    }

    #[test]
    fn test_conversion_direction() {
        let uf = UpstreamFormats(vec![Protocol::Chat]);
        assert_eq!(conversion_direction(Protocol::Chat, &uf), None); // passthrough
        assert_eq!(
            conversion_direction(Protocol::Responses, &uf),
            Some((Protocol::Responses, Protocol::Chat))
        );
    }
}
