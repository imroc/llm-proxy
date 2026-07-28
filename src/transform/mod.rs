//! Protocol transform module — bidirectional conversion between all protocol pairs.
//!
//! Supported conversions:
//! - responses ↔ chat
//! - responses ↔ anthropic
//! - chat ↔ anthropic
//!
//! Each conversion has:
//! - Request body transform (JSON → JSON)
//! - Response body transform (non-streaming, JSON → JSON)
//! - Response SSE stream transform (streaming, chunk → SSE events)

pub mod chat_anthropic;
pub mod common;
pub mod history;
pub mod responses_anthropic;
pub mod responses_chat;

use crate::format::Protocol;
use bytes::Bytes;

/// The result of a request transform: the converted body bytes and the
/// upstream API path to use (e.g., `/v1/chat/completions`).
pub struct TransformedRequest {
    pub body: Bytes,
    pub upstream_path: String,
}

/// Dispatch request transformation based on the conversion direction.
///
/// Returns `Ok(TransformedRequest)` if conversion succeeded,
/// `Err(message)` if the conversion failed (caller should decide whether
/// to fall back to passthrough or return an error).
pub fn transform_request(
    from: Protocol,
    to: Protocol,
    body: &[u8],
) -> Result<TransformedRequest, String> {
    if from == to {
        // Passthrough — no conversion needed.
        return Ok(TransformedRequest {
            body: Bytes::copy_from_slice(body),
            upstream_path: to.api_path().to_string(),
        });
    }

    let body_val: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| format!("failed to parse request body as JSON: {}", e))?;

    let converted = match (from, to) {
        (Protocol::Responses, Protocol::Chat) => {
            responses_chat::request_responses_to_chat(&body_val)
                .map_err(|e| format!("responses→chat request conversion failed: {}", e))?
        }
        (Protocol::Chat, Protocol::Responses) => {
            responses_chat::request_chat_to_responses(&body_val)
                .map_err(|e| format!("chat→responses request conversion failed: {}", e))?
        }
        (Protocol::Responses, Protocol::Anthropic) => {
            responses_anthropic::request_responses_to_anthropic(&body_val)
                .map_err(|e| format!("responses→anthropic request conversion failed: {}", e))?
        }
        (Protocol::Anthropic, Protocol::Responses) => {
            responses_anthropic::request_anthropic_to_responses(&body_val)
                .map_err(|e| format!("anthropic→responses request conversion failed: {}", e))?
        }
        (Protocol::Chat, Protocol::Anthropic) => {
            chat_anthropic::request_chat_to_anthropic(&body_val)
                .map_err(|e| format!("chat→anthropic request conversion failed: {}", e))?
        }
        (Protocol::Anthropic, Protocol::Chat) => {
            chat_anthropic::request_anthropic_to_chat(&body_val)
                .map_err(|e| format!("anthropic→chat request conversion failed: {}", e))?
        }
        _ => return Err(format!("no conversion needed: {} to {}", from, to)),
    };

    let body = serde_json::to_vec(&converted)
        .map_err(|e| format!("failed to serialize converted request: {}", e))?;

    Ok(TransformedRequest {
        body: Bytes::from(body),
        upstream_path: to.api_path().to_string(),
    })
}

/// Dispatch non-streaming response transformation.
///
/// `from` is the upstream's protocol, `to` is the client's protocol.
pub fn transform_response_body(
    from: Protocol,
    to: Protocol,
    body: &[u8],
    client_model: Option<&str>,
) -> Result<Bytes, String> {
    if from == to {
        // Passthrough — optionally rewrite model name.
        return Ok(common::rewrite_model_in_response(body, client_model));
    }

    let body_val: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| format!("failed to parse response body: {}", e))?;

    let converted = match (from, to) {
        (Protocol::Chat, Protocol::Responses) => {
            responses_chat::response_chat_to_responses(&body_val, client_model)
        }
        (Protocol::Responses, Protocol::Chat) => {
            responses_chat::response_responses_to_chat(&body_val, client_model)
        }
        (Protocol::Anthropic, Protocol::Responses) => {
            responses_anthropic::response_anthropic_to_responses(&body_val, client_model)
        }
        (Protocol::Responses, Protocol::Anthropic) => {
            responses_anthropic::response_responses_to_anthropic(&body_val, client_model)
        }
        (Protocol::Anthropic, Protocol::Chat) => {
            chat_anthropic::response_anthropic_to_chat(&body_val, client_model)
        }
        (Protocol::Chat, Protocol::Anthropic) => {
            chat_anthropic::response_chat_to_anthropic(&body_val, client_model)
        }
        _ => return Err(format!("no conversion needed: {} to {}", from, to)),
    };

    let converted =
        converted.ok_or_else(|| format!("response conversion {}→{} produced None", from, to))?;

    serde_json::to_vec(&converted)
        .map(Bytes::from)
        .map_err(|e| format!("failed to serialize converted response: {}", e))
}

/// Create a streaming SSE transformer for the given conversion direction.
///
/// `from` is the upstream's protocol, `to` is the client's protocol.
/// Returns a `StreamTransformDispatch` that can process SSE chunks.
pub enum StreamTransformer {
    Passthrough {
        client_model: Option<String>,
    },
    ChatToResponses {
        state: responses_chat::ChatToResponsesStreamState,
    },
    ResponsesToChat {
        state: responses_chat::ResponsesToChatStreamState,
    },
    AnthropicToResponses {
        state: responses_anthropic::AnthropicToResponsesStreamState,
    },
    ResponsesToAnthropic {
        state: responses_anthropic::ResponsesToAnthropicStreamState,
    },
    AnthropicToChat {
        state: chat_anthropic::AnthropicToChatStreamState,
    },
    ChatToAnthropic {
        state: chat_anthropic::ChatToAnthropicStreamState,
    },
}

impl StreamTransformer {
    /// Create a new stream transformer for the given conversion.
    pub fn new(from: Protocol, to: Protocol, client_model: Option<&str>) -> Self {
        match (from, to) {
            (a, b) if a == b => StreamTransformer::Passthrough {
                client_model: client_model.map(|s| s.to_string()),
            },
            (Protocol::Chat, Protocol::Responses) => StreamTransformer::ChatToResponses {
                state: responses_chat::ChatToResponsesStreamState::new(client_model.unwrap_or("")),
            },
            (Protocol::Responses, Protocol::Chat) => StreamTransformer::ResponsesToChat {
                state: responses_chat::ResponsesToChatStreamState::new(client_model.unwrap_or("")),
            },
            (Protocol::Anthropic, Protocol::Responses) => StreamTransformer::AnthropicToResponses {
                state: responses_anthropic::AnthropicToResponsesStreamState::new(
                    client_model.unwrap_or(""),
                ),
            },
            (Protocol::Responses, Protocol::Anthropic) => StreamTransformer::ResponsesToAnthropic {
                state: responses_anthropic::ResponsesToAnthropicStreamState::new(
                    client_model.unwrap_or(""),
                ),
            },
            (Protocol::Anthropic, Protocol::Chat) => StreamTransformer::AnthropicToChat {
                state: chat_anthropic::AnthropicToChatStreamState::new(client_model.unwrap_or("")),
            },
            (Protocol::Chat, Protocol::Anthropic) => StreamTransformer::ChatToAnthropic {
                state: chat_anthropic::ChatToAnthropicStreamState::new(client_model.unwrap_or("")),
            },
            _ => StreamTransformer::Passthrough {
                client_model: client_model.map(|s| s.to_string()),
            },
        }
    }

    /// Process one SSE line from the upstream and return SSE lines to send to the client.
    ///
    /// For passthrough mode, this rewrites the model field in `data:` lines.
    /// For conversion modes, this transforms the chunk into the target protocol's SSE format.
    ///
    /// Returns `Some(output)` with SSE data lines to send, or `None` if nothing to emit.
    /// When the `[DONE]` marker is received, the transformer flushes any pending state.
    pub fn transform_sse_line(&mut self, line: &str) -> Option<String> {
        match self {
            StreamTransformer::Passthrough { client_model } => {
                common::passthrough_sse_line(line, client_model.as_deref())
            }
            StreamTransformer::ChatToResponses { state } => {
                responses_chat::chat_sse_to_responses_sse(line, state)
            }
            StreamTransformer::ResponsesToChat { state } => {
                responses_chat::responses_sse_to_chat_sse(line, state)
            }
            StreamTransformer::AnthropicToResponses { state } => {
                responses_anthropic::anthropic_sse_to_responses_sse(line, state)
            }
            StreamTransformer::ResponsesToAnthropic { state } => {
                responses_anthropic::responses_sse_to_anthropic_sse(line, state)
            }
            StreamTransformer::AnthropicToChat { state } => {
                chat_anthropic::anthropic_sse_to_chat_sse(line, state)
            }
            StreamTransformer::ChatToAnthropic { state } => {
                chat_anthropic::chat_sse_to_anthropic_sse(line, state)
            }
        }
    }

    /// Whether this transformer is passthrough (no protocol conversion).
    pub fn is_passthrough(&self) -> bool {
        matches!(self, StreamTransformer::Passthrough { .. })
    }
}
