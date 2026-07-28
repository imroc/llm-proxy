//! Shared utilities for protocol transforms: ID generation, SSE formatting,
//! model name rewriting, and passthrough helpers.

use bytes::Bytes;
use serde_json::Value;
use uuid::Uuid;

/// Generate an ID with a prefix, using a UUID (hyphens removed).
pub fn make_id(prefix: &str) -> String {
    let id = Uuid::new_v4().to_string().replace('-', "");
    format!("{}_{}", prefix, &id[..24])
}

/// Format a JSON value as an SSE `data:` line.
pub fn sse_line(data: &Value) -> String {
    format!(
        "data: {}\n\n",
        serde_json::to_string(data).unwrap_or_default()
    )
}

/// Rewrite the `model` field in a JSON response body (non-streaming).
///
/// If the body is not valid JSON or has no `model` field, returns the original bytes.
pub fn rewrite_model_in_response(body: &[u8], client_model: Option<&str>) -> Bytes {
    let Some(model) = client_model else {
        return Bytes::copy_from_slice(body);
    };
    let mut value: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => return Bytes::copy_from_slice(body),
    };
    if let Some(obj) = value.as_object_mut() {
        obj.insert("model".to_string(), Value::String(model.to_string()));
        serde_json::to_vec(&value)
            .map(Bytes::from)
            .unwrap_or_else(|_| Bytes::copy_from_slice(body))
    } else {
        Bytes::copy_from_slice(body)
    }
}

/// Process a passthrough SSE line, optionally rewriting the `model` field.
///
/// For `data: {...}` lines containing a `model` field, rewrites it to `client_model`.
/// For `data: [DONE]` lines, passes through unchanged.
/// Non-`data:` lines (comments, event lines) are passed through unchanged.
pub fn passthrough_sse_line(line: &str, client_model: Option<&str>) -> Option<String> {
    let trimmed = line.trim();

    // Pass through non-data lines as-is
    if !trimmed.starts_with("data: ") {
        return Some(line.to_string());
    }

    let data_str = trimmed.strip_prefix("data: ").unwrap_or("");

    if data_str == "[DONE]" {
        return Some(line.to_string());
    }

    let Some(model) = client_model else {
        return Some(line.to_string());
    };

    // Try to parse and rewrite model field
    let mut value: Value = match serde_json::from_str(data_str) {
        Ok(v) => v,
        Err(_) => return Some(line.to_string()),
    };

    if let Some(obj) = value.as_object_mut() {
        if obj.contains_key("model") {
            obj.insert("model".to_string(), Value::String(model.to_string()));
            return Some(sse_line(&value));
        }
    }

    Some(line.to_string())
}

/// Extract the `model` field from a JSON body (if present).
pub fn extract_model_from_json(body: &[u8]) -> Option<String> {
    let value: Value = serde_json::from_slice(body).ok()?;
    value.get("model")?.as_str().map(|s| s.to_string())
}

/// Current Unix timestamp in seconds.
pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_make_id_prefix() {
        let id = make_id("msg");
        assert!(id.starts_with("msg_"));
        assert!(id.len() > 24);
    }

    #[test]
    fn test_sse_line_format() {
        let data = json!({"type": "test"});
        let line = sse_line(&data);
        assert!(line.starts_with("data: "));
        assert!(line.ends_with("\n\n"));
    }

    #[test]
    fn test_rewrite_model_in_response() {
        let body = br#"{"model": "upstream-model", "content": "hi"}"#;
        let result = rewrite_model_in_response(body, Some("client-model"));
        let v: Value = serde_json::from_slice(&result).unwrap();
        assert_eq!(v["model"], "client-model");
    }

    #[test]
    fn test_rewrite_model_no_client_model() {
        let body = br#"{"model": "upstream-model"}"#;
        let result = rewrite_model_in_response(body, None);
        assert_eq!(result.as_ref(), body);
    }

    #[test]
    fn test_passthrough_sse_rewrite_model() {
        let line = r#"data: {"model":"upstream","choices":[]}"#;
        let result = passthrough_sse_line(line, Some("client-model")).unwrap();
        assert!(result.contains("client-model"));
        assert!(!result.contains("upstream"));
    }

    #[test]
    fn test_passthrough_sse_done() {
        let line = "data: [DONE]";
        let result = passthrough_sse_line(line, Some("model")).unwrap();
        assert_eq!(result, line);
    }

    #[test]
    fn test_passthrough_sse_no_model() {
        let line = r#"data: {"type":"some_event","delta":"text"}"#;
        let result = passthrough_sse_line(line, Some("model")).unwrap();
        assert_eq!(result, line);
    }
}
