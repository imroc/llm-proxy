//! Bidirectional conversion between OpenAI Responses API and Anthropic Messages API.
//! TODO: Full implementation. Currently stubbed for compilation.

use super::common::{make_id, now_secs};
use serde_json::{json, Value};

/// Convert a Responses API request to an Anthropic Messages request.
pub fn request_responses_to_anthropic(body: &Value) -> Result<Value, String> {
    let mut result = json!({});
    if let Some(model) = body.get("model") {
        result["model"] = model.clone();
    }
    if let Some(max_tokens) = body.get("max_output_tokens") {
        result["max_tokens"] = max_tokens.clone();
    }
    if result.get("max_tokens").is_none() {
        result["max_tokens"] = json!(4096);
    }
    if let Some(temp) = body.get("temperature") {
        result["temperature"] = temp.clone();
    }
    if let Some(top_p) = body.get("top_p") {
        result["top_p"] = top_p.clone();
    }
    if let Some(stream) = body.get("stream") {
        result["stream"] = stream.clone();
    }

    // System prompt
    if let Some(instructions) = body.get("instructions") {
        if let Some(text) = instructions.as_str() {
            result["system"] = json!(text);
        }
    }

    // Convert input array to messages
    let mut messages: Vec<Value> = Vec::new();
    if let Some(input) = body.get("input") {
        match input {
            Value::String(text) => messages.push(json!({"role": "user", "content": text})),
            Value::Array(items) => {
                for item in items {
                    let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    match item_type {
                        "message" => {
                            let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                            let content = extract_text_from_responses_content(item);
                            let anthropic_role = if role == "assistant" {
                                "assistant"
                            } else {
                                "user"
                            };
                            if !content.is_empty() {
                                messages.push(json!({"role": anthropic_role, "content": content}));
                            }
                        }
                        "function_call" => {
                            let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let args = item
                                .get("arguments")
                                .and_then(|v| v.as_str())
                                .unwrap_or("{}");
                            let id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                            messages.push(json!({
                                "role": "assistant",
                                "content": [{"type": "tool_use", "id": id, "name": name, "input": serde_json::from_str(args).unwrap_or(json!({}))}],
                            }));
                        }
                        "function_call_output" => {
                            let call_id =
                                item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                            let output = item.get("output").and_then(|v| v.as_str()).unwrap_or("");
                            messages.push(json!({
                                "role": "user",
                                "content": [{"type": "tool_result", "tool_use_id": call_id, "content": output}],
                            }));
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    result["messages"] = json!(messages);

    // Convert tools
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let anthropic_tools: Vec<Value> = tools
            .iter()
            .filter_map(|tool| {
                if tool.get("type").and_then(|v| v.as_str()) == Some("function") {
                    Some(json!({
                        "name": tool.get("name").cloned().unwrap_or(json!("")),
                        "description": tool.get("description").cloned().unwrap_or(json!("")),
                        "input_schema": tool.get("parameters").cloned().unwrap_or(json!({})),
                    }))
                } else {
                    None
                }
            })
            .collect();
        if !anthropic_tools.is_empty() {
            result["tools"] = json!(anthropic_tools);
        }
    }

    Ok(result)
}

/// Convert an Anthropic Messages request to a Responses API request.
pub fn request_anthropic_to_responses(body: &Value) -> Result<Value, String> {
    let mut result = json!({});
    if let Some(model) = body.get("model") {
        result["model"] = model.clone();
    }
    if let Some(max_tokens) = body.get("max_tokens") {
        result["max_output_tokens"] = max_tokens.clone();
    }
    if let Some(temp) = body.get("temperature") {
        result["temperature"] = temp.clone();
    }
    if let Some(top_p) = body.get("top_p") {
        result["top_p"] = top_p.clone();
    }
    if let Some(stream) = body.get("stream") {
        result["stream"] = stream.clone();
    }

    if let Some(system) = body.get("system") {
        if let Some(text) = system.as_str() {
            result["instructions"] = json!(text);
        }
    }

    let mut input_items: Vec<Value> = Vec::new();
    if let Some(messages) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in messages {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            if let Some(content) = msg.get("content") {
                match content {
                    Value::String(text) => {
                        input_items.push(json!({
                            "type": "message", "role": role,
                            "content": [{"type": if role == "assistant" { "output_text" } else { "input_text" }, "text": text}],
                        }));
                    }
                    Value::Array(parts) => {
                        let mut msg_parts: Vec<Value> = Vec::new();
                        let mut tool_uses: Vec<Value> = Vec::new();
                        for part in parts {
                            let part_type = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            match part_type {
                                "text" => {
                                    let text =
                                        part.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                    msg_parts.push(json!({
                                        "type": if role == "assistant" { "output_text" } else { "input_text" },
                                        "text": text,
                                    }));
                                }
                                "tool_use" => {
                                    let id = part.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                    let name =
                                        part.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                    let input = part.get("input").cloned().unwrap_or(json!({}));
                                    tool_uses.push(json!({
                                        "type": "function_call", "call_id": id, "name": name,
                                        "arguments": serde_json::to_string(&input).unwrap_or_default(),
                                    }));
                                }
                                "tool_result" => {
                                    let tool_use_id = part
                                        .get("tool_use_id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    let content =
                                        part.get("content").and_then(|v| v.as_str()).unwrap_or("");
                                    input_items.push(json!({
                                        "type": "function_call_output", "call_id": tool_use_id, "output": content,
                                    }));
                                }
                                _ => {}
                            }
                        }
                        if !msg_parts.is_empty() {
                            input_items.push(
                                json!({"type": "message", "role": role, "content": msg_parts}),
                            );
                        }
                        input_items.extend(tool_uses);
                    }
                    _ => {}
                }
            }
        }
    }
    result["input"] = json!(input_items);

    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let resp_tools: Vec<Value> = tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.get("name").cloned().unwrap_or(json!("")),
                    "description": tool.get("description").cloned().unwrap_or(json!("")),
                    "parameters": tool.get("input_schema").cloned().unwrap_or(json!({})),
                })
            })
            .collect();
        if !resp_tools.is_empty() {
            result["tools"] = json!(resp_tools);
        }
    }

    Ok(result)
}

// ── Non-streaming response conversions ─────────────────────────────────────

/// Convert an Anthropic Messages response to a Responses API response.
pub fn response_anthropic_to_responses(
    anthro: &Value,
    client_model: Option<&str>,
) -> Option<Value> {
    let model =
        client_model.unwrap_or_else(|| anthro.get("model").and_then(|v| v.as_str()).unwrap_or(""));

    let mut output_items: Vec<Value> = Vec::new();
    let mut all_text = String::new();

    if let Some(content) = anthro.get("content").and_then(|v| v.as_array()) {
        let mut text_parts: Vec<Value> = Vec::new();
        for block in content {
            match block.get("type").and_then(|v| v.as_str()) {
                Some("text") => {
                    let text = block.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    all_text.push_str(text);
                    text_parts
                        .push(json!({"type": "output_text", "text": text, "annotations": []}));
                }
                Some("tool_use") => {
                    let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let input = block.get("input").cloned().unwrap_or(json!({}));
                    output_items.push(json!({
                        "type": "function_call", "id": id, "call_id": id,
                        "name": name, "arguments": serde_json::to_string(&input).unwrap_or_default(),
                    }));
                }
                _ => {}
            }
        }
        if !text_parts.is_empty() {
            output_items.insert(
                0,
                json!({
                    "id": make_id("msg"), "type": "message", "role": "assistant",
                    "status": "completed", "content": text_parts,
                }),
            );
        }
    }

    let empty_usage = json!({});
    let usage = anthro.get("usage").unwrap_or(&empty_usage);
    let stop_reason = anthro
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("end_turn");
    let status = match stop_reason {
        "end_turn" => "completed",
        _ => "completed",
    };

    Some(json!({
        "id": make_id("resp"), "object": "response", "created_at": now_secs(),
        "status": status, "model": model, "output": output_items,
        "usage": {
            "input_tokens": usage.get("input_tokens").unwrap_or(&json!(0)),
            "output_tokens": usage.get("output_tokens").unwrap_or(&json!(0)),
            "total_tokens": usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0)
                + usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
        },
        "parallel_tool_calls": true, "previous_response_id": null,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "text": {"format": {"type": "text"}}, "tools": [], "truncation": "disabled",
    }))
}

/// Convert a Responses API response to an Anthropic Messages response.
pub fn response_responses_to_anthropic(resp: &Value, client_model: Option<&str>) -> Option<Value> {
    let model =
        client_model.unwrap_or_else(|| resp.get("model").and_then(|v| v.as_str()).unwrap_or(""));

    let mut content_blocks: Vec<Value> = Vec::new();
    let stop_reason;

    if let Some(output) = resp.get("output").and_then(|v| v.as_array()) {
        let mut has_tool_use = false;
        for item in output {
            match item.get("type").and_then(|v| v.as_str()) {
                Some("message") => {
                    if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
                        let text: String = parts
                            .iter()
                            .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
                            .collect::<Vec<_>>()
                            .join("");
                        if !text.is_empty() {
                            content_blocks.push(json!({"type": "text", "text": text}));
                        }
                    }
                }
                Some("function_call") => {
                    has_tool_use = true;
                    let id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let args_str = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");
                    let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                    content_blocks
                        .push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
                }
                _ => {}
            }
        }
        stop_reason = if has_tool_use { "tool_use" } else { "end_turn" };
    } else {
        stop_reason = "end_turn";
    }

    let empty_usage = json!({});
    let usage = resp.get("usage").unwrap_or(&empty_usage);

    Some(json!({
        "id": resp.get("id").cloned().unwrap_or_else(|| json!(make_id("msg"))),
        "type": "message", "role": "assistant", "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": usage.get("input_tokens").unwrap_or(&json!(0)),
            "output_tokens": usage.get("output_tokens").unwrap_or(&json!(0)),
        },
    }))
}

// ── Streaming state ───────────────────────────────────────────────────────

pub struct AnthropicToResponsesStreamState {
    pub resp_id: String,
    pub msg_id: String,
    pub model: String,
    pub created: u64,
    pub full_text: String,
    pub total_input: u64,
    pub total_output: u64,
    pub msg_closed: bool,
    pub output_index: usize,
    pub headers_emitted: bool,
    pub completed: bool,
}

impl AnthropicToResponsesStreamState {
    pub fn new(model: &str) -> Self {
        Self {
            resp_id: make_id("resp"),
            msg_id: make_id("msg"),
            model: model.to_string(),
            created: now_secs(),
            full_text: String::new(),
            total_input: 0,
            total_output: 0,
            msg_closed: false,
            output_index: 0,
            headers_emitted: false,
            completed: false,
        }
    }
}

pub struct ResponsesToAnthropicStreamState {
    pub msg_id: String,
    pub model: String,
    pub full_text: String,
    pub headers_emitted: bool,
    pub completed: bool,
}

impl ResponsesToAnthropicStreamState {
    pub fn new(model: &str) -> Self {
        Self {
            msg_id: make_id("msg"),
            model: model.to_string(),
            full_text: String::new(),
            headers_emitted: false,
            completed: false,
        }
    }
}

/// Convert Anthropic SSE to Responses SSE.
pub fn anthropic_sse_to_responses_sse(
    line: &str,
    state: &mut AnthropicToResponsesStreamState,
) -> Option<String> {
    let data_str = line.strip_prefix("data: ")?.trim();
    let chunk: Value = serde_json::from_str(data_str).ok()?;
    let event_type = chunk.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let mut output = String::new();

    if !state.headers_emitted {
        state.headers_emitted = true;
        let empty = json!({"id": state.resp_id, "object": "response", "created_at": state.created, "status": "in_progress", "model": &state.model, "output": [], "usage": null});
        output.push_str(&super::common::sse_line(
            &json!({"type": "response.created", "response": empty}),
        ));
        output.push_str(&super::common::sse_line(
            &json!({"type": "response.in_progress", "response": empty}),
        ));
        output.push_str(&super::common::sse_line(&json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"id": state.msg_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []},
        })));
        output.push_str(&super::common::sse_line(&json!({
            "type": "response.content_part.added", "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        })));
    }

    match event_type {
        "content_block_delta" => {
            if let Some(delta) = chunk.get("delta") {
                if delta.get("type").and_then(|v| v.as_str()) == Some("text_delta") {
                    let text = delta.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    if !text.is_empty() {
                        state.full_text.push_str(text);
                        output.push_str(&super::common::sse_line(&json!({
                            "type": "response.output_text.delta", "item_id": state.msg_id,
                            "output_index": 0, "content_index": 0, "delta": text,
                        })));
                    }
                }
                if delta.get("type").and_then(|v| v.as_str()) == Some("thinking_delta") {
                    let text = delta.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                    if !text.is_empty() {
                        output.push_str(&super::common::sse_line(&json!({
                            "type": "response.reasoning_text.delta", "item_id": state.msg_id,
                            "output_index": 0, "content_index": 0, "delta": text,
                        })));
                    }
                }
            }
        }
        "message_delta" => {
            if let Some(usage) = chunk.get("usage") {
                state.total_output = usage
                    .get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
            }
        }
        "message_start" => {
            if let Some(usage) = chunk.get("message").and_then(|m| m.get("usage")) {
                state.total_input = usage
                    .get("input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
            }
        }
        "message_stop" => {
            // Close message and emit completed
            if !state.msg_closed {
                state.msg_closed = true;
                output.push_str(&super::common::sse_line(&json!({
                    "type": "response.content_part.done", "output_index": 0, "content_index": 0,
                    "part": {"type": "output_text", "text": state.full_text, "annotations": []},
                })));
                output.push_str(&super::common::sse_line(&json!({
                    "type": "response.output_item.done", "output_index": 0,
                    "item": {"id": state.msg_id, "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": state.full_text, "annotations": []}]},
                })));
            }
            state.completed = true;
            output.push_str(&super::common::sse_line(&json!({
                "type": "response.completed",
                "response": {
                    "id": state.resp_id, "object": "response", "created_at": state.created,
                    "status": "completed", "model": &state.model,
                    "output": [{"id": state.msg_id, "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": state.full_text, "annotations": []}]}],
                    "usage": {"input_tokens": state.total_input, "output_tokens": state.total_output,
                        "total_tokens": state.total_input + state.total_output},
                    "parallel_tool_calls": true, "previous_response_id": null,
                    "reasoning": {"effort": "medium", "summary": "auto"},
                    "text": {"format": {"type": "text"}}, "tools": [], "truncation": "disabled",
                },
            })));
            output.push_str("data: [DONE]\n\n");
        }
        _ => {}
    }

    if output.is_empty() {
        None
    } else {
        Some(output)
    }
}

/// Convert Responses SSE to Anthropic SSE.
pub fn responses_sse_to_anthropic_sse(
    line: &str,
    state: &mut ResponsesToAnthropicStreamState,
) -> Option<String> {
    let data_str = line.strip_prefix("data: ")?.trim();
    if data_str == "[DONE]" {
        return None;
    }
    let chunk: Value = serde_json::from_str(data_str).ok()?;
    let event_type = chunk.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let mut output = String::new();

    // Emit message_start on first chunk
    if !state.headers_emitted {
        state.headers_emitted = true;
        output.push_str(&super::common::sse_line(&json!({
            "type": "message_start",
            "message": {"id": state.msg_id, "type": "message", "role": "assistant",
                "model": &state.model, "content": [], "stop_reason": null,
                "usage": {"input_tokens": 0, "output_tokens": 0}},
        })));
        output.push_str(&super::common::sse_line(&json!({
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        })));
    }

    match event_type {
        "response.output_text.delta" => {
            let delta = chunk.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            if !delta.is_empty() {
                state.full_text.push_str(delta);
                output.push_str(&super::common::sse_line(&json!({
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": delta},
                })));
            }
        }
        "response.reasoning_text.delta" => {
            let delta = chunk.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            if !delta.is_empty() {
                output.push_str(&super::common::sse_line(&json!({
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": delta},
                })));
            }
        }
        "response.completed" => {
            let response = chunk.get("response").unwrap_or(&serde_json::Value::Null);
            let empty_usage = json!({});
            let usage = response.get("usage").unwrap_or(&empty_usage);
            let input_tokens = usage
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let output_tokens = usage
                .get("output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            output.push_str(&super::common::sse_line(
                &json!({"type": "content_block_stop", "index": 0}),
            ));
            output.push_str(&super::common::sse_line(&json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": output_tokens},
            })));
            output.push_str(&super::common::sse_line(&json!({
                "type": "message_stop",
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            })));
            state.completed = true;
        }
        _ => {}
    }

    if output.is_empty() {
        None
    } else {
        Some(output)
    }
}

fn extract_text_from_responses_content(item: &Value) -> String {
    match item.get("content") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| {
                let ct = p.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match ct {
                    "input_text" | "output_text" => p.get("text").and_then(|v| v.as_str()),
                    _ => None,
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_responses_to_anthropic_simple() {
        let body = json!({"model": "claude-3", "instructions": "Be helpful", "input": "Hello", "max_output_tokens": 1024});
        let result = request_responses_to_anthropic(&body).unwrap();
        assert_eq!(result["model"], "claude-3");
        assert_eq!(result["system"], "Be helpful");
        assert_eq!(result["max_tokens"], 1024);
        let msgs = result["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn test_request_anthropic_to_responses_simple() {
        let body = json!({
            "model": "claude-3", "system": "Be helpful", "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        });
        let result = request_anthropic_to_responses(&body).unwrap();
        assert_eq!(result["model"], "claude-3");
        assert_eq!(result["instructions"], "Be helpful");
        let input = result["input"].as_array().unwrap();
        assert_eq!(input.len(), 1);
    }

    #[test]
    fn test_response_anthropic_to_responses() {
        let anthro = json!({
            "model": "claude-3",
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        });
        let result = response_anthropic_to_responses(&anthro, None).unwrap();
        assert_eq!(result["object"], "response");
        assert_eq!(result["status"], "completed");
    }

    #[test]
    fn test_response_responses_to_anthropic() {
        let resp = json!({
            "model": "claude-3",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi!"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        });
        let result = response_responses_to_anthropic(&resp, None).unwrap();
        assert_eq!(result["type"], "message");
        assert_eq!(result["role"], "assistant");
        assert_eq!(result["stop_reason"], "end_turn");
    }
}
