//! Bidirectional conversion between OpenAI Chat Completions and Anthropic Messages.

use super::common::{make_id, now_secs, sse_line};
use serde_json::{json, Value};

/// Convert a Chat Completions request to an Anthropic Messages request.
pub fn request_chat_to_anthropic(body: &Value) -> Result<Value, String> {
    let mut result = json!({});
    if let Some(model) = body.get("model") {
        result["model"] = model.clone();
    }
    if let Some(max_tokens) = body.get("max_tokens") {
        result["max_tokens"] = max_tokens.clone();
    }
    if let Some(max_tokens) = body.get("max_completion_tokens") {
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

    let mut system_text = String::new();
    let mut messages: Vec<Value> = Vec::new();

    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            match role {
                "system" => {
                    if let Some(text) = msg.get("content").and_then(|v| v.as_str()) {
                        if !system_text.is_empty() {
                            system_text.push('\n');
                        }
                        system_text.push_str(text);
                    }
                }
                "user" => {
                    let content = chat_content_to_anthropic(msg);
                    messages.push(json!({"role": "user", "content": content}));
                }
                "assistant" => {
                    let mut blocks: Vec<Value> = Vec::new();
                    if let Some(text) = msg.get("content").and_then(|v| v.as_str()) {
                        if !text.is_empty() {
                            blocks.push(json!({"type": "text", "text": text}));
                        }
                    }
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                        for tc in tool_calls {
                            let empty_fn = json!({});
                            let fn_obj = tc.get("function").unwrap_or(&empty_fn);
                            let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                            let name = fn_obj.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let args_str = fn_obj
                                .get("arguments")
                                .and_then(|v| v.as_str())
                                .unwrap_or("{}");
                            let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                            blocks.push(
                                json!({"type": "tool_use", "id": id, "name": name, "input": input}),
                            );
                        }
                    }
                    if !blocks.is_empty() {
                        messages.push(json!({"role": "assistant", "content": blocks}));
                    }
                }
                "tool" => {
                    let call_id = msg
                        .get("tool_call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    messages.push(json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": content}]}));
                }
                _ => {}
            }
        }
    }

    if !system_text.is_empty() {
        result["system"] = json!(system_text);
    }
    result["messages"] = json!(messages);

    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let anthro_tools: Vec<Value> = tools
            .iter()
            .filter_map(|t| {
                if t.get("type").and_then(|v| v.as_str()) == Some("function") {
                    let fn_obj = t.get("function").unwrap_or(&serde_json::Value::Null);
                    Some(json!({
                        "name": fn_obj.get("name").cloned().unwrap_or(json!("")),
                        "description": fn_obj.get("description").cloned().unwrap_or(json!("")),
                        "input_schema": fn_obj.get("parameters").cloned().unwrap_or(json!({})),
                    }))
                } else {
                    None
                }
            })
            .collect();
        if !anthro_tools.is_empty() {
            result["tools"] = json!(anthro_tools);
        }
    }

    Ok(result)
}

/// Convert an Anthropic Messages request to a Chat Completions request.
pub fn request_anthropic_to_chat(body: &Value) -> Result<Value, String> {
    let mut result = json!({});
    if let Some(model) = body.get("model") {
        result["model"] = model.clone();
    }
    if let Some(max_tokens) = body.get("max_tokens") {
        result["max_tokens"] = max_tokens.clone();
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

    let mut messages: Vec<Value> = Vec::new();

    if let Some(system) = body.get("system") {
        if let Some(text) = system.as_str() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }

    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            if let Some(content) = msg.get("content") {
                match content {
                    Value::String(text) => {
                        messages.push(json!({"role": role, "content": text}));
                    }
                    Value::Array(blocks) => {
                        let mut text_parts = String::new();
                        let mut tool_uses: Vec<Value> = Vec::new();
                        let mut tool_results: Vec<Value> = Vec::new();
                        for block in blocks {
                            match block.get("type").and_then(|v| v.as_str()) {
                                Some("text") => {
                                    let text =
                                        block.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                    text_parts.push_str(text);
                                }
                                Some("tool_use") => {
                                    let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                    let name =
                                        block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                    let input = block.get("input").cloned().unwrap_or(json!({}));
                                    tool_uses.push(json!({
                                        "id": id, "type": "function",
                                        "function": {"name": name, "arguments": serde_json::to_string(&input).unwrap_or_default()},
                                    }));
                                }
                                Some("tool_result") => {
                                    let tool_use_id = block
                                        .get("tool_use_id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    let content =
                                        block.get("content").and_then(|v| v.as_str()).unwrap_or("");
                                    tool_results.push(json!({"role": "tool", "tool_call_id": tool_use_id, "content": content}));
                                }
                                _ => {}
                            }
                        }
                        if !text_parts.is_empty() || tool_uses.is_empty() {
                            messages.push(json!({"role": role, "content": text_parts}));
                        }
                        if !tool_uses.is_empty() {
                            messages.push(json!({"role": "assistant", "tool_calls": tool_uses}));
                        }
                        messages.extend(tool_results);
                    }
                    _ => {}
                }
            }
        }
    }
    result["messages"] = json!(messages);

    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let chat_tools: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.get("name").cloned().unwrap_or(json!("")),
                        "description": t.get("description").cloned().unwrap_or(json!("")),
                        "parameters": t.get("input_schema").cloned().unwrap_or(json!({})),
                    },
                })
            })
            .collect();
        if !chat_tools.is_empty() {
            result["tools"] = json!(chat_tools);
        }
    }

    Ok(result)
}

// ── Non-streaming response conversions ─────────────────────────────────────

/// Convert an Anthropic response to a Chat Completions response.
pub fn response_anthropic_to_chat(anthro: &Value, client_model: Option<&str>) -> Option<Value> {
    let model =
        client_model.unwrap_or_else(|| anthro.get("model").and_then(|v| v.as_str()).unwrap_or(""));
    let mut content = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    if let Some(blocks) = anthro.get("content").and_then(|v| v.as_array()) {
        for block in blocks {
            match block.get("type").and_then(|v| v.as_str()) {
                Some("text") => {
                    content.push_str(block.get("text").and_then(|v| v.as_str()).unwrap_or(""));
                }
                Some("tool_use") => {
                    let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let input = block.get("input").cloned().unwrap_or(json!({}));
                    tool_calls.push(json!({"id": id, "type": "function", "function": {"name": name, "arguments": serde_json::to_string(&input).unwrap_or_default()}}));
                }
                _ => {}
            }
        }
    }

    let finish_reason = if !tool_calls.is_empty() {
        "tool_calls"
    } else {
        "stop"
    };
    let mut message = json!({"role": "assistant", "content": content});
    if !tool_calls.is_empty() {
        message["tool_calls"] = json!(tool_calls);
    }

    let empty_usage = json!({});
    let usage = anthro.get("usage").unwrap_or(&empty_usage);

    Some(json!({
        "id": make_id("chatcmpl"), "object": "chat.completion", "created": now_secs(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens").unwrap_or(&json!(0)),
            "completion_tokens": usage.get("output_tokens").unwrap_or(&json!(0)),
            "total_tokens": usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0)
                + usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
        },
    }))
}

/// Convert a Chat Completions response to an Anthropic response.
pub fn response_chat_to_anthropic(chat: &Value, client_model: Option<&str>) -> Option<Value> {
    let model =
        client_model.unwrap_or_else(|| chat.get("model").and_then(|v| v.as_str()).unwrap_or(""));
    let choices = chat.get("choices").and_then(|v| v.as_array())?;
    let message = choices.first()?.get("message")?;

    let mut content_blocks: Vec<Value> = Vec::new();
    if let Some(text) = message.get("content").and_then(|v| v.as_str()) {
        if !text.is_empty() {
            content_blocks.push(json!({"type": "text", "text": text}));
        }
    }
    let mut has_tool_use = false;
    if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tool_calls {
            has_tool_use = true;
            let empty_fn = json!({});
            let fn_obj = tc.get("function").unwrap_or(&empty_fn);
            let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let name = fn_obj.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args_str = fn_obj
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
            content_blocks
                .push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
        }
    }

    let stop_reason = if has_tool_use { "tool_use" } else { "end_turn" };
    let empty_usage = json!({});
    let usage = chat.get("usage").unwrap_or(&empty_usage);

    Some(json!({
        "id": make_id("msg"), "type": "message", "role": "assistant", "model": model,
        "content": content_blocks, "stop_reason": stop_reason,
        "usage": {
            "input_tokens": usage.get("prompt_tokens").unwrap_or(&json!(0)),
            "output_tokens": usage.get("completion_tokens").unwrap_or(&json!(0)),
        },
    }))
}

// ── Streaming ──────────────────────────────────────────────────────────────

pub struct AnthropicToChatStreamState {
    pub chat_id: String,
    pub model: String,
    pub created: u64,
    pub full_text: String,
    pub total_input: u64,
    pub total_output: u64,
    pub headers_emitted: bool,
    pub completed: bool,
}

impl AnthropicToChatStreamState {
    pub fn new(model: &str) -> Self {
        Self {
            chat_id: make_id("chatcmpl"),
            model: model.to_string(),
            created: now_secs(),
            full_text: String::new(),
            total_input: 0,
            total_output: 0,
            headers_emitted: false,
            completed: false,
        }
    }
}

pub struct ChatToAnthropicStreamState {
    pub msg_id: String,
    pub model: String,
    pub full_text: String,
    pub headers_emitted: bool,
    pub content_block_started: bool,
    pub completed: bool,
}

impl ChatToAnthropicStreamState {
    pub fn new(model: &str) -> Self {
        Self {
            msg_id: make_id("msg"),
            model: model.to_string(),
            full_text: String::new(),
            headers_emitted: false,
            content_block_started: false,
            completed: false,
        }
    }
}

/// Convert Anthropic SSE to Chat Completions SSE.
pub fn anthropic_sse_to_chat_sse(
    line: &str,
    state: &mut AnthropicToChatStreamState,
) -> Option<String> {
    let data_str = line.strip_prefix("data: ")?.trim();
    let chunk: Value = serde_json::from_str(data_str).ok()?;
    let event_type = chunk.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let mut output = String::new();

    if !state.headers_emitted {
        state.headers_emitted = true;
        output.push_str(&sse_line(&json!({
            "id": state.chat_id, "object": "chat.completion.chunk", "created": state.created, "model": &state.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
        })));
    }

    match event_type {
        "content_block_delta" => {
            if let Some(delta) = chunk.get("delta") {
                if delta.get("type").and_then(|v| v.as_str()) == Some("text_delta") {
                    let text = delta.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    if !text.is_empty() {
                        state.full_text.push_str(text);
                        output.push_str(&sse_line(&json!({
                            "id": state.chat_id, "object": "chat.completion.chunk", "created": state.created, "model": &state.model,
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}],
                        })));
                    }
                }
                if delta.get("type").and_then(|v| v.as_str()) == Some("thinking_delta") {
                    let text = delta.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                    if !text.is_empty() {
                        output.push_str(&sse_line(&json!({
                            "id": state.chat_id, "object": "chat.completion.chunk", "created": state.created, "model": &state.model,
                            "choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": null}],
                        })));
                    }
                }
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
        "message_delta" => {
            if let Some(usage) = chunk.get("usage") {
                state.total_output = usage
                    .get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
            }
            let stop_reason = chunk
                .get("delta")
                .and_then(|d| d.get("stop_reason"))
                .and_then(|v| v.as_str());
            if let Some(sr) = stop_reason {
                let finish_reason = match sr {
                    "tool_use" => "tool_calls",
                    _ => "stop",
                };
                output.push_str(&sse_line(&json!({
                    "id": state.chat_id, "object": "chat.completion.chunk", "created": state.created, "model": &state.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": state.total_input, "completion_tokens": state.total_output,
                        "total_tokens": state.total_input + state.total_output},
                })));
                state.completed = true;
                output.push_str("data: [DONE]\n\n");
            }
        }
        "message_stop" if !state.completed => {
            output.push_str(&sse_line(&json!({
                    "id": state.chat_id, "object": "chat.completion.chunk", "created": state.created, "model": &state.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": state.total_input, "completion_tokens": state.total_output,
                        "total_tokens": state.total_input + state.total_output},
                })));
            state.completed = true;
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

/// Flush an Anthropic→Chat stream that ended without `message_stop`.
/// Emits a final chat chunk with `finish_reason` + `[DONE]` so the client
/// receives a properly terminated stream.
pub fn flush_anthropic_to_chat(state: &mut AnthropicToChatStreamState) -> Option<String> {
    if state.completed {
        return None;
    }
    state.completed = true;

    let mut output = String::new();
    if !state.headers_emitted {
        state.headers_emitted = true;
        output.push_str(&sse_line(&json!({
            "id": state.chat_id, "object": "chat.completion.chunk", "created": state.created, "model": &state.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
        })));
    }
    output.push_str(&sse_line(&json!({
        "id": state.chat_id, "object": "chat.completion.chunk", "created": state.created, "model": &state.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": state.total_input, "completion_tokens": state.total_output,
            "total_tokens": state.total_input + state.total_output},
    })));
    output.push_str("data: [DONE]\n\n");
    Some(output)
}

/// Flush a Chat→Anthropic stream that ended without `[DONE]`.
/// Emits `content_block_stop` + `message_delta` + `message_stop` so the client
/// receives a properly terminated Anthropic stream.
pub fn flush_chat_to_anthropic(state: &mut ChatToAnthropicStreamState) -> Option<String> {
    if state.completed {
        return None;
    }
    state.completed = true;

    let mut output = String::new();
    if !state.headers_emitted {
        state.headers_emitted = true;
        output.push_str(&sse_line(&json!({
            "type": "message_start",
            "message": {"id": state.msg_id, "type": "message", "role": "assistant",
                "model": &state.model, "content": [], "stop_reason": null,
                "usage": {"input_tokens": 0, "output_tokens": 0}},
        })));
    }
    if state.content_block_started {
        output.push_str(&sse_line(
            &json!({"type": "content_block_stop", "index": 0}),
        ));
    }
    output.push_str(&sse_line(&json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 0}})));
    output.push_str(&sse_line(
        &json!({"type": "message_stop", "usage": {"input_tokens": 0, "output_tokens": 0}}),
    ));
    Some(output)
}

/// Convert Chat Completions SSE to Anthropic SSE.
pub fn chat_sse_to_anthropic_sse(
    line: &str,
    state: &mut ChatToAnthropicStreamState,
) -> Option<String> {
    let data_str = line.strip_prefix("data: ")?.trim();
    if data_str == "[DONE]" {
        if !state.completed {
            state.completed = true;
            let mut out = String::new();
            if state.content_block_started {
                out.push_str(&sse_line(
                    &json!({"type": "content_block_stop", "index": 0}),
                ));
            }
            out.push_str(&sse_line(&json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 0}})));
            out.push_str(&sse_line(
                &json!({"type": "message_stop", "usage": {"input_tokens": 0, "output_tokens": 0}}),
            ));
            return Some(out);
        }
        return None;
    }

    let chunk: Value = serde_json::from_str(data_str).ok()?;
    let mut output = String::new();

    if !state.headers_emitted {
        state.headers_emitted = true;
        output.push_str(&sse_line(&json!({
            "type": "message_start",
            "message": {"id": state.msg_id, "type": "message", "role": "assistant",
                "model": &state.model, "content": [], "stop_reason": null,
                "usage": {"input_tokens": 0, "output_tokens": 0}},
        })));
    }

    let choices = chunk.get("choices").and_then(|v| v.as_array());
    if let Some(choices) = choices {
        if !choices.is_empty() {
            let empty_obj = json!({});
            let delta = choices[0].get("delta").unwrap_or(&empty_obj);

            if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    if !state.content_block_started {
                        state.content_block_started = true;
                        output.push_str(&sse_line(&json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})));
                    }
                    state.full_text.push_str(text);
                    output.push_str(&sse_line(&json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})));
                }
            }

            if let Some(reasoning) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                if !reasoning.is_empty() {
                    output.push_str(&sse_line(&json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": reasoning}})));
                }
            }

            if let Some(finish_reason) = choices[0].get("finish_reason").and_then(|v| v.as_str()) {
                if state.content_block_started {
                    output.push_str(&sse_line(
                        &json!({"type": "content_block_stop", "index": 0}),
                    ));
                }
                let stop_reason = match finish_reason {
                    "tool_calls" => "tool_use",
                    _ => "end_turn",
                };
                output.push_str(&sse_line(&json!({"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 0}})));
                output.push_str(&sse_line(&json!({"type": "message_stop", "usage": {"input_tokens": 0, "output_tokens": 0}})));
                state.completed = true;
            }
        }
    }

    if let Some(_usage) = chunk.get("usage") {
        // Update usage in the message_start (best effort)
    }

    if output.is_empty() {
        None
    } else {
        Some(output)
    }
}

fn chat_content_to_anthropic(msg: &Value) -> Value {
    match msg.get("content") {
        Some(Value::String(s)) => json!([{"type": "text", "text": s}]),
        Some(Value::Array(arr)) => json!(arr),
        _ => json!([{"type": "text", "text": ""}]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_chat_to_anthropic_simple() {
        let body = json!({
            "model": "claude-3", "max_tokens": 1024,
            "messages": [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Hello"}],
        });
        let result = request_chat_to_anthropic(&body).unwrap();
        assert_eq!(result["model"], "claude-3");
        assert_eq!(result["system"], "Be helpful");
        let msgs = result["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn test_request_anthropic_to_chat_simple() {
        let body = json!({
            "model": "claude-3", "system": "Be helpful", "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        });
        let result = request_anthropic_to_chat(&body).unwrap();
        assert_eq!(result["model"], "claude-3");
        let msgs = result["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn test_response_anthropic_to_chat() {
        let anthro = json!({
            "model": "claude-3",
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        });
        let result = response_anthropic_to_chat(&anthro, None).unwrap();
        assert_eq!(result["object"], "chat.completion");
        let choices = result["choices"].as_array().unwrap();
        assert_eq!(choices[0]["message"]["content"], "Hello!");
    }

    #[test]
    fn test_response_chat_to_anthropic() {
        let chat = json!({
            "model": "claude-3",
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        });
        let result = response_chat_to_anthropic(&chat, None).unwrap();
        assert_eq!(result["type"], "message");
        assert_eq!(result["role"], "assistant");
        assert_eq!(result["stop_reason"], "end_turn");
    }
}
