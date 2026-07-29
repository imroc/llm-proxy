//! Bidirectional conversion between OpenAI Responses API and Chat Completions API.
//!
//! Request direction:
//! - responses → chat: `input` array → `messages` array, `instructions` → system message
//! - chat → responses: `messages` array → `input` array, system message → `instructions`
//!
//! Response direction (non-streaming):
//! - chat → responses: `choices[0].message` → `output` items with `response` envelope
//! - responses → chat: `output` items → `choices[0].message` with chat envelope
//!
//! Response direction (streaming):
//! - chat SSE → responses SSE: `choices[0].delta` → `response.output_text.delta` etc.
//! - responses SSE → chat SSE: `response.output_text.delta` → `choices[0].delta` etc.

use super::common::{make_id, now_secs, sse_line};
use serde_json::{json, Value};

// ── Request: Responses → Chat ──────────────────────────────────────────────

/// Convert a Responses API request body to a Chat Completions request body.
pub fn request_responses_to_chat(body: &Value) -> Result<Value, String> {
    let mut result = json!({});

    if let Some(model) = body.get("model") {
        result["model"] = model.clone();
    }

    let mut messages = Vec::new();

    // instructions → system message
    if let Some(instructions) = body.get("instructions") {
        let text = instruction_text(instructions);
        if !text.is_empty() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }

    // input → messages
    if let Some(input) = body.get("input") {
        append_responses_input_as_chat_messages(input, &mut messages);
    }

    // Collapse consecutive system messages into one
    let messages = collapse_system_messages_to_head(messages);
    result["messages"] = json!(messages);

    // max_output_tokens → max_tokens (or max_completion_tokens for o-series)
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("");
    if let Some(max_tokens) = body.get("max_output_tokens") {
        if is_openai_o_series(model) {
            result["max_completion_tokens"] = max_tokens.clone();
        } else {
            result["max_tokens"] = max_tokens.clone();
        }
    }
    if let Some(max_tokens) = body.get("max_tokens") {
        result["max_tokens"] = max_tokens.clone();
    }
    if let Some(max_tokens) = body.get("max_completion_tokens") {
        result["max_completion_tokens"] = max_tokens.clone();
    }

    // Pass through common fields
    for key in ["temperature", "top_p", "stream"] {
        if let Some(value) = body.get(key) {
            result[key] = value.clone();
        }
    }

    // reasoning → reasoning_effort
    if let Some(reasoning) = body.get("reasoning") {
        if let Some(effort) = reasoning.get("effort").and_then(|v| v.as_str()) {
            result["reasoning_effort"] = json!(effort);
        }
    }

    // tools: Responses format → Chat format
    let chat_tools = responses_tools_to_chat_tools(body);
    if !chat_tools.is_empty() {
        result["tools"] = json!(chat_tools);
        if let Some(tool_choice) = body.get("tool_choice") {
            result["tool_choice"] = responses_tool_choice_to_chat(tool_choice);
        }
    }

    // Extra passthrough fields
    for key in &[
        "metadata",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "seed",
        "user",
    ] {
        if let Some(value) = body.get(*key) {
            result[*key] = value.clone();
        }
    }

    // Inject stream_options.include_usage for streaming requests
    let is_stream = result
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if is_stream {
        match result.get_mut("stream_options") {
            Some(Value::Object(opts)) => {
                opts.insert("include_usage".to_string(), json!(true));
            }
            None => {
                result["stream_options"] = json!({"include_usage": true});
            }
            _ => {}
        }
    }

    Ok(result)
}

/// Convert a Chat Completions request body to a Responses API request body.
pub fn request_chat_to_responses(body: &Value) -> Result<Value, String> {
    let mut result = json!({});

    if let Some(model) = body.get("model") {
        result["model"] = model.clone();
    }

    let mut instructions = String::new();
    let mut input_items: Vec<Value> = Vec::new();

    if let Some(messages) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in messages {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            match role {
                "system" => {
                    if !instructions.is_empty() {
                        instructions.push('\n');
                    }
                    instructions
                        .push_str(msg.get("content").and_then(|v| v.as_str()).unwrap_or(""));
                }
                "user" => {
                    input_items.push(json!({
                        "type": "message",
                        "role": "user",
                        "content": msg_content_to_responses_content(msg),
                    }));
                }
                "assistant" => {
                    let mut parts: Vec<Value> = Vec::new();
                    if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
                        parts.push(json!({"type": "output_text", "text": content}));
                    }
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                        for tc in tool_calls {
                            let empty_fn = json!({});
                            let fn_obj = tc.get("function").unwrap_or(&empty_fn);
                            input_items.push(json!({
                                "type": "function_call",
                                "id": tc.get("id").cloned().unwrap_or(json!("")),
                                "call_id": tc.get("id").cloned().unwrap_or(json!("")),
                                "name": fn_obj.get("name").cloned().unwrap_or(json!("")),
                                "arguments": fn_obj.get("arguments").cloned().unwrap_or(json!("{}")),
                            }));
                        }
                    }
                    if !parts.is_empty() {
                        input_items.push(json!({
                            "type": "message",
                            "role": "assistant",
                            "content": parts,
                        }));
                    }
                }
                "tool" => {
                    input_items.push(json!({
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id").cloned().unwrap_or(json!("")),
                        "output": msg.get("content").cloned().unwrap_or(json!("")),
                    }));
                }
                _ => {}
            }
        }
    }

    if !instructions.is_empty() {
        result["instructions"] = json!(instructions);
    }
    result["input"] = json!(input_items);

    // max_tokens
    if let Some(max_tokens) = body.get("max_tokens") {
        result["max_output_tokens"] = max_tokens.clone();
    }
    if let Some(max_tokens) = body.get("max_completion_tokens") {
        result["max_output_tokens"] = max_tokens.clone();
    }

    for key in ["temperature", "top_p", "stream"] {
        if let Some(value) = body.get(key) {
            result[key] = value.clone();
        }
    }

    if let Some(effort) = body.get("reasoning_effort").and_then(|v| v.as_str()) {
        result["reasoning"] = json!({"effort": effort, "summary": "auto"});
    }

    // tools
    let resp_tools = chat_tools_to_responses_tools(body);
    if !resp_tools.is_empty() {
        result["tools"] = json!(resp_tools);
    }

    for key in &["metadata", "stop", "seed", "user"] {
        if let Some(value) = body.get(*key) {
            result[*key] = value.clone();
        }
    }

    Ok(result)
}

// ── Response (non-streaming) ───────────────────────────────────────────────

/// Convert a Chat Completions response to a Responses API response.
pub fn response_chat_to_responses(chat: &Value, client_model: Option<&str>) -> Option<Value> {
    let model =
        client_model.unwrap_or_else(|| chat.get("model").and_then(|v| v.as_str()).unwrap_or(""));

    let choices = chat.get("choices").and_then(|v| v.as_array())?;
    let message = choices.first()?.get("message")?;
    let content_text = message
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let mut content_parts: Vec<Value> = vec![json!({
        "type": "output_text",
        "text": content_text,
        "annotations": [],
    })];

    // tool_calls → function_call content parts
    if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tool_calls {
            let empty_fn = json!({});
            let fn_obj = tc.get("function").unwrap_or(&empty_fn);
            let tc_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
            content_parts.push(json!({
                "type": "function_call",
                "id": tc_id,
                "call_id": tc_id,
                "name": fn_obj.get("name").unwrap_or(&json!("")),
                "arguments": fn_obj.get("arguments").unwrap_or(&json!("{}")),
            }));
        }
    }

    let output_item = json!({
        "id": make_id("msg"),
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": content_parts,
    });

    let empty_usage = json!({});
    let usage = chat.get("usage").unwrap_or(&empty_usage);

    Some(json!({
        "id": make_id("resp"),
        "object": "response",
        "created_at": now_secs(),
        "status": "completed",
        "model": model,
        "output": [output_item],
        "usage": {
            "input_tokens": usage.get("prompt_tokens").unwrap_or(&json!(0)),
            "output_tokens": usage.get("completion_tokens").unwrap_or(&json!(0)),
            "total_tokens": usage.get("total_tokens").unwrap_or(&json!(0)),
        },
        "parallel_tool_calls": true,
        "previous_response_id": null,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "text": {"format": {"type": "text"}},
        "tools": [],
        "truncation": "disabled",
    }))
}

/// Convert a Responses API response to a Chat Completions response.
pub fn response_responses_to_chat(resp: &Value, client_model: Option<&str>) -> Option<Value> {
    let model =
        client_model.unwrap_or_else(|| resp.get("model").and_then(|v| v.as_str()).unwrap_or(""));

    let output = resp.get("output").and_then(|v| v.as_array())?;
    let mut content = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    for item in output {
        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match item_type {
            "message" => {
                if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
                    for part in parts {
                        if part.get("type").and_then(|v| v.as_str()) == Some("output_text") {
                            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                content.push_str(text);
                            }
                        }
                    }
                }
            }
            "function_call" => {
                let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");
                tool_calls.push(json!({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }));
            }
            _ => {}
        }
    }

    let mut message = json!({"role": "assistant", "content": content});
    if !tool_calls.is_empty() {
        message["tool_calls"] = json!(tool_calls);
    }

    let finish_reason = if !tool_calls.is_empty() {
        "tool_calls"
    } else {
        "stop"
    };

    let empty_usage = json!({});
    let usage = resp.get("usage").unwrap_or(&empty_usage);

    Some(json!({
        "id": resp.get("id").cloned().unwrap_or_else(|| json!(make_id("chatcmpl"))),
        "object": "chat.completion",
        "created": resp.get("created_at").cloned().unwrap_or_else(|| json!(now_secs())),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens").unwrap_or(&json!(0)),
            "completion_tokens": usage.get("output_tokens").unwrap_or(&json!(0)),
            "total_tokens": usage.get("total_tokens").unwrap_or(&json!(0)),
        },
    }))
}

// ── Streaming: Chat SSE → Responses SSE ────────────────────────────────────

/// State for converting Chat Completions SSE stream to Responses API SSE stream.
pub struct ChatToResponsesStreamState {
    pub msg_id: String,
    pub resp_id: String,
    pub model: String,
    pub created: u64,
    pub full_text: String,
    pub total_input: u64,
    pub total_output: u64,
    pub msg_closed: bool,
    pub output_index: usize,
    pub active_tool_calls: Vec<ToolCallState>,
    pub completed_tool_calls: Vec<ToolCallState>,
    pub headers_emitted: bool,
    pub completed: bool,
}

#[derive(Debug, Clone)]
pub struct ToolCallState {
    pub id: String,
    pub index: usize,
    pub name: String,
    pub arguments: String,
    pub output_index: usize,
}

impl ChatToResponsesStreamState {
    pub fn new(model: &str) -> Self {
        Self {
            msg_id: make_id("msg"),
            resp_id: make_id("resp"),
            model: model.to_string(),
            created: now_secs(),
            full_text: String::new(),
            total_input: 0,
            total_output: 0,
            msg_closed: false,
            output_index: 0,
            active_tool_calls: Vec::new(),
            completed_tool_calls: Vec::new(),
            headers_emitted: false,
            completed: false,
        }
    }
}

/// Convert one SSE line from Chat Completions format to Responses API format.
pub fn chat_sse_to_responses_sse(
    line: &str,
    state: &mut ChatToResponsesStreamState,
) -> Option<String> {
    let data_str = line.strip_prefix("data: ")?.trim();

    if data_str == "[DONE]" {
        return flush_stream_completion(state);
    }

    let chunk: Value = serde_json::from_str(data_str).ok()?;
    transform_sse_chunk(&chunk, state)
}

fn transform_sse_chunk(chunk: &Value, state: &mut ChatToResponsesStreamState) -> Option<String> {
    let mut output = String::new();

    if !state.headers_emitted {
        state.headers_emitted = true;
        output.push_str(&emit_stream_headers(state));
    }

    let choices = chunk.get("choices").and_then(|v| v.as_array());

    if choices.is_none() || choices.map(|a| a.is_empty()) == Some(true) {
        if let Some(u) = chunk.get("usage") {
            state.total_input = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
            state.total_output = u
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
        }
        return if output.is_empty() {
            None
        } else {
            Some(output)
        };
    }

    let choices = choices.unwrap();
    let empty_obj = json!({});
    let delta = choices[0].get("delta").unwrap_or(&empty_obj);
    let finish_reason = choices[0].get("finish_reason");

    // reasoning content
    if let Some(r) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
        if !r.is_empty() {
            output.push_str(&sse_line(&json!({
                "type": "response.reasoning_text.delta",
                "item_id": state.msg_id,
                "output_index": 0,
                "content_index": 0,
                "delta": r,
            })));
        }
    }

    // text content
    if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
        if !text.is_empty() {
            state.full_text.push_str(text);
            output.push_str(&sse_line(&json!({
                "type": "response.output_text.delta",
                "item_id": state.msg_id,
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            })));
        }
    }

    // tool calls
    if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tool_calls {
            let tc_index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let tc_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let empty_fn = json!({});
            let fn_obj = tc.get("function").unwrap_or(&empty_fn);

            let existing = state.active_tool_calls.iter().find(|t| t.index == tc_index);

            if existing.is_none() {
                if !state.msg_closed {
                    output.push_str(&close_msg_item(state));
                }

                let new_tc = ToolCallState {
                    id: tc_id.to_string(),
                    index: tc_index,
                    name: fn_obj
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    arguments: fn_obj
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    output_index: state.output_index + tc_index,
                };

                output.push_str(&sse_line(&json!({
                    "type": "response.output_item.added",
                    "output_index": new_tc.output_index,
                    "item": {
                        "id": new_tc.id,
                        "type": "function_call",
                        "call_id": new_tc.id,
                        "name": new_tc.name,
                        "arguments": "",
                        "status": "in_progress",
                    },
                })));

                state.active_tool_calls.push(new_tc);
            } else if let Some(existing) = state
                .active_tool_calls
                .iter_mut()
                .find(|t| t.index == tc_index)
            {
                let args_delta = fn_obj
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                existing.arguments.push_str(args_delta);
                output.push_str(&sse_line(&json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": existing.id,
                    "output_index": existing.output_index,
                    "delta": args_delta,
                })));
            }
        }
    }

    if finish_reason.and_then(|v| v.as_str()) == Some("tool_calls") {
        if !state.msg_closed {
            output.push_str(&close_msg_item(state));
        }
        for tc in &state.active_tool_calls {
            output.push_str(&sse_line(&json!({
                "type": "response.function_call_arguments.done",
                "item_id": tc.id,
                "output_index": tc.output_index,
                "arguments": tc.arguments,
            })));
            output.push_str(&sse_line(&json!({
                "type": "response.output_item.done",
                "output_index": tc.output_index,
                "item": {
                    "id": tc.id,
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "status": "completed",
                },
            })));
        }
        state.completed_tool_calls = state.active_tool_calls.clone();
        state.active_tool_calls.clear();
    }

    if let Some(u) = chunk.get("usage") {
        state.total_input = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        state.total_output = u
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
    }

    if output.is_empty() {
        None
    } else {
        Some(output)
    }
}

fn emit_stream_headers(state: &ChatToResponsesStreamState) -> String {
    let empty = json!({
        "id": state.resp_id,
        "object": "response",
        "created_at": state.created,
        "status": "in_progress",
        "model": &state.model,
        "output": [],
        "usage": null,
    });
    let mut out = String::new();
    out.push_str(&sse_line(
        &json!({"type": "response.created", "response": empty}),
    ));
    out.push_str(&sse_line(
        &json!({"type": "response.in_progress", "response": empty}),
    ));
    out.push_str(&sse_line(&json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"id": state.msg_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []},
    })));
    out.push_str(&sse_line(&json!({
        "type": "response.content_part.added",
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    })));
    out
}

fn close_msg_item(state: &mut ChatToResponsesStreamState) -> String {
    if state.msg_closed {
        return String::new();
    }
    state.msg_closed = true;
    let mut out = String::new();
    out.push_str(&sse_line(&json!({
        "type": "response.content_part.done",
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": state.full_text, "annotations": []},
    })));
    out.push_str(&sse_line(&json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "id": state.msg_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": state.full_text, "annotations": []}],
        },
    })));
    state.output_index = 1;
    out
}

fn flush_stream_completion(state: &mut ChatToResponsesStreamState) -> Option<String> {
    if state.completed {
        return None;
    }
    state.completed = true;
    let mut out = String::new();

    if !state.msg_closed {
        out.push_str(&close_msg_item(state));
    }

    for tc in &state.active_tool_calls {
        if !state.completed_tool_calls.iter().any(|c| c.id == tc.id) {
            out.push_str(&sse_line(&json!({
                "type": "response.function_call_arguments.done",
                "item_id": tc.id,
                "output_index": tc.output_index,
                "arguments": tc.arguments,
            })));
            out.push_str(&sse_line(&json!({
                "type": "response.output_item.done",
                "output_index": tc.output_index,
                "item": {
                    "id": tc.id,
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "status": "completed",
                },
            })));
        }
    }

    let all_completed: Vec<&ToolCallState> = state
        .completed_tool_calls
        .iter()
        .chain(
            state
                .active_tool_calls
                .iter()
                .filter(|a| !state.completed_tool_calls.iter().any(|c| c.id == a.id)),
        )
        .collect();

    let mut output_items: Vec<Value> = Vec::new();
    if !state.full_text.is_empty() {
        output_items.push(json!({
            "id": state.msg_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": state.full_text, "annotations": []}],
        }));
    }
    for tc in &all_completed {
        output_items.push(json!({
            "id": tc.id,
            "type": "function_call",
            "call_id": tc.id,
            "name": tc.name,
            "arguments": tc.arguments,
            "status": "completed",
        }));
    }

    out.push_str(&sse_line(&json!({
        "type": "response.completed",
        "response": {
            "id": state.resp_id,
            "object": "response",
            "created_at": state.created,
            "status": "completed",
            "model": &state.model,
            "output": output_items,
            "usage": {
                "input_tokens": state.total_input,
                "output_tokens": state.total_output,
                "total_tokens": state.total_input + state.total_output,
            },
            "parallel_tool_calls": true,
            "previous_response_id": null,
            "reasoning": {"effort": "medium", "summary": "auto"},
            "text": {"format": {"type": "text"}},
            "tools": [],
            "truncation": "disabled",
        },
    })));
    out.push_str("data: [DONE]\n\n");
    Some(out)
}

// ── Streaming: Responses SSE → Chat SSE ────────────────────────────────────

/// State for converting Responses API SSE stream to Chat Completions SSE stream.
pub struct ResponsesToChatStreamState {
    pub chat_id: String,
    pub model: String,
    pub created: u64,
    pub full_text: String,
    pub total_input: u64,
    pub total_output: u64,
    pub headers_emitted: bool,
    pub completed: bool,
    pub active_tool_calls: Vec<ResponsesToolCallState>,
}

#[derive(Debug, Clone)]
pub struct ResponsesToolCallState {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

impl ResponsesToChatStreamState {
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
            active_tool_calls: Vec::new(),
        }
    }
}

/// Convert one SSE line from Responses API format to Chat Completions format.
pub fn responses_sse_to_chat_sse(
    line: &str,
    state: &mut ResponsesToChatStreamState,
) -> Option<String> {
    let data_str = line.strip_prefix("data: ")?.trim();

    if data_str == "[DONE]" {
        return None; // Already handled by flush
    }

    let chunk: Value = serde_json::from_str(data_str).ok()?;
    let event_type = chunk.get("type").and_then(|v| v.as_str()).unwrap_or("");

    let mut output = String::new();

    // Emit initial chat chunk with role
    if !state.headers_emitted {
        state.headers_emitted = true;
        output.push_str(&sse_line(&json!({
            "id": state.chat_id,
            "object": "chat.completion.chunk",
            "created": state.created,
            "model": &state.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
        })));
    }

    match event_type {
        "response.output_text.delta" => {
            let delta = chunk.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            if !delta.is_empty() {
                state.full_text.push_str(delta);
                output.push_str(&sse_line(&json!({
                    "id": state.chat_id,
                    "object": "chat.completion.chunk",
                    "created": state.created,
                    "model": &state.model,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": null}],
                })));
            }
        }
        "response.reasoning_text.delta" => {
            let delta = chunk.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            if !delta.is_empty() {
                output.push_str(&sse_line(&json!({
                    "id": state.chat_id,
                    "object": "chat.completion.chunk",
                    "created": state.created,
                    "model": &state.model,
                    "choices": [{"index": 0, "delta": {"reasoning_content": delta}, "finish_reason": null}],
                })));
            }
        }
        "response.output_item.added" => {
            let empty_item = json!({});
            let item = chunk.get("item").unwrap_or(&empty_item);
            if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                let id = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let call_id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                state.active_tool_calls.push(ResponsesToolCallState {
                    id: id.clone(),
                    call_id: call_id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                });
                let tc_index = state.active_tool_calls.len() - 1;
                output.push_str(&sse_line(&json!({
                    "id": state.chat_id,
                    "object": "chat.completion.chunk",
                    "created": state.created,
                    "model": &state.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": tc_index,
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }],
                        },
                        "finish_reason": null,
                    }],
                })));
            }
        }
        "response.function_call_arguments.delta" => {
            let item_id = chunk.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
            let delta = chunk.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            if let Some(pos) = state.active_tool_calls.iter().position(|t| t.id == item_id) {
                state.active_tool_calls[pos].arguments.push_str(delta);
                output.push_str(&sse_line(&json!({
                    "id": state.chat_id,
                    "object": "chat.completion.chunk",
                    "created": state.created,
                    "model": &state.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": pos,
                                "function": {"arguments": delta},
                            }],
                        },
                        "finish_reason": null,
                    }],
                })));
            }
        }
        "response.completed" => {
            let empty_resp = json!({});
            let response = chunk.get("response").unwrap_or(&empty_resp);
            if let Some(usage) = response.get("usage") {
                state.total_input = usage
                    .get("input_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                state.total_output = usage
                    .get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
            }
            let finish_reason = if !state.active_tool_calls.is_empty() {
                "tool_calls"
            } else {
                "stop"
            };
            output.push_str(&sse_line(&json!({
                "id": state.chat_id,
                "object": "chat.completion.chunk",
                "created": state.created,
                "model": &state.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": state.total_input,
                    "completion_tokens": state.total_output,
                    "total_tokens": state.total_input + state.total_output,
                },
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

// ── Helper functions ───────────────────────────────────────────────────────

fn instruction_text(instructions: &Value) -> String {
    match instructions {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn is_openai_o_series(model: &str) -> bool {
    model.starts_with("o1") || model.starts_with("o3") || model.starts_with("o4")
}

fn collapse_system_messages_to_head(mut messages: Vec<Value>) -> Vec<Value> {
    let mut system_texts: Vec<String> = Vec::new();
    let mut first_system = None;
    for (i, msg) in messages.iter().enumerate() {
        if msg.get("role").and_then(|v| v.as_str()) == Some("system") {
            if first_system.is_none() {
                first_system = Some(i);
            }
            if let Some(text) = msg.get("content").and_then(|v| v.as_str()) {
                system_texts.push(text.to_string());
            }
        }
    }
    if let Some(pos) = first_system {
        messages[pos]["content"] = json!(system_texts.join("\n"));
        // Keep system message at pos, remove others
        let mut new_messages: Vec<Value> = Vec::new();
        for (i, msg) in messages.drain(..).enumerate() {
            let is_system = msg.get("role").and_then(|v| v.as_str()) == Some("system");
            if !is_system || i == pos {
                new_messages.push(msg);
            }
        }
        messages = new_messages;
    }
    messages
}

fn append_responses_input_as_chat_messages(input: &Value, messages: &mut Vec<Value>) {
    let mut pending_tool_calls: Vec<Value> = Vec::new();

    match input {
        Value::String(text) => {
            messages.push(json!({"role": "user", "content": text}));
        }
        Value::Array(items) => {
            for item in items {
                append_responses_item_as_chat_message(item, messages, &mut pending_tool_calls);
            }
        }
        Value::Object(_) => {
            append_responses_item_as_chat_message(input, messages, &mut pending_tool_calls);
        }
        _ => {}
    }

    flush_pending_tool_calls(messages, &mut pending_tool_calls);
}

fn append_responses_item_as_chat_message(
    item: &Value,
    messages: &mut Vec<Value>,
    pending_tool_calls: &mut Vec<Value>,
) {
    let item_type = item.get("type").and_then(|v| v.as_str());
    match item_type {
        Some("message") => {
            flush_pending_tool_calls(messages, pending_tool_calls);
            let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let role = if role == "developer" { "system" } else { role };
            let content = responses_content_to_chat_text(item);
            messages.push(json!({"role": role, "content": content}));
        }
        Some("local_shell_call") | Some("custom_tool_call") | Some("function_call") => {
            let id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
            let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let arguments = item
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            pending_tool_calls.push(json!({
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }));
        }
        Some("function_call_output") => {
            flush_pending_tool_calls(messages, pending_tool_calls);
            let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
            let output = item.get("output").and_then(|v| v.as_str()).unwrap_or("");
            messages.push(json!({"role": "tool", "tool_call_id": call_id, "content": output}));
        }
        // Item without a recognized type — try role-based handling
        _ => {
            if item.get("role").is_some() {
                // Message-like item without explicit type
                flush_pending_tool_calls(messages, pending_tool_calls);
                let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                // Map "developer" role to "system" for broader compatibility
                // (OpenAI introduced "developer" as a replacement for "system",
                // but many third-party APIs only accept "system")
                let role = if role == "developer" { "system" } else { role };
                let content = responses_content_to_chat_text(item);
                messages.push(json!({"role": role, "content": content}));
            } else if let Some(text) = item.as_str() {
                // Bare string input
                messages.push(json!({"role": "user", "content": text}));
            }
            // Skip unknown item types (reasoning, local_shell_call, web_search_call,
            // image_generation_call, compaction, tool_search_call, etc.)
            // These don't have a chat completions equivalent.
        }
    }
}

fn flush_pending_tool_calls(messages: &mut Vec<Value>, pending_tool_calls: &mut Vec<Value>) {
    if pending_tool_calls.is_empty() {
        return;
    }
    let tool_calls = std::mem::take(pending_tool_calls);
    messages.push(json!({"role": "assistant", "tool_calls": tool_calls}));
}

fn responses_content_to_chat_text(item: &Value) -> String {
    match item.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let texts: Vec<&str> = parts
                .iter()
                .filter_map(|c| {
                    let ct = c.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    match ct {
                        "input_text" | "output_text" => c.get("text").and_then(|v| v.as_str()),
                        "input_image" => Some("[image]"),
                        _ => c.get("text").and_then(|v| v.as_str()),
                    }
                })
                .collect();
            texts.join("\n")
        }
        _ => String::new(),
    }
}

fn responses_tools_to_chat_tools(body: &Value) -> Vec<Value> {
    let tools = body.get("tools").and_then(|v| v.as_array());
    let Some(tools) = tools else {
        return vec![];
    };
    tools
        .iter()
        .filter_map(|tool| {
            let tool_type = tool.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if tool_type == "function" {
                let name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let description = tool
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let params = tool.get("parameters").cloned().unwrap_or(json!({}));
                Some(json!({
                    "type": "function",
                    "function": {"name": name, "description": description, "parameters": params},
                }))
            } else {
                None
            }
        })
        .collect()
}

fn chat_tools_to_responses_tools(body: &Value) -> Vec<Value> {
    let tools = body.get("tools").and_then(|v| v.as_array());
    let Some(tools) = tools else {
        return vec![];
    };
    tools
        .iter()
        .filter_map(|tool| {
            let tool_type = tool.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if tool_type == "function" {
                let empty_fn = json!({});
                let fn_obj = tool.get("function").unwrap_or(&empty_fn);
                Some(json!({
                    "type": "function",
                    "name": fn_obj.get("name").cloned().unwrap_or(json!("")),
                    "description": fn_obj.get("description").cloned().unwrap_or(json!("")),
                    "parameters": fn_obj.get("parameters").cloned().unwrap_or(json!({})),
                }))
            } else {
                None
            }
        })
        .collect()
}

fn responses_tool_choice_to_chat(tool_choice: &Value) -> Value {
    match tool_choice {
        Value::String(s) => json!(s),
        Value::Object(obj) => {
            if let Some(name) = obj.get("name").and_then(|v| v.as_str()) {
                json!({"type": "function", "function": {"name": name}})
            } else {
                json!("auto")
            }
        }
        _ => json!("auto"),
    }
}

fn msg_content_to_responses_content(msg: &Value) -> Value {
    match msg.get("content") {
        Some(Value::String(s)) => json!([{"type": "input_text", "text": s}]),
        Some(Value::Array(arr)) => json!(arr),
        _ => json!([{"type": "input_text", "text": ""}]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_responses_to_chat_simple() {
        let body = json!({
            "model": "gpt-4",
            "instructions": "You are helpful.",
            "input": "Hello"
        });
        let result = request_responses_to_chat(&body).unwrap();
        assert_eq!(result["model"], "gpt-4");
        let msgs = result["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn test_request_responses_to_chat_with_tools() {
        let body = json!({
            "model": "gpt-4",
            "input": [
                {"role": "user", "content": "What is the weather?"},
                {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "Sunny, 20C"},
            ]
        });
        let result = request_responses_to_chat(&body).unwrap();
        let msgs = result["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[2]["role"], "tool");
    }

    #[test]
    fn test_request_chat_to_responses_simple() {
        let body = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
        });
        let result = request_chat_to_responses(&body).unwrap();
        assert_eq!(result["model"], "gpt-4");
        assert_eq!(result["instructions"], "You are helpful.");
        let input = result["input"].as_array().unwrap();
        assert_eq!(input.len(), 1);
    }

    #[test]
    fn test_response_chat_to_responses() {
        let chat = json!({
            "model": "test",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let result = response_chat_to_responses(&chat, None).unwrap();
        assert_eq!(result["object"], "response");
        assert_eq!(result["status"], "completed");
    }

    #[test]
    fn test_response_responses_to_chat() {
        let resp = json!({
            "model": "test",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there!"}]
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        });
        let result = response_responses_to_chat(&resp, None).unwrap();
        assert_eq!(result["object"], "chat.completion");
        let choices = result["choices"].as_array().unwrap();
        assert_eq!(choices[0]["message"]["content"], "Hi there!");
    }
}
