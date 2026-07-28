//! Tool call history store for Responses→Chat conversion.
//!
//! When Codex (Responses API) sends `previous_response_id` + tool output,
//! the upstream Chat Completions API requires the original tool call to
//! appear before the tool output. This store caches tool calls from
//! upstream responses and enriches subsequent requests.
//!
//! Based on cc-switch-cli's `CodexChatHistoryStore` design.

use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

const MAX_CACHED_RESPONSES: usize = 512;

#[derive(Debug, Clone, Default)]
struct CachedResponse {
    /// function_call items keyed by call_id
    calls_by_id: HashMap<String, Value>,
    /// Original call order
    call_order: Vec<String>,
}

#[derive(Debug, Default)]
struct Inner {
    responses: HashMap<String, CachedResponse>,
    response_order: VecDeque<String>,
    /// Fallback index: call_id → response_ids
    call_index: HashMap<String, VecDeque<String>>,
}

/// Cross-request history store for tool call enrichment.
#[derive(Debug, Default)]
pub struct ToolCallHistoryStore {
    inner: RwLock<Inner>,
}

impl ToolCallHistoryStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record tool calls from an upstream response (Responses format).
    /// Call this after receiving a response from the upstream.
    pub async fn record_response(&self, response: &Value) {
        let Some(response_id) = response
            .get("id")
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty())
            .map(|s| s.to_string())
        else {
            return;
        };

        let calls: Vec<(String, Value)> = response
            .get("output")
            .and_then(|v| v.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(cached_tool_call)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        if calls.is_empty() {
            return;
        }

        let mut inner = self.inner.write().await;
        inner.insert_calls(&response_id, calls);
    }

    /// Enrich a Responses API request body with cached tool calls.
    ///
    /// If the request contains `previous_response_id` and the input has
    /// `function_call_output` items without preceding `function_call` items,
    /// this method inserts the cached function_call items.
    pub async fn enrich_request(&self, body: &mut Value) -> usize {
        let previous_response_id = body
            .get("previous_response_id")
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty())
            .map(|s| s.to_string());

        let Some(input) = body.get_mut("input") else {
            return 0;
        };

        let original_input = std::mem::take(input);
        let original_input_backup = original_input.clone();
        let items = match original_input {
            Value::Array(items) => items,
            Value::Object(obj) => vec![Value::Object(obj)],
            other => {
                *input = other;
                return 0;
            }
        };

        // Collect tool output call_ids and existing tool call call_ids
        let output_call_ids: HashSet<String> = items
            .iter()
            .filter(|item| {
                item.get("type")
                    .and_then(|v| v.as_str())
                    .is_some_and(is_tool_output_type)
            })
            .filter_map(response_item_call_id)
            .collect();

        let existing_call_ids: HashSet<String> = items
            .iter()
            .filter(|item| {
                item.get("type")
                    .and_then(|v| v.as_str())
                    .is_some_and(is_tool_call_type)
            })
            .filter_map(response_item_call_id)
            .collect();

        let requested_call_ids: HashSet<String> =
            output_call_ids.union(&existing_call_ids).cloned().collect();

        let lookup = self
            .lookup(previous_response_id.as_deref(), &requested_call_ids)
            .await;

        let restore_group = lookup.restore_group(&output_call_ids, &existing_call_ids);
        let restore_group_ids: HashSet<String> = restore_group
            .iter()
            .map(|(call_id, _)| call_id.clone())
            .collect();

        let mut restore_group = Some(restore_group);
        let mut seen_call_ids = HashSet::new();
        let mut restored = 0usize;
        let mut new_items = Vec::new();

        for mut item in items {
            let item_type = item.get("type").and_then(|v| v.as_str());
            match item_type {
                Some(t) if is_tool_call_type(t) => {
                    if let Some(call_id) = response_item_call_id(&item) {
                        if let Some(cached) = lookup.call(&call_id) {
                            if enrich_call_item_from_cache(&mut item, cached) {
                                // enriched
                            }
                        }
                        seen_call_ids.insert(call_id);
                    }
                    new_items.push(item);
                }
                Some(t) if is_tool_output_type(t) => {
                    // Restore missing tool calls before this output
                    if let Some(group) = restore_group.take().filter(|g| !g.is_empty()) {
                        for (call_id, cached_item) in group {
                            seen_call_ids.insert(call_id);
                            new_items.push(cached_item);
                            restored += 1;
                        }
                    }

                    // Also try to restore by call_id lookup
                    if let Some(call_id) = response_item_call_id(&item) {
                        if !seen_call_ids.contains(&call_id)
                            && !restore_group_ids.contains(&call_id)
                        {
                            if let Some(cached) = lookup.call(&call_id).cloned() {
                                seen_call_ids.insert(call_id);
                                new_items.push(cached);
                                restored += 1;
                            }
                        }
                    }
                    new_items.push(item);
                }
                _ => {
                    new_items.push(item);
                }
            }
        }

        if restored > 0 {
            *input = Value::Array(new_items);
        } else {
            *input = original_input_backup; // no changes, put back original
        }

        restored
    }

    async fn lookup(
        &self,
        previous_response_id: Option<&str>,
        requested_call_ids: &HashSet<String>,
    ) -> CachedLookup {
        let inner = self.inner.read().await;
        let previous = previous_response_id.and_then(|id| inner.responses.get(id).cloned());
        let fallback = inner.unique_fallback_calls(requested_call_ids, previous.as_ref());
        CachedLookup { previous, fallback }
    }
}

#[derive(Debug, Clone, Default)]
struct CachedLookup {
    previous: Option<CachedResponse>,
    fallback: CachedResponse,
}

impl CachedLookup {
    fn call(&self, call_id: &str) -> Option<&Value> {
        self.previous
            .as_ref()
            .and_then(|r| r.calls_by_id.get(call_id))
            .or_else(|| self.fallback.calls_by_id.get(call_id))
    }

    fn restore_group(
        &self,
        output_call_ids: &HashSet<String>,
        existing_call_ids: &HashSet<String>,
    ) -> Vec<(String, Value)> {
        let source = self.previous.as_ref().unwrap_or(&self.fallback);
        let mut result = Vec::new();

        for call_id in &source.call_order {
            if output_call_ids.contains(call_id) && !existing_call_ids.contains(call_id) {
                if let Some(item) = source.calls_by_id.get(call_id) {
                    result.push((call_id.clone(), item.clone()));
                }
            }
        }

        result
    }
}

impl Inner {
    fn insert_calls(&mut self, response_id: &str, calls: Vec<(String, Value)>) {
        if !self.responses.contains_key(response_id) {
            self.response_order.push_back(response_id.to_string());
        }

        let cached = self.responses.entry(response_id.to_string()).or_default();
        for (call_id, item) in calls {
            if !cached.calls_by_id.contains_key(&call_id) {
                cached.call_order.push(call_id.clone());
            }
            let cid = call_id.clone();
            cached.calls_by_id.insert(call_id, item);
            self.call_index
                .entry(cid)
                .or_default()
                .push_back(response_id.to_string());
        }

        // LRU eviction
        while self.response_order.len() > MAX_CACHED_RESPONSES {
            if let Some(old_id) = self.response_order.pop_front() {
                if let Some(old_resp) = self.responses.remove(&old_id) {
                    for call_id in &old_resp.call_order {
                        if let Some(queue) = self.call_index.get_mut(call_id) {
                            queue.retain(|id| id != &old_id);
                            if queue.is_empty() {
                                self.call_index.remove(call_id);
                            }
                        }
                    }
                }
            }
        }
    }

    fn unique_fallback_calls(
        &self,
        requested_call_ids: &HashSet<String>,
        previous: Option<&CachedResponse>,
    ) -> CachedResponse {
        let mut result = CachedResponse::default();

        for call_id in requested_call_ids {
            if let Some(queue) = self.call_index.get(call_id) {
                // Find a response that is not the previous one
                for response_id in queue {
                    if previous
                        .as_ref()
                        .map(|p| p.calls_by_id.contains_key(call_id))
                        .unwrap_or(false)
                    {
                        continue;
                    }
                    if let Some(resp) = self.responses.get(response_id) {
                        if let Some(item) = resp.calls_by_id.get(call_id) {
                            if !result.calls_by_id.contains_key(call_id) {
                                result.calls_by_id.insert(call_id.clone(), item.clone());
                                result.call_order.push(call_id.clone());
                            }
                        }
                    }
                }
            }
        }

        result
    }
}

/// Extract a cached tool call from a Responses API output item.
fn cached_tool_call(item: &Value) -> Option<(String, Value)> {
    let item_type = item.get("type").and_then(|v| v.as_str())?;
    if !is_tool_call_type(item_type) {
        return None;
    }
    let call_id = response_item_call_id(item)?;
    Some((call_id, item.clone()))
}

fn is_tool_call_type(t: &str) -> bool {
    matches!(t, "function_call")
}

fn is_tool_output_type(t: &str) -> bool {
    matches!(t, "function_call_output")
}

fn response_item_call_id(item: &Value) -> Option<String> {
    item.get("call_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

fn enrich_call_item_from_cache(item: &mut Value, cached: &Value) -> bool {
    // If the item is missing fields that the cached version has, copy them over.
    let mut changed = false;
    if item.get("name").is_none() && cached.get("name").is_some() {
        if let Some(name) = cached.get("name").cloned() {
            item["name"] = name;
            changed = true;
        }
    }
    if item.get("arguments").is_none() && cached.get("arguments").is_some() {
        if let Some(args) = cached.get("arguments").cloned() {
            item["arguments"] = args;
            changed = true;
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_record_and_enrich() {
        let store = ToolCallHistoryStore::default();

        // Simulate a response with a function_call
        let response = json!({
            "id": "resp_123",
            "output": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{\"city\":\"Tokyo\"}",
            }],
        });

        // We need to run async
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            store.record_response(&response).await;

            // Now simulate a follow-up request with previous_response_id + tool output
            let mut request = json!({
                "model": "test",
                "previous_response_id": "resp_123",
                "input": [
                    {"type": "function_call_output", "call_id": "call_1", "output": "Sunny, 20C"},
                ],
            });

            let restored = store.enrich_request(&mut request).await;
            assert_eq!(restored, 1);

            let input = request["input"].as_array().unwrap();
            // Should have function_call restored before function_call_output
            assert_eq!(input.len(), 2);
            assert_eq!(input[0]["type"], "function_call");
            assert_eq!(input[0]["call_id"], "call_1");
            assert_eq!(input[0]["name"], "get_weather");
            assert_eq!(input[1]["type"], "function_call_output");
        });
    }

    #[test]
    fn test_no_enrichment_without_previous_response_id() {
        let store = ToolCallHistoryStore::default();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let response = json!({
                "id": "resp_456",
                "output": [{
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "get_time",
                    "arguments": "{}",
                }],
            });
            store.record_response(&response).await;

            let mut request = json!({
                "model": "test",
                "input": [
                    {"type": "function_call_output", "call_id": "call_2", "output": "12:00"},
                ],
            });

            let restored = store.enrich_request(&mut request).await;
            assert_eq!(restored, 1);
            let input = request["input"].as_array().unwrap();
            assert_eq!(input.len(), 2);
            assert_eq!(input[0]["type"], "function_call");
            assert_eq!(input[1]["type"], "function_call_output");
        });
    }
}
