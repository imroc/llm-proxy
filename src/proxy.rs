use arc_swap::ArcSwap;
use bytes::Bytes;
use http::{HeaderMap, Method, Request, Response, StatusCode};
use http_body_util::{BodyExt, Full, StreamBody};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tracing::{info, trace, warn};

use crate::config::{Config, DEFAULT_ROUTE};
use crate::format::{self, Protocol};
use crate::health::ResponseBody;
use crate::log::extract_model;
use crate::metrics::Metrics;
use crate::retry::{
    compute_backoff_ms, compute_delay_ms, is_retryable_status, parse_retry_after_ms,
};
use crate::transform::{self, StreamTransformer, TransformedRequest};

const HOP_BY_HOP_REQUEST: &[&str] = &[
    "host",
    "connection",
    "content-length",
    "transfer-encoding",
    "keep-alive",
    "proxy-authorization",
    "te",
    "trailer",
    "upgrade",
];

const HOP_BY_HOP_RESPONSE: &[&str] = &[
    "content-length",
    "content-encoding",
    "transfer-encoding",
    "connection",
    "keep-alive",
];

fn error_response(status: StatusCode, message: &str, error_type: &str) -> Response<ResponseBody> {
    let body = serde_json::json!({"error": {"message": message, "type": error_type}});
    let mut resp = Response::new(
        Full::new(Bytes::from(body.to_string()))
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            .boxed(),
    );
    *resp.status_mut() = status;
    resp.headers_mut().insert(
        http::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );
    resp
}

/// Route match result: either a named route or the default route.
enum RouteMatch {
    Named { name: String, rest: String },
    Default { rest: String },
    NotFound,
}

/// Match the URL path against configured routes.
/// Named routes take priority; if no named route matches, try default route.
fn match_route(path: &str, config: &Config) -> RouteMatch {
    let path = path.strip_prefix('/').unwrap_or(path);

    // Try to split into route_name/rest
    if let Some((first_segment, rest)) = path.split_once('/') {
        // Check if first_segment is a named route
        if config.routes.contains_key(first_segment) {
            return RouteMatch::Named {
                name: first_segment.to_string(),
                rest: rest.to_string(),
            };
        }
    }

    // Check if default route is configured
    if config.routes.contains_key(DEFAULT_ROUTE) {
        // The rest is the full path (including the first segment)
        RouteMatch::Default {
            rest: path.to_string(),
        }
    } else {
        RouteMatch::NotFound
    }
}

fn build_forward_headers(req_headers: &HeaderMap, has_config_api_key: bool) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (key, value) in req_headers {
        let lower = key.as_str().to_lowercase();
        if HOP_BY_HOP_REQUEST.contains(&lower.as_str()) {
            continue;
        }
        // Drop Authorization only when a config-level api_key will replace it
        if has_config_api_key && lower == "authorization" {
            continue;
        }
        headers.insert(key.clone(), value.clone());
    }
    headers
}

fn strip_response_hop_by_hop(resp_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (key, value) in resp_headers {
        let lower = key.as_str().to_lowercase();
        if HOP_BY_HOP_RESPONSE.contains(&lower.as_str()) {
            continue;
        }
        headers.insert(key.clone(), value.clone());
    }
    headers
}

/// Rewrite the `model` field in a JSON request body.
fn rewrite_model_in_body(body: &[u8], new_model: &str) -> Bytes {
    let mut value: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => return Bytes::copy_from_slice(body),
    };
    if let Some(obj) = value.as_object_mut() {
        obj.insert(
            "model".to_string(),
            serde_json::Value::String(new_model.to_string()),
        );
        serde_json::to_vec(&value)
            .map(Bytes::from)
            .unwrap_or_else(|_| Bytes::copy_from_slice(body))
    } else {
        Bytes::copy_from_slice(body)
    }
}

/// Handle GET /v1/models — return all configured model names.
fn handle_models(config: &Config) -> Response<ResponseBody> {
    let models: Vec<serde_json::Value> = config
        .all_model_names()
        .iter()
        .map(|name| serde_json::json!({"id": name, "object": "model", "owned_by": "llm-proxy"}))
        .collect();
    let body = serde_json::json!({"object": "list", "data": models});
    let mut resp = Response::new(
        Full::new(Bytes::from(body.to_string()))
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            .boxed(),
    );
    resp.headers_mut().insert(
        http::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );
    resp
}

pub async fn handle_request(
    req: Request<hyper::body::Incoming>,
    config: Arc<ArcSwap<Config>>,
    metrics: Arc<Metrics>,
    client: reqwest::Client,
    disconnect_token: CancellationToken,
    version: &'static str,
) -> Response<ResponseBody> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    // Health check
    if path == "/healthz" {
        return crate::health::handle_healthz(config, version);
    }

    // Metrics
    if path == "/metrics" {
        return crate::health::handle_metrics(metrics);
    }

    let config = config.load();

    // GET /v1/models or /models — return configured model list
    if method == Method::GET && (path.ends_with("/v1/models") || path.ends_with("/models")) {
        return handle_models(&config);
    }

    // Route matching
    let (route_name, rest, route_config) = match match_route(&path, &config) {
        RouteMatch::Named { name, rest } => {
            let rc = match config.resolve_route(&name) {
                Some(rc) => rc,
                None => {
                    return error_response(
                        StatusCode::NOT_FOUND,
                        &format!(
                            "unknown route \"{}\". Available: {}",
                            name,
                            config.route_names().join(", ")
                        ),
                        "unknown_route",
                    );
                }
            };
            (name, rest, rc)
        }
        RouteMatch::Default { rest } => {
            let rc = match config.resolve_default_route() {
                Some(rc) => rc,
                None => {
                    return error_response(
                        StatusCode::NOT_FOUND,
                        &format!(
                            "no route matched \"{}\" and no default route configured",
                            path
                        ),
                        "unknown_route",
                    );
                }
            };
            (DEFAULT_ROUTE.to_string(), rest, rc)
        }
        RouteMatch::NotFound => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!(
                    "no route matched \"{}\". Available: {}",
                    path,
                    config.route_names().join(", ")
                ),
                "unknown_route",
            );
        }
    };

    // Extract headers before consuming body
    let forward_headers = build_forward_headers(req.headers(), route_config.api_key.is_some());

    // Read full request body
    let has_body = method != Method::GET && method != Method::HEAD;
    let body_bytes: Bytes = if has_body {
        match req.collect().await {
            Ok(collected) => collected.to_bytes(),
            Err(e) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    &format!("failed to read request body: {}", e),
                    "read_body_failed",
                );
            }
        }
    } else {
        Bytes::new()
    };

    // Extract client model and resolve model-level config
    let client_model = extract_model(&body_bytes);
    let model_str = client_model.as_deref().unwrap_or("");

    // Two-step resolve: route-level → model-level
    let route_config = if let Some(ref model) = client_model {
        route_config.resolve_model(model)
    } else {
        route_config
    };

    // For default route, model must be present (to determine target)
    if route_name == DEFAULT_ROUTE && route_config.target.is_none() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "default route requires a model field in the request body to determine the upstream target",
            "missing_model",
        );
    }

    // Detect inbound protocol from URL path
    let inbound_protocol = format::Protocol::from_path(&rest).unwrap_or(format::Protocol::Chat); // default to chat if unknown

    // Determine conversion direction
    let conversion = format::conversion_direction(inbound_protocol, &route_config.upstream_formats);

    // Determine the upstream protocol (target format)
    let upstream_protocol = conversion.map(|(_, to)| to).unwrap_or(inbound_protocol);

    // Determine target URL — use anthropic_target when the upstream protocol is Anthropic
    let raw_target = if upstream_protocol == format::Protocol::Anthropic {
        route_config
            .anthropic_target
            .as_ref()
            .or(route_config.target.as_ref())
    } else {
        route_config.target.as_ref()
    };
    let target = match raw_target {
        Some(t) => t.clone(),
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "no target configured for this route/model",
                "no_target",
            );
        }
    };

    // Inject API key from config
    let forward_headers = if let Some(ref api_key) = route_config.api_key {
        let mut h = forward_headers.clone();
        h.insert(
            http::header::AUTHORIZATION,
            http::HeaderValue::from_str(&format!("Bearer {}", api_key))
                .unwrap_or_else(|_| http::HeaderValue::from_static("Bearer")),
        );
        h
    } else {
        forward_headers
    };

    // Rewrite model in request body if upstream_model is configured
    let body_bytes = if let Some(ref upstream_model) = route_config.upstream_model {
        if upstream_model != model_str {
            tracing::debug!(
                "[{}|{}] rewriting model: {} → {}",
                route_name,
                model_str,
                model_str,
                upstream_model
            );
            rewrite_model_in_body(&body_bytes, upstream_model)
        } else {
            body_bytes
        }
    } else {
        body_bytes
    };

    // The client's model name (for response rewriting)
    let response_model = client_model.as_deref();

    // Determine upstream URL path
    let upstream_path = upstream_protocol.api_path();
    let upstream_url = format!("{}{}", target.trim_end_matches('/'), upstream_path);

    let tag = if !model_str.is_empty() {
        format!("[{}|{}]", route_name, model_str)
    } else {
        format!("[{}]", route_name)
    };

    metrics.record_request(&route_name, model_str);

    // Apply request transform if needed
    let request_body = if let Some((from, to)) = conversion {
        match transform::transform_request(from, to, &body_bytes) {
            Ok(TransformedRequest { body, .. }) => {
                tracing::debug!(
                    "{} transform: {} → {} ({} bytes)",
                    tag,
                    from,
                    to,
                    body.len()
                );
                body
            }
            Err(e) => {
                warn!("{} transform failed: {}, sending original body", tag, e);
                body_bytes.clone()
            }
        }
    } else {
        body_bytes.clone()
    };

    let conversion_desc = match conversion {
        Some((from, to)) => format!("{}→{}", from, to),
        None => "passthrough".to_string(),
    };
    info!(
        "{} -> {} {} (body={} bytes)",
        tag,
        upstream_url,
        conversion_desc,
        request_body.len()
    );

    // Detailed routing info for debugging
    let model_mapping = match &route_config.upstream_model {
        Some(um) if um != model_str => format!(" model={}→{}", model_str, um),
        _ => format!(" model={}", model_str),
    };
    tracing::debug!(
        "{} {} {}{} {} body={}bytes",
        tag,
        method,
        path,
        model_mapping,
        upstream_url,
        request_body.len()
    );
    if conversion.is_some() {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&request_body) {
            if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
                let summary: Vec<String> = msgs
                    .iter()
                    .enumerate()
                    .map(|(i, m)| {
                        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("MISSING");
                        let tc = m
                            .get("tool_calls")
                            .and_then(|t| t.as_array())
                            .map(|a| a.len())
                            .unwrap_or(0);
                        let clen = match m.get("content") {
                            Some(serde_json::Value::String(s)) => s.len(),
                            Some(serde_json::Value::Array(arr)) => arr.len(),
                            _ => 0,
                        };
                        format!("msg[{}]:role={}:tc={}:len={}", i, role, tc, clen)
                    })
                    .collect();
                tracing::debug!("{} converted messages: {}", tag, summary.join(", "));
            }
        }
    }

    // Retry loop
    let start_time = Instant::now();
    let mut attempt: u32 = 0;
    let mut total_wait_ms: u64 = 0;

    loop {
        if attempt >= route_config.max_retries {
            warn!(
                "{} -> {} retry exhausted (max_retries={})",
                tag, upstream_url, route_config.max_retries
            );
            return error_response(
                StatusCode::BAD_GATEWAY,
                "upstream request failed (max retries exceeded)",
                "upstream_failed",
            );
        }

        if route_config.max_total_wait_ms > 0 && total_wait_ms >= route_config.max_total_wait_ms {
            warn!(
                "{} -> {} total wait budget exceeded ({}ms)",
                tag, upstream_url, route_config.max_total_wait_ms
            );
            return error_response(
                StatusCode::BAD_GATEWAY,
                "upstream request failed (total wait exceeded)",
                "upstream_failed",
            );
        }

        let req_builder = client
            .request(method.clone(), &upstream_url)
            .headers(forward_headers.clone());

        let req_builder = if has_body {
            req_builder.body(request_body.clone())
        } else {
            req_builder
        };

        let send_result = tokio::select! {
            r = req_builder.send() => r,
            _ = disconnect_token.cancelled() => {
                info!("{} -> {} client disconnected during request", tag, upstream_url);
                return error_response(StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_GATEWAY), "client disconnected", "client_disconnected");
            }
        };

        match send_result {
            Ok(response) => {
                let status = response.status().as_u16();
                let retryable = is_retryable_status(status, &route_config.retry_status_codes);

                if retryable && attempt < route_config.max_retries {
                    let retry_after_ms = parse_retry_after_ms(response.headers());
                    let _body_text = response.text().await.unwrap_or_default();
                    let backoff_ms = compute_backoff_ms(
                        route_config.base_delay_ms,
                        route_config.max_delay_ms,
                        attempt,
                    );
                    let delay_ms = compute_delay_ms(retry_after_ms, backoff_ms);

                    if route_config.max_total_wait_ms > 0
                        && total_wait_ms + delay_ms > route_config.max_total_wait_ms
                    {
                        warn!("{} -> {} total wait would exceed budget", tag, upstream_url);
                        return error_response(
                            StatusCode::BAD_GATEWAY,
                            "upstream request failed (budget exceeded)",
                            "upstream_failed",
                        );
                    }

                    info!(
                        "{} -> {} HTTP {} retry {}/{} in {}ms",
                        tag,
                        upstream_url,
                        status,
                        attempt + 1,
                        route_config.max_retries,
                        delay_ms
                    );
                    metrics.record_retry(&route_name, model_str);
                    attempt += 1;
                    total_wait_ms += delay_ms;

                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_millis(delay_ms)) => {}
                        _ = disconnect_token.cancelled() => {
                            info!("{} -> {} client disconnected during backoff", tag, upstream_url);
                            return error_response(StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_GATEWAY), "client disconnected", "client_disconnected");
                        }
                    }
                    continue;
                }

                if retryable {
                    warn!(
                        "{} -> {} retry exhausted, returning status={}",
                        tag, upstream_url, status
                    );
                }

                metrics.record_upstream_status(&route_name, model_str, status);
                let duration = start_time.elapsed();
                metrics.record_duration(&route_name, model_str, duration);

                // Build response with appropriate transform
                let is_stream = response
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .map(|ct| ct.contains("text/event-stream"))
                    .unwrap_or(false);

                if is_stream {
                    return build_streaming_response(
                        response,
                        upstream_protocol,
                        inbound_protocol,
                        response_model,
                        &tag,
                        &upstream_url,
                        disconnect_token,
                    )
                    .await;
                } else {
                    let resp = build_non_streaming_response(
                        response,
                        upstream_protocol,
                        inbound_protocol,
                        response_model,
                        &tag,
                        &upstream_url,
                    )
                    .await;
                    info!(
                        "{} -> {} {} ({}ms)",
                        tag,
                        upstream_url,
                        resp.status(),
                        duration.as_millis()
                    );
                    return resp;
                }
            }
            Err(e) => {
                if attempt >= route_config.max_retries {
                    warn!("{} -> {} network error exhausted: {}", tag, upstream_url, e);
                    return error_response(
                        StatusCode::BAD_GATEWAY,
                        &format!("upstream request failed: {}", e),
                        "upstream_failed",
                    );
                }

                let backoff_ms = compute_backoff_ms(
                    route_config.base_delay_ms,
                    route_config.max_delay_ms,
                    attempt,
                );

                if route_config.max_total_wait_ms > 0
                    && total_wait_ms + backoff_ms > route_config.max_total_wait_ms
                {
                    return error_response(
                        StatusCode::BAD_GATEWAY,
                        "upstream request failed (budget exceeded)",
                        "upstream_failed",
                    );
                }

                info!(
                    "{} -> {} network error, retry {}/{} in {}ms: {}",
                    tag,
                    upstream_url,
                    attempt + 1,
                    route_config.max_retries,
                    backoff_ms,
                    e
                );
                metrics.record_retry(&route_name, model_str);
                attempt += 1;
                total_wait_ms += backoff_ms;

                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_millis(backoff_ms)) => {}
                    _ = disconnect_token.cancelled() => {
                        return error_response(StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_GATEWAY), "client disconnected", "client_disconnected");
                    }
                }
                continue;
            }
        }
    }
}

/// Build a non-streaming response, applying protocol transform if needed.
async fn build_non_streaming_response(
    response: reqwest::Response,
    upstream_protocol: Protocol,
    inbound_protocol: Protocol,
    client_model: Option<&str>,
    tag: &str,
    _upstream_url: &str,
) -> Response<ResponseBody> {
    let status = response.status();
    let resp_headers = strip_response_hop_by_hop(response.headers());
    let body_bytes = response.bytes().await.unwrap_or_default();

    tracing::debug!(
        "{} upstream {} | response body={}bytes",
        tag,
        status,
        body_bytes.len()
    );

    // Debug log non-200 response bodies
    if !status.is_success() {
        let body_preview = String::from_utf8_lossy(&body_bytes);
        let p = if body_preview.len() > 1000 {
            &body_preview[..1000]
        } else {
            &body_preview
        };
        tracing::debug!("{} upstream returned {}: response body: {}", tag, status, p);
    }

    // Apply response transform only on 2xx success responses.
    // Error responses (4xx/5xx) are passed through as-is.
    let body_bytes = if status.is_success() && upstream_protocol != inbound_protocol {
        match transform::transform_response_body(
            upstream_protocol,
            inbound_protocol,
            &body_bytes,
            client_model,
        ) {
            Ok(transformed) => transformed,
            Err(e) => {
                warn!(
                    "{} response transform failed: {}, passing through original",
                    tag, e
                );
                body_bytes
            }
        }
    } else {
        // Passthrough — rewrite model if needed
        transform::common::rewrite_model_in_response(&body_bytes, client_model)
    };

    let mut resp = Response::new(
        Full::new(body_bytes)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            .boxed(),
    );
    *resp.status_mut() = status;
    for (key, value) in &resp_headers {
        resp.headers_mut().insert(key.clone(), value.clone());
    }
    resp
}

/// Build a streaming response, applying protocol transform if needed.
async fn build_streaming_response(
    response: reqwest::Response,
    upstream_protocol: Protocol,
    inbound_protocol: Protocol,
    client_model: Option<&str>,
    tag: &str,
    upstream_url: &str,
    _disconnect_token: CancellationToken,
) -> Response<ResponseBody> {
    let status = response.status();
    let resp_headers = strip_response_hop_by_hop(response.headers());
    let start_time = Instant::now();

    tracing::debug!("{} upstream {} | streaming", tag, status);

    // Only apply stream transform on 2xx success. Error responses passthrough.
    let mut stream_transformer = if status.is_success() {
        StreamTransformer::new(upstream_protocol, inbound_protocol, client_model)
    } else {
        StreamTransformer::new(upstream_protocol, upstream_protocol, client_model)
    };

    let (tx, rx) = mpsc::channel::<Result<Bytes, Box<dyn std::error::Error + Send + Sync>>>(128);
    let tag_owned = tag.to_string();
    let upstream_url_owned = upstream_url.to_string();

    // Spawn the streaming task
    tokio::spawn(async move {
        use futures::StreamExt;
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    buffer.push_str(&String::from_utf8_lossy(&chunk));

                    // Process complete lines
                    while let Some(pos) = buffer.find('\n') {
                        let line = buffer[..=pos].to_string();
                        buffer = buffer[pos + 1..].to_string();

                        let transformed = stream_transformer.transform_sse_line(&line);
                        if let Some(output) = transformed {
                            if tx.send(Ok(Bytes::from(output))).await.is_err() {
                                // Client disconnected
                                trace!(
                                    "{} -> {} client disconnected during streaming",
                                    tag_owned,
                                    upstream_url_owned
                                );
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx
                        .send(Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>))
                        .await;
                    break;
                }
            }
        }

        // Flush remaining buffer
        if !buffer.is_empty() {
            let transformed = stream_transformer.transform_sse_line(&buffer);
            if let Some(output) = transformed {
                let _ = tx.send(Ok(Bytes::from(output))).await;
            }
        }

        // If the upstream stream ended without a proper termination event
        // (e.g. connection dropped mid-stream without `[DONE]` /
        // `response.completed` / `message_stop`), flush the transformer to
        // emit a well-formed completion so the client doesn't treat the
        // response as a silently finished turn.
        if let Some(output) = stream_transformer.flush_if_incomplete() {
            let _ = tx.send(Ok(Bytes::from(output))).await;
        }

        // Signal end of stream
        let _ = tx.send(Ok(Bytes::new())).await;
        info!(
            "{} -> {} stream completed ({}ms)",
            tag_owned,
            upstream_url_owned,
            start_time.elapsed().as_millis()
        );
        drop(tag_owned);
        drop(upstream_url_owned);
    });

    let body = ReceiverStream::new(rx);
    let mapped = tokio_stream::StreamExt::map(body, |chunk: Result<Bytes, Box<dyn std::error::Error + Send + Sync>>| -> Result<http_body::Frame<Bytes>, std::convert::Infallible> {
        Ok(http_body::Frame::data(chunk.unwrap_or_default()))
    });
    let boxed = http_body_util::BodyExt::boxed(http_body_util::BodyExt::map_err(
        StreamBody::new(mapped),
        |e: std::convert::Infallible| -> Box<dyn std::error::Error + Send + Sync> { match e {} },
    ));

    let mut resp = Response::new(boxed);
    *resp.status_mut() = status;
    for (key, value) in &resp_headers {
        resp.headers_mut().insert(key.clone(), value.clone());
    }
    // Ensure SSE content type for streaming
    if !resp.headers().contains_key("content-type") {
        resp.headers_mut().insert(
            http::header::CONTENT_TYPE,
            "text/event-stream".parse().unwrap(),
        );
    }

    resp
}
