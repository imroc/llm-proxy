# LLM Proxy

[中文文档](./README.zh-CN.md)

A thin local proxy for LLM APIs with automatic retry, flexible protocol conversion, and model-level routing. Designed for team-shared API scenarios where rate limiting (429) frequently interrupts AI CLI tool sessions.

## Why?

When using AI CLI tools (CodeBuddy Code, Claude Code, Codex CLI) with team-shared model APIs, global rate limits cause 429 errors that interrupt sessions. Most CLI tools have limited or no retry capability, requiring manual "continue" to resume work.

This proxy sits between the CLI tool and the API, transparently retrying requests that receive 429/5xx errors with exponential backoff + jitter, so the CLI tool never sees the error.

It also solves a second problem: different AI tools use different API protocols (OpenAI Responses, Chat Completions, Anthropic Messages), but upstream usually does not provide full support. The proxy automatically detects the inbound protocol and converts it to whatever the upstream supports — no manual transform configuration needed.

## Features

- **Unlimited retry** — keeps retrying until the client disconnects or succeeds
- **Flexible protocol conversion** — auto-detects inbound format (Responses/Chat/Anthropic) and converts to any supported upstream format
- **Default route** — all AI tools point to the same address; the `model` field in the request body determines the upstream target
- **API key management** — per-model API keys with `${ENV_VAR}` expansion, independent of client-side keys
- **Model name mapping** — automatic bidirectional model name rewriting (`upstream_model`)
- **GET /v1/models** — returns all configured models for client-side model discovery
- **Transparent** — CLI tools don't need any retry support; just point them at the proxy
- **Streaming support** — SSE streaming passthrough and real-time protocol conversion
- **Client-aware** — detects client disconnect immediately (even during backoff) and stops retrying
- **Hot-reload config** — add/remove routes and models without restart
- **Tool call history cache** — handles `previous_response_id` across protocol conversion
- **Prometheus metrics** — monitor retry rates and upstream status (per route + model)
- **Single binary** — low memory footprint, written in Rust

## Quick Start

```bash
# Build
make build

# Create config
cp config.example.toml config.toml
# Edit config.toml to point to your API

# Run
./target/release/llm-proxy --config config.toml
```

Point your AI CLI tool's API URL to `http://127.0.0.1:8888` (default route) or `http://127.0.0.1:8888/{route_name}/{api_path}` (named route).

## Configuration

```toml
# Log level (error, warn, info, debug, trace). Hot-reloadable.
log_level = "info"

[defaults]
max_retries = 9999           # Effectively unlimited
base_delay_ms = 1000         # Exponential backoff base
max_delay_ms = 60000         # Backoff cap
max_total_wait_ms = 0        # 0 = rely on client disconnect
connect_timeout_secs = 30
retry_status_codes = [429, 500, 502, 503, 504, 408, 529]

# Named route: URL path starts with /tkehub/...
[routes.tkehub]
target = "http://tkehub.woa.com"

# Default route: URL without route name, model field determines upstream
[routes.default]

# Each model has its own target, API key, and supported formats
[routes.default.models."my-glm"]
target = "http://tkehub.woa.com"
upstream_formats = ["responses", "chat"]  # what the upstream supports
api_key = "${TKEHUB_API_KEY}"              # env var expansion
upstream_model = "tke/glm-latest"         # rewrite model name for upstream

[routes.default.models."my-deepseek"]
target = "https://tokenhub.tencentmaas.com"
upstream_formats = ["chat"]               # only supports chat
api_key = "${DEEPSEEK_API_KEY}"
upstream_model = "deepseek-chat"
```

See [config.example.toml](./config.example.toml) for a complete example.

### Protocol Auto-Detection

The proxy detects the inbound protocol from the URL path:

| URL Path               | Protocol                |
| ---------------------- | ----------------------- |
| `/v1/responses`        | OpenAI Responses API    |
| `/v1/chat/completions` | OpenAI Chat Completions |
| `/v1/messages`         | Anthropic Messages      |

### `upstream_formats`

Declare what the upstream supports, ordered by preference. The proxy auto-decides:

- If the inbound format **is in** the list → **passthrough** (no conversion)
- If the inbound format **is not in** the list → **convert** to the first format in the list
- Empty (unset) → passthrough anything

This replaces the old `transform` field. No more manual transform configuration — the proxy infers the conversion direction automatically.

### Default Route

The default route (`routes.default`) enables unified access: all AI CLI tools point to the same address (`http://127.0.0.1:8888`), and the `model` field in the request body determines which upstream to use. Each model can have its own target, API key, and supported formats.

Named routes and the default route coexist — named routes are matched first (first URL path segment), then the default route catches everything else.

### Model-Level Fields

All fields are optional — only specified fields override the route-level config.

| Field              | Description                                                                                                       |
| ------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `target`           | Upstream base URL (proxy auto-appends standard API path)                                                          |
| `upstream_formats` | List of protocols the upstream supports, ordered by preference                                                    |
| `api_key`          | API key for the upstream (supports `${ENV_VAR}` expansion)                                                        |
| `upstream_model`   | Rewrite the `model` field in requests; auto-rewrite responses back                                                |
| retry params       | `max_retries`, `base_delay_ms`, `max_delay_ms`, `max_total_wait_ms`, `connect_timeout_secs`, `retry_status_codes` |

### API Key Management

When `api_key` is configured at the model or route level, the proxy:

1. Strips the client's `Authorization` header
2. Injects the configured API key (with `${ENV_VAR}` expansion)
3. Forwards to the upstream

When no `api_key` is configured (named routes without `api_key`), the client's original `Authorization` header is forwarded as-is (backward compatible).

## Installation

```bash
make install
```

This interactively installs the binary and optionally sets up a systemd/launchd service.

## License

[MIT](./LICENSE)
