# LLM Proxy

[English](./README.md)

轻量级 LLM API 本地代理，提供自动重试、灵活协议转换和模型级路由。专为团队共享 API 场景下限频（429）频繁中断 AI CLI 工具会话而设计。

## 为什么需要？

使用 AI CLI 工具（CodeBuddy Code、Claude Code、Codex CLI）连接团队共享模型 API 时，全局限频会导致 429 错误中断会话。大多数 CLI 工具的重试能力有限或没有，需��手动 "continue" 恢复工作。

本代理位于 CLI 工具和 API 之间，对收到 429/5xx 错误的请求进行指数退避 + 抖动的透明重试，CLI 工具完全无感知。

同时解决第二个问题：不同 AI 工具使用不同的 API 协议（OpenAI Responses、Chat Completions、Anthropic Messages），但上游通常只支持一种。代理自动检测入站协议并转换为上游支持的格式——无需手动配置转换。

## 功能

- **无限重试** — 持续重试直到客户端断开或成功
- **灵活协议转换** — 自动检测入站格式（Responses/Chat/Anthropic）并转换为上游支持的任意格式
- **默认路由** — 所有 AI 工具指向同一地址，请求体中的 `model` 字段决定上游目标
- **API key 管理** — 按 model 配置 API key，支持 `${ENV_VAR}` 环境变量展开，独立于客户端 key
- **模型名称映射** — `upstream_model` 双向自动改写（请求改写 + 响应回写）
- **GET /v1/models** — 返回配置中的模型列表，供客户端动态发现
- **透明** — CLI 工具无需任何重试支持，只需指向代理
- **流式支持** — SSE 流式透传和实时协议转换
- **客户端感知** — 即时检测客户端断开（即使在退避等待期间），立即停止重试
- **热加载配置** — 无需重启即可添加/删除路由和模型
- **tool call 历史缓存** — 处理跨协议转换时 `previous_response_id` 的上下文恢复
- **Prometheus 指标** — 监控重试率和上游状态（按 route + model 维度）
- **单二进制** — 低内存占用，Rust 编写

## 快速开始

```bash
# 构建
make build

# 创建配置
cp config.example.toml config.toml
# 编辑 config.toml 指向你的 API

# 运行
./target/release/llm-proxy --config config.toml --log-level info
```

将 AI CLI 工具的 API URL 指向 `http://127.0.0.1:8888`（默认路由）或 `http://127.0.0.1:8888/{route_name}/{api_path}`（命名路由）。

## 配置

```toml
[defaults]
max_retries = 9999           # 实际无限
base_delay_ms = 1000         # 指数退避基数
max_delay_ms = 60000         # 退避上限
max_total_wait_ms = 0        # 0 = 依赖客户端断开
connect_timeout_secs = 30
retry_status_codes = [429, 500, 502, 503, 504, 408, 529]

# 命名路由：URL 路径以 /tkehub/ 开头时命中
[routes.tkehub]
target = "http://tkehub.woa.com"

# 默认路由：URL 不带路由名，model 字段决定上游
[routes.default]

# 每个模型有自己的 target、api_key 和支持的格式
[routes.default.models."my-glm"]
target = "http://tkehub.woa.com"
upstream_formats = ["responses", "chat"]  # 上游支持的格式
api_key = "${TKEHUB_API_KEY}"              # 环境变量展开
upstream_model = "tke/glm-latest"         # 请求模型名改写

[routes.default.models."my-deepseek"]
target = "https://tokenhub.tencentmaas.com"
upstream_formats = ["chat"]               # 仅支持 chat
api_key = "${DEEPSEEK_API_KEY}"
upstream_model = "deepseek-chat"
```

完整示例见 [config.example.toml](./config.example.toml)。

### 协议自动检测

代理根据 URL 路径自动检测入站协议：

| URL 路径 | 协议 |
|----------|------|
| `/v1/responses` | OpenAI Responses API |
| `/v1/chat/completions` | OpenAI Chat Completions |
| `/v1/messages` | Anthropic Messages |

### `upstream_formats`

声明上游支持的协议列表（按优先级排序），代理自动决策：

- 入站格式**在**列表中 → **直接透传**（不转换）
- 入站格式**不在**列表中 → **转换**为列表中的第一个格式
- 为空（未设置）→ 透传任意格式

替代了旧的 `transform` 字段，无需手动配置转换策略——代理自动推断转换方向。

### 默认路由

默认路由（`routes.default`）实现统一接入：所有 AI CLI 工具指向同一地址（`http://127.0.0.1:8888`），请求体中的 `model` 字段决定使用哪个上游。每个模型可以有自己的 target、api_key 和支持的格式。

命名路由与默认路由共存——先匹配命名路由（URL 第一段），未命中则走默认路由。

### Model 级配置字段

所有字段可选——仅指定的字段覆盖路由级配置。

| 字段 | 说明 |
|------|------|
| `target` | 上游 base URL（代理自动拼接标准 API 路径） |
| `upstream_formats` | 上游支持的协议列表，按优先级排序 |
| `api_key` | 上游 API key（支持 `${ENV_VAR}` 环境变量展开） |
| `upstream_model` | 改写请求中的 `model` 字段；响应自动回写为客户端原始模型名 |
| 重试参数 | `max_retries`、`base_delay_ms`、`max_delay_ms`、`max_total_wait_ms`、`connect_timeout_secs`、`retry_status_codes` |

### API Key 管理

当 model 或 route 级别配置了 `api_key` 时，代理会：
1. 剥离客户端的 `Authorization` header
2. 注入配置的 API key（支持 `${ENV_VAR}` 环境变量展开）
3. 转发给上游

未配置 `api_key` 时（如命名路由未设 api_key），客户端原始 `Authorization` header 原样转发（向后兼容）。

## 安装

```bash
make install
```

交互式安装二进制文件，可选安装 systemd/launchd 服务自启。

## 许可证

[MIT](./LICENSE)
