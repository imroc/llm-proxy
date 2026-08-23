// Bring crate into scope for tests
use llm_proxy::{config, metrics, server};

use arc_swap::ArcSwap;
use bytes::Bytes;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;

use http_body_util::BodyExt as _;

/// Mock upstream server that fails the first N requests, then returns 200.
struct MockUpstream {
    fail_count: u32,
    fail_status: u16,
    fail_headers: Vec<(String, String)>,
    fail_body: Vec<u8>,
    success_body: Vec<u8>,
    request_count: AtomicU32,
}

impl MockUpstream {
    fn new(fail_count: u32) -> Self {
        Self {
            fail_count,
            fail_status: 429,
            fail_headers: vec![],
            fail_body: br#"{"error":{"message":"Rate limit exceeded (mock)}}"#.to_vec(),
            success_body: br#"{"id":"chatcmpl-123","choices":[{"message":{"content":"Hello!"}}]}"#
                .to_vec(),
            request_count: AtomicU32::new(0),
        }
    }

    fn with_retry_after(mut self, secs: u64) -> Self {
        self.fail_headers
            .push(("retry-after".into(), secs.to_string()));
        self
    }

    fn with_fail_status(mut self, status: u16) -> Self {
        self.fail_status = status;
        self
    }
}

async fn start_mock_upstream(mock: Arc<MockUpstream>) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        loop {
            let (stream, _) = match listener.accept().await {
                Ok(v) => v,
                Err(_) => break,
            };

            let mock = mock.clone();
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let service = service_fn(move |req: hyper::Request<Incoming>| {
                    let mock = mock.clone();
                    async move {
                        let count = mock.request_count.fetch_add(1, Ordering::SeqCst) + 1;
                        let method = req.method().clone();

                        // Read and discard request body
                        let _ = req.into_body().collect().await;

                        if count <= mock.fail_count {
                            // Return error
                            let mut resp = hyper::Response::builder().status(mock.fail_status);
                            for (k, v) in &mock.fail_headers {
                                resp = resp.header(k, v);
                            }
                            let body = Bytes::from(mock.fail_body.clone());
                            Ok::<_, std::convert::Infallible>(
                                resp.body(http_body_util::Full::new(body).boxed()).unwrap(),
                            )
                        } else {
                            // Return success
                            let body = Bytes::from(mock.success_body.clone());
                            Ok::<_, std::convert::Infallible>(
                                hyper::Response::builder()
                                    .status(200)
                                    .header("content-type", "application/json")
                                    .body(http_body_util::Full::new(body).boxed())
                                    .unwrap(),
                            )
                        }
                    }
                });

                let _ = http1::Builder::new().serve_connection(io, service).await;
            });
        }
    });

    addr
}

/// Start a fake HTTP CONNECT proxy that records whether it received a
/// CONNECT request and always rejects the tunnel with 403.
async fn start_fake_connect_proxy() -> (SocketAddr, Arc<AtomicBool>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let saw_connect = Arc::new(AtomicBool::new(false));

    let saw_connect_clone = saw_connect.clone();
    tokio::spawn(async move {
        loop {
            let (stream, _) = match listener.accept().await {
                Ok(s) => s,
                Err(_) => break,
            };
            let saw_connect = saw_connect_clone.clone();
            tokio::spawn(async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut stream = stream;
                let mut buf = vec![0u8; 1024];
                let Ok(n) = stream.read(&mut buf).await else {
                    return;
                };
                let head = String::from_utf8_lossy(&buf[..n]);
                if head.starts_with("CONNECT") {
                    saw_connect.store(true, Ordering::SeqCst);
                }
                let _ = stream
                    .write_all(b"HTTP/1.1 403 Filtered\r\nContent-Length: 0\r\n\r\n")
                    .await;
            });
        }
    });

    (addr, saw_connect)
}

/// Start the proxy with a given config.
async fn start_proxy(
    config_str: &str,
) -> (
    SocketAddr,
    Arc<ArcSwap<config::Config>>,
    Arc<metrics::Metrics>,
) {
    let cfg: config::Config = toml::from_str(config_str).unwrap();
    let config = Arc::new(ArcSwap::from_pointee(cfg));
    let metrics = metrics::Metrics::new();

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let config_clone = config.clone();
    let metrics_clone = metrics.clone();
    let client_clone = client.clone();
    let addr_clone = addr;

    tokio::spawn(async move {
        let (_tx, rx) = tokio::sync::broadcast::channel::<()>(1);
        let _ = server::run(
            config_clone,
            metrics_clone,
            client_clone,
            addr_clone,
            "0.1.0",
            rx,
        )
        .await;
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    (addr, config, metrics)
}

async fn http_get(addr: SocketAddr, path: &str) -> (u16, String) {
    let resp = reqwest::get(format!("http://{}{}", addr, path))
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap();
    (status, body)
}

async fn http_post(addr: SocketAddr, path: &str, body: &str) -> (u16, String) {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{}{}", addr, path))
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap();
    (status, body)
}

#[tokio::test]
async fn test_retry_then_success() {
    let mock = Arc::new(MockUpstream::new(3));
    let mock_addr = start_mock_upstream(mock.clone()).await;

    let config_str = format!(
        r#"
[defaults]
max_retries = 9999
base_delay_ms = 10
max_delay_ms = 100
max_total_wait_ms = 0
connect_timeout_secs = 5
retry_status_codes = [429, 500, 502, 503, 504, 408, 529]

[routes.mock]
target = "http://{}"
"#,
        mock_addr
    );

    let (proxy_addr, _, _) = start_proxy(&config_str).await;

    let (status, body) = http_post(
        proxy_addr,
        "/mock/v1/chat/completions",
        r#"{"model":"test-model","messages":[]}"#,
    )
    .await;

    assert_eq!(status, 200, "should succeed after retries: body={}", body);
    assert!(body.contains("Hello!"));
    assert_eq!(mock.request_count.load(Ordering::SeqCst), 4); // 3 failures + 1 success
}

#[tokio::test]
async fn test_max_retries_exhausted() {
    let mock = Arc::new(MockUpstream::new(999)); // Always fail
    let mock_addr = start_mock_upstream(mock.clone()).await;

    let config_str = format!(
        r#"
[defaults]
max_retries = 3
base_delay_ms = 10
max_delay_ms = 50
max_total_wait_ms = 0
connect_timeout_secs = 5
retry_status_codes = [429]

[routes.mock]
target = "http://{}"
"#,
        mock_addr
    );

    let (proxy_addr, _, _) = start_proxy(&config_str).await;

    let (status, body) = http_post(
        proxy_addr,
        "/mock/v1/chat/completions",
        r#"{"model":"test","messages":[]}"#,
    )
    .await;

    assert_eq!(status, 502, "should return 502 after exhausting retries");
    assert!(body.contains("upstream_failed"));
}

#[tokio::test]
async fn test_non_retryable_status_passed_through() {
    let mock = Arc::new(MockUpstream::new(0).with_fail_status(400));
    // Override: make 400 the fail status, but it's not in retry codes
    let mock = Arc::new(MockUpstream {
        fail_count: 999,
        fail_status: 400,
        fail_headers: vec![],
        fail_body: br#"{"error":"bad request"}"#.to_vec(),
        success_body: br#"{"ok":true}"#.to_vec(),
        request_count: AtomicU32::new(0),
    });
    let mock_addr = start_mock_upstream(mock.clone()).await;

    let config_str = format!(
        r#"
[defaults]
max_retries = 9999
base_delay_ms = 10
max_delay_ms = 100
connect_timeout_secs = 5
retry_status_codes = [429, 500, 502, 503, 504, 408, 529]

[routes.mock]
target = "http://{}"
"#,
        mock_addr
    );

    let (proxy_addr, _, _) = start_proxy(&config_str).await;

    let (status, body) = http_post(
        proxy_addr,
        "/mock/v1/chat/completions",
        r#"{"model":"test","messages":[]}"#,
    )
    .await;

    assert_eq!(status, 400, "400 should be passed through without retry");
    assert!(body.contains("bad request"));
    assert_eq!(mock.request_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_unknown_route_404() {
    let config_str = r#"
[defaults]
max_retries = 9999
base_delay_ms = 10
max_delay_ms = 100

[routes.known]
target = "http://127.0.0.1:9999"
"#;

    let (proxy_addr, _, _) = start_proxy(config_str).await;

    let (status, body) = http_post(
        proxy_addr,
        "/unknown/v1/chat/completions",
        r#"{"model":"test","messages":[]}"#,
    )
    .await;

    assert_eq!(status, 404);
    assert!(body.contains("unknown_route"));
}

#[tokio::test]
async fn test_healthz() {
    let config_str = r#"
[defaults]
[routes.test]
target = "http://127.0.0.1:9999"
"#;

    let (proxy_addr, _, _) = start_proxy(config_str).await;

    let (status, body) = http_get(proxy_addr, "/healthz").await;

    assert_eq!(status, 200);
    assert!(body.contains("ok"));
    assert!(body.contains("0.1.0"));
    assert!(body.contains("test"));
}

#[tokio::test]
async fn test_metrics() {
    // Start a mock upstream that returns 200
    let mock = Arc::new(MockUpstream::new(0));
    let mock_addr = start_mock_upstream(mock.clone()).await;

    let config_str = format!(
        r#"
[defaults]
[routes.mock]
target = "http://{}"
"#,
        mock_addr
    );

    let (proxy_addr, _, _) = start_proxy(&config_str).await;

    // Make a request to generate metrics
    let _ = http_post(
        proxy_addr,
        "/mock/v1/chat/completions",
        r#"{"model":"test","messages":[]}"#,
    )
    .await;

    let (status, body) = http_get(proxy_addr, "/metrics").await;

    assert_eq!(status, 200);
    assert!(
        body.contains("proxy_"),
        "metrics should contain proxy_ prefix: {}",
        body
    );
}

#[tokio::test]
async fn test_retry_after_header() {
    let mock = Arc::new(MockUpstream::new(1).with_retry_after(1)); // 1 second retry-after
    let mock_addr = start_mock_upstream(mock.clone()).await;

    let config_str = format!(
        r#"
[defaults]
max_retries = 9999
base_delay_ms = 5000   # High base so backoff > retry-after
max_delay_ms = 10000
connect_timeout_secs = 5
retry_status_codes = [429]

[routes.mock]
target = "http://{}"
"#,
        mock_addr
    );

    let (proxy_addr, _, _) = start_proxy(&config_str).await;

    let start = std::time::Instant::now();
    let (status, _) = http_post(
        proxy_addr,
        "/mock/v1/chat/completions",
        r#"{"model":"test","messages":[]}"#,
    )
    .await;
    let elapsed = start.elapsed();

    assert_eq!(status, 200);
    // retry-after is 1s, backoff is ~5s. min(1s, 5s) = 1s.
    // So total wait should be around 1 second, not 5 seconds.
    assert!(
        elapsed < Duration::from_secs(3),
        "should use min(retry_after, backoff), elapsed: {:?}",
        elapsed
    );
}

#[tokio::test]
async fn test_network_error_retry() {
    // Use a port that's not listening to simulate network error
    let config_str = r#"
[defaults]
max_retries = 2
base_delay_ms = 10
max_delay_ms = 50
connect_timeout_secs = 1
retry_status_codes = [429, 500]

[routes.dead]
target = "http://127.0.0.1:1"
"#;

    let (proxy_addr, _, _) = start_proxy(config_str).await;

    let (status, body) = http_post(
        proxy_addr,
        "/dead/v1/chat/completions",
        r#"{"model":"test","messages":[]}"#,
    )
    .await;

    assert_eq!(status, 502);
    assert!(body.contains("upstream_failed"));
}

#[tokio::test]
async fn test_https_proxy_env_var_routes_upstream_via_proxy() {
    let (fake_proxy_addr, saw_connect) = start_fake_connect_proxy().await;

    // Must be set before the reqwest client is built inside start_proxy,
    // mirroring how the variable must exist at process startup.
    std::env::set_var("HTTPS_PROXY", format!("http://{}", fake_proxy_addr));

    let config_str = r#"
[defaults]
max_retries = 1
base_delay_ms = 10
max_delay_ms = 20
connect_timeout_secs = 1
retry_status_codes = [429, 500]

[routes.tls]
target = "https://upstream.example.test"
"#;
    let (proxy_addr, _, _) = start_proxy(config_str).await;

    let (status, body) = http_post(
        proxy_addr,
        "/tls/v1/chat/completions",
        r#"{"model":"test","messages":[]}"#,
    )
    .await;

    std::env::remove_var("HTTPS_PROXY");

    assert_eq!(status, 502);
    assert!(body.contains("upstream_failed"));
    assert!(
        saw_connect.load(Ordering::SeqCst),
        "upstream request should be routed through HTTPS_PROXY"
    );
}

#[tokio::test]
async fn test_max_total_wait_exceeded() {
    let mock = Arc::new(MockUpstream::new(999).with_retry_after(10));
    let mock_addr = start_mock_upstream(mock.clone()).await;

    let config_str = format!(
        r#"
[defaults]
max_retries = 9999
base_delay_ms = 100
max_delay_ms = 200
max_total_wait_ms = 500
connect_timeout_secs = 5
retry_status_codes = [429]

[routes.mock]
target = "http://{}"
"#,
        mock_addr
    );

    let (proxy_addr, _, _) = start_proxy(&config_str).await;

    let (status, body) = http_post(
        proxy_addr,
        "/mock/v1/chat/completions",
        r#"{"model":"test","messages":[]}"#,
    )
    .await;

    assert_eq!(
        status, 502,
        "should give up after total wait budget exceeded"
    );
    assert!(body.contains("upstream_failed"));
}

#[tokio::test]
async fn test_streaming_passthrough() {
    // Mock that returns 200 immediately with a body
    let mock = Arc::new(MockUpstream::new(0));
    let mock_addr = start_mock_upstream(mock.clone()).await;

    let config_str = format!(
        r#"
[defaults]
max_retries = 9999
base_delay_ms = 10
max_delay_ms = 100
connect_timeout_secs = 5
retry_status_codes = [429]

[routes.mock]
target = "http://{}"
"#,
        mock_addr
    );

    let (proxy_addr, _, _) = start_proxy(&config_str).await;

    let (status, body) = http_post(
        proxy_addr,
        "/mock/v1/chat/completions",
        r#"{"model":"test","messages":[]}"#,
    )
    .await;

    assert_eq!(status, 200);
    assert!(body.contains("Hello!"));
}
