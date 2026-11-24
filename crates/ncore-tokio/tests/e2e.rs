use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::{BodyExt, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use ncore::{
    Auth, CancelToken, Endpoint, Format, InboundRequest, PassthroughBody, PassthroughOptions,
    RetryPolicy,
};
use tokio::net::TcpListener;

const SSE: &[u8] = include_bytes!("../../ncore-core/tests/fixtures/openai_chat.sse");

/// What the fake provider received.
#[derive(Debug, Clone)]
struct Seen {
    path: String,
    auth: Option<String>,
    body: Bytes,
}

type Body = StreamBody<futures_util::stream::BoxStream<'static, Result<Frame<Bytes>, Infallible>>>;

// Serves `fail_first` 503s, then the SSE fixture in small delayed chunks.
async fn provider(fail_first: usize) -> (SocketAddr, Arc<Mutex<Vec<Seen>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen2 = seen.clone();
    tokio::spawn(async move {
        loop {
            let (sock, _) = listener.accept().await.unwrap();
            let seen = seen2.clone();
            tokio::spawn(async move {
                let svc = service_fn(move |req: Request<Incoming>| {
                    let seen = seen.clone();
                    async move {
                        let path = req.uri().to_string();
                        let auth = req
                            .headers()
                            .get("authorization")
                            .map(|v| v.to_str().unwrap().to_owned());
                        let body = req.into_body().collect().await.unwrap().to_bytes();
                        let n = {
                            let mut s = seen.lock().unwrap();
                            s.push(Seen { path, auth, body });
                            s.len()
                        };
                        let resp: Response<Body> = if n <= fail_first {
                            let b = futures_util::stream::iter([Ok(Frame::data(
                                Bytes::from_static(b"{\"error\":{\"message\":\"busy\"}}"),
                            ))]);
                            Response::builder()
                                .status(503)
                                .body(StreamBody::new(b.boxed()))
                                .unwrap()
                        } else {
                            let chunks: Vec<Bytes> =
                                SSE.chunks(17).map(Bytes::copy_from_slice).collect();
                            let s = futures_util::stream::iter(chunks).then(|c| async move {
                                tokio::time::sleep(Duration::from_millis(1)).await;
                                Ok(Frame::data(c))
                            });
                            Response::builder()
                                .header("content-type", "text/event-stream")
                                .body(StreamBody::new(s.boxed()))
                                .unwrap()
                        };
                        Ok::<_, Infallible>(resp)
                    }
                });
                let _ = http1::Builder::new()
                    .serve_connection(TokioIo::new(sock), svc)
                    .await;
            });
        }
    });
    (addr, seen)
}

fn chat_request(body: &str) -> InboundRequest {
    InboundRequest {
        method: "POST".into(),
        path_and_query: "/v1/chat/completions".into(),
        headers: vec![
            ("authorization".into(), "Bearer client-key".into()),
            ("content-type".into(), "application/json".into()),
        ],
        body: Bytes::copy_from_slice(body.as_bytes()),
    }
}

#[tokio::test]
async fn streams_through_real_http() {
    let (addr, seen) = provider(1).await;
    let endpoint = Endpoint::new(Format::OpenAiChat, format!("http://{addr}"))
        .with_auth(Auth::Bearer("server-key".into()));
    let policy = RetryPolicy {
        base_delay: Duration::from_millis(5),
        ..Default::default()
    };
    let core = ncore_tokio::core(Some(policy));
    let body = r#"{"model": "gpt-client", "stream": true, "stream_options": {"include_usage": true}, "messages": []}"#;
    let opts = PassthroughOptions {
        model: Some("gpt-x".into()),
        ..Default::default()
    };
    let resp = core
        .passthrough(&endpoint, chat_request(body), opts, &CancelToken::new())
        .await
        .unwrap();
    assert_eq!((resp.status, resp.attempts), (200, 2));

    let PassthroughBody::Stream(mut stream) = resp.body else {
        panic!("expected stream")
    };
    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        out.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(out, SSE);
    let obs = stream.observation();
    assert_eq!(obs.usage.cache_read, Some(32));
    assert_eq!(obs.stop_reason.as_deref(), Some("stop"));

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1].path, "/v1/chat/completions");
    assert_eq!(seen[1].auth.as_deref(), Some("Bearer server-key"));
    assert_eq!(
        seen[1].body.as_ref(),
        br#"{"model": "gpt-x", "stream": true, "stream_options": {"include_usage": true}, "messages": []}"#
    );
}

#[tokio::test]
#[ignore = "hits the real Anthropic API without a key"]
async fn live_tls_and_error_parsing() {
    let endpoint = Endpoint::new(Format::AnthropicMessages, "https://api.anthropic.com");
    let req = InboundRequest {
        method: "POST".into(),
        path_and_query: "/v1/messages".into(),
        headers: vec![
            ("anthropic-version".into(), "2023-06-01".into()),
            ("content-type".into(), "application/json".into()),
        ],
        body: Bytes::from_static(br#"{"model":"claude-haiku-5-5","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}"#),
    };
    let core = ncore_tokio::core(None);
    let resp = core
        .passthrough(
            &endpoint,
            req,
            PassthroughOptions::default(),
            &CancelToken::new(),
        )
        .await
        .unwrap();
    let PassthroughBody::Full { observation, .. } = resp.body else {
        panic!("expected full body")
    };
    assert_eq!(resp.status, 401);
    assert_eq!(
        observation.error.unwrap().kind,
        ncore::core::ErrorKind::Auth
    );
}
