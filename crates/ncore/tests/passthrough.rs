use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use futures::executor::block_on;
use futures::stream::{self, StreamExt};
use ncore::core::SseEvent;
use ncore::rt::BoxFuture;
use ncore::{
    CancelToken, Clock, Core, Endpoint, Error, EventAction, Hook, HttpRequest, HttpResponse,
    InboundRequest, Interests, Middleware, Next, Observed, PassthroughBody, PassthroughOptions,
    RequestCtx, ResponseHead, RetryPolicy, Transport, TransportError,
};

const ANTHROPIC_SSE: &[u8] = include_bytes!("../../ncore-core/tests/fixtures/anthropic.sse");

/// Scripted response: status, headers, body chunks.
type Scripted = (u16, Vec<(String, String)>, Vec<Bytes>);

/// Transport that replays scripted responses and records requests.
#[derive(Default)]
struct MockTransport {
    script: Mutex<Vec<Result<Scripted, TransportError>>>,
    seen: Mutex<Vec<HttpRequest>>,
}

impl MockTransport {
    fn new(script: Vec<Result<Scripted, TransportError>>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(script.into_iter().rev().collect()),
            seen: Mutex::default(),
        })
    }
}

impl Transport for MockTransport {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        self.seen.lock().unwrap().push(req);
        let next = self
            .script
            .lock()
            .unwrap()
            .pop()
            .expect("unexpected request");
        Box::pin(async move {
            let (status, headers, chunks) = next?;
            Ok(HttpResponse {
                status,
                headers,
                body: Box::pin(stream::iter(chunks.into_iter().map(Ok))),
            })
        })
    }
}

/// Clock that records requested sleeps and returns immediately.
#[derive(Default)]
struct MockClock(Mutex<Vec<Duration>>);

impl Clock for MockClock {
    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> {
        self.0.lock().unwrap().push(d);
        Box::pin(async {})
    }
}

#[derive(Default)]
struct Recorder(Mutex<Vec<Observed>>);

impl Hook for Recorder {
    fn name(&self) -> &str {
        "recorder"
    }

    fn on_observed(&self, _: &RequestCtx, ev: &Observed) {
        self.0.lock().unwrap().push(ev.clone());
    }
}

fn chunks(b: &[u8], n: usize) -> Vec<Bytes> {
    b.chunks(n).map(Bytes::copy_from_slice).collect()
}

fn anthropic_request(body: &str) -> InboundRequest {
    InboundRequest {
        method: "POST".into(),
        path_and_query: "/v1/messages".into(),
        headers: vec![("anthropic-version".into(), "2023-06-01".into())],
        body: Bytes::copy_from_slice(body.as_bytes()),
    }
}

fn collect(body: PassthroughBody) -> (Vec<u8>, ncore::Observation) {
    match body {
        PassthroughBody::Stream(mut s) => block_on(async {
            let mut out = Vec::new();
            while let Some(c) = s.next().await {
                out.extend_from_slice(&c.unwrap());
            }
            (out, s.observation())
        }),
        PassthroughBody::Full { bytes, observation } => (bytes.to_vec(), observation),
    }
}

#[test]
fn stream_is_forwarded_byte_for_byte() {
    let t = MockTransport::new(vec![Ok((200, vec![], chunks(ANTHROPIC_SSE, 13)))]);
    let rec = Arc::new(Recorder::default());
    let core = Core::builder(t.clone()).hook(rec.clone()).build();
    let resp = block_on(core.passthrough(
        &Endpoint::anthropic("k"),
        anthropic_request(r#"{"model":"m","stream":true,"messages":[]}"#),
        PassthroughOptions::default(),
        &CancelToken::new(),
    ))
    .unwrap();
    let (bytes, obs) = collect(resp.body);
    assert_eq!(bytes, ANTHROPIC_SSE);
    assert_eq!(obs.usage.input_total, Some(125));
    assert_eq!(obs.usage.output_total, Some(15));
    assert!(
        rec.0
            .lock()
            .unwrap()
            .iter()
            .any(|o| matches!(o, Observed::Stop(s) if s == "end_turn"))
    );
}

#[test]
fn response_model_is_rewritten_in_stream() {
    let t = MockTransport::new(vec![Ok((200, vec![], chunks(ANTHROPIC_SSE, 64)))]);
    let core = Core::builder(t.clone()).build();
    let opts = PassthroughOptions {
        model: Some("provider-model".into()),
        response_model: Some("client-model".into()),
        ..Default::default()
    };
    let resp = block_on(core.passthrough(
        &Endpoint::anthropic("k"),
        anthropic_request(r#"{"model":"client-model","stream":true}"#),
        opts,
        &CancelToken::new(),
    ))
    .unwrap();
    let sent = t.seen.lock().unwrap()[0].body.clone();
    assert_eq!(
        sent.as_ref(),
        br#"{"model":"provider-model","stream":true}"#
    );
    let (bytes, _) = collect(resp.body);
    let text = String::from_utf8(bytes).unwrap();
    let expected = String::from_utf8(ANTHROPIC_SSE.to_vec()).unwrap().replace(
        r#""model": "claude-opus-5-5""#,
        r#""model": "client-model""#,
    );
    assert_eq!(text, expected);
}

#[test]
fn retries_overload_then_succeeds() {
    let overloaded =
        br#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
    let ok = br#"{"id":"msg_1","model":"m","stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}"#;
    let t = MockTransport::new(vec![
        Err(TransportError::connect("refused")),
        Ok((
            529,
            vec![("retry-after".into(), "1".into())],
            chunks(overloaded, 8),
        )),
        Ok((200, vec![], chunks(ok, 8))),
    ]);
    let clock = Arc::new(MockClock::default());
    let core = Core::builder(t.clone())
        .retry(RetryPolicy::default(), clock.clone())
        .build();
    let resp = block_on(core.passthrough(
        &Endpoint::anthropic("k"),
        anthropic_request(r#"{"model":"m"}"#),
        PassthroughOptions::default(),
        &CancelToken::new(),
    ))
    .unwrap();
    assert_eq!((resp.status, resp.attempts), (200, 3));
    assert_eq!(
        *clock.0.lock().unwrap(),
        vec![Duration::from_millis(500), Duration::from_secs(1)]
    );
    let (bytes, obs) = collect(resp.body);
    assert_eq!(bytes, ok);
    assert_eq!(obs.usage.input_total, Some(3));
}

#[test]
fn endpoint_replays_default_to_allowed() {
    for endpoint in [
        Endpoint::new(ncore::Format::OpenAiChat, "http://local"),
        Endpoint::anthropic("k"),
        Endpoint::openai_chat("k"),
        Endpoint::openai_responses("k"),
        Endpoint::gemini("k"),
    ] {
        assert!(endpoint.allow_replays);
        assert!(
            !endpoint
                .without_replays()
                .with_header("x-test", "1")
                .allow_replays
        );
    }
}

#[test]
fn endpoint_without_replays_forwards_the_original_overload() {
    let body =
        br#"{"type":"error","error":{"type":"overloaded_error","message":"original overload"}}"#;
    let transport = MockTransport::new(vec![
        Ok((
            503,
            vec![("retry-after".into(), "0".into())],
            chunks(body, 7),
        )),
        Ok((200, vec![], vec![Bytes::from_static(b"must not be sent")])),
    ]);
    let clock = Arc::new(MockClock::default());
    let core = Core::builder(transport.clone())
        .retry(RetryPolicy::default(), clock.clone())
        .build();
    let response = block_on(core.passthrough(
        &Endpoint::anthropic("k").without_replays(),
        anthropic_request(r#"{"model":"m","stream":true}"#),
        PassthroughOptions::default(),
        &CancelToken::new(),
    ))
    .unwrap();
    assert_eq!((response.status, response.attempts), (503, 1));
    assert_eq!(response.headers, vec![("retry-after".into(), "0".into())]);
    let (bytes, observation) = collect(response.body);
    assert_eq!(bytes, body);
    assert_eq!(observation.error.unwrap().message, "original overload");
    let requests = transport.seen.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert!(clock.0.lock().unwrap().is_empty());
}

#[test]
fn endpoint_without_replays_returns_the_first_connection_error() {
    let transport = MockTransport::new(vec![
        Err(TransportError::connect("original refusal")),
        Ok((200, vec![], vec![])),
    ]);
    let clock = Arc::new(MockClock::default());
    let core = Core::builder(transport.clone())
        .retry(RetryPolicy::default(), clock.clone())
        .build();
    let error = block_on(core.passthrough(
        &Endpoint::anthropic("k").without_replays(),
        anthropic_request(r#"{"model":"m"}"#),
        PassthroughOptions::default(),
        &CancelToken::new(),
    ))
    .unwrap_err();
    assert!(
        matches!(error, Error::Transport(error) if error.connect && error.message == "original refusal")
    );
    assert_eq!(transport.seen.lock().unwrap().len(), 1);
    assert!(clock.0.lock().unwrap().is_empty());
}

#[test]
fn without_policy_error_is_forwarded() {
    let body = br#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#;
    let t = MockTransport::new(vec![Ok((429, vec![], chunks(body, 100)))]);
    let core = Core::builder(t).build();
    let resp = block_on(core.passthrough(
        &Endpoint::anthropic("k"),
        anthropic_request(r#"{"model":"m","stream":true}"#),
        PassthroughOptions::default(),
        &CancelToken::new(),
    ))
    .unwrap();
    assert_eq!(resp.status, 429);
    let (bytes, obs) = collect(resp.body);
    assert_eq!(bytes, body);
    assert_eq!(
        obs.error.unwrap().kind,
        ncore::core::ErrorKind::RateLimited
    );
}

#[test]
fn cancel_stops_the_stream() {
    let t = MockTransport::new(vec![Ok((200, vec![], chunks(ANTHROPIC_SSE, 50)))]);
    let core = Core::builder(t).build();
    let cancel = CancelToken::new();
    let resp = block_on(core.passthrough(
        &Endpoint::anthropic("k"),
        anthropic_request(r#"{"model":"m","stream":true}"#),
        PassthroughOptions::default(),
        &cancel,
    ))
    .unwrap();
    let PassthroughBody::Stream(mut s) = resp.body else {
        panic!("expected stream")
    };
    block_on(async {
        assert!(s.next().await.unwrap().is_ok());
        cancel.cancel();
        assert!(matches!(s.next().await, Some(Err(Error::Cancelled))));
        assert!(s.next().await.is_none());
    });
}

/// Hook that logs every call and rewrites what it is told to.
struct Tagger {
    name: &'static str,
    log: Arc<Mutex<Vec<String>>>,
    drop_pings: bool,
}

impl Tagger {
    fn new(name: &'static str, log: &Arc<Mutex<Vec<String>>>) -> Arc<Self> {
        Arc::new(Self {
            name,
            log: log.clone(),
            drop_pings: false,
        })
    }
}

impl Hook for Tagger {
    fn name(&self) -> &str {
        self.name
    }

    fn on_request<'a>(
        &'a self,
        _ctx: &'a RequestCtx,
        req: &'a mut HttpRequest,
    ) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move {
            self.log.lock().unwrap().push(format!("req:{}", self.name));
            req.headers.push(("x-tag".into(), self.name.into()));
            Ok(())
        })
    }

    fn on_response<'a>(
        &'a self,
        _ctx: &'a RequestCtx,
        head: &'a mut ResponseHead,
    ) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move {
            self.log.lock().unwrap().push(format!("resp:{}", self.name));
            head.headers.push(("x-seen-by".into(), self.name.into()));
            Ok(())
        })
    }

    fn on_stream_event(&self, _ctx: &RequestCtx, ev: &mut SseEvent) -> Result<EventAction, String> {
        if self.drop_pings && ev.event.as_deref() == Some("ping") {
            return Ok(EventAction::Drop);
        }
        if ev.event.as_deref() == Some("message_stop") {
            ev.set_data(r#"{"type":"message_stop","tagged":true}"#);
        }
        Ok(EventAction::Keep)
    }
}

#[test]
fn hooks_run_in_onion_order_and_edit_events() {
    let t = MockTransport::new(vec![Ok((200, vec![], chunks(ANTHROPIC_SSE, 7)))]);
    let log = Arc::new(Mutex::new(Vec::new()));
    let outer = Tagger::new("outer", &log);
    let inner = Arc::new(Tagger {
        drop_pings: true,
        ..Arc::try_unwrap(Tagger::new("inner", &log)).ok().unwrap()
    });
    let core = Core::builder(t.clone()).hook(outer).build();
    let ep = Endpoint::anthropic("k").with_hook(inner);
    let resp = block_on(core.passthrough(
        &ep,
        anthropic_request(r#"{"model":"m","stream":true}"#),
        PassthroughOptions::default(),
        &CancelToken::new(),
    ))
    .unwrap();
    assert_eq!(
        *log.lock().unwrap(),
        ["req:outer", "req:inner", "resp:inner", "resp:outer"]
    );
    let tags: Vec<_> = t.seen.lock().unwrap()[0]
        .headers
        .iter()
        .filter(|(k, _)| k == "x-tag")
        .map(|(_, v)| v.clone())
        .collect();
    assert_eq!(tags, ["outer", "inner"]);
    let (bytes, obs) = collect(resp.body);
    let text = String::from_utf8(bytes).unwrap();
    assert!(!text.contains("event: ping"));
    // The fixture may be checked out with CRLF on Windows; hooks preserve that framing.
    assert!(
        text.replace("\r\n", "\n").ends_with(
            "event: message_stop\ndata: {\"type\":\"message_stop\",\"tagged\":true}\n\n"
        )
    );
    assert_eq!(obs.stop_reason.as_deref(), Some("end_turn"));
}

/// Answers from a canned body, or forwards and retries once on 401 with a new key.
struct Mw {
    canned: Option<&'static [u8]>,
}

impl Middleware for Mw {
    fn name(&self) -> &str {
        "mw"
    }

    fn handle<'a>(
        &'a self,
        _ctx: &'a RequestCtx,
        mut req: HttpRequest,
        next: Next<'a>,
    ) -> BoxFuture<'a, Result<HttpResponse, Error>> {
        Box::pin(async move {
            if let Some(body) = self.canned {
                return Ok(HttpResponse {
                    status: 200,
                    headers: vec![],
                    body: Box::pin(stream::iter([Ok(Bytes::from_static(body))])),
                });
            }
            let resp = next.run(req.clone()).await?;
            if resp.status != 401 {
                return Ok(resp);
            }
            req.headers.retain(|(k, _)| k != "x-api-key");
            req.headers.push(("x-api-key".into(), "fresh".into()));
            next.run(req).await
        })
    }
}

#[test]
fn middleware_can_answer_without_network() {
    let t = MockTransport::new(vec![]);
    let body = br#"{"id":"m1","model":"cached","usage":{"input_tokens":1,"output_tokens":1}}"#;
    let core = Core::builder(t.clone())
        .middleware(Arc::new(Mw { canned: Some(body) }))
        .build();
    let resp = block_on(core.passthrough(
        &Endpoint::anthropic("k"),
        anthropic_request(r#"{"model":"m"}"#),
        PassthroughOptions::default(),
        &CancelToken::new(),
    ))
    .unwrap();
    let (bytes, obs) = collect(resp.body);
    assert_eq!(bytes, body);
    assert_eq!(obs.model.as_deref(), Some("cached"));
    assert!(t.seen.lock().unwrap().is_empty());
}

#[test]
fn middleware_can_retry_with_new_auth() {
    let ok = br#"{"id":"m1","model":"m"}"#;
    let t = MockTransport::new(vec![
        Ok((401, vec![], chunks(b"{}", 2))),
        Ok((200, vec![], chunks(ok, 4))),
    ]);
    let ep = Endpoint::anthropic("stale").with_middleware(Arc::new(Mw { canned: None }));
    let core = Core::builder(t.clone()).build();
    let resp = block_on(core.passthrough(
        &ep,
        anthropic_request(r#"{"model":"m"}"#),
        PassthroughOptions::default(),
        &CancelToken::new(),
    ))
    .unwrap();
    assert_eq!((resp.status, resp.attempts), (200, 1));
    let seen = t.seen.lock().unwrap();
    assert_eq!(
        ncore::rt::header(&seen[1].headers, "x-api-key"),
        Some("fresh")
    );
}

struct Failing;

impl Hook for Failing {
    fn name(&self) -> &str {
        "failing"
    }

    fn interests(&self) -> Interests {
        Interests {
            stream: true,
            ..Interests::NONE
        }
    }

    fn on_request<'a>(
        &'a self,
        _ctx: &'a RequestCtx,
        _req: &'a mut HttpRequest,
    ) -> BoxFuture<'a, Result<(), String>> {
        panic!("not declared, must not be called")
    }

    fn on_stream_event(&self, _ctx: &RequestCtx, ev: &mut SseEvent) -> Result<EventAction, String> {
        if ev.event.as_deref() == Some("content_block_delta") {
            return Err("bad delta".into());
        }
        Ok(EventAction::Keep)
    }
}

#[test]
fn hook_error_ends_stream_after_flushing() {
    let t = MockTransport::new(vec![Ok((200, vec![], chunks(ANTHROPIC_SSE, 4096)))]);
    let core = Core::builder(t).hook(Arc::new(Failing)).build();
    let resp = block_on(core.passthrough(
        &Endpoint::anthropic("k"),
        anthropic_request(r#"{"model":"m","stream":true}"#),
        PassthroughOptions::default(),
        &CancelToken::new(),
    ))
    .unwrap();
    let PassthroughBody::Stream(mut s) = resp.body else {
        panic!("expected stream")
    };
    block_on(async {
        let first = s.next().await.unwrap().unwrap();
        let text = String::from_utf8(first.to_vec()).unwrap();
        assert!(text.contains("event: ping") && !text.contains("content_block_delta"));
        assert!(matches!(
            s.next().await,
            Some(Err(Error::Plugin { name, .. })) if name == "failing"
        ));
        assert!(s.next().await.is_none());
    });
}
