use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use futures::StreamExt;
use futures::stream;
use ncore::core::cache::BreakReason;
use ncore::core::ir::{
    Block, Message, Request, Role, StopReason, StreamEvent, ToolCall, ToolDef, ToolResult,
};
use ncore::rt::{BoxFuture, Clock};
use ncore::{
    Agent, AgentHook, CacheTracker, CancelToken, Core, Endpoint, Error, HttpRequest, HttpResponse,
    LoopEvent, RetryPolicy, RunOptions, RunStop, ToolDecision, Tools, Transport, TransportError,
};
use serde_json::{Value, json};

struct TokioClock;

impl Clock for TokioClock {
    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> {
        Box::pin(tokio::time::sleep(d))
    }
}

/// One scripted provider reply.
enum Reply {
    Sse(String),
    /// Sends the events, then never finishes.
    SseHang(String),
    Json(u16, Value),
}

#[derive(Default)]
struct Mock {
    script: Mutex<Vec<Reply>>,
    seen: Mutex<Vec<Value>>,
}

impl Mock {
    fn new(script: Vec<Reply>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(script.into_iter().rev().collect()),
            seen: Mutex::default(),
        })
    }

    fn requests(&self) -> Vec<Value> {
        self.seen.lock().unwrap().clone()
    }
}

fn chunks(s: String) -> Vec<Result<Bytes, TransportError>> {
    // Small chunks split events and UTF-8 characters.
    s.into_bytes()
        .chunks(7)
        .map(|c| Ok(Bytes::copy_from_slice(c)))
        .collect()
}

impl Transport for Mock {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        self.seen
            .lock()
            .unwrap()
            .push(serde_json::from_slice(&req.body).unwrap());
        let reply = self
            .script
            .lock()
            .unwrap()
            .pop()
            .expect("unexpected request");
        Box::pin(async move {
            let (status, body): (u16, ncore::rt::BoxStream<'static, _>) = match reply {
                Reply::Sse(s) => (200, Box::pin(stream::iter(chunks(s)))),
                Reply::SseHang(s) => (
                    200,
                    Box::pin(stream::iter(chunks(s)).chain(stream::pending())),
                ),
                Reply::Json(status, v) => (status, Box::pin(stream::iter(chunks(v.to_string())))),
            };
            Ok(HttpResponse {
                status,
                headers: Vec::new(),
                body,
            })
        })
    }
}

fn ev(v: Value) -> String {
    format!("event: {}\ndata: {v}\n\n", v["type"].as_str().unwrap())
}

fn start() -> String {
    start_cached(0, 0)
}

fn start_cached(read: u64, write: u64) -> String {
    ev(
        json!({"type": "message_start", "message": {"id": "msg", "type": "message", "role": "assistant",
        "model": "m", "content": [], "stop_reason": null, "usage": {"input_tokens": 10, "output_tokens": 1,
        "cache_read_input_tokens": read, "cache_creation_input_tokens": write}}}),
    )
}

fn cached_turn(text: &str, read: u64, write: u64) -> Reply {
    Reply::Sse(start_cached(read, write) + &text_block(0, text) + &end("end_turn"))
}

fn text_block(i: usize, text: &str) -> String {
    ev(
        json!({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}),
    ) + &ev(
        json!({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": text}}),
    ) + &ev(json!({"type": "content_block_stop", "index": i}))
}

fn tool_block(i: usize, id: &str, name: &str, args: &str) -> String {
    ev(json!({"type": "content_block_start", "index": i,
        "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}}))
        + &ev(json!({"type": "content_block_delta", "index": i,
            "delta": {"type": "input_json_delta", "partial_json": args}}))
        + &ev(json!({"type": "content_block_stop", "index": i}))
}

fn end(stop: &str) -> String {
    ev(
        json!({"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 5}}),
    ) + &ev(json!({"type": "message_stop"}))
}

fn text_turn(text: &str) -> Reply {
    Reply::Sse(start() + &text_block(0, text) + &end("end_turn"))
}

fn tools_turn(calls: &[(&str, &str, &str)], stop: &str) -> Reply {
    let mut s = start();
    for (i, (id, name, args)) in calls.iter().enumerate() {
        s += &tool_block(i, id, name, args);
    }
    Reply::Sse(s + &end(stop))
}

/// In-process tools with scripted behavior.
struct Fake;

impl Tools for Fake {
    fn definitions(&self) -> Vec<ToolDef> {
        ["add", "hang", "polite"]
            .into_iter()
            .map(|n| {
                ToolDef::Function(ncore::core::ir::Tool {
                    name: n.into(),
                    description: None,
                    schema: json!({"type": "object"}),
                    strict: None,
                    cache: None,
                    ext: Default::default(),
                })
            })
            .collect()
    }

    fn call<'a>(
        &'a self,
        call: &'a ToolCall,
        cancel: &'a CancelToken,
    ) -> BoxFuture<'a, ToolResult> {
        Box::pin(async move {
            let args: Value = serde_json::from_str(&call.args).unwrap_or(Value::Null);
            let (text, is_error) = match call.name.as_str() {
                "add" => {
                    let ms = args["ms"].as_u64().unwrap_or(0);
                    tokio::time::sleep(Duration::from_millis(ms)).await;
                    let sum = args["a"].as_i64().unwrap_or(0) + args["b"].as_i64().unwrap_or(0);
                    (sum.to_string(), false)
                }
                // Ignores cancellation completely.
                "hang" => std::future::pending().await,
                // Stops as soon as it is told to.
                "polite" => {
                    cancel.cancelled().await;
                    ("stopped".to_owned(), true)
                }
                other => (format!("unknown tool {other}"), true),
            };
            ToolResult {
                call_id: call.id.clone(),
                name: None,
                content: vec![Block::text(text)],
                is_error,
                cache: None,
            }
        })
    }
}

fn request(stream: bool) -> Request {
    Request {
        model: "m".into(),
        max_tokens: Some(100),
        messages: vec![Message::new(Role::User, vec![Block::text("go")])],
        stream,
        ..Default::default()
    }
}

fn core(mock: &Arc<Mock>, retry: bool) -> Core {
    let mut b = Core::builder(mock.clone());
    if retry {
        b = b.retry(
            RetryPolicy {
                max_attempts: 3,
                base_delay: Duration::from_millis(10),
                ..Default::default()
            },
            Arc::new(TokioClock),
        );
    }
    b.build()
}

fn endpoint() -> Endpoint {
    Endpoint::anthropic("k")
}

fn results(m: &Message) -> Vec<(String, String, bool)> {
    m.content
        .iter()
        .map(|b| match b {
            Block::ToolResult(r) => (
                r.call_id.clone(),
                ncore::core::ir::plain_text(&r.content),
                r.is_error,
            ),
            other => panic!("not a tool result: {other:?}"),
        })
        .collect()
}

fn call_ids(m: &Message) -> Vec<String> {
    m.content
        .iter()
        .filter_map(|b| match b {
            Block::ToolCall(c) => Some(c.id.clone()),
            _ => None,
        })
        .collect()
}

fn cancel_after(token: &CancelToken, d: Duration) {
    let t = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(d).await;
        t.cancel();
    });
}

#[tokio::test(start_paused = true)]
async fn multi_step_with_parallel_tools_in_call_order() {
    let mock = Mock::new(vec![
        tools_turn(
            &[
                ("t1", "add", r#"{"a": 1, "b": 2, "ms": 50}"#),
                ("t2", "add", r#"{"a": 5, "b": 10}"#),
            ],
            "tool_use",
        ),
        text_turn("готово"),
    ]);
    let core = core(&mock, false);
    let ep = endpoint();
    let agent = Agent::new(&core, &ep, Arc::new(TokioClock)).tools(&Fake);
    let mut run = agent.run(request(true), RunOptions::default());
    let mut events = Vec::new();
    while let Some(e) = run.next().await {
        events.push(e);
    }
    let Some(LoopEvent::Done(out)) = events.pop() else {
        panic!("no Done")
    };
    assert!(matches!(
        out.stop,
        RunStop::Finished(Some(StopReason::EndTurn))
    ));
    assert_eq!(out.steps, 2);
    assert_eq!(out.messages.len(), 3);
    assert_eq!(
        results(&out.messages[1]),
        [
            ("t1".into(), "3".into(), false),
            ("t2".into(), "15".into(), false)
        ]
    );
    assert_eq!(
        ncore::core::ir::plain_text(&out.messages[2].content),
        "готово"
    );
    assert_eq!(out.usage.input_total, Some(20));
    assert_eq!(out.usage.output_total, Some(10));

    // The faster call finishes first, but the history keeps call order.
    let ends: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            LoopEvent::ToolEnd { result, .. } => Some(result.call_id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(ends, ["t2", "t1"]);
    assert!(matches!(events[0], LoopEvent::StepStart { step: 1 }));
    assert!(events.iter().any(|e| matches!(
        e,
        LoopEvent::Stream {
            step: 2,
            event: StreamEvent::Delta { .. }
        }
    )));

    let reqs = mock.requests();
    assert_eq!(reqs.len(), 2);
    let tools: Vec<_> = reqs[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(tools, ["add", "hang", "polite"]);
    let msgs = reqs[1]["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[1]["content"][1]["id"], "t2");
    assert_eq!(msgs[2]["content"][0]["tool_use_id"], "t1");
    assert_eq!(msgs[2]["content"][1]["tool_use_id"], "t2");
    // The last tool and the conversation tail carry the breakpoints.
    let mark = json!({"type": "ephemeral"});
    assert_eq!(reqs[0]["tools"][2]["cache_control"], mark);
    assert_eq!(reqs[0]["messages"][0]["content"][0]["cache_control"], mark);
    assert_eq!(msgs[2]["content"][1]["cache_control"], mark);
    assert!(msgs[0]["content"][0].get("cache_control").is_none());
    // The second request starts with the first one up to the moved marker.
    assert_eq!(reqs[0]["tools"], reqs[1]["tools"]);
    let mut first = reqs[0]["messages"][0].clone();
    first["content"][0]
        .as_object_mut()
        .unwrap()
        .remove("cache_control");
    assert_eq!(first, msgs[0]);
}

#[tokio::test(start_paused = true)]
async fn cancel_during_tools_answers_every_call() {
    let mock = Mock::new(vec![tools_turn(
        &[
            ("a", "hang", "{}"),
            ("b", "polite", "{}"),
            ("c", "add", r#"{"a": 2, "b": 2}"#),
        ],
        "tool_use",
    )]);
    let core = core(&mock, false);
    let ep = endpoint();
    let agent = Agent::new(&core, &ep, Arc::new(TokioClock)).tools(&Fake);
    let opts = RunOptions::default();
    cancel_after(&opts.cancel, Duration::from_secs(1));
    let started = tokio::time::Instant::now();
    let out = agent.run(request(true), opts).complete().await;
    assert!(matches!(out.stop, RunStop::Cancelled));
    assert_eq!(out.messages.len(), 2);
    let r = results(&out.messages[1]);
    assert_eq!(
        r[0],
        (
            "a".into(),
            "the call was cancelled by the user".into(),
            true
        )
    );
    assert_eq!(
        r[1],
        (
            "b".into(),
            "the call was cancelled by the user".into(),
            true
        )
    );
    assert_eq!(r[2], ("c".into(), "4".into(), false));
    // The tool that ignores cancellation is abandoned after the grace period.
    assert!(started.elapsed() <= Duration::from_secs(4));
    assert_eq!(mock.requests().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn cancel_mid_stream_keeps_history_valid() {
    let mock = Mock::new(vec![
        tools_turn(&[("t1", "add", r#"{"a": 1}"#)], "tool_use"),
        Reply::SseHang(
            start()
                + &ev(json!({"type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}}))
                + &ev(json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "Наполовину"}})),
        ),
    ]);
    let core = core(&mock, true);
    let ep = endpoint();
    let agent = Agent::new(&core, &ep, Arc::new(TokioClock)).tools(&Fake);
    let opts = RunOptions::default();
    cancel_after(&opts.cancel, Duration::from_secs(5));
    let out = agent.run(request(true), opts).complete().await;
    assert!(matches!(out.stop, RunStop::Cancelled));
    // Assistant call and its results; the unfinished turn is not in the history.
    assert_eq!(out.messages.len(), 2);
    assert_eq!(out.messages[1].role, Role::User);
    let partial = out.partial.expect("partial output");
    assert_eq!(
        ncore::core::ir::plain_text(&partial.content),
        "Наполовину"
    );
}

#[tokio::test(start_paused = true)]
async fn broken_streams_are_restarted() {
    let cut = Reply::Sse(start() + &text_block(0, "обры"));
    let overloaded = Reply::Sse(
        start()
            + &ev(
                json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
            ),
    );
    let mock = Mock::new(vec![cut, overloaded, text_turn("целиком")]);
    let core = core(&mock, true);
    let ep = endpoint();
    let agent = Agent::new(&core, &ep, Arc::new(TokioClock));
    let mut run = agent.run(request(true), RunOptions::default());
    let mut restarts = Vec::new();
    let mut out = None;
    while let Some(e) = run.next().await {
        match e {
            LoopEvent::StreamRestart { attempt, error, .. } => restarts.push((attempt, error)),
            LoopEvent::Done(o) => out = Some(o),
            _ => {}
        }
    }
    let out = out.unwrap();
    assert!(matches!(
        out.stop,
        RunStop::Finished(Some(StopReason::EndTurn))
    ));
    assert_eq!(out.messages.len(), 1);
    assert_eq!(
        ncore::core::ir::plain_text(&out.messages[0].content),
        "целиком"
    );
    assert_eq!(restarts.len(), 2);
    assert!(
        matches!(&restarts[0].1, Error::Transport(t) if t.message.contains("before the response was complete"))
    );
    assert!(matches!(&restarts[1].1, Error::Provider(p) if p.message == "Overloaded"));

    // Without a retry policy the run fails and keeps what was streamed aside.
    let mock = Mock::new(vec![Reply::Sse(start() + &text_block(0, "обры"))]);
    let core = self::core(&mock, false);
    let agent = Agent::new(&core, &ep, Arc::new(TokioClock));
    let out = agent
        .run(request(true), RunOptions::default())
        .complete()
        .await;
    assert!(matches!(out.stop, RunStop::Failed(Error::Transport(_))));
    assert!(out.messages.is_empty());
    assert_eq!(
        ncore::core::ir::plain_text(&out.partial.unwrap().content),
        "обры"
    );
}

#[tokio::test(start_paused = true)]
async fn endpoint_without_replays_keeps_the_first_broken_turn() {
    let mock = Mock::new(vec![
        Reply::Sse(start() + &text_block(0, "partial")),
        text_turn("must not be sent"),
    ]);
    let core = core(&mock, true);
    let endpoint = endpoint().without_replays();
    let agent = Agent::new(&core, &endpoint, Arc::new(TokioClock));
    let options = RunOptions::default();
    assert!(options.restart_streams);
    let events: Vec<_> = agent.run(request(true), options).collect().await;
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, LoopEvent::StreamRestart { .. }))
    );
    let Some(LoopEvent::Done(output)) = events.last() else {
        panic!("missing output")
    };
    assert!(
        matches!(&output.stop, RunStop::Failed(Error::Transport(error)) if error.message.contains("before the response was complete"))
    );
    assert!(output.messages.is_empty());
    assert_eq!(
        ncore::core::ir::plain_text(&output.partial.as_ref().unwrap().content),
        "partial"
    );
    assert_eq!(mock.requests().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn provider_error_keeps_earlier_steps() {
    let mock = Mock::new(vec![
        tools_turn(&[("t1", "add", "{}")], "tool_use"),
        Reply::Json(
            400,
            json!({"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long"}}),
        ),
    ]);
    let core = core(&mock, true);
    let ep = endpoint();
    let agent = Agent::new(&core, &ep, Arc::new(TokioClock)).tools(&Fake);
    let out = agent
        .run(request(true), RunOptions::default())
        .complete()
        .await;
    match &out.stop {
        RunStop::Failed(Error::Provider(p)) => {
            assert_eq!(p.status, Some(400));
            assert_eq!(p.message, "prompt is too long");
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(out.messages.len(), 2);
    assert!(out.partial.is_none());
}

#[tokio::test(start_paused = true)]
async fn host_can_run_the_tools_itself() {
    let mock = Mock::new(vec![
        tools_turn(&[("t1", "add", r#"{"a": 1, "b": 1}"#)], "tool_use"),
        text_turn("ok"),
    ]);
    let core = core(&mock, false);
    let ep = endpoint();
    let agent = Agent::new(&core, &ep, Arc::new(TokioClock)).tools(&Fake);
    let opts = RunOptions {
        execute_tools: false,
        ..Default::default()
    };
    let out = agent.run(request(true), opts.clone()).complete().await;
    let RunStop::ToolCalls(calls) = &out.stop else {
        panic!("unexpected {:?}", out.stop)
    };
    assert_eq!(calls[0].id, "t1");
    assert_eq!(out.messages.len(), 1);

    // A history ending in unanswered calls is never sent as is.
    let mut req = request(true);
    req.messages.extend(out.messages);
    let again = agent.run(req.clone(), opts).complete().await;
    assert!(matches!(again.stop, RunStop::ToolCalls(_)));
    assert_eq!(mock.requests().len(), 1);

    // With execution on, the pending calls run before the next model turn.
    let out = agent.run(req, RunOptions::default()).complete().await;
    assert!(matches!(out.stop, RunStop::Finished(_)));
    assert_eq!(
        results(&out.messages[0]),
        [("t1".into(), "2".into(), false)]
    );
    let reqs = mock.requests();
    assert_eq!(reqs[1]["messages"][2]["content"][0]["tool_use_id"], "t1");
}

struct Gate;

impl AgentHook for Gate {
    fn before_step<'a>(
        &'a self,
        step: u32,
        req: &'a mut Request,
    ) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move {
            req.system = vec![Block::text(format!("step {step}"))];
            Ok(())
        })
    }

    fn on_tool_call<'a>(&'a self, call: &'a ToolCall) -> BoxFuture<'a, ToolDecision> {
        Box::pin(async move {
            match call.args.as_str() {
                r#"{"deny": true}"# => ToolDecision::Deny("not allowed".into()),
                r#"{"replace": true}"# => ToolDecision::Replace(ToolResult {
                    call_id: "wrong".into(),
                    name: None,
                    content: vec![Block::text("from the host")],
                    is_error: false,
                    cache: None,
                }),
                _ => ToolDecision::Allow,
            }
        })
    }

    fn on_tool_result<'a>(
        &'a self,
        _call: &'a ToolCall,
        result: &'a mut ToolResult,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            if let Some(Block::Text { text, .. }) = result.content.first_mut() {
                text.push('!');
            }
        })
    }
}

#[tokio::test(start_paused = true)]
async fn hooks_truncated_calls_timeouts_and_ids() {
    let mock = Mock::new(vec![
        tools_turn(
            &[
                ("", "add", r#"{"a": 1, "b": 1}"#),
                ("x", "add", r#"{"deny": true}"#),
                ("x", "add", r#"{"replace": true}"#),
                ("y", "hang", "{}"),
                ("z", "add", r#"{"a": 1, "b"#),
            ],
            "max_tokens",
        ),
        text_turn("ok"),
    ]);
    let core = core(&mock, false);
    let ep = endpoint();
    let agent = Agent::new(&core, &ep, Arc::new(TokioClock))
        .tools(&Fake)
        .hook(Arc::new(Gate));
    let opts = RunOptions {
        tool_timeout: Some(Duration::from_secs(5)),
        max_parallel_tools: Some(2),
        ..Default::default()
    };
    let out = agent.run(request(true), opts).complete().await;
    assert!(matches!(out.stop, RunStop::Finished(_)));
    assert_eq!(
        call_ids(&out.messages[0]),
        ["call_1_0", "x", "x_2", "y", "z"]
    );
    let r = results(&out.messages[1]);
    assert_eq!(r[0], ("call_1_0".into(), "2!".into(), false));
    assert_eq!(
        r[1],
        ("x".into(), "the call was denied: not allowed!".into(), true)
    );
    assert_eq!(r[2], ("x_2".into(), "from the host!".into(), false));
    assert_eq!(
        r[3],
        ("y".into(), "the call timed out after 5s!".into(), true)
    );
    assert!(r[4].1.contains("cut off by the output token limit") && r[4].2);

    let reqs = mock.requests();
    assert_eq!(reqs[0]["system"][0]["text"], "step 1");
    assert_eq!(reqs[1]["system"][0]["text"], "step 2");
    // The truncated arguments still encode as an object.
    assert!(reqs[1]["messages"][1]["content"][4]["input"].is_object());
}

#[tokio::test(start_paused = true)]
async fn limits_pauses_and_empty_turns() {
    let mock = Mock::new(vec![tools_turn(&[("t1", "add", "{}")], "tool_use")]);
    let core = core(&mock, false);
    let ep = endpoint();
    let agent = Agent::new(&core, &ep, Arc::new(TokioClock)).tools(&Fake);
    let out = agent
        .step(request(true), RunOptions::default())
        .complete()
        .await;
    assert!(matches!(out.stop, RunStop::MaxSteps));
    assert_eq!((out.steps, out.messages.len()), (1, 2));

    let pause = Reply::Sse(start() + &text_block(0, "ищу") + &end("pause_turn"));
    let empty = Reply::Sse(start() + &end("end_turn"));
    let mock = Mock::new(vec![pause, empty]);
    let core = self::core(&mock, false);
    let agent = Agent::new(&core, &ep, Arc::new(TokioClock));
    let out = agent
        .run(request(true), RunOptions::default())
        .complete()
        .await;
    assert!(matches!(
        out.stop,
        RunStop::Finished(Some(StopReason::EndTurn))
    ));
    // The paused turn is kept and continued; the empty answer is not added.
    assert_eq!(out.messages.len(), 1);
    assert_eq!(out.steps, 2);
    let reqs = mock.requests();
    assert_eq!(reqs[1]["messages"][1]["role"], "assistant");

    let cancel = CancelToken::new();
    cancel.cancel();
    let out = agent
        .run(
            request(true),
            RunOptions {
                cancel,
                ..Default::default()
            },
        )
        .complete()
        .await;
    assert!(matches!(out.stop, RunStop::Cancelled));
    assert_eq!(out.steps, 0);
}

#[tokio::test(start_paused = true)]
async fn non_streaming_requests() {
    let mock = Mock::new(vec![
        Reply::Json(
            200,
            json!({"id": "msg", "type": "message", "role": "assistant", "model": "m",
                "content": [{"type": "tool_use", "id": "t1", "name": "add", "input": {"a": 2, "b": 3}}],
                "stop_reason": "tool_use", "usage": {"input_tokens": 3, "output_tokens": 4}}),
        ),
        Reply::Json(
            200,
            json!({"id": "msg2", "type": "message", "role": "assistant", "model": "m",
                "content": [{"type": "text", "text": "5"}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 4}}),
        ),
    ]);
    let core = core(&mock, false);
    let ep = endpoint();
    let agent = Agent::new(&core, &ep, Arc::new(TokioClock)).tools(&Fake);
    let mut run = agent.run(request(false), RunOptions::default());
    let mut deltas = 0;
    let mut out = None;
    while let Some(e) = run.next().await {
        match e {
            LoopEvent::Stream {
                event: StreamEvent::Delta { .. },
                ..
            } => deltas += 1,
            LoopEvent::Done(o) => out = Some(o),
            _ => {}
        }
    }
    let out = out.unwrap();
    assert!(matches!(
        out.stop,
        RunStop::Finished(Some(StopReason::EndTurn))
    ));
    assert_eq!(
        results(&out.messages[1]),
        [("t1".into(), "5".into(), false)]
    );
    assert_eq!(deltas, 2);
    assert_eq!(out.usage.output_total, Some(8));
}

#[tokio::test(start_paused = true)]
async fn every_format_detects_cut_streams() {
    use ncore::core::Format;
    let chat = |finish: &str| {
        format!(
            "data: {}\n\n",
            json!({"id": "c", "object": "chat.completion.chunk", "model": "m",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": null}]})
        ) + &if finish.is_empty() {
            String::new()
        } else {
            format!(
                "data: {}\n\n",
                json!({"id": "c", "object": "chat.completion.chunk", "model": "m",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            )
        }
    };
    let gemini = |finish: Option<&str>| {
        let mut c = json!({"content": {"role": "model", "parts": [{"text": "hi"}]}});
        if let Some(f) = finish {
            c["finishReason"] = json!(f);
        }
        format!(
            "data: {}\n\n",
            json!({"candidates": [c], "modelVersion": "m"})
        )
    };
    let responses_start = format!(
        "event: response.created\ndata: {}\n\n",
        json!({"type": "response.created", "response": {"id": "r", "model": "m", "output": []}})
    );
    let cases = [
        (Format::OpenAiChat, chat(""), false),
        (Format::OpenAiChat, chat("stop"), true),
        (Format::OpenAiChat, chat("stop") + "data: [DONE]\n\n", true),
        (Format::Gemini, gemini(None), false),
        (Format::Gemini, gemini(Some("STOP")), true),
        (Format::OpenAiResponses, responses_start, false),
        (
            Format::AnthropicMessages,
            start() + &text_block(0, "hi"),
            false,
        ),
        (
            Format::AnthropicMessages,
            start() + &text_block(0, "hi") + &end("end_turn"),
            true,
        ),
        // An event without its closing blank line does not count.
        (
            Format::AnthropicMessages,
            start() + &text_block(0, "hi") + end("end_turn").trim_end(),
            false,
        ),
    ];
    for (i, (format, body, complete)) in cases.into_iter().enumerate() {
        let mock = Mock::new(vec![Reply::Sse(body)]);
        let core = core(&mock, false);
        let ep = Endpoint::new(format, "http://x").with_auth(ncore::Auth::Bearer("k".into()));
        let agent = Agent::new(&core, &ep, Arc::new(TokioClock));
        let out = agent
            .run(request(true), RunOptions::default())
            .complete()
            .await;
        match (&out.stop, complete) {
            (RunStop::Finished(_), true) => {
                assert_eq!(
                    ncore::core::ir::plain_text(&out.messages[0].content),
                    "hi"
                )
            }
            (RunStop::Failed(Error::Transport(_)), false) => {}
            (other, _) => panic!("case {i} {format:?}: {other:?}"),
        }
    }
}

/// Edits the first message of the history before step 2.
struct Rewrite;

impl AgentHook for Rewrite {
    fn before_step<'a>(
        &'a self,
        step: u32,
        req: &'a mut Request,
    ) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move {
            if step == 2 {
                req.messages[0].content[0] = Block::text("rewritten");
            }
            Ok(())
        })
    }
}

#[tokio::test(start_paused = true)]
async fn cache_breaks_and_misses_are_reported() {
    let mock = Mock::new(vec![
        tools_turn(&[("t1", "add", r#"{"a": 1, "b": 2}"#)], "tool_use"),
        cached_turn("3", 4000, 50),
        // Same prefix, nothing read: expired or another node.
        cached_turn("again", 0, 4100),
        // The host changed the system prompt.
        cached_turn("new rules", 0, 4200),
        tools_turn(&[("t2", "add", "{}")], "tool_use"),
        text_turn("done"),
    ]);
    let core = core(&mock, false);
    let ep = endpoint();
    let tracker = Arc::new(Mutex::new(CacheTracker::new()));
    let agent = Agent::new(&core, &ep, Arc::new(TokioClock))
        .tools(&Fake)
        .tracker(tracker.clone());
    let cache_events = |events: &[LoopEvent]| -> Vec<String> {
        events
            .iter()
            .filter_map(|e| match e {
                LoopEvent::CacheBreak { step, info } => {
                    Some(format!("break {step} {:?} {}", info.reason, info.segment))
                }
                LoopEvent::CacheMiss { step, .. } => Some(format!("miss {step}")),
                _ => None,
            })
            .collect()
    };

    let mut history = request(true);
    let events: Vec<_> = agent
        .run(history.clone(), RunOptions::default())
        .collect()
        .await;
    assert!(cache_events(&events).is_empty());
    let Some(LoopEvent::Done(out)) = events.last() else {
        panic!()
    };
    history.messages.extend(out.messages.clone());

    history
        .messages
        .push(Message::new(Role::User, vec![Block::text("more")]));
    let events: Vec<_> = agent
        .run(history.clone(), RunOptions::default())
        .collect()
        .await;
    assert_eq!(cache_events(&events), ["miss 1"]);
    let Some(LoopEvent::Done(out)) = events.last() else {
        panic!()
    };
    history.messages.extend(out.messages.clone());

    history.system = vec![Block::text("be brief")];
    let events: Vec<_> = agent
        .run(history.clone(), RunOptions::default())
        .collect()
        .await;
    assert_eq!(cache_events(&events), ["break 1 System system/0"]);

    // Without a shared tracker a run still sees its own breaks, here a hook editing history.
    let agent = Agent::new(&core, &ep, Arc::new(TokioClock))
        .tools(&Fake)
        .hook(Arc::new(Rewrite));
    let events: Vec<_> = agent
        .run(request(true), RunOptions::default())
        .collect()
        .await;
    assert_eq!(
        cache_events(&events),
        [format!(
            "break 2 {:?} messages/0/content/0",
            BreakReason::History
        )]
    );
}
