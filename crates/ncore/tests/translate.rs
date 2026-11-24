use std::sync::{Arc, Mutex};

use bytes::Bytes;
use futures::executor::block_on;
use futures::stream::{self, StreamExt};
use ncore::core::codec;
use ncore::core::ir::{Accumulator, Block, StopReason};
use ncore::rt::{BoxFuture, header};
use ncore::{
    Auth, CancelToken, Core, Endpoint, Error, Format, HttpRequest, HttpResponse, InboundRequest,
    LossPolicy, TranslateBody, TranslateOptions, Transport, TransportError,
};
use serde_json::{Value, json};

const CHAT_SSE: &[u8] =
    include_bytes!("../../ncore-core/tests/fixtures/translate/chat_tools.sse");

type Scripted = (u16, Vec<u8>);

#[derive(Default)]
struct Mock {
    script: Mutex<Vec<Scripted>>,
    seen: Mutex<Vec<HttpRequest>>,
}

impl Mock {
    fn new(script: Vec<Scripted>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(script.into_iter().rev().collect()),
            seen: Mutex::default(),
        })
    }
}

impl Transport for Mock {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        self.seen.lock().unwrap().push(req);
        let (status, body) = self
            .script
            .lock()
            .unwrap()
            .pop()
            .expect("unexpected request");
        Box::pin(async move {
            let chunks: Vec<_> = body
                .chunks(29)
                .map(|c| Ok(Bytes::copy_from_slice(c)))
                .collect();
            Ok(HttpResponse {
                status,
                headers: vec![
                    ("x-request-id".into(), "req-1".into()),
                    ("server".into(), "x".into()),
                ],
                body: Box::pin(stream::iter(chunks)),
            })
        })
    }
}

fn claude_code(stream: bool) -> InboundRequest {
    let body = json!({
        "model": "claude-sonnet",
        "max_tokens": 1000,
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        "stream": stream
    });
    InboundRequest {
        method: "POST".into(),
        path_and_query: "/v1/messages?beta=true".into(),
        headers: vec![
            ("x-api-key".into(), "sk-client".into()),
            ("anthropic-version".into(), "2023-06-01".into()),
            ("anthropic-beta".into(), "x".into()),
        ],
        body: Bytes::from(serde_json::to_vec(&body).unwrap()),
    }
}

fn opts() -> TranslateOptions {
    TranslateOptions {
        model: Some("gpt-x".into()),
        response_model: Some("claude-sonnet".into()),
        ..Default::default()
    }
}

#[test]
fn anthropic_client_streams_from_chat_provider() {
    let t = Mock::new(vec![(200, CHAT_SSE.to_vec())]);
    let core = Core::builder(t.clone()).build();
    let ep = Endpoint::openai_chat("sk-server");
    let resp =
        block_on(core.translate(&ep, claude_code(true), opts(), &CancelToken::new())).unwrap();
    assert_eq!(resp.status, 200);
    assert_eq!(
        header(&resp.headers, "content-type"),
        Some("text/event-stream")
    );
    assert_eq!(header(&resp.headers, "x-request-id"), Some("req-1"));
    assert_eq!(header(&resp.headers, "server"), None);
    assert_eq!(resp.losses.len(), 1, "{:?}", resp.losses);

    let sent = t.seen.lock().unwrap()[0].clone();
    assert_eq!(sent.url, "https://api.openai.com/v1/chat/completions");
    assert_eq!(
        header(&sent.headers, "authorization"),
        Some("Bearer sk-server")
    );
    assert_eq!(header(&sent.headers, "x-api-key"), None);
    assert_eq!(header(&sent.headers, "anthropic-beta"), None);
    let body: Value = serde_json::from_slice(&sent.body).unwrap();
    assert_eq!(body["model"], "gpt-x");
    assert_eq!(body["stream_options"]["include_usage"], true);

    let TranslateBody::Stream(mut s) = resp.body else {
        panic!("expected stream")
    };
    let bytes = block_on(async {
        let mut out = Vec::new();
        while let Some(c) = s.next().await {
            out.extend_from_slice(&c.unwrap());
        }
        out
    });
    assert_eq!(s.observation().usage.cache_read, Some(100));

    let mut parser = ncore::core::SseParser::new();
    let mut dec = codec::stream_decoder(Format::AnthropicMessages);
    let mut acc = Accumulator::new();
    for ev in parser.feed(&bytes) {
        for e in dec.feed(&ev).unwrap() {
            acc.push(&e);
        }
    }
    assert!(acc.is_done());
    let r = acc.finish();
    assert_eq!(r.model.as_deref(), Some("claude-sonnet"));
    assert_eq!(r.stop, Some(StopReason::ToolUse));
    let names: Vec<&str> = r
        .content
        .iter()
        .filter_map(|b| match b {
            Block::ToolCall(c) => Some(c.name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, ["get_weather", "get_time"]);
}

#[test]
fn drop_reasoning_renumbers_stream_blocks() {
    let t = Mock::new(vec![(200, CHAT_SSE.to_vec())]);
    let core = Core::builder(t).build();
    let o = TranslateOptions {
        drop_reasoning: true,
        ..opts()
    };
    let resp = block_on(core.translate(
        &Endpoint::openai_chat("k"),
        claude_code(true),
        o,
        &CancelToken::new(),
    ))
    .unwrap();
    let TranslateBody::Stream(mut s) = resp.body else {
        panic!("expected stream")
    };
    let bytes = block_on(async {
        let mut out = Vec::new();
        while let Some(c) = s.next().await {
            out.extend_from_slice(&c.unwrap());
        }
        out
    });
    let text = String::from_utf8_lossy(&bytes);
    assert!(!text.contains(r#""type":"thinking""#), "{text}");
    assert!(text.contains(r#""index":0"#));
    let mut parser = ncore::core::SseParser::new();
    let mut dec = codec::stream_decoder(Format::AnthropicMessages);
    let mut acc = Accumulator::new();
    for ev in parser.feed(&bytes) {
        for e in dec.feed(&ev).unwrap() {
            acc.push(&e);
        }
    }
    let r = acc.finish();
    assert!(r.content.iter().all(|b| !matches!(b, Block::Thinking(_))));
    assert_eq!(r.content.len(), 3, "{:?}", r.content);
}

#[test]
fn full_response_and_errors_are_converted() {
    let ok = json!({
        "id": "chatcmpl-9", "object": "chat.completion", "created": 1, "model": "gpt-x",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
    });
    let err = json!({"error": {"message": "slow down", "type": "requests", "code": "rate_limit_exceeded"}});
    let t = Mock::new(vec![
        (200, serde_json::to_vec(&ok).unwrap()),
        (429, serde_json::to_vec(&err).unwrap()),
    ]);
    let core = Core::builder(t).build();
    let ep = Endpoint::openai_chat("k");

    let resp =
        block_on(core.translate(&ep, claude_code(false), opts(), &CancelToken::new())).unwrap();
    let TranslateBody::Full {
        bytes, response, ..
    } = resp.body
    else {
        panic!()
    };
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["type"], "message");
    assert_eq!(v["model"], "claude-sonnet");
    assert_eq!(v["content"], json!([{"type": "text", "text": "hi"}]));
    assert_eq!(v["stop_reason"], "end_turn");
    assert_eq!(v["usage"]["input_tokens"], 5);
    assert!(response.is_some());

    let resp =
        block_on(core.translate(&ep, claude_code(false), opts(), &CancelToken::new())).unwrap();
    assert_eq!(resp.status, 429);
    let TranslateBody::Full {
        bytes, observation, ..
    } = resp.body
    else {
        panic!()
    };
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        v,
        json!({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})
    );
    assert_eq!(
        observation.error.unwrap().kind,
        ncore::core::ErrorKind::RateLimited
    );
}

#[test]
fn inherited_key_moves_to_target_header() {
    let ok = json!({"candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP"}]});
    let t = Mock::new(vec![(200, serde_json::to_vec(&ok).unwrap())]);
    let core = Core::builder(t.clone()).build();
    let ep = Endpoint::new(Format::Gemini, "https://g").with_auth(Auth::Inherit);
    block_on(core.translate(&ep, claude_code(false), opts(), &CancelToken::new())).unwrap();
    let sent = t.seen.lock().unwrap()[0].clone();
    assert_eq!(sent.url, "https://g/v1beta/models/gpt-x:generateContent");
    assert_eq!(header(&sent.headers, "x-goog-api-key"), Some("sk-client"));
    assert_eq!(header(&sent.headers, "x-api-key"), None);
}

#[test]
fn loss_policy_error_refuses_before_sending() {
    let t = Mock::new(vec![]);
    let core = Core::builder(t.clone()).build();
    let o = TranslateOptions {
        loss_policy: LossPolicy::Error,
        ..opts()
    };
    let err = block_on(core.translate(
        &Endpoint::openai_chat("k"),
        claude_code(false),
        o,
        &CancelToken::new(),
    ))
    .unwrap_err();
    let Error::Loss(l) = err else { panic!("{err}") };
    assert_eq!(l[0].path, "/cache");
    assert!(t.seen.lock().unwrap().is_empty());
}

#[test]
fn antigravity_signatures_round_trip_in_full_and_streamed_responses() {
    for client in [Format::AnthropicMessages, Format::OpenAiResponses] {
        for streaming in [false, true] {
            let signature = "s".repeat(80);
            let response = json!({"candidates":[{"content":{"role":"model","parts":[{"text":"thinking","thought":true,"thoughtSignature":signature},{"text":"answer"}]},"finishReason":"STOP"}]});
            let body = if streaming {
                format!("data: {response}\n\n").into_bytes()
            } else {
                serde_json::to_vec(&response).unwrap()
            };
            let transport = Mock::new(vec![(200, body)]);
            let core = Core::builder(transport).build();
            let endpoint = Endpoint::new(Format::Gemini, "https://cloud.example")
                .with_signature_namespace("antigravity");
            let request = ncore::core::ir::Request {
                model: "claude-x".into(),
                stream: streaming,
                max_tokens: Some(4096),
                ..Default::default()
            };
            let inbound = InboundRequest {
                method: "POST".into(),
                path_and_query: "/v1/messages".into(),
                headers: vec![],
                body: Bytes::new(),
            };
            let translated = block_on(core.translate_request(
                &endpoint,
                client,
                request,
                &inbound,
                TranslateOptions {
                    strict_stream: true,
                    ..Default::default()
                },
                &CancelToken::new(),
            ))
            .unwrap();
            let value: Value = match translated.body {
                TranslateBody::Full { bytes, .. } => serde_json::from_slice(&bytes).unwrap(),
                TranslateBody::Stream(mut stream) => {
                    let bytes = block_on(async {
                        let mut out = Vec::new();
                        while let Some(chunk) = stream.next().await {
                            out.extend_from_slice(&chunk.unwrap());
                        }
                        out
                    });
                    let events = ncore::core::SseParser::new().feed(&bytes);
                    if client == Format::AnthropicMessages {
                        let mut blocks: Vec<Value> = Vec::new();
                        for ev in events {
                            let value: Value = serde_json::from_str(&ev.data).unwrap();
                            if value["type"] == "content_block_start" {
                                blocks.push(value["content_block"].clone());
                            }
                            if value["type"] == "content_block_delta" {
                                let b = &mut blocks[value["index"].as_u64().unwrap() as usize];
                                for key in ["text", "thinking", "signature"] {
                                    if let Some(delta) = value["delta"][key].as_str() {
                                        b[key] = json!(format!(
                                            "{}{delta}",
                                            b[key].as_str().unwrap_or("")
                                        ));
                                    }
                                }
                            }
                        }
                        json!({"content":blocks})
                    } else {
                        events
                            .iter()
                            .filter_map(|e| serde_json::from_str::<Value>(&e.data).ok())
                            .find(|v| v["type"] == "response.completed")
                            .unwrap()["response"]
                            .clone()
                    }
                }
            };
            let (path, next) = if client == Format::AnthropicMessages {
                (
                    "/v1/messages",
                    json!({"model":"claude-x","max_tokens":4096,"messages":[{"role":"assistant","content":value["content"]},{"role":"user","content":"next"}]}),
                )
            } else {
                (
                    "/v1/responses",
                    json!({"model":"claude-x","input":value["output"],"max_output_tokens":4096}),
                )
            };
            let mut ir =
                codec::decode_request(client, path, &serde_json::to_vec(&next).unwrap()).unwrap();
            let unsigned = codec::encode_request(Format::Gemini, &ir, &Default::default()).unwrap();
            assert!(!String::from_utf8_lossy(&unsigned.body).contains("thoughtSignature"));
            let Block::Thinking(thought) = &mut ir.messages[0].content[0] else {
                panic!("thinking missing: {ir:?}")
            };
            assert_eq!(thought.text, "thinking");
            let sig = thought.sig.as_mut().unwrap();
            assert_eq!(sig.namespace.as_deref(), Some("antigravity"));
            assert_eq!(sig.format, Format::Gemini);
            assert_eq!(sig.data, signature);
            let replay = Mock::new(vec![(200, serde_json::to_vec(&response).unwrap())]);
            let replay_core = Core::builder(replay.clone()).build();
            block_on(replay_core.translate_request(
                &endpoint,
                client,
                ir,
                &inbound,
                TranslateOptions::default(),
                &CancelToken::new(),
            ))
            .unwrap();
            let restored: Value =
                serde_json::from_slice(&replay.seen.lock().unwrap()[0].body).unwrap();
            assert_eq!(
                restored["contents"][0]["parts"][0]["thoughtSignature"],
                signature
            );
        }
    }
}
