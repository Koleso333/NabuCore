use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::Stream;
use futures::executor::block_on;
use futures::stream::StreamExt;
use ncore::core::SseEvent;
use ncore::core::ir::{Accumulator, Delta, Request, StreamEvent, plain_text};
use ncore::rt::BoxFuture;
use ncore::{
    CancelToken, Core, Endpoint, Error, EventAction, Format, GenerateOptions, Hook, HttpRequest,
    HttpResponse, InboundRequest, Interests, PassthroughBody, PassthroughOptions, RequestCtx,
    TranslateBody, TranslateOptions, Transport, TransportError,
};
use serde_json::{Value, json};

type Chunk = Result<Bytes, TransportError>;

struct GuardedBody {
    chunks: VecDeque<Chunk>,
    pending: bool,
    drops: Arc<AtomicUsize>,
}

impl Stream for GuardedBody {
    type Item = Chunk;

    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(chunk) = self.chunks.pop_front() {
            Poll::Ready(Some(chunk))
        } else if self.pending {
            Poll::Pending
        } else {
            Poll::Ready(None)
        }
    }
}

impl Drop for GuardedBody {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct MockTransport {
    chunks: Mutex<Option<Vec<Chunk>>>,
    pending: bool,
    drops: Arc<AtomicUsize>,
}

impl MockTransport {
    fn new(chunks: Vec<Chunk>, pending: bool) -> Arc<Self> {
        Arc::new(Self {
            chunks: Mutex::new(Some(chunks)),
            pending,
            drops: Arc::default(),
        })
    }

    fn drops(&self) -> usize {
        self.drops.load(Ordering::SeqCst)
    }
}

impl Transport for MockTransport {
    fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        assert_eq!(request.method, "POST");
        let chunks = self.chunks.lock().unwrap().take().expect("one request");
        let body = GuardedBody {
            chunks: chunks.into(),
            pending: self.pending,
            drops: self.drops.clone(),
        };
        Box::pin(async move {
            Ok(HttpResponse {
                status: 200,
                headers: Vec::new(),
                body: Box::pin(body),
            })
        })
    }
}

fn endpoint() -> Endpoint {
    Endpoint::new(Format::OpenAiChat, "http://local")
}

fn request() -> Request {
    Request {
        model: "model".into(),
        stream: true,
        ..Default::default()
    }
}

fn inbound() -> InboundRequest {
    InboundRequest {
        method: "POST".into(),
        path_and_query: "/v1/chat/completions".into(),
        headers: Vec::new(),
        body: Bytes::from_static(br#"{"model":"model","messages":[],"stream":true}"#),
    }
}

fn event(value: Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}

fn chat(delta: Value, finish: Value) -> Bytes {
    event(
        json!({"id": "chat-1", "model": "model", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}),
    )
}

fn partial() -> Bytes {
    chat(json!({"content": "partial"}), Value::Null)
}

fn joined(left: Bytes, right: Bytes) -> Bytes {
    let mut bytes = left.to_vec();
    bytes.extend_from_slice(&right);
    Bytes::from(bytes)
}

fn provider_error() -> Bytes {
    event(json!({"error": {"type": "server_error", "message": "stream failed"}}))
}

fn complete() -> Vec<Chunk> {
    vec![
        Ok(partial()),
        Ok(chat(json!({}), json!("stop"))),
        Ok(event(
            json!({"choices": [], "usage": {"prompt_tokens": 23, "completion_tokens": 17}}),
        )),
        Ok(Bytes::from_static(b"data: [DONE]\n\n")),
    ]
}

struct FailingHook;

impl Hook for FailingHook {
    fn name(&self) -> &str {
        "failing"
    }

    fn interests(&self) -> Interests {
        Interests {
            stream: true,
            ..Interests::NONE
        }
    }

    fn on_stream_event(&self, _: &RequestCtx, event: &mut SseEvent) -> Result<EventAction, String> {
        let value: Value = serde_json::from_str(&event.data).unwrap();
        if value.get("fail").and_then(Value::as_bool) == Some(true) {
            Err("stream failed".into())
        } else {
            Ok(EventAction::Keep)
        }
    }
}

#[test]
fn cancelled_passthrough_releases_the_body_while_response_is_retained() {
    block_on(async {
        let transport = MockTransport::new(vec![Ok(partial())], true);
        let core = Core::builder(transport.clone()).build();
        let cancel = CancelToken::new();
        let mut response = core
            .passthrough(
                &endpoint(),
                inbound(),
                PassthroughOptions::default(),
                &cancel,
            )
            .await
            .unwrap();
        let PassthroughBody::Stream(body) = &mut response.body else {
            panic!()
        };
        assert_eq!(body.next().await.unwrap().unwrap(), partial());
        assert_eq!(transport.drops(), 0);
        cancel.cancel();
        assert!(matches!(body.next().await, Some(Err(Error::Cancelled))));
        assert_eq!(transport.drops(), 1);
        assert!(body.next().await.is_none());
        assert_eq!(response.status, 200);
    });
}

#[test]
fn failed_passthrough_releases_the_body_and_keeps_pending_bytes() {
    block_on(async {
        for hook_error in [false, true] {
            let suffix = if hook_error {
                event(json!({"fail": true}))
            } else {
                Bytes::from_static(b"data: {bad\n\n")
            };
            let transport = MockTransport::new(vec![Ok(joined(partial(), suffix))], true);
            let mut builder = Core::builder(transport.clone());
            if hook_error {
                builder = builder.hook(Arc::new(FailingHook));
            }
            let core = builder.build();
            let mut response = core
                .passthrough(
                    &endpoint(),
                    inbound(),
                    PassthroughOptions {
                        strict_stream: true,
                        ..Default::default()
                    },
                    &CancelToken::new(),
                )
                .await
                .unwrap();
            let PassthroughBody::Stream(body) = &mut response.body else {
                panic!()
            };
            assert_eq!(body.next().await.unwrap().unwrap(), partial());
            assert_eq!(
                transport.drops(),
                1,
                "release before pending error is consumed"
            );
            let error = body.next().await.unwrap().unwrap_err();
            assert!(matches!(
                (&error, hook_error),
                (Error::Plugin { .. }, true) | (Error::Codec(_), false)
            ));
            assert!(body.next().await.is_none());
            assert_eq!(response.status, 200);
        }
    });
}

#[test]
fn transport_error_releases_a_retained_passthrough_body() {
    block_on(async {
        let transport = MockTransport::new(
            vec![Ok(partial()), Err(TransportError::new("broken body"))],
            true,
        );
        let core = Core::builder(transport.clone()).build();
        let mut response = core
            .passthrough(
                &endpoint(),
                inbound(),
                PassthroughOptions::default(),
                &CancelToken::new(),
            )
            .await
            .unwrap();
        let PassthroughBody::Stream(body) = &mut response.body else {
            panic!()
        };
        assert_eq!(body.next().await.unwrap().unwrap(), partial());
        assert!(
            matches!(body.next().await, Some(Err(Error::Transport(error))) if error.message == "broken body")
        );
        assert_eq!(transport.drops(), 1);
        assert!(body.next().await.is_none());
        assert_eq!(response.status, 200);
    });
}

#[test]
fn passthrough_eof_releases_the_body_without_losing_tail_bytes_or_late_usage() {
    block_on(async {
        for strict in [false, true] {
            let mut chunks = complete();
            let tail = Bytes::from_static(b"data: incomplete tail");
            if !strict {
                chunks.push(Ok(tail.clone()));
            }
            let expected: Vec<_> = chunks
                .iter()
                .flat_map(|chunk| chunk.as_ref().unwrap().iter().copied())
                .collect();
            let transport = MockTransport::new(chunks, false);
            let core = Core::builder(transport.clone()).build();
            let mut response = core
                .passthrough(
                    &endpoint(),
                    inbound(),
                    PassthroughOptions {
                        strict_stream: strict,
                        ..Default::default()
                    },
                    &CancelToken::new(),
                )
                .await
                .unwrap();
            let PassthroughBody::Stream(body) = &mut response.body else {
                panic!()
            };
            let mut bytes = Vec::new();
            while let Some(chunk) = body.next().await {
                let chunk = chunk.unwrap();
                if chunk == tail {
                    assert_eq!(transport.drops(), 1);
                } else {
                    assert_eq!(
                        transport.drops(),
                        0,
                        "do not stop on finish_reason or DONE alone"
                    );
                }
                bytes.extend_from_slice(&chunk);
            }
            assert_eq!(transport.drops(), 1);
            assert_eq!(bytes, expected);
            assert_eq!(body.observation().usage.output_total, Some(17));
            assert_eq!(response.status, 200);
        }
    });
}

#[test]
fn cancelled_generation_releases_the_body_and_keeps_partial_output() {
    block_on(async {
        let transport = MockTransport::new(vec![Ok(partial())], true);
        let core = Core::builder(transport.clone()).build();
        let cancel = CancelToken::new();
        let mut generation = core
            .generate(
                &endpoint(),
                &request(),
                &GenerateOptions::default(),
                &cancel,
            )
            .await
            .unwrap();
        let mut accumulator = Accumulator::new();
        loop {
            let event = generation.next().await.unwrap().unwrap();
            accumulator.push(&event);
            if matches!(
                event,
                StreamEvent::Delta {
                    delta: Delta::Text(_),
                    ..
                }
            ) {
                break;
            }
        }
        assert_eq!(transport.drops(), 0);
        cancel.cancel();
        assert!(matches!(
            generation.next().await,
            Some(Err(Error::Cancelled))
        ));
        assert_eq!(transport.drops(), 1);
        assert_eq!(plain_text(&accumulator.finish().content), "partial");
        assert!(generation.next().await.is_none());
        assert_eq!(generation.status, 200);
    });
}

#[test]
fn generation_errors_release_the_body_and_keep_decoded_output() {
    block_on(async {
        for failure in ["transport", "provider", "codec", "hook", "eof"] {
            let suffix = match failure {
                "provider" => provider_error(),
                "codec" => Bytes::from_static(b"data: {bad\n\n"),
                "hook" => event(json!({"fail": true})),
                _ => Bytes::new(),
            };
            let mut chunks = vec![Ok(joined(partial(), suffix))];
            if failure == "transport" {
                chunks.push(Err(TransportError::new("broken body")));
            }
            let transport = MockTransport::new(chunks, failure != "eof");
            let mut builder = Core::builder(transport.clone());
            if failure == "hook" {
                builder = builder.hook(Arc::new(FailingHook));
            }
            let core = builder.build();
            let mut generation = core
                .generate(
                    &endpoint(),
                    &request(),
                    &GenerateOptions::default(),
                    &CancelToken::new(),
                )
                .await
                .unwrap();
            let mut accumulator = Accumulator::new();
            let error = loop {
                match generation.next().await {
                    Some(Ok(event)) => {
                        accumulator.push(&event);
                        if matches!(failure, "provider" | "codec" | "hook") {
                            assert_eq!(
                                transport.drops(),
                                1,
                                "release before decoded outputs are drained"
                            );
                        }
                    }
                    Some(Err(error)) => break error,
                    None => panic!("missing {failure} error"),
                }
            };
            assert!(
                matches!(
                    (&error, failure),
                    (Error::Transport(_), "transport" | "eof")
                        | (Error::Provider(_), "provider")
                        | (Error::Codec(_), "codec")
                        | (Error::Plugin { .. }, "hook")
                ),
                "{failure}: {error}"
            );
            assert_eq!(transport.drops(), 1);
            assert_eq!(plain_text(&accumulator.finish().content), "partial");
            assert!(generation.next().await.is_none());
            assert_eq!(generation.status, 200);
        }
    });
}

#[test]
fn generation_waits_for_late_usage_and_releases_the_body_on_eof() {
    block_on(async {
        let transport = MockTransport::new(complete(), false);
        let core = Core::builder(transport.clone()).build();
        let mut generation = core
            .generate(
                &endpoint(),
                &request(),
                &GenerateOptions::default(),
                &CancelToken::new(),
            )
            .await
            .unwrap();
        let mut accumulator = Accumulator::new();
        let mut usage_seen = false;
        while let Some(event) = generation.next().await {
            let event = event.unwrap();
            assert_eq!(transport.drops(), 0);
            if let StreamEvent::Usage(usage) = &event {
                assert_eq!(usage.output_total, Some(17));
                usage_seen = true;
            }
            if let StreamEvent::End { usage, .. } = &event {
                assert!(usage_seen);
                assert_eq!(usage.output_total, Some(17));
            }
            accumulator.push(&event);
        }
        assert_eq!(transport.drops(), 1);
        assert!(accumulator.is_done());
        let response = accumulator.finish();
        assert_eq!(plain_text(&response.content), "partial");
        assert_eq!(response.usage.input_total, Some(23));
        assert_eq!(response.usage.output_total, Some(17));
        assert_eq!(generation.observation().usage.output_total, Some(17));
        assert_eq!(generation.status, 200);
    });
}

#[test]
fn translated_terminal_errors_release_the_body_while_response_is_retained() {
    block_on(async {
        for cancelled in [false, true] {
            let chunk = if cancelled {
                partial()
            } else {
                joined(partial(), provider_error())
            };
            let transport = MockTransport::new(vec![Ok(chunk)], true);
            let core = Core::builder(transport.clone()).build();
            let cancel = CancelToken::new();
            let mut response = core
                .translate_request(
                    &endpoint(),
                    Format::AnthropicMessages,
                    request(),
                    &inbound(),
                    TranslateOptions {
                        strict_stream: true,
                        ..Default::default()
                    },
                    &cancel,
                )
                .await
                .unwrap();
            let TranslateBody::Stream(body) = &mut response.body else {
                panic!()
            };
            let bytes = body.next().await.unwrap().unwrap();
            assert!(String::from_utf8_lossy(&bytes).contains("partial"));
            if cancelled {
                assert_eq!(transport.drops(), 0);
                cancel.cancel();
            } else {
                assert_eq!(transport.drops(), 1);
            }
            let error = body.next().await.unwrap().unwrap_err();
            assert!(matches!(
                (&error, cancelled),
                (Error::Cancelled, true) | (Error::Provider(_), false)
            ));
            assert_eq!(transport.drops(), 1);
            assert_eq!(response.status, 200);
        }
    });
}
