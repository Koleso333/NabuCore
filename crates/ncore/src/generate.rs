use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use ncore_core::cache::{self, CachePolicy};
use ncore_core::codec::{self, EncodeOptions, StreamDecoder};
use ncore_core::ir::{Block, BlockStart, Delta, Loss, Losses, Request, Response, StreamEvent};
use ncore_core::{Format, Observation, ProviderError, SseEvent, SseParser, StreamTap};

use crate::Core;
use crate::cancel::CancelToken;
use crate::endpoint::Endpoint;
use crate::error::Error;
use crate::hooks::{EventAction, Pipeline, RequestCtx, plugin_error};
use crate::passthrough::{Exchange, InboundRequest, observed_events, read_all};
use crate::rt::{BoxStream, TransportError};
use crate::translate::{LossPolicy, provider_request};

#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    pub encode: EncodeOptions,
    pub loss_policy: LossPolicy,
    /// Extra request headers such as `anthropic-beta`; they override endpoint headers.
    pub headers: Vec<(String, String)>,
    /// Breakpoints to add before encoding; `None` sends the request's own marks only.
    pub cache: Option<CachePolicy>,
}

/// Provider reply to an IR request, read as IR stream events.
pub struct Generation {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub attempts: u32,
    /// Losses of the request conversion.
    pub losses: Vec<Loss>,
    /// Provider body after request hooks, before transport middleware.
    pub request: Bytes,
    source: Source,
}

enum Source {
    Full {
        events: VecDeque<StreamEvent>,
        response: Option<Response>,
        observation: Observation,
    },
    Stream {
        events: EventStream,
        ready: VecDeque<StreamEvent>,
    },
}

impl std::fmt::Debug for Generation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Generation")
            .field("status", &self.status)
            .field("streaming", &matches!(self.source, Source::Stream { .. }))
            .finish()
    }
}

impl Generation {
    /// Exact decoded response of a non-streamed request; streamed replies are rebuilt with `Accumulator`.
    pub fn take_response(&mut self) -> Option<Response> {
        match &mut self.source {
            Source::Full { response, .. } => response.take(),
            Source::Stream { .. } => None,
        }
    }

    /// Provider-side facts observed so far.
    pub fn observation(&self) -> Observation {
        match &self.source {
            Source::Full { observation, .. } => observation.clone(),
            Source::Stream { events, .. } => events.shared.lock().unwrap().clone(),
        }
    }
}

impl Stream for Generation {
    type Item = Result<StreamEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match &mut self.source {
            Source::Full { events, .. } => Poll::Ready(events.pop_front().map(Ok)),
            Source::Stream { events, ready } => loop {
                if let Some(ev) = ready.pop_front() {
                    return Poll::Ready(Some(Ok(ev)));
                }
                match events.poll_chunk(cx) {
                    Poll::Ready(Some(Ok(batch))) => ready.extend(batch),
                    Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                    Poll::Ready(None) => return Poll::Ready(None),
                    Poll::Pending => return Poll::Pending,
                }
            },
        }
    }
}

/// Events a stream would have carried for a complete response.
fn events_of(resp: &Response) -> VecDeque<StreamEvent> {
    let mut out = VecDeque::new();
    out.push_back(StreamEvent::Start {
        id: resp.id.clone(),
        model: resp.model.clone(),
    });
    for (index, b) in resp.content.iter().enumerate() {
        let (start, deltas) = match b {
            Block::Text { text, sig, .. } => (
                BlockStart::Text,
                [Some(Delta::Text(text.clone())), sig.clone().map(Delta::Sig)],
            ),
            Block::Thinking(t) => (
                BlockStart::Thinking {
                    redacted: t.redacted,
                },
                [
                    (!t.text.is_empty()).then(|| Delta::Thinking(t.text.clone())),
                    t.sig.clone().map(Delta::Sig),
                ],
            ),
            Block::ToolCall(c) => (
                BlockStart::ToolCall {
                    id: c.id.clone(),
                    name: c.name.clone(),
                },
                [
                    Some(Delta::ToolArgs(c.args.clone())),
                    c.sig.clone().map(Delta::Sig),
                ],
            ),
            Block::Opaque(o) => (BlockStart::Opaque(o.clone()), [None, None]),
            // Media and tool results have no stream form; the response itself carries them.
            Block::Media { .. } | Block::ToolResult(_) => continue,
        };
        out.push_back(StreamEvent::BlockStart {
            index,
            block: start,
        });
        for delta in deltas.into_iter().flatten() {
            out.push_back(StreamEvent::Delta { index, delta });
        }
        out.push_back(StreamEvent::BlockEnd { index });
    }
    out.push_back(StreamEvent::End {
        stop: resp.stop.clone(),
        usage: resp.usage.clone(),
    });
    // Indices must stay dense even when blocks were skipped.
    let mut map = std::collections::HashMap::new();
    for ev in out.iter_mut() {
        if let StreamEvent::BlockStart { index, .. }
        | StreamEvent::Delta { index, .. }
        | StreamEvent::BlockEnd { index } = ev
        {
            let next = map.len();
            *index = *map.entry(*index).or_insert(next);
        }
    }
    out
}

impl Core {
    /// Sends an IR request to an endpoint; provider errors come back as `Error::Provider`.
    pub async fn generate(
        &self,
        endpoint: &Endpoint,
        req: &Request,
        opts: &GenerateOptions,
        cancel: &CancelToken,
    ) -> Result<Generation, Error> {
        let format = endpoint.format;
        let (encoded, losses) = encode(format, req, &opts.encode, opts.cache.as_ref())?;
        if opts.loss_policy == LossPolicy::Error && !losses.is_empty() {
            return Err(Error::Loss(losses));
        }
        let inbound = InboundRequest {
            method: "POST".into(),
            path_and_query: encoded.path.clone(),
            headers: opts.headers.clone(),
            body: Bytes::new(),
        };
        let mut http = provider_request(endpoint, format, &inbound, encoded);
        for (name, value) in &opts.headers {
            http.headers.retain(|(k, _)| !k.eq_ignore_ascii_case(name));
            http.headers.push((name.clone(), value.clone()));
        }
        let Exchange {
            ctx,
            pipe,
            status,
            headers,
            body,
            attempts,
            request,
        } = self.exchange(endpoint, http, req.stream, cancel).await?;

        if req.stream && status < 400 {
            return Ok(Generation {
                status,
                headers,
                attempts,
                losses,
                request,
                source: Source::Stream {
                    events: EventStream::new(ctx, body, pipe, cancel.clone(), true),
                    ready: VecDeque::new(),
                },
            });
        }
        let mut bytes = read_all(body, cancel).await?;
        for h in &pipe.body {
            h.on_body(&ctx, status, &mut bytes)
                .map_err(|m| plugin_error(h.name(), m))?;
        }
        let observation = Observation::from_body(format, status, &bytes);
        for ev in observed_events(&observation) {
            pipe.observe(&ctx, &ev);
        }
        if status >= 400 {
            return Err(Error::Provider(ProviderError::parse(
                format,
                Some(status),
                bytes,
            )));
        }
        let response = codec::decode_response(format, &bytes).map_err(|e| Error::Codec(e.0))?;
        Ok(Generation {
            status,
            headers,
            attempts,
            losses,
            request,
            source: Source::Full {
                events: events_of(&response),
                response: Some(response),
                observation,
            },
        })
    }
}

/// Encodes a request after planning cache breakpoints; losses of both steps are returned.
pub(crate) fn encode(
    format: Format,
    req: &Request,
    opts: &EncodeOptions,
    cache: Option<&CachePolicy>,
) -> Result<(codec::Encoded, Vec<Loss>), Error> {
    let mut planned;
    let mut losses = Losses::default();
    let req = match cache {
        Some(policy) => {
            planned = req.clone();
            cache::plan(format, &mut planned, policy, opts, &mut losses);
            &planned
        }
        None => req,
    };
    let encoded = codec::encode_request(format, req, opts).map_err(|e| Error::Codec(e.0))?;
    losses.0.extend(encoded.losses.0.iter().cloned());
    Ok((encoded, losses.0))
}

/// Provider SSE body decoded into IR events, with hooks and observation applied.
pub(crate) struct EventStream {
    pub ctx: RequestCtx,
    body: BoxStream<'static, Result<Bytes, TransportError>>,
    parser: SseParser,
    tap: StreamTap,
    pub shared: Arc<Mutex<Observation>>,
    decoder: Box<dyn StreamDecoder>,
    pipe: Arc<Pipeline>,
    cancel: CancelToken,
    failed: Option<Error>,
    done: bool,
    // Strict streams turn provider error events and cut-off streams into errors.
    strict: bool,
    ended: bool,
}

impl EventStream {
    pub fn new(
        ctx: RequestCtx,
        body: BoxStream<'static, Result<Bytes, TransportError>>,
        pipe: Arc<Pipeline>,
        cancel: CancelToken,
        strict: bool,
    ) -> Self {
        Self {
            tap: StreamTap::new(ctx.format),
            decoder: codec::stream_decoder(ctx.format),
            ctx,
            body,
            parser: SseParser::new(),
            shared: Arc::default(),
            pipe,
            cancel,
            failed: None,
            done: false,
            strict,
            ended: false,
        }
    }

    fn stop(&mut self) {
        self.done = true;
        self.body = Box::pin(futures_util::stream::empty());
    }

    fn process(&mut self, mut ev: SseEvent, out: &mut Vec<StreamEvent>) -> Result<(), Error> {
        for h in &self.pipe.stream {
            match h.on_stream_event(&self.ctx, &mut ev) {
                Ok(EventAction::Keep) => {}
                Ok(EventAction::Drop) => return Ok(()),
                Err(m) => return Err(plugin_error(h.name(), m)),
            }
        }
        let observed = self.tap.on_event(&ev);
        if !observed.is_empty() {
            *self.shared.lock().unwrap() = self.tap.observation().clone();
            for o in &observed {
                self.pipe.observe(&self.ctx, o);
            }
        }
        for e in self.decoder.feed(&ev).map_err(|e| Error::Codec(e.0))? {
            if self.strict {
                if self.ended {
                    continue;
                }
                if let StreamEvent::Error(pe) = e {
                    return Err(Error::Provider(pe));
                }
            }
            if matches!(e, StreamEvent::End { .. }) {
                self.ended = true;
            }
            out.push(e);
        }
        Ok(())
    }

    fn finish(&mut self, out: &mut Vec<StreamEvent>) -> Result<(), Error> {
        // An event without its closing blank line is discarded, as the SSE spec says.
        let tail_bytes = self.parser.finish();
        if self.strict && !tail_bytes.is_empty() {
            return Err(Error::Transport(TransportError::new(
                "stream ended with an incomplete SSE event",
            )));
        }
        let tail = self.decoder.finish();
        // Formats without a terminator (Gemini) end with a synthetic event carrying the finish reason.
        let complete = self.ended
            || tail
                .iter()
                .any(|e| matches!(e, StreamEvent::End { stop: Some(_), .. }));
        if self.strict && !complete {
            return Err(Error::Transport(TransportError::new(
                "stream ended before the response was complete",
            )));
        }
        out.extend(tail);
        Ok(())
    }

    /// Events decoded from the next body chunk.
    pub fn poll_chunk(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Vec<StreamEvent>, Error>>> {
        loop {
            if let Some(e) = self.failed.take() {
                return Poll::Ready(Some(Err(e)));
            }
            if self.done {
                return Poll::Ready(None);
            }
            if self.cancel.poll_cancelled(cx) {
                self.stop();
                return Poll::Ready(Some(Err(Error::Cancelled)));
            }
            let mut out = Vec::new();
            match self.body.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(chunk))) => {
                    for ev in self.parser.feed(&chunk) {
                        if let Err(e) = self.process(ev, &mut out) {
                            self.failed = Some(e);
                            self.stop();
                            break;
                        }
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    self.stop();
                    self.failed = Some(e.into());
                }
                Poll::Ready(None) => {
                    self.stop();
                    if let Err(e) = self.finish(&mut out) {
                        self.failed = Some(e);
                    }
                }
            }
            if !out.is_empty() {
                return Poll::Ready(Some(Ok(out)));
            }
        }
    }
}
