use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use bytes::{Bytes, BytesMut};
use futures_core::Stream;
use futures_util::StreamExt;
use ncore_core::format::replace_gemini_model;
use ncore_core::json::{self, Patch};
use ncore_core::{Format, Observation, Observed, SseEvent, SseParser, StreamTap};

use crate::Core;
use crate::cancel::{CancelToken, or_cancel};
use crate::endpoint::{Auth, Endpoint};
use crate::error::Error;
use crate::hooks::{EventAction, Next, Pipeline, RequestCtx, ResponseHead, plugin_error};
use crate::rt::{BoxStream, HttpRequest, HttpResponse, TransportError};

/// Request as received by the host, in the endpoint's own format.
#[derive(Debug, Clone)]
pub struct InboundRequest {
    pub method: String,
    pub path_and_query: String,
    pub headers: Vec<(String, String)>,
    pub body: Bytes,
}

/// Point edits applied on the way through; everything else is forwarded byte for byte.
#[derive(Debug, Clone, Default)]
pub struct PassthroughOptions {
    /// Model id to send to the provider.
    pub model: Option<String>,
    /// Model id to report back to the client in responses and stream events.
    pub response_model: Option<String>,
    /// Extra body edits.
    pub patches: Vec<Patch>,
    pub strict_stream: bool,
}

/// Provider response ready to be written back to the client.
#[derive(Debug)]
pub struct PassthroughResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: PassthroughBody,
    pub attempts: u32,
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum PassthroughBody {
    Full {
        bytes: Bytes,
        observation: Observation,
    },
    Stream(PassthroughStream),
}

const HOP_BY_HOP: [&str; 10] = [
    "connection",
    "keep-alive",
    "proxy-connection",
    "transfer-encoding",
    "te",
    "trailer",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
];

const AUTH_HEADERS: [&str; 3] = ["authorization", "x-api-key", "x-goog-api-key"];

fn drop_query_param(path_and_query: &str, name: &str) -> String {
    let Some((path, query)) = path_and_query.split_once('?') else {
        return path_and_query.to_owned();
    };
    let kept: Vec<&str> = query
        .split('&')
        .filter(|kv| kv.split('=').next() != Some(name))
        .collect();
    if kept.is_empty() {
        path.to_owned()
    } else {
        format!("{path}?{}", kept.join("&"))
    }
}

/// Builds the provider request from the inbound one.
pub fn build_request(
    endpoint: &Endpoint,
    req: InboundRequest,
    opts: &PassthroughOptions,
) -> Result<HttpRequest, Error> {
    let format = endpoint.format;
    if let Some(found) = Format::detect(&req.path_and_query)
        && found != format
    {
        return Err(Error::FormatMismatch {
            endpoint: format,
            request: found,
        });
    }
    let mut path = req.path_and_query;
    let mut patches = Vec::new();
    if let Some(model) = &opts.model {
        match format.request_model_pointer() {
            Some(p) => patches.push(Patch::set(p, model)?),
            None => {
                path = replace_gemini_model(&path, model)
                    .ok_or_else(|| Error::Config("cannot find model segment in path".into()))?;
            }
        }
    }
    patches.extend(opts.patches.iter().cloned());
    let body = if patches.is_empty() {
        req.body
    } else {
        Bytes::from(json::apply(&req.body, &patches)?)
    };

    let mut headers: Vec<(String, String)> = req
        .headers
        .into_iter()
        .filter(|(k, _)| !HOP_BY_HOP.iter().any(|h| k.eq_ignore_ascii_case(h)))
        .collect();
    match &endpoint.auth {
        Auth::Inherit => {}
        auth => {
            headers.retain(|(k, _)| !AUTH_HEADERS.iter().any(|h| k.eq_ignore_ascii_case(h)));
            path = drop_query_param(&path, "key");
            match auth {
                Auth::Header { name, value } => headers.push((name.clone(), value.clone())),
                Auth::Bearer(t) => headers.push(("authorization".into(), format!("Bearer {t}"))),
                Auth::Inherit => unreachable!(),
            }
        }
    }
    for (name, value) in &endpoint.headers {
        headers.retain(|(k, _)| !k.eq_ignore_ascii_case(name));
        headers.push((name.clone(), value.clone()));
    }
    Ok(HttpRequest {
        method: req.method,
        url: format!("{}{}", endpoint.base_url.trim_end_matches('/'), path),
        headers,
        body,
    })
}

fn patch_model(format: Format, body: &[u8], model: &str) -> Option<Vec<u8>> {
    for p in format.response_model_pointers() {
        let path = json::JsonPath::parse(p).ok()?;
        if json::find(body, &path).ok()?.is_some() {
            return json::apply(body, &[Patch::set(p, model).ok()?]).ok();
        }
    }
    None
}

fn response_headers(headers: Vec<(String, String)>) -> Vec<(String, String)> {
    headers
        .into_iter()
        .filter(|(k, _)| !HOP_BY_HOP.iter().any(|h| k.eq_ignore_ascii_case(h)))
        .collect()
}

impl Core {
    /// Forwards a request in the endpoint's own format, editing only what `opts` and plugins ask for.
    pub async fn passthrough(
        &self,
        endpoint: &Endpoint,
        req: InboundRequest,
        opts: PassthroughOptions,
        cancel: &CancelToken,
    ) -> Result<PassthroughResponse, Error> {
        let format = endpoint.format;
        let streaming = format.is_streaming(&req.path_and_query, &req.body);
        let http = build_request(endpoint, req, &opts)?;
        let Exchange {
            ctx,
            pipe,
            status,
            headers,
            body,
            attempts,
            ..
        } = self.exchange(endpoint, http, streaming, cancel).await?;

        if streaming && status < 400 {
            let mut stream =
                PassthroughStream::new(ctx, body, opts.response_model, pipe, cancel.clone());
            if opts.strict_stream {
                stream.validator = Some(ncore_core::codec::stream_decoder(format));
            }
            return Ok(PassthroughResponse {
                status,
                headers,
                body: PassthroughBody::Stream(stream),
                attempts,
            });
        }

        let mut bytes = read_all(body, cancel).await?;
        for h in &pipe.body {
            h.on_body(&ctx, status, &mut bytes)
                .map_err(|m| plugin_error(h.name(), m))?;
        }
        if let (Some(m), true) = (&opts.response_model, status < 400)
            && let Some(patched) = patch_model(format, &bytes, m)
        {
            bytes = Bytes::from(patched);
        }
        let observation = Observation::from_body(format, status, &bytes);
        for ev in observed_events(&observation) {
            pipe.observe(&ctx, &ev);
        }
        Ok(PassthroughResponse {
            status,
            headers,
            body: PassthroughBody::Full { bytes, observation },
            attempts,
        })
    }

    /// Runs request hooks, middleware and retries, then response-head hooks.
    pub(crate) async fn exchange(
        &self,
        endpoint: &Endpoint,
        mut http: HttpRequest,
        streaming: bool,
        cancel: &CancelToken,
    ) -> Result<Exchange, Error> {
        let pipe = Arc::new(Pipeline::new(&self.plugins, &endpoint.plugins));
        let mut ctx = RequestCtx {
            id: self.request_id(),
            format: endpoint.format,
            streaming,
            attempt: 0,
        };
        for h in &pipe.request {
            or_cancel(cancel, h.on_request(&ctx, &mut http))
                .await
                .ok_or(Error::Cancelled)?
                .map_err(|m| plugin_error(h.name(), m))?;
        }
        let request = http.body.clone();
        let (resp, attempts) = self
            .send_with_retry(&pipe, &mut ctx, http, endpoint.allow_replays, cancel)
            .await?;
        let HttpResponse {
            status,
            headers,
            body,
        } = resp;
        let mut head = ResponseHead {
            status,
            headers: response_headers(headers),
        };
        for h in &pipe.response {
            or_cancel(cancel, h.on_response(&ctx, &mut head))
                .await
                .ok_or(Error::Cancelled)?
                .map_err(|m| plugin_error(h.name(), m))?;
        }
        Ok(Exchange {
            request,
            ctx,
            pipe,
            status: head.status,
            headers: head.headers,
            body,
            attempts,
        })
    }

    async fn send_with_retry(
        &self,
        pipe: &Pipeline,
        ctx: &mut RequestCtx,
        http: HttpRequest,
        allow_replays: bool,
        cancel: &CancelToken,
    ) -> Result<(HttpResponse, u32), Error> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            ctx.attempt = attempt;
            let retry = self
                .retry
                .as_ref()
                .filter(|(p, _)| allow_replays && attempt < p.max_attempts);
            let next = Next::new(ctx, &pipe.middleware, &*self.transport);
            let res = or_cancel(cancel, next.run(http.clone()))
                .await
                .ok_or(Error::Cancelled)?;
            let delay = match (res, retry) {
                (Err(Error::Transport(e)), Some((p, _))) if e.connect && p.retry_connect_errors => {
                    p.delay(attempt, None)
                }
                (Err(e), _) => return Err(e),
                (Ok(resp), Some((p, _))) if p.retries_status(resp.status) => {
                    let d = p.delay(attempt, resp.header("retry-after"));
                    drop(resp);
                    d
                }
                (Ok(resp), _) => return Ok((resp, attempt)),
            };
            let (_, clock) = retry.expect("retry policy present");
            or_cancel(cancel, clock.sleep(delay))
                .await
                .ok_or(Error::Cancelled)?;
        }
    }
}

/// Provider response after the shared request path.
pub(crate) struct Exchange {
    pub request: Bytes,
    pub ctx: RequestCtx,
    pub pipe: Arc<Pipeline>,
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: BoxStream<'static, Result<Bytes, TransportError>>,
    pub attempts: u32,
}

pub(crate) fn observed_events(o: &Observation) -> Vec<Observed> {
    let mut out = Vec::new();
    if let Some(m) = &o.model {
        out.push(Observed::Model(m.clone()));
    }
    if !o.usage.is_empty() {
        out.push(Observed::Usage(o.usage.clone()));
    }
    if let Some(s) = &o.stop_reason {
        out.push(Observed::Stop(s.clone()));
    }
    if let Some(e) = &o.error {
        out.push(Observed::Error(e.clone()));
    }
    out
}

pub(crate) async fn read_all(
    mut body: BoxStream<'static, Result<Bytes, TransportError>>,
    cancel: &CancelToken,
) -> Result<Bytes, Error> {
    let mut buf = BytesMut::new();
    loop {
        match or_cancel(cancel, body.next())
            .await
            .ok_or(Error::Cancelled)?
        {
            Some(chunk) => buf.extend_from_slice(&chunk?),
            None => return Ok(buf.freeze()),
        }
    }
}

/// Byte stream for the client; observes and optionally edits events in flight.
pub struct PassthroughStream {
    ctx: RequestCtx,
    body: BoxStream<'static, Result<Bytes, TransportError>>,
    parser: SseParser,
    tap: StreamTap,
    shared: Arc<Mutex<Observation>>,
    response_model: Option<String>,
    pipe: Arc<Pipeline>,
    cancel: CancelToken,
    pending: VecDeque<Bytes>,
    failed: Option<Error>,
    done: bool,
    validator: Option<Box<dyn ncore_core::codec::StreamDecoder>>,
    ended: bool,
}

impl std::fmt::Debug for PassthroughStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassthroughStream")
            .field("format", &self.ctx.format)
            .field("done", &self.done)
            .finish()
    }
}

impl PassthroughStream {
    fn new(
        ctx: RequestCtx,
        body: BoxStream<'static, Result<Bytes, TransportError>>,
        response_model: Option<String>,
        pipe: Arc<Pipeline>,
        cancel: CancelToken,
    ) -> Self {
        Self {
            tap: StreamTap::new(ctx.format),
            ctx,
            body,
            parser: SseParser::new(),
            shared: Arc::default(),
            response_model,
            pipe,
            cancel,
            pending: VecDeque::new(),
            failed: None,
            done: false,
            validator: None,
            ended: false,
        }
    }

    /// Snapshot of what has been observed so far.
    pub fn observation(&self) -> Observation {
        self.shared.lock().unwrap().clone()
    }

    /// Handle that stays readable after the stream is moved or dropped.
    pub fn observation_handle(&self) -> Arc<Mutex<Observation>> {
        self.shared.clone()
    }

    fn stop(&mut self) {
        self.done = true;
        self.body = Box::pin(futures_util::stream::empty());
    }

    fn process(&mut self, mut ev: SseEvent, out: &mut BytesMut) -> Result<(), Error> {
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
        if let Some(decoder) = &mut self.validator {
            for event in decoder.feed(&ev).map_err(|e| Error::Codec(e.0))? {
                if matches!(
                    event,
                    ncore_core::ir::StreamEvent::End { .. }
                        | ncore_core::ir::StreamEvent::Error(_)
                ) {
                    self.ended = true;
                }
            }
        }
        if let (Some(m), true) = (&self.response_model, ev.has_data())
            && let Some(patched) = patch_model(self.ctx.format, ev.data.as_bytes(), m)
        {
            out.extend_from_slice(&ev.with_data(&String::from_utf8_lossy(&patched)));
            return Ok(());
        }
        out.extend_from_slice(&ev.raw);
        Ok(())
    }
}

impl Stream for PassthroughStream {
    type Item = Result<Bytes, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            if let Some(b) = this.pending.pop_front() {
                return Poll::Ready(Some(Ok(b)));
            }
            if let Some(e) = this.failed.take() {
                return Poll::Ready(Some(Err(e)));
            }
            if this.done {
                return Poll::Ready(None);
            }
            if this.cancel.poll_cancelled(cx) {
                this.stop();
                return Poll::Ready(Some(Err(Error::Cancelled)));
            }
            match this.body.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(chunk))) => {
                    let mut out = BytesMut::new();
                    for ev in this.parser.feed(&chunk) {
                        if let Err(e) = this.process(ev, &mut out) {
                            this.failed = Some(e);
                            this.stop();
                            break;
                        }
                    }
                    if !out.is_empty() {
                        this.pending.push_back(out.freeze());
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    this.stop();
                    return Poll::Ready(Some(Err(e.into())));
                }
                Poll::Ready(None) => {
                    this.stop();
                    let tail = this.parser.finish();
                    if let Some(decoder) = &mut this.validator {
                        let complete = decoder.finish().iter().any(|event| {
                            matches!(
                                event,
                                ncore_core::ir::StreamEvent::End { stop: Some(_), .. }
                            )
                        });
                        if !tail.is_empty() || (!this.ended && !complete) {
                            return Poll::Ready(Some(Err(Error::Transport(TransportError::new(
                                "stream ended before the response was complete",
                            )))));
                        }
                    } else if !tail.is_empty() {
                        this.pending.push_back(tail);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::header;

    fn inbound(path: &str, headers: &[(&str, &str)], body: &str) -> InboundRequest {
        InboundRequest {
            method: "POST".into(),
            path_and_query: path.into(),
            headers: headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            body: Bytes::copy_from_slice(body.as_bytes()),
        }
    }

    #[test]
    fn swaps_auth_and_model_only() {
        let body = r#"{ "model":"claude-a", "unknown_future_field": {"x": 1.50}, "stream": true }"#;
        let req = inbound(
            "/v1/messages",
            &[
                ("x-api-key", "client"),
                ("anthropic-version", "2023-06-01"),
                ("Host", "localhost"),
                ("content-length", "5"),
            ],
            body,
        );
        let opts = PassthroughOptions {
            model: Some("claude-b".into()),
            ..Default::default()
        };
        let out = build_request(&Endpoint::anthropic("server"), req, &opts).unwrap();
        assert_eq!(out.url, "https://api.anthropic.com/v1/messages");
        assert_eq!(
            out.body.as_ref(),
            br#"{ "model":"claude-b", "unknown_future_field": {"x": 1.50}, "stream": true }"#
        );
        assert_eq!(header(&out.headers, "x-api-key"), Some("server"));
        assert_eq!(
            header(&out.headers, "anthropic-version"),
            Some("2023-06-01")
        );
        assert_eq!(header(&out.headers, "host"), None);
        assert_eq!(header(&out.headers, "content-length"), None);
    }

    #[test]
    fn untouched_body_without_patches() {
        let body = "{\n  \"model\" : \"m\",\"messages\":[]\n}";
        let req = inbound(
            "/v1/chat/completions",
            &[("authorization", "Bearer c")],
            body,
        );
        let ep = Endpoint::new(Format::OpenAiChat, "http://x/");
        let out = build_request(&ep, req, &PassthroughOptions::default()).unwrap();
        assert_eq!(out.body.as_ref(), body.as_bytes());
        assert_eq!(header(&out.headers, "authorization"), Some("Bearer c"));
        assert_eq!(out.url, "http://x/v1/chat/completions");
    }

    #[test]
    fn gemini_model_and_key_live_in_url() {
        let req = inbound(
            "/v1beta/models/a:streamGenerateContent?alt=sse&key=client",
            &[],
            "{}",
        );
        let opts = PassthroughOptions {
            model: Some("b".into()),
            ..Default::default()
        };
        let out = build_request(&Endpoint::gemini("server"), req, &opts).unwrap();
        assert_eq!(
            out.url,
            "https://generativelanguage.googleapis.com/v1beta/models/b:streamGenerateContent?alt=sse"
        );
        assert_eq!(header(&out.headers, "x-goog-api-key"), Some("server"));
    }

    #[test]
    fn rejects_mismatched_format() {
        let req = inbound("/v1/chat/completions", &[], "{}");
        let err = build_request(
            &Endpoint::anthropic("k"),
            req,
            &PassthroughOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::FormatMismatch { .. }));
    }
}
