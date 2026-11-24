use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use bytes::{Bytes, BytesMut};
use futures_core::Stream;
use ncore_core::cache::CachePolicy;
use ncore_core::codec::{self, EncodeOptions, StreamEncoder};
use ncore_core::ir::{Block, BlockStart, Delta, Loss, Opaque, Request, Response, StreamEvent};
use ncore_core::{Format, Observation, ProviderError};

use crate::Core;
use crate::cancel::CancelToken;
use crate::endpoint::{Auth, Endpoint};
use crate::error::Error;
use crate::generate::{EventStream, encode};
use crate::hooks::{Pipeline, RequestCtx, plugin_error};
use crate::passthrough::{Exchange, InboundRequest, observed_events, read_all};
use crate::rt::{BoxStream, HttpRequest, TransportError, header};

/// What to do when the target format cannot carry something.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LossPolicy {
    /// Convert anyway and return the losses with the response.
    #[default]
    Report,
    /// Refuse requests that would lose anything.
    Error,
}

#[derive(Debug, Clone, Default)]
pub struct TranslateOptions {
    /// Model id to send to the provider.
    pub model: Option<String>,
    /// Model id to report back to the client.
    pub response_model: Option<String>,
    pub encode: EncodeOptions,
    pub loss_policy: LossPolicy,
    /// Remove thinking blocks from the response before it reaches the client.
    pub drop_reasoning: bool,
    /// Breakpoints to add for the provider; `None` forwards the client's own marks only.
    pub cache: Option<CachePolicy>,
    pub strict_stream: bool,
}

#[derive(Debug)]
pub struct TranslateResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: TranslateBody,
    pub attempts: u32,
    /// Losses of the request conversion, plus the response for full bodies.
    pub losses: Vec<Loss>,
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum TranslateBody {
    Full {
        bytes: Bytes,
        /// Decoded provider response; `None` for errors.
        response: Option<Response>,
        observation: Observation,
    },
    Stream(TranslateStream),
}

// Response headers worth keeping when the body changes format.
const KEEP_RESPONSE: [&str; 4] = [
    "retry-after",
    "request-id",
    "x-request-id",
    "openai-processing-ms",
];

/// Credential the client sent, in any provider's style.
fn client_key(req: &InboundRequest) -> Option<String> {
    if let Some(v) = header(&req.headers, "x-api-key").or(header(&req.headers, "x-goog-api-key")) {
        return Some(v.to_owned());
    }
    if let Some(v) = header(&req.headers, "authorization") {
        return Some(v.strip_prefix("Bearer ").unwrap_or(v).trim().to_owned());
    }
    let query = req.path_and_query.split_once('?')?.1;
    query
        .split('&')
        .find_map(|kv| kv.strip_prefix("key="))
        .map(str::to_owned)
}

fn auth_header(format: Format, key: String) -> (String, String) {
    match format {
        Format::AnthropicMessages => ("x-api-key".into(), key),
        Format::Gemini => ("x-goog-api-key".into(), key),
        Format::OpenAiChat | Format::OpenAiResponses => {
            ("authorization".into(), format!("Bearer {key}"))
        }
    }
}

/// Builds the provider request for an IR request already encoded for `endpoint`.
pub(crate) fn provider_request(
    endpoint: &Endpoint,
    client: Format,
    inbound: &InboundRequest,
    encoded: codec::Encoded,
) -> HttpRequest {
    let target = endpoint.format;
    let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
    if target == Format::AnthropicMessages {
        let version = (client == target)
            .then(|| header(&inbound.headers, "anthropic-version"))
            .flatten()
            .unwrap_or("2023-06-01");
        headers.push(("anthropic-version".into(), version.into()));
        if client == target
            && let Some(beta) = header(&inbound.headers, "anthropic-beta")
        {
            headers.push(("anthropic-beta".into(), beta.into()));
        }
    }
    if let Some(ua) = header(&inbound.headers, "user-agent") {
        headers.push(("user-agent".into(), ua.into()));
    }
    match &endpoint.auth {
        Auth::Inherit => {
            if let Some(key) = client_key(inbound) {
                headers.push(auth_header(target, key));
            }
        }
        Auth::Header { name, value } => headers.push((name.clone(), value.clone())),
        Auth::Bearer(t) => headers.push(("authorization".into(), format!("Bearer {t}"))),
    }
    for (name, value) in &endpoint.headers {
        headers.retain(|(k, _)| !k.eq_ignore_ascii_case(name));
        headers.push((name.clone(), value.clone()));
    }
    HttpRequest {
        method: "POST".into(),
        url: format!(
            "{}{}",
            endpoint.base_url.trim_end_matches('/'),
            encoded.path
        ),
        headers,
        body: encoded.body,
    }
}

fn response_headers(headers: &[(String, String)], streaming: bool) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = headers
        .iter()
        .filter(|(k, _)| {
            KEEP_RESPONSE.iter().any(|h| k.eq_ignore_ascii_case(h))
                || k.to_ascii_lowercase().contains("ratelimit")
        })
        .cloned()
        .collect();
    let ct = if streaming {
        "text/event-stream"
    } else {
        "application/json"
    };
    out.push(("content-type".into(), ct.into()));
    out
}

impl Core {
    /// Decodes an inbound request into IR.
    pub fn decode(&self, req: &InboundRequest) -> Result<(Format, Request), Error> {
        let format = Format::detect(&req.path_and_query).ok_or_else(|| {
            Error::Config(format!("unknown request format: {}", req.path_and_query))
        })?;
        let ir = codec::decode_request(format, &req.path_and_query, &req.body)
            .map_err(|e| Error::Codec(e.0))?;
        Ok((format, ir))
    }

    /// Serves a request in the client's format through an endpoint of another format.
    pub async fn translate(
        &self,
        endpoint: &Endpoint,
        req: InboundRequest,
        opts: TranslateOptions,
        cancel: &CancelToken,
    ) -> Result<TranslateResponse, Error> {
        let (client, ir) = self.decode(&req)?;
        self.translate_request(endpoint, client, ir, &req, opts, cancel)
            .await
    }

    /// Like `translate`, for a request the host already decoded and possibly edited.
    /// `inbound` supplies the client's headers and credentials.
    pub async fn translate_request(
        &self,
        endpoint: &Endpoint,
        client: Format,
        mut ir: Request,
        inbound: &InboundRequest,
        opts: TranslateOptions,
        cancel: &CancelToken,
    ) -> Result<TranslateResponse, Error> {
        let req = inbound;
        if let Some(m) = &opts.model {
            ir.model = m.clone();
        }
        if let Some(namespace) = &endpoint.signature_namespace {
            for message in &mut ir.messages {
                signatures(&mut message.content, &mut |sig| {
                    if sig.namespace.as_deref() == Some(namespace) && sig.format == endpoint.format
                    {
                        sig.namespace = None;
                    }
                });
            }
        }
        let streaming = ir.stream;
        let (encoded, mut losses) =
            encode(endpoint.format, &ir, &opts.encode, opts.cache.as_ref())?;
        if opts.loss_policy == LossPolicy::Error && !losses.is_empty() {
            return Err(Error::Loss(losses));
        }
        let http = provider_request(endpoint, client, req, encoded);
        let Exchange {
            ctx,
            pipe,
            status,
            headers,
            body,
            attempts,
            ..
        } = self.exchange(endpoint, http, streaming, cancel).await?;
        let target = endpoint.format;

        if streaming && status < 400 {
            let mut stream = TranslateStream::new(
                ctx,
                client,
                body,
                opts.response_model,
                pipe,
                cancel.clone(),
                opts.strict_stream,
            );
            stream.drop_reasoning = opts.drop_reasoning;
            stream.signature_namespace = endpoint.signature_namespace.clone();
            stream.client = client;
            return Ok(TranslateResponse {
                status,
                headers: response_headers(&headers, true),
                body: TranslateBody::Stream(stream),
                attempts,
                losses,
            });
        }

        let mut bytes = read_all(body, cancel).await?;
        for h in &pipe.body {
            h.on_body(&ctx, status, &mut bytes)
                .map_err(|m| plugin_error(h.name(), m))?;
        }
        let observation = Observation::from_body(target, status, &bytes);
        for ev in observed_events(&observation) {
            pipe.observe(&ctx, &ev);
        }
        if status >= 400 {
            let err = ProviderError::parse(target, Some(status), bytes);
            let (_, body) = codec::encode_error(client, &err);
            return Ok(TranslateResponse {
                status,
                headers: response_headers(&headers, false),
                body: TranslateBody::Full {
                    bytes: body,
                    response: None,
                    observation,
                },
                attempts,
                losses,
            });
        }
        let mut resp = codec::decode_response(target, &bytes).map_err(|e| Error::Codec(e.0))?;
        if let Some(m) = &opts.response_model {
            resp.model = Some(m.clone());
        }
        if opts.drop_reasoning {
            resp.content.retain(|b| !matches!(b, Block::Thinking(_)));
        }
        let mut client_resp = resp.clone();
        if let Some(namespace) = &endpoint.signature_namespace
            && matches!(client, Format::AnthropicMessages | Format::OpenAiResponses)
            && client != target
        {
            signatures(&mut client_resp.content, &mut |sig| {
                *sig = sig.tunnel(namespace, client);
            });
        }
        let (out, resp_losses) = codec::encode_response(client, &client_resp);
        losses.extend(resp_losses.0);
        Ok(TranslateResponse {
            status,
            headers: response_headers(&headers, false),
            body: TranslateBody::Full {
                bytes: out,
                response: Some(resp),
                observation,
            },
            attempts,
            losses,
        })
    }
}

fn signatures(blocks: &mut [Block], f: &mut impl FnMut(&mut Opaque)) {
    for block in blocks {
        match block {
            Block::Thinking(t) => {
                if let Some(sig) = &mut t.sig {
                    f(sig);
                }
            }
            Block::Text { sig: Some(sig), .. } => f(sig),
            Block::ToolCall(t) => {
                if let Some(sig) = &mut t.sig {
                    f(sig);
                }
            }
            Block::ToolResult(t) => signatures(&mut t.content, f),
            _ => {}
        }
    }
}

/// Client-format byte stream converted from the provider stream.
pub struct TranslateStream {
    events: EventStream,
    encoder: Box<dyn StreamEncoder>,
    response_model: Option<String>,
    done: bool,
    drop_reasoning: bool,
    // Thinking block being dropped, and how many were dropped before the current one.
    skipping: Option<usize>,
    dropped: usize,
    signature_namespace: Option<String>,
    client: Format,
}

impl std::fmt::Debug for TranslateStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranslateStream")
            .field("provider", &self.events.ctx.format)
            .field("done", &self.done)
            .finish()
    }
}

impl TranslateStream {
    fn new(
        ctx: RequestCtx,
        client: Format,
        body: BoxStream<'static, Result<Bytes, TransportError>>,
        response_model: Option<String>,
        pipe: Arc<Pipeline>,
        cancel: CancelToken,
        strict: bool,
    ) -> Self {
        Self {
            events: EventStream::new(ctx, body, pipe, cancel, strict),
            encoder: codec::stream_encoder(client),
            response_model,
            done: false,
            drop_reasoning: false,
            skipping: None,
            dropped: 0,
            signature_namespace: None,
            client,
        }
    }

    // Drops thinking blocks and renumbers the rest so indices stay dense.
    fn keep(&mut self, ev: &mut StreamEvent) -> bool {
        if !self.drop_reasoning {
            return true;
        }
        let index = match ev {
            StreamEvent::BlockStart {
                index,
                block: BlockStart::Thinking { .. },
            } => {
                self.skipping = Some(*index);
                self.dropped += 1;
                return false;
            }
            StreamEvent::BlockStart { index, .. } | StreamEvent::Delta { index, .. } => index,
            StreamEvent::BlockEnd { index } => {
                if self.skipping == Some(*index) {
                    self.skipping = None;
                    return false;
                }
                index
            }
            _ => return true,
        };
        if self.skipping == Some(*index) {
            return false;
        }
        *index -= self.dropped;
        true
    }

    /// Provider-side facts observed so far.
    pub fn observation(&self) -> Observation {
        self.events.shared.lock().unwrap().clone()
    }

    pub fn observation_handle(&self) -> Arc<Mutex<Observation>> {
        self.events.shared.clone()
    }

    fn emit(&mut self, events: Vec<StreamEvent>, out: &mut BytesMut) {
        for mut ev in events {
            if let (
                Some(namespace),
                StreamEvent::Delta {
                    delta: Delta::Sig(sig),
                    ..
                },
            ) = (&self.signature_namespace, &mut ev)
                && matches!(
                    self.client,
                    Format::AnthropicMessages | Format::OpenAiResponses
                )
                && sig.format != self.client
            {
                *sig = sig.tunnel(namespace, self.client);
            }
            if !self.keep(&mut ev) {
                continue;
            }
            if let (StreamEvent::Start { model, .. }, Some(m)) = (&mut ev, &self.response_model) {
                *model = Some(m.clone());
            }
            for b in self.encoder.encode(&ev) {
                out.extend_from_slice(&b);
            }
        }
    }
}

impl Stream for TranslateStream {
    type Item = Result<Bytes, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            if this.done {
                return Poll::Ready(None);
            }
            let mut out = BytesMut::new();
            match this.events.poll_chunk(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(events))) => this.emit(events, &mut out),
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    this.done = true;
                    for b in this.encoder.finish() {
                        out.extend_from_slice(&b);
                    }
                }
            }
            if !out.is_empty() {
                return Poll::Ready(Some(Ok(out.freeze())));
            }
        }
    }
}
