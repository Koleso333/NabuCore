use std::fmt;
use std::sync::Arc;

use bytes::Bytes;
use ncore_core::{Format, Observed, SseEvent};

use crate::error::Error;
use crate::rt::{BoxFuture, HttpRequest, HttpResponse, MaybeSend, MaybeSync, Transport, boxed};

/// Per-request facts shared with plugins.
#[derive(Debug, Clone)]
pub struct RequestCtx {
    pub id: u64,
    pub format: Format,
    pub streaming: bool,
    /// Attempt number of the core retry loop, starting at 1.
    pub attempt: u32,
}

/// Status and headers of a provider response before the body is read.
#[derive(Debug, Clone)]
pub struct ResponseHead {
    pub status: u16,
    pub headers: Vec<(String, String)>,
}

/// What happens to a stream event after a hook saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventAction {
    Keep,
    Drop,
}

/// Which hook methods a plugin implements, so the core skips the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interests {
    pub request: bool,
    pub response: bool,
    pub body: bool,
    pub stream: bool,
    pub observed: bool,
}

impl Interests {
    pub const ALL: Self = Self {
        request: true,
        response: true,
        body: true,
        stream: true,
        observed: true,
    };
    pub const NONE: Self = Self {
        request: false,
        response: false,
        body: false,
        stream: false,
        observed: false,
    };
}

/// Format-aware plugin hooks; requests run them in registration order, responses in reverse.
pub trait Hook: MaybeSend + MaybeSync {
    fn name(&self) -> &str;

    fn interests(&self) -> Interests {
        Interests::ALL
    }

    /// Edits the outgoing request once, before retries and middleware.
    fn on_request<'a>(
        &'a self,
        _ctx: &'a RequestCtx,
        _req: &'a mut HttpRequest,
    ) -> BoxFuture<'a, Result<(), String>> {
        boxed(async { Ok(()) })
    }

    /// Edits status and headers of the final response.
    fn on_response<'a>(
        &'a self,
        _ctx: &'a RequestCtx,
        _head: &'a mut ResponseHead,
    ) -> BoxFuture<'a, Result<(), String>> {
        boxed(async { Ok(()) })
    }

    /// Edits a complete (non-streamed or error) response body.
    fn on_body(&self, _ctx: &RequestCtx, _status: u16, _body: &mut Bytes) -> Result<(), String> {
        Ok(())
    }

    /// Edits or drops one stream event; runs before observation.
    fn on_stream_event(
        &self,
        _ctx: &RequestCtx,
        _ev: &mut SseEvent,
    ) -> Result<EventAction, String> {
        Ok(EventAction::Keep)
    }

    /// Called for each fact extracted from the response or stream.
    fn on_observed(&self, _ctx: &RequestCtx, _event: &Observed) {}
}

/// HTTP-level plugin wrapping the transport; may retry, rewrite or answer without the network.
pub trait Middleware: MaybeSend + MaybeSync {
    fn name(&self) -> &str;

    fn handle<'a>(
        &'a self,
        ctx: &'a RequestCtx,
        req: HttpRequest,
        next: Next<'a>,
    ) -> BoxFuture<'a, Result<HttpResponse, Error>>;
}

/// Rest of the middleware chain; can be run any number of times.
#[derive(Clone, Copy)]
pub struct Next<'a> {
    ctx: &'a RequestCtx,
    rest: &'a [Arc<dyn Middleware>],
    transport: &'a dyn Transport,
}

impl<'a> Next<'a> {
    pub(crate) fn new(
        ctx: &'a RequestCtx,
        rest: &'a [Arc<dyn Middleware>],
        transport: &'a dyn Transport,
    ) -> Self {
        Self {
            ctx,
            rest,
            transport,
        }
    }

    pub fn run(self, req: HttpRequest) -> BoxFuture<'a, Result<HttpResponse, Error>> {
        match self.rest.split_first() {
            Some((m, rest)) => m.handle(self.ctx, req, Next { rest, ..self }),
            None => boxed(async move { self.transport.send(req).await.map_err(Error::from) }),
        }
    }
}

/// Plugins attached to a `Core` or an `Endpoint`.
#[derive(Clone, Default)]
pub struct Plugins {
    pub(crate) hooks: Vec<Arc<dyn Hook>>,
    pub(crate) middleware: Vec<Arc<dyn Middleware>>,
}

impl Plugins {
    pub fn hook(&mut self, h: Arc<dyn Hook>) {
        self.hooks.push(h);
    }

    pub fn middleware(&mut self, m: Arc<dyn Middleware>) {
        self.middleware.push(m);
    }

    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty() && self.middleware.is_empty()
    }
}

impl fmt::Debug for Plugins {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plugins")
            .field(
                "hooks",
                &self.hooks.iter().map(|h| h.name()).collect::<Vec<_>>(),
            )
            .field(
                "middleware",
                &self.middleware.iter().map(|m| m.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Hooks of one request, pre-split by interest; response-side lists are already reversed.
#[derive(Clone, Default)]
pub(crate) struct Pipeline {
    pub request: Vec<Arc<dyn Hook>>,
    pub response: Vec<Arc<dyn Hook>>,
    pub body: Vec<Arc<dyn Hook>>,
    pub stream: Vec<Arc<dyn Hook>>,
    pub observed: Vec<Arc<dyn Hook>>,
    pub middleware: Vec<Arc<dyn Middleware>>,
}

impl Pipeline {
    /// Core plugins wrap endpoint plugins.
    pub fn new(outer: &Plugins, inner: &Plugins) -> Self {
        let mut p = Pipeline::default();
        for h in outer.hooks.iter().chain(&inner.hooks) {
            let i = h.interests();
            for (on, list) in [
                (i.request, &mut p.request),
                (i.response, &mut p.response),
                (i.body, &mut p.body),
                (i.stream, &mut p.stream),
                (i.observed, &mut p.observed),
            ] {
                if on {
                    list.push(h.clone());
                }
            }
        }
        p.response.reverse();
        p.body.reverse();
        p.stream.reverse();
        p.observed.reverse();
        p.middleware = outer
            .middleware
            .iter()
            .chain(&inner.middleware)
            .cloned()
            .collect();
        p
    }

    pub fn observe(&self, ctx: &RequestCtx, ev: &Observed) {
        for h in &self.observed {
            h.on_observed(ctx, ev);
        }
    }
}

pub(crate) fn plugin_error(name: &str, message: String) -> Error {
    Error::Plugin {
        name: name.to_owned(),
        message,
    }
}
