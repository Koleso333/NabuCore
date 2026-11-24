use std::sync::{Arc, Mutex};

use bytes::{Bytes, BytesMut};
use futures_util::StreamExt;
use ncore::core::sse::SseParser;
use ncore::rt::BoxFuture;
use ncore::{HttpRequest, HttpResponse, Transport};
use serde_json::Value;

use crate::error::McpError;
use crate::jsonrpc::{self, RpcError};
use crate::tool::header_value;
use crate::wire::{Inbox, Outgoing, Sent, Wire};

const MAX_ERROR_BODY: usize = 64 * 1024;

/// Streamable HTTP binding, both the stateless (2026-07-28) and the session-based shape.
pub(crate) struct HttpWire {
    transport: Arc<dyn Transport>,
    url: String,
    headers: Vec<(String, String)>,
    inbox: Arc<Inbox>,
    max_bytes: usize,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    version: Option<String>,
    modern: bool,
    session: Option<String>,
}

impl HttpWire {
    pub fn new(
        transport: Arc<dyn Transport>,
        url: String,
        headers: Vec<(String, String)>,
        inbox: Arc<Inbox>,
        max_bytes: usize,
    ) -> Self {
        Self {
            transport,
            url,
            headers,
            inbox,
            max_bytes,
            state: Mutex::default(),
        }
    }

    /// Sets the protocol version header; `modern` adds the per-request metadata headers.
    pub fn set_version(&self, version: Option<String>, modern: bool) {
        let mut s = self.state.lock().unwrap();
        s.version = version;
        s.modern = modern;
        if modern {
            s.session = None;
        }
    }

    pub fn has_session(&self) -> bool {
        self.state.lock().unwrap().session.is_some()
    }

    pub fn reset_session(&self) {
        self.state.lock().unwrap().session = None;
    }

    fn request(&self, msg: &Outgoing) -> HttpRequest {
        let mut headers = vec![
            ("content-type".into(), "application/json".into()),
            (
                "accept".into(),
                "application/json, text/event-stream".into(),
            ),
        ];
        headers.extend(self.headers.iter().cloned());
        let s = self.state.lock().unwrap();
        if let Some(v) = &s.version {
            headers.push(("MCP-Protocol-Version".into(), v.clone()));
        }
        if let Some(id) = &s.session {
            headers.push(("Mcp-Session-Id".into(), id.clone()));
        }
        if s.modern && !msg.method.is_empty() {
            headers.push(("Mcp-Method".into(), msg.method.clone()));
            if let Some(n) = &msg.name {
                headers.push(("Mcp-Name".into(), header_value(n)));
            }
            headers.extend(msg.headers.iter().cloned());
        }
        HttpRequest {
            method: "POST".into(),
            url: self.url.clone(),
            headers,
            body: Bytes::from(msg.body.to_string()),
        }
    }

    async fn post(&self, msg: &Outgoing) -> Result<Sent, McpError> {
        let resp = self.transport.send(self.request(msg)).await.map_err(|e| {
            if e.connect {
                McpError::Connect(e.message)
            } else {
                McpError::closed(e.message)
            }
        })?;
        if let Some(id) = resp
            .header("mcp-session-id")
            .filter(|_| !self.state.lock().unwrap().modern)
        {
            self.state.lock().unwrap().session = Some(id.to_owned());
        }
        let status = resp.status;
        if status == 401 || status == 403 {
            return Err(McpError::Auth {
                status,
                challenge: resp.header("www-authenticate").map(str::to_owned),
            });
        }
        if status >= 300 {
            let body = read_all(resp, MAX_ERROR_BODY).await.unwrap_or_default();
            let body = String::from_utf8_lossy(&body).into_owned();
            let rpc = serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|v| v.get("error").cloned())
                .filter(Value::is_object)
                .map(|e| {
                    match jsonrpc::parse(serde_json::json!({"id": 0, "error": e}))
                        .0
                        .pop()
                    {
                        Some(jsonrpc::Incoming::Response { result: Err(e), .. }) => e,
                        _ => RpcError::new(jsonrpc::INTERNAL_ERROR, "unreadable error"),
                    }
                });
            let rpc = rpc.map(Box::new);
            return Err(McpError::Http { status, body, rpc });
        }
        if status == 202 || status == 204 || !msg.is_request {
            return Ok(Sent::Done);
        }
        let id = jsonrpc::id_key(&msg.body["id"]).unwrap_or_default();
        let ctype = resp
            .header("content-type")
            .unwrap_or("")
            .to_ascii_lowercase();
        if ctype.contains("text/event-stream") {
            self.read_sse(resp, &id).await?;
        } else {
            let body = read_all(resp, self.max_bytes).await?;
            let text = trim(&body);
            if text.starts_with(b"data:") || text.starts_with(b"event:") || text.starts_with(b":") {
                // Missing or wrong content type on an SSE body.
                self.sse_events(&mut SseParser::new(), &body, &id).await;
                self.sse_tail(&mut SseParser::new(), &id).await;
            } else if !text.is_empty() {
                let v: Value = serde_json::from_slice(text).map_err(|e| {
                    McpError::Protocol(format!(
                        "response is not JSON ({e}): {}",
                        crate::content::clip(&String::from_utf8_lossy(text), 200)
                    ))
                })?;
                self.deliver(v).await;
            }
        }
        Ok(Sent::Done)
    }

    async fn read_sse(&self, resp: HttpResponse, id: &str) -> Result<(), McpError> {
        let mut parser = SseParser::new();
        let mut body = resp.body;
        let mut total = 0usize;
        while let Some(chunk) = body.next().await {
            let chunk =
                chunk.map_err(|e| McpError::closed(format!("response stream broke: {e}")))?;
            total += chunk.len();
            if total > self.max_bytes {
                return Err(McpError::Protocol(format!(
                    "response exceeds {} bytes",
                    self.max_bytes
                )));
            }
            if self.sse_events(&mut parser, &chunk, id).await {
                // Got our response; some servers never close the stream.
                return Ok(());
            }
        }
        self.sse_tail(&mut parser, id).await;
        Ok(())
    }

    // Returns true once the response with `id` has been delivered.
    async fn sse_events(&self, parser: &mut SseParser, chunk: &[u8], id: &str) -> bool {
        for ev in parser.feed(chunk) {
            let data = ev.data.trim();
            if !ev.has_data() || data.is_empty() {
                continue;
            }
            match serde_json::from_str::<Value>(data) {
                Ok(v) => {
                    let ours = answers(&v, id);
                    self.deliver(v).await;
                    if ours {
                        return true;
                    }
                }
                Err(_) => self.inbox.warn(format!(
                    "ignored non-JSON event: {}",
                    crate::content::clip(data, 200)
                )),
            }
        }
        false
    }

    // An unterminated last event is still dispatched.
    async fn sse_tail(&self, parser: &mut SseParser, id: &str) {
        let rest = parser.finish();
        if !rest.is_empty() {
            let mut buf = BytesMut::from(&rest[..]);
            buf.extend_from_slice(b"\n\n");
            self.sse_events(&mut SseParser::new(), &buf, id).await;
        }
    }

    async fn deliver(&self, v: Value) {
        for reply in self.inbox.dispatch(v) {
            let body = reply.await;
            let out = Outgoing {
                body,
                method: String::new(),
                name: None,
                headers: Vec::new(),
                is_request: false,
            };
            match self.transport.send(self.request(&out)).await {
                Ok(r) if r.status < 300 => {}
                Ok(r) => self
                    .inbox
                    .warn(format!("server rejected our reply with HTTP {}", r.status)),
                Err(e) => self
                    .inbox
                    .warn(format!("cannot answer a server request: {e}")),
            }
        }
    }
}

// True when `v` is the response to the request with id key `id`.
fn answers(v: &Value, id: &str) -> bool {
    v.get("method").is_none() && jsonrpc::id_key(&v["id"]).as_deref() == Some(id)
}

fn trim(b: &[u8]) -> &[u8] {
    let start = b
        .iter()
        .position(|c| !c.is_ascii_whitespace())
        .unwrap_or(b.len());
    &b[start..]
}

async fn read_all(resp: HttpResponse, max: usize) -> Result<Bytes, McpError> {
    let mut buf = BytesMut::new();
    let mut body = resp.body;
    while let Some(chunk) = body.next().await {
        let chunk = chunk.map_err(|e| McpError::closed(format!("response broke: {e}")))?;
        if buf.len() + chunk.len() > max {
            return Err(McpError::Protocol(format!("response exceeds {max} bytes")));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf.freeze())
}

impl Wire for HttpWire {
    fn send<'a>(&'a self, msg: &'a Outgoing) -> BoxFuture<'a, Result<Sent, McpError>> {
        Box::pin(self.post(msg))
    }

    fn close(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            let session = self.state.lock().unwrap().session.take();
            if let Some(id) = session {
                let mut headers = self.headers.clone();
                headers.push(("Mcp-Session-Id".into(), id));
                let req = HttpRequest {
                    method: "DELETE".into(),
                    url: self.url.clone(),
                    headers,
                    body: Bytes::new(),
                };
                if let Ok(r) = self.transport.send(req).await {
                    drop(r);
                }
            }
        })
    }

    fn alive(&self) -> bool {
        true
    }
}
