use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures_channel::oneshot;
use ncore::rt::{BoxFuture, MaybeSend, MaybeSync};
use serde_json::{Value, json};

use crate::error::McpError;
use crate::jsonrpc::{self, Incoming, RpcError};

/// Something the host may want to see; never required for correctness.
#[derive(Debug, Clone, PartialEq)]
pub enum McpEvent {
    Connected {
        server: String,
        version: String,
        name: Option<String>,
    },
    /// The server was started again after it died.
    Restarted {
        server: String,
        reason: String,
    },
    Stderr {
        server: String,
        line: String,
    },
    /// Something odd the client tolerated.
    Warning {
        server: String,
        message: String,
    },
    Progress {
        server: String,
        token: String,
        progress: f64,
        total: Option<f64>,
        message: Option<String>,
    },
    Log {
        server: String,
        level: String,
        data: Value,
    },
    ToolsChanged {
        server: String,
    },
}

pub type EventSink = Arc<dyn Fn(McpEvent) + Send + Sync>;

/// Answers server-to-client requests: legacy requests and MRTR input requests.
pub trait ClientHandler: MaybeSend + MaybeSync {
    /// Client capabilities to declare, e.g. `{"elicitation": {}}`.
    fn capabilities(&self) -> Value {
        json!({})
    }

    /// Handles `elicitation/create`, `sampling/createMessage`, `roots/list` and the like.
    fn handle<'a>(
        &'a self,
        server: &'a str,
        method: &'a str,
        params: Value,
    ) -> BoxFuture<'a, Result<Value, RpcError>> {
        let _ = (server, params);
        Box::pin(async move { Err(RpcError::method_not_found(method)) })
    }
}

/// Handler that supports nothing.
pub struct NoHandler;

impl ClientHandler for NoHandler {}

type Reply = Result<Value, RpcError>;

/// Correlates responses with pending requests and routes everything else.
pub(crate) struct Inbox {
    pub server: String,
    pending: Mutex<HashMap<String, oneshot::Sender<Reply>>>,
    closed: Mutex<Option<McpError>>,
    pub events: EventSink,
    pub handler: Arc<dyn ClientHandler>,
    /// Bumped on every progress notification; long calls use it to extend their deadline.
    pub progress: AtomicU64,
    pub tools_changed: AtomicBool,
}

impl Inbox {
    pub fn new(server: String, events: EventSink, handler: Arc<dyn ClientHandler>) -> Self {
        Self {
            server,
            pending: Mutex::default(),
            closed: Mutex::default(),
            events,
            handler,
            progress: AtomicU64::new(0),
            tools_changed: AtomicBool::new(false),
        }
    }

    pub fn warn(&self, message: impl Into<String>) {
        (self.events)(McpEvent::Warning {
            server: self.server.clone(),
            message: message.into(),
        });
    }

    pub fn expect(&self, id: u64) -> Result<oneshot::Receiver<Reply>, McpError> {
        if let Some(e) = self.closed.lock().unwrap().clone() {
            return Err(e);
        }
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id.to_string(), tx);
        Ok(rx)
    }

    pub fn forget(&self, id: u64) {
        self.pending.lock().unwrap().remove(&id.to_string());
    }

    /// Fails every pending request and refuses new ones.
    pub fn close(&self, err: McpError) {
        self.closed.lock().unwrap().get_or_insert(err);
        self.pending.lock().unwrap().clear();
    }

    pub fn closed(&self) -> Option<McpError> {
        self.closed.lock().unwrap().clone()
    }

    /// Routes one received JSON value; returns replies the wire must send back.
    pub fn dispatch(self: &Arc<Self>, v: Value) -> Vec<BoxFuture<'static, Value>> {
        let (msgs, warnings) = jsonrpc::parse(v);
        for w in warnings {
            self.warn(w);
        }
        let mut replies = Vec::new();
        for m in msgs {
            match m {
                Incoming::Response { id, result } => {
                    match self.pending.lock().unwrap().remove(&id) {
                        Some(tx) => {
                            let _ = tx.send(result);
                        }
                        None => self.warn(format!("response for unknown request id {id}")),
                    }
                }
                Incoming::Notification { method, params } => self.notify(&method, params),
                Incoming::Request { id, method, params } => {
                    let me = self.clone();
                    replies.push(Box::pin(async move {
                        let result = match method.as_str() {
                            "ping" => Ok(json!({})),
                            _ => me.handler.handle(&me.server, &method, params).await,
                        };
                        jsonrpc::response(id, result)
                    }) as BoxFuture<'static, Value>);
                }
            }
        }
        replies
    }

    fn notify(&self, method: &str, p: Value) {
        let server = self.server.clone();
        match method {
            "notifications/progress" => {
                self.progress.fetch_add(1, Ordering::Relaxed);
                let token = match &p["progressToken"] {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                (self.events)(McpEvent::Progress {
                    server,
                    token,
                    progress: p["progress"].as_f64().unwrap_or(0.0),
                    total: p["total"].as_f64(),
                    message: p["message"].as_str().map(str::to_owned),
                });
            }
            "notifications/message" => (self.events)(McpEvent::Log {
                server,
                level: p["level"].as_str().unwrap_or("info").to_owned(),
                data: p.get("data").cloned().unwrap_or(Value::Null),
            }),
            "notifications/tools/list_changed" => {
                self.tools_changed.store(true, Ordering::Relaxed);
                (self.events)(McpEvent::ToolsChanged { server });
            }
            _ => {}
        }
    }
}

/// Message to send plus what the HTTP binding mirrors into headers.
pub(crate) struct Outgoing {
    pub body: Value,
    pub method: String,
    pub name: Option<String>,
    pub headers: Vec<(String, String)>,
    pub is_request: bool,
}

/// How far a send got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sent {
    /// The reply arrives later through the inbox.
    Later,
    /// Every reply for this message has already been dispatched.
    Done,
}

/// Moves messages to and from one server.
pub(crate) trait Wire: Send + Sync {
    fn send<'a>(&'a self, msg: &'a Outgoing) -> BoxFuture<'a, Result<Sent, McpError>>;
    fn close(&self) -> BoxFuture<'_, ()>;
    fn alive(&self) -> bool;
    /// Recent stderr lines for diagnostics.
    fn stderr(&self) -> Vec<String> {
        Vec::new()
    }
}
