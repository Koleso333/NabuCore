use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::future::{Either, select};
use futures_util::lock::Mutex as AsyncMutex;
use ncore::{CancelToken, Clock, ProcessLauncher, ProcessSpec, Spawner, Transport};
use serde_json::{Map, Value, json};

use crate::content::CallResult;
use crate::error::McpError;
use crate::http::HttpWire;
use crate::jsonrpc::{self, HEADER_MISMATCH, UNSUPPORTED_VERSION};
use crate::stdio::{StdioParams, StdioWire};
use crate::tool::{McpTool, param_headers};
use crate::wire::{ClientHandler, EventSink, Inbox, McpEvent, NoHandler, Outgoing, Sent, Wire};

/// Stateless protocol versions, preferred first.
pub const MODERN_VERSIONS: &[&str] = &["2026-07-28"];
/// Handshake-based protocol versions, preferred first.
pub const LEGACY_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

const META: &str = "io.modelcontextprotocol/";
const MAX_PAGES: usize = 100;
const MAX_INPUT_ROUNDS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerTransport {
    Stdio(ProcessSpec),
    /// Streamable HTTP endpoint; `headers` go on every request (e.g. authorization).
    Http {
        url: String,
        headers: Vec<(String, String)>,
    },
}

/// Which protocol generation to speak; `Auto` probes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Era {
    #[default]
    Auto,
    Modern,
    Legacy,
}

#[derive(Debug, Clone, Copy)]
pub struct Timeouts {
    /// Starting the server and agreeing on a protocol version.
    pub connect: Duration,
    /// Waiting for a legacy server to reject the modern probe.
    pub probe: Duration,
    /// Any request other than `tools/call`.
    pub request: Duration,
    /// `tools/call` without progress; each progress notification extends it.
    pub call: Duration,
    /// Hard cap for a `tools/call`, progress or not.
    pub call_max: Duration,
    /// Grace period after closing stdin before the process is killed.
    pub shutdown: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(60),
            probe: Duration::from_secs(5),
            request: Duration::from_secs(60),
            call: Duration::from_secs(300),
            call_max: Duration::from_secs(3600),
            shutdown: Duration::from_secs(2),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub name: String,
    pub transport: ServerTransport,
    pub era: Era,
    pub timeouts: Timeouts,
    /// Restarts in a row without a successful request before giving up.
    pub max_restarts: u32,
    /// Largest single message accepted from the server.
    pub max_message_bytes: usize,
    /// Stderr lines kept for error reports.
    pub stderr_lines: usize,
    pub client_name: String,
    pub client_version: String,
}

impl ServerConfig {
    pub fn new(name: impl Into<String>, transport: ServerTransport) -> Self {
        Self {
            name: name.into(),
            transport,
            era: Era::Auto,
            timeouts: Timeouts::default(),
            max_restarts: 3,
            max_message_bytes: 64 * 1024 * 1024,
            stderr_lines: 50,
            client_name: "nabucore".into(),
            client_version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    pub fn stdio(name: impl Into<String>, program: impl Into<String>, args: &[&str]) -> Self {
        Self::new(
            name,
            ServerTransport::Stdio(ProcessSpec {
                program: program.into(),
                args: args.iter().map(|a| a.to_string()).collect(),
                ..ProcessSpec::default()
            }),
        )
    }

    pub fn http(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self::new(
            name,
            ServerTransport::Http {
                url: url.into(),
                headers: Vec::new(),
            },
        )
    }
}

/// Runtime pieces the client needs; `transport` for HTTP servers, `launcher` for stdio ones.
#[derive(Clone)]
pub struct Runtime {
    pub clock: Arc<dyn Clock>,
    pub spawner: Arc<dyn Spawner>,
    pub transport: Option<Arc<dyn Transport>>,
    pub launcher: Option<Arc<dyn ProcessLauncher>>,
    pub events: EventSink,
    pub handler: Arc<dyn ClientHandler>,
}

impl Runtime {
    pub fn new(clock: Arc<dyn Clock>, spawner: Arc<dyn Spawner>) -> Self {
        Self {
            clock,
            spawner,
            transport: None,
            launcher: None,
            events: Arc::new(|_| {}),
            handler: Arc::new(NoHandler),
        }
    }
}

/// What the server said about itself.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ServerInfo {
    pub protocol: String,
    pub modern: bool,
    pub name: Option<String>,
    pub version: Option<String>,
    pub capabilities: Value,
    pub instructions: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct CallOptions {
    pub cancel: CancelToken,
    /// Overrides `Timeouts::call`.
    pub timeout: Option<Duration>,
}

/// Started wire, plus its HTTP side for header and session control.
type Wires = (Arc<dyn Wire>, Option<Arc<HttpWire>>);

struct Conn {
    wire: Arc<dyn Wire>,
    http: Option<Arc<HttpWire>>,
    inbox: Arc<Inbox>,
    info: ServerInfo,
}

struct Inner {
    cfg: ServerConfig,
    rt: Runtime,
    conn: AsyncMutex<Option<Arc<Conn>>>,
    next_id: AtomicU64,
    failures: AtomicU32,
    /// Era found by the first successful connect, reused on restarts.
    era: Mutex<Option<(bool, String)>>,
}

/// Connection to one MCP server that survives crashes and restarts it on demand.
#[derive(Clone)]
pub struct McpClient {
    inner: Arc<Inner>,
}

enum Probe {
    Modern(String, Value),
    Legacy,
}

impl McpClient {
    /// Starts the server and agrees on a protocol version.
    pub async fn connect(cfg: ServerConfig, rt: Runtime) -> Result<Self, McpError> {
        let c = Self {
            inner: Arc::new(Inner {
                cfg,
                rt,
                conn: AsyncMutex::new(None),
                next_id: AtomicU64::new(1),
                failures: AtomicU32::new(0),
                era: Mutex::new(None),
            }),
        };
        c.conn().await?;
        Ok(c)
    }

    pub fn name(&self) -> &str {
        &self.inner.cfg.name
    }

    /// Info from the current connection.
    pub async fn info(&self) -> Option<ServerInfo> {
        self.inner
            .conn
            .lock()
            .await
            .as_ref()
            .map(|c| c.info.clone())
    }

    /// True once since the server last announced a tool list change.
    pub async fn take_tools_changed(&self) -> bool {
        match self.inner.conn.lock().await.as_ref() {
            Some(c) => c.inbox.tools_changed.swap(false, Ordering::Relaxed),
            None => false,
        }
    }

    pub async fn close(&self) {
        let conn = self.inner.conn.lock().await.take();
        if let Some(c) = conn {
            c.inbox
                .close(McpError::closed("client closed the connection"));
            c.wire.close().await;
        }
    }

    /// Allows restarts again after the client gave up on a crashing server.
    pub fn reset_failures(&self) {
        self.inner.failures.store(0, Ordering::SeqCst);
    }

    fn emit(&self, e: McpEvent) {
        (self.inner.rt.events)(e);
    }

    async fn conn(&self) -> Result<Arc<Conn>, McpError> {
        let inner = &self.inner;
        let mut slot = inner.conn.lock().await;
        let mut why = None;
        if let Some(c) = slot.as_ref() {
            match c.inbox.closed() {
                None if c.wire.alive() => return Ok(c.clone()),
                closed => why = Some(closed.unwrap_or_else(|| McpError::closed("server stopped"))),
            }
            inner.failures.fetch_add(1, Ordering::SeqCst);
        }
        let n = inner.failures.load(Ordering::SeqCst);
        if n > inner.cfg.max_restarts {
            let (reason, stderr) = match &why {
                Some(McpError::Closed { reason, stderr }) => (reason.clone(), stderr.clone()),
                Some(e) => (e.to_string(), Vec::new()),
                None => ("the server keeps failing to start".into(), Vec::new()),
            };
            return Err(McpError::Closed {
                reason: format!(
                    "gave up after {} restarts: {reason}",
                    inner.cfg.max_restarts
                ),
                stderr,
            });
        }
        if n > 0 {
            // Back off so a crash loop does not spin.
            inner
                .rt
                .clock
                .sleep(Duration::from_millis(250 << n.min(5)))
                .await;
        }
        if let Some(err) = &why {
            self.emit(McpEvent::Restarted {
                server: inner.cfg.name.clone(),
                reason: err.to_string(),
            });
        }
        if let Some(old) = slot.take() {
            old.wire.close().await;
        }
        let c = match self
            .timed(inner.cfg.timeouts.connect, "connect", self.open())
            .await
        {
            Ok(c) => c,
            Err(e) => {
                inner.failures.fetch_add(1, Ordering::SeqCst);
                return Err(e);
            }
        };
        self.emit(McpEvent::Connected {
            server: inner.cfg.name.clone(),
            version: c.info.protocol.clone(),
            name: c.info.name.clone(),
        });
        *slot = Some(c.clone());
        Ok(c)
    }

    async fn timed<T>(
        &self,
        after: Duration,
        method: &str,
        f: impl Future<Output = Result<T, McpError>>,
    ) -> Result<T, McpError> {
        let f = std::pin::pin!(f);
        match select(f, self.inner.rt.clock.sleep(after)).await {
            Either::Left((r, _)) => r,
            Either::Right(_) => Err(McpError::Timeout {
                method: method.into(),
                after,
            }),
        }
    }

    fn start_wire(&self, inbox: &Arc<Inbox>) -> Result<Wires, McpError> {
        let cfg = &self.inner.cfg;
        let rt = &self.inner.rt;
        match &cfg.transport {
            ServerTransport::Stdio(spec) => {
                let launcher = rt.launcher.as_deref().ok_or_else(|| {
                    McpError::Unsupported("no process launcher on this platform".into())
                })?;
                let w = StdioWire::start(
                    spec,
                    launcher,
                    rt.spawner.as_ref(),
                    rt.clock.clone(),
                    inbox.clone(),
                    StdioParams {
                        max_line: cfg.max_message_bytes,
                        stderr_lines: cfg.stderr_lines,
                        shutdown: cfg.timeouts.shutdown,
                    },
                )?;
                Ok((w, None))
            }
            ServerTransport::Http { url, headers } => {
                let t = rt
                    .transport
                    .clone()
                    .ok_or_else(|| McpError::Unsupported("no HTTP transport configured".into()))?;
                let w = Arc::new(HttpWire::new(
                    t,
                    url.clone(),
                    headers.clone(),
                    inbox.clone(),
                    cfg.max_message_bytes,
                ));
                Ok((w.clone(), Some(w)))
            }
        }
    }

    fn new_inbox(&self) -> Arc<Inbox> {
        let rt = &self.inner.rt;
        Arc::new(Inbox::new(
            self.inner.cfg.name.clone(),
            rt.events.clone(),
            rt.handler.clone(),
        ))
    }

    async fn open(&self) -> Result<Arc<Conn>, McpError> {
        let inbox = self.new_inbox();
        let (wire, http) = self.start_wire(&inbox)?;
        let mut conn = Conn {
            wire,
            http,
            inbox,
            info: ServerInfo::default(),
        };
        let known = self.inner.era.lock().unwrap().clone();
        let era = match (known, self.inner.cfg.era) {
            (Some((true, v)), _) => Probe::Modern(v, Value::Null),
            (Some((false, _)), _) | (None, Era::Legacy) => Probe::Legacy,
            (None, era) => self.probe(&mut conn, era == Era::Modern).await?,
        };
        match era {
            Probe::Modern(v, discovered) => {
                let discovered = if discovered.is_null() {
                    self.set_modern(&conn, &v);
                    self.discover(&conn, &v, self.inner.cfg.timeouts.request)
                        .await?
                } else {
                    discovered
                };
                conn.info = modern_info(&v, &discovered);
            }
            Probe::Legacy => conn.info = self.initialize(&conn).await?,
        }
        *self.inner.era.lock().unwrap() = Some((conn.info.modern, conn.info.protocol.clone()));
        Ok(Arc::new(conn))
    }

    fn set_modern(&self, conn: &Conn, version: &str) {
        if let Some(h) = &conn.http {
            h.set_version(Some(version.into()), true);
        }
    }

    // Era detection per the 2026-07-28 backward compatibility rules.
    async fn probe(&self, conn: &mut Conn, modern_only: bool) -> Result<Probe, McpError> {
        let mut version = MODERN_VERSIONS[0].to_owned();
        let wait = if modern_only {
            self.inner.cfg.timeouts.request
        } else {
            self.inner.cfg.timeouts.probe
        };
        for _ in 0..2 {
            self.set_modern(conn, &version);
            match self.discover(conn, &version, wait).await {
                Ok(v) => return Ok(Probe::Modern(version, v)),
                Err(e) if e.rpc().is_some_and(|r| r.code == UNSUPPORTED_VERSION) => {
                    let supported = e.rpc().map(|r| r.supported_versions()).unwrap_or_default();
                    if let Some(v) = MODERN_VERSIONS
                        .iter()
                        .find(|v| supported.iter().any(|s| s == *v))
                    {
                        version = (*v).to_owned();
                        continue;
                    }
                    if LEGACY_VERSIONS
                        .iter()
                        .any(|v| supported.iter().any(|s| s == v))
                        && !modern_only
                    {
                        break;
                    }
                    return Err(McpError::Unsupported(format!(
                        "server supports protocol versions {supported:?}, client supports {MODERN_VERSIONS:?} and {LEGACY_VERSIONS:?}"
                    )));
                }
                Err(e) if e.rpc().is_some_and(|r| r.is_modern()) || modern_only => return Err(e),
                Err(e @ (McpError::Auth { .. } | McpError::Connect(_))) => return Err(e),
                Err(e)
                    if conn.http.is_none()
                        && (matches!(e, McpError::Closed { .. }) || !conn.wire.alive()) =>
                {
                    // Some legacy servers exit on an unknown first request; start a fresh one.
                    (self.inner.rt.events)(McpEvent::Warning {
                        server: self.inner.cfg.name.clone(),
                        message: format!(
                            "server died on the protocol probe ({e}); restarting it in legacy mode"
                        ),
                    });
                    let inbox = self.new_inbox();
                    let (wire, http) = self.start_wire(&inbox)?;
                    *conn = Conn {
                        wire,
                        http,
                        inbox,
                        info: ServerInfo::default(),
                    };
                    break;
                }
                Err(_) => break,
            }
        }
        if let Some(h) = &conn.http {
            h.set_version(None, false);
        }
        Ok(Probe::Legacy)
    }

    async fn discover(
        &self,
        conn: &Conn,
        version: &str,
        wait: Duration,
    ) -> Result<Value, McpError> {
        let params = json!({"_meta": self.meta(version)});
        self.raw(
            conn,
            "server/discover",
            params,
            None,
            Vec::new(),
            wait,
            None,
        )
        .await
    }

    fn meta(&self, version: &str) -> Value {
        let cfg = &self.inner.cfg;
        let mut m = Map::new();
        m.insert(format!("{META}protocolVersion"), json!(version));
        m.insert(
            format!("{META}clientInfo"),
            json!({"name": cfg.client_name, "version": cfg.client_version}),
        );
        m.insert(
            format!("{META}clientCapabilities"),
            self.inner.rt.handler.capabilities(),
        );
        Value::Object(m)
    }

    async fn initialize(&self, conn: &Conn) -> Result<ServerInfo, McpError> {
        let cfg = &self.inner.cfg;
        let params = json!({
            "protocolVersion": LEGACY_VERSIONS[0],
            "capabilities": self.inner.rt.handler.capabilities(),
            "clientInfo": {"name": cfg.client_name, "version": cfg.client_version},
        });
        let r = self
            .raw(
                conn,
                "initialize",
                params,
                None,
                Vec::new(),
                cfg.timeouts.connect,
                None,
            )
            .await?;
        let mut version = r["protocolVersion"].as_str().unwrap_or("").to_owned();
        if !LEGACY_VERSIONS.contains(&version.as_str()) {
            conn.inbox.warn(format!(
                "server answered with unknown protocol version {version:?}; continuing with {}",
                LEGACY_VERSIONS[0]
            ));
            version = LEGACY_VERSIONS[0].to_owned();
        }
        if let Some(h) = &conn.http {
            // The 2024-11-05 revision predates the version header.
            let header = (version != "2024-11-05").then(|| version.clone());
            h.set_version(header, false);
        }
        self.notify(conn, "notifications/initialized", Value::Null)
            .await;
        Ok(ServerInfo {
            protocol: version,
            modern: false,
            name: r["serverInfo"]["name"].as_str().map(str::to_owned),
            version: r["serverInfo"]["version"].as_str().map(str::to_owned),
            capabilities: r.get("capabilities").cloned().unwrap_or(json!({})),
            instructions: r["instructions"].as_str().map(str::to_owned),
        })
    }

    async fn notify(&self, conn: &Conn, method: &str, params: Value) {
        let out = Outgoing {
            body: jsonrpc::notification(method, params),
            method: method.into(),
            name: None,
            headers: Vec::new(),
            is_request: false,
        };
        if let Err(e) = conn.wire.send(&out).await {
            conn.inbox.warn(format!("{method} failed: {e}"));
        }
    }

    /// Sends one request on `conn` and waits for its result.
    #[allow(clippy::too_many_arguments)]
    async fn raw(
        &self,
        conn: &Conn,
        method: &str,
        mut params: Value,
        name: Option<String>,
        headers: Vec<(String, String)>,
        wait: Duration,
        call: Option<&CallOptions>,
    ) -> Result<Value, McpError> {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        if conn.info.modern && method != "server/discover" {
            let meta = params
                .as_object_mut()
                .expect("params object")
                .entry("_meta")
                .or_insert_with(|| json!({}));
            if let (Value::Object(m), Value::Object(extra)) = (meta, self.meta(&conn.info.protocol))
            {
                m.extend(extra);
            }
        }
        if call.is_some() {
            params["_meta"]["progressToken"] = json!(id.to_string());
        }
        let rx = conn.inbox.expect(id)?;
        // Dropping this future mid-wait still forgets the request and tells the server.
        let mut guard = Abandon {
            client: self,
            conn,
            id,
            method,
            armed: true,
        };
        let out = Outgoing {
            body: jsonrpc::request(id, method, params),
            method: method.into(),
            name,
            headers,
            is_request: true,
        };
        let exchange = async {
            match conn.wire.send(&out).await? {
                Sent::Done => {
                    let mut rx = rx;
                    match rx.try_recv() {
                        Ok(Some(r)) => r.map_err(|e| McpError::Rpc(Box::new(e))),
                        _ => Err(McpError::Protocol(format!(
                            "server ended the {method} response without a result"
                        ))),
                    }
                }
                Sent::Later => match rx.await {
                    Ok(r) => r.map_err(|e| McpError::Rpc(Box::new(e))),
                    Err(_) => Err(conn
                        .inbox
                        .closed()
                        .unwrap_or_else(|| McpError::closed("connection lost"))),
                },
            }
        };
        let mut exchange = std::pin::pin!(exchange);
        let clock = &self.inner.rt.clock;
        let max = call.map_or(wait, |_| self.inner.cfg.timeouts.call_max.max(wait));
        let mut waited = Duration::ZERO;
        let cancel = call.map(|c| c.cancel.clone()).unwrap_or_default();
        let outcome = loop {
            let seen = conn.inbox.progress.load(Ordering::Relaxed);
            let step = wait.min(max - waited);
            let timer = select(clock.sleep(step), cancel.cancelled());
            match select(exchange.as_mut(), timer).await {
                Either::Left((r, _)) => {
                    guard.armed = false;
                    if r.is_ok() {
                        self.inner.failures.store(0, Ordering::SeqCst);
                    }
                    return r;
                }
                Either::Right((Either::Right(_), _)) => break McpError::Cancelled,
                Either::Right((Either::Left(_), _)) => {
                    waited += step;
                    let progressed = conn.inbox.progress.load(Ordering::Relaxed) != seen;
                    if !(call.is_some() && progressed && waited < max) {
                        break McpError::Timeout {
                            method: method.into(),
                            after: waited,
                        };
                    }
                }
            }
        };
        let reason = if matches!(outcome, McpError::Cancelled) {
            "cancelled"
        } else {
            "timed out"
        };
        guard.release(reason);
        Err(outcome)
    }

    // Forgets an unanswered request and sends notifications/cancelled where the transport needs it.
    fn abandon(&self, conn: &Conn, id: u64, method: &str, reason: &str) {
        conn.inbox.forget(id);
        // Over modern HTTP dropping the stream is the cancellation.
        if (conn.http.is_none() || !conn.info.modern) && method != "server/discover" {
            let wire = conn.wire.clone();
            let body = jsonrpc::notification(
                "notifications/cancelled",
                json!({"requestId": id, "reason": reason}),
            );
            self.inner.rt.spawner.spawn(Box::pin(async move {
                let out = Outgoing {
                    body,
                    method: "notifications/cancelled".into(),
                    name: None,
                    headers: Vec::new(),
                    is_request: false,
                };
                let _ = wire.send(&out).await;
            }));
        }
    }

    /// Request with reconnects: restarts a dead server and renews an expired HTTP session.
    async fn request(
        &self,
        method: &str,
        params: Value,
        name: Option<String>,
        headers: Vec<(String, String)>,
        call: Option<&CallOptions>,
        idempotent: bool,
    ) -> Result<Value, McpError> {
        let t = self.inner.cfg.timeouts;
        let wait = match call {
            Some(c) => c.timeout.unwrap_or(t.call),
            None => t.request,
        };
        let mut retried = false;
        loop {
            let conn = self.conn().await?;
            let r = self
                .raw(
                    &conn,
                    method,
                    params.clone(),
                    name.clone(),
                    headers.clone(),
                    wait,
                    call,
                )
                .await;
            let expired = matches!(
                &r,
                Err(McpError::Http {
                    status: 404,
                    rpc: None,
                    ..
                })
            ) && conn.http.as_ref().is_some_and(|h| h.has_session());
            let died = matches!(&r, Err(McpError::Closed { .. })) && idempotent;
            if expired && !retried {
                // An expired session is routine: reconnect at once, not as a failure.
                retried = true;
                if let Some(h) = &conn.http {
                    h.reset_session();
                }
                conn.inbox.close(McpError::closed("session expired"));
                let mut slot = self.inner.conn.lock().await;
                if slot.as_ref().is_some_and(|c| Arc::ptr_eq(c, &conn)) {
                    *slot = None;
                }
                continue;
            }
            if died && !retried {
                retried = true;
                continue;
            }
            return r;
        }
    }

    /// Lists every tool, following pagination and repairing bad entries.
    pub async fn list_tools(&self) -> Result<Vec<McpTool>, McpError> {
        let http = matches!(self.inner.cfg.transport, ServerTransport::Http { .. });
        let mut tools: Vec<McpTool> = Vec::new();
        let mut cursor: Option<String> = None;
        let mut seen_cursors = Vec::new();
        let mut warnings = Vec::new();
        for _ in 0..MAX_PAGES {
            let params = match &cursor {
                Some(c) => json!({"cursor": c}),
                None => json!({}),
            };
            let r = self
                .request("tools/list", params, None, Vec::new(), None, true)
                .await?;
            let items = match r.get("tools") {
                Some(Value::Array(a)) => a.as_slice(),
                Some(Value::Null) | None => &[],
                Some(other) => {
                    return Err(McpError::Protocol(format!(
                        "tools/list returned tools = {}",
                        crate::content::clip(&other.to_string(), 80)
                    )));
                }
            };
            for item in items {
                match McpTool::from_value(item, &mut warnings) {
                    Ok(mut t) => {
                        if tools.iter().any(|x| x.name == t.name) {
                            warnings.push(format!("duplicate tool {} ignored", t.name));
                            continue;
                        }
                        if http && let Err(e) = t.read_headers() {
                            warnings.push(format!("tool {} rejected: {e}", t.name));
                            continue;
                        }
                        tools.push(t);
                    }
                    Err(e) => warnings.push(e),
                }
            }
            cursor = r
                .get("nextCursor")
                .and_then(Value::as_str)
                .filter(|c| !c.is_empty())
                .map(str::to_owned);
            match &cursor {
                None => break,
                Some(c) if seen_cursors.contains(c) => {
                    warnings.push("tools/list repeated a cursor; stopped paging".into());
                    break;
                }
                Some(c) => seen_cursors.push(c.clone()),
            }
        }
        for w in warnings {
            self.emit(McpEvent::Warning {
                server: self.inner.cfg.name.clone(),
                message: w,
            });
        }
        Ok(tools)
    }

    /// Calls a tool; tool-level failures come back as `is_error` results, not errors.
    pub async fn call_tool(
        &self,
        tool: &McpTool,
        args: Value,
        opts: &CallOptions,
    ) -> Result<CallResult, McpError> {
        let mut tool = tool.clone();
        let mut extra = Map::new();
        let mut header_retry = false;
        for _ in 0..MAX_INPUT_ROUNDS {
            let mut params = json!({"name": tool.name, "arguments": args});
            params.as_object_mut().unwrap().extend(extra.clone());
            let r = self
                .request(
                    "tools/call",
                    params,
                    Some(tool.name.clone()),
                    param_headers(&tool, &args),
                    Some(opts),
                    false,
                )
                .await;
            let v = match r {
                Err(e) if e.rpc().is_some_and(|r| r.code == HEADER_MISMATCH) && !header_retry => {
                    header_retry = true;
                    if let Some(t) = self
                        .list_tools()
                        .await?
                        .into_iter()
                        .find(|t| t.name == tool.name)
                    {
                        tool = t;
                    }
                    continue;
                }
                Err(McpError::Rpc(e)) => {
                    return Ok(CallResult::error(format!(
                        "{} (code {})",
                        e.message, e.code
                    )));
                }
                r => r?,
            };
            match v.get("resultType").and_then(Value::as_str) {
                None | Some("complete") => return Ok(CallResult::from_value(&v)),
                Some("input_required") => {}
                Some(other) => {
                    self.emit(McpEvent::Warning {
                        server: self.inner.cfg.name.clone(),
                        message: format!("unknown resultType {other:?} treated as complete"),
                    });
                    return Ok(CallResult::from_value(&v));
                }
            }
            let mut responses = Map::new();
            if let Some(Value::Object(reqs)) = v.get("inputRequests") {
                for (key, req) in reqs {
                    let method = req["method"].as_str().unwrap_or("");
                    let params = req.get("params").cloned().unwrap_or(json!({}));
                    match self
                        .inner
                        .rt
                        .handler
                        .handle(&self.inner.cfg.name, method, params)
                        .await
                    {
                        Ok(r) => {
                            responses.insert(key.clone(), r);
                        }
                        Err(e) => {
                            return Ok(CallResult::error(format!(
                                "the tool needs {method} input that this client cannot provide: {e}"
                            )));
                        }
                    }
                }
            }
            extra.clear();
            if !responses.is_empty() {
                extra.insert("inputResponses".into(), Value::Object(responses));
            }
            if let Some(s) = v.get("requestState").filter(|s| !s.is_null()) {
                extra.insert("requestState".into(), s.clone());
            }
        }
        Ok(CallResult::error(
            "the tool kept asking for more input; gave up",
        ))
    }
}

fn modern_info(version: &str, d: &Value) -> ServerInfo {
    let info = &d["_meta"][format!("{META}serverInfo")];
    ServerInfo {
        protocol: version.to_owned(),
        modern: true,
        name: info["name"].as_str().map(str::to_owned),
        version: info["version"].as_str().map(str::to_owned),
        capabilities: d.get("capabilities").cloned().unwrap_or(json!({})),
        instructions: d["instructions"].as_str().map(str::to_owned),
    }
}

/// Pending request that is abandoned unless it gets its answer.
struct Abandon<'a> {
    client: &'a McpClient,
    conn: &'a Conn,
    id: u64,
    method: &'a str,
    armed: bool,
}

impl Abandon<'_> {
    fn release(&mut self, reason: &str) {
        if std::mem::take(&mut self.armed) {
            self.client.abandon(self.conn, self.id, self.method, reason);
        }
    }
}

impl Drop for Abandon<'_> {
    fn drop(&mut self) {
        self.release("cancelled");
    }
}
