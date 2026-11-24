#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use futures_channel::mpsc;
use futures_util::{StreamExt, stream};
use ncore::rt::BoxFuture;
use ncore::{
    ChildProcess, HttpRequest, HttpResponse, ProcessControl, ProcessLauncher, ProcessSpec,
    ProcessStdin, Transport, TransportError,
};
use ncore_mcp::{McpEvent, Runtime};
use ncore_tokio::{TokioClock, TokioSpawner};
use serde_json::{Value, json};
use tokio::sync::watch;

/// Fake child: what the client wrote, and pipes back to it.
pub struct Proc {
    pub launch: usize,
    lines: mpsc::UnboundedReceiver<Value>,
    out: mpsc::UnboundedSender<Bytes>,
    err: mpsc::UnboundedSender<Bytes>,
}

impl Proc {
    pub async fn next(&mut self) -> Option<Value> {
        self.lines.next().await
    }

    /// Next message that is not a notification the test does not care about.
    pub async fn next_request(&mut self) -> Option<Value> {
        loop {
            let m = self.next().await?;
            if m.get("id").is_some() {
                return Some(m);
            }
        }
    }

    pub fn raw(&self, s: &str) {
        let _ = self.out.unbounded_send(Bytes::from(format!("{s}\n")));
    }

    pub fn send(&self, v: Value) {
        self.raw(&v.to_string());
    }

    pub fn reply(&self, req: &Value, result: Value) {
        self.send(json!({"jsonrpc": "2.0", "id": req["id"], "result": result}));
    }

    pub fn fail(&self, req: &Value, code: i64, message: &str) {
        self.send(
            json!({"jsonrpc": "2.0", "id": req["id"], "error": {"code": code, "message": message}}),
        );
    }

    pub fn stderr(&self, s: &str) {
        let _ = self.err.unbounded_send(Bytes::from(format!("{s}\n")));
    }
}

pub type Script = Arc<dyn Fn(Proc) -> BoxFuture<'static, i32> + Send + Sync>;

/// Launcher that runs an async script instead of a process.
pub struct FakeLauncher {
    script: Script,
    pub launches: Arc<AtomicUsize>,
}

impl FakeLauncher {
    pub fn new(script: Script) -> Arc<Self> {
        Arc::new(Self {
            script,
            launches: Arc::default(),
        })
    }
}

struct Stdin {
    buf: Vec<u8>,
    tx: Option<mpsc::UnboundedSender<Value>>,
}

impl ProcessStdin for Stdin {
    fn write(&mut self, data: Bytes) -> BoxFuture<'_, Result<(), TransportError>> {
        Box::pin(async move {
            let tx = self
                .tx
                .as_ref()
                .ok_or_else(|| TransportError::new("broken pipe"))?;
            self.buf.extend_from_slice(&data);
            while let Some(p) = self.buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = self.buf.drain(..=p).collect();
                let v: Value = serde_json::from_slice(&line).expect("client wrote valid JSON");
                if tx.unbounded_send(v).is_err() {
                    return Err(TransportError::new("broken pipe"));
                }
            }
            Ok(())
        })
    }

    fn close(&mut self) -> BoxFuture<'_, ()> {
        self.tx = None;
        Box::pin(async {})
    }
}

struct Control {
    abort: tokio::task::AbortHandle,
    tx: watch::Sender<Option<Option<i32>>>,
}

impl ProcessControl for Control {
    fn kill(&self) {
        self.abort.abort();
        let _ = self.tx.send_if_modified(|s| {
            if s.is_none() {
                *s = Some(None);
                true
            } else {
                false
            }
        });
    }

    fn wait(&self) -> BoxFuture<'_, Option<i32>> {
        let mut rx = self.tx.subscribe();
        Box::pin(async move {
            rx.wait_for(Option::is_some)
                .await
                .map(|v| v.flatten())
                .unwrap_or(None)
        })
    }
}

impl ProcessLauncher for FakeLauncher {
    fn launch(&self, _spec: &ProcessSpec) -> Result<ChildProcess, TransportError> {
        let launch = self.launches.fetch_add(1, Ordering::SeqCst) + 1;
        let (in_tx, in_rx) = mpsc::unbounded();
        let (out_tx, out_rx) = mpsc::unbounded();
        let (err_tx, err_rx) = mpsc::unbounded();
        let (tx, _) = watch::channel(None);
        let done = tx.clone();
        let proc = Proc {
            launch,
            lines: in_rx,
            out: out_tx,
            err: err_tx,
        };
        let fut = (self.script)(proc);
        let handle = tokio::spawn(async move {
            let code = fut.await;
            done.send_replace(Some(Some(code)));
        });
        Ok(ChildProcess {
            stdin: Box::new(Stdin {
                buf: Vec::new(),
                tx: Some(in_tx),
            }),
            stdout: Box::pin(out_rx.map(Ok)),
            stderr: Box::pin(err_rx.map(Ok)),
            control: Arc::new(Control {
                abort: handle.abort_handle(),
                tx,
            }),
        })
    }
}

pub type Events = Arc<Mutex<Vec<McpEvent>>>;

pub fn runtime() -> (Runtime, Events) {
    let events: Events = Arc::default();
    let sink = events.clone();
    let mut rt = Runtime::new(Arc::new(TokioClock), Arc::new(TokioSpawner));
    rt.events = Arc::new(move |e| sink.lock().unwrap().push(e));
    (rt, events)
}

pub fn warnings(events: &Events) -> Vec<String> {
    events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|e| match e {
            McpEvent::Warning { message, .. } => Some(message.clone()),
            _ => None,
        })
        .collect()
}

/// Canned HTTP answer; `hang` keeps the body open after the chunks.
pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub chunks: Vec<String>,
    pub hang: bool,
}

impl Reply {
    pub fn json(v: Value) -> Self {
        Self {
            status: 200,
            headers: vec![("content-type".into(), "application/json".into())],
            chunks: vec![v.to_string()],
            hang: false,
        }
    }

    pub fn status(status: u16, body: &str) -> Self {
        Self {
            status,
            headers: Vec::new(),
            chunks: vec![body.into()],
            hang: false,
        }
    }

    pub fn sse(events: &[&str], hang: bool) -> Self {
        Self {
            status: 200,
            headers: vec![("content-type".into(), "text/event-stream".into())],
            chunks: events.iter().map(|e| e.to_string()).collect(),
            hang,
        }
    }

    pub fn header(mut self, k: &str, v: &str) -> Self {
        self.headers.push((k.into(), v.into()));
        self
    }
}

type Handler = Arc<dyn Fn(&HttpRequest, &Value) -> Reply + Send + Sync>;

/// Transport that answers from a closure and records requests.
pub struct MockHttp {
    handler: Handler,
    pub log: Mutex<Vec<(HttpRequest, Value)>>,
}

impl MockHttp {
    pub fn new(
        handler: impl Fn(&HttpRequest, &Value) -> Reply + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            handler: Arc::new(handler),
            log: Mutex::default(),
        })
    }

    pub fn methods(&self) -> Vec<String> {
        self.log
            .lock()
            .unwrap()
            .iter()
            .map(|(r, b)| match b["method"].as_str() {
                Some(m) => m.to_owned(),
                None if r.method == "POST" => "<reply>".to_owned(),
                None => r.method.clone(),
            })
            .collect()
    }
}

pub fn header<'a>(r: &'a HttpRequest, name: &str) -> Option<&'a str> {
    ncore::rt::header(&r.headers, name)
}

impl Transport for MockHttp {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        let body: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        let reply = (self.handler)(&req, &body);
        self.log.lock().unwrap().push((req, body));
        Box::pin(async move {
            let chunks = stream::iter(reply.chunks.into_iter().map(|c| Ok(Bytes::from(c))));
            let body = if reply.hang {
                chunks.chain(stream::pending()).boxed()
            } else {
                chunks.boxed()
            };
            Ok(HttpResponse {
                status: reply.status,
                headers: reply.headers,
                body,
            })
        })
    }
}
