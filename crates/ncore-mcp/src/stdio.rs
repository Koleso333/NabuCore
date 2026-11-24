use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;
use futures_util::future::{Either, select};
use futures_util::lock::Mutex as AsyncMutex;
use ncore::rt::{BoxFuture, BoxStream};
use ncore::{
    Clock, ProcessControl, ProcessLauncher, ProcessSpec, ProcessStdin, Spawner, TransportError,
};
use serde_json::Value;

use crate::content::clip;
use crate::error::McpError;
use crate::wire::{Inbox, McpEvent, Outgoing, Sent, Wire};

const MAX_GARBAGE_WARNINGS: usize = 20;
const MAX_STDERR_LINE: usize = 2000;

/// Newline-delimited JSON-RPC over a child's stdin and stdout.
pub(crate) struct StdioWire {
    stdin: AsyncMutex<Box<dyn ProcessStdin>>,
    control: Arc<dyn ProcessControl>,
    alive: Arc<AtomicBool>,
    stderr: Arc<Mutex<VecDeque<String>>>,
    clock: Arc<dyn Clock>,
    shutdown: Duration,
}

pub(crate) struct StdioParams {
    pub max_line: usize,
    pub stderr_lines: usize,
    pub shutdown: Duration,
}

impl StdioWire {
    pub fn start(
        spec: &ProcessSpec,
        launcher: &dyn ProcessLauncher,
        spawner: &dyn Spawner,
        clock: Arc<dyn Clock>,
        inbox: Arc<Inbox>,
        p: StdioParams,
    ) -> Result<Arc<Self>, McpError> {
        let child = launcher
            .launch(spec)
            .map_err(|e| McpError::Connect(e.message))?;
        let wire = Arc::new(Self {
            stdin: AsyncMutex::new(child.stdin),
            control: child.control,
            alive: Arc::new(AtomicBool::new(true)),
            stderr: Arc::default(),
            clock,
            shutdown: p.shutdown,
        });
        let (done_tx, done_rx) = futures_channel::oneshot::channel::<()>();
        spawner.spawn(Box::pin(read_stderr(
            child.stderr,
            wire.stderr.clone(),
            p.stderr_lines,
            inbox.clone(),
            done_tx,
        )));
        spawner.spawn(Box::pin(read_stdout(
            child.stdout,
            Arc::downgrade(&wire),
            wire.alive.clone(),
            wire.control.clone(),
            wire.stderr.clone(),
            wire.clock.clone(),
            inbox,
            spawner_ref(spawner),
            p.max_line,
            done_rx,
        )));
        Ok(wire)
    }

    async fn write(&self, v: &Value) -> Result<(), McpError> {
        let mut line = serde_json::to_vec(v).map_err(|e| McpError::Protocol(e.to_string()))?;
        line.push(b'\n');
        let r = self.stdin.lock().await.write(Bytes::from(line)).await;
        r.map_err(|e: TransportError| {
            self.alive.store(false, Ordering::SeqCst);
            McpError::Closed {
                reason: format!("cannot write to the server: {e}"),
                stderr: self.stderr(),
            }
        })
    }
}

// Spawning replies needs an owned spawner; replies are rare, so a tiny shim is enough.
fn spawner_ref(s: &dyn Spawner) -> Replies {
    let (tx, rx) = futures_channel::mpsc::unbounded::<BoxFuture<'static, ()>>();
    s.spawn(Box::pin(rx.for_each_concurrent(None, |f| f)));
    Replies(tx)
}

struct Replies(futures_channel::mpsc::UnboundedSender<BoxFuture<'static, ()>>);

impl Wire for StdioWire {
    fn send<'a>(&'a self, msg: &'a Outgoing) -> BoxFuture<'a, Result<Sent, McpError>> {
        Box::pin(async move {
            if !self.alive() {
                return Err(McpError::Closed {
                    reason: "server is not running".into(),
                    stderr: self.stderr(),
                });
            }
            self.write(&msg.body).await.map(|_| Sent::Later)
        })
    }

    fn close(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            self.stdin.lock().await.close().await;
            let exited = select(self.control.wait(), self.clock.sleep(self.shutdown)).await;
            if matches!(exited, Either::Right(_)) {
                self.control.kill();
            }
            self.alive.store(false, Ordering::SeqCst);
        })
    }

    fn alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    fn stderr(&self) -> Vec<String> {
        self.stderr.lock().unwrap().iter().cloned().collect()
    }
}

impl Drop for StdioWire {
    fn drop(&mut self) {
        self.control.kill();
    }
}

async fn read_stderr(
    mut s: BoxStream<'static, Result<Bytes, TransportError>>,
    ring: Arc<Mutex<VecDeque<String>>>,
    cap: usize,
    inbox: Arc<Inbox>,
    done: futures_channel::oneshot::Sender<()>,
) {
    let mut buf = Vec::new();
    let emit = |raw: &[u8]| {
        let text = String::from_utf8_lossy(raw);
        let line = clip(text.trim_end_matches('\r'), MAX_STDERR_LINE).to_owned();
        if line.trim().is_empty() {
            return;
        }
        {
            let mut r = ring.lock().unwrap();
            if r.len() == cap.max(1) {
                r.pop_front();
            }
            r.push_back(line.clone());
        }
        (inbox.events)(McpEvent::Stderr {
            server: inbox.server.clone(),
            line,
        });
    };
    while let Some(Ok(chunk)) = s.next().await {
        for &b in chunk.iter() {
            if b == b'\n' {
                emit(&buf);
                buf.clear();
            } else if buf.len() < MAX_STDERR_LINE * 4 {
                buf.push(b);
            }
        }
    }
    if !buf.is_empty() {
        emit(&buf);
    }
    let _ = done.send(());
}

#[allow(clippy::too_many_arguments)]
async fn read_stdout(
    mut s: BoxStream<'static, Result<Bytes, TransportError>>,
    wire: Weak<StdioWire>,
    alive: Arc<AtomicBool>,
    control: Arc<dyn ProcessControl>,
    ring: Arc<Mutex<VecDeque<String>>>,
    clock: Arc<dyn Clock>,
    inbox: Arc<Inbox>,
    replies: Replies,
    max_line: usize,
    stderr_done: futures_channel::oneshot::Receiver<()>,
) {
    let mut lines = Lines {
        buf: Vec::new(),
        skipping: false,
        garbage: 0,
        max_line,
    };
    let mut reason = "server closed its output".to_owned();
    while let Some(chunk) = s.next().await {
        match chunk {
            Ok(chunk) => {
                for line in lines.feed(&chunk, &inbox) {
                    handle(line, &inbox, &wire, &replies);
                }
            }
            Err(e) => {
                reason = format!("cannot read server output: {e}");
                break;
            }
        }
    }
    if let Some(line) = lines.finish() {
        handle(line, &inbox, &wire, &replies);
    }
    alive.store(false, Ordering::SeqCst);
    let code = match select(control.wait(), clock.sleep(Duration::from_secs(2))).await {
        Either::Left((code, _)) => Some(code),
        Either::Right(_) => None,
    };
    let _ = select(stderr_done, clock.sleep(Duration::from_millis(500))).await;
    if let Some(code) = code {
        reason = match code {
            Some(c) => format!("server exited with code {c}"),
            None => "server was killed".into(),
        };
    }
    inbox.close(McpError::Closed {
        reason,
        stderr: ring.lock().unwrap().iter().cloned().collect(),
    });
}

fn handle(line: Value, inbox: &Arc<Inbox>, wire: &Weak<StdioWire>, replies: &Replies) {
    for reply in inbox.dispatch(line) {
        let w = wire.clone();
        let _ = replies.0.unbounded_send(Box::pin(async move {
            let v = reply.await;
            if let Some(w) = w.upgrade() {
                let _ = w.write(&v).await;
            }
        }));
    }
}

/// Splits stdout into JSON lines, skipping garbage and oversized lines.
struct Lines {
    buf: Vec<u8>,
    skipping: bool,
    garbage: usize,
    max_line: usize,
}

impl Lines {
    fn feed(&mut self, chunk: &[u8], inbox: &Inbox) -> Vec<Value> {
        let mut out = Vec::new();
        let mut rest = chunk;
        while let Some(pos) = rest.iter().position(|&b| b == b'\n') {
            if !self.skipping {
                self.buf.extend_from_slice(&rest[..pos]);
                let line = std::mem::take(&mut self.buf);
                if line.len() > self.max_line {
                    inbox.warn(format!(
                        "dropped an output line longer than {} bytes",
                        self.max_line
                    ));
                } else if let Some(v) = self.parse(&line, inbox) {
                    out.push(v);
                }
            }
            self.skipping = false;
            self.buf.clear();
            rest = &rest[pos + 1..];
        }
        if !self.skipping {
            self.buf.extend_from_slice(rest);
            if self.buf.len() > self.max_line {
                inbox.warn(format!(
                    "dropped an output line longer than {} bytes",
                    self.max_line
                ));
                self.buf.clear();
                self.skipping = true;
            }
        }
        out
    }

    fn finish(&mut self) -> Option<Value> {
        let line = std::mem::take(&mut self.buf);
        if self.skipping || line.is_empty() {
            return None;
        }
        serde_json::from_slice(trim(&line)).ok()
    }

    fn parse(&mut self, line: &[u8], inbox: &Inbox) -> Option<Value> {
        let line = trim(line);
        if line.is_empty() {
            return None;
        }
        match serde_json::from_slice(line) {
            Ok(v) => Some(v),
            Err(_) => {
                self.garbage += 1;
                if self.garbage <= MAX_GARBAGE_WARNINGS {
                    let text = String::from_utf8_lossy(line);
                    inbox.warn(format!("ignored non-JSON output: {}", clip(&text, 200)));
                } else if self.garbage == MAX_GARBAGE_WARNINGS + 1 {
                    inbox.warn("further non-JSON output is ignored silently");
                }
                None
            }
        }
    }
}

fn trim(b: &[u8]) -> &[u8] {
    let b = b.strip_prefix("\u{feff}".as_bytes()).unwrap_or(b);
    let start = b
        .iter()
        .position(|c| !c.is_ascii_whitespace())
        .unwrap_or(b.len());
    let end = b
        .iter()
        .rposition(|c| !c.is_ascii_whitespace())
        .map_or(start, |e| e + 1);
    &b[start..end]
}
