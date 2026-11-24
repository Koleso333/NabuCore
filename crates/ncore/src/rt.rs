use std::fmt;
use std::future::Future;
use std::time::Duration;

use bytes::Bytes;

#[cfg(not(target_arch = "wasm32"))]
mod send {
    pub trait MaybeSend: Send {}
    impl<T: Send + ?Sized> MaybeSend for T {}
    pub trait MaybeSync: Sync {}
    impl<T: Sync + ?Sized> MaybeSync for T {}
    pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;
    pub type BoxStream<'a, T> = std::pin::Pin<Box<dyn futures_core::Stream<Item = T> + Send + 'a>>;
}

#[cfg(target_arch = "wasm32")]
mod send {
    pub trait MaybeSend {}
    impl<T: ?Sized> MaybeSend for T {}
    pub trait MaybeSync {}
    impl<T: ?Sized> MaybeSync for T {}
    pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + 'a>>;
    pub type BoxStream<'a, T> = std::pin::Pin<Box<dyn futures_core::Stream<Item = T> + 'a>>;
}

pub use send::{BoxFuture, BoxStream, MaybeSend, MaybeSync};

/// Boxes a future into the platform's `BoxFuture`.
pub fn boxed<'a, T>(f: impl Future<Output = T> + MaybeSend + 'a) -> BoxFuture<'a, T> {
    Box::pin(f)
}

/// Outgoing HTTP request with a fully buffered body.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Bytes,
}

/// HTTP response whose body arrives as a stream of chunks.
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: BoxStream<'static, Result<Bytes, TransportError>>,
}

impl HttpResponse {
    /// First header value with the given case-insensitive name.
    pub fn header(&self, name: &str) -> Option<&str> {
        header(&self.headers, name)
    }
}

impl fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .finish()
    }
}

/// Case-insensitive header lookup.
pub fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

/// Network-level failure; `connect` marks errors that happened before anything was sent.
#[derive(Debug, Clone)]
pub struct TransportError {
    pub message: String,
    pub connect: bool,
}

impl TransportError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            connect: false,
        }
    }

    pub fn connect(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            connect: true,
        }
    }
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TransportError {}

/// Sends HTTP requests; implemented by the host or a runtime crate.
pub trait Transport: MaybeSend + MaybeSync {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>>;
}

/// Timers for retries and timeouts.
pub trait Clock: MaybeSend + MaybeSync {
    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()>;
}

/// Runs background tasks.
pub trait Spawner: MaybeSend + MaybeSync {
    fn spawn(&self, fut: BoxFuture<'static, ()>);
}

/// Command line of a child process.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProcessSpec {
    pub program: String,
    pub args: Vec<String>,
    /// Added to the inherited environment, or the whole environment when `clear_env` is set.
    pub env: Vec<(String, String)>,
    pub clear_env: bool,
    pub cwd: Option<String>,
}

/// Writable end of a child's stdin.
pub trait ProcessStdin: MaybeSend {
    fn write(&mut self, data: Bytes) -> BoxFuture<'_, Result<(), TransportError>>;
    /// Closes stdin, which asks a well-behaved child to exit.
    fn close(&mut self) -> BoxFuture<'_, ()>;
}

/// Lifecycle control of a running child.
pub trait ProcessControl: MaybeSend + MaybeSync {
    fn kill(&self);
    /// Resolves with the exit code once the child has exited (`None` when killed by a signal).
    fn wait(&self) -> BoxFuture<'_, Option<i32>>;
}

/// Running child process with piped standard streams.
pub struct ChildProcess {
    pub stdin: Box<dyn ProcessStdin>,
    pub stdout: BoxStream<'static, Result<Bytes, TransportError>>,
    pub stderr: BoxStream<'static, Result<Bytes, TransportError>>,
    pub control: std::sync::Arc<dyn ProcessControl>,
}

/// Starts child processes; absent on platforms that forbid them.
pub trait ProcessLauncher: MaybeSend + MaybeSync {
    fn launch(&self, spec: &ProcessSpec) -> Result<ChildProcess, TransportError>;
}
