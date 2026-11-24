mod process;
mod proxy;

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::{BodyDataStream, Full};
use hyper_rustls::{HttpsConnector, HttpsConnectorBuilder};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use ncore::rt::BoxFuture;
use ncore::{Clock, HttpRequest, HttpResponse, Spawner, Transport, TransportError};

pub use process::TokioLauncher;
use proxy::Connector;
pub use proxy::Proxy;

/// HTTP/1.1 and HTTP/2 transport on hyper with rustls and bundled webpki roots.
#[derive(Clone)]
pub struct HyperTransport {
    client: Client<HttpsConnector<Connector>, Full<Bytes>>,
    timeouts: TransportTimeouts,
}

#[derive(Debug, Clone, Copy)]
pub struct TransportTimeouts {
    pub connect: Duration,
    pub headers: Duration,
    pub idle: Duration,
}

impl Default for TransportTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(15),
            headers: Duration::from_secs(120),
            idle: Duration::from_secs(180),
        }
    }
}

impl Default for HyperTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperTransport {
    pub fn new() -> Self {
        Self::with_proxy(None)
    }

    /// Sends every request through `proxy` when given.
    pub fn with_proxy(proxy: Option<Proxy>) -> Self {
        Self::with_timeouts(proxy, TransportTimeouts::default())
    }

    pub fn with_timeouts(proxy: Option<Proxy>, timeouts: TransportTimeouts) -> Self {
        let connector = HttpsConnectorBuilder::new()
            .with_webpki_roots()
            .https_or_http()
            .enable_http1()
            .enable_http2()
            .wrap_connector(Connector::with_timeout(proxy, timeouts.connect));
        Self {
            client: Client::builder(TokioExecutor::new()).build(connector),
            timeouts,
        }
    }
}

impl Transport for HyperTransport {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        Box::pin(async move {
            let mut builder = http::Request::builder()
                .method(req.method.as_str())
                .uri(req.url.as_str());
            for (k, v) in &req.headers {
                builder = builder.header(k.as_str(), v.as_str());
            }
            let request = builder
                .body(Full::new(req.body))
                .map_err(|e| TransportError::new(e.to_string()))?;
            let resp = tokio::time::timeout(self.timeouts.headers, self.client.request(request))
                .await
                .map_err(|_| TransportError::new("upstream response headers timed out"))?
                .map_err(|e| {
                    if e.is_connect() {
                        TransportError::connect(e.to_string())
                    } else {
                        TransportError::new(e.to_string())
                    }
                })?;
            let status = resp.status().as_u16();
            let headers = resp
                .headers()
                .iter()
                .map(|(k, v)| {
                    (
                        k.as_str().to_owned(),
                        String::from_utf8_lossy(v.as_bytes()).into_owned(),
                    )
                })
                .collect();
            let body = BodyDataStream::new(resp.into_body())
                .map(|r| r.map_err(|e| TransportError::new(e.to_string())));
            let idle = self.timeouts.idle;
            let body =
                futures_util::stream::unfold((body, false), move |(mut body, done)| async move {
                    if done {
                        return None;
                    }
                    match tokio::time::timeout(idle, body.next()).await {
                        Ok(Some(chunk)) => {
                            let failed = chunk.is_err();
                            Some((chunk, (body, failed)))
                        }
                        Ok(None) => None,
                        Err(_) => Some((
                            Err(TransportError::new(
                                "upstream response body timed out while idle",
                            )),
                            (body, true),
                        )),
                    }
                });
            Ok(HttpResponse {
                status,
                headers,
                body: Box::pin(body),
            })
        })
    }
}

/// Clock backed by `tokio::time`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokioClock;

impl Clock for TokioClock {
    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> {
        Box::pin(tokio::time::sleep(d))
    }
}

/// Spawner backed by the current tokio runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokioSpawner;

impl Spawner for TokioSpawner {
    fn spawn(&self, fut: BoxFuture<'static, ()>) {
        tokio::spawn(fut);
    }
}

/// Core with the hyper transport, plus retries on the tokio clock when a policy is given.
pub fn core(retry: Option<ncore::RetryPolicy>) -> ncore::Core {
    let mut b = ncore::Core::builder(Arc::new(HyperTransport::new()));
    if let Some(p) = retry {
        b = b.retry(p, Arc::new(TokioClock));
    }
    b.build()
}
