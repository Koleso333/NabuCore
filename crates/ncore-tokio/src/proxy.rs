use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use http::Uri;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioIo;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tower_service::Service;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Http,
    Socks5,
}

/// Outbound proxy: `http://[user:pass@]host:port` (CONNECT) or `socks5://[user:pass@]host:port`.
#[derive(Debug, Clone)]
pub struct Proxy {
    kind: Kind,
    host: String,
    port: u16,
    auth: Option<(String, String)>,
}

impl Proxy {
    pub fn parse(url: &str) -> Result<Self, String> {
        let (scheme, rest) = url
            .split_once("://")
            .ok_or_else(|| format!("proxy URL needs a scheme: {url}"))?;
        let kind = match scheme.to_ascii_lowercase().as_str() {
            "http" => Kind::Http,
            "socks5" | "socks5h" => Kind::Socks5,
            other => return Err(format!("unsupported proxy scheme {other}")),
        };
        let rest = rest.trim_end_matches('/');
        let (auth, addr) = match rest.rsplit_once('@') {
            Some((a, h)) => {
                let (u, p) = a.split_once(':').unwrap_or((a, ""));
                (Some((u.to_owned(), p.to_owned())), h)
            }
            None => (None, rest),
        };
        let (host, port) = addr
            .rsplit_once(':')
            .ok_or_else(|| format!("proxy URL needs a port: {url}"))?;
        let port = port
            .parse()
            .map_err(|_| format!("bad proxy port in {url}"))?;
        Ok(Self {
            kind,
            host: host.trim_matches(['[', ']']).to_owned(),
            port,
            auth,
        })
    }

    async fn tunnel(&self, host: &str, port: u16) -> Result<TcpStream, BoxError> {
        let mut s = TcpStream::connect((self.host.as_str(), self.port)).await?;
        s.set_nodelay(true)?;
        match self.kind {
            Kind::Http => self.connect_http(&mut s, host, port).await?,
            Kind::Socks5 => self.connect_socks(&mut s, host, port).await?,
        }
        Ok(s)
    }

    async fn connect_http(&self, s: &mut TcpStream, host: &str, port: u16) -> Result<(), BoxError> {
        let target = if host.contains(':') {
            format!("[{host}]:{port}")
        } else {
            format!("{host}:{port}")
        };
        let mut req = format!("CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n");
        if let Some((u, p)) = &self.auth {
            let cred = STANDARD.encode(format!("{u}:{p}"));
            req.push_str(&format!("Proxy-Authorization: Basic {cred}\r\n"));
        }
        req.push_str("\r\n");
        s.write_all(req.as_bytes()).await?;
        let mut buf = Vec::with_capacity(256);
        let mut byte = [0u8; 1];
        while !buf.ends_with(b"\r\n\r\n") {
            if s.read(&mut byte).await? == 0 || buf.len() > 8192 {
                return Err("proxy closed the CONNECT response".into());
            }
            buf.push(byte[0]);
        }
        let head = String::from_utf8_lossy(&buf);
        let status = head.split_whitespace().nth(1).unwrap_or("");
        if status != "200" {
            let line = head.lines().next().unwrap_or("");
            return Err(format!("proxy refused CONNECT {target}: {line}").into());
        }
        Ok(())
    }

    async fn connect_socks(
        &self,
        s: &mut TcpStream,
        host: &str,
        port: u16,
    ) -> Result<(), BoxError> {
        let method = if self.auth.is_some() { 2 } else { 0 };
        s.write_all(&[5, 1, method]).await?;
        let mut r = [0u8; 2];
        s.read_exact(&mut r).await?;
        if r != [5, method] {
            return Err("socks5 proxy rejected the auth method".into());
        }
        if let Some((u, p)) = &self.auth {
            let mut m = vec![1, u.len() as u8];
            m.extend_from_slice(u.as_bytes());
            m.push(p.len() as u8);
            m.extend_from_slice(p.as_bytes());
            s.write_all(&m).await?;
            s.read_exact(&mut r).await?;
            if r[1] != 0 {
                return Err("socks5 proxy rejected the credentials".into());
            }
        }
        let mut m = vec![5, 1, 0, 3, host.len() as u8];
        m.extend_from_slice(host.as_bytes());
        m.extend_from_slice(&port.to_be_bytes());
        s.write_all(&m).await?;
        let mut head = [0u8; 4];
        s.read_exact(&mut head).await?;
        if head[1] != 0 {
            return Err(format!(
                "socks5 connect to {host}:{port} failed with code {}",
                head[1]
            )
            .into());
        }
        let skip = match head[3] {
            1 => 4,
            4 => 16,
            3 => {
                let mut n = [0u8; 1];
                s.read_exact(&mut n).await?;
                n[0] as usize
            }
            _ => return Err("socks5 reply has an unknown address type".into()),
        };
        let mut rest = vec![0u8; skip + 2];
        s.read_exact(&mut rest).await?;
        Ok(())
    }
}

/// TCP connector that goes direct or through a proxy.
#[derive(Clone)]
pub struct Connector {
    direct: HttpConnector,
    proxy: Option<Arc<Proxy>>,
    timeout: std::time::Duration,
}

impl Connector {
    pub fn with_timeout(proxy: Option<Proxy>, timeout: std::time::Duration) -> Self {
        let mut direct = HttpConnector::new();
        direct.enforce_http(false);
        direct.set_nodelay(true);
        direct.set_connect_timeout(Some(timeout));
        Self {
            direct,
            proxy: proxy.map(Arc::new),
            timeout,
        }
    }
}

impl Service<Uri> for Connector {
    type Response = TokioIo<TcpStream>;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, BoxError>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), BoxError>> {
        self.direct.poll_ready(cx).map_err(Into::into)
    }

    fn call(&mut self, uri: Uri) -> Self::Future {
        let Some(proxy) = self.proxy.clone() else {
            let fut = self.direct.call(uri);
            return Box::pin(async move { fut.await.map_err(Into::into) });
        };
        let timeout = self.timeout;
        Box::pin(async move {
            let host = uri
                .host()
                .ok_or("URI without host")?
                .trim_matches(['[', ']']);
            let port = uri
                .port_u16()
                .unwrap_or(if uri.scheme_str() == Some("https") {
                    443
                } else {
                    80
                });
            let stream = tokio::time::timeout(timeout, proxy.tunnel(host, port))
                .await
                .map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::TimedOut, "proxy connection timed out")
                })??;
            Ok(TokioIo::new(stream))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_proxy_urls() {
        let p = Proxy::parse("http://127.0.0.1:10808").unwrap();
        assert_eq!(
            (p.kind, p.host.as_str(), p.port),
            (Kind::Http, "127.0.0.1", 10808)
        );
        let p = Proxy::parse("socks5://u:p@[::1]:1080/").unwrap();
        assert_eq!(
            (p.kind, p.host.as_str(), p.port),
            (Kind::Socks5, "::1", 1080)
        );
        assert_eq!(p.auth, Some(("u".into(), "p".into())));
        assert!(Proxy::parse("ftp://x:1").is_err());
        assert!(Proxy::parse("http://x").is_err());
    }
}
