use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;
use ncore::{HttpRequest, Transport};
use ncore_tokio::{HyperTransport, Proxy, TransportTimeouts};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

fn limits() -> TransportTimeouts {
    TransportTimeouts {
        connect: Duration::from_millis(80),
        headers: Duration::from_millis(160),
        idle: Duration::from_millis(80),
    }
}

fn request(url: String) -> HttpRequest {
    HttpRequest {
        method: "GET".into(),
        url,
        headers: vec![],
        body: Bytes::new(),
    }
}

#[tokio::test]
async fn headers_and_proxy_handshakes_have_deadlines() {
    for proxy in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0; 4096];
            let _ = socket.read(&mut buffer).await;
            std::future::pending::<()>().await;
        });
        let transport = HyperTransport::with_timeouts(
            proxy.then(|| Proxy::parse(&format!("http://{address}")).unwrap()),
            limits(),
        );
        let url = if proxy {
            "https://example.test/".into()
        } else {
            format!("http://{address}/")
        };
        let error = tokio::time::timeout(Duration::from_secs(2), transport.send(request(url)))
            .await
            .unwrap()
            .unwrap_err();
        assert!(
            error.connect || error.message.contains("headers timed out"),
            "{error}"
        );
        server.abort();
    }
}

#[tokio::test]
async fn idle_body_timeout_is_reported_once() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buffer = [0; 4096];
        let _ = socket.read(&mut buffer).await;
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1\r\na\r\n")
            .await
            .unwrap();
        std::future::pending::<()>().await;
    });
    let transport = HyperTransport::with_timeouts(None, limits());
    let mut response = transport
        .send(request(format!("http://{address}/")))
        .await
        .unwrap();
    assert_eq!(
        response.body.next().await.unwrap().unwrap(),
        Bytes::from_static(b"a")
    );
    let error = response.body.next().await.unwrap().unwrap_err();
    assert!(error.message.contains("idle"));
    assert!(response.body.next().await.is_none());
    server.abort();
}

#[tokio::test]
async fn active_stream_can_outlive_the_idle_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buffer = [0; 4096];
        let _ = socket.read(&mut buffer).await;
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
            .await
            .unwrap();
        for _ in 0..5 {
            tokio::time::sleep(Duration::from_millis(30)).await;
            socket.write_all(b"1\r\na\r\n").await.unwrap();
        }
        socket.write_all(b"0\r\n\r\n").await.unwrap();
    });
    let transport = Arc::new(HyperTransport::with_timeouts(None, limits()));
    let mut response = transport
        .send(request(format!("http://{address}/")))
        .await
        .unwrap();
    let mut bytes = Vec::new();
    while let Some(chunk) = response.body.next().await {
        bytes.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(bytes, b"aaaaa");
    server.await.unwrap();
}
