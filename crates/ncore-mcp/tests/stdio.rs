mod common;

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use common::{FakeLauncher, Proc, runtime, warnings};
use ncore::CancelToken;
use ncore::core::ir::{Block, Media};
use ncore::rt::BoxFuture;
use ncore_mcp::content::to_blocks;
use ncore_mcp::{
    CallOptions, ClientHandler, McpClient, McpError, McpEvent, ResultLimits, RpcError, ServerConfig,
};
use serde_json::{Value, json};

const PNG: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3];

fn cfg() -> ServerConfig {
    let mut c = ServerConfig::stdio("fake", "fake-server", &[]);
    c.timeouts.probe = Duration::from_millis(300);
    c
}

/// Answers the handshake of a legacy server: rejects the modern probe, then `initialize`.
async fn legacy_handshake(p: &mut Proc, version: &str) {
    loop {
        let m = p.next().await.expect("client message");
        match m["method"].as_str() {
            Some("server/discover") => p.fail(&m, -32601, "Method not found"),
            Some("initialize") => p.reply(
                &m,
                json!({
                    "protocolVersion": version,
                    "serverInfo": {"name": "legacy"},
                    "capabilities": {"tools": {}}
                }),
            ),
            Some("notifications/initialized") => return,
            other => panic!("unexpected {other:?}"),
        }
    }
}

fn script<F>(f: F) -> Arc<FakeLauncher>
where
    F: Fn(Proc) -> BoxFuture<'static, i32> + Send + Sync + 'static,
{
    FakeLauncher::new(Arc::new(f))
}

#[tokio::test]
async fn crooked_legacy_server() {
    let launcher = script(|mut p| {
        Box::pin(async move {
            let m = p.next().await.unwrap();
            assert_eq!(m["method"], "server/discover");
            // Ids echoed back as strings.
            p.send(json!({"jsonrpc": "2.0", "id": m["id"].to_string(), "error": {"code": -32601, "message": "nope"}}));
            let m = p.next().await.unwrap();
            assert_eq!(m["method"], "initialize");
            p.raw("Starting crooked server v0.1 ...");
            p.stderr("debug: init");
            p.send(json!({"jsonrpc": "2.0", "id": m["id"].to_string(), "result": {
                "protocolVersion": "2025-06-18", "serverInfo": {"name": "crooked"}, "capabilities": {"tools": {}}
            }}));
            assert_eq!(
                p.next().await.unwrap()["method"],
                "notifications/initialized"
            );
            loop {
                let Some(m) = p.next_request().await else {
                    return 0;
                };
                match (
                    m["method"].as_str().unwrap(),
                    m["params"]["cursor"].as_str(),
                ) {
                    ("tools/list", None) => {
                        p.raw(&"x".repeat(5000));
                        p.raw("");
                        p.reply(&m, json!({"tools": [
                            {"name": "echo", "inputSchema": {"properties": {"text": {"type": "string"}}}},
                            {"name": "echo", "inputSchema": {}},
                            {"name": ""},
                            {"name": "shout", "inputSchema": {"type": "string"}}
                        ], "nextCursor": "p2"}));
                    }
                    ("tools/list", Some("p2")) => {
                        p.reply(&m, json!({"tools": [{"name": "pic"}], "nextCursor": "p2"}))
                    }
                    ("tools/call", _) => {
                        p.send(json!({"jsonrpc": "2.0", "id": "s1", "method": "ping"}));
                        p.send(json!({"jsonrpc": "2.0", "id": "s2", "method": "roots/list"}));
                        p.send(json!({"jsonrpc": "2.0", "method": "notifications/progress",
                            "params": {"progressToken": m["params"]["_meta"]["progressToken"], "progress": 1, "total": 2}}));
                        let mut answers = Vec::new();
                        while answers.len() < 2 {
                            answers.push(p.next().await.unwrap());
                        }
                        answers.sort_by_key(|a| a["id"].as_str().unwrap().to_owned());
                        assert_eq!(answers[0]["result"], json!({}));
                        assert_eq!(answers[1]["error"]["code"], -32601);
                        p.reply(&m, json!({"content": [
                            {"type": "image", "data": STANDARD.encode(PNG), "mimeType": "image/jpeg"},
                            {"type": "text", "text": "a picture"}
                        ]}));
                    }
                    other => panic!("unexpected {other:?}"),
                }
            }
        })
    });
    let (mut rt, events) = runtime();
    rt.launcher = Some(launcher);
    let mut c = cfg();
    c.max_message_bytes = 4096;
    let client = McpClient::connect(c, rt).await.unwrap();
    let info = client.info().await.unwrap();
    assert_eq!((info.protocol.as_str(), info.modern), ("2025-06-18", false));
    assert_eq!(info.name.as_deref(), Some("crooked"));

    let tools = client.list_tools().await.unwrap();
    let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["echo", "shout", "pic"]);
    assert_eq!(tools[0].input_schema["type"], "object");
    assert!(tools[1].wrapped);

    let r = client
        .call_tool(&tools[2], json!({}), &CallOptions::default())
        .await
        .unwrap();
    let blocks = to_blocks(&r, &ResultLimits::default());
    assert!(
        matches!(&blocks[0], Block::Media { media: Media { mime: Some(m), .. }, .. } if m == "image/png")
    );
    assert!(matches!(&blocks[1], Block::Text { text, .. } if text == "a picture"));

    let w = warnings(&events).join("\n");
    for needle in [
        "non-JSON output: Starting crooked",
        "longer than 4096",
        "duplicate tool echo",
        "without a name",
        "repeated a cursor",
        "wrapped",
    ] {
        assert!(w.contains(needle), "missing warning {needle:?} in:\n{w}");
    }
    let ev = events.lock().unwrap();
    assert!(
        ev.iter()
            .any(|e| matches!(e, McpEvent::Stderr { line, .. } if line == "debug: init"))
    );
    assert!(
        ev.iter()
            .any(|e| matches!(e, McpEvent::Progress { total: Some(t), .. } if *t == 2.0))
    );
}

struct Elicit;

impl ClientHandler for Elicit {
    fn capabilities(&self) -> Value {
        json!({"elicitation": {}})
    }

    fn handle<'a>(
        &'a self,
        _server: &'a str,
        method: &'a str,
        params: Value,
    ) -> BoxFuture<'a, Result<Value, RpcError>> {
        Box::pin(async move {
            assert_eq!(method, "elicitation/create");
            assert_eq!(params["message"], "name?");
            Ok(json!({"action": "accept", "content": {"name": "octo"}}))
        })
    }
}

#[tokio::test]
async fn modern_server_with_multi_round_trip() {
    let launcher = script(|mut p| {
        Box::pin(async move {
            while let Some(m) = p.next_request().await {
                let meta = &m["params"]["_meta"];
                assert_eq!(
                    meta["io.modelcontextprotocol/protocolVersion"],
                    "2026-07-28"
                );
                assert_eq!(
                    meta["io.modelcontextprotocol/clientCapabilities"],
                    json!({"elicitation": {}})
                );
                assert_eq!(
                    meta["io.modelcontextprotocol/clientInfo"]["name"],
                    "nabucore"
                );
                let params = &m["params"];
                match m["method"].as_str().unwrap() {
                    "server/discover" => p.reply(&m, json!({
                        "resultType": "complete", "supportedVersions": ["2026-07-28"],
                        "capabilities": {"tools": {}},
                        "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "modern", "version": "2"}},
                        "instructions": "be nice"
                    })),
                    "tools/list" => p.reply(&m, json!({"resultType": "complete", "tools": [{"name": "login", "inputSchema": {"type": "object"}}], "ttlMs": 1000})),
                    "tools/call" if params.get("inputResponses").is_none() => p.reply(&m, json!({
                        "resultType": "input_required",
                        "inputRequests": {"who": {"method": "elicitation/create", "params": {"message": "name?"}}},
                        "requestState": "st1"
                    })),
                    "tools/call" => {
                        assert_eq!(params["requestState"], "st1");
                        let name = params["inputResponses"]["who"]["content"]["name"].as_str().unwrap();
                        p.reply(&m, json!({"resultType": "complete", "content": [{"type": "text", "text": format!("hi {name}")}]}));
                    }
                    other => panic!("unexpected {other}"),
                }
            }
            0
        })
    });
    let (mut rt, _) = runtime();
    rt.launcher = Some(launcher);
    rt.handler = Arc::new(Elicit);
    let client = McpClient::connect(cfg(), rt).await.unwrap();
    let info = client.info().await.unwrap();
    assert!(info.modern);
    assert_eq!(info.name.as_deref(), Some("modern"));
    assert_eq!(info.instructions.as_deref(), Some("be nice"));
    let tools = client.list_tools().await.unwrap();
    let r = client
        .call_tool(&tools[0], json!({}), &CallOptions::default())
        .await
        .unwrap();
    assert!(!r.is_error);
    assert_eq!(r.content, [ncore_mcp::Content::Text("hi octo".into())]);
}

#[tokio::test]
async fn version_mismatch_falls_back_to_a_common_legacy_version() {
    let launcher = script(|mut p| {
        Box::pin(async move {
            let m = p.next().await.unwrap();
            p.send(json!({"jsonrpc": "2.0", "id": m["id"], "error": {
                "code": -32022, "message": "Unsupported protocol version",
                "data": {"supported": ["2099-01-01", "2025-11-25"], "requested": "2026-07-28"}
            }}));
            let m = p.next().await.unwrap();
            assert_eq!(m["method"], "initialize");
            p.reply(
                &m,
                json!({"protocolVersion": "2025-11-25", "capabilities": {}}),
            );
            while p.next().await.is_some() {}
            0
        })
    });
    let (mut rt, _) = runtime();
    rt.launcher = Some(launcher);
    let client = McpClient::connect(cfg(), rt).await.unwrap();
    assert_eq!(client.info().await.unwrap().protocol, "2025-11-25");
}

#[tokio::test]
async fn server_that_dies_on_the_probe_is_restarted_in_legacy_mode() {
    let launcher = script(|mut p| {
        Box::pin(async move {
            if p.launch == 1 {
                let m = p.next().await.unwrap();
                assert_eq!(m["method"], "server/discover");
                p.stderr("fatal: unknown method server/discover");
                return 1;
            }
            legacy_handshake(&mut p, "2025-03-26").await;
            while p.next().await.is_some() {}
            0
        })
    });
    let (mut rt, events) = runtime();
    rt.launcher = Some(launcher.clone());
    let client = McpClient::connect(cfg(), rt).await.unwrap();
    assert_eq!(client.info().await.unwrap().protocol, "2025-03-26");
    assert_eq!(launcher.launches.load(Ordering::SeqCst), 2);
    assert!(
        warnings(&events)
            .iter()
            .any(|w| w.contains("died on the protocol probe"))
    );
}

#[tokio::test]
async fn silent_legacy_server_gets_initialize_after_the_probe_times_out() {
    let launcher = script(|mut p| {
        Box::pin(async move {
            assert_eq!(p.next().await.unwrap()["method"], "server/discover");
            // No answer, and nothing but `initialize` may follow.
            let m = p.next().await.unwrap();
            assert_eq!(m["method"], "initialize");
            p.reply(&m, json!({"protocolVersion": "2025-06-18"}));
            while p.next().await.is_some() {}
            0
        })
    });
    let (mut rt, _) = runtime();
    rt.launcher = Some(launcher);
    let client = McpClient::connect(cfg(), rt).await.unwrap();
    assert_eq!(client.info().await.unwrap().protocol, "2025-06-18");
}

#[tokio::test]
async fn crash_during_a_call_reports_stderr_and_restarts_later() {
    let launcher = script(|mut p| {
        Box::pin(async move {
            if p.launch == 1 {
                legacy_handshake(&mut p, "2025-06-18").await;
            } else {
                // Era is remembered: no second probe.
                let m = p.next().await.unwrap();
                assert_eq!(m["method"], "initialize");
                p.reply(&m, json!({"protocolVersion": "2025-06-18"}));
            }
            while let Some(m) = p.next_request().await {
                match m["method"].as_str().unwrap() {
                    "tools/list" => p.reply(&m, json!({"tools": [{"name": "boom"}]})),
                    "tools/call" => {
                        p.stderr("thread 'main' panicked at 'boom'");
                        return 3;
                    }
                    _ => {}
                }
            }
            0
        })
    });
    let (mut rt, events) = runtime();
    rt.launcher = Some(launcher.clone());
    let client = McpClient::connect(cfg(), rt).await.unwrap();
    let tools = client.list_tools().await.unwrap();
    let err = client
        .call_tool(&tools[0], json!({}), &CallOptions::default())
        .await
        .unwrap_err();
    match &err {
        McpError::Closed { reason, stderr } => {
            assert_eq!(reason, "server exited with code 3");
            assert!(stderr.iter().any(|l| l.contains("panicked")));
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(client.list_tools().await.unwrap().len(), 1);
    assert_eq!(launcher.launches.load(Ordering::SeqCst), 2);
    assert!(
        events
            .lock()
            .unwrap()
            .iter()
            .any(|e| matches!(e, McpEvent::Restarted { .. }))
    );
}

#[tokio::test]
async fn hung_calls_time_out_or_cancel_and_the_server_is_told() {
    let cancels = Arc::new(Mutex::new(Vec::new()));
    let seen = cancels.clone();
    let launcher = script(move |mut p| {
        let seen = seen.clone();
        Box::pin(async move {
            legacy_handshake(&mut p, "2025-06-18").await;
            while let Some(m) = p.next().await {
                match m["method"].as_str() {
                    Some("tools/list") => {
                        p.reply(&m, json!({"tools": [{"name": "hang"}, {"name": "slow"}]}))
                    }
                    Some("tools/call") if m["params"]["name"] == "slow" => {
                        let token = m["params"]["_meta"]["progressToken"].clone();
                        for i in 0..5 {
                            tokio::time::sleep(Duration::from_millis(60)).await;
                            p.send(json!({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progressToken": token, "progress": i}}));
                        }
                        p.reply(&m, json!({"content": [{"type": "text", "text": "done"}]}));
                    }
                    Some("notifications/cancelled") => {
                        seen.lock().unwrap().push(m["params"].clone())
                    }
                    _ => {}
                }
            }
            0
        })
    });
    let (mut rt, _) = runtime();
    rt.launcher = Some(launcher);
    let mut c = cfg();
    c.timeouts.call = Duration::from_millis(150);
    let client = McpClient::connect(c, rt).await.unwrap();
    let tools = client.list_tools().await.unwrap();

    let err = client
        .call_tool(&tools[0], json!({}), &CallOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(err, McpError::Timeout { .. }), "{err:?}");

    let cancel = CancelToken::new();
    let c2 = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        c2.cancel();
    });
    let opts = CallOptions {
        cancel,
        timeout: None,
    };
    let err = client
        .call_tool(&tools[0], json!({}), &opts)
        .await
        .unwrap_err();
    assert!(matches!(err, McpError::Cancelled), "{err:?}");

    // A call whose future is simply dropped is abandoned the same way.
    let dropped = tokio::time::timeout(
        Duration::from_millis(30),
        client.call_tool(&tools[0], json!({}), &CallOptions::default()),
    )
    .await;
    assert!(dropped.is_err());

    // Progress keeps a slow call alive past its timeout.
    let r = client
        .call_tool(&tools[1], json!({}), &CallOptions::default())
        .await
        .unwrap();
    assert_eq!(r.content, [ncore_mcp::Content::Text("done".into())]);

    let got = cancels.lock().unwrap().clone();
    let reasons: Vec<_> = got.iter().map(|c| c["reason"].as_str().unwrap()).collect();
    assert_eq!(reasons, ["timed out", "cancelled", "cancelled"]);
    assert!(got.iter().all(|c| c["requestId"].is_u64()));
}

#[tokio::test]
async fn server_that_never_starts_properly_fails_to_connect() {
    let launcher = script(|p| {
        Box::pin(async move {
            p.stderr("error: missing API key");
            2
        })
    });
    let (mut rt, _) = runtime();
    rt.launcher = Some(launcher);
    let err = match McpClient::connect(cfg(), rt).await {
        Err(e) => e,
        Ok(_) => panic!("connected"),
    };
    assert!(err.to_string().contains("missing API key"), "{err}");
}

#[tokio::test]
async fn real_process_through_tokio_launcher() {
    // A shell pretending to be a broken server: prints junk and exits.
    #[cfg(windows)]
    let mut c = ServerConfig::stdio(
        "cmd",
        "cmd.exe",
        &["/D", "/C", "echo not-json & echo oops 1>&2 & exit /b 7"],
    );
    #[cfg(not(windows))]
    let mut c = ServerConfig::stdio(
        "sh",
        "/bin/sh",
        &["-c", "echo not-json; echo oops >&2; exit 7"],
    );
    c.timeouts.probe = Duration::from_millis(500);
    let (mut rt, events) = runtime();
    rt.launcher = Some(Arc::new(ncore_tokio::TokioLauncher));
    let err = match McpClient::connect(c, rt).await {
        Err(e) => e,
        Ok(_) => panic!("connected"),
    };
    assert!(
        err.to_string().contains("code 7") || err.to_string().contains("oops"),
        "{err}"
    );
    assert!(warnings(&events).iter().any(|w| w.contains("not-json")));
}
