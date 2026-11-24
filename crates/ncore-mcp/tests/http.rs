mod common;

use std::sync::Mutex;

use common::{MockHttp, Reply, header, runtime, warnings};
use ncore_mcp::{CallOptions, Content, McpClient, McpError, McpEvent, ServerConfig};
use serde_json::json;

fn result(id: &serde_json::Value, r: serde_json::Value) -> serde_json::Value {
    json!({"jsonrpc": "2.0", "id": id, "result": r})
}

#[tokio::test]
async fn modern_http_headers_and_endless_sse() {
    let http = MockHttp::new(|req, body| {
        let method = body["method"].as_str().unwrap_or("");
        assert_eq!(header(req, "MCP-Protocol-Version"), Some("2026-07-28"));
        assert_eq!(header(req, "Mcp-Method"), Some(method));
        assert_eq!(header(req, "authorization"), Some("Bearer t"));
        assert!(header(req, "accept").unwrap().contains("text/event-stream"));
        assert!(header(req, "Mcp-Session-Id").is_none());
        let id = &body["id"];
        match method {
            "server/discover" => Reply::json(result(id, json!({"resultType": "complete", "supportedVersions": ["2026-07-28"], "capabilities": {"tools": {}}})))
                .header("mcp-session-id", "must-be-ignored"),
            "tools/list" => Reply::json(result(id, json!({"resultType": "complete", "tools": [
                {"name": "sql", "inputSchema": {"type": "object", "properties": {
                    "region": {"type": "string", "x-mcp-header": "Region"}, "query": {"type": "string"}}}},
                {"name": "bad", "inputSchema": {"type": "object", "properties": {
                    "xs": {"type": "array", "items": {"type": "string", "x-mcp-header": "X"}}}}}
            ]}))),
            "tools/call" => {
                assert_eq!(header(req, "Mcp-Name"), Some("sql"));
                assert_eq!(header(req, "Mcp-Param-Region"), Some("eu-west"));
                let progress = json!({"jsonrpc": "2.0", "method": "notifications/progress",
                    "params": {"progressToken": body["params"]["_meta"]["progressToken"], "progress": 0.5}});
                let done = result(id, json!({"resultType": "complete", "content": [{"type": "text", "text": "3 rows"}]}));
                Reply::sse(&[
                    ": keep-alive\n\n",
                    &format!("event: message\ndata: {progress}\n\n"),
                    "data: not json\n\n",
                    &format!("data: {done}\n\n"),
                ], true)
            }
            other => panic!("unexpected {other}"),
        }
    });
    let (mut rt, events) = runtime();
    rt.transport = Some(http.clone());
    let mut cfg = ServerConfig::http("db", "https://mcp.example/mcp");
    if let ncore_mcp::ServerTransport::Http { headers, .. } = &mut cfg.transport {
        headers.push(("authorization".into(), "Bearer t".into()));
    }
    let client = McpClient::connect(cfg, rt).await.unwrap();
    assert!(client.info().await.unwrap().modern);
    let tools = client.list_tools().await.unwrap();
    assert_eq!(
        tools.len(),
        1,
        "tool with an invalid x-mcp-header is rejected"
    );
    let r = client
        .call_tool(
            &tools[0],
            json!({"region": "eu-west", "query": "select"}),
            &CallOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(r.content, [Content::Text("3 rows".into())]);
    assert!(
        warnings(&events)
            .iter()
            .any(|w| w.contains("tool bad rejected"))
    );
    assert!(
        warnings(&events)
            .iter()
            .any(|w| w.contains("non-JSON event"))
    );
    assert!(
        events
            .lock()
            .unwrap()
            .iter()
            .any(|e| matches!(e, McpEvent::Progress { .. }))
    );
}

#[tokio::test]
async fn legacy_http_with_sessions_and_server_requests() {
    let sessions = Mutex::new(0);
    let http = MockHttp::new(move |req, body| {
        let id = &body["id"];
        let session = header(req, "Mcp-Session-Id").map(str::to_owned);
        if req.method == "DELETE" {
            return Reply::status(200, "");
        }
        match body["method"].as_str() {
            Some("server/discover") => Reply::status(404, "<html>Not Found</html>"),
            Some("initialize") => {
                assert!(session.is_none());
                assert!(header(req, "MCP-Protocol-Version").is_none());
                let mut n = sessions.lock().unwrap();
                *n += 1;
                let mut r = Reply::json(result(
                    id,
                    json!({"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}}),
                ));
                r.headers.clear(); // no content type at all
                r.header("Mcp-Session-Id", &format!("s{n}"))
            }
            Some("notifications/initialized") => Reply::status(202, ""),
            Some("tools/list") => {
                assert_eq!(header(req, "MCP-Protocol-Version"), Some("2025-06-18"));
                assert!(header(req, "Mcp-Method").is_none());
                if session.as_deref() == Some("s1") {
                    return Reply::status(404, "session expired");
                }
                Reply::json(result(id, json!({"tools": [{"name": "files"}]})))
            }
            Some("tools/call") => {
                let ask = json!({"jsonrpc": "2.0", "id": "r1", "method": "roots/list"});
                let done = result(id, json!({"content": [{"type": "text", "text": "ok"}]}));
                Reply::sse(
                    &[&format!("data: {ask}\n\n"), &format!("data: {done}\n\n")],
                    false,
                )
            }
            None => {
                assert_eq!(body["id"], "r1");
                assert_eq!(body["error"]["code"], -32601);
                Reply::status(202, "")
            }
            Some(other) => panic!("unexpected {other}"),
        }
    });
    let (mut rt, _) = runtime();
    rt.transport = Some(http.clone());
    let client = McpClient::connect(ServerConfig::http("files", "http://localhost:1/mcp"), rt)
        .await
        .unwrap();
    assert_eq!(client.info().await.unwrap().protocol, "2025-06-18");
    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools[0].name, "files");
    let r = client
        .call_tool(&tools[0], json!({}), &CallOptions::default())
        .await
        .unwrap();
    assert_eq!(r.content, [Content::Text("ok".into())]);
    assert_eq!(
        http.methods(),
        [
            "server/discover",
            "initialize",
            "notifications/initialized",
            "tools/list",
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/call",
            "<reply>"
        ]
    );
    client.close().await;
    let log = http.log.lock().unwrap();
    let (last, _) = log.last().unwrap();
    assert_eq!(
        (last.method.as_str(), header(last, "Mcp-Session-Id")),
        ("DELETE", Some("s2"))
    );
}

#[tokio::test]
async fn unauthorized_server_is_reported_as_auth() {
    let http = MockHttp::new(|_, _| {
        Reply::status(401, "").header("WWW-Authenticate", "Bearer realm=\"mcp\"")
    });
    let (mut rt, _) = runtime();
    rt.transport = Some(http);
    let err = match McpClient::connect(ServerConfig::http("x", "https://x/mcp"), rt).await {
        Err(e) => e,
        Ok(_) => panic!("connected"),
    };
    match err {
        McpError::Auth { status, challenge } => {
            assert_eq!(status, 401);
            assert_eq!(challenge.as_deref(), Some("Bearer realm=\"mcp\""));
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[tokio::test]
async fn http_tool_errors_become_error_results() {
    let http = MockHttp::new(|_, body| {
        let id = &body["id"];
        match body["method"].as_str().unwrap() {
            "server/discover" => {
                Reply::json(result(id, json!({"supportedVersions": ["2026-07-28"]})))
            }
            "tools/list" => Reply::json(result(id, json!({"tools": [{"name": "t"}]}))),
            _ => Reply::json(
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32602, "message": "Unknown tool: t"}}),
            ),
        }
    });
    let (mut rt, _) = runtime();
    rt.transport = Some(http);
    let client = McpClient::connect(ServerConfig::http("x", "https://x/mcp"), rt)
        .await
        .unwrap();
    let tools = client.list_tools().await.unwrap();
    let r = client
        .call_tool(&tools[0], json!({}), &CallOptions::default())
        .await
        .unwrap();
    assert!(r.is_error);
    assert_eq!(
        r.content,
        [Content::Text("Unknown tool: t (code -32602)".into())]
    );
}
