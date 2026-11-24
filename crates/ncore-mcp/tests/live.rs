//! Real MCP servers; run with `MCP_BIN=<dir with mcp-server-* binaries> cargo test -p ncore-mcp --test live -- --ignored --nocapture`.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{runtime, warnings};
use ncore::CancelToken;
use ncore::core::ir::{Block, ToolCall};
use ncore_mcp::content::to_blocks;
use ncore_mcp::{CallOptions, McpClient, McpEvent, ResultLimits, ServerConfig, Toolbox};
use serde_json::json;

fn bin(name: &str) -> String {
    let dir = std::env::var("MCP_BIN").expect("set MCP_BIN to the node_modules/.bin directory");
    format!("{dir}/{name}")
}

fn kinds(blocks: &[Block]) -> Vec<String> {
    blocks
        .iter()
        .map(|b| match b {
            Block::Text { text, .. } => format!(
                "text({})",
                ncore_mcp::content::clip(text, 60).replace('\n', " ")
            ),
            Block::Media { media, .. } => {
                format!("media({})", media.mime.as_deref().unwrap_or("?"))
            }
            other => format!("{other:?}"),
        })
        .collect()
}

async fn everything(client: &McpClient, events: &common::Events) {
    let info = client.info().await.unwrap();
    println!(
        "server {:?} {:?} protocol {} modern {}",
        info.name, info.version, info.protocol, info.modern
    );
    let tools = client.list_tools().await.unwrap();
    println!("  {} tools", tools.len());
    let find = |n: &str| {
        tools
            .iter()
            .find(|t| t.name.to_lowercase().contains(n))
            .unwrap_or_else(|| panic!("no {n} tool"))
    };
    let opts = CallOptions::default();

    let r = client
        .call_tool(find("echo"), json!({"message": "привет"}), &opts)
        .await
        .unwrap();
    println!(
        "echo -> {:?}",
        kinds(&to_blocks(&r, &ResultLimits::default()))
    );
    assert!(!r.is_error);

    let img = tools
        .iter()
        .find(|t| t.name.to_lowercase().contains("image"))
        .expect("image tool");
    let r = client.call_tool(img, json!({}), &opts).await.unwrap();
    let blocks = to_blocks(&r, &ResultLimits::default());
    println!("{} -> {:?}", img.name, kinds(&blocks));
    assert!(
        blocks
            .iter()
            .any(|b| matches!(b, Block::Media { media, .. } if media.is_image()))
    );

    if let Some(long) = tools
        .iter()
        .find(|t| t.name.to_lowercase().contains("long"))
    {
        let before = events.lock().unwrap().len();
        let r = client
            .call_tool(long, json!({"duration": 2, "steps": 4}), &opts)
            .await
            .unwrap();
        let progress = events.lock().unwrap()[before..]
            .iter()
            .filter(|e| matches!(e, McpEvent::Progress { .. }))
            .count();
        println!(
            "{} -> {:?}, {progress} progress events",
            long.name,
            kinds(&to_blocks(&r, &ResultLimits::default()))
        );
        assert!(progress >= 1);
    }
    for t in tools.iter().filter(|t| {
        ["resource", "structured", "annotated", "audio", "sum", "add"]
            .iter()
            .any(|k| t.name.to_lowercase().contains(k))
    }) {
        let args = match t.name.to_lowercase() {
            n if n.contains("sum") || n.contains("add") => json!({"a": 2, "b": 3}),
            n if n.contains("structured") => json!({"location": "Paris"}),
            n if n.contains("annotated") => json!({"messageType": "success", "includeImage": true}),
            n if n.contains("resource") => {
                json!({"count": 2, "resourceId": 1, "resourceType": "Text"})
            }
            _ => json!({}),
        };
        match client.call_tool(t, args, &opts).await {
            Ok(r) => println!(
                "{} -> error={} {:?}",
                t.name,
                r.is_error,
                kinds(&to_blocks(&r, &ResultLimits::default()))
            ),
            Err(e) => println!("{} -> transport error {e}", t.name),
        }
    }
    let r = client
        .call_tool(
            find("header-params"),
            json!({"region": "eu-west", "repeat": 2}),
            &opts,
        )
        .await
        .unwrap();
    println!(
        "get-header-params -> error={} {:?}",
        r.is_error,
        kinds(&to_blocks(&r, &ResultLimits::default()))
    );
    assert!(!r.is_error);
    let r = client
        .call_tool(find("trigger-elicitation-request"), json!({}), &opts)
        .await
        .unwrap();
    println!(
        "trigger-elicitation-request -> error={} {:?}",
        r.is_error,
        kinds(&to_blocks(&r, &ResultLimits::default()))
    );
    let w = warnings(events);
    println!("warnings: {w:#?}");
}

struct Accept;

impl ncore_mcp::ClientHandler for Accept {
    fn capabilities(&self) -> serde_json::Value {
        json!({"elicitation": {"form": {}}})
    }

    fn handle<'a>(
        &'a self,
        _: &'a str,
        method: &'a str,
        params: serde_json::Value,
    ) -> ncore::rt::BoxFuture<'a, Result<serde_json::Value, ncore_mcp::RpcError>> {
        Box::pin(async move {
            println!(
                "  handler got {method}: {}",
                ncore_mcp::content::clip(&params.to_string(), 300)
            );
            match method {
                "elicitation/create" => {
                    // Fill every requested field with something plausible.
                    let mut content = serde_json::Map::new();
                    if let Some(props) = params["requestedSchema"]["properties"].as_object() {
                        for (k, v) in props {
                            let val = match v["type"].as_str() {
                                Some("boolean") => json!(true),
                                Some("number") | Some("integer") => json!(1),
                                _ => match v["enum"].as_array().and_then(|e| e.first()) {
                                    Some(e) => e.clone(),
                                    None => json!("test"),
                                },
                            };
                            content.insert(k.clone(), val);
                        }
                    }
                    Ok(json!({"action": "accept", "content": content}))
                }
                _ => Err(ncore_mcp::RpcError::method_not_found(method)),
            }
        })
    }
}

#[tokio::test]
#[ignore]
async fn everything_over_stdio() {
    for era in [ncore_mcp::Era::Auto, ncore_mcp::Era::Legacy] {
        println!("=== era {era:?}");
        let (mut rt, events) = runtime();
        rt.launcher = Some(Arc::new(ncore_tokio::TokioLauncher));
        rt.handler = Arc::new(Accept);
        let mut cfg = ServerConfig::stdio("everything", bin("mcp-server-everything"), &["stdio"]);
        cfg.era = era;
        let client = McpClient::connect(cfg, rt).await.unwrap();
        everything(&client, &events).await;
        client.close().await;
    }
}

#[tokio::test]
#[ignore]
async fn everything_over_http() {
    let mut server = tokio::process::Command::new(bin("mcp-server-everything"))
        .arg("streamableHttp")
        .env("PORT", "3917")
        .kill_on_drop(true)
        .stdout(std::process::Stdio::null())
        .spawn()
        .unwrap();
    tokio::time::sleep(Duration::from_secs(2)).await;
    for era in [ncore_mcp::Era::Auto, ncore_mcp::Era::Legacy] {
        println!("=== era {era:?}");
        let (mut rt, events) = runtime();
        rt.transport = Some(Arc::new(ncore_tokio::HyperTransport::new()));
        rt.handler = Arc::new(Accept);
        let mut cfg = ServerConfig::http("everything", "http://127.0.0.1:3917/mcp");
        cfg.era = era;
        let client = McpClient::connect(cfg, rt).await.unwrap();
        everything(&client, &events).await;
        client.close().await;
    }
    server.kill().await.unwrap();
}

#[tokio::test]
#[ignore]
async fn filesystem_through_toolbox() {
    let dir = std::env::temp_dir().join(format!("ncore-mcp-fs-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("note.txt"), "hello from a file").unwrap();
    let png = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
    )
    .unwrap();
    std::fs::write(dir.join("dot.png"), png).unwrap();
    let root = std::fs::canonicalize(&dir)
        .unwrap()
        .to_string_lossy()
        .into_owned();

    let (mut rt, events) = runtime();
    rt.launcher = Some(Arc::new(ncore_tokio::TokioLauncher));
    let client = McpClient::connect(
        ServerConfig::stdio("fs", bin("mcp-server-filesystem"), &[&root]),
        rt,
    )
    .await
    .unwrap();
    println!("info {:?}", client.info().await.unwrap());
    let mut tb = Toolbox::new(ResultLimits::default());
    tb.add("fs", Arc::new(client.clone())).unwrap();
    let r = tb.refresh().await;
    println!("tools {:?}", r.added);
    assert!(r.failed.is_empty());
    let cancel = CancelToken::new();
    let call = |id: &str, name: &str, args: serde_json::Value| ToolCall {
        id: id.into(),
        name: format!("mcp__fs__{name}"),
        args: args.to_string(),
        sig: None,
    };
    let results = tb
        .call_all(
            &[
                call("1", "list_directory", json!({"path": root})),
                call(
                    "2",
                    "read_text_file",
                    json!({"path": format!("{root}/note.txt")}),
                ),
                call(
                    "3",
                    "read_media_file",
                    json!({"path": format!("{root}/dot.png")}),
                ),
                call("4", "read_text_file", json!({"path": "/etc/passwd"})),
            ],
            &cancel,
        )
        .await;
    for r in &results {
        println!("{} error={} {:?}", r.call_id, r.is_error, kinds(&r.content));
    }
    assert!(ncore::core::ir::plain_text(&results[1].content).contains("hello from a file"));
    assert!(
        matches!(&results[2].content[0], Block::Media { media, .. } if media.mime.as_deref() == Some("image/png"))
    );
    assert!(results[3].is_error, "outside the allowed root");
    println!("warnings: {:#?}", warnings(&events));
    client.close().await;
    std::fs::remove_dir_all(&dir).unwrap();
}
