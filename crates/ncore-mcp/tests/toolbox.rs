use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use base64::Engine;
use ncore::CancelToken;
use ncore::core::ir::{Block, ToolCall, ToolDef};
use ncore::rt::BoxFuture;
use ncore_mcp::{CallResult, Content, McpError, McpTool, ResultLimits, ToolProvider, Toolbox};
use serde_json::{Value, json};

/// Host tools served in-process.
struct Local {
    more: AtomicBool,
}

impl ToolProvider for Local {
    fn list_tools(&self) -> BoxFuture<'_, Result<Vec<McpTool>, McpError>> {
        Box::pin(async move {
            let mut t = vec![
                McpTool::new(
                    "add",
                    "Adds numbers",
                    json!({"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}),
                ),
                McpTool::new("snap", "Takes a picture", json!({"type": "object"})),
            ];
            if self.more.load(Ordering::SeqCst) {
                t[1].description = Some("Takes a better picture".into());
                t.insert(
                    0,
                    McpTool::new("late", "Added later", json!({"type": "object"})),
                );
            }
            Ok(t)
        })
    }

    fn call_tool<'a>(
        &'a self,
        tool: &'a McpTool,
        args: Value,
        _cancel: &'a CancelToken,
    ) -> BoxFuture<'a, Result<CallResult, McpError>> {
        Box::pin(async move {
            match tool.name.as_str() {
                "add" => {
                    // Later calls finish first; results must still keep call order.
                    let a = args["a"].as_f64().unwrap_or(0.0);
                    tokio::time::sleep(Duration::from_millis(if a > 1.0 { 1 } else { 40 })).await;
                    Ok(CallResult::text(
                        (a + args["b"].as_f64().unwrap_or(0.0)).to_string(),
                    ))
                }
                "snap" => Ok(CallResult {
                    content: vec![Content::Image {
                        data: base64::engine::general_purpose::STANDARD
                            .encode([0xFF, 0xD8, 0xFF, 0xE0]),
                        mime: None,
                    }],
                    ..CallResult::default()
                }),
                _ => Err(McpError::Protocol("boom".into())),
            }
        })
    }
}

struct Broken;

impl ToolProvider for Broken {
    fn list_tools(&self) -> BoxFuture<'_, Result<Vec<McpTool>, McpError>> {
        Box::pin(async { Err(McpError::closed("server exited with code 1")) })
    }

    fn call_tool<'a>(
        &'a self,
        _: &'a McpTool,
        _: Value,
        _: &'a CancelToken,
    ) -> BoxFuture<'a, Result<CallResult, McpError>> {
        Box::pin(async { unreachable!() })
    }
}

fn names(tb: &Toolbox) -> Vec<String> {
    tb.tools()
        .into_iter()
        .map(|t| match t {
            ToolDef::Function(f) => f.name,
            ToolDef::Opaque(_) => unreachable!(),
        })
        .collect()
}

fn call(id: &str, name: &str, args: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        args: args.into(),
        sig: None,
    }
}

fn text(b: &[Block]) -> String {
    ncore::core::ir::plain_text(b)
}

#[tokio::test]
async fn toolbox_keeps_order_and_turns_failures_into_results() {
    let local = Arc::new(Local {
        more: AtomicBool::new(false),
    });
    let mut tb = Toolbox::new(ResultLimits::default());
    tb.add("host", local.clone()).unwrap();
    tb.add("dead", Arc::new(Broken)).unwrap();
    assert!(tb.add("host", Arc::new(Broken)).is_err());

    let r = tb.refresh().await;
    assert_eq!(r.added, ["mcp__host__add", "mcp__host__snap"]);
    assert_eq!(r.failed.len(), 1);
    assert_eq!(tb.errors()[0].0, "dead");

    local.more.store(true, Ordering::SeqCst);
    let r = tb.refresh().await;
    assert!(r.changed());
    // A new description changes the prompt as much as a new tool.
    assert_eq!(r.updated, ["mcp__host__snap"]);
    // The new tool goes last even though the server lists it first.
    assert_eq!(
        names(&tb),
        ["mcp__host__add", "mcp__host__snap", "mcp__host__late"]
    );
    assert!(!tb.refresh().await.changed());

    let cancel = CancelToken::new();
    let results = tb
        .call_all(
            &[
                call("1", "mcp__host__add", r#"{"a": 1, "b": 2}"#),
                call("2", "mcp__host__add", r#"{"a": 5, "b": 5}"#),
                call("3", "mcp__host__snap", ""),
                call("4", "mcp__nope__x", "{}"),
                call("5", "mcp__host__add", "{not json"),
                call("6", "mcp__host__late", "{}"),
            ],
            &cancel,
        )
        .await;
    let ids: Vec<_> = results.iter().map(|r| r.call_id.as_str()).collect();
    assert_eq!(ids, ["1", "2", "3", "4", "5", "6"]);
    assert_eq!(
        (text(&results[0].content), results[0].is_error),
        ("3".into(), false)
    );
    assert_eq!(text(&results[1].content), "10");
    assert!(
        matches!(&results[2].content[0], Block::Media { media, .. } if media.mime.as_deref() == Some("image/jpeg"))
    );
    assert!(results[3].is_error && text(&results[3].content).contains("unknown tool"));
    assert!(results[4].is_error && text(&results[4].content).contains("not valid JSON"));
    assert!(results[5].is_error && text(&results[5].content).contains("MCP server host failed"));
    assert_eq!(
        tb.resolve("mcp__host__late")
            .map(|(s, t)| (s, t.name.as_str())),
        Some(("host", "late"))
    );
}
