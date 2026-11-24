use std::sync::Arc;

use futures_util::future::join_all;
use ncore::CancelToken;
use ncore::core::ir::{Block, Ext, Tool, ToolCall, ToolDef, ToolResult};
use ncore::rt::{BoxFuture, MaybeSend, MaybeSync};
use serde_json::Value;

use crate::client::{CallOptions, McpClient};
use crate::content::{CallResult, ResultLimits, clip, to_blocks};
use crate::error::McpError;
use crate::tool::{McpTool, exposed_name, prepare_args};

/// Source of tools: an MCP server or the host's own in-process tools.
pub trait ToolProvider: MaybeSend + MaybeSync {
    fn list_tools(&self) -> BoxFuture<'_, Result<Vec<McpTool>, McpError>>;
    fn call_tool<'a>(
        &'a self,
        tool: &'a McpTool,
        args: Value,
        cancel: &'a CancelToken,
    ) -> BoxFuture<'a, Result<CallResult, McpError>>;
}

impl ToolProvider for McpClient {
    fn list_tools(&self) -> BoxFuture<'_, Result<Vec<McpTool>, McpError>> {
        Box::pin(McpClient::list_tools(self))
    }

    fn call_tool<'a>(
        &'a self,
        tool: &'a McpTool,
        args: Value,
        cancel: &'a CancelToken,
    ) -> BoxFuture<'a, Result<CallResult, McpError>> {
        Box::pin(async move {
            let opts = CallOptions {
                cancel: cancel.clone(),
                timeout: None,
            };
            McpClient::call_tool(self, tool, args, &opts).await
        })
    }
}

struct Source {
    name: String,
    provider: Arc<dyn ToolProvider>,
    error: Option<McpError>,
}

struct Entry {
    exposed: String,
    source: usize,
    tool: McpTool,
}

/// Outcome of `Toolbox::refresh`.
#[derive(Debug, Default)]
pub struct Refresh {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    /// Tools whose description or schema changed.
    pub updated: Vec<String>,
    /// Servers whose listing failed; their previous tools are kept.
    pub failed: Vec<(String, McpError)>,
}

impl Refresh {
    /// True when the tool list sent to the model changed, which breaks the prompt cache.
    pub fn changed(&self) -> bool {
        !self.added.is_empty() || !self.removed.is_empty() || !self.updated.is_empty()
    }
}

/// Tools of several providers under stable, provider-safe names in a frozen order.
#[derive(Default)]
pub struct Toolbox {
    sources: Vec<Source>,
    entries: Vec<Entry>,
    limits: ResultLimits,
}

impl Toolbox {
    pub fn new(limits: ResultLimits) -> Self {
        Self {
            limits,
            ..Self::default()
        }
    }

    /// Registers a provider under a unique server name.
    pub fn add(
        &mut self,
        server: impl Into<String>,
        provider: Arc<dyn ToolProvider>,
    ) -> Result<(), String> {
        let name = server.into();
        if self.sources.iter().any(|s| s.name == name) {
            return Err(format!("server {name} is already registered"));
        }
        self.sources.push(Source {
            name,
            provider,
            error: None,
        });
        Ok(())
    }

    /// Last listing error of each server that failed.
    pub fn errors(&self) -> Vec<(&str, &McpError)> {
        self.sources
            .iter()
            .filter_map(|s| s.error.as_ref().map(|e| (s.name.as_str(), e)))
            .collect()
    }

    /// Re-lists all servers; known tools keep their place, new ones go to the end.
    pub async fn refresh(&mut self) -> Refresh {
        let lists = join_all(self.sources.iter().map(|s| s.provider.list_tools())).await;
        let mut out = Refresh::default();
        for (i, list) in lists.into_iter().enumerate() {
            let tools = match list {
                Ok(t) => {
                    self.sources[i].error = None;
                    t
                }
                Err(e) => {
                    out.failed.push((self.sources[i].name.clone(), e.clone()));
                    self.sources[i].error = Some(e);
                    continue;
                }
            };
            self.entries.retain(|e| {
                let keep = e.source != i || tools.iter().any(|t| t.name == e.tool.name);
                if !keep {
                    out.removed.push(e.exposed.clone());
                }
                keep
            });
            for t in tools {
                if let Some(e) = self
                    .entries
                    .iter_mut()
                    .find(|e| e.source == i && e.tool.name == t.name)
                {
                    if e.tool != t {
                        out.updated.push(e.exposed.clone());
                        e.tool = t;
                    }
                    continue;
                }
                let exposed = self.unique(exposed_name(&self.sources[i].name, &t.name));
                out.added.push(exposed.clone());
                self.entries.push(Entry {
                    exposed,
                    source: i,
                    tool: t,
                });
            }
        }
        out
    }

    fn unique(&self, base: String) -> String {
        if !self.entries.iter().any(|e| e.exposed == base) {
            return base;
        }
        (2..)
            .map(|n| {
                let suffix = format!("_{n}");
                format!("{}{suffix}", clip(&base, 64 - suffix.len()))
            })
            .find(|c| !self.entries.iter().any(|e| e.exposed == *c))
            .expect("free name")
    }

    /// Tool definitions for the request, in the frozen order.
    pub fn tools(&self) -> Vec<ToolDef> {
        self.entries
            .iter()
            .map(|e| {
                let t = &e.tool;
                ToolDef::Function(Tool {
                    name: e.exposed.clone(),
                    description: t.description.clone().or_else(|| t.title.clone()),
                    schema: t.input_schema.clone(),
                    strict: None,
                    cache: None,
                    ext: Ext::default(),
                })
            })
            .collect()
    }

    /// Server and original tool behind an exposed name.
    pub fn resolve(&self, exposed: &str) -> Option<(&str, &McpTool)> {
        self.entries
            .iter()
            .find(|e| e.exposed == exposed)
            .map(|e| (self.sources[e.source].name.as_str(), &e.tool))
    }

    /// Runs one call; every failure becomes an error result the model can read.
    pub async fn call(&self, call: &ToolCall, cancel: &CancelToken) -> ToolResult {
        let (blocks, is_error) = self.run(call, cancel).await;
        ToolResult {
            call_id: call.id.clone(),
            name: Some(call.name.clone()),
            content: blocks,
            is_error,
            cache: None,
        }
    }

    async fn run(&self, call: &ToolCall, cancel: &CancelToken) -> (Vec<Block>, bool) {
        let fail = |m: String| (vec![Block::text(m)], true);
        let Some(entry) = self.entries.iter().find(|e| e.exposed == call.name) else {
            return fail(format!("unknown tool {}", call.name));
        };
        let args = match prepare_args(&entry.tool, &call.args) {
            Ok(a) => a,
            Err(e) => return fail(e),
        };
        let source = &self.sources[entry.source];
        match source.provider.call_tool(&entry.tool, args, cancel).await {
            Ok(r) => (to_blocks(&r, &self.limits), r.is_error),
            Err(McpError::Cancelled) => fail("the tool call was cancelled".into()),
            Err(e) => fail(format!("MCP server {} failed: {e}", source.name)),
        }
    }

    /// Runs calls concurrently; results keep the order of the calls.
    pub async fn call_all(&self, calls: &[ToolCall], cancel: &CancelToken) -> Vec<ToolResult> {
        join_all(calls.iter().map(|c| self.call(c, cancel))).await
    }
}

impl ncore::Tools for Toolbox {
    fn definitions(&self) -> Vec<ToolDef> {
        self.tools()
    }

    fn call<'a>(
        &'a self,
        call: &'a ToolCall,
        cancel: &'a CancelToken,
    ) -> BoxFuture<'a, ToolResult> {
        Box::pin(Toolbox::call(self, call, cancel))
    }
}
