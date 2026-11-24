mod client;
pub mod content;
mod error;
mod http;
pub mod jsonrpc;
mod stdio;
pub mod tool;
mod toolbox;
mod wire;

pub use client::{
    CallOptions, Era, LEGACY_VERSIONS, MODERN_VERSIONS, McpClient, Runtime, ServerConfig,
    ServerInfo, ServerTransport, Timeouts,
};
pub use content::{CallResult, Content, ResultLimits};
pub use error::McpError;
pub use jsonrpc::RpcError;
pub use tool::McpTool;
pub use toolbox::{Refresh, ToolProvider, Toolbox};
pub use wire::{ClientHandler, EventSink, McpEvent, NoHandler};
