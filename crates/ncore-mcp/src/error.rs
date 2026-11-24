use std::fmt;
use std::time::Duration;

use crate::jsonrpc::RpcError;

/// Failure talking to an MCP server.
#[derive(Debug, Clone)]
pub enum McpError {
    /// The server could not be started or reached.
    Connect(String),
    /// The server went away; `stderr` holds its last lines when it was a process.
    Closed {
        reason: String,
        stderr: Vec<String>,
    },
    Timeout {
        method: String,
        after: Duration,
    },
    Cancelled,
    /// Non-success HTTP status; `rpc` is the JSON-RPC error in the body, if any.
    Http {
        status: u16,
        body: String,
        rpc: Option<Box<RpcError>>,
    },
    /// HTTP 401/403; the host has to supply credentials.
    Auth {
        status: u16,
        challenge: Option<String>,
    },
    /// JSON-RPC error returned by the server.
    Rpc(Box<RpcError>),
    /// Malformed or unexpected traffic from the server.
    Protocol(String),
    /// No protocol version or feature both sides support.
    Unsupported(String),
}

impl McpError {
    pub fn closed(reason: impl Into<String>) -> Self {
        Self::Closed {
            reason: reason.into(),
            stderr: Vec::new(),
        }
    }

    /// The JSON-RPC error carried by this failure, if any.
    pub fn rpc(&self) -> Option<&RpcError> {
        match self {
            Self::Rpc(e) => Some(e),
            Self::Http { rpc, .. } => rpc.as_deref(),
            _ => None,
        }
    }
}

impl fmt::Display for McpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connect(m) => write!(f, "cannot connect: {m}"),
            Self::Closed { reason, stderr } => {
                write!(f, "connection closed: {reason}")?;
                if let Some(last) = stderr.last() {
                    write!(f, " (stderr: {last})")?;
                }
                Ok(())
            }
            Self::Timeout { method, after } => {
                write!(f, "{method} timed out after {}s", after.as_secs_f32())
            }
            Self::Cancelled => f.write_str("cancelled"),
            Self::Http { status, body, rpc } => match rpc {
                Some(e) => write!(f, "HTTP {status}: {e}"),
                None => write!(f, "HTTP {status}: {}", crate::content::clip(body, 300)),
            },
            Self::Auth { status, .. } => write!(f, "HTTP {status}: authorization required"),
            Self::Rpc(e) => write!(f, "{e}"),
            Self::Protocol(m) => write!(f, "protocol error: {m}"),
            Self::Unsupported(m) => write!(f, "unsupported: {m}"),
        }
    }
}

impl std::error::Error for McpError {}
