use std::fmt;

use ncore_core::ir::Loss;
use ncore_core::{Format, JsonError, ProviderError};

use crate::rt::TransportError;

/// Failures of the core; proxy paths report provider errors via `Observation`, `generate` as `Provider`.
#[derive(Debug, Clone)]
pub enum Error {
    Cancelled,
    Transport(TransportError),
    Json(JsonError),
    FormatMismatch {
        endpoint: Format,
        request: Format,
    },
    Config(String),
    Plugin {
        name: String,
        message: String,
    },
    /// Body that cannot be converted between formats.
    Codec(String),
    /// Losses refused by `LossPolicy::Error`.
    Loss(Vec<Loss>),
    /// Error response or error event from the provider.
    Provider(ProviderError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => f.write_str("cancelled"),
            Self::Transport(e) => write!(f, "transport: {e}"),
            Self::Json(e) => write!(f, "{e}"),
            Self::FormatMismatch { endpoint, request } => {
                write!(
                    f,
                    "request format {request:?} does not match endpoint format {endpoint:?}"
                )
            }
            Self::Config(m) => write!(f, "config: {m}"),
            Self::Plugin { name, message } => write!(f, "plugin {name}: {message}"),
            Self::Codec(m) => write!(f, "codec: {m}"),
            Self::Loss(l) => {
                let paths: Vec<&str> = l.iter().map(|l| l.path.as_str()).collect();
                write!(f, "conversion would lose: {}", paths.join(", "))
            }
            Self::Provider(e) => match e.status {
                Some(s) => write!(f, "provider error {s}: {}", e.message),
                None => write!(f, "provider error: {}", e.message),
            },
        }
    }
}

impl std::error::Error for Error {}

impl From<TransportError> for Error {
    fn from(e: TransportError) -> Self {
        Self::Transport(e)
    }
}

impl From<JsonError> for Error {
    fn from(e: JsonError) -> Self {
        Self::Json(e)
    }
}
