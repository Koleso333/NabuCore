use bytes::Bytes;
use serde_json::Value;

use crate::Format;

/// Provider-independent error category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidRequest,
    Auth,
    Billing,
    PermissionDenied,
    NotFound,
    Conflict,
    RequestTooLarge,
    RateLimited,
    Overloaded,
    Server,
    Timeout,
    Other,
}

impl ErrorKind {
    /// Whether retrying the same request may succeed.
    pub fn is_transient(self) -> bool {
        matches!(
            self,
            Self::RateLimited | Self::Overloaded | Self::Server | Self::Timeout
        )
    }

    pub fn from_status(status: u16) -> Self {
        match status {
            400 | 422 => Self::InvalidRequest,
            401 => Self::Auth,
            402 => Self::Billing,
            403 => Self::PermissionDenied,
            404 => Self::NotFound,
            409 => Self::Conflict,
            408 | 504 => Self::Timeout,
            413 => Self::RequestTooLarge,
            429 => Self::RateLimited,
            503 | 529 => Self::Overloaded,
            500..=599 => Self::Server,
            _ => Self::Other,
        }
    }

    // Maps documented provider error type strings.
    fn from_code(code: &str) -> Option<Self> {
        Some(match code {
            "invalid_request_error" | "INVALID_ARGUMENT" | "FAILED_PRECONDITION" => {
                Self::InvalidRequest
            }
            "authentication_error" | "UNAUTHENTICATED" => Self::Auth,
            "billing_error" => Self::Billing,
            "permission_error" | "PERMISSION_DENIED" => Self::PermissionDenied,
            "not_found_error" | "NOT_FOUND" => Self::NotFound,
            "conflict_error" => Self::Conflict,
            "request_too_large" => Self::RequestTooLarge,
            "rate_limit_error" | "RESOURCE_EXHAUSTED" | "rate_limit_exceeded" | "too_many_requests" => {
                Self::RateLimited
            }
            "overloaded_error" | "UNAVAILABLE" => Self::Overloaded,
            "api_error" | "INTERNAL" | "server_error" => Self::Server,
            "timeout_error" | "DEADLINE_EXCEEDED" => Self::Timeout,
            _ => return None,
        })
    }
}

/// Error reported by a provider, with the original body kept intact.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderError {
    pub kind: ErrorKind,
    pub status: Option<u16>,
    pub code: Option<String>,
    pub message: String,
    pub raw: Bytes,
}

impl ProviderError {
    /// Parses an error body; `status` is `None` for errors that arrive inside a stream.
    pub fn parse(format: Format, status: Option<u16>, body: Bytes) -> Self {
        let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        Self::from_value(format, status, &v, body)
    }

    pub fn from_value(format: Format, status: Option<u16>, v: &Value, raw: Bytes) -> Self {
        let s = |p: &str| v.pointer(p).and_then(Value::as_str).map(str::to_owned);
        let (code, message) = match format {
            Format::AnthropicMessages => (s("/error/type"), s("/error/message")),
            Format::Gemini => (s("/error/status"), s("/error/message")),
            // `type` may be generic while `code` is the known one, so the first known wins.
            Format::OpenAiChat | Format::OpenAiResponses => {
                let codes: Vec<String> = [s("/error/type"), s("/error/code"), s("/code")]
                    .into_iter()
                    .flatten()
                    .collect();
                let code = codes
                    .iter()
                    .find(|c| ErrorKind::from_code(c).is_some())
                    .or(codes.first())
                    .cloned();
                (code, s("/error/message").or_else(|| s("/message")))
            }
        };
        let kind = code
            .as_deref()
            .and_then(ErrorKind::from_code)
            .or_else(|| status.map(ErrorKind::from_status))
            .unwrap_or(ErrorKind::Other);
        let message = message.unwrap_or_else(|| String::from_utf8_lossy(&raw).into_owned());
        Self {
            kind,
            status,
            code,
            message,
            raw,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_error_shape() {
        let body = Bytes::from_static(
            br#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"},"request_id":"req_1"}"#,
        );
        let e = ProviderError::parse(Format::AnthropicMessages, Some(529), body);
        assert_eq!(e.kind, ErrorKind::Overloaded);
        assert_eq!(e.message, "Overloaded");
    }

    #[test]
    fn openai_code_behind_unknown_type() {
        let body = Bytes::from_static(
            br#"{"error":{"type":"too_many_requests","code":"rate_limit_exceeded","message":"slow down"}}"#,
        );
        let e = ProviderError::parse(Format::OpenAiResponses, None, body);
        assert_eq!(e.kind, ErrorKind::RateLimited);
        assert_eq!(e.message, "slow down");
    }

    #[test]
    fn falls_back_to_status() {
        let e = ProviderError::parse(Format::OpenAiChat, Some(429), Bytes::from_static(b"busy"));
        assert_eq!(e.kind, ErrorKind::RateLimited);
        assert_eq!(e.message, "busy");
    }
}
