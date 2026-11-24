pub mod anthropic;
pub mod chat;
pub mod gemini;
pub mod responses;

use std::fmt;

use bytes::Bytes;
use serde_json::{Map, Value, json};

use crate::ir::{Ext, Fields, Losses, Request, Response, StreamEvent};
use crate::{ErrorKind, Format, ProviderError, SseEvent};

/// Malformed input that cannot be decoded or encoded at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecError(pub String);

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CodecError {}

impl From<serde_json::Error> for CodecError {
    fn from(e: serde_json::Error) -> Self {
        Self(format!("invalid JSON: {e}"))
    }
}

pub(crate) fn err<T>(m: impl Into<String>) -> Result<T, CodecError> {
    Err(CodecError(m.into()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaxTokensField {
    /// Current OpenAI name, required by reasoning models.
    #[default]
    MaxCompletionTokens,
    /// Legacy name many OpenAI-compatible servers still expect.
    MaxTokens,
}

/// Host choices for fields the formats disagree on; the core keeps no model database.
#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    /// Used when the request has no `max_tokens` and the target requires one.
    pub default_max_tokens: Option<u64>,
    pub chat_max_tokens_field: MaxTokensField,
    /// Assistant-message field for reasoning text on OpenAI-compatible servers, e.g. `reasoning_content`.
    pub chat_reasoning_field: Option<String>,
    /// Send cache marks as OpenAI `prompt_cache_breakpoint` (newer models only).
    pub openai_cache_breakpoints: bool,
    /// Send temperature, top_p and top_k to Anthropic, which current models reject.
    pub anthropic_sampling: bool,
    /// Responses `store`; when false, encrypted reasoning is requested for stateless turns.
    pub responses_store: bool,
}

/// Encoded provider request.
#[derive(Debug, Clone)]
pub struct Encoded {
    pub path: String,
    pub body: Bytes,
    pub losses: Losses,
}

/// Turns provider stream events into IR events.
pub trait StreamDecoder: Send {
    fn feed(&mut self, ev: &SseEvent) -> Result<Vec<StreamEvent>, CodecError>;
    /// Called when the byte stream ends; may emit a synthetic end.
    fn finish(&mut self) -> Vec<StreamEvent>;
}

/// Turns IR events into provider SSE bytes.
pub trait StreamEncoder: Send {
    fn encode(&mut self, ev: &StreamEvent) -> Vec<Bytes>;
    fn finish(&mut self) -> Vec<Bytes>;
}

/// Parses a request body; `path` matters for Gemini, which keeps model and streaming in the URL.
pub fn decode_request(format: Format, path: &str, body: &[u8]) -> Result<Request, CodecError> {
    let v: Value = serde_json::from_slice(body)?;
    match format {
        Format::AnthropicMessages => anthropic::decode_request(v),
        Format::OpenAiChat => chat::decode_request(v),
        Format::OpenAiResponses => responses::decode_request(v),
        Format::Gemini => gemini::decode_request(path, v),
    }
}

pub fn encode_request(
    format: Format,
    req: &Request,
    opts: &EncodeOptions,
) -> Result<Encoded, CodecError> {
    let mut losses = Losses::default();
    let (path, body) = match format {
        Format::AnthropicMessages => (
            "/v1/messages".to_owned(),
            anthropic::encode_request(req, opts, &mut losses)?,
        ),
        Format::OpenAiChat => (
            "/v1/chat/completions".to_owned(),
            chat::encode_request(req, opts, &mut losses)?,
        ),
        Format::OpenAiResponses => (
            "/v1/responses".to_owned(),
            responses::encode_request(req, opts, &mut losses)?,
        ),
        Format::Gemini => (
            gemini::path(&req.model, req.stream),
            gemini::encode_request(req, opts, &mut losses)?,
        ),
    };
    Ok(Encoded {
        path,
        body: Bytes::from(serde_json::to_vec(&body)?),
        losses,
    })
}

pub fn decode_response(format: Format, body: &[u8]) -> Result<Response, CodecError> {
    let v: Value = serde_json::from_slice(body)?;
    match format {
        Format::AnthropicMessages => anthropic::decode_response(v),
        Format::OpenAiChat => chat::decode_response(v),
        Format::OpenAiResponses => responses::decode_response(v),
        Format::Gemini => gemini::decode_response(v),
    }
}

pub fn encode_response(format: Format, resp: &Response) -> (Bytes, Losses) {
    let mut losses = Losses::default();
    let v = match format {
        Format::AnthropicMessages => anthropic::encode_response(resp, &mut losses),
        Format::OpenAiChat => chat::encode_response(resp, &mut losses),
        Format::OpenAiResponses => responses::encode_response(resp, &mut losses),
        Format::Gemini => gemini::encode_response(resp, &mut losses),
    };
    (
        Bytes::from(serde_json::to_vec(&v).expect("serializable")),
        losses,
    )
}

pub fn stream_decoder(format: Format) -> Box<dyn StreamDecoder> {
    match format {
        Format::AnthropicMessages => Box::new(anthropic::Decoder::default()),
        Format::OpenAiChat => Box::new(chat::Decoder::default()),
        Format::OpenAiResponses => Box::new(responses::Decoder::default()),
        Format::Gemini => Box::new(gemini::Decoder::default()),
    }
}

pub fn stream_encoder(format: Format) -> Box<dyn StreamEncoder> {
    match format {
        Format::AnthropicMessages => Box::new(anthropic::Encoder::default()),
        Format::OpenAiChat => Box::new(chat::Encoder::default()),
        Format::OpenAiResponses => Box::new(responses::Encoder::default()),
        Format::Gemini => Box::new(gemini::Encoder::default()),
    }
}

/// Error body in `format`'s own shape, with the status to send.
pub fn encode_error(format: Format, e: &ProviderError) -> (u16, Bytes) {
    let status = e.status.unwrap_or_else(|| default_status(e.kind));
    let v = match format {
        Format::AnthropicMessages => json!({
            "type": "error",
            "error": {"type": anthropic::error_type(e.kind), "message": e.message},
        }),
        Format::OpenAiChat | Format::OpenAiResponses => json!({
            "error": {
                "message": e.message,
                "type": openai_error_type(e.kind),
                "param": null,
                "code": e.code,
            }
        }),
        Format::Gemini => json!({
            "error": {"code": status, "message": e.message, "status": gemini::error_status(e.kind)}
        }),
    };
    (
        status,
        Bytes::from(serde_json::to_vec(&v).expect("serializable")),
    )
}

pub(crate) fn default_status(kind: ErrorKind) -> u16 {
    match kind {
        ErrorKind::InvalidRequest => 400,
        ErrorKind::Auth => 401,
        ErrorKind::Billing => 402,
        ErrorKind::PermissionDenied => 403,
        ErrorKind::NotFound => 404,
        ErrorKind::Conflict => 409,
        ErrorKind::RequestTooLarge => 413,
        ErrorKind::RateLimited => 429,
        ErrorKind::Overloaded => 503,
        ErrorKind::Timeout => 504,
        ErrorKind::Server | ErrorKind::Other => 500,
    }
}

fn openai_error_type(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::InvalidRequest | ErrorKind::RequestTooLarge | ErrorKind::NotFound => {
            "invalid_request_error"
        }
        ErrorKind::Auth => "authentication_error",
        ErrorKind::PermissionDenied => "permission_error",
        ErrorKind::RateLimited | ErrorKind::Billing => "rate_limit_error",
        _ => "server_error",
    }
}

// ---- shared helpers ----

/// Splits off fields not listed in `known`.
pub(crate) fn rest(map: &Map<String, Value>, known: &[&str]) -> Fields {
    map.iter()
        .filter(|(k, _)| !known.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

pub(crate) fn ext(format: Format, map: &Map<String, Value>, known: &[&str]) -> Ext {
    Ext::new(format, rest(map, known))
}

/// Adds extension fields without overriding what the encoder wrote; objects merge one level.
pub(crate) fn merge(out: &mut Map<String, Value>, fields: Option<Fields>) {
    for (k, v) in fields.into_iter().flatten() {
        match (out.get_mut(&k), v) {
            (Some(Value::Object(dst)), Value::Object(src)) => {
                for (sk, sv) in src {
                    dst.entry(sk).or_insert(sv);
                }
            }
            (Some(_), _) => {}
            (None, v) => {
                out.insert(k, v);
            }
        }
    }
}

pub(crate) fn s(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_owned)
}

pub(crate) fn obj(v: &Value) -> Result<&Map<String, Value>, CodecError> {
    v.as_object()
        .ok_or_else(|| CodecError(format!("expected object, got {v}")))
}

/// `data:<mime>;base64,<data>` → (mime, data).
pub(crate) fn parse_data_url(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (head, data) = rest.split_once(',')?;
    let mime = head.strip_suffix(";base64")?;
    Some((mime.to_owned(), data.to_owned()))
}

pub(crate) fn data_url(mime: &str, data: &str) -> String {
    format!("data:{mime};base64,{data}")
}

/// Arguments as a JSON object; invalid or non-object text is wrapped and reported.
pub(crate) fn args_object(args: &str, path: &str, losses: &mut Losses) -> Value {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        return json!({});
    }
    match serde_json::from_str::<Value>(trimmed) {
        Ok(v @ Value::Object(_)) => v,
        Ok(other) => {
            losses.approx(
                path,
                "tool arguments are not an object; wrapped in {\"value\": ...}",
            );
            json!({ "value": other })
        }
        Err(_) => {
            losses.approx(
                path,
                "tool arguments are not valid JSON; wrapped in {\"raw\": ...}",
            );
            json!({ "raw": args })
        }
    }
}

pub(crate) fn sse(event: Option<&str>, data: &Value) -> Bytes {
    crate::sse::encode(
        event,
        None,
        &serde_json::to_string(data).expect("serializable"),
    )
}

/// Deterministic id for a call that arrived without one.
/// Text placed before tool-result media moved into a user turn.
pub(crate) fn attachments_label(call_id: &str) -> String {
    format!("[attachments returned by tool call {call_id}]")
}

pub(crate) fn synth_call_id(msg: usize, block: usize) -> String {
    format!("call_{msg}_{block}")
}

/// Effort for a token budget, used where a format has only levels.
pub(crate) fn effort_for_budget(budget: u64) -> &'static str {
    match budget {
        0..=2048 => "low",
        2049..=16384 => "medium",
        _ => "high",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_urls() {
        assert_eq!(
            parse_data_url("data:image/png;base64,AAA="),
            Some(("image/png".into(), "AAA=".into()))
        );
        assert_eq!(parse_data_url("https://x/y.png"), None);
    }

    #[test]
    fn merge_keeps_encoder_fields() {
        let mut out = json!({"a": 1, "o": {"x": 1}}).as_object().unwrap().clone();
        let f = json!({"a": 2, "b": 3, "o": {"x": 2, "y": 3}})
            .as_object()
            .unwrap()
            .clone();
        merge(&mut out, Some(f));
        assert_eq!(
            Value::Object(out),
            json!({"a": 1, "o": {"x": 1, "y": 3}, "b": 3})
        );
    }
}
