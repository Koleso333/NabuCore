use std::collections::HashMap;

use bytes::Bytes;
use serde_json::{Map, Value, json};

use super::{
    CodecError, EncodeOptions, StreamDecoder, StreamEncoder, args_object, err, ext, merge, obj,
    rest, s, sse,
};
use crate::ir::*;
use crate::{ErrorKind, Format, ProviderError, SseEvent, Usage};

const F: Format = Format::AnthropicMessages;

const REQ_KNOWN: &[&str] = &[
    "model",
    "max_tokens",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "thinking",
    "output_config",
    "stream",
];

fn cache_of(b: &Value) -> Option<CacheMark> {
    let c = b.get("cache_control")?;
    Some(CacheMark { ttl: s(c, "ttl") })
}

fn cache_json(c: &Option<CacheMark>) -> Option<Value> {
    c.as_ref().map(|c| match &c.ttl {
        Some(t) if t == "5m" || t == "1h" => json!({"type": "ephemeral", "ttl": t}),
        _ => json!({"type": "ephemeral"}),
    })
}

fn decode_media(b: &Value, default_mime: Option<&str>) -> Option<Media> {
    let src = b.get("source")?;
    let mime = s(src, "media_type").or(default_mime.map(str::to_owned));
    let source = match src.get("type")?.as_str()? {
        "base64" => MediaSource::Base64(s(src, "data")?),
        "url" => MediaSource::Url(s(src, "url")?),
        "file" => MediaSource::FileId(s(src, "file_id")?),
        _ => return None,
    };
    Some(Media {
        mime,
        source,
        filename: s(b, "title"),
    })
}

pub(crate) fn decode_block(b: &Value) -> Result<Block, CodecError> {
    let m = obj(b)?;
    let ty = m.get("type").and_then(Value::as_str).unwrap_or("");
    let opaque = || Block::Opaque(Opaque::new(F, b.clone()));
    Ok(match ty {
        "text" => Block::Text {
            text: s(b, "text").unwrap_or_default(),
            cache: cache_of(b),
            sig: None,
            ext: ext(F, m, &["type", "text", "cache_control"]),
        },
        "image" | "document" => {
            let default = (ty == "document").then_some("application/pdf");
            match decode_media(b, default) {
                Some(media) => Block::Media {
                    media,
                    cache: cache_of(b),
                    ext: ext(F, m, &["type", "source", "cache_control", "title"]),
                },
                None => opaque(),
            }
        }
        "tool_use" => Block::ToolCall(ToolCall {
            id: s(b, "id").unwrap_or_default(),
            name: s(b, "name").unwrap_or_default(),
            args: serde_json::to_string(b.get("input").unwrap_or(&json!({})))?,
            sig: None,
        }),
        "tool_result" => {
            let content = match b.get("content") {
                Some(Value::String(t)) => vec![Block::text(t.clone())],
                Some(Value::Array(items)) => {
                    items.iter().map(decode_block).collect::<Result<_, _>>()?
                }
                _ => Vec::new(),
            };
            Block::ToolResult(ToolResult {
                call_id: s(b, "tool_use_id").unwrap_or_default(),
                name: None,
                content,
                is_error: b.get("is_error").and_then(Value::as_bool).unwrap_or(false),
                cache: cache_of(b),
            })
        }
        "thinking" => Block::Thinking(Thinking {
            text: s(b, "thinking").unwrap_or_default(),
            sig: b.get("signature").map(|v| Opaque::signature(F, v.clone())),
            redacted: false,
        }),
        "redacted_thinking" => Block::Thinking(Thinking {
            text: String::new(),
            sig: b.get("data").map(|v| Opaque::new(F, v.clone())),
            redacted: true,
        }),
        _ => opaque(),
    })
}

fn decode_content(v: Option<&Value>) -> Result<Vec<Block>, CodecError> {
    match v {
        Some(Value::String(t)) => Ok(vec![Block::text(t.clone())]),
        Some(Value::Array(items)) => items.iter().map(decode_block).collect(),
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(other) => err(format!("unexpected content: {other}")),
    }
}

pub fn decode_request(v: Value) -> Result<Request, CodecError> {
    let m = obj(&v)?;
    let mut req = Request {
        model: s(&v, "model").unwrap_or_default(),
        max_tokens: v.get("max_tokens").and_then(Value::as_u64),
        temperature: v.get("temperature").and_then(Value::as_f64),
        top_p: v.get("top_p").and_then(Value::as_f64),
        top_k: v.get("top_k").and_then(Value::as_u64),
        stream: v.get("stream").and_then(Value::as_bool).unwrap_or(false),
        system: decode_content(v.get("system"))?,
        ..Request::default()
    };
    let mut fields = rest(m, REQ_KNOWN);
    if let Some(Value::Array(stops)) = v.get("stop_sequences") {
        req.stop = stops
            .iter()
            .filter_map(|s| s.as_str().map(str::to_owned))
            .collect();
    }
    for msg in v
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let mm = obj(msg)?;
        let role = match mm.get("role").and_then(Value::as_str) {
            Some("assistant") => Role::Assistant,
            _ => Role::User,
        };
        req.messages.push(Message {
            role,
            content: decode_content(mm.get("content"))?,
            ext: ext(F, mm, &["role", "content"]),
        });
    }
    for t in v
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let tm = obj(t)?;
        let custom = matches!(
            tm.get("type").and_then(Value::as_str),
            None | Some("custom")
        );
        req.tools
            .push(if custom && tm.contains_key("input_schema") {
                ToolDef::Function(Tool {
                    name: s(t, "name").unwrap_or_default(),
                    description: s(t, "description"),
                    schema: t["input_schema"].clone(),
                    strict: t.get("strict").and_then(Value::as_bool),
                    cache: cache_of(t),
                    ext: ext(
                        F,
                        tm,
                        &[
                            "type",
                            "name",
                            "description",
                            "input_schema",
                            "strict",
                            "cache_control",
                        ],
                    ),
                })
            } else {
                ToolDef::Opaque(Opaque::new(F, t.clone()))
            });
    }
    if let Some(tc) = v.get("tool_choice") {
        req.tool_choice = match tc.get("type").and_then(Value::as_str) {
            Some("auto") => Some(ToolChoice::Auto),
            Some("any") => Some(ToolChoice::Required),
            Some("none") => Some(ToolChoice::None),
            Some("tool") => Some(ToolChoice::Tool(s(tc, "name").unwrap_or_default())),
            _ => None,
        };
        if let Some(d) = tc.get("disable_parallel_tool_use").and_then(Value::as_bool) {
            req.parallel_tool_calls = Some(!d);
        }
    }
    let mut reasoning = ReasoningCfg::default();
    if let Some(th) = v.get("thinking") {
        match th.get("type").and_then(Value::as_str) {
            Some("enabled") => {
                reasoning.mode = Some(ReasoningMode::On);
                reasoning.budget_tokens = th.get("budget_tokens").and_then(Value::as_u64);
            }
            Some("adaptive") => reasoning.mode = Some(ReasoningMode::Adaptive),
            Some("disabled") => reasoning.mode = Some(ReasoningMode::Off),
            _ => {
                fields.insert("thinking".into(), th.clone());
            }
        }
        match th.get("display").and_then(Value::as_str) {
            Some("summarized") => reasoning.include = Some(true),
            Some("omitted") => reasoning.include = Some(false),
            _ => {}
        }
    }
    if let Some(oc) = v.get("output_config").and_then(Value::as_object) {
        reasoning.effort = oc.get("effort").and_then(Value::as_str).map(str::to_owned);
        if let Some(f) = oc.get("format")
            && f.get("type").and_then(Value::as_str) == Some("json_schema")
        {
            req.output = Some(OutputFormat::JsonSchema {
                name: None,
                schema: f.get("schema").cloned().unwrap_or(json!({})),
                strict: None,
            });
        }
        let other = rest(oc, &["effort", "format"]);
        if !other.is_empty() {
            fields.insert("output_config".into(), Value::Object(other));
        }
    }
    if reasoning != ReasoningCfg::default() {
        req.reasoning = Some(reasoning);
    }
    req.ext = Ext::new(F, fields);
    Ok(req)
}

fn media_json(
    media: &Media,
    cache: &Option<CacheMark>,
    path: &str,
    losses: &mut Losses,
) -> Option<Value> {
    let mime = media.mime.clone();
    let is_pdf = mime.as_deref() == Some("application/pdf");
    if !media.is_image() && !is_pdf {
        losses.dropped(
            path,
            format!("media type {mime:?} is not supported by Anthropic"),
        );
        return None;
    }
    let source = match &media.source {
        MediaSource::Base64(d) => {
            json!({"type": "base64", "media_type": mime.unwrap_or_default(), "data": d})
        }
        MediaSource::Url(u) => json!({"type": "url", "url": u}),
        MediaSource::FileId(id) => json!({"type": "file", "file_id": id}),
    };
    let mut b = Map::new();
    b.insert(
        "type".into(),
        json!(if is_pdf { "document" } else { "image" }),
    );
    b.insert("source".into(), source);
    if let (true, Some(name)) = (is_pdf, &media.filename) {
        b.insert("title".into(), json!(name));
    }
    if let Some(c) = cache_json(cache) {
        b.insert("cache_control".into(), c);
    }
    Some(Value::Object(b))
}

fn encode_block(b: &Block, path: &str, losses: &mut Losses) -> Option<Value> {
    encode_block_for(b, path, losses, false)
}

// Responses to clients may show unsigned reasoning; requests to Anthropic may not.
fn encode_block_for(b: &Block, path: &str, losses: &mut Losses, response: bool) -> Option<Value> {
    match b {
        Block::Text {
            text,
            cache,
            sig,
            ext,
        } => {
            if sig.is_some() {
                losses.dropped(
                    format!("{path}/signature"),
                    "text signature from another provider",
                );
            }
            if text.is_empty() {
                return None;
            }
            let mut o = Map::new();
            o.insert("type".into(), json!("text"));
            o.insert("text".into(), json!(text));
            if let Some(c) = cache_json(cache) {
                o.insert("cache_control".into(), c);
            }
            merge(&mut o, losses.ext(path, ext, F));
            Some(Value::Object(o))
        }
        Block::Media { media, cache, ext } => {
            let mut v = media_json(media, cache, path, losses)?;
            merge(v.as_object_mut()?, losses.ext(path, ext, F));
            Some(v)
        }
        Block::ToolCall(c) => {
            if c.sig.as_ref().is_some_and(|o| o.format != F) {
                losses.dropped(
                    format!("{path}/signature"),
                    "tool call signature from another provider",
                );
            }
            Some(json!({
                "type": "tool_use",
                "id": c.id,
                "name": c.name,
                "input": args_object(&c.args, &format!("{path}/input"), losses),
            }))
        }
        Block::ToolResult(r) => {
            let mut o = Map::new();
            o.insert("type".into(), json!("tool_result"));
            o.insert("tool_use_id".into(), json!(r.call_id));
            let simple = matches!(r.content.as_slice(), [Block::Text { cache: None, sig: None, ext, .. }] if ext.is_empty());
            if simple {
                o.insert("content".into(), json!(plain_text(&r.content)));
            } else if !r.content.is_empty() {
                let items: Vec<Value> = r
                    .content
                    .iter()
                    .enumerate()
                    .filter_map(|(i, b)| encode_block(b, &format!("{path}/content/{i}"), losses))
                    .collect();
                o.insert("content".into(), Value::Array(items));
            }
            if r.is_error {
                o.insert("is_error".into(), json!(true));
            }
            if let Some(c) = cache_json(&r.cache) {
                o.insert("cache_control".into(), c);
            }
            Some(Value::Object(o))
        }
        Block::Thinking(t) => match t.sig.as_ref().and_then(|o| o.for_format(F)) {
            Some(sig) if t.redacted => Some(json!({"type": "redacted_thinking", "data": sig})),
            Some(sig) => Some(json!({"type": "thinking", "thinking": t.text, "signature": sig})),
            None if response && !t.text.is_empty() => {
                Some(json!({"type": "thinking", "thinking": t.text, "signature": ""}))
            }
            None => {
                losses.dropped(path, "reasoning without an Anthropic signature");
                None
            }
        },
        Block::Opaque(o) => match o.for_format(F) {
            Some(v) => Some(v.clone()),
            None => {
                losses.dropped(
                    path,
                    format!("{:?} block has no Anthropic equivalent", o.format),
                );
                None
            }
        },
    }
}

fn encode_blocks(blocks: &[Block], path: &str, losses: &mut Losses) -> Vec<Value> {
    blocks
        .iter()
        .enumerate()
        .filter_map(|(i, b)| encode_block(b, &format!("{path}/{i}"), losses))
        .collect()
}

pub fn encode_request(
    req: &Request,
    opts: &EncodeOptions,
    losses: &mut Losses,
) -> Result<Value, CodecError> {
    let mut o = Map::new();
    o.insert("model".into(), json!(req.model));
    let Some(mut max) = req.max_tokens.or(opts.default_max_tokens) else {
        return err("Anthropic requires max_tokens; set it or EncodeOptions::default_max_tokens");
    };
    let system = encode_blocks(&req.system, "/system", losses);
    let mut messages = Vec::new();
    for (i, m) in req.messages.iter().enumerate() {
        let path = format!("/messages/{i}");
        let role = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => {
                losses.approx(
                    &path,
                    "mid-conversation system message sent as a user message",
                );
                "user"
            }
        };
        let content = encode_blocks(&m.content, &format!("{path}/content"), losses);
        if content.is_empty() {
            continue;
        }
        let mut mo = Map::new();
        mo.insert("role".into(), json!(role));
        mo.insert("content".into(), Value::Array(content));
        merge(&mut mo, losses.ext(&path, &m.ext, F));
        messages.push(Value::Object(mo));
    }
    let mut tools = Vec::new();
    for (i, t) in req.tools.iter().enumerate() {
        let path = format!("/tools/{i}");
        match t {
            ToolDef::Function(t) => {
                let mut to = Map::new();
                to.insert("name".into(), json!(t.name));
                if let Some(d) = &t.description {
                    to.insert("description".into(), json!(d));
                }
                to.insert("input_schema".into(), t.schema.clone());
                if let Some(st) = t.strict {
                    to.insert("strict".into(), json!(st));
                }
                if let Some(c) = cache_json(&t.cache) {
                    to.insert("cache_control".into(), c);
                }
                merge(&mut to, losses.ext(&path, &t.ext, F));
                tools.push(Value::Object(to));
            }
            ToolDef::Opaque(op) => match op.for_format(F) {
                Some(v) => tools.push(v.clone()),
                None => losses.dropped(path, format!("{:?} hosted tool", op.format)),
            },
        }
    }

    let mut thinking = None;
    let mut output_config = Map::new();
    if let Some(r) = &req.reasoning {
        match (r.mode, r.budget_tokens) {
            (Some(ReasoningMode::Off), _) => thinking = Some(json!({"type": "disabled"})),
            (_, Some(budget)) => {
                let budget = budget.max(1024);
                if budget >= max {
                    losses.approx("/max_tokens", "raised above the thinking budget");
                    max = budget + 1024;
                }
                thinking = Some(json!({"type": "enabled", "budget_tokens": budget}));
            }
            (Some(_), None) => thinking = Some(json!({"type": "adaptive"})),
            (None, None) => {}
        }
        if let (Some(th), Some(inc)) = (thinking.as_mut(), r.include)
            && th["type"] != "disabled"
        {
            th["display"] = json!(if inc { "summarized" } else { "omitted" });
        }
        if let Some(e) = &r.effort {
            let mapped = match e.as_str() {
                "none" => None,
                "minimal" => {
                    losses.approx("/output_config/effort", "minimal mapped to low");
                    Some("low")
                }
                "low" | "medium" | "high" | "xhigh" | "max" => Some(e.as_str()),
                other => {
                    losses.dropped("/output_config/effort", format!("unknown effort {other}"));
                    None
                }
            };
            if e == "none" && thinking.is_none() {
                thinking = Some(json!({"type": "disabled"}));
            }
            if let Some(m) = mapped {
                output_config.insert("effort".into(), json!(m));
            }
        }
    }
    match &req.output {
        Some(OutputFormat::JsonSchema { schema, .. }) => {
            output_config.insert(
                "format".into(),
                json!({"type": "json_schema", "schema": schema}),
            );
        }
        Some(OutputFormat::JsonObject) => losses.dropped(
            "/output_config/format",
            "Anthropic needs a schema for JSON output",
        ),
        None => {}
    }

    o.insert("max_tokens".into(), json!(max));
    match system.as_slice() {
        [] => {}
        [one] if one.as_object().is_some_and(|m| m.len() == 2) => {
            o.insert("system".into(), one["text"].clone());
        }
        _ => {
            o.insert("system".into(), Value::Array(system));
        }
    }
    o.insert("messages".into(), Value::Array(messages));
    if !tools.is_empty() {
        o.insert("tools".into(), Value::Array(tools));
    }
    let disable = req.parallel_tool_calls == Some(false);
    let mut choice = match &req.tool_choice {
        Some(ToolChoice::Auto) => Some(json!({"type": "auto"})),
        Some(ToolChoice::None) => Some(json!({"type": "none"})),
        Some(ToolChoice::Required) => Some(json!({"type": "any"})),
        Some(ToolChoice::Tool(n)) => Some(json!({"type": "tool", "name": n})),
        None if disable => Some(json!({"type": "auto"})),
        None => None,
    };
    if let (Some(c), true) = (choice.as_mut(), disable)
        && c["type"] != "none"
    {
        c["disable_parallel_tool_use"] = json!(true);
    }
    if let Some(c) = choice {
        o.insert("tool_choice".into(), c);
    }
    if !req.stop.is_empty() {
        o.insert("stop_sequences".into(), json!(req.stop));
    }
    for (k, v) in [
        ("temperature", req.temperature.map(|x| json!(x))),
        ("top_p", req.top_p.map(|x| json!(x))),
        ("top_k", req.top_k.map(|x| json!(x))),
    ] {
        if let Some(v) = v {
            if opts.anthropic_sampling {
                o.insert(k.into(), v);
            } else {
                losses.dropped(format!("/{k}"), "not accepted by current Anthropic models");
            }
        }
    }
    if req.seed.is_some() {
        losses.dropped("/seed", "Anthropic has no seed");
    }
    if let Some(t) = thinking {
        o.insert("thinking".into(), t);
    }
    if !output_config.is_empty() {
        o.insert("output_config".into(), Value::Object(output_config));
    }
    if req.stream {
        o.insert("stream".into(), json!(true));
    }
    merge(&mut o, losses.ext("", &req.ext, F));
    Ok(Value::Object(o))
}

pub(crate) fn stop_of(reason: Option<&str>, seq: Option<String>) -> Option<StopReason> {
    Some(match reason? {
        "end_turn" => StopReason::EndTurn,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence(seq),
        "tool_use" => StopReason::ToolUse,
        "pause_turn" => StopReason::Pause,
        "refusal" => StopReason::Refusal,
        "model_context_window_exceeded" => StopReason::ContextWindow,
        other => StopReason::Other(other.to_owned()),
    })
}

fn stop_json(stop: &Option<StopReason>) -> (Value, Value) {
    let r = match stop {
        None => return (Value::Null, Value::Null),
        Some(StopReason::StopSequence(seq)) => return (json!("stop_sequence"), json!(seq)),
        Some(StopReason::EndTurn) | Some(StopReason::Other(_)) => "end_turn",
        Some(StopReason::MaxTokens) => "max_tokens",
        Some(StopReason::ToolUse) => "tool_use",
        Some(StopReason::Pause) => "pause_turn",
        Some(StopReason::Refusal) | Some(StopReason::ContentFilter) => "refusal",
        Some(StopReason::ContextWindow) => "model_context_window_exceeded",
    };
    (json!(r), Value::Null)
}

pub(crate) fn usage_json(u: &Usage) -> Value {
    let read = u.cache_read.unwrap_or(0);
    let write = u.cache_write.unwrap_or(0);
    let input = u.input_total.unwrap_or(0).saturating_sub(read + write);
    let mut o = json!({
        "input_tokens": input,
        "cache_creation_input_tokens": write,
        "cache_read_input_tokens": read,
        "output_tokens": u.output_total.unwrap_or(0),
    });
    if u.cache_write_5m.is_some() || u.cache_write_1h.is_some() {
        o["cache_creation"] = json!({
            "ephemeral_5m_input_tokens": u.cache_write_5m.unwrap_or(0),
            "ephemeral_1h_input_tokens": u.cache_write_1h.unwrap_or(0),
        });
    }
    if let Some(r) = u.reasoning {
        o["output_tokens_details"] = json!({"thinking_tokens": r});
    }
    o
}

pub fn decode_response(v: Value) -> Result<Response, CodecError> {
    let m = obj(&v)?;
    Ok(Response {
        id: s(&v, "id"),
        model: s(&v, "model"),
        content: decode_content(v.get("content"))?,
        stop: stop_of(
            v.get("stop_reason").and_then(Value::as_str),
            s(&v, "stop_sequence"),
        ),
        usage: v
            .get("usage")
            .map(|u| Usage::from_provider(F, u))
            .unwrap_or_default(),
        ext: ext(
            F,
            m,
            &[
                "id",
                "type",
                "role",
                "model",
                "content",
                "stop_reason",
                "stop_sequence",
                "usage",
            ],
        ),
    })
}

pub fn encode_response(r: &Response, losses: &mut Losses) -> Value {
    let (reason, seq) = stop_json(&r.stop);
    let mut o = Map::new();
    o.insert(
        "id".into(),
        json!(r.id.clone().unwrap_or_else(|| "msg_nabucore".into())),
    );
    o.insert("type".into(), json!("message"));
    o.insert("role".into(), json!("assistant"));
    o.insert("model".into(), json!(r.model.clone().unwrap_or_default()));
    let content: Vec<Value> = r
        .content
        .iter()
        .enumerate()
        .filter_map(|(i, b)| encode_block_for(b, &format!("/content/{i}"), losses, true))
        .collect();
    o.insert("content".into(), Value::Array(content));
    o.insert("stop_reason".into(), reason);
    o.insert("stop_sequence".into(), seq);
    o.insert("usage".into(), usage_json(&r.usage));
    merge(&mut o, losses.ext("", &r.ext, F));
    Value::Object(o)
}

pub(crate) fn error_type(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::InvalidRequest | ErrorKind::Conflict => "invalid_request_error",
        ErrorKind::Auth => "authentication_error",
        ErrorKind::Billing => "billing_error",
        ErrorKind::PermissionDenied => "permission_error",
        ErrorKind::NotFound => "not_found_error",
        ErrorKind::RequestTooLarge => "request_too_large",
        ErrorKind::RateLimited => "rate_limit_error",
        ErrorKind::Overloaded => "overloaded_error",
        ErrorKind::Timeout => "timeout_error",
        ErrorKind::Server | ErrorKind::Other => "api_error",
    }
}

// Anthropic block whose start is buffered until its input is complete.
struct Pending {
    block: Value,
    input: String,
}

/// Anthropic SSE → IR events.
#[derive(Default)]
pub struct Decoder {
    map: HashMap<u64, usize>,
    pending: HashMap<u64, Pending>,
    next: usize,
    usage: Usage,
    stop: Option<StopReason>,
    ended: bool,
}

impl Decoder {
    fn open(&mut self, idx: u64) -> usize {
        let i = self.next;
        self.next += 1;
        self.map.insert(idx, i);
        i
    }
}

impl StreamDecoder for Decoder {
    fn feed(&mut self, ev: &SseEvent) -> Result<Vec<StreamEvent>, CodecError> {
        if !ev.has_data() {
            return Ok(Vec::new());
        }
        let v: Value = serde_json::from_str(&ev.data)?;
        let mut out = Vec::new();
        let idx = v.get("index").and_then(Value::as_u64).unwrap_or(0);
        match v.get("type").and_then(Value::as_str).unwrap_or("") {
            "message_start" => {
                let msg = &v["message"];
                out.push(StreamEvent::Start {
                    id: s(msg, "id"),
                    model: s(msg, "model"),
                });
                if let Some(u) = msg.get("usage") {
                    self.usage.merge(&Usage::from_provider(F, u));
                    out.push(StreamEvent::Usage(self.usage.clone()));
                }
            }
            "content_block_start" => {
                let cb = &v["content_block"];
                match cb.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text" => {
                        let i = self.open(idx);
                        out.push(StreamEvent::BlockStart {
                            index: i,
                            block: BlockStart::Text,
                        });
                        if let Some(t) = cb
                            .get("text")
                            .and_then(Value::as_str)
                            .filter(|t| !t.is_empty())
                        {
                            out.push(StreamEvent::Delta {
                                index: i,
                                delta: Delta::Text(t.to_owned()),
                            });
                        }
                    }
                    "thinking" => {
                        let i = self.open(idx);
                        out.push(StreamEvent::BlockStart {
                            index: i,
                            block: BlockStart::Thinking { redacted: false },
                        });
                    }
                    "redacted_thinking" => {
                        let i = self.open(idx);
                        out.push(StreamEvent::BlockStart {
                            index: i,
                            block: BlockStart::Thinking { redacted: true },
                        });
                        if let Some(d) = cb.get("data") {
                            out.push(StreamEvent::Delta {
                                index: i,
                                delta: Delta::Sig(Opaque::new(F, d.clone())),
                            });
                        }
                    }
                    "tool_use" => {
                        let i = self.open(idx);
                        out.push(StreamEvent::BlockStart {
                            index: i,
                            block: BlockStart::ToolCall {
                                id: s(cb, "id").unwrap_or_default(),
                                name: s(cb, "name").unwrap_or_default(),
                            },
                        });
                    }
                    _ => {
                        self.pending.insert(
                            idx,
                            Pending {
                                block: cb.clone(),
                                input: String::new(),
                            },
                        );
                    }
                }
            }
            "content_block_delta" => {
                let d = &v["delta"];
                if let Some(p) = self.pending.get_mut(&idx) {
                    if let Some(j) = d.get("partial_json").and_then(Value::as_str) {
                        p.input.push_str(j);
                    }
                    return Ok(out);
                }
                let Some(&i) = self.map.get(&idx) else {
                    return Ok(out);
                };
                let delta = match d.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text_delta" => Delta::Text(s(d, "text").unwrap_or_default()),
                    "thinking_delta" => Delta::Thinking(s(d, "thinking").unwrap_or_default()),
                    "signature_delta" => Delta::Sig(Opaque::signature(
                        F,
                        d.get("signature").cloned().unwrap_or_default(),
                    )),
                    "input_json_delta" => Delta::ToolArgs(s(d, "partial_json").unwrap_or_default()),
                    _ => return Ok(out),
                };
                out.push(StreamEvent::Delta { index: i, delta });
            }
            "content_block_stop" => {
                if let Some(mut p) = self.pending.remove(&idx) {
                    if !p.input.is_empty()
                        && let Ok(input) = serde_json::from_str::<Value>(&p.input)
                    {
                        p.block["input"] = input;
                    }
                    let i = self.open(idx);
                    out.push(StreamEvent::BlockStart {
                        index: i,
                        block: BlockStart::Opaque(Opaque::new(F, p.block)),
                    });
                    out.push(StreamEvent::BlockEnd { index: i });
                } else if let Some(&i) = self.map.get(&idx) {
                    out.push(StreamEvent::BlockEnd { index: i });
                }
            }
            "message_delta" => {
                let d = &v["delta"];
                if let Some(stop) = stop_of(
                    d.get("stop_reason").and_then(Value::as_str),
                    s(d, "stop_sequence"),
                ) {
                    self.stop = Some(stop);
                }
                if let Some(u) = v.get("usage") {
                    self.usage.merge(&Usage::from_provider(F, u));
                    out.push(StreamEvent::Usage(self.usage.clone()));
                }
            }
            "message_stop" => {
                self.ended = true;
                out.push(StreamEvent::End {
                    stop: self.stop.clone(),
                    usage: self.usage.clone(),
                });
            }
            "error" => out.push(StreamEvent::Error(ProviderError::from_value(
                F,
                None,
                &v,
                Bytes::copy_from_slice(ev.data.as_bytes()),
            ))),
            _ => {}
        }
        Ok(out)
    }

    fn finish(&mut self) -> Vec<StreamEvent> {
        Vec::new()
    }
}

/// IR events → Anthropic SSE.
#[derive(Default)]
pub struct Encoder {
    started: bool,
    id: Option<String>,
    model: Option<String>,
    map: HashMap<usize, u64>,
    skipped: Vec<usize>,
    next: u64,
    usage: Usage,
    ended: bool,
}

impl Encoder {
    fn start(&mut self, out: &mut Vec<Bytes>) {
        if self.started {
            return;
        }
        self.started = true;
        let msg = json!({
            "id": self.id.clone().unwrap_or_else(|| "msg_nabucore".into()),
            "type": "message",
            "role": "assistant",
            "model": self.model.clone().unwrap_or_default(),
            "content": [],
            "stop_reason": null,
            "stop_sequence": null,
            "usage": usage_json(&self.usage),
        });
        out.push(sse(
            Some("message_start"),
            &json!({"type": "message_start", "message": msg}),
        ));
    }

    fn block_start(&mut self, index: usize, block: Value, out: &mut Vec<Bytes>) -> u64 {
        let i = self.next;
        self.next += 1;
        self.map.insert(index, i);
        out.push(sse(
            Some("content_block_start"),
            &json!({"type": "content_block_start", "index": i, "content_block": block}),
        ));
        i
    }
}

impl StreamEncoder for Encoder {
    fn encode(&mut self, ev: &StreamEvent) -> Vec<Bytes> {
        let mut out = Vec::new();
        match ev {
            // message_start waits for the first content so it can carry input usage.
            StreamEvent::Start { id, model } => {
                self.id = id.clone();
                self.model = model.clone();
                return out;
            }
            StreamEvent::Usage(u) if !self.started => {
                self.usage.merge(u);
                return out;
            }
            StreamEvent::Error(_) => {}
            _ => self.start(&mut out),
        }
        match ev {
            StreamEvent::Start { .. } => {}
            StreamEvent::BlockStart { index, block } => {
                let b = match block {
                    BlockStart::Text => json!({"type": "text", "text": ""}),
                    BlockStart::Thinking { redacted: true } => {
                        // Data arrives as a signature; redacted blocks are emitted whole on end.
                        self.skipped.push(*index);
                        return out;
                    }
                    BlockStart::Thinking { .. } => {
                        json!({"type": "thinking", "thinking": "", "signature": ""})
                    }
                    BlockStart::ToolCall { id, name } => {
                        json!({"type": "tool_use", "id": id, "name": name, "input": {}})
                    }
                    BlockStart::Opaque(o) => match o.for_format(F) {
                        Some(v) => v.clone(),
                        None => {
                            self.skipped.push(*index);
                            return out;
                        }
                    },
                };
                self.block_start(*index, b, &mut out);
            }
            StreamEvent::Delta { index, delta } => {
                if self.skipped.contains(index) {
                    if let Delta::Sig(o) = delta
                        && let Some(data) = o.for_format(F)
                    {
                        self.skipped.retain(|i| i != index);
                        let i = self.block_start(
                            *index,
                            json!({"type": "redacted_thinking", "data": data}),
                            &mut out,
                        );
                        self.map.insert(*index, i);
                    }
                    return out;
                }
                let Some(&i) = self.map.get(index) else {
                    return out;
                };
                let d = match delta {
                    Delta::Text(t) => json!({"type": "text_delta", "text": t}),
                    Delta::Thinking(t) => json!({"type": "thinking_delta", "thinking": t}),
                    Delta::ToolArgs(a) => json!({"type": "input_json_delta", "partial_json": a}),
                    Delta::Sig(o) => match o.for_format(F) {
                        Some(sig) if sig.is_string() => {
                            json!({"type": "signature_delta", "signature": sig})
                        }
                        _ => return out,
                    },
                };
                out.push(sse(
                    Some("content_block_delta"),
                    &json!({"type": "content_block_delta", "index": i, "delta": d}),
                ));
            }
            StreamEvent::BlockEnd { index } => {
                if let Some(i) = self.map.remove(index) {
                    out.push(sse(
                        Some("content_block_stop"),
                        &json!({"type": "content_block_stop", "index": i}),
                    ));
                }
                self.skipped.retain(|i| i != index);
            }
            StreamEvent::Usage(u) => self.usage.merge(u),
            StreamEvent::End { stop, usage } => {
                self.usage.merge(usage);
                let mut open: Vec<u64> = self.map.drain().map(|(_, i)| i).collect();
                open.sort();
                for i in open {
                    out.push(sse(
                        Some("content_block_stop"),
                        &json!({"type": "content_block_stop", "index": i}),
                    ));
                }
                let (reason, seq) = stop_json(stop);
                let reason = if reason.is_null() {
                    json!("end_turn")
                } else {
                    reason
                };
                out.push(sse(
                    Some("message_delta"),
                    &json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": reason, "stop_sequence": seq},
                        "usage": usage_json(&self.usage),
                    }),
                ));
                out.push(sse(Some("message_stop"), &json!({"type": "message_stop"})));
                self.ended = true;
            }
            StreamEvent::Error(e) => {
                out.push(sse(
                    Some("error"),
                    &json!({"type": "error", "error": {"type": error_type(e.kind), "message": e.message}}),
                ));
                self.ended = true;
            }
        }
        out
    }

    fn finish(&mut self) -> Vec<Bytes> {
        if self.ended {
            return Vec::new();
        }
        self.encode(&StreamEvent::End {
            stop: None,
            usage: Usage::default(),
        })
    }
}
