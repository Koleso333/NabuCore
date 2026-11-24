use std::collections::{HashMap, HashSet};

use bytes::Bytes;
use serde_json::{Map, Value, json};

use super::{
    CodecError, EncodeOptions, StreamDecoder, StreamEncoder, data_url, effort_for_budget, ext,
    merge, obj, parse_data_url, rest, s,
};
use crate::ir::*;
use crate::{ErrorKind, Format, ProviderError, SseEvent, Usage};

const F: Format = Format::OpenAiResponses;

const REQ_KNOWN: &[&str] = &[
    "model",
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "max_output_tokens",
    "temperature",
    "top_p",
    "reasoning",
    "text",
    "stream",
    // Transport settings the encoder derives from EncodeOptions.
    "store",
    "include",
];

fn breakpoint(p: &Value) -> Option<CacheMark> {
    p.get("prompt_cache_breakpoint")
        .map(|_| CacheMark { ttl: None })
}

fn decode_part(p: &Value) -> Result<Block, CodecError> {
    let m = obj(p)?;
    Ok(match m.get("type").and_then(Value::as_str).unwrap_or("") {
        "input_text" | "output_text" => Block::Text {
            text: s(p, "text").unwrap_or_default(),
            cache: breakpoint(p),
            sig: None,
            ext: ext(
                F,
                m,
                &[
                    "type",
                    "text",
                    "prompt_cache_breakpoint",
                    "annotations",
                    "logprobs",
                ],
            ),
        },
        "refusal" => Block::text(s(p, "refusal").unwrap_or_default()),
        "input_image" => {
            let media = if let Some(url) = p.get("image_url").and_then(Value::as_str) {
                match parse_data_url(url) {
                    Some((mime, data)) => Media {
                        mime: Some(mime),
                        source: MediaSource::Base64(data),
                        filename: None,
                    },
                    None => Media {
                        mime: Some("image/*".into()),
                        source: MediaSource::Url(url.into()),
                        filename: None,
                    },
                }
            } else if let Some(id) = s(p, "file_id") {
                Media {
                    mime: Some("image/*".into()),
                    source: MediaSource::FileId(id),
                    filename: None,
                }
            } else {
                return Ok(Block::Opaque(Opaque::new(F, p.clone())));
            };
            Block::Media {
                media,
                cache: breakpoint(p),
                ext: ext(
                    F,
                    m,
                    &["type", "image_url", "file_id", "prompt_cache_breakpoint"],
                ),
            }
        }
        "input_file" => {
            let filename = s(p, "filename");
            let source = if let Some((mime, data)) = p
                .get("file_data")
                .and_then(Value::as_str)
                .and_then(parse_data_url)
            {
                (Some(mime), MediaSource::Base64(data))
            } else if let Some(id) = s(p, "file_id") {
                (Some("application/pdf".into()), MediaSource::FileId(id))
            } else if let Some(u) = s(p, "file_url") {
                (Some("application/pdf".into()), MediaSource::Url(u))
            } else {
                return Ok(Block::Opaque(Opaque::new(F, p.clone())));
            };
            Block::Media {
                media: Media {
                    mime: source.0,
                    source: source.1,
                    filename,
                },
                cache: breakpoint(p),
                ext: Ext::default(),
            }
        }
        _ => Block::Opaque(Opaque::new(F, p.clone())),
    })
}

fn decode_content(v: Option<&Value>) -> Result<Vec<Block>, CodecError> {
    match v {
        Some(Value::String(t)) => Ok(vec![Block::text(t.clone())]),
        Some(Value::Array(parts)) => parts.iter().map(decode_part).collect(),
        _ => Ok(Vec::new()),
    }
}

fn reasoning_block(item: &Value) -> Block {
    let summary = |key: &str| -> Vec<String> {
        item.get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|p| p.get("text").and_then(Value::as_str).map(str::to_owned))
            .collect()
    };
    let mut parts = summary("summary");
    if parts.is_empty() {
        parts = summary("content");
    }
    Block::Thinking(Thinking {
        text: parts.join("\n\n"),
        sig: Some(Opaque::signature(F, item.clone())),
        redacted: false,
    })
}

fn call_block(item: &Value) -> Block {
    Block::ToolCall(ToolCall {
        id: s(item, "call_id").unwrap_or_default(),
        name: s(item, "name").unwrap_or_default(),
        args: s(item, "arguments").unwrap_or_default(),
        sig: item.get("id").map(|id| Opaque::new(F, json!({"id": id}))),
    })
}

fn push_assistant(msgs: &mut Vec<Message>, b: Block) {
    match msgs.last_mut() {
        Some(m) if m.role == Role::Assistant => m.content.push(b),
        _ => msgs.push(Message::new(Role::Assistant, vec![b])),
    }
}

pub fn decode_request(v: Value) -> Result<Request, CodecError> {
    let m = obj(&v)?;
    let mut req = Request {
        model: s(&v, "model").unwrap_or_default(),
        max_tokens: v.get("max_output_tokens").and_then(Value::as_u64),
        temperature: v.get("temperature").and_then(Value::as_f64),
        top_p: v.get("top_p").and_then(Value::as_f64),
        parallel_tool_calls: v.get("parallel_tool_calls").and_then(Value::as_bool),
        stream: v.get("stream").and_then(Value::as_bool).unwrap_or(false),
        ..Request::default()
    };
    let mut fields = rest(m, REQ_KNOWN);
    if let Some(i) = s(&v, "instructions") {
        req.system.push(Block::text(i));
    }
    let mut leading = true;
    match v.get("input") {
        Some(Value::String(t)) => req
            .messages
            .push(Message::new(Role::User, vec![Block::text(t.clone())])),
        Some(Value::Array(items)) => {
            for item in items {
                let im = obj(item)?;
                let ty = im.get("type").and_then(Value::as_str).unwrap_or("message");
                match ty {
                    "message" => {
                        let role = im.get("role").and_then(Value::as_str).unwrap_or("user");
                        let content = decode_content(im.get("content"))?;
                        let e = ext(F, im, &["type", "role", "content", "status"]);
                        match role {
                            "system" | "developer" if leading => req.system.extend(content),
                            "system" | "developer" => req.messages.push(Message {
                                role: Role::System,
                                content,
                                ext: e,
                            }),
                            "assistant" => {
                                leading = false;
                                match req.messages.last_mut() {
                                    Some(last) if last.role == Role::Assistant && e.is_empty() => {
                                        last.content.extend(content)
                                    }
                                    _ => req.messages.push(Message {
                                        role: Role::Assistant,
                                        content,
                                        ext: e,
                                    }),
                                }
                            }
                            _ => {
                                leading = false;
                                req.messages.push(Message {
                                    role: Role::User,
                                    content,
                                    ext: e,
                                });
                            }
                        }
                    }
                    "function_call" => {
                        leading = false;
                        push_assistant(&mut req.messages, call_block(item));
                    }
                    "reasoning" => {
                        leading = false;
                        push_assistant(&mut req.messages, reasoning_block(item));
                    }
                    "function_call_output" => {
                        leading = false;
                        let content = match item.get("output") {
                            Some(Value::String(t)) => vec![Block::text(t.clone())],
                            other => decode_content(other)?,
                        };
                        let r = Block::ToolResult(ToolResult {
                            call_id: s(item, "call_id").unwrap_or_default(),
                            name: s(item, "name"),
                            content,
                            is_error: false,
                            cache: None,
                        });
                        match req.messages.last_mut() {
                            Some(last)
                                if last.role == Role::User
                                    && last
                                        .content
                                        .iter()
                                        .all(|b| matches!(b, Block::ToolResult(_))) =>
                            {
                                last.content.push(r)
                            }
                            _ => req.messages.push(Message::new(Role::User, vec![r])),
                        }
                    }
                    _ => {
                        leading = false;
                        push_assistant(
                            &mut req.messages,
                            Block::Opaque(Opaque::new(F, item.clone())),
                        );
                    }
                }
            }
        }
        _ => {}
    }
    for t in v
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let tm = obj(t)?;
        req.tools.push(
            if tm.get("type").and_then(Value::as_str) == Some("function") {
                ToolDef::Function(Tool {
                    name: s(t, "name").unwrap_or_default(),
                    description: s(t, "description"),
                    schema: t
                        .get("parameters")
                        .filter(|p| !p.is_null())
                        .cloned()
                        .unwrap_or(json!({"type": "object", "properties": {}})),
                    strict: t.get("strict").and_then(Value::as_bool),
                    cache: None,
                    ext: ext(
                        F,
                        tm,
                        &["type", "name", "description", "parameters", "strict"],
                    ),
                })
            } else {
                ToolDef::Opaque(Opaque::new(F, t.clone()))
            },
        );
    }
    match v.get("tool_choice") {
        Some(Value::String(c)) => {
            req.tool_choice = match c.as_str() {
                "auto" => Some(ToolChoice::Auto),
                "none" => Some(ToolChoice::None),
                "required" => Some(ToolChoice::Required),
                _ => None,
            }
        }
        Some(c) if c.get("type").and_then(Value::as_str) == Some("function") => {
            req.tool_choice = Some(ToolChoice::Tool(s(c, "name").unwrap_or_default()))
        }
        Some(c) => {
            fields.insert("tool_choice".into(), c.clone());
        }
        None => {}
    }
    if let Some(r) = v.get("reasoning").and_then(Value::as_object) {
        let effort = r.get("effort").and_then(Value::as_str);
        req.reasoning = Some(ReasoningCfg {
            mode: (effort == Some("none")).then_some(ReasoningMode::Off),
            effort: effort.filter(|e| *e != "none").map(str::to_owned),
            budget_tokens: None,
            include: r.get("summary").filter(|s| !s.is_null()).map(|_| true),
        });
        let other = rest(r, &["effort"]);
        if !other.is_empty() {
            fields.insert("reasoning".into(), Value::Object(other));
        }
    }
    if let Some(t) = v.get("text").and_then(Value::as_object) {
        if let Some(f) = t.get("format") {
            match f.get("type").and_then(Value::as_str) {
                Some("json_object") => req.output = Some(OutputFormat::JsonObject),
                Some("json_schema") => {
                    req.output = Some(OutputFormat::JsonSchema {
                        name: s(f, "name"),
                        schema: f.get("schema").cloned().unwrap_or(json!({})),
                        strict: f.get("strict").and_then(Value::as_bool),
                    })
                }
                _ => {}
            }
        }
        let other = rest(t, &["format"]);
        if !other.is_empty() {
            fields.insert("text".into(), Value::Object(other));
        }
    }
    req.ext = Ext::new(F, fields);
    Ok(req)
}

struct Enc<'a> {
    opts: &'a EncodeOptions,
    losses: &'a mut Losses,
    marks_dropped: bool,
}

impl Enc<'_> {
    fn mark(&mut self, p: &mut Map<String, Value>, cache: &Option<CacheMark>) {
        if cache.is_none() {
            return;
        }
        if self.opts.openai_cache_breakpoints {
            p.insert(
                "prompt_cache_breakpoint".into(),
                json!({"mode": "explicit"}),
            );
        } else if !self.marks_dropped {
            self.marks_dropped = true;
            self.losses.dropped(
                "/cache",
                "cache marks dropped; enable openai_cache_breakpoints for models that support them",
            );
        }
    }

    fn part(&mut self, b: &Block, assistant: bool, path: &str) -> Option<Value> {
        let mut p = Map::new();
        match b {
            Block::Text {
                text, cache, ext, ..
            } => {
                p.insert(
                    "type".into(),
                    json!(if assistant {
                        "output_text"
                    } else {
                        "input_text"
                    }),
                );
                p.insert("text".into(), json!(text));
                if assistant {
                    p.insert("annotations".into(), json!([]));
                }
                self.mark(&mut p, cache);
                merge(&mut p, self.losses.ext(path, ext, F));
            }
            Block::Media { media, cache, ext } => {
                let mime = media.mime.as_deref().unwrap_or("");
                if media.is_image() || mime == "image/*" {
                    p.insert("type".into(), json!("input_image"));
                    match &media.source {
                        MediaSource::Base64(d) => {
                            p.insert("image_url".into(), json!(data_url(mime, d)))
                        }
                        MediaSource::Url(u) => p.insert("image_url".into(), json!(u)),
                        MediaSource::FileId(id) => p.insert("file_id".into(), json!(id)),
                    };
                    let detail = self
                        .losses
                        .ext(path, ext, F)
                        .and_then(|f| f.get("detail").cloned());
                    p.insert("detail".into(), detail.unwrap_or(json!("auto")));
                } else {
                    p.insert("type".into(), json!("input_file"));
                    match &media.source {
                        MediaSource::Base64(d) => {
                            p.insert("file_data".into(), json!(data_url(mime, d)))
                        }
                        MediaSource::Url(u) => p.insert("file_url".into(), json!(u)),
                        MediaSource::FileId(id) => p.insert("file_id".into(), json!(id)),
                    };
                    if let Some(n) = &media.filename {
                        p.insert("filename".into(), json!(n));
                    }
                }
                self.mark(&mut p, cache);
            }
            Block::Opaque(o) if o.format == F => return Some(o.data.clone()),
            other => {
                self.losses.dropped(
                    path,
                    format!("{} has no Responses content part", kind(other)),
                );
                return None;
            }
        }
        Some(Value::Object(p))
    }

    fn message(&mut self, role: &str, blocks: &[&Block], path: &str) -> Option<Value> {
        let assistant = role == "assistant";
        let marks = self.opts.openai_cache_breakpoints;
        let plain = blocks.iter().all(
            |b| matches!(b, Block::Text { cache, ext, .. } if ext.is_empty() && (cache.is_none() || !marks)),
        );
        if plain {
            for b in blocks {
                if let Block::Text { cache, .. } = b {
                    self.mark(&mut Map::new(), cache);
                }
            }
        }
        let content = if plain {
            json!(plain_text(
                &blocks.iter().map(|b| (*b).clone()).collect::<Vec<_>>()
            ))
        } else {
            let parts: Vec<Value> = blocks
                .iter()
                .enumerate()
                .filter_map(|(i, b)| self.part(b, assistant, &format!("{path}/{i}")))
                .collect();
            if parts.is_empty() {
                return None;
            }
            Value::Array(parts)
        };
        Some(json!({"type": "message", "role": role, "content": content}))
    }
}

fn kind(b: &Block) -> &'static str {
    match b {
        Block::Text { .. } => "text",
        Block::Media { .. } => "media",
        Block::ToolCall(_) => "tool call",
        Block::ToolResult(_) => "tool result",
        Block::Thinking(_) => "reasoning",
        Block::Opaque(_) => "provider block",
    }
}

pub fn encode_request(
    req: &Request,
    opts: &EncodeOptions,
    losses: &mut Losses,
) -> Result<Value, CodecError> {
    let mut enc = Enc {
        opts,
        losses,
        marks_dropped: false,
    };
    let mut o = Map::new();
    o.insert("model".into(), json!(req.model));
    let mut input = Vec::new();
    let system_plain = req.system.iter().all(|b| {
        matches!(b, Block::Text { cache, ext, .. } if ext.is_empty() && (cache.is_none() || !opts.openai_cache_breakpoints))
    });
    if !req.system.is_empty() {
        if system_plain {
            for b in &req.system {
                if let Block::Text { cache, .. } = b {
                    enc.mark(&mut Map::new(), cache);
                }
            }
            o.insert("instructions".into(), json!(plain_text(&req.system)));
        } else {
            let blocks: Vec<&Block> = req.system.iter().collect();
            input.extend(enc.message("system", &blocks, "/system"));
        }
    }
    let mut has_reasoning = false;
    for (i, m) in req.messages.iter().enumerate() {
        let path = format!("/messages/{i}");
        let mut pending: Vec<&Block> = Vec::new();
        let role = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        };
        let flush = |enc: &mut Enc, pending: &mut Vec<&Block>, input: &mut Vec<Value>| {
            if !pending.is_empty() {
                input.extend(enc.message(role, pending, &format!("{path}/content")));
                pending.clear();
            }
        };
        for (j, b) in m.content.iter().enumerate() {
            let bp = format!("{path}/content/{j}");
            match b {
                Block::ToolResult(r) => {
                    flush(&mut enc, &mut pending, &mut input);
                    let texts: Vec<&Block> = r
                        .content
                        .iter()
                        .filter(|b| matches!(b, Block::Text { .. } | Block::Media { .. }))
                        .collect();
                    let simple = texts
                        .iter()
                        .all(|b| matches!(b, Block::Text { cache: None, .. }));
                    let output = if simple {
                        json!(plain_text(&r.content))
                    } else {
                        Value::Array(
                            texts
                                .iter()
                                .enumerate()
                                .filter_map(|(k, b)| {
                                    enc.part(b, false, &format!("{bp}/content/{k}"))
                                })
                                .collect(),
                        )
                    };
                    if r.is_error {
                        enc.losses.approx(
                            format!("{bp}/is_error"),
                            "error flag has no Responses field",
                        );
                    }
                    input.push(json!({"type": "function_call_output", "call_id": r.call_id, "output": output}));
                }
                Block::ToolCall(c) => {
                    flush(&mut enc, &mut pending, &mut input);
                    let mut item = Map::new();
                    item.insert("type".into(), json!("function_call"));
                    if let Some(id) = c
                        .sig
                        .as_ref()
                        .and_then(|s| s.for_format(F))
                        .and_then(|s| s.get("id"))
                    {
                        item.insert("id".into(), id.clone());
                    }
                    item.insert("call_id".into(), json!(c.id));
                    item.insert("name".into(), json!(c.name));
                    item.insert(
                        "arguments".into(),
                        json!(if c.args.is_empty() {
                            "{}"
                        } else {
                            c.args.as_str()
                        }),
                    );
                    input.push(Value::Object(item));
                }
                Block::Thinking(t) => {
                    flush(&mut enc, &mut pending, &mut input);
                    match t.sig.as_ref().and_then(|s| s.for_format(F)) {
                        Some(item) => {
                            has_reasoning = true;
                            input.push(item.clone());
                        }
                        None => enc.losses.dropped(bp, "reasoning without a Responses item"),
                    }
                }
                Block::Opaque(op)
                    if op.format == F
                        && op.data.get("type").is_some_and(|t| t != "input_text")
                        && m.role == Role::Assistant =>
                {
                    flush(&mut enc, &mut pending, &mut input);
                    input.push(op.data.clone());
                }
                other => pending.push(other),
            }
        }
        flush(&mut enc, &mut pending, &mut input);
    }
    o.insert("input".into(), Value::Array(input));

    let mut tools = Vec::new();
    for (i, t) in req.tools.iter().enumerate() {
        match t {
            ToolDef::Function(t) => {
                let mut to = Map::new();
                to.insert("type".into(), json!("function"));
                to.insert("name".into(), json!(t.name));
                if let Some(d) = &t.description {
                    to.insert("description".into(), json!(d));
                }
                to.insert("parameters".into(), t.schema.clone());
                // Responses defaults to strict schemas, which arbitrary tool schemas fail.
                to.insert("strict".into(), json!(t.strict.unwrap_or(false)));
                merge(&mut to, enc.losses.ext(&format!("/tools/{i}"), &t.ext, F));
                tools.push(Value::Object(to));
            }
            ToolDef::Opaque(op) => match op.for_format(F) {
                Some(v) => tools.push(v.clone()),
                None => enc.losses.dropped(
                    format!("/tools/{i}"),
                    format!("{:?} hosted tool", op.format),
                ),
            },
        }
    }
    if !tools.is_empty() {
        o.insert("tools".into(), Value::Array(tools));
    }
    if let Some(c) = &req.tool_choice {
        o.insert(
            "tool_choice".into(),
            match c {
                ToolChoice::Auto => json!("auto"),
                ToolChoice::None => json!("none"),
                ToolChoice::Required => json!("required"),
                ToolChoice::Tool(n) => json!({"type": "function", "name": n}),
            },
        );
    }
    if let Some(p) = req.parallel_tool_calls {
        o.insert("parallel_tool_calls".into(), json!(p));
    }
    if let Some(m) = req.max_tokens {
        o.insert("max_output_tokens".into(), json!(m));
    }
    if let Some(t) = req.temperature {
        o.insert("temperature".into(), json!(t));
    }
    if let Some(t) = req.top_p {
        o.insert("top_p".into(), json!(t));
    }
    if req.top_k.is_some() {
        enc.losses.dropped("/top_k", "Responses has no top_k");
    }
    if !req.stop.is_empty() {
        enc.losses
            .dropped("/stop", "Responses has no stop sequences");
    }
    if req.seed.is_some() {
        enc.losses.dropped("/seed", "Responses has no seed");
    }
    if let Some(r) = &req.reasoning {
        let mut ro = Map::new();
        let effort = match (r.mode, &r.effort, r.budget_tokens) {
            (Some(ReasoningMode::Off), _, _) => Some("none".to_owned()),
            (_, Some(e), _) => Some(e.clone()),
            (_, None, Some(b)) => {
                enc.losses.approx(
                    "/reasoning/effort",
                    "thinking budget mapped to an effort level",
                );
                Some(effort_for_budget(b).to_owned())
            }
            _ => None,
        };
        if let Some(e) = effort {
            ro.insert("effort".into(), json!(e));
        }
        if r.include == Some(true) {
            ro.insert("summary".into(), json!("auto"));
        }
        if !ro.is_empty() {
            o.insert("reasoning".into(), Value::Object(ro));
        }
        has_reasoning = has_reasoning || r.mode != Some(ReasoningMode::Off);
    }
    match &req.output {
        Some(OutputFormat::JsonObject) => {
            o.insert("text".into(), json!({"format": {"type": "json_object"}}));
        }
        Some(OutputFormat::JsonSchema {
            name,
            schema,
            strict,
        }) => {
            let mut f = Map::new();
            f.insert("type".into(), json!("json_schema"));
            f.insert(
                "name".into(),
                json!(name.clone().unwrap_or_else(|| "output".into())),
            );
            f.insert("schema".into(), schema.clone());
            if let Some(st) = strict {
                f.insert("strict".into(), json!(st));
            }
            o.insert("text".into(), json!({"format": f}));
        }
        None => {}
    }
    o.insert("store".into(), json!(opts.responses_store));
    if !opts.responses_store && has_reasoning {
        o.insert("include".into(), json!(["reasoning.encrypted_content"]));
    }
    if req.stream {
        o.insert("stream".into(), json!(true));
    }
    merge(&mut o, enc.losses.ext("", &req.ext, F));
    Ok(Value::Object(o))
}

fn stop_of(resp: &Value, has_tools: bool) -> Option<StopReason> {
    Some(match resp.get("status").and_then(Value::as_str)? {
        "completed" if has_tools => StopReason::ToolUse,
        "completed" => StopReason::EndTurn,
        "incomplete" => match resp["incomplete_details"]["reason"].as_str() {
            Some("max_output_tokens") => StopReason::MaxTokens,
            Some("content_filter") => StopReason::ContentFilter,
            Some(other) => StopReason::Other(other.into()),
            None => StopReason::Other("incomplete".into()),
        },
        other => StopReason::Other(other.into()),
    })
}

fn decode_output(items: Option<&Value>) -> Result<Vec<Block>, CodecError> {
    let mut out = Vec::new();
    for item in items.and_then(Value::as_array).into_iter().flatten() {
        match item.get("type").and_then(Value::as_str).unwrap_or("") {
            "message" => out.extend(decode_content(item.get("content"))?),
            "function_call" => out.push(call_block(item)),
            "reasoning" => out.push(reasoning_block(item)),
            _ => out.push(Block::Opaque(Opaque::new(F, item.clone()))),
        }
    }
    Ok(out)
}

pub fn decode_response(v: Value) -> Result<Response, CodecError> {
    let content = decode_output(v.get("output"))?;
    let has_tools = content.iter().any(|b| matches!(b, Block::ToolCall(_)));
    Ok(Response {
        id: s(&v, "id"),
        model: s(&v, "model"),
        stop: stop_of(&v, has_tools),
        usage: v
            .get("usage")
            .filter(|u| u.is_object())
            .map(|u| Usage::from_provider(F, u))
            .unwrap_or_default(),
        content,
        ext: Ext::default(),
    })
}

pub(crate) fn usage_json(u: &Usage) -> Value {
    let input = u.input_total.unwrap_or(0);
    let output = u.output_total.unwrap_or(0);
    json!({
        "input_tokens": input,
        "input_tokens_details": {"cached_tokens": u.cache_read.unwrap_or(0)},
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": u.reasoning.unwrap_or(0)},
        "total_tokens": input + output,
    })
}

/// Output item for a finished block at `index`; `None` for blocks Responses cannot carry.
fn item_for(b: &Block, index: usize) -> Option<Value> {
    Some(match b {
        Block::Text { text, .. } => json!({
            "type": "message",
            "id": format!("msg_{index}"),
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }),
        Block::ToolCall(c) => json!({
            "type": "function_call",
            "id": format!("fc_{index}"),
            "call_id": c.id,
            "name": c.name,
            "arguments": c.args,
            "status": "completed",
        }),
        Block::Thinking(t) => match t.sig.as_ref().and_then(|s| s.for_format(F)) {
            Some(item) => {
                let mut item = item.clone();
                if item
                    .get("encrypted_content")
                    .and_then(Value::as_str)
                    .is_some_and(|s| s.starts_with("nabu-signature-v1:"))
                    && !t.text.is_empty()
                {
                    item["summary"] = json!([{"type": "summary_text", "text": t.text}]);
                    item["id"] = json!(format!("rs_{index}"));
                }
                item
            }
            None => {
                let summary: Vec<Value> = if t.text.is_empty() {
                    Vec::new()
                } else {
                    vec![json!({"type": "summary_text", "text": t.text})]
                };
                json!({"type": "reasoning", "id": format!("rs_{index}"), "summary": summary})
            }
        },
        Block::Opaque(o) => o.for_format(F)?.clone(),
        _ => return None,
    })
}

fn output_items(blocks: &[Block], losses: &mut Losses) -> Vec<Value> {
    let mut out = Vec::new();
    for (i, b) in blocks.iter().enumerate() {
        match item_for(b, out.len()) {
            Some(item) => out.push(item),
            None => losses.dropped(format!("/content/{i}"), "no Responses output item"),
        }
    }
    out
}

fn status_of(stop: &Option<StopReason>) -> (&'static str, Value) {
    match stop {
        Some(StopReason::MaxTokens) | Some(StopReason::ContextWindow) => {
            ("incomplete", json!({"reason": "max_output_tokens"}))
        }
        Some(StopReason::ContentFilter) => ("incomplete", json!({"reason": "content_filter"})),
        _ => ("completed", Value::Null),
    }
}

fn response_json(r: &Response, output: Vec<Value>, status: &str, incomplete: Value) -> Value {
    json!({
        "id": r.id.clone().unwrap_or_else(|| "resp_nabucore".into()),
        "object": "response",
        "created_at": 0,
        "status": status,
        "error": null,
        "incomplete_details": incomplete,
        "instructions": null,
        "metadata": {},
        "model": r.model.clone().unwrap_or_default(),
        "output": output,
        "parallel_tool_calls": true,
        "temperature": null,
        "tool_choice": "auto",
        "tools": [],
        "top_p": null,
        "usage": usage_json(&r.usage),
    })
}

pub fn encode_response(r: &Response, losses: &mut Losses) -> Value {
    let output = output_items(&r.content, losses);
    let (status, incomplete) = status_of(&r.stop);
    response_json(r, output, status, incomplete)
}

/// Responses SSE → IR events.
#[derive(Default)]
pub struct Decoder {
    started: bool,
    items: HashMap<u64, usize>,
    parts: HashMap<(u64, u64), usize>,
    pending: HashMap<u64, Value>,
    /// Function calls whose arguments already arrived.
    streamed: HashSet<u64>,
    next: usize,
    usage: Usage,
    saw_tool: bool,
    ended: bool,
}

impl Decoder {
    fn open(&mut self, block: BlockStart, out: &mut Vec<StreamEvent>) -> usize {
        let i = self.next;
        self.next += 1;
        out.push(StreamEvent::BlockStart { index: i, block });
        i
    }
}

impl StreamDecoder for Decoder {
    fn feed(&mut self, ev: &SseEvent) -> Result<Vec<StreamEvent>, CodecError> {
        let mut out = Vec::new();
        if !ev.has_data() {
            return Ok(out);
        }
        let v: Value = serde_json::from_str(&ev.data)?;
        let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
        let oi = v.get("output_index").and_then(Value::as_u64).unwrap_or(0);
        let ci = v.get("content_index").and_then(Value::as_u64).unwrap_or(0);
        match ty {
            "response.created" | "response.in_progress" | "response.queued" => {
                if !self.started {
                    self.started = true;
                    let r = &v["response"];
                    out.push(StreamEvent::Start {
                        id: s(r, "id"),
                        model: s(r, "model"),
                    });
                }
            }
            "response.output_item.added" => {
                let item = &v["item"];
                match item.get("type").and_then(Value::as_str).unwrap_or("") {
                    "message" => {}
                    "function_call" => {
                        self.saw_tool = true;
                        let i = self.open(
                            BlockStart::ToolCall {
                                id: s(item, "call_id").unwrap_or_default(),
                                name: s(item, "name").unwrap_or_default(),
                            },
                            &mut out,
                        );
                        if let Some(id) = item.get("id") {
                            out.push(StreamEvent::Delta {
                                index: i,
                                delta: Delta::Sig(Opaque::new(F, json!({"id": id}))),
                            });
                        }
                        self.items.insert(oi, i);
                    }
                    "reasoning" => {
                        let i = self.open(BlockStart::Thinking { redacted: false }, &mut out);
                        self.items.insert(oi, i);
                    }
                    _ => {
                        self.pending.insert(oi, item.clone());
                    }
                }
            }
            "response.content_part.added" => {
                let i = self.open(BlockStart::Text, &mut out);
                self.parts.insert((oi, ci), i);
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                if let Some(&i) = self.parts.get(&(oi, ci)) {
                    out.push(StreamEvent::Delta {
                        index: i,
                        delta: Delta::Text(s(&v, "delta").unwrap_or_default()),
                    });
                }
            }
            "response.content_part.done" => {
                if let Some(i) = self.parts.remove(&(oi, ci)) {
                    out.push(StreamEvent::BlockEnd { index: i });
                }
            }
            "response.reasoning_summary_part.added" => {
                let si = v.get("summary_index").and_then(Value::as_u64).unwrap_or(0);
                if si > 0
                    && let Some(&i) = self.items.get(&oi)
                {
                    out.push(StreamEvent::Delta {
                        index: i,
                        delta: Delta::Thinking("\n\n".into()),
                    });
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                if let Some(&i) = self.items.get(&oi) {
                    out.push(StreamEvent::Delta {
                        index: i,
                        delta: Delta::Thinking(s(&v, "delta").unwrap_or_default()),
                    });
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(&i) = self.items.get(&oi) {
                    self.streamed.insert(oi);
                    out.push(StreamEvent::Delta {
                        index: i,
                        delta: Delta::ToolArgs(s(&v, "delta").unwrap_or_default()),
                    });
                }
            }
            // Some backends (ChatGPT's Codex for parallel calls) send the arguments only here, with no deltas.
            "response.function_call_arguments.done" => {
                if let Some(&i) = self.items.get(&oi)
                    && !self.streamed.contains(&oi)
                {
                    self.streamed.insert(oi);
                    out.push(StreamEvent::Delta {
                        index: i,
                        delta: Delta::ToolArgs(s(&v, "arguments").unwrap_or_default()),
                    });
                }
            }
            "response.output_item.done" => {
                let item = &v["item"];
                let unstreamed = !self.streamed.remove(&oi);
                if let Some(i) = self.items.remove(&oi) {
                    if unstreamed
                        && item.get("type").and_then(Value::as_str) == Some("function_call")
                        && let Some(args) = s(item, "arguments").filter(|a| !a.is_empty())
                    {
                        out.push(StreamEvent::Delta {
                            index: i,
                            delta: Delta::ToolArgs(args),
                        });
                    }
                    if item.get("type").and_then(Value::as_str) == Some("reasoning") {
                        out.push(StreamEvent::Delta {
                            index: i,
                            delta: Delta::Sig(Opaque::signature(F, item.clone())),
                        });
                    }
                    out.push(StreamEvent::BlockEnd { index: i });
                } else if self.pending.remove(&oi).is_some() {
                    let i = self.open(BlockStart::Opaque(Opaque::new(F, item.clone())), &mut out);
                    out.push(StreamEvent::BlockEnd { index: i });
                }
            }
            "response.completed" | "response.incomplete" | "response.failed" => {
                let r = &v["response"];
                if let Some(u) = r.get("usage").filter(|u| u.is_object()) {
                    self.usage.merge(&Usage::from_provider(F, u));
                }
                if ty == "response.failed"
                    && let Some(e) = r.get("error").filter(|e| !e.is_null())
                {
                    out.push(StreamEvent::Error(ProviderError::from_value(
                        F,
                        None,
                        &json!({"error": e}),
                        Bytes::from(e.to_string()),
                    )));
                }
                self.ended = true;
                out.push(StreamEvent::End {
                    stop: stop_of(r, self.saw_tool),
                    usage: self.usage.clone(),
                });
            }
            // OpenAI sends the error fields flat; some proxies nest them under `error`.
            "error" => out.push(StreamEvent::Error(ProviderError::from_value(
                F,
                None,
                &json!({"error": v.get("error").filter(|e| e.is_object()).unwrap_or(&v)}),
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

struct OpenItem {
    output_index: usize,
    id: String,
    kind: BlockStart,
    text: String,
}

/// IR events → Responses SSE.
#[derive(Default)]
pub struct Encoder {
    seq: u64,
    started: bool,
    acc: Accumulator,
    open: HashMap<usize, OpenItem>,
    next_output: usize,
    skipped: Vec<usize>,
    ended: bool,
}

impl Encoder {
    fn emit(&mut self, out: &mut Vec<Bytes>, mut v: Value) {
        v["sequence_number"] = json!(self.seq);
        self.seq += 1;
        let ty = v["type"].as_str().unwrap_or("").to_owned();
        out.push(super::sse(Some(&ty), &v));
    }

    fn start(&mut self, out: &mut Vec<Bytes>) {
        if self.started {
            return;
        }
        self.started = true;
        let r = response_json(self.acc.response(), Vec::new(), "in_progress", Value::Null);
        self.emit(out, json!({"type": "response.created", "response": r}));
        let r = response_json(self.acc.response(), Vec::new(), "in_progress", Value::Null);
        self.emit(out, json!({"type": "response.in_progress", "response": r}));
    }

    fn close(&mut self, index: usize, out: &mut Vec<Bytes>) {
        let Some(it) = self.open.remove(&index) else {
            return;
        };
        let (oi, id) = (it.output_index, it.id.clone());
        match &it.kind {
            BlockStart::Text => {
                let part = json!({"type": "output_text", "text": it.text, "annotations": []});
                self.emit(out, json!({"type": "response.output_text.done", "item_id": id, "output_index": oi, "content_index": 0, "text": it.text, "logprobs": []}));
                self.emit(out, json!({"type": "response.content_part.done", "item_id": id, "output_index": oi, "content_index": 0, "part": part}));
            }
            BlockStart::ToolCall { name, .. } => {
                self.emit(out, json!({"type": "response.function_call_arguments.done", "item_id": id, "output_index": oi, "name": name, "arguments": it.text}));
            }
            BlockStart::Thinking { .. } => {
                if !it.text.is_empty() {
                    let part = json!({"type": "summary_text", "text": it.text});
                    self.emit(out, json!({"type": "response.reasoning_summary_text.done", "item_id": id, "output_index": oi, "summary_index": 0, "text": it.text}));
                    self.emit(out, json!({"type": "response.reasoning_summary_part.done", "item_id": id, "output_index": oi, "summary_index": 0, "part": part}));
                }
            }
            BlockStart::Opaque(_) => {}
        }
        let block = self.acc.response().content.last().cloned();
        if let Some(item) = block.as_ref().and_then(|b| item_for(b, oi)) {
            self.emit(
                out,
                json!({"type": "response.output_item.done", "output_index": oi, "item": item}),
            );
        }
    }
}

impl StreamEncoder for Encoder {
    fn encode(&mut self, ev: &StreamEvent) -> Vec<Bytes> {
        let mut out = Vec::new();
        if let StreamEvent::BlockStart {
            index,
            block: BlockStart::Opaque(o),
        } = ev
            && o.format != F
        {
            self.skipped.push(*index);
            return out;
        }
        if let StreamEvent::Delta { index, .. } | StreamEvent::BlockEnd { index } = ev
            && self.skipped.contains(index)
        {
            if matches!(ev, StreamEvent::BlockEnd { .. }) {
                self.skipped.retain(|i| i != index);
            }
            return out;
        }
        self.acc.push(ev);
        match ev {
            StreamEvent::Start { .. } | StreamEvent::Usage(_) => {}
            StreamEvent::BlockStart { index, block } => {
                self.start(&mut out);
                let oi = self.next_output;
                self.next_output += 1;
                let (id, item) = match block {
                    BlockStart::Text => {
                        let id = format!("msg_{oi}");
                        (
                            id.clone(),
                            json!({"type": "message", "id": id, "status": "in_progress", "role": "assistant", "content": []}),
                        )
                    }
                    BlockStart::ToolCall { id: call_id, name } => {
                        let id = format!("fc_{oi}");
                        (
                            id.clone(),
                            json!({"type": "function_call", "id": id, "call_id": call_id, "name": name, "arguments": "", "status": "in_progress"}),
                        )
                    }
                    BlockStart::Thinking { .. } => {
                        let id = format!("rs_{oi}");
                        (
                            id.clone(),
                            json!({"type": "reasoning", "id": id, "summary": []}),
                        )
                    }
                    BlockStart::Opaque(o) => (String::new(), o.data.clone()),
                };
                self.emit(
                    &mut out,
                    json!({"type": "response.output_item.added", "output_index": oi, "item": item}),
                );
                if matches!(block, BlockStart::Text) {
                    self.emit(&mut out, json!({"type": "response.content_part.added", "item_id": id, "output_index": oi, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}));
                }
                self.open.insert(
                    *index,
                    OpenItem {
                        output_index: oi,
                        id,
                        kind: block.clone(),
                        text: String::new(),
                    },
                );
            }
            StreamEvent::Delta { index, delta } => {
                let Some(it) = self.open.get_mut(index) else {
                    return out;
                };
                let (oi, id) = (it.output_index, it.id.clone());
                let first = it.text.is_empty();
                let ev = match delta {
                    Delta::Text(t) => {
                        it.text.push_str(t);
                        json!({"type": "response.output_text.delta", "item_id": id, "output_index": oi, "content_index": 0, "delta": t, "logprobs": []})
                    }
                    Delta::ToolArgs(a) => {
                        it.text.push_str(a);
                        json!({"type": "response.function_call_arguments.delta", "item_id": id, "output_index": oi, "delta": a})
                    }
                    Delta::Thinking(t) => {
                        it.text.push_str(t);
                        if first {
                            self.emit(&mut out, json!({"type": "response.reasoning_summary_part.added", "item_id": id, "output_index": oi, "summary_index": 0, "part": {"type": "summary_text", "text": ""}}));
                        }
                        json!({"type": "response.reasoning_summary_text.delta", "item_id": id, "output_index": oi, "summary_index": 0, "delta": t})
                    }
                    Delta::Sig(_) => return out,
                };
                self.emit(&mut out, ev);
            }
            StreamEvent::BlockEnd { index } => self.close(*index, &mut out),
            StreamEvent::End { .. } => {
                self.start(&mut out);
                let mut open: Vec<usize> = self.open.keys().copied().collect();
                open.sort();
                for i in open {
                    self.close(i, &mut out);
                }
                let r = self.acc.response().clone();
                let mut losses = Losses::default();
                let output = output_items(&r.content, &mut losses);
                let (status, incomplete) = status_of(&r.stop);
                let ty = if status == "completed" {
                    "response.completed"
                } else {
                    "response.incomplete"
                };
                let resp = response_json(&r, output, status, incomplete);
                self.emit(&mut out, json!({"type": ty, "response": resp}));
                self.ended = true;
            }
            StreamEvent::Error(e) => {
                let code = match e.kind {
                    ErrorKind::RateLimited => "rate_limit_exceeded",
                    ErrorKind::InvalidRequest => "invalid_request_error",
                    _ => "server_error",
                };
                self.emit(&mut out, json!({"type": "error", "code": e.code.clone().unwrap_or_else(|| code.into()), "message": e.message, "param": null}));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::SseParser;

    fn stream_error(event: Value) -> ProviderError {
        let mut parser = SseParser::new();
        let sse = parser.feed(format!("data: {event}

").as_bytes());
        match Decoder::default().feed(&sse[0]).unwrap().remove(0) {
            StreamEvent::Error(e) => e,
            other => panic!("not an error: {other:?}"),
        }
    }

    #[test]
    fn stream_errors_flat_and_nested() {
        let flat = stream_error(json!({"type": "error", "code": "rate_limit_exceeded", "message": "slow down"}));
        let nested = stream_error(json!({"type": "error", "error": {"type": "too_many_requests", "code": "rate_limit_exceeded", "message": "slow down"}}));
        for e in [flat, nested] {
            assert_eq!(e.kind, ErrorKind::RateLimited);
            assert_eq!(e.message, "slow down");
        }
    }

    fn args_of(events: &[Value]) -> Vec<String> {
        let mut decoder = Decoder::default();
        let mut parser = SseParser::new();
        let mut args: HashMap<usize, String> = HashMap::new();
        for event in events {
            for sse in parser.feed(format!("data: {event}\n\n").as_bytes()) {
                for out in decoder.feed(&sse).unwrap() {
                    if let StreamEvent::Delta { index, delta: Delta::ToolArgs(a) } = out {
                        args.entry(index).or_default().push_str(&a);
                    }
                }
            }
        }
        let mut list: Vec<(usize, String)> = args.into_iter().collect();
        list.sort();
        list.into_iter().map(|(_, a)| a).collect()
    }

    #[test]
    fn arguments_without_deltas_come_from_done() {
        let call = |oi: u64, name: &str| json!({"type": "response.output_item.added", "output_index": oi, "item": {"type": "function_call", "call_id": format!("c{oi}"), "name": name, "arguments": ""}});
        let parallel = args_of(&[
            call(0, "glob"),
            json!({"type": "response.function_call_arguments.done", "output_index": 0, "arguments": "{\"pattern\":\"*.toml\"}"}),
            json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "function_call", "arguments": "{\"pattern\":\"*.toml\"}"}}),
            call(1, "read"),
            json!({"type": "response.output_item.done", "output_index": 1, "item": {"type": "function_call", "arguments": "{\"path\":\"a\"}"}}),
        ]);
        assert_eq!(parallel, ["{\"pattern\":\"*.toml\"}", "{\"path\":\"a\"}"]);
        let streamed = args_of(&[
            call(0, "glob"),
            json!({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{\"pattern\""}),
            json!({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": ":\"x\"}"}),
            json!({"type": "response.function_call_arguments.done", "output_index": 0, "arguments": "{\"pattern\":\"x\"}"}),
            json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "function_call", "arguments": "{\"pattern\":\"x\"}"}}),
        ]);
        assert_eq!(streamed, ["{\"pattern\":\"x\"}"]);
    }
}
