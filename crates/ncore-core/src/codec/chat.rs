use std::collections::{BTreeMap, HashMap};

use bytes::Bytes;
use serde_json::{Map, Value, json};

use super::{
    CodecError, EncodeOptions, MaxTokensField, StreamDecoder, StreamEncoder, attachments_label,
    data_url, effort_for_budget, ext, merge, obj, parse_data_url, rest, s, synth_call_id,
};
use crate::ir::*;
use crate::{Format, ProviderError, SseEvent, Usage};

const F: Format = Format::OpenAiChat;

const REQ_KNOWN: &[&str] = &[
    "model",
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "seed",
    "reasoning_effort",
    "response_format",
    "stream",
    "stream_options",
];

// OpenAI-compatible servers put reasoning text under one of these names.
const REASONING_FIELDS: [&str; 2] = ["reasoning_content", "reasoning"];

fn breakpoint(part: &Value) -> Option<CacheMark> {
    part.get("prompt_cache_breakpoint")
        .map(|_| CacheMark { ttl: None })
}

fn decode_part(p: &Value) -> Result<Block, CodecError> {
    let m = obj(p)?;
    let ty = m.get("type").and_then(Value::as_str).unwrap_or("");
    Ok(match ty {
        "text" => Block::Text {
            text: s(p, "text").unwrap_or_default(),
            cache: breakpoint(p),
            sig: None,
            ext: ext(F, m, &["type", "text", "prompt_cache_breakpoint"]),
        },
        "refusal" => Block::text(s(p, "refusal").unwrap_or_default()),
        "image_url" => {
            let url = p["image_url"]["url"].as_str().unwrap_or_default();
            let media = match parse_data_url(url) {
                Some((mime, data)) => Media {
                    mime: Some(mime),
                    source: MediaSource::Base64(data),
                    filename: None,
                },
                None => Media {
                    mime: Some("image/*".into()),
                    source: MediaSource::Url(url.to_owned()),
                    filename: None,
                },
            };
            let mut e = rest(m, &["type", "image_url", "prompt_cache_breakpoint"]);
            if let Some(d) = p["image_url"].get("detail") {
                e.insert("detail".into(), d.clone());
            }
            Block::Media {
                media,
                cache: breakpoint(p),
                ext: Ext::new(F, e),
            }
        }
        "input_audio" => Block::Media {
            media: Media {
                mime: Some(format!(
                    "audio/{}",
                    p["input_audio"]["format"].as_str().unwrap_or("wav")
                )),
                source: MediaSource::Base64(
                    p["input_audio"]["data"].as_str().unwrap_or_default().into(),
                ),
                filename: None,
            },
            cache: None,
            ext: Ext::default(),
        },
        "file" => {
            let f = &p["file"];
            let filename = s(f, "filename");
            let media = if let Some((mime, data)) = f
                .get("file_data")
                .and_then(Value::as_str)
                .and_then(parse_data_url)
            {
                Media {
                    mime: Some(mime),
                    source: MediaSource::Base64(data),
                    filename,
                }
            } else if let Some(id) = s(f, "file_id") {
                Media {
                    mime: Some("application/pdf".into()),
                    source: MediaSource::FileId(id),
                    filename,
                }
            } else {
                return Ok(Block::Opaque(Opaque::new(F, p.clone())));
            };
            Block::Media {
                media,
                cache: breakpoint(p),
                ext: Ext::default(),
            }
        }
        _ => Block::Opaque(Opaque::new(F, p.clone())),
    })
}

fn decode_content(v: Option<&Value>) -> Result<Vec<Block>, CodecError> {
    match v {
        Some(Value::String(t)) => Ok(if t.is_empty() {
            Vec::new()
        } else {
            vec![Block::text(t.clone())]
        }),
        Some(Value::Array(parts)) => parts.iter().map(decode_part).collect(),
        _ => Ok(Vec::new()),
    }
}

fn decode_tool_calls(v: Option<&Value>, msg: usize, offset: usize) -> Vec<Block> {
    v.and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(i, tc)| {
            Block::ToolCall(ToolCall {
                id: s(tc, "id").unwrap_or_else(|| synth_call_id(msg, offset + i)),
                name: tc["function"]["name"].as_str().unwrap_or_default().into(),
                args: tc["function"]["arguments"]
                    .as_str()
                    .unwrap_or_default()
                    .into(),
                sig: None,
            })
        })
        .collect()
}

fn reasoning_text(m: &Value) -> Option<String> {
    REASONING_FIELDS
        .iter()
        .find_map(|f| m.get(*f).and_then(Value::as_str))
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
}

fn details_text(details: &Value) -> String {
    details
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|detail| match detail.get("type").and_then(Value::as_str) {
            Some("reasoning.text") => detail.get("text").and_then(Value::as_str),
            Some("reasoning.summary") => detail.get("summary").and_then(Value::as_str),
            _ => None,
        })
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

// Message-level details live in the signature; derived display text is not replayed separately.
fn details_block(details: Value, plain_reasoning: bool) -> Thinking {
    Thinking {
        text: if plain_reasoning {
            String::new()
        } else {
            details_text(&details)
        },
        sig: Some(Opaque::new(F, json!({"reasoning_details": details}))),
        redacted: false,
    }
}

fn decode_reasoning(msg: &Value) -> Vec<Block> {
    let mut blocks = Vec::new();
    let plain = reasoning_text(msg);
    if let Some(text) = &plain {
        blocks.push(Block::Thinking(Thinking {
            text: text.clone(),
            sig: None,
            redacted: false,
        }));
    }
    if let Some(details) = msg.get("reasoning_details") {
        blocks.push(Block::Thinking(details_block(
            details.clone(),
            plain.is_some(),
        )));
    }
    blocks
}

fn reasoning_details(sig: &Opaque) -> Option<&Value> {
    sig.for_format(F)?.get("reasoning_details")
}

fn collect_details(dst: &mut Option<Value>, details: &Value) {
    match (dst.as_mut(), details) {
        (Some(Value::Array(all)), Value::Array(parts)) => all.extend(parts.iter().cloned()),
        _ => *dst = Some(details.clone()),
    }
}

fn detail_type(detail: &Value) -> Option<&str> {
    detail.get("type").and_then(Value::as_str).or_else(|| {
        [
            ("text", "reasoning.text"),
            ("summary", "reasoning.summary"),
            ("data", "reasoning.encrypted"),
        ]
        .into_iter()
        .find_map(|(field, kind)| detail.get(field).map(|_| kind))
    })
}

fn detail_match(left: &Value, right: &Value, position: bool) -> Option<u8> {
    if !left.is_object() || !right.is_object() {
        return None;
    }
    if let (Some(a), Some(b)) = (detail_type(left), detail_type(right))
        && a != b
    {
        return None;
    }
    let field =
        |detail: &Value, key: &str| detail.get(key).filter(|value| !value.is_null()).cloned();
    for key in ["id", "format"] {
        if let (Some(a), Some(b)) = (field(left, key), field(right, key))
            && a != b
        {
            return None;
        }
    }
    match (field(left, "index"), field(right, "index")) {
        (Some(a), Some(b)) => (a == b).then_some(3),
        _ => match (field(left, "id"), field(right, "id")) {
            (Some(a), Some(b)) => (a == b).then_some(2),
            _ => position.then_some(1),
        },
    }
}

fn merge_detail(dst: &mut Value, fragment: &Value) {
    let (Some(dst), Some(fragment)) = (dst.as_object_mut(), fragment.as_object()) else {
        return;
    };
    for (key, value) in fragment {
        if value.is_null() && dst.contains_key(key) {
            continue;
        }
        if matches!(key.as_str(), "text" | "summary" | "data")
            && let Some(text) = value.as_str()
            && let Some(Value::String(current)) = dst.get_mut(key)
        {
            current.push_str(text);
        } else {
            dst.insert(key.clone(), value.clone());
        }
    }
}

fn append_details(dst: &mut Option<Value>, fragment: &Value) {
    if fragment.is_null() && dst.is_some() {
        return;
    }
    let (Some(Value::Array(all)), Value::Array(parts)) = (dst.as_mut(), fragment) else {
        *dst = Some(fragment.clone());
        return;
    };
    for (position, part) in parts.iter().enumerate() {
        let found = all
            .iter()
            .enumerate()
            .filter_map(|(i, detail)| {
                detail_match(detail, part, i == position).map(|rank| (i, rank))
            })
            .max_by_key(|(_, rank)| *rank)
            .map(|(i, _)| i);
        match found {
            Some(i) => merge_detail(&mut all[i], part),
            None => all.push(part.clone()),
        }
    }
}

pub fn decode_request(v: Value) -> Result<Request, CodecError> {
    let m = obj(&v)?;
    let mut req = Request {
        model: s(&v, "model").unwrap_or_default(),
        max_tokens: v
            .get("max_completion_tokens")
            .or(v.get("max_tokens"))
            .and_then(Value::as_u64),
        temperature: v.get("temperature").and_then(Value::as_f64),
        top_p: v.get("top_p").and_then(Value::as_f64),
        seed: v.get("seed").and_then(Value::as_i64),
        parallel_tool_calls: v.get("parallel_tool_calls").and_then(Value::as_bool),
        stream: v.get("stream").and_then(Value::as_bool).unwrap_or(false),
        ..Request::default()
    };
    req.stop = match v.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|s| s.as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    };
    let mut leading = true;
    for (i, msg) in v
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let mm = obj(msg)?;
        let role = mm.get("role").and_then(Value::as_str).unwrap_or("user");
        if matches!(role, "system" | "developer") {
            let content = decode_content(mm.get("content"))?;
            if leading {
                req.system.extend(content);
            } else {
                req.messages.push(Message::new(Role::System, content));
            }
            continue;
        }
        leading = false;
        match role {
            "tool" => {
                let result = Block::ToolResult(ToolResult {
                    call_id: s(msg, "tool_call_id").unwrap_or_default(),
                    name: None,
                    content: decode_content(mm.get("content"))?,
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
                        last.content.push(result)
                    }
                    _ => req.messages.push(Message::new(Role::User, vec![result])),
                }
            }
            "assistant" => {
                let mut content = decode_reasoning(msg);
                content.extend(decode_content(mm.get("content"))?);
                if let Some(r) = msg.get("refusal").and_then(Value::as_str) {
                    content.push(Block::text(r));
                }
                let n = content.len();
                content.extend(decode_tool_calls(mm.get("tool_calls"), i, n));
                let mut known = vec![
                    "role",
                    "content",
                    "tool_calls",
                    "refusal",
                    "reasoning_details",
                ];
                known.extend(REASONING_FIELDS);
                req.messages.push(Message {
                    role: Role::Assistant,
                    content,
                    ext: ext(F, mm, &known),
                });
            }
            _ => req.messages.push(Message {
                role: Role::User,
                content: decode_content(mm.get("content"))?,
                ext: ext(F, mm, &["role", "content"]),
            }),
        }
    }
    for t in v
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        req.tools.push(
            if t.get("type").and_then(Value::as_str) == Some("function") {
                let f = &t["function"];
                ToolDef::Function(Tool {
                    name: s(f, "name").unwrap_or_default(),
                    description: s(f, "description"),
                    schema: f
                        .get("parameters")
                        .cloned()
                        .unwrap_or(json!({"type": "object", "properties": {}})),
                    strict: f.get("strict").and_then(Value::as_bool),
                    cache: None,
                    ext: ext(F, obj(f)?, &["name", "description", "parameters", "strict"]),
                })
            } else {
                ToolDef::Opaque(Opaque::new(F, t.clone()))
            },
        );
    }
    let mut fields = rest(m, REQ_KNOWN);
    match v.get("tool_choice") {
        Some(Value::String(c)) => {
            req.tool_choice = match c.as_str() {
                "auto" => Some(ToolChoice::Auto),
                "none" => Some(ToolChoice::None),
                "required" => Some(ToolChoice::Required),
                _ => None,
            }
        }
        Some(c @ Value::Object(_)) => match c["function"]["name"].as_str() {
            Some(n) => req.tool_choice = Some(ToolChoice::Tool(n.into())),
            None => {
                fields.insert("tool_choice".into(), c.clone());
            }
        },
        _ => {}
    }
    if let Some(e) = v.get("reasoning_effort").and_then(Value::as_str) {
        req.reasoning = Some(ReasoningCfg {
            mode: (e == "none").then_some(ReasoningMode::Off),
            effort: (e != "none").then(|| e.to_owned()),
            ..ReasoningCfg::default()
        });
    }
    if let Some(rf) = v.get("response_format") {
        match rf.get("type").and_then(Value::as_str) {
            Some("json_object") => req.output = Some(OutputFormat::JsonObject),
            Some("json_schema") => {
                let js = &rf["json_schema"];
                req.output = Some(OutputFormat::JsonSchema {
                    name: s(js, "name"),
                    schema: js.get("schema").cloned().unwrap_or(json!({})),
                    strict: js.get("strict").and_then(Value::as_bool),
                });
            }
            _ => {}
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
    fn mark(&mut self, part: &mut Map<String, Value>, cache: &Option<CacheMark>) {
        if cache.is_none() {
            return;
        }
        if self.opts.openai_cache_breakpoints {
            part.insert(
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

    fn part(&mut self, b: &Block, path: &str) -> Option<Value> {
        match b {
            Block::Text {
                text, cache, ext, ..
            } => {
                let mut p = Map::new();
                p.insert("type".into(), json!("text"));
                p.insert("text".into(), json!(text));
                self.mark(&mut p, cache);
                merge(&mut p, self.losses.ext(path, ext, F));
                Some(Value::Object(p))
            }
            Block::Media { media, cache, ext } => {
                let mime = media.mime.as_deref().unwrap_or("");
                let mut p = Map::new();
                if media.is_image() || mime == "image/*" {
                    let url = match &media.source {
                        MediaSource::Base64(d) => data_url(mime, d),
                        MediaSource::Url(u) => u.clone(),
                        MediaSource::FileId(_) => {
                            self.losses
                                .dropped(path, "Chat cannot reference images by file id");
                            return None;
                        }
                    };
                    let mut iu = Map::new();
                    iu.insert("url".into(), json!(url));
                    if let Some(f) = self.losses.ext(path, ext, F)
                        && let Some(d) = f.get("detail")
                    {
                        iu.insert("detail".into(), d.clone());
                    }
                    p.insert("type".into(), json!("image_url"));
                    p.insert("image_url".into(), Value::Object(iu));
                } else if let (Some(fmt), MediaSource::Base64(d)) =
                    (mime.strip_prefix("audio/"), &media.source)
                {
                    let fmt = match fmt {
                        "mpeg" | "mp3" => "mp3",
                        "x-wav" | "wave" => "wav",
                        other => other,
                    };
                    p.insert("type".into(), json!("input_audio"));
                    p.insert("input_audio".into(), json!({"data": d, "format": fmt}));
                } else {
                    let mut f = Map::new();
                    match &media.source {
                        MediaSource::Base64(d) => {
                            f.insert("file_data".into(), json!(data_url(mime, d)));
                        }
                        MediaSource::FileId(id) => {
                            f.insert("file_id".into(), json!(id));
                        }
                        MediaSource::Url(_) => {
                            self.losses
                                .dropped(path, "Chat cannot reference files by URL");
                            return None;
                        }
                    }
                    if let Some(n) = &media.filename {
                        f.insert("filename".into(), json!(n));
                    }
                    p.insert("type".into(), json!("file"));
                    p.insert("file".into(), Value::Object(f));
                }
                self.mark(&mut p, cache);
                Some(Value::Object(p))
            }
            Block::Opaque(o) => match o.for_format(F) {
                Some(v) => Some(v.clone()),
                None => {
                    self.losses
                        .dropped(path, format!("{:?} block has no Chat equivalent", o.format));
                    None
                }
            },
            _ => None,
        }
    }

    // String when content is plain text without marks, parts otherwise.
    fn content(&mut self, blocks: &[&Block], path: &str) -> Value {
        let marks = self.opts.openai_cache_breakpoints;
        let plain = blocks.iter().all(|b| {
            matches!(b, Block::Text { cache, ext, .. } if ext.is_empty() && (cache.is_none() || !marks))
        });
        if plain {
            for b in blocks {
                if let Block::Text { cache, .. } = b {
                    let mut sink = Map::new();
                    self.mark(&mut sink, cache);
                }
            }
            let texts: Vec<&Block> = blocks.to_vec();
            return json!(
                texts
                    .iter()
                    .filter_map(|b| match b {
                        Block::Text { text, .. } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("")
            );
        }
        Value::Array(
            blocks
                .iter()
                .enumerate()
                .filter_map(|(i, b)| self.part(b, &format!("{path}/{i}")))
                .collect(),
        )
    }

    // Media cannot go in a tool message; it is returned to be sent in a user message.
    fn tool_result<'r>(&mut self, r: &'r ToolResult, path: &str) -> (Value, Vec<&'r Block>) {
        let mut parts = Vec::new();
        let mut media = Vec::new();
        for (i, b) in r.content.iter().enumerate() {
            match b {
                Block::Text { .. } => parts.push(b),
                Block::Media { .. } => {
                    self.losses.approx(
                        format!("{path}/content/{i}"),
                        "Chat tool messages carry text only; media moved to a user message",
                    );
                    media.push(b);
                }
                _ => self.losses.dropped(
                    format!("{path}/content/{i}"),
                    "Chat tool messages carry text only",
                ),
            }
        }
        let mut content = self.content(&parts, &format!("{path}/content"));
        if r.is_error {
            self.losses
                .approx(format!("{path}/is_error"), "error flag has no Chat field");
        }
        if r.cache.is_some()
            && let Value::String(t) = &content
        {
            let mut p = Map::new();
            p.insert("type".into(), json!("text"));
            p.insert("text".into(), json!(t));
            self.mark(&mut p, &r.cache);
            content = json!([p]);
        }
        let msg = json!({"role": "tool", "tool_call_id": r.call_id, "content": content});
        (msg, media)
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
    let mut messages = Vec::new();
    if !req.system.is_empty() {
        let sys: Vec<&Block> = req.system.iter().collect();
        let content = enc.content(&sys, "/system");
        messages.push(json!({"role": "system", "content": content}));
    }
    for (i, m) in req.messages.iter().enumerate() {
        let path = format!("/messages/{i}");
        match m.role {
            Role::System | Role::User => {
                let mut rest_blocks = Vec::new();
                let mut hoisted = Vec::new();
                for (j, b) in m.content.iter().enumerate() {
                    match b {
                        Block::ToolResult(r) => {
                            let (t, media) = enc.tool_result(r, &format!("{path}/content/{j}"));
                            messages.push(t);
                            if !media.is_empty() {
                                hoisted.push(Block::text(attachments_label(&r.call_id)));
                                hoisted.extend(media.into_iter().cloned());
                            }
                        }
                        Block::ToolCall(_) | Block::Thinking(_) => enc.losses.dropped(
                            format!("{path}/content/{j}"),
                            "unexpected block in a user message",
                        ),
                        other => rest_blocks.push(other),
                    }
                }
                if !hoisted.is_empty() {
                    rest_blocks.splice(0..0, hoisted.iter());
                }
                if !rest_blocks.is_empty() {
                    let role = if m.role == Role::System {
                        "system"
                    } else {
                        "user"
                    };
                    let content = enc.content(&rest_blocks, &format!("{path}/content"));
                    let mut mo = Map::new();
                    mo.insert("role".into(), json!(role));
                    mo.insert("content".into(), content);
                    merge(&mut mo, enc.losses.ext(&path, &m.ext, F));
                    messages.push(Value::Object(mo));
                }
            }
            Role::Assistant => {
                let mut texts = Vec::new();
                let mut calls = Vec::new();
                let mut reasoning = String::new();
                let mut details = None;
                for (j, b) in m.content.iter().enumerate() {
                    let bp = format!("{path}/content/{j}");
                    match b {
                        Block::Text { .. } => texts.push(b),
                        Block::ToolCall(c) => calls.push(json!({
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": if c.args.is_empty() { "{}" } else { c.args.as_str() }},
                        })),
                        Block::Thinking(t) => {
                            if let Some(raw) = t.sig.as_ref().and_then(reasoning_details) {
                                collect_details(&mut details, raw);
                            } else {
                                if !t.text.is_empty() && opts.chat_reasoning_field.is_some() {
                                    reasoning.push_str(&t.text);
                                    if t.sig.is_some() {
                                        enc.losses.dropped(format!("{bp}/signature"), "no Chat reasoning details");
                                    }
                                } else {
                                    enc.losses.dropped(&bp, "reasoning has no Chat field");
                                }
                            }
                        }
                        Block::Opaque(o) if o.format == F => texts.push(b),
                        _ => enc
                            .losses
                            .dropped(bp, "only text and tool calls fit an assistant message"),
                    }
                }
                let mut mo = Map::new();
                mo.insert("role".into(), json!("assistant"));
                let content = if texts.is_empty() {
                    Value::Null
                } else {
                    enc.content(&texts, &format!("{path}/content"))
                };
                mo.insert("content".into(), content);
                if let (Some(field), false) = (&opts.chat_reasoning_field, reasoning.is_empty()) {
                    mo.insert(field.clone(), json!(reasoning));
                }
                if let Some(details) = details {
                    mo.insert("reasoning_details".into(), details);
                }
                if !calls.is_empty() {
                    mo.insert("tool_calls".into(), Value::Array(calls));
                }
                merge(&mut mo, enc.losses.ext(&path, &m.ext, F));
                messages.push(Value::Object(mo));
            }
        }
    }

    let mut o = Map::new();
    o.insert("model".into(), json!(req.model));
    o.insert("messages".into(), Value::Array(messages));
    let mut tools = Vec::new();
    for (i, t) in req.tools.iter().enumerate() {
        match t {
            ToolDef::Function(t) => {
                let mut f = Map::new();
                f.insert("name".into(), json!(t.name));
                if let Some(d) = &t.description {
                    f.insert("description".into(), json!(d));
                }
                f.insert("parameters".into(), t.schema.clone());
                if let Some(st) = t.strict {
                    f.insert("strict".into(), json!(st));
                }
                merge(&mut f, enc.losses.ext(&format!("/tools/{i}"), &t.ext, F));
                tools.push(json!({"type": "function", "function": f}));
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
                ToolChoice::Tool(n) => json!({"type": "function", "function": {"name": n}}),
            },
        );
    }
    if let Some(p) = req.parallel_tool_calls {
        o.insert("parallel_tool_calls".into(), json!(p));
    }
    if let Some(m) = req.max_tokens {
        let key = match opts.chat_max_tokens_field {
            MaxTokensField::MaxCompletionTokens => "max_completion_tokens",
            MaxTokensField::MaxTokens => "max_tokens",
        };
        o.insert(key.into(), json!(m));
    }
    if let Some(t) = req.temperature {
        o.insert("temperature".into(), json!(t));
    }
    if let Some(t) = req.top_p {
        o.insert("top_p".into(), json!(t));
    }
    if req.top_k.is_some() {
        enc.losses.dropped("/top_k", "OpenAI has no top_k");
    }
    if !req.stop.is_empty() {
        let mut stop = req.stop.clone();
        if stop.len() > 4 {
            enc.losses
                .dropped("/stop", "OpenAI accepts at most 4 stop sequences");
            stop.truncate(4);
        }
        o.insert("stop".into(), json!(stop));
    }
    if let Some(sd) = req.seed {
        o.insert("seed".into(), json!(sd));
    }
    if let Some(r) = &req.reasoning {
        let effort = match (r.mode, &r.effort, r.budget_tokens) {
            (Some(ReasoningMode::Off), _, _) => Some("none".to_owned()),
            (_, Some(e), _) => Some(e.clone()),
            (_, None, Some(b)) => {
                enc.losses.approx(
                    "/reasoning_effort",
                    "thinking budget mapped to an effort level",
                );
                Some(effort_for_budget(b).to_owned())
            }
            _ => None,
        };
        if let Some(e) = effort {
            o.insert("reasoning_effort".into(), json!(e));
        }
    }
    match &req.output {
        Some(OutputFormat::JsonObject) => {
            o.insert("response_format".into(), json!({"type": "json_object"}));
        }
        Some(OutputFormat::JsonSchema {
            name,
            schema,
            strict,
        }) => {
            let mut js = Map::new();
            js.insert(
                "name".into(),
                json!(name.clone().unwrap_or_else(|| "output".into())),
            );
            js.insert("schema".into(), schema.clone());
            if let Some(st) = strict {
                js.insert("strict".into(), json!(st));
            }
            o.insert(
                "response_format".into(),
                json!({"type": "json_schema", "json_schema": js}),
            );
        }
        None => {}
    }
    if req.stream {
        o.insert("stream".into(), json!(true));
        o.insert("stream_options".into(), json!({"include_usage": true}));
    }
    merge(&mut o, enc.losses.ext("", &req.ext, F));
    Ok(Value::Object(o))
}

fn finish_of(r: Option<&str>) -> Option<StopReason> {
    Some(match r? {
        "stop" => StopReason::EndTurn,
        "length" => StopReason::MaxTokens,
        "tool_calls" | "function_call" => StopReason::ToolUse,
        "content_filter" => StopReason::ContentFilter,
        other => StopReason::Other(other.to_owned()),
    })
}

fn finish_json(stop: &Option<StopReason>, tools: bool) -> &'static str {
    match stop {
        _ if tools => "tool_calls",
        Some(StopReason::MaxTokens) | Some(StopReason::ContextWindow) => "length",
        Some(StopReason::ToolUse) => "tool_calls",
        Some(StopReason::ContentFilter) | Some(StopReason::Refusal) => "content_filter",
        _ => "stop",
    }
}

pub(crate) fn usage_json(u: &Usage) -> Value {
    let input = u.input_total.unwrap_or(0);
    let output = u.output_total.unwrap_or(0);
    let mut o = json!({
        "prompt_tokens": input,
        "completion_tokens": output,
        "total_tokens": input + output,
        "prompt_tokens_details": {"cached_tokens": u.cache_read.unwrap_or(0)},
        "completion_tokens_details": {"reasoning_tokens": u.reasoning.unwrap_or(0)},
    });
    if let Some(w) = u.cache_write {
        o["prompt_tokens_details"]["cache_write_tokens"] = json!(w);
    }
    o
}

pub fn decode_response(v: Value) -> Result<Response, CodecError> {
    let m = obj(&v)?;
    let choice = v
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .ok_or_else(|| CodecError("response has no choices".into()))?;
    let msg = &choice["message"];
    let mut content = decode_reasoning(msg);
    content.extend(decode_content(msg.get("content"))?);
    if let Some(r) = msg.get("refusal").and_then(Value::as_str) {
        content.push(Block::text(r));
    }
    let n = content.len();
    content.extend(decode_tool_calls(msg.get("tool_calls"), 0, n));
    let mut stop = finish_of(choice.get("finish_reason").and_then(Value::as_str));
    if msg.get("refusal").and_then(Value::as_str).is_some() {
        stop = Some(StopReason::Refusal);
    }
    Ok(Response {
        id: s(&v, "id"),
        model: s(&v, "model"),
        content,
        stop,
        usage: v
            .get("usage")
            .map(|u| Usage::from_provider(F, u))
            .unwrap_or_default(),
        ext: ext(
            F,
            m,
            &["id", "object", "created", "model", "choices", "usage"],
        ),
    })
}

pub fn encode_response(r: &Response, losses: &mut Losses) -> Value {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut details = None;
    let mut calls = Vec::new();
    for (i, b) in r.content.iter().enumerate() {
        match b {
            Block::Text { text: t, .. } => text.push_str(t),
            Block::Thinking(t) => {
                if let Some(raw) = t.sig.as_ref().and_then(reasoning_details) {
                    collect_details(&mut details, raw);
                } else {
                    reasoning.push_str(&t.text);
                    if t.sig.is_some() {
                        losses.dropped(
                            format!("/content/{i}/signature"),
                            "no Chat reasoning details",
                        );
                    }
                }
            }
            Block::ToolCall(c) => calls.push(json!({
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.args},
            })),
            _ => losses.dropped(format!("/content/{i}"), "no Chat equivalent"),
        }
    }
    let mut msg = Map::new();
    msg.insert("role".into(), json!("assistant"));
    msg.insert(
        "content".into(),
        if text.is_empty() {
            Value::Null
        } else {
            json!(text)
        },
    );
    if !reasoning.is_empty() {
        msg.insert("reasoning_content".into(), json!(reasoning));
    }
    if let Some(details) = details {
        msg.insert("reasoning_details".into(), details);
    }
    if !calls.is_empty() {
        msg.insert("tool_calls".into(), Value::Array(calls));
    }
    msg.insert("refusal".into(), Value::Null);
    let mut o = Map::new();
    o.insert(
        "id".into(),
        json!(r.id.clone().unwrap_or_else(|| "chatcmpl-nabucore".into())),
    );
    o.insert("object".into(), json!("chat.completion"));
    o.insert("created".into(), json!(0));
    o.insert("model".into(), json!(r.model.clone().unwrap_or_default()));
    o.insert(
        "choices".into(),
        json!([{
            "index": 0,
            "message": msg,
            "finish_reason": finish_json(&r.stop, r.has_tool_calls()),
            "logprobs": null,
        }]),
    );
    o.insert("usage".into(), usage_json(&r.usage));
    merge(&mut o, losses.ext("", &r.ext, F));
    Value::Object(o)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Open {
    Text,
    Thinking,
}

#[derive(Default)]
struct ToolChunks {
    id: String,
    name: String,
    args: String,
}

impl ToolChunks {
    fn append(&mut self, chunk: &Value) {
        if let Some(id) = chunk.get("id").and_then(Value::as_str) {
            self.id.push_str(id);
        }
        if let Some(name) = chunk["function"]["name"].as_str() {
            self.name.push_str(name);
        }
        if let Some(args) = chunk["function"]["arguments"].as_str() {
            self.args.push_str(args);
        }
    }
}

/// Chat chunks → IR events.
#[derive(Default)]
pub struct Decoder {
    started: bool,
    open: Option<(Open, usize)>,
    tools: BTreeMap<u64, ToolChunks>,
    details: Option<Value>,
    plain_reasoning: bool,
    pending_reasoning: String,
    pending_text: String,
    next: usize,
    usage: Usage,
    stop: Option<StopReason>,
    ended: bool,
}

impl Decoder {
    fn switch(&mut self, kind: Open, start: BlockStart, out: &mut Vec<StreamEvent>) -> usize {
        if let Some((k, i)) = self.open
            && k == kind
        {
            return i;
        }
        self.close(out);
        let i = self.next;
        self.next += 1;
        self.open = Some((kind, i));
        out.push(StreamEvent::BlockStart {
            index: i,
            block: start,
        });
        i
    }

    fn close(&mut self, out: &mut Vec<StreamEvent>) {
        if let Some((_, i)) = self.open.take() {
            out.push(StreamEvent::BlockEnd { index: i });
        }
    }

    fn buffered(&self) -> bool {
        self.details.is_some() || !self.tools.is_empty()
    }

    fn flush(&mut self, out: &mut Vec<StreamEvent>) {
        self.close(out);
        if !self.pending_reasoning.is_empty() {
            let i = self.switch(
                Open::Thinking,
                BlockStart::Thinking { redacted: false },
                out,
            );
            out.push(StreamEvent::Delta {
                index: i,
                delta: Delta::Thinking(std::mem::take(&mut self.pending_reasoning)),
            });
            self.close(out);
        }
        if let Some(details) = self.details.take() {
            let thinking = details_block(details, self.plain_reasoning);
            let i = self.switch(
                Open::Thinking,
                BlockStart::Thinking { redacted: false },
                out,
            );
            if !thinking.text.is_empty() {
                out.push(StreamEvent::Delta {
                    index: i,
                    delta: Delta::Thinking(thinking.text),
                });
            }
            out.push(StreamEvent::Delta {
                index: i,
                delta: Delta::Sig(thinking.sig.expect("reasoning details")),
            });
            self.close(out);
        }
        if !self.pending_text.is_empty() {
            let i = self.switch(Open::Text, BlockStart::Text, out);
            out.push(StreamEvent::Delta {
                index: i,
                delta: Delta::Text(std::mem::take(&mut self.pending_text)),
            });
            self.close(out);
        }
        // IR blocks cannot interleave, so calls are emitted complete in provider-index order.
        for (_, call) in std::mem::take(&mut self.tools) {
            let i = self.next;
            self.next += 1;
            out.push(StreamEvent::BlockStart {
                index: i,
                block: BlockStart::ToolCall {
                    id: if call.id.is_empty() {
                        synth_call_id(0, i)
                    } else {
                        call.id
                    },
                    name: call.name,
                },
            });
            if !call.args.is_empty() {
                out.push(StreamEvent::Delta {
                    index: i,
                    delta: Delta::ToolArgs(call.args),
                });
            }
            out.push(StreamEvent::BlockEnd { index: i });
        }
    }

    fn end(&mut self, out: &mut Vec<StreamEvent>) {
        if self.ended {
            return;
        }
        self.flush(out);
        self.ended = true;
        out.push(StreamEvent::End {
            stop: self.stop.clone(),
            usage: self.usage.clone(),
        });
    }
}

impl StreamDecoder for Decoder {
    fn feed(&mut self, ev: &SseEvent) -> Result<Vec<StreamEvent>, CodecError> {
        let mut out = Vec::new();
        if !ev.has_data() {
            return Ok(out);
        }
        if ev.data.trim() == "[DONE]" {
            self.end(&mut out);
            return Ok(out);
        }
        let v: Value = serde_json::from_str(&ev.data)?;
        if v.get("error").is_some() {
            out.push(StreamEvent::Error(ProviderError::from_value(
                F,
                None,
                &v,
                Bytes::copy_from_slice(ev.data.as_bytes()),
            )));
            return Ok(out);
        }
        if !self.started {
            self.started = true;
            out.push(StreamEvent::Start {
                id: s(&v, "id"),
                model: s(&v, "model"),
            });
        }
        if let Some(choice) = v
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
        {
            let d = &choice["delta"];
            if let Some(details) = d.get("reasoning_details") {
                self.close(&mut out);
                append_details(&mut self.details, details);
            }
            if let Some(t) = reasoning_text(d) {
                self.plain_reasoning = true;
                if self.buffered() {
                    self.pending_reasoning.push_str(&t);
                } else {
                    let i = self.switch(
                        Open::Thinking,
                        BlockStart::Thinking { redacted: false },
                        &mut out,
                    );
                    out.push(StreamEvent::Delta {
                        index: i,
                        delta: Delta::Thinking(t),
                    });
                }
            }
            for key in ["content", "refusal"] {
                if let Some(t) = d.get(key).and_then(Value::as_str).filter(|t| !t.is_empty()) {
                    if self.buffered() {
                        self.pending_text.push_str(t);
                    } else {
                        let i = self.switch(Open::Text, BlockStart::Text, &mut out);
                        out.push(StreamEvent::Delta {
                            index: i,
                            delta: Delta::Text(t.to_owned()),
                        });
                    }
                }
            }
            for tc in d
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                self.close(&mut out);
                let k = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
                self.tools.entry(k).or_default().append(tc);
            }
            if let Some(stop) = finish_of(choice.get("finish_reason").and_then(Value::as_str)) {
                self.stop = Some(stop);
                self.flush(&mut out);
            }
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            self.usage.merge(&Usage::from_provider(F, u));
            out.push(StreamEvent::Usage(self.usage.clone()));
        }
        Ok(out)
    }

    fn finish(&mut self) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        if self.started {
            self.end(&mut out);
        }
        out
    }
}

/// IR events → Chat chunks.
#[derive(Default)]
pub struct Encoder {
    id: Option<String>,
    model: Option<String>,
    started: bool,
    tools: HashMap<usize, u64>,
    skipped: Vec<usize>,
    next_tool: u64,
    thinking: HashMap<usize, (Vec<String>, Option<Value>)>,
    usage: Usage,
    ended: bool,
}

impl Encoder {
    fn chunk(&self, delta: Value, finish: Value) -> Bytes {
        let v = json!({
            "id": self.id.clone().unwrap_or_else(|| "chatcmpl-nabucore".into()),
            "object": "chat.completion.chunk",
            "created": 0,
            "model": self.model.clone().unwrap_or_default(),
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": null}],
        });
        super::sse(None, &v)
    }

    fn start(&mut self, out: &mut Vec<Bytes>) {
        if !self.started {
            self.started = true;
            out.push(self.chunk(
                json!({"role": "assistant", "content": "", "refusal": null}),
                Value::Null,
            ));
        }
    }

    fn close_thinking(&mut self, index: usize, out: &mut Vec<Bytes>) {
        let Some((texts, details)) = self.thinking.remove(&index) else {
            return;
        };
        if let Some(details) = details {
            out.push(self.chunk(json!({"reasoning_details": details}), Value::Null));
        } else {
            for text in texts {
                out.push(self.chunk(json!({"reasoning_content": text}), Value::Null));
            }
        }
    }
}

impl StreamEncoder for Encoder {
    fn encode(&mut self, ev: &StreamEvent) -> Vec<Bytes> {
        let mut out = Vec::new();
        match ev {
            StreamEvent::Start { id, model } => {
                self.id = id.clone();
                self.model = model.clone();
            }
            StreamEvent::BlockStart { index, block } => {
                self.start(&mut out);
                match block {
                    BlockStart::ToolCall { id, name } => {
                        let k = self.next_tool;
                        self.next_tool += 1;
                        self.tools.insert(*index, k);
                        out.push(self.chunk(
                            json!({"tool_calls": [{"index": k, "id": id, "type": "function", "function": {"name": name, "arguments": ""}}]}),
                            Value::Null,
                        ));
                    }
                    BlockStart::Thinking { .. } => {
                        self.thinking.insert(*index, (Vec::new(), None));
                    }
                    BlockStart::Opaque(_) => self.skipped.push(*index),
                    _ => {}
                }
            }
            StreamEvent::Delta { index, delta } => {
                if self.skipped.contains(index) {
                    return out;
                }
                self.start(&mut out);
                let d = match delta {
                    Delta::Text(t) => json!({"content": t}),
                    Delta::Thinking(t) => {
                        self.thinking.entry(*index).or_default().0.push(t.clone());
                        return out;
                    }
                    Delta::ToolArgs(a) => match self.tools.get(index) {
                        Some(k) => {
                            json!({"tool_calls": [{"index": k, "function": {"arguments": a}}]})
                        }
                        None => return out,
                    },
                    Delta::Sig(sig) => {
                        if let Some(details) = reasoning_details(sig)
                            && let Some((_, saved)) = self.thinking.get_mut(index)
                        {
                            *saved = Some(details.clone());
                        }
                        return out;
                    }
                };
                out.push(self.chunk(d, Value::Null));
            }
            StreamEvent::BlockEnd { index } => {
                self.close_thinking(*index, &mut out);
                self.skipped.retain(|i| i != index);
            }
            StreamEvent::Usage(u) => self.usage.merge(u),
            StreamEvent::End { stop, usage } => {
                self.start(&mut out);
                let mut pending: Vec<_> = self.thinking.keys().copied().collect();
                pending.sort_unstable();
                for index in pending {
                    self.close_thinking(index, &mut out);
                }
                self.usage.merge(usage);
                let f = finish_json(stop, !self.tools.is_empty());
                out.push(self.chunk(json!({}), json!(f)));
                let u = json!({
                    "id": self.id.clone().unwrap_or_else(|| "chatcmpl-nabucore".into()),
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": self.model.clone().unwrap_or_default(),
                    "choices": [],
                    "usage": usage_json(&self.usage),
                });
                out.push(super::sse(None, &u));
                out.push(crate::sse::encode(None, None, "[DONE]"));
                self.ended = true;
            }
            StreamEvent::Error(e) => {
                let (_, body) = super::encode_error(F, e);
                out.push(crate::sse::encode(
                    None,
                    None,
                    &String::from_utf8_lossy(&body),
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
