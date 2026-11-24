use std::collections::VecDeque;

use bytes::Bytes;
use serde_json::{Map, Value, json};

use super::{
    CodecError, EncodeOptions, StreamDecoder, StreamEncoder, args_object, ext, merge, obj, rest, s,
    synth_call_id,
};
use crate::ir::*;
use crate::{ErrorKind, Format, ProviderError, SseEvent, Usage};

const F: Format = Format::Gemini;

/// Placeholder Google accepts for function calls whose signature was lost in transit.
pub const SKIP_SIGNATURE: &str = "skip_thought_signature_validator";

const REQ_KNOWN: &[&str] = &[
    "contents",
    "systemInstruction",
    "system_instruction",
    "tools",
    "toolConfig",
    "generationConfig",
    "model",
];

const GEN_KNOWN: &[&str] = &[
    "maxOutputTokens",
    "temperature",
    "topP",
    "topK",
    "stopSequences",
    "seed",
    "thinkingConfig",
    "responseMimeType",
    "responseJsonSchema",
    "responseSchema",
];

pub fn path(model: &str, stream: bool) -> String {
    let model = model.strip_prefix("models/").unwrap_or(model);
    if stream {
        format!("/v1beta/models/{model}:streamGenerateContent?alt=sse")
    } else {
        format!("/v1beta/models/{model}:generateContent")
    }
}

fn model_and_stream(path: &str) -> (Option<String>, bool) {
    let p = path.split('?').next().unwrap_or(path);
    let Some(start) = p.find("/models/") else {
        return (None, false);
    };
    let tail = &p[start + "/models/".len()..];
    match tail.split_once(':') {
        Some((m, method)) => (Some(m.to_owned()), method == "streamGenerateContent"),
        None => (Some(tail.to_owned()), false),
    }
}

// Gemini `Schema` uses upper-case types; JSON Schema wants lower case.
fn lower_types(v: &mut Value) {
    match v {
        Value::Object(m) => {
            for (k, val) in m.iter_mut() {
                if k == "type"
                    && let Value::String(t) = val
                {
                    *t = t.to_ascii_lowercase();
                } else {
                    lower_types(val);
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(lower_types),
        _ => {}
    }
}

fn sig_of(p: &Value) -> Option<Opaque> {
    p.get("thoughtSignature").map(|v| Opaque::new(F, v.clone()))
}

/// Matches tool results to calls that came without ids.
#[derive(Default)]
struct Ids {
    unnamed: VecDeque<(String, String)>,
}

impl Ids {
    fn call(&mut self, p: &Value, msg: usize, block: usize) -> ToolCall {
        let fc = &p["functionCall"];
        let name = s(fc, "name").unwrap_or_default();
        let id = s(fc, "id").unwrap_or_else(|| {
            let id = synth_call_id(msg, block);
            self.unnamed.push_back((name.clone(), id.clone()));
            id
        });
        ToolCall {
            id,
            name,
            args: serde_json::to_string(fc.get("args").unwrap_or(&json!({}))).unwrap_or_default(),
            sig: sig_of(p),
        }
    }

    fn result(&mut self, p: &Value) -> ToolResult {
        let fr = &p["functionResponse"];
        let name = s(fr, "name").unwrap_or_default();
        let call_id = s(fr, "id").unwrap_or_else(|| {
            match self.unnamed.iter().position(|(n, _)| *n == name) {
                Some(pos) => self
                    .unnamed
                    .remove(pos)
                    .map(|(_, id)| id)
                    .unwrap_or_default(),
                None => name.clone(),
            }
        });
        let resp = fr.get("response").cloned().unwrap_or(json!({}));
        let (text, is_error) = match (resp.get("output"), resp.get("error")) {
            (Some(Value::String(t)), _) => (t.clone(), false),
            (Some(o), _) => (o.to_string(), false),
            (None, Some(Value::String(e))) => (e.clone(), true),
            (None, Some(e)) => (e.to_string(), true),
            (None, None) => (resp.to_string(), false),
        };
        ToolResult {
            call_id,
            name: Some(name),
            content: vec![Block::text(text)],
            is_error,
            cache: None,
        }
    }
}

fn decode_part(p: &Value, ids: &mut Ids, msg: usize, block: usize) -> Result<Block, CodecError> {
    let m = obj(p)?;
    if m.contains_key("functionCall") {
        return Ok(Block::ToolCall(ids.call(p, msg, block)));
    }
    if m.contains_key("functionResponse") {
        return Ok(Block::ToolResult(ids.result(p)));
    }
    if let Some(t) = m.get("text").and_then(Value::as_str) {
        if p.get("thought").and_then(Value::as_bool) == Some(true) {
            return Ok(Block::Thinking(Thinking {
                text: t.to_owned(),
                sig: sig_of(p),
                redacted: false,
            }));
        }
        return Ok(Block::Text {
            text: t.to_owned(),
            cache: None,
            sig: sig_of(p),
            ext: ext(F, m, &["text", "thought", "thoughtSignature"]),
        });
    }
    if let Some(d) = m.get("inlineData") {
        return Ok(Block::Media {
            media: Media {
                mime: s(d, "mimeType"),
                source: MediaSource::Base64(s(d, "data").unwrap_or_default()),
                filename: s(d, "displayName"),
            },
            cache: None,
            ext: ext(F, m, &["inlineData"]),
        });
    }
    if let Some(d) = m.get("fileData") {
        return Ok(Block::Media {
            media: Media {
                mime: s(d, "mimeType"),
                source: MediaSource::Url(s(d, "fileUri").unwrap_or_default()),
                filename: s(d, "displayName"),
            },
            cache: None,
            ext: ext(F, m, &["fileData"]),
        });
    }
    Ok(Block::Opaque(Opaque::new(F, p.clone())))
}

fn decode_parts(v: Option<&Value>, ids: &mut Ids, msg: usize) -> Result<Vec<Block>, CodecError> {
    v.and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(i, p)| decode_part(p, ids, msg, i))
        .collect()
}

pub fn decode_request(path: &str, v: Value) -> Result<Request, CodecError> {
    let m = obj(&v)?;
    let (model, stream) = model_and_stream(path);
    let mut req = Request {
        model: model.or_else(|| s(&v, "model")).unwrap_or_default(),
        stream,
        ..Request::default()
    };
    let mut fields = rest(m, REQ_KNOWN);
    let mut ids = Ids::default();
    if let Some(si) = v.get("systemInstruction").or(v.get("system_instruction")) {
        req.system = decode_parts(si.get("parts"), &mut ids, 0)?;
    }
    for (i, c) in v
        .get("contents")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let role = match c.get("role").and_then(Value::as_str) {
            Some("model") => Role::Assistant,
            _ => Role::User,
        };
        req.messages.push(Message {
            role,
            content: decode_parts(c.get("parts"), &mut ids, i)?,
            ext: ext(F, obj(c)?, &["role", "parts"]),
        });
    }
    for t in v
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let tm = obj(t)?;
        if let Some(Value::Array(decls)) = tm.get("functionDeclarations") {
            for d in decls {
                let schema = match (d.get("parametersJsonSchema"), d.get("parameters")) {
                    (Some(js), _) => js.clone(),
                    (None, Some(p)) => {
                        let mut p = p.clone();
                        lower_types(&mut p);
                        p
                    }
                    (None, None) => json!({"type": "object", "properties": {}}),
                };
                req.tools.push(ToolDef::Function(Tool {
                    name: s(d, "name").unwrap_or_default(),
                    description: s(d, "description"),
                    schema,
                    strict: None,
                    cache: None,
                    ext: ext(
                        F,
                        obj(d)?,
                        &["name", "description", "parametersJsonSchema", "parameters"],
                    ),
                }));
            }
            let other = rest(tm, &["functionDeclarations"]);
            if !other.is_empty() {
                req.tools
                    .push(ToolDef::Opaque(Opaque::new(F, Value::Object(other))));
            }
        } else {
            req.tools.push(ToolDef::Opaque(Opaque::new(F, t.clone())));
        }
    }
    if let Some(tc) = v.get("toolConfig") {
        let fc = &tc["functionCallingConfig"];
        let allowed: Vec<&str> = fc["allowedFunctionNames"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        req.tool_choice = match fc["mode"].as_str() {
            Some("AUTO") => Some(ToolChoice::Auto),
            Some("NONE") => Some(ToolChoice::None),
            Some("ANY") if allowed.len() == 1 => Some(ToolChoice::Tool(allowed[0].into())),
            Some("ANY") => Some(ToolChoice::Required),
            _ => None,
        };
        if req.tool_choice.is_none() || allowed.len() > 1 {
            fields.insert("toolConfig".into(), tc.clone());
        }
    }
    if let Some(gc) = v.get("generationConfig").and_then(Value::as_object) {
        let g = Value::Object(gc.clone());
        req.max_tokens = g.get("maxOutputTokens").and_then(Value::as_u64);
        req.temperature = g.get("temperature").and_then(Value::as_f64);
        req.top_p = g.get("topP").and_then(Value::as_f64);
        req.top_k = g.get("topK").and_then(Value::as_f64).map(|k| k as u64);
        req.seed = g.get("seed").and_then(Value::as_i64);
        req.stop = g["stopSequences"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|s| s.as_str().map(str::to_owned))
            .collect();
        if let Some(tc) = g.get("thinkingConfig") {
            let mut r = ReasoningCfg {
                include: tc.get("includeThoughts").and_then(Value::as_bool),
                effort: tc
                    .get("thinkingLevel")
                    .and_then(Value::as_str)
                    .map(str::to_ascii_lowercase),
                ..ReasoningCfg::default()
            };
            match tc.get("thinkingBudget").and_then(Value::as_i64) {
                Some(0) => r.mode = Some(ReasoningMode::Off),
                Some(b) if b < 0 => r.mode = Some(ReasoningMode::Adaptive),
                Some(b) => {
                    r.mode = Some(ReasoningMode::On);
                    r.budget_tokens = Some(b as u64);
                }
                None => {}
            }
            req.reasoning = Some(r);
        }
        if g.get("responseMimeType").and_then(Value::as_str) == Some("application/json") {
            req.output = Some(
                match g.get("responseJsonSchema").or(g.get("responseSchema")) {
                    Some(schema) => {
                        let mut schema = schema.clone();
                        lower_types(&mut schema);
                        OutputFormat::JsonSchema {
                            name: None,
                            schema,
                            strict: None,
                        }
                    }
                    None => OutputFormat::JsonObject,
                },
            );
        }
        let other = rest(gc, GEN_KNOWN);
        if !other.is_empty() {
            fields.insert("generationConfig".into(), Value::Object(other));
        }
    }
    req.ext = Ext::new(F, fields);
    Ok(req)
}

struct Enc<'a> {
    losses: &'a mut Losses,
    names: Vec<(String, String)>,
    /// Encoding a response for a client, where unsigned thoughts are fine.
    response: bool,
    /// Tool-result media waiting to follow the function responses of the current turn.
    hoisted: Vec<Value>,
}

impl Enc<'_> {
    fn sig(&mut self, p: &mut Map<String, Value>, sig: &Option<Opaque>, path: &str) {
        match sig {
            Some(o) if o.for_format(F).is_some() => {
                p.insert("thoughtSignature".into(), o.data.clone());
            }
            Some(_) => self.losses.dropped(
                format!("{path}/signature"),
                "signature from another provider",
            ),
            None => {}
        }
    }

    fn part(&mut self, b: &Block, first_call: &mut bool, path: &str) -> Option<Value> {
        let mut p = Map::new();
        match b {
            Block::Text {
                text,
                sig,
                ext,
                cache,
            } => {
                if text.is_empty() && !sig.as_ref().is_some_and(|s| s.for_format(F).is_some()) {
                    return None;
                }
                if cache.is_some() {
                    self.losses
                        .dropped(format!("{path}/cache"), "Gemini has no cache breakpoints");
                }
                p.insert("text".into(), json!(text));
                self.sig(&mut p, sig, path);
                merge(&mut p, self.losses.ext(path, ext, F));
            }
            Block::Thinking(t) => match t.sig.as_ref().and_then(|s| s.for_format(F)) {
                Some(sig) => {
                    p.insert("text".into(), json!(t.text));
                    p.insert("thought".into(), json!(true));
                    p.insert("thoughtSignature".into(), sig.clone());
                }
                None if self.response => {
                    p.insert("text".into(), json!(t.text));
                    p.insert("thought".into(), json!(true));
                }
                None => {
                    self.losses
                        .dropped(path, "reasoning without a Gemini signature");
                    return None;
                }
            },
            Block::Media { media, ext, .. } => {
                let mime = media
                    .mime
                    .clone()
                    .unwrap_or_else(|| "application/octet-stream".into());
                match &media.source {
                    MediaSource::Base64(d) => {
                        p.insert("inlineData".into(), json!({"mimeType": mime, "data": d}));
                    }
                    MediaSource::Url(u) | MediaSource::FileId(u) => {
                        if !u.starts_with("https://generativelanguage.googleapis.com/")
                            && !u.starts_with("gs://")
                        {
                            self.losses
                                .approx(path, "Gemini may not fetch arbitrary URLs");
                        }
                        p.insert("fileData".into(), json!({"mimeType": mime, "fileUri": u}));
                    }
                }
                merge(&mut p, self.losses.ext(path, ext, F));
            }
            Block::ToolCall(c) => {
                self.names.push((c.id.clone(), c.name.clone()));
                p.insert(
                    "functionCall".into(),
                    json!({"id": c.id, "name": c.name, "args": args_object(&c.args, &format!("{path}/args"), self.losses)}),
                );
                match c.sig.as_ref().and_then(|s| s.for_format(F)) {
                    Some(sig) => {
                        p.insert("thoughtSignature".into(), sig.clone());
                    }
                    None if *first_call => {
                        self.losses.approx(
                            format!("{path}/signature"),
                            "missing Gemini signature replaced by the skip placeholder",
                        );
                        p.insert("thoughtSignature".into(), json!(SKIP_SIGNATURE));
                    }
                    None => {}
                }
                *first_call = false;
            }
            Block::ToolResult(r) => {
                let name = r
                    .name
                    .clone()
                    .or_else(|| {
                        self.names
                            .iter()
                            .rev()
                            .find(|(id, _)| *id == r.call_id)
                            .map(|(_, n)| n.clone())
                    })
                    .unwrap_or_else(|| {
                        self.losses
                            .approx(path, "tool name unknown for this result");
                        "unknown".into()
                    });
                let mut label = true;
                for (i, b) in r.content.iter().enumerate() {
                    let bp = format!("{path}/content/{i}");
                    match b {
                        Block::Text { .. } => {}
                        Block::Media { .. } => {
                            self.losses.approx(
                                &bp,
                                "Gemini tool results carry text only; media moved after the function response",
                            );
                            if let Some(m) = self.part(b, &mut false, &bp) {
                                if std::mem::take(&mut label) {
                                    self.hoisted.push(
                                        json!({"text": super::attachments_label(&r.call_id)}),
                                    );
                                }
                                self.hoisted.push(m);
                            }
                        }
                        _ => self
                            .losses
                            .dropped(bp, "Gemini tool results carry text only"),
                    }
                }
                let text = plain_text(&r.content);
                let key = if r.is_error { "error" } else { "output" };
                p.insert(
                    "functionResponse".into(),
                    json!({"id": r.call_id, "name": name, "response": {key: text}}),
                );
            }
            Block::Opaque(o) => match o.for_format(F) {
                Some(v) => return Some(v.clone()),
                None => {
                    self.losses.dropped(
                        path,
                        format!("{:?} block has no Gemini equivalent", o.format),
                    );
                    return None;
                }
            },
        }
        Some(Value::Object(p))
    }
}

pub fn encode_request(
    req: &Request,
    _opts: &EncodeOptions,
    losses: &mut Losses,
) -> Result<Value, CodecError> {
    let mut enc = Enc {
        losses,
        names: Vec::new(),
        hoisted: Vec::new(),
        response: false,
    };
    let mut o = Map::new();
    let mut contents: Vec<Value> = Vec::new();
    for (i, m) in req.messages.iter().enumerate() {
        let path = format!("/messages/{i}");
        let role = match m.role {
            Role::Assistant => "model",
            Role::User => "user",
            Role::System => {
                enc.losses.approx(
                    &path,
                    "mid-conversation system message sent as user content",
                );
                "user"
            }
        };
        let mut first_call = true;
        let mut parts: Vec<Value> = m
            .content
            .iter()
            .enumerate()
            .filter_map(|(j, b)| enc.part(b, &mut first_call, &format!("{path}/content/{j}")))
            .collect();
        if !enc.hoisted.is_empty() {
            let at = parts
                .iter()
                .rposition(|p| p.get("functionResponse").is_some())
                .map_or(parts.len(), |i| i + 1);
            parts.splice(at..at, enc.hoisted.drain(..));
        }
        if parts.is_empty() {
            continue;
        }
        let ext = enc.losses.ext(&path, &m.ext, F);
        match contents.last_mut() {
            Some(last) if last["role"] == role && ext.is_none() => {
                last["parts"].as_array_mut().expect("parts").extend(parts);
            }
            _ => {
                let mut c = Map::new();
                c.insert("role".into(), json!(role));
                c.insert("parts".into(), Value::Array(parts));
                merge(&mut c, ext);
                contents.push(Value::Object(c));
            }
        }
    }
    o.insert("contents".into(), Value::Array(contents));
    if !req.system.is_empty() {
        let mut first = false;
        let parts: Vec<Value> = req
            .system
            .iter()
            .enumerate()
            .filter_map(|(i, b)| enc.part(b, &mut first, &format!("/system/{i}")))
            .collect();
        o.insert("systemInstruction".into(), json!({"parts": parts}));
    }
    let mut decls = Vec::new();
    let mut tools = Vec::new();
    for (i, t) in req.tools.iter().enumerate() {
        match t {
            ToolDef::Function(t) => {
                let mut d = Map::new();
                d.insert("name".into(), json!(t.name));
                if let Some(desc) = &t.description {
                    d.insert("description".into(), json!(desc));
                }
                d.insert("parametersJsonSchema".into(), t.schema.clone());
                if t.cache.is_some() {
                    enc.losses.dropped(
                        format!("/tools/{i}/cache"),
                        "Gemini has no cache breakpoints",
                    );
                }
                merge(&mut d, enc.losses.ext(&format!("/tools/{i}"), &t.ext, F));
                decls.push(Value::Object(d));
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
    if !decls.is_empty() {
        tools.insert(0, json!({"functionDeclarations": decls}));
    }
    if !tools.is_empty() {
        o.insert("tools".into(), Value::Array(tools));
    }
    if let Some(c) = &req.tool_choice {
        let fc = match c {
            ToolChoice::Auto => json!({"mode": "AUTO"}),
            ToolChoice::None => json!({"mode": "NONE"}),
            ToolChoice::Required => json!({"mode": "ANY"}),
            ToolChoice::Tool(n) => json!({"mode": "ANY", "allowedFunctionNames": [n]}),
        };
        o.insert("toolConfig".into(), json!({"functionCallingConfig": fc}));
    }
    if req.parallel_tool_calls == Some(false) {
        enc.losses.dropped(
            "/parallel_tool_calls",
            "Gemini cannot disable parallel calls",
        );
    }
    let mut gc = Map::new();
    if let Some(m) = req.max_tokens {
        gc.insert("maxOutputTokens".into(), json!(m));
    }
    if let Some(t) = req.temperature {
        gc.insert("temperature".into(), json!(t));
    }
    if let Some(t) = req.top_p {
        gc.insert("topP".into(), json!(t));
    }
    if let Some(k) = req.top_k {
        gc.insert("topK".into(), json!(k));
    }
    if !req.stop.is_empty() {
        gc.insert("stopSequences".into(), json!(req.stop));
    }
    if let Some(sd) = req.seed {
        gc.insert("seed".into(), json!(sd));
    }
    if let Some(r) = &req.reasoning {
        let mut tc = Map::new();
        match (r.mode, r.budget_tokens, r.effort.as_deref()) {
            (Some(ReasoningMode::Off), _, _) | (_, _, Some("none")) => {
                tc.insert("thinkingBudget".into(), json!(0));
            }
            (_, Some(b), _) => {
                tc.insert("thinkingBudget".into(), json!(b));
            }
            (_, None, Some(e)) => {
                let level = match e {
                    "minimal" | "low" | "medium" | "high" | "xhigh" => e.to_ascii_uppercase(),
                    _ => {
                        enc.losses.approx(
                            "/generationConfig/thinkingConfig/thinkingLevel",
                            format!("{e} mapped to HIGH"),
                        );
                        "HIGH".into()
                    }
                };
                tc.insert("thinkingLevel".into(), json!(level));
            }
            (Some(ReasoningMode::Adaptive), None, None) => {
                tc.insert("thinkingBudget".into(), json!(-1));
            }
            _ => {}
        }
        if let Some(inc) = r.include {
            tc.insert("includeThoughts".into(), json!(inc));
        }
        if !tc.is_empty() {
            gc.insert("thinkingConfig".into(), Value::Object(tc));
        }
    }
    match &req.output {
        Some(OutputFormat::JsonObject) => {
            gc.insert("responseMimeType".into(), json!("application/json"));
        }
        Some(OutputFormat::JsonSchema { schema, .. }) => {
            gc.insert("responseMimeType".into(), json!("application/json"));
            gc.insert("responseJsonSchema".into(), schema.clone());
        }
        None => {}
    }
    if !gc.is_empty() {
        o.insert("generationConfig".into(), Value::Object(gc));
    }
    merge(&mut o, enc.losses.ext("", &req.ext, F));
    Ok(Value::Object(o))
}

pub(crate) fn finish_of(r: Option<&str>, has_tools: bool) -> Option<StopReason> {
    Some(match r? {
        "STOP" if has_tools => StopReason::ToolUse,
        "STOP" => StopReason::EndTurn,
        "MAX_TOKENS" => StopReason::MaxTokens,
        "SAFETY"
        | "RECITATION"
        | "BLOCKLIST"
        | "PROHIBITED_CONTENT"
        | "SPII"
        | "IMAGE_SAFETY"
        | "IMAGE_PROHIBITED_CONTENT"
        | "IMAGE_RECITATION" => StopReason::ContentFilter,
        other => StopReason::Other(other.to_owned()),
    })
}

fn finish_json(stop: &Option<StopReason>) -> &'static str {
    match stop {
        Some(StopReason::MaxTokens) | Some(StopReason::ContextWindow) => "MAX_TOKENS",
        Some(StopReason::ContentFilter) | Some(StopReason::Refusal) => "SAFETY",
        _ => "STOP",
    }
}

pub(crate) fn usage_json(u: &Usage) -> Value {
    let input = u.input_total.unwrap_or(0);
    let output = u.output_total.unwrap_or(0);
    let thoughts = u.reasoning.unwrap_or(0);
    let mut o = json!({
        "promptTokenCount": input,
        "candidatesTokenCount": output.saturating_sub(thoughts),
        "totalTokenCount": input + output,
    });
    if thoughts > 0 {
        o["thoughtsTokenCount"] = json!(thoughts);
    }
    if let Some(c) = u.cache_read {
        o["cachedContentTokenCount"] = json!(c);
    }
    o
}

pub(crate) fn error_status(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::InvalidRequest | ErrorKind::RequestTooLarge => "INVALID_ARGUMENT",
        ErrorKind::Auth => "UNAUTHENTICATED",
        ErrorKind::PermissionDenied | ErrorKind::Billing => "PERMISSION_DENIED",
        ErrorKind::NotFound => "NOT_FOUND",
        ErrorKind::Conflict => "ABORTED",
        ErrorKind::RateLimited => "RESOURCE_EXHAUSTED",
        ErrorKind::Overloaded => "UNAVAILABLE",
        ErrorKind::Timeout => "DEADLINE_EXCEEDED",
        ErrorKind::Server | ErrorKind::Other => "INTERNAL",
    }
}

pub fn decode_response(v: Value) -> Result<Response, CodecError> {
    let cand = &v["candidates"][0];
    let mut ids = Ids::default();
    let content = decode_parts(cand["content"].get("parts"), &mut ids, 0)?;
    let has_tools = content.iter().any(|b| matches!(b, Block::ToolCall(_)));
    let mut stop = finish_of(cand.get("finishReason").and_then(Value::as_str), has_tools);
    if cand.is_null() && v["promptFeedback"].get("blockReason").is_some() {
        stop = Some(StopReason::ContentFilter);
    }
    Ok(Response {
        id: s(&v, "responseId"),
        model: s(&v, "modelVersion"),
        content,
        stop,
        usage: v
            .get("usageMetadata")
            .map(|u| Usage::from_provider(F, u))
            .unwrap_or_default(),
        ext: Ext::default(),
    })
}

pub fn encode_response(r: &Response, losses: &mut Losses) -> Value {
    let mut enc = Enc {
        losses,
        names: Vec::new(),
        hoisted: Vec::new(),
        response: true,
    };
    let mut first = false;
    let parts: Vec<Value> = r
        .content
        .iter()
        .enumerate()
        .filter_map(|(i, b)| enc.part(b, &mut first, &format!("/content/{i}")))
        .collect();
    let mut o = json!({
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish_json(&r.stop), "index": 0}],
        "usageMetadata": usage_json(&r.usage),
    });
    if let Some(m) = &r.model {
        o["modelVersion"] = json!(m);
    }
    if let Some(id) = &r.id {
        o["responseId"] = json!(id);
    }
    o
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Open {
    Text,
    Thinking,
}

/// Gemini chunks → IR events.
#[derive(Default)]
pub struct Decoder {
    started: bool,
    id: Option<String>,
    open: Option<(Open, usize)>,
    next: usize,
    usage: Usage,
    stop: Option<StopReason>,
    finish: Option<String>,
    saw_tool: bool,
    ended: bool,
}

impl Decoder {
    fn close(&mut self, out: &mut Vec<StreamEvent>) {
        if let Some((_, i)) = self.open.take() {
            out.push(StreamEvent::BlockEnd { index: i });
        }
    }

    fn text(&mut self, kind: Open, text: &str, sig: Option<Opaque>, out: &mut Vec<StreamEvent>) {
        let i = match self.open {
            Some((k, i)) if k == kind => i,
            _ => {
                self.close(out);
                let i = self.next;
                self.next += 1;
                let block = match kind {
                    Open::Text => BlockStart::Text,
                    Open::Thinking => BlockStart::Thinking { redacted: false },
                };
                out.push(StreamEvent::BlockStart { index: i, block });
                self.open = Some((kind, i));
                i
            }
        };
        if !text.is_empty() {
            let delta = match kind {
                Open::Text => Delta::Text(text.to_owned()),
                Open::Thinking => Delta::Thinking(text.to_owned()),
            };
            out.push(StreamEvent::Delta { index: i, delta });
        }
        if let Some(sig) = sig {
            out.push(StreamEvent::Delta {
                index: i,
                delta: Delta::Sig(sig),
            });
            // A signed part must stay a part of its own.
            self.close(out);
        }
    }

    fn whole(&mut self, block: BlockStart, deltas: Vec<Delta>, out: &mut Vec<StreamEvent>) {
        self.close(out);
        let i = self.next;
        self.next += 1;
        out.push(StreamEvent::BlockStart { index: i, block });
        for delta in deltas {
            out.push(StreamEvent::Delta { index: i, delta });
        }
        out.push(StreamEvent::BlockEnd { index: i });
    }
}

impl StreamDecoder for Decoder {
    fn feed(&mut self, ev: &SseEvent) -> Result<Vec<StreamEvent>, CodecError> {
        let mut out = Vec::new();
        if !ev.has_data() {
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
            self.id = s(&v, "responseId");
            out.push(StreamEvent::Start {
                id: self.id.clone(),
                model: s(&v, "modelVersion"),
            });
        }
        let cand = &v["candidates"][0];
        for p in cand["content"]["parts"].as_array().into_iter().flatten() {
            if let Some(fc) = p.get("functionCall") {
                self.saw_tool = true;
                let id = s(fc, "id").unwrap_or_else(|| {
                    format!("{}_{}", self.id.as_deref().unwrap_or("call"), self.next)
                });
                let mut deltas = vec![Delta::ToolArgs(
                    serde_json::to_string(fc.get("args").unwrap_or(&json!({}))).unwrap_or_default(),
                )];
                if let Some(sig) = sig_of(p) {
                    deltas.push(Delta::Sig(sig));
                }
                self.whole(
                    BlockStart::ToolCall {
                        id,
                        name: s(fc, "name").unwrap_or_default(),
                    },
                    deltas,
                    &mut out,
                );
            } else if let Some(t) = p.get("text").and_then(Value::as_str) {
                let kind = if p.get("thought").and_then(Value::as_bool) == Some(true) {
                    Open::Thinking
                } else {
                    Open::Text
                };
                self.text(kind, t, sig_of(p), &mut out);
            } else {
                self.whole(
                    BlockStart::Opaque(Opaque::new(F, p.clone())),
                    Vec::new(),
                    &mut out,
                );
            }
        }
        if let Some(fr) = cand.get("finishReason").and_then(Value::as_str) {
            self.finish = Some(fr.to_owned());
        }
        if let Some(u) = v.get("usageMetadata") {
            self.usage.merge(&Usage::from_provider(F, u));
            out.push(StreamEvent::Usage(self.usage.clone()));
        }
        Ok(out)
    }

    fn finish(&mut self) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        if self.ended || !self.started {
            return out;
        }
        self.close(&mut out);
        self.ended = true;
        self.stop = finish_of(self.finish.as_deref(), self.saw_tool);
        out.push(StreamEvent::End {
            stop: self.stop.clone(),
            usage: self.usage.clone(),
        });
        out
    }
}

/// IR events → Gemini chunks.
#[derive(Default)]
pub struct Encoder {
    id: Option<String>,
    model: Option<String>,
    blocks: std::collections::HashMap<usize, (BlockStart, String, Option<Opaque>)>,
    called: bool,
    usage: Usage,
    ended: bool,
}

impl Encoder {
    fn chunk(&self, parts: Vec<Value>, finish: Option<&str>, usage: bool) -> Bytes {
        let mut cand = json!({"content": {"role": "model", "parts": parts}, "index": 0});
        if let Some(f) = finish {
            cand["finishReason"] = json!(f);
        }
        let mut v = json!({"candidates": [cand]});
        if usage {
            v["usageMetadata"] = usage_json(&self.usage);
        }
        if let Some(m) = &self.model {
            v["modelVersion"] = json!(m);
        }
        if let Some(id) = &self.id {
            v["responseId"] = json!(id);
        }
        super::sse(None, &v)
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
                self.blocks
                    .insert(*index, (block.clone(), String::new(), None));
            }
            StreamEvent::Delta { index, delta } => {
                let Some((kind, buf, sig)) = self.blocks.get_mut(index) else {
                    return out;
                };
                match (kind, delta) {
                    (BlockStart::Text, Delta::Text(t)) => {
                        let part = json!({"text": t});
                        out.push(self.chunk(vec![part], None, false));
                    }
                    (BlockStart::Thinking { .. }, Delta::Thinking(t)) => {
                        let part = json!({"text": t, "thought": true});
                        out.push(self.chunk(vec![part], None, false));
                    }
                    (BlockStart::ToolCall { .. }, Delta::ToolArgs(a)) => buf.push_str(a),
                    (_, Delta::Sig(o)) if o.for_format(F).is_some() => *sig = Some(o.clone()),
                    _ => {}
                }
            }
            StreamEvent::BlockEnd { index } => {
                let Some((kind, buf, sig)) = self.blocks.remove(index) else {
                    return out;
                };
                let part = match kind {
                    BlockStart::ToolCall { id, name } => {
                        let mut losses = Losses::default();
                        let mut p = json!({"functionCall": {"id": id, "name": name, "args": args_object(&buf, "", &mut losses)}});
                        match sig {
                            Some(s) => p["thoughtSignature"] = s.data,
                            None if !self.called => p["thoughtSignature"] = json!(SKIP_SIGNATURE),
                            None => {}
                        }
                        self.called = true;
                        Some(p)
                    }
                    BlockStart::Text => {
                        sig.map(|s| json!({"text": "", "thoughtSignature": s.data}))
                    }
                    BlockStart::Thinking { .. } => sig
                        .map(|s| json!({"text": "", "thought": true, "thoughtSignature": s.data})),
                    BlockStart::Opaque(o) => o.for_format(F).cloned(),
                };
                if let Some(p) = part {
                    out.push(self.chunk(vec![p], None, false));
                }
            }
            StreamEvent::Usage(u) => self.usage.merge(u),
            StreamEvent::End { stop, usage } => {
                self.usage.merge(usage);
                out.push(self.chunk(vec![json!({"text": ""})], Some(finish_json(stop)), true));
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
