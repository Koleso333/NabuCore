use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::codec::{CodecError, EncodeOptions};
use crate::ir::{Block, CacheMark, Losses, Request, Role, ToolDef};
use crate::{Format, Usage};

/// Breakpoints Anthropic accepts in one request.
pub const ANTHROPIC_BREAKPOINTS: usize = 4;
/// Positions Anthropic checks backwards from each breakpoint, the breakpoint included.
pub const ANTHROPIC_LOOKBACK: usize = 20;
// OpenAI's implicit breakpoint takes one of the four cache writes.
const OPENAI_BREAKPOINTS: usize = 3;

/// Where the planner adds breakpoints; host marks are kept and count against the limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachePolicy {
    /// Last tool definition.
    pub tools: bool,
    /// End of the system prompt, or the last tool when there is no system prompt.
    pub system: bool,
    /// Last cacheable block of the conversation.
    pub tail: bool,
    /// Extra breakpoints so a long stretch since the previous one stays inside the lookback window.
    pub bridge: bool,
    /// TTL of added marks, e.g. `1h`; `None` is the provider default.
    pub ttl: Option<String>,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self {
            tools: true,
            system: true,
            tail: true,
            bridge: true,
            ttl: None,
        }
    }
}

impl CachePolicy {
    /// Adds nothing; host marks are still checked against provider limits.
    pub fn none() -> Self {
        Self {
            tools: false,
            system: false,
            tail: false,
            bridge: false,
            ttl: None,
        }
    }
}

/// Marks the planner added, as request paths such as `/messages/3/content/1`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CachePlan {
    pub added: Vec<String>,
}

// Mark position in prompt order: tools, then system, then messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Slot {
    Tool(usize),
    System(usize),
    Msg(usize, usize),
}

impl Slot {
    fn path(self) -> String {
        match self {
            Slot::Tool(i) => format!("/tools/{i}"),
            Slot::System(i) => format!("/system/{i}"),
            Slot::Msg(m, b) => format!("/messages/{m}/content/{b}"),
        }
    }
}

fn block_mark(b: &mut Block) -> Option<&mut Option<CacheMark>> {
    match b {
        Block::Text { cache, .. } | Block::Media { cache, .. } => Some(cache),
        Block::ToolResult(r) => Some(&mut r.cache),
        _ => None,
    }
}

fn mark_of(req: &mut Request, slot: Slot) -> Option<&mut Option<CacheMark>> {
    match slot {
        Slot::Tool(i) => match req.tools.get_mut(i)? {
            ToolDef::Function(t) => Some(&mut t.cache),
            ToolDef::Opaque(_) => None,
        },
        Slot::System(i) => block_mark(req.system.get_mut(i)?),
        Slot::Msg(m, b) => block_mark(req.messages.get_mut(m)?.content.get_mut(b)?),
    }
}

fn marked(req: &Request) -> Vec<Slot> {
    let mut out = Vec::new();
    for (i, t) in req.tools.iter().enumerate() {
        if matches!(t, ToolDef::Function(t) if t.cache.is_some()) {
            out.push(Slot::Tool(i));
        }
    }
    for (i, b) in req.system.iter().enumerate() {
        if b.cache().is_some() {
            out.push(Slot::System(i));
        }
    }
    for (m, msg) in req.messages.iter().enumerate() {
        for (i, b) in msg.content.iter().enumerate() {
            if b.cache().is_some() {
                out.push(Slot::Msg(m, i));
            }
        }
    }
    out
}

/// Adds breakpoints for the target format and fixes host marks the provider would reject.
pub fn plan(
    format: Format,
    req: &mut Request,
    policy: &CachePolicy,
    opts: &EncodeOptions,
    losses: &mut Losses,
) -> CachePlan {
    match format {
        Format::AnthropicMessages => plan_anthropic(req, policy, losses),
        // OpenAI caches the prefix by itself; explicit marks only need the write limit.
        Format::OpenAiChat | Format::OpenAiResponses if opts.openai_cache_breakpoints => {
            let host = marked(req);
            cap(req, host, OPENAI_BREAKPOINTS, losses);
            CachePlan::default()
        }
        // Gemini has no breakpoints; the codec reports host marks as losses.
        _ => CachePlan::default(),
    }
}

/// Keeps the first `max - 1` marks and the last one.
fn cap(req: &mut Request, host: Vec<Slot>, max: usize, losses: &mut Losses) -> Vec<Slot> {
    if host.len() <= max {
        return host;
    }
    let last = host.len() - 1;
    let mut kept = Vec::new();
    for (i, slot) in host.into_iter().enumerate() {
        if i + 1 < max || i == last {
            kept.push(slot);
        } else if let Some(m) = mark_of(req, slot) {
            *m = None;
            losses.dropped(
                slot.path() + "/cache",
                format!("more than {max} cache breakpoints"),
            );
        }
    }
    kept
}

// Whether the Anthropic encoder sends the block, so it takes a lookback position.
fn sent(b: &Block) -> bool {
    match b {
        Block::Text { text, .. } => !text.is_empty(),
        Block::Media { media, .. } => {
            media.is_image() || media.mime.as_deref() == Some("application/pdf")
        }
        Block::Thinking(t) => t
            .sig
            .as_ref()
            .is_some_and(|s| s.format == Format::AnthropicMessages),
        Block::Opaque(o) => o.format == Format::AnthropicMessages,
        Block::ToolCall(_) | Block::ToolResult(_) => true,
    }
}

// Blocks that can carry `cache_control`; tool_use can too, but the IR has no mark for it.
fn cacheable(b: &Block, role: Role) -> bool {
    match b {
        Block::Text { .. } | Block::ToolResult(_) => sent(b),
        Block::Media { .. } => role != Role::Assistant && sent(b),
        _ => false,
    }
}

// One lookback position of the prompt and the slot that marks it for the planner.
struct Pos {
    pos: usize,
    slot: Option<Slot>,
}

/// Lookback positions in prompt order (a tool_use or tool_result run is one) and each slot's position.
fn positions(req: &Request) -> (Vec<Pos>, HashMap<Slot, usize>) {
    let mut out = Vec::new();
    let mut at = HashMap::new();
    let mut pos = 0;
    for (i, t) in req.tools.iter().enumerate() {
        let slot = matches!(t, ToolDef::Function(_)).then_some(Slot::Tool(i));
        at.extend(slot.map(|s| (s, pos)));
        out.push(Pos { pos, slot });
        pos += 1;
    }
    for (i, b) in req.system.iter().enumerate() {
        if matches!(b, Block::Text { text, .. } if !text.is_empty()) {
            at.insert(Slot::System(i), pos);
            out.push(Pos {
                pos,
                slot: Some(Slot::System(i)),
            });
            pos += 1;
        }
    }
    // 0: other, 1: tool_use run, 2: tool_result run.
    let mut run = 0;
    for (m, msg) in req.messages.iter().enumerate() {
        for (i, b) in msg.content.iter().enumerate() {
            if !sent(b) {
                continue;
            }
            let slot = cacheable(b, msg.role).then_some(Slot::Msg(m, i));
            let kind = match b {
                Block::ToolCall(_) => 1,
                Block::ToolResult(_) => 2,
                _ => 0,
            };
            if kind != 0 && kind == run {
                // Same position; the planner marks the end of the run.
                at.extend(slot.map(|s| (s, pos - 1)));
                if let Some(last) = out.last_mut() {
                    last.slot = slot;
                }
                continue;
            }
            run = kind;
            at.extend(slot.map(|s| (s, pos)));
            out.push(Pos { pos, slot });
            pos += 1;
        }
    }
    (out, at)
}

fn plan_anthropic(req: &mut Request, policy: &CachePolicy, losses: &mut Losses) -> CachePlan {
    let (all, at) = positions(req);
    // Marks on blocks the encoder drops never reach the provider.
    let host: Vec<Slot> = marked(req)
        .into_iter()
        .filter(|s| at.contains_key(s))
        .collect();
    let mut marks = cap(req, host, ANTHROPIC_BREAKPOINTS, losses);
    let mut added = Vec::new();
    let pos_of = |s: Slot| at.get(&s).copied();

    let last_tool = all.iter().rev().find_map(|p| match p.slot {
        Some(s @ Slot::Tool(_)) => Some(s),
        _ => None,
    });
    let last_system = all.iter().rev().find_map(|p| match p.slot {
        Some(s @ Slot::System(_)) => Some(s),
        _ => None,
    });
    let tail = all.iter().rev().find_map(|p| match p.slot {
        Some(s @ Slot::Msg(..)) => Some(s),
        _ => None,
    });

    let place = |s: Slot, marks: &mut Vec<Slot>, added: &mut Vec<Slot>| {
        if !marks.contains(&s) && marks.len() < ANTHROPIC_BREAKPOINTS {
            marks.push(s);
            marks.sort();
            added.push(s);
        }
    };

    if policy.tail
        && let Some(s) = tail
    {
        place(s, &mut marks, &mut added);
    }
    let stable = if policy.system {
        last_system.or(last_tool)
    } else {
        None
    };
    if let Some(s) = stable {
        place(s, &mut marks, &mut added);
    }

    // A mark reads back 20 positions; a wider gap gets a bridge low enough to keep windows touching.
    let bridge = |marks: &[Slot]| -> Option<Slot> {
        let mut ps: Vec<usize> = marks.iter().filter_map(|s| pos_of(*s)).collect();
        ps.sort();
        for w in (0..ps.len()).rev() {
            let hi = ps[w];
            let lo = w.checked_sub(1).map(|i| ps[i]);
            let reach = hi.saturating_sub(ANTHROPIC_LOOKBACK);
            let covered = match lo {
                Some(lo) => lo >= reach,
                None => hi < ANTHROPIC_LOOKBACK,
            };
            if covered {
                continue;
            }
            let found = all
                .iter()
                .filter(|p| p.pos >= reach && p.pos < hi && lo.is_none_or(|lo| p.pos > lo))
                .find_map(|p| p.slot.filter(|s| matches!(s, Slot::Msg(..))));
            if found.is_some() {
                return found;
            }
        }
        None
    };

    if policy.bridge
        && let Some(s) = bridge(&marks)
    {
        place(s, &mut marks, &mut added);
    }
    if policy.tools
        && let Some(s) = last_tool
    {
        place(s, &mut marks, &mut added);
    }
    while policy.bridge && marks.len() < ANTHROPIC_BREAKPOINTS {
        match bridge(&marks) {
            Some(s) if !marks.contains(&s) => place(s, &mut marks, &mut added),
            _ => break,
        }
    }

    for s in &added {
        if let Some(m) = mark_of(req, *s) {
            *m = Some(CacheMark {
                ttl: policy.ttl.clone(),
            });
        }
    }
    order_ttls(req, &marks, losses);
    added.sort();
    CachePlan {
        added: added.into_iter().map(Slot::path).collect(),
    }
}

fn long_ttl(m: &CacheMark) -> bool {
    m.ttl.as_deref() == Some("1h")
}

/// Anthropic rejects a 1h entry after a 5m one; later 1h marks are lowered.
fn order_ttls(req: &mut Request, marks: &[Slot], losses: &mut Losses) {
    let mut short_seen = false;
    for s in marks {
        let Some(Some(m)) = mark_of(req, *s) else {
            continue;
        };
        if !long_ttl(m) {
            short_seen = true;
        } else if short_seen {
            m.ttl = None;
            losses.approx(
                s.path() + "/cache/ttl",
                "1h cache mark after a 5m one lowered to 5m",
            );
        }
    }
}

// ---- tracking ----

/// Hash of one prompt unit as the provider sees it, cache markers removed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment {
    /// `settings/<field>`, `tools/<i>`, `system/...` or a message path.
    pub label: String,
    pub hash: u64,
}

/// Cache-relevant shape of an encoded request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    pub format: Format,
    pub model: String,
    pub segments: Vec<Segment>,
}

// Top-level fields that do not affect prompt caching.
const NEUTRAL: &[&str] = &[
    "model",
    "stream",
    "stream_options",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop",
    "stop_sequences",
    "seed",
    "metadata",
    "user",
    "n",
    "store",
    "safety_identifier",
];

const GEMINI_NEUTRAL: &[&str] = &[
    "temperature",
    "topP",
    "topK",
    "maxOutputTokens",
    "stopSequences",
    "seed",
    "candidateCount",
];

const MARKERS: &[&str] = &["cache_control", "prompt_cache_breakpoint"];

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn hash(v: &Value) -> u64 {
    fnv(&serde_json::to_vec(v).expect("serializable"))
}

/// Text content as blocks, so string and one-block forms hash alike.
fn as_blocks(v: &Value, text_type: &str) -> Value {
    match v {
        Value::String(s) => Value::Array(vec![serde_json::json!({"type": text_type, "text": s})]),
        other => other.clone(),
    }
}

// Removes cache markers from a block and from the blocks nested in it.
fn unmark(v: &mut Value) {
    let Some(o) = v.as_object_mut() else { return };
    for k in MARKERS {
        o.remove(*k);
    }
    for key in ["content", "output"] {
        if let Some(Value::Array(items)) = o.get_mut(key) {
            for it in items {
                if let Some(io) = it.as_object_mut() {
                    for k in MARKERS {
                        io.remove(*k);
                    }
                }
            }
        }
    }
}

// OpenAI carries the system prompt as messages; their label says so.
fn system_prefix(msg: &Value) -> &'static str {
    match msg.get("role").and_then(Value::as_str) {
        Some("system" | "developer") => "system/",
        _ => "",
    }
}

struct Segs(Vec<Segment>);

impl Segs {
    fn push(&mut self, label: String, mut v: Value) {
        unmark(&mut v);
        self.0.push(Segment {
            label,
            hash: hash(&v),
        });
    }
}

/// Fingerprint of an encoded request body; Gemini keeps the model in the URL, so it is passed in.
pub fn fingerprint(format: Format, model: &str, body: &[u8]) -> Result<Fingerprint, CodecError> {
    let v: Value = serde_json::from_slice(body)?;
    let o = v
        .as_object()
        .ok_or_else(|| CodecError("request body is not an object".into()))?;
    let content: &[&str] = match format {
        Format::AnthropicMessages => &["tools", "system", "messages"],
        Format::OpenAiChat => &["tools", "messages"],
        Format::OpenAiResponses => &["tools", "instructions", "input"],
        Format::Gemini => &["tools", "systemInstruction", "contents", "generationConfig"],
    };
    let mut segs = Segs(Vec::new());

    let mut keys: Vec<&String> = o
        .keys()
        .filter(|k| !content.contains(&k.as_str()) && !NEUTRAL.contains(&k.as_str()))
        .collect();
    keys.sort();
    for k in keys {
        segs.push(format!("settings/{k}"), o[k].clone());
    }
    if format == Format::Gemini
        && let Some(Value::Object(g)) = o.get("generationConfig")
    {
        let rest: Map<String, Value> = g
            .iter()
            .filter(|(k, _)| !GEMINI_NEUTRAL.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if !rest.is_empty() {
            segs.push("settings/generationConfig".into(), Value::Object(rest));
        }
    }

    let array = |key: &str| -> &[Value] {
        o.get(key)
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
    };
    for (i, t) in array("tools").iter().enumerate() {
        segs.push(format!("tools/{i}"), t.clone());
    }

    match format {
        Format::AnthropicMessages => {
            if let Some(Value::Array(sys)) = o.get("system").map(|s| as_blocks(s, "text")) {
                for (i, b) in sys.into_iter().enumerate() {
                    segs.push(format!("system/{i}"), b);
                }
            }
            for (m, msg) in array("messages").iter().enumerate() {
                let role = msg.get("role").cloned().unwrap_or(Value::Null);
                let blocks = msg.get("content").map(|c| as_blocks(c, "text"));
                for (i, mut b) in blocks
                    .and_then(|b| b.as_array().cloned())
                    .unwrap_or_default()
                    .into_iter()
                    .enumerate()
                {
                    if let Some(c) = b.get("content").filter(|c| c.is_string()) {
                        b["content"] = as_blocks(c, "text");
                    }
                    unmark(&mut b);
                    segs.push(
                        format!("messages/{m}/content/{i}"),
                        serde_json::json!([role, b]),
                    );
                }
            }
        }
        Format::OpenAiChat => {
            for (m, msg) in array("messages").iter().enumerate() {
                let mut msg = msg.clone();
                if let Some(c) = msg.get("content").filter(|c| c.is_string()) {
                    msg["content"] = as_blocks(c, "text");
                }
                segs.push(format!("{}messages/{m}", system_prefix(&msg)), msg);
            }
        }
        Format::OpenAiResponses => {
            if let Some(i) = o.get("instructions") {
                segs.push("system/instructions".into(), i.clone());
            }
            for (n, item) in array("input").iter().enumerate() {
                let mut item = item.clone();
                if let Some(c) = item.get("content").filter(|c| c.is_string()) {
                    item["content"] = as_blocks(c, "input_text");
                }
                segs.push(format!("{}input/{n}", system_prefix(&item)), item);
            }
        }
        Format::Gemini => {
            if let Some(parts) = o
                .get("systemInstruction")
                .and_then(|s| s.get("parts"))
                .and_then(Value::as_array)
            {
                for (i, p) in parts.iter().enumerate() {
                    segs.push(format!("system/{i}"), p.clone());
                }
            }
            for (c, content) in array("contents").iter().enumerate() {
                let role = content.get("role").cloned().unwrap_or(Value::Null);
                for (i, p) in content
                    .get("parts")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .enumerate()
                {
                    segs.push(
                        format!("contents/{c}/parts/{i}"),
                        serde_json::json!([role, p]),
                    );
                }
            }
        }
    }
    Ok(Fingerprint {
        format,
        model: model.to_owned(),
        segments: segs.0,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BreakReason {
    /// Another provider format.
    Format,
    Model,
    /// A request setting such as `tool_choice` or thinking changed.
    Settings,
    Tools,
    System,
    /// Earlier messages were edited, removed or reordered.
    History,
}

fn reason_of(label: &str) -> BreakReason {
    if label.starts_with("settings/") {
        BreakReason::Settings
    } else if label.starts_with("tools/") {
        BreakReason::Tools
    } else if label.starts_with("system/") {
        BreakReason::System
    } else {
        BreakReason::History
    }
}

/// The new request does not extend the previous one, so cached prefixes past `index` are lost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheBreak {
    pub reason: BreakReason,
    /// Label of the first differing segment, from the new request when it has one.
    pub segment: String,
    /// Segments still shared with the previous request.
    pub index: usize,
    /// Segments of the previous request.
    pub previous: usize,
}

/// The prefix was unchanged and caching worked before, yet nothing was read from cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheMiss {
    /// Segments shared with the previous request.
    pub shared: usize,
}

/// Remembers the last successful request of a conversation; serializable for the host to keep.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheTracker {
    last: Option<Fingerprint>,
    /// The last response read or wrote cache.
    warm: bool,
}

impl CacheTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compares a request about to be sent with the last successful one.
    pub fn check(&self, fp: &Fingerprint) -> Option<CacheBreak> {
        let last = self.last.as_ref()?;
        let at = |reason, segment: &str| CacheBreak {
            reason,
            segment: segment.to_owned(),
            index: 0,
            previous: last.segments.len(),
        };
        if last.format != fp.format {
            return Some(at(BreakReason::Format, ""));
        }
        if last.model != fp.model {
            return Some(at(BreakReason::Model, ""));
        }
        let shared = last
            .segments
            .iter()
            .zip(&fp.segments)
            .take_while(|(a, b)| a == b)
            .count();
        if shared == last.segments.len() {
            return None;
        }
        let old = &last.segments[shared].label;
        let (segment, reason) = match fp.segments.get(shared) {
            Some(new) => (new.label.clone(), reason_of(&new.label).min(reason_of(old))),
            None => (old.clone(), reason_of(old)),
        };
        Some(CacheBreak {
            reason,
            segment,
            index: shared,
            previous: last.segments.len(),
        })
    }

    /// Records a request that got a response, with that response's usage.
    pub fn commit(&mut self, fp: Fingerprint, usage: &Usage) -> Option<CacheMiss> {
        let miss =
            (self.warm && self.check(&fp).is_none() && usage.cache_read == Some(0)).then(|| {
                CacheMiss {
                    shared: self.last.as_ref().map_or(0, |l| l.segments.len()),
                }
            });
        self.warm = usage.cache_read.unwrap_or(0) > 0 || usage.cache_write.unwrap_or(0) > 0;
        self.last = Some(fp);
        miss
    }

    /// Forgets the previous request, e.g. when the host starts a new conversation.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
