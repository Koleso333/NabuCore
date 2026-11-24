use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{Format, ProviderError, Usage};

pub type Fields = Map<String, Value>;

/// Provider fields the IR has no slot for; restored only when encoding to the same format.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Ext {
    pub format: Option<Format>,
    pub fields: Fields,
}

impl Ext {
    pub fn new(format: Format, fields: Fields) -> Self {
        if fields.is_empty() {
            return Self::default();
        }
        Self {
            format: Some(format),
            fields,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// Provider value only the same provider can interpret, e.g. a reasoning signature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Opaque {
    pub format: Format,
    pub data: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

impl Opaque {
    pub fn new(format: Format, data: Value) -> Self {
        Self {
            format,
            data,
            namespace: None,
        }
    }

    pub fn signature(format: Format, data: Value) -> Self {
        let text = data
            .as_str()
            .or_else(|| data.get("encrypted_content").and_then(Value::as_str));
        if let Some(encoded) = text.and_then(|s| s.strip_prefix("nabu-signature-v1:"))
            && let Ok(sig) = serde_json::from_str::<Self>(encoded)
            && sig.namespace.is_some()
        {
            return sig;
        }
        Self::new(format, data)
    }

    pub fn tunnel(&self, namespace: &str, client: Format) -> Self {
        let mut payload = self.clone();
        payload.namespace = Some(namespace.into());
        let text = format!(
            "nabu-signature-v1:{}",
            serde_json::to_string(&payload).expect("signature")
        );
        let data = if client == Format::OpenAiResponses {
            serde_json::json!({"type": "reasoning", "encrypted_content": text, "summary": []})
        } else {
            Value::String(text)
        };
        Self::new(client, data)
    }

    /// Data when it belongs to `format`.
    pub fn for_format(&self, format: Format) -> Option<&Value> {
        (self.format == format && self.namespace.is_none()).then_some(&self.data)
    }
}

/// Prompt-cache breakpoint at the end of a block.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheMark {
    /// Provider TTL such as `5m`, `1h` or `30m`; `None` is the provider default.
    pub ttl: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    User,
    Assistant,
    /// Instruction in the middle of the conversation (OpenAI system/developer).
    System,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Block>,
    #[serde(default, skip_serializing_if = "Ext::is_empty")]
    pub ext: Ext,
}

impl Message {
    pub fn new(role: Role, content: Vec<Block>) -> Self {
        Self {
            role,
            content,
            ext: Ext::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MediaSource {
    Base64(String),
    Url(String),
    FileId(String),
}

/// Image, document or audio input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Media {
    pub mime: Option<String>,
    pub source: MediaSource,
    pub filename: Option<String>,
}

impl Media {
    pub fn is_image(&self) -> bool {
        self.mime
            .as_deref()
            .is_some_and(|m| m.starts_with("image/"))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Arguments as raw JSON text, kept verbatim.
    pub args: String,
    pub sig: Option<Opaque>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    /// Tool name when the source format carries it.
    pub name: Option<String>,
    pub content: Vec<Block>,
    pub is_error: bool,
    pub cache: Option<CacheMark>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Thinking {
    /// Visible reasoning or summary text; empty when redacted.
    pub text: String,
    pub sig: Option<Opaque>,
    pub redacted: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Block {
    Text {
        text: String,
        cache: Option<CacheMark>,
        /// Gemini may sign plain text parts.
        sig: Option<Opaque>,
        #[serde(default, skip_serializing_if = "Ext::is_empty")]
        ext: Ext,
    },
    Media {
        media: Media,
        cache: Option<CacheMark>,
        #[serde(default, skip_serializing_if = "Ext::is_empty")]
        ext: Ext,
    },
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    Thinking(Thinking),
    /// Provider block with no IR counterpart, e.g. server tool results.
    Opaque(Opaque),
}

impl Block {
    pub fn text(t: impl Into<String>) -> Self {
        Block::Text {
            text: t.into(),
            cache: None,
            sig: None,
            ext: Ext::default(),
        }
    }

    pub fn cache(&self) -> Option<&CacheMark> {
        match self {
            Block::Text { cache, .. } | Block::Media { cache, .. } => cache.as_ref(),
            Block::ToolResult(r) => r.cache.as_ref(),
            _ => None,
        }
    }
}

/// Concatenated text of text blocks.
pub fn plain_text(blocks: &[Block]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            Block::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema of the arguments.
    pub schema: Value,
    pub strict: Option<bool>,
    pub cache: Option<CacheMark>,
    #[serde(default, skip_serializing_if = "Ext::is_empty")]
    pub ext: Ext,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToolDef {
    Function(Tool),
    /// Provider-hosted tool such as web search.
    Opaque(Opaque),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolChoice {
    Auto,
    None,
    /// Model must call some tool.
    Required,
    Tool(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningMode {
    Off,
    On,
    /// Model decides how much to think.
    Adaptive,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningCfg {
    pub mode: Option<ReasoningMode>,
    /// Lower-case effort level: minimal, low, medium, high, xhigh, max.
    pub effort: Option<String>,
    pub budget_tokens: Option<u64>,
    /// Whether thoughts or summaries are returned.
    pub include: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutputFormat {
    JsonObject,
    JsonSchema {
        name: Option<String>,
        schema: Value,
        strict: Option<bool>,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub model: String,
    /// Leading system instructions as text blocks.
    pub system: Vec<Block>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub max_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u64>,
    pub stop: Vec<String>,
    pub seed: Option<i64>,
    pub reasoning: Option<ReasoningCfg>,
    pub output: Option<OutputFormat>,
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Ext::is_empty")]
    pub ext: Ext,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence(Option<String>),
    ToolUse,
    ContentFilter,
    Refusal,
    /// Server paused a long turn (Anthropic `pause_turn`).
    Pause,
    ContextWindow,
    Other(String),
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub id: Option<String>,
    pub model: Option<String>,
    pub content: Vec<Block>,
    pub stop: Option<StopReason>,
    pub usage: Usage,
    #[serde(default, skip_serializing_if = "Ext::is_empty")]
    pub ext: Ext,
}

impl Response {
    pub fn has_tool_calls(&self) -> bool {
        self.content.iter().any(|b| matches!(b, Block::ToolCall(_)))
    }
}

/// What a streamed block is when it opens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BlockStart {
    Text,
    Thinking {
        redacted: bool,
    },
    ToolCall {
        id: String,
        name: String,
    },
    /// Arrives complete; no deltas follow.
    Opaque(Opaque),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Delta {
    Text(String),
    Thinking(String),
    ToolArgs(String),
    /// Signature for the block, delivered whole.
    Sig(Opaque),
}

/// Provider-neutral stream event; block indices are dense and blocks never interleave.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StreamEvent {
    Start {
        id: Option<String>,
        model: Option<String>,
    },
    BlockStart {
        index: usize,
        block: BlockStart,
    },
    Delta {
        index: usize,
        delta: Delta,
    },
    BlockEnd {
        index: usize,
    },
    /// Cumulative usage seen so far.
    Usage(Usage),
    End {
        stop: Option<StopReason>,
        usage: Usage,
    },
    #[serde(skip)]
    Error(ProviderError),
}

/// Builds a `Response` from stream events.
#[derive(Debug, Default)]
pub struct Accumulator {
    resp: Response,
    open: Option<Block>,
    error: Option<ProviderError>,
    done: bool,
}

impl Accumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, ev: &StreamEvent) {
        match ev {
            StreamEvent::Start { id, model } => {
                if id.is_some() {
                    self.resp.id = id.clone();
                }
                if model.is_some() {
                    self.resp.model = model.clone();
                }
            }
            StreamEvent::BlockStart { block, .. } => {
                self.close();
                self.open = Some(match block {
                    BlockStart::Text => Block::text(""),
                    BlockStart::Thinking { redacted } => Block::Thinking(Thinking {
                        text: String::new(),
                        sig: None,
                        redacted: *redacted,
                    }),
                    BlockStart::ToolCall { id, name } => Block::ToolCall(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        args: String::new(),
                        sig: None,
                    }),
                    BlockStart::Opaque(o) => Block::Opaque(o.clone()),
                });
            }
            StreamEvent::Delta { delta, .. } => match (self.open.as_mut(), delta) {
                (Some(Block::Text { text, .. }), Delta::Text(t)) => text.push_str(t),
                (Some(Block::Thinking(th)), Delta::Thinking(t)) => th.text.push_str(t),
                (Some(Block::ToolCall(c)), Delta::ToolArgs(a)) => c.args.push_str(a),
                (Some(Block::Text { sig, .. }), Delta::Sig(o)) => *sig = Some(o.clone()),
                (Some(Block::Thinking(th)), Delta::Sig(o)) => th.sig = Some(o.clone()),
                (Some(Block::ToolCall(c)), Delta::Sig(o)) => c.sig = Some(o.clone()),
                _ => {}
            },
            StreamEvent::BlockEnd { .. } => self.close(),
            StreamEvent::Usage(u) => self.resp.usage.merge(u),
            StreamEvent::End { stop, usage } => {
                self.close();
                self.resp.usage.merge(usage);
                if stop.is_some() {
                    self.resp.stop = stop.clone();
                }
                self.done = true;
            }
            StreamEvent::Error(e) => self.error = Some(e.clone()),
        }
    }

    fn close(&mut self) {
        if let Some(mut b) = self.open.take() {
            if let Block::ToolCall(c) = &mut b
                && c.args.is_empty()
            {
                c.args.push_str("{}");
            }
            self.resp.content.push(b);
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn error(&self) -> Option<&ProviderError> {
        self.error.as_ref()
    }

    pub fn response(&self) -> &Response {
        &self.resp
    }

    pub fn finish(mut self) -> Response {
        self.close();
        self.resp
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossKind {
    /// Removed from the output.
    Dropped,
    /// Expressed with a near equivalent.
    Approximated,
}

/// Something the target format could not carry exactly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Loss {
    pub path: String,
    pub kind: LossKind,
    pub detail: String,
}

/// Collects losses during one conversion.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Losses(pub Vec<Loss>);

impl Losses {
    pub fn dropped(&mut self, path: impl Into<String>, detail: impl Into<String>) {
        self.0.push(Loss {
            path: path.into(),
            kind: LossKind::Dropped,
            detail: detail.into(),
        });
    }

    pub fn approx(&mut self, path: impl Into<String>, detail: impl Into<String>) {
        self.0.push(Loss {
            path: path.into(),
            kind: LossKind::Approximated,
            detail: detail.into(),
        });
    }

    /// Reports extension fields that cannot travel to `target`.
    pub fn ext(&mut self, path: &str, ext: &Ext, target: Format) -> Option<Fields> {
        if ext.is_empty() {
            return None;
        }
        if ext.format == Some(target) {
            return Some(ext.fields.clone());
        }
        for k in ext.fields.keys() {
            self.dropped(format!("{path}/{k}"), "no equivalent in target format");
        }
        None
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
