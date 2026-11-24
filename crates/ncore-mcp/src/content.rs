use base64::Engine;
use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD};
use ncore::core::ir::{Block, Media, MediaSource};
use serde_json::{Value, json};

/// One item of a tool result, as MCP defines it.
#[derive(Debug, Clone, PartialEq)]
pub enum Content {
    Text(String),
    Image {
        data: String,
        mime: Option<String>,
    },
    Audio {
        data: String,
        mime: Option<String>,
    },
    ResourceLink {
        uri: String,
        name: Option<String>,
        description: Option<String>,
        mime: Option<String>,
    },
    Resource {
        uri: String,
        mime: Option<String>,
        text: Option<String>,
        blob: Option<String>,
    },
    /// Content type this client does not know; kept as sent.
    Other(Value),
}

/// Result of `tools/call`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CallResult {
    pub content: Vec<Content>,
    pub structured: Option<Value>,
    pub is_error: bool,
}

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).map(str::to_owned)
}

impl Content {
    pub fn from_value(v: &Value) -> Self {
        let kind = v.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "text" => match v.get("text") {
                Some(Value::String(t)) => Self::Text(t.clone()),
                Some(other) => Self::Text(other.to_string()),
                None => Self::Other(v.clone()),
            },
            "image" | "audio" => {
                let Some(data) = s(v, "data") else {
                    return Self::Other(v.clone());
                };
                let mime = s(v, "mimeType").or_else(|| s(v, "mime_type"));
                if kind == "image" {
                    Self::Image { data, mime }
                } else {
                    Self::Audio { data, mime }
                }
            }
            "resource_link" => match s(v, "uri") {
                Some(uri) => Self::ResourceLink {
                    uri,
                    name: s(v, "name"),
                    description: s(v, "description"),
                    mime: s(v, "mimeType"),
                },
                None => Self::Other(v.clone()),
            },
            "resource" => {
                let r = &v["resource"];
                match s(r, "uri") {
                    Some(uri) => Self::Resource {
                        uri,
                        mime: s(r, "mimeType"),
                        text: s(r, "text"),
                        blob: s(r, "blob"),
                    },
                    None => Self::Other(v.clone()),
                }
            }
            _ => Self::Other(v.clone()),
        }
    }
}

impl CallResult {
    pub fn text(t: impl Into<String>) -> Self {
        Self {
            content: vec![Content::Text(t.into())],
            ..Self::default()
        }
    }

    pub fn error(t: impl Into<String>) -> Self {
        Self {
            is_error: true,
            ..Self::text(t)
        }
    }

    /// Lenient parse: a missing or odd `content` never fails the call.
    pub fn from_value(v: &Value) -> Self {
        let content = match v.get("content") {
            Some(Value::Array(items)) => items.iter().map(Content::from_value).collect(),
            Some(Value::String(t)) => vec![Content::Text(t.clone())],
            Some(Value::Object(_)) => vec![Content::from_value(&v["content"])],
            _ => Vec::new(),
        };
        let is_error = match v.get("isError") {
            Some(Value::Bool(b)) => *b,
            Some(Value::String(t)) => t == "true",
            _ => false,
        };
        Self {
            content,
            structured: v.get("structuredContent").cloned().filter(|s| !s.is_null()),
            is_error,
        }
    }
}

/// Bounds applied when a result is turned into model input.
#[derive(Debug, Clone, Copy)]
pub struct ResultLimits {
    /// Total characters of text across the whole result.
    pub max_text_bytes: usize,
    /// Decoded size of one image, audio clip or file.
    pub max_media_bytes: usize,
}

impl Default for ResultLimits {
    fn default() -> Self {
        Self {
            max_text_bytes: 400_000,
            max_media_bytes: 20 * 1024 * 1024,
        }
    }
}

/// Cuts `s` to at most `max` bytes on a char boundary.
pub fn clip(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Mime type from magic bytes.
pub fn sniff(b: &[u8]) -> Option<&'static str> {
    let riff = |tag: &[u8]| b.len() >= 12 && &b[..4] == b"RIFF" && &b[8..12] == tag;
    Some(match b {
        [0x89, b'P', b'N', b'G', ..] => "image/png",
        [0xFF, 0xD8, 0xFF, ..] => "image/jpeg",
        [b'G', b'I', b'F', b'8', ..] => "image/gif",
        _ if riff(b"WEBP") => "image/webp",
        _ if riff(b"WAVE") => "audio/wav",
        [b'%', b'P', b'D', b'F', ..] => "application/pdf",
        [b'I', b'D', b'3', ..] | [0xFF, 0xFB | 0xF3 | 0xF2, ..] => "audio/mpeg",
        [b'O', b'g', b'g', b'S', ..] => "audio/ogg",
        [b'f', b'L', b'a', b'C', ..] => "audio/flac",
        _ => return None,
    })
}

const IMAGES: [&str; 4] = ["image/png", "image/jpeg", "image/gif", "image/webp"];

// Accepts data URLs, whitespace, URL-safe alphabets and missing padding; returns standard base64.
fn decode(data: &str) -> Option<(String, Vec<u8>)> {
    let data = match data.strip_prefix("data:") {
        Some(rest) => rest.split_once(',')?.1,
        None => data,
    };
    let clean: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    if let Ok(b) = STANDARD.decode(&clean) {
        return Some((clean, b));
    }
    let b = [STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD]
        .iter()
        .find_map(|e| e.decode(&clean).ok())?;
    Some((STANDARD.encode(&b), b))
}

fn media(
    data: &str,
    declared: Option<&str>,
    filename: Option<String>,
    what: &str,
    limits: &ResultLimits,
) -> Block {
    let Some((data, bytes)) = decode(data) else {
        return Block::text(format!("[{what} omitted: data is not valid base64]"));
    };
    if bytes.len() > limits.max_media_bytes {
        return Block::text(format!(
            "[{what} omitted: {} bytes exceeds the {} byte limit]",
            bytes.len(),
            limits.max_media_bytes
        ));
    }
    let declared = declared.map(|m| m.split(';').next().unwrap_or(m).trim().to_ascii_lowercase());
    // Providers reject a media type that does not match the bytes, so trust the bytes.
    let mime = match (sniff(&bytes), declared) {
        (Some(s), _) => s.to_owned(),
        (None, Some(d)) => d,
        (None, None) => return Block::text(format!("[{what} omitted: unknown format]")),
    };
    if mime.starts_with("image/") && !IMAGES.contains(&mime.as_str()) {
        return Block::text(format!(
            "[{what} omitted: {mime} is not supported by model providers]"
        ));
    }
    Block::Media {
        media: Media {
            mime: Some(mime),
            source: MediaSource::Base64(data),
            filename,
        },
        cache: None,
        ext: Default::default(),
    }
}

struct Texts {
    left: usize,
    cut: usize,
}

impl Texts {
    fn push(&mut self, out: &mut Vec<Block>, t: &str) {
        let kept = clip(t, self.left);
        self.left -= kept.len();
        self.cut += t.len() - kept.len();
        if !kept.is_empty() || t.is_empty() {
            out.push(Block::text(kept));
        }
    }
}

/// Converts a tool result into IR blocks for the model.
pub fn to_blocks(r: &CallResult, limits: &ResultLimits) -> Vec<Block> {
    let mut out = Vec::new();
    let mut texts = Texts {
        left: limits.max_text_bytes,
        cut: 0,
    };
    for c in &r.content {
        match c {
            Content::Text(t) => texts.push(&mut out, t),
            Content::Image { data, mime } => {
                out.push(media(data, mime.as_deref(), None, "image", limits))
            }
            Content::Audio { data, mime } => {
                out.push(media(data, mime.as_deref(), None, "audio", limits))
            }
            Content::ResourceLink {
                uri,
                name,
                description,
                mime,
            } => {
                let mut t = format!("[resource] {}", name.as_deref().unwrap_or(uri));
                if name.is_some() {
                    t.push_str(&format!(" <{uri}>"));
                }
                if let Some(m) = mime {
                    t.push_str(&format!(" ({m})"));
                }
                if let Some(d) = description {
                    t.push_str(&format!(": {d}"));
                }
                texts.push(&mut out, &t);
            }
            Content::Resource {
                uri,
                mime,
                text,
                blob,
            } => match (text, blob) {
                (Some(t), _) => texts.push(&mut out, &format!("[resource {uri}]\n{t}")),
                (None, Some(b)) => {
                    let name = uri
                        .rsplit('/')
                        .next()
                        .filter(|n| !n.is_empty())
                        .map(str::to_owned);
                    let m = mime.as_deref().unwrap_or("");
                    if m.starts_with("text/") || m.ends_with("json") || m.ends_with("xml") {
                        match decode(b).and_then(|(_, b)| String::from_utf8(b).ok()) {
                            Some(t) => texts.push(&mut out, &format!("[resource {uri}]\n{t}")),
                            None => texts.push(
                                &mut out,
                                &format!("[resource {uri} omitted: undecodable text]"),
                            ),
                        }
                    } else {
                        out.push(media(
                            b,
                            mime.as_deref(),
                            name,
                            &format!("resource {uri}"),
                            limits,
                        ));
                    }
                }
                (None, None) => texts.push(&mut out, &format!("[resource {uri}]")),
            },
            Content::Other(v) => texts.push(&mut out, &v.to_string()),
        }
    }
    if r.content.is_empty()
        && let Some(sv) = &r.structured
    {
        texts.push(&mut out, &sv.to_string());
    }
    if texts.cut > 0 {
        out.push(Block::text(format!(
            "\n[… {} bytes of output truncated]",
            texts.cut
        )));
    }
    if out.is_empty() {
        out.push(Block::text(if r.is_error {
            "tool failed without a message"
        } else {
            "(no output)"
        }));
    }
    out
}

/// Inverse of `to_blocks` for in-process providers that build IR directly.
pub fn result_value(r: &CallResult) -> Value {
    let content: Vec<Value> = r
        .content
        .iter()
        .map(|c| match c {
            Content::Text(t) => json!({"type": "text", "text": t}),
            Content::Image { data, mime } => json!({"type": "image", "data": data, "mimeType": mime}),
            Content::Audio { data, mime } => json!({"type": "audio", "data": data, "mimeType": mime}),
            Content::ResourceLink { uri, name, description, mime } => json!({
                "type": "resource_link", "uri": uri, "name": name, "description": description, "mimeType": mime
            }),
            Content::Resource { uri, mime, text, blob } => json!({
                "type": "resource", "resource": {"uri": uri, "mimeType": mime, "text": text, "blob": blob}
            }),
            Content::Other(v) => v.clone(),
        })
        .collect();
    json!({"content": content, "isError": r.is_error})
}

#[cfg(test)]
mod tests {
    use super::*;

    const PNG: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0];

    fn mime_of(b: &Block) -> Option<&str> {
        match b {
            Block::Media { media, .. } => media.mime.as_deref(),
            _ => None,
        }
    }

    #[test]
    fn media_is_repaired_or_explained() {
        let png = STANDARD.encode(PNG);
        let r = CallResult::from_value(&json!({"content": [
            {"type": "image", "data": png, "mimeType": "image/jpeg"},
            {"type": "image", "data": format!("data:image/png;base64,{}", URL_SAFE_NO_PAD.encode(PNG))},
            {"type": "image", "data": "!!!not base64", "mimeType": "image/png"},
            {"type": "image", "data": STANDARD.encode(b"BMxx"), "mimeType": "image/bmp"},
            {"type": "audio", "data": STANDARD.encode(b"RIFF\0\0\0\0WAVEfmt "), "mimeType": "audio/x-wav"},
            {"type": "resource", "resource": {"uri": "file:///a/b.png", "blob": png}},
            {"type": "resource", "resource": {"uri": "file:///a/c.txt", "mimeType": "text/plain", "blob": STANDARD.encode("hi")}},
            {"type": "resource_link", "uri": "file:///x", "name": "x", "mimeType": "text/plain"},
            {"type": "hologram", "data": 1}
        ]}));
        let b = to_blocks(&r, &ResultLimits::default());
        assert_eq!(mime_of(&b[0]), Some("image/png"));
        assert_eq!(mime_of(&b[1]), Some("image/png"));
        assert!(
            matches!(&b[1], Block::Media { media: Media { source: MediaSource::Base64(d), .. }, .. } if *d == png)
        );
        assert!(matches!(&b[2], Block::Text { text, .. } if text.contains("not valid base64")));
        assert!(matches!(&b[3], Block::Text { text, .. } if text.contains("image/bmp")));
        assert_eq!(mime_of(&b[4]), Some("audio/wav"));
        assert!(
            matches!(&b[5], Block::Media { media: Media { filename: Some(f), .. }, .. } if f == "b.png")
        );
        assert!(matches!(&b[6], Block::Text { text, .. } if text.ends_with("\nhi")));
        assert!(
            matches!(&b[7], Block::Text { text, .. } if text == "[resource] x <file:///x> (text/plain)")
        );
        assert!(matches!(&b[8], Block::Text { text, .. } if text.contains("hologram")));
    }

    #[test]
    fn text_budget_and_fallbacks() {
        let r = CallResult::from_value(&json!({"content": [
            {"type": "text", "text": "ab€cd"}, {"type": "text", "text": "zzz"}
        ], "isError": "true"}));
        assert!(r.is_error);
        let b = to_blocks(
            &r,
            &ResultLimits {
                max_text_bytes: 4,
                ..Default::default()
            },
        );
        assert!(matches!(&b[0], Block::Text { text, .. } if text == "ab"));
        assert!(matches!(b.last(), Some(Block::Text { text, .. }) if text.contains("6 bytes")));

        let r = CallResult::from_value(&json!({"structuredContent": {"t": 1}}));
        assert!(
            matches!(&to_blocks(&r, &ResultLimits::default())[0], Block::Text { text, .. } if text == r#"{"t":1}"#)
        );
        let r = CallResult::from_value(&json!({"content": "plain"}));
        assert_eq!(r.content, [Content::Text("plain".into())]);
        assert!(
            matches!(&to_blocks(&CallResult::default(), &ResultLimits::default())[0], Block::Text { text, .. } if text == "(no output)")
        );
    }
}
