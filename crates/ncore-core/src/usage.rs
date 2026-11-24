use serde_json::Value;

use crate::Format;

/// Token usage normalized across providers; `None` means the provider did not report it.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Usage {
    /// All prompt tokens, cached or not.
    pub input_total: Option<u64>,
    pub cache_read: Option<u64>,
    pub cache_write: Option<u64>,
    pub cache_write_5m: Option<u64>,
    pub cache_write_1h: Option<u64>,
    /// All generated tokens, reasoning included.
    pub output_total: Option<u64>,
    pub reasoning: Option<u64>,
}

fn n(v: &Value, pointer: &str) -> Option<u64> {
    v.pointer(pointer).and_then(Value::as_u64)
}

fn sum(parts: &[Option<u64>]) -> Option<u64> {
    if parts.iter().all(Option::is_none) {
        return None;
    }
    Some(parts.iter().map(|p| p.unwrap_or(0)).sum())
}

impl Usage {
    /// Parses a provider usage object (`usage` or Gemini `usageMetadata`).
    pub fn from_provider(format: Format, u: &Value) -> Self {
        match format {
            // input_tokens counts only tokens after the last cache breakpoint.
            Format::AnthropicMessages => {
                let input = n(u, "/input_tokens");
                let read = n(u, "/cache_read_input_tokens");
                let write = n(u, "/cache_creation_input_tokens");
                Self {
                    input_total: if input.is_some() {
                        sum(&[input, read, write])
                    } else {
                        None
                    },
                    cache_read: read,
                    cache_write: write,
                    cache_write_5m: n(u, "/cache_creation/ephemeral_5m_input_tokens"),
                    cache_write_1h: n(u, "/cache_creation/ephemeral_1h_input_tokens"),
                    output_total: n(u, "/output_tokens"),
                    reasoning: n(u, "/output_tokens_details/thinking_tokens"),
                }
            }
            Format::OpenAiChat => Self {
                input_total: n(u, "/prompt_tokens"),
                cache_read: n(u, "/prompt_tokens_details/cached_tokens"),
                cache_write: n(u, "/prompt_tokens_details/cache_write_tokens"),
                output_total: n(u, "/completion_tokens"),
                reasoning: n(u, "/completion_tokens_details/reasoning_tokens"),
                ..Self::default()
            },
            Format::OpenAiResponses => Self {
                input_total: n(u, "/input_tokens"),
                cache_read: n(u, "/input_tokens_details/cached_tokens"),
                cache_write: n(u, "/input_tokens_details/cache_write_tokens"),
                output_total: n(u, "/output_tokens"),
                reasoning: n(u, "/output_tokens_details/reasoning_tokens"),
                ..Self::default()
            },
            // Thoughts are reported apart from candidates and billed as output.
            Format::Gemini => {
                let candidates = n(u, "/candidatesTokenCount");
                let thoughts = n(u, "/thoughtsTokenCount");
                Self {
                    input_total: n(u, "/promptTokenCount"),
                    cache_read: n(u, "/cachedContentTokenCount"),
                    output_total: sum(&[candidates, thoughts]),
                    reasoning: thoughts,
                    ..Self::default()
                }
            }
        }
    }

    /// Overlays fields reported later in a stream; providers send cumulative values.
    pub fn merge(&mut self, later: &Usage) {
        macro_rules! take {
            ($($f:ident),*) => { $( if later.$f.is_some() { self.$f = later.$f; } )* };
        }
        take!(
            input_total,
            cache_read,
            cache_write,
            cache_write_5m,
            cache_write_1h,
            output_total,
            reasoning
        );
    }

    /// Adds another request's usage, e.g. to total a multi-step run.
    pub fn add(&mut self, other: &Usage) {
        macro_rules! add {
            ($($f:ident),*) => { $( if let Some(v) = other.$f { self.$f = Some(self.$f.unwrap_or(0) + v); } )* };
        }
        add!(
            input_total,
            cache_read,
            cache_write,
            cache_write_5m,
            cache_write_1h,
            output_total,
            reasoning
        );
    }

    pub fn is_empty(&self) -> bool {
        *self == Usage::default()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn anthropic_input_total_includes_cache() {
        let u = Usage::from_provider(
            Format::AnthropicMessages,
            &json!({"input_tokens": 10, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 5,
                    "cache_creation": {"ephemeral_5m_input_tokens": 5, "ephemeral_1h_input_tokens": 0},
                    "output_tokens": 7}),
        );
        assert_eq!(u.input_total, Some(115));
        assert_eq!(u.cache_read, Some(100));
        assert_eq!(u.cache_write_5m, Some(5));
        assert_eq!(u.output_total, Some(7));
    }

    #[test]
    fn anthropic_stream_merge() {
        let mut u = Usage::from_provider(
            Format::AnthropicMessages,
            &json!({"input_tokens": 25, "output_tokens": 1}),
        );
        u.merge(&Usage::from_provider(
            Format::AnthropicMessages,
            &json!({"output_tokens": 15}),
        ));
        assert_eq!(u.input_total, Some(25));
        assert_eq!(u.output_total, Some(15));
    }

    #[test]
    fn openai_and_gemini() {
        let c = Usage::from_provider(
            Format::OpenAiChat,
            &json!({"prompt_tokens": 50, "completion_tokens": 9, "total_tokens": 59,
                    "prompt_tokens_details": {"cached_tokens": 32}, "completion_tokens_details": {"reasoning_tokens": 4}}),
        );
        assert_eq!(
            (c.input_total, c.cache_read, c.reasoning),
            (Some(50), Some(32), Some(4))
        );
        let g = Usage::from_provider(
            Format::Gemini,
            &json!({"promptTokenCount": 20, "candidatesTokenCount": 8, "thoughtsTokenCount": 30}),
        );
        assert_eq!((g.output_total, g.reasoning), (Some(38), Some(30)));
    }
}
