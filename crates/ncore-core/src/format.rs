use crate::json;

/// Provider wire format of request and response bodies.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Format {
    AnthropicMessages,
    OpenAiChat,
    OpenAiResponses,
    Gemini,
}

impl Format {
    /// Guesses the format from a request path such as `/v1/messages`.
    pub fn detect(path: &str) -> Option<Self> {
        let path = path.split('?').next().unwrap_or(path).trim_end_matches('/');
        if path.ends_with("/messages") {
            Some(Self::AnthropicMessages)
        } else if path.ends_with("/chat/completions") {
            Some(Self::OpenAiChat)
        } else if path.ends_with("/responses") {
            Some(Self::OpenAiResponses)
        } else if path.ends_with(":generateContent") || path.ends_with(":streamGenerateContent") {
            Some(Self::Gemini)
        } else {
            None
        }
    }

    /// Whether the request asks for a streamed response.
    pub fn is_streaming(self, path: &str, body: &[u8]) -> bool {
        match self {
            Self::Gemini => path
                .split('?')
                .next()
                .unwrap_or(path)
                .ends_with(":streamGenerateContent"),
            _ => json::get_bool(body, "/stream").unwrap_or(false),
        }
    }

    /// Body pointer holding the model id; Gemini keeps it in the URL path instead.
    pub fn request_model_pointer(self) -> Option<&'static str> {
        match self {
            Self::Gemini => None,
            _ => Some("/model"),
        }
    }

    /// Pointers that may hold the model id in a response body or stream event.
    pub fn response_model_pointers(self) -> &'static [&'static str] {
        match self {
            Self::AnthropicMessages => &["/model", "/message/model"],
            Self::OpenAiChat => &["/model"],
            Self::OpenAiResponses => &["/model", "/response/model"],
            Self::Gemini => &["/modelVersion"],
        }
    }
}

/// Replaces the model segment in a Gemini path like `/v1beta/models/x:generateContent`.
pub fn replace_gemini_model(path_and_query: &str, model: &str) -> Option<String> {
    let start = path_and_query.find("/models/")? + "/models/".len();
    let end = start + path_and_query[start..].find(':')?;
    Some(format!(
        "{}{}{}",
        &path_and_query[..start],
        model,
        &path_and_query[end..]
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_formats() {
        assert_eq!(
            Format::detect("/v1/messages"),
            Some(Format::AnthropicMessages)
        );
        assert_eq!(
            Format::detect("/v1/chat/completions"),
            Some(Format::OpenAiChat)
        );
        assert_eq!(
            Format::detect("/v1/responses"),
            Some(Format::OpenAiResponses)
        );
        assert_eq!(
            Format::detect("/v1beta/models/gemini-x:streamGenerateContent?alt=sse"),
            Some(Format::Gemini)
        );
        assert_eq!(Format::detect("/v1/models"), None);
    }

    #[test]
    fn streaming_flag() {
        assert!(Format::AnthropicMessages.is_streaming("/v1/messages", br#"{"stream":true}"#));
        assert!(!Format::OpenAiChat.is_streaming("/v1/chat/completions", br#"{"stream":false}"#));
        assert!(
            Format::Gemini.is_streaming("/v1beta/models/m:streamGenerateContent?alt=sse", b"{}")
        );
    }

    #[test]
    fn rewrites_gemini_model() {
        assert_eq!(
            replace_gemini_model("/v1beta/models/a:streamGenerateContent?alt=sse", "b").as_deref(),
            Some("/v1beta/models/b:streamGenerateContent?alt=sse")
        );
    }
}
