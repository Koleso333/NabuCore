use std::sync::Arc;

use ncore_core::Format;

use crate::hooks::{Hook, Middleware, Plugins};

/// How the core authenticates against the provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Auth {
    /// Keep whatever credentials the inbound request carries.
    Inherit,
    /// Send `name: value`, e.g. `x-api-key`.
    Header { name: String, value: String },
    /// Send `Authorization: Bearer <token>`.
    Bearer(String),
}

/// Where and how requests of one format are sent; routing decisions stay with the host.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub format: Format,
    pub base_url: String,
    pub auth: Auth,
    /// Headers added or overridden on every request.
    pub headers: Vec<(String, String)>,
    /// Plugins that apply only to this endpoint, inside the core-wide ones.
    pub plugins: Plugins,
    pub signature_namespace: Option<String>,
    /// Allow core HTTP retries and Agent restarts of broken streams.
    pub allow_replays: bool,
}

impl Endpoint {
    pub fn new(format: Format, base_url: impl Into<String>) -> Self {
        Self {
            format,
            base_url: base_url.into(),
            auth: Auth::Inherit,
            headers: Vec::new(),
            plugins: Plugins::default(),
            signature_namespace: None,
            allow_replays: true,
        }
    }

    pub fn anthropic(api_key: impl Into<String>) -> Self {
        Self::new(Format::AnthropicMessages, "https://api.anthropic.com").with_auth(Auth::Header {
            name: "x-api-key".into(),
            value: api_key.into(),
        })
    }

    pub fn openai_chat(api_key: impl Into<String>) -> Self {
        Self::new(Format::OpenAiChat, "https://api.openai.com")
            .with_auth(Auth::Bearer(api_key.into()))
    }

    pub fn openai_responses(api_key: impl Into<String>) -> Self {
        Self::new(Format::OpenAiResponses, "https://api.openai.com")
            .with_auth(Auth::Bearer(api_key.into()))
    }

    pub fn gemini(api_key: impl Into<String>) -> Self {
        Self::new(Format::Gemini, "https://generativelanguage.googleapis.com").with_auth(
            Auth::Header {
                name: "x-goog-api-key".into(),
                value: api_key.into(),
            },
        )
    }

    pub fn with_signature_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.signature_namespace = Some(namespace.into());
        self
    }

    /// Disables automatic core retries and Agent stream restarts.
    pub fn without_replays(mut self) -> Self {
        self.allow_replays = false;
        self
    }

    pub fn with_auth(mut self, auth: Auth) -> Self {
        self.auth = auth;
        self
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn with_hook(mut self, h: Arc<dyn Hook>) -> Self {
        self.plugins.hook(h);
        self
    }

    pub fn with_middleware(mut self, m: Arc<dyn Middleware>) -> Self {
        self.plugins.middleware(m);
        self
    }

    /// Registers a plugin that is both a hook and a middleware.
    pub fn with_plugin<P: Hook + Middleware + 'static>(self, p: Arc<P>) -> Self {
        self.with_hook(p.clone()).with_middleware(p)
    }
}
