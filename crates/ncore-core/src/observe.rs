use bytes::Bytes;
use serde_json::Value;

use crate::{Format, ProviderError, SseEvent, Usage};

/// Facts extracted from a response while it is forwarded untouched.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Observation {
    pub id: Option<String>,
    pub model: Option<String>,
    pub stop_reason: Option<String>,
    pub usage: Usage,
    pub error: Option<ProviderError>,
}

/// Incremental update produced by the tap.
#[derive(Debug, Clone, PartialEq)]
pub enum Observed {
    Model(String),
    Usage(Usage),
    Stop(String),
    Error(ProviderError),
}

impl Observation {
    fn apply(&mut self, o: &Observed) {
        match o {
            Observed::Model(m) => self.model = Some(m.clone()),
            Observed::Usage(u) => self.usage.merge(u),
            Observed::Stop(s) => self.stop_reason = Some(s.clone()),
            Observed::Error(e) => self.error = Some(e.clone()),
        }
    }

    /// Observes a complete (non-streamed) response body.
    pub fn from_body(format: Format, status: u16, body: &Bytes) -> Self {
        let mut obs = Observation::default();
        if status >= 400 {
            obs.error = Some(ProviderError::parse(format, Some(status), body.clone()));
            return obs;
        }
        let Ok(v) = serde_json::from_slice::<Value>(body) else {
            return obs;
        };
        let mut out = Vec::new();
        match format {
            Format::AnthropicMessages => {
                message_facts(format, &v, &mut obs, &mut out, "/stop_reason")
            }
            Format::OpenAiChat => {
                message_facts(format, &v, &mut obs, &mut out, "");
                chat_finish(&v, &mut out);
            }
            Format::OpenAiResponses => message_facts(format, &v, &mut obs, &mut out, "/status"),
            Format::Gemini => gemini_facts(&v, &mut obs, &mut out),
        }
        for o in &out {
            obs.apply(o);
        }
        obs
    }
}

fn str_at(v: &Value, p: &str) -> Option<String> {
    v.pointer(p).and_then(Value::as_str).map(str::to_owned)
}

// Common shape: top-level id, model, usage and a stop field.
fn message_facts(
    format: Format,
    v: &Value,
    obs: &mut Observation,
    out: &mut Vec<Observed>,
    stop: &str,
) {
    if let Some(id) = str_at(v, "/id") {
        obs.id = Some(id);
    }
    if let Some(m) = str_at(v, "/model") {
        out.push(Observed::Model(m));
    }
    if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
        out.push(Observed::Usage(Usage::from_provider(format, u)));
    }
    if !stop.is_empty()
        && let Some(s) = str_at(v, stop)
    {
        out.push(Observed::Stop(s));
    }
}

fn chat_finish(v: &Value, out: &mut Vec<Observed>) {
    if let Some(choices) = v.get("choices").and_then(Value::as_array)
        && let Some(r) = choices
            .iter()
            .find_map(|c| c.get("finish_reason").and_then(Value::as_str))
    {
        out.push(Observed::Stop(r.to_owned()));
    }
}

fn gemini_facts(v: &Value, obs: &mut Observation, out: &mut Vec<Observed>) {
    if let Some(id) = str_at(v, "/responseId") {
        obs.id = Some(id);
    }
    if let Some(m) = str_at(v, "/modelVersion") {
        out.push(Observed::Model(m));
    }
    if let Some(u) = v.get("usageMetadata") {
        out.push(Observed::Usage(Usage::from_provider(Format::Gemini, u)));
    }
    if let Some(r) = str_at(v, "/candidates/0/finishReason") {
        out.push(Observed::Stop(r));
    }
}

/// Watches stream events of one response and accumulates an `Observation`.
#[derive(Debug)]
pub struct StreamTap {
    format: Format,
    obs: Observation,
}

impl StreamTap {
    pub fn new(format: Format) -> Self {
        Self {
            format,
            obs: Observation::default(),
        }
    }

    pub fn observation(&self) -> &Observation {
        &self.obs
    }

    /// Inspects one event; unknown or non-JSON events yield nothing.
    pub fn on_event(&mut self, ev: &SseEvent) -> Vec<Observed> {
        if !ev.has_data() || ev.data == "[DONE]" {
            return Vec::new();
        }
        let Ok(v) = serde_json::from_str::<Value>(&ev.data) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        if v.get("error").is_some()
            && v.get("type").and_then(Value::as_str) != Some("response.failed")
        {
            out.push(Observed::Error(ProviderError::from_value(
                self.format,
                None,
                &v,
                Bytes::copy_from_slice(ev.data.as_bytes()),
            )));
        }
        let mut obs = std::mem::take(&mut self.obs);
        match self.format {
            Format::AnthropicMessages => self.anthropic(&v, &mut obs, &mut out),
            Format::OpenAiChat => {
                message_facts(Format::OpenAiChat, &v, &mut obs, &mut out, "");
                chat_finish(&v, &mut out);
            }
            Format::OpenAiResponses => self.responses(&v, &mut obs, &mut out),
            Format::Gemini => gemini_facts(&v, &mut obs, &mut out),
        }
        for o in &out {
            obs.apply(o);
        }
        self.obs = obs;
        out
    }

    fn anthropic(&self, v: &Value, obs: &mut Observation, out: &mut Vec<Observed>) {
        match v.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                if let Some(m) = v.get("message") {
                    message_facts(self.format, m, obs, out, "");
                }
            }
            Some("message_delta") => {
                if let Some(s) = str_at(v, "/delta/stop_reason") {
                    out.push(Observed::Stop(s));
                }
                if let Some(u) = v.get("usage") {
                    out.push(Observed::Usage(Usage::from_provider(self.format, u)));
                }
            }
            _ => {}
        }
    }

    fn responses(&self, v: &Value, obs: &mut Observation, out: &mut Vec<Observed>) {
        let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
        if ty == "error" {
            out.push(Observed::Error(ProviderError::from_value(
                self.format,
                None,
                v,
                Bytes::from(v.to_string()),
            )));
            return;
        }
        let Some(r) = v.get("response") else {
            return;
        };
        message_facts(self.format, r, obs, out, "");
        if matches!(
            ty,
            "response.completed" | "response.incomplete" | "response.failed"
        ) && let Some(s) = str_at(r, "/status")
        {
            out.push(Observed::Stop(s));
        }
        if ty == "response.failed"
            && let Some(e) = r.get("error").filter(|e| !e.is_null())
        {
            let wrapped = serde_json::json!({ "error": e });
            out.push(Observed::Error(ProviderError::from_value(
                self.format,
                None,
                &wrapped,
                Bytes::from(e.to_string()),
            )));
        }
    }
}
