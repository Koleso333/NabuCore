use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

/// Tool as listed by a server, after repairs.
#[derive(Debug, Clone, PartialEq)]
pub struct McpTool {
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    /// Always an object schema.
    pub input_schema: Value,
    pub output_schema: Option<Value>,
    pub annotations: Option<Value>,
    /// Set when the server's schema was not an object and was wrapped as `{"input": ...}`.
    pub wrapped: bool,
    /// `x-mcp-header` annotations: property path and header name part.
    pub headers: Vec<(Vec<String>, String)>,
}

impl McpTool {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        let mut t = Self {
            name: name.into(),
            title: None,
            description: Some(description.into()),
            input_schema,
            output_schema: None,
            annotations: None,
            wrapped: false,
            headers: Vec::new(),
        };
        let mut w = Vec::new();
        t.repair(&mut w);
        t
    }

    /// Parses one `tools/list` entry; `Err` means the entry is unusable.
    pub fn from_value(v: &Value, warnings: &mut Vec<String>) -> Result<Self, String> {
        let name = match v.get("name") {
            Some(Value::String(n)) if !n.trim().is_empty() => n.clone(),
            _ => {
                return Err(format!(
                    "tool without a name: {}",
                    crate::content::clip(&v.to_string(), 120)
                ));
            }
        };
        let text = |k: &str| match v.get(k) {
            Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
            _ => None,
        };
        let mut t = Self {
            title: text("title").or_else(|| v["annotations"]["title"].as_str().map(str::to_owned)),
            description: text("description"),
            input_schema: v
                .get("inputSchema")
                .or_else(|| v.get("input_schema"))
                .or_else(|| v.get("parameters"))
                .cloned()
                .unwrap_or(Value::Null),
            output_schema: v.get("outputSchema").cloned().filter(Value::is_object),
            annotations: v.get("annotations").cloned().filter(Value::is_object),
            wrapped: false,
            headers: Vec::new(),
            name,
        };
        t.repair(warnings);
        Ok(t)
    }

    fn repair(&mut self, warnings: &mut Vec<String>) {
        let schema = std::mem::take(&mut self.input_schema);
        self.input_schema = match schema {
            Value::Object(mut o) => match o.get("type") {
                None | Some(Value::Null) => {
                    o.insert("type".into(), json!("object"));
                    Value::Object(o)
                }
                Some(Value::String(t)) if t == "object" => Value::Object(o),
                Some(Value::Array(ts)) if ts.iter().any(|t| t == "object") => {
                    o.insert("type".into(), json!("object"));
                    Value::Object(o)
                }
                Some(_) => {
                    warnings.push(format!(
                        "tool {}: non-object input schema wrapped as {{\"input\": ...}}",
                        self.name
                    ));
                    self.wrapped = true;
                    json!({"type": "object", "properties": {"input": Value::Object(o)}, "required": ["input"]})
                }
            },
            Value::Null => json!({"type": "object", "properties": {}}),
            other => {
                warnings.push(format!(
                    "tool {}: invalid input schema {} replaced",
                    self.name,
                    crate::content::clip(&other.to_string(), 60)
                ));
                json!({"type": "object", "properties": {}})
            }
        };
        if let Value::Object(o) = &mut self.input_schema
            && !o.get("properties").is_none_or(Value::is_object)
        {
            warnings.push(format!("tool {}: invalid properties dropped", self.name));
            o.insert("properties".into(), json!({}));
        }
    }

    /// Reads `x-mcp-header` annotations; an invalid one makes the tool unusable over HTTP.
    pub fn read_headers(&mut self) -> Result<(), String> {
        let mut found = Vec::new();
        walk(&self.input_schema, &mut Vec::new(), true, &mut found)?;
        let mut seen: Vec<String> = Vec::new();
        for (_, h) in &found {
            let low = h.to_ascii_lowercase();
            if seen.contains(&low) {
                return Err(format!("duplicate x-mcp-header {h}"));
            }
            seen.push(low);
        }
        self.headers = found;
        Ok(())
    }
}

fn tchar(c: char) -> bool {
    c.is_ascii_alphanumeric() || "!#$%&'*+-.^_`|~".contains(c)
}

// `reachable` is false below `items`, composition keywords or `$ref`.
fn walk(
    s: &Value,
    path: &mut Vec<String>,
    reachable: bool,
    out: &mut Vec<(Vec<String>, String)>,
) -> Result<(), String> {
    let Value::Object(o) = s else { return Ok(()) };
    if let Some(h) = o.get("x-mcp-header") {
        let h = h
            .as_str()
            .filter(|h| !h.is_empty() && h.chars().all(tchar))
            .ok_or("invalid x-mcp-header value")?;
        if !reachable || path.is_empty() {
            return Err(format!(
                "x-mcp-header {h} is not on a statically reachable property"
            ));
        }
        let ok = matches!(
            o.get("type").and_then(Value::as_str),
            Some("string" | "integer" | "boolean")
        );
        if !ok {
            return Err(format!("x-mcp-header {h} is on a non-primitive property"));
        }
        out.push((path.clone(), h.to_owned()));
    }
    for (k, v) in o {
        match k.as_str() {
            "properties" => {
                if let Value::Object(props) = v {
                    for (name, sub) in props {
                        path.push(name.clone());
                        walk(sub, path, reachable, out)?;
                        path.pop();
                    }
                }
            }
            "items"
            | "prefixItems"
            | "oneOf"
            | "anyOf"
            | "allOf"
            | "not"
            | "if"
            | "then"
            | "else"
            | "additionalProperties"
            | "$defs"
            | "definitions" => match v {
                Value::Array(a) => {
                    for sub in a {
                        walk(sub, path, false, out)?;
                    }
                }
                Value::Object(_) => walk(v, path, false, out)?,
                _ => {}
            },
            _ => {}
        }
    }
    Ok(())
}

/// Header value per the Streamable HTTP encoding rules.
pub fn header_value(v: &str) -> String {
    let plain = v.bytes().all(|b| (0x20..=0x7e).contains(&b) || b == b'\t')
        && v.trim() == v
        && !(v.starts_with("=?base64?") && v.ends_with("?="));
    if plain {
        v.to_owned()
    } else {
        use base64::Engine;
        format!(
            "=?base64?{}?=",
            base64::engine::general_purpose::STANDARD.encode(v)
        )
    }
}

/// `Mcp-Param-*` headers for a call.
pub fn param_headers(tool: &McpTool, args: &Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (path, name) in &tool.headers {
        let mut v = args;
        for p in path {
            v = &v[p.as_str()];
        }
        let text = match v {
            Value::String(s) => s.clone(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
            _ => continue,
        };
        out.push((format!("Mcp-Param-{name}"), header_value(&text)));
    }
    out
}

const MAX_NAME: usize = 64;

fn sanitize(s: &str) -> String {
    let out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() { "_".into() } else { out }
}

/// Name exposed to the model: `mcp__{server}__{tool}`, valid for every provider (`[A-Za-z0-9_-]{1,64}`).
pub fn exposed_name(server: &str, tool: &str) -> String {
    let name = format!("mcp__{}__{}", sanitize(server), sanitize(tool));
    if name.len() <= MAX_NAME && name == format!("mcp__{server}__{tool}") {
        return name;
    }
    // Lossy names get a hash of the original so distinct tools stay distinct.
    let digest = Sha256::digest(format!("{server}\0{tool}").as_bytes());
    let hash: String = digest[..4].iter().map(|b| format!("{b:02x}")).collect();
    let keep = MAX_NAME - hash.len() - 1;
    format!("{}_{hash}", crate::content::clip(&name, keep))
}

/// Unwraps arguments of a wrapped tool and checks they form an object.
pub fn prepare_args(tool: &McpTool, raw: &str) -> Result<Value, String> {
    let raw = raw.trim();
    let v: Value = if raw.is_empty() {
        Value::Object(Map::new())
    } else {
        serde_json::from_str(raw).map_err(|e| format!("arguments are not valid JSON: {e}"))?
    };
    if tool.wrapped {
        return Ok(v.get("input").cloned().unwrap_or(Value::Null));
    }
    match v {
        Value::Object(_) => Ok(v),
        Value::Null => Ok(Value::Object(Map::new())),
        other => Err(format!(
            "arguments must be a JSON object, got {}",
            crate::content::clip(&other.to_string(), 80)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repairs_schemas() {
        let mut w = Vec::new();
        let t = McpTool::from_value(&json!({"name": "a"}), &mut w).unwrap();
        assert_eq!(t.input_schema, json!({"type": "object", "properties": {}}));
        let t = McpTool::from_value(
            &json!({"name": "b", "inputSchema": {"properties": {"x": {}}}}),
            &mut w,
        )
        .unwrap();
        assert_eq!(t.input_schema["type"], "object");
        let t = McpTool::from_value(
            &json!({"name": "c", "inputSchema": {"type": "string"}}),
            &mut w,
        )
        .unwrap();
        assert!(t.wrapped);
        assert_eq!(prepare_args(&t, r#"{"input":"hi"}"#).unwrap(), json!("hi"));
        let t = McpTool::from_value(&json!({"name": "d", "inputSchema": "junk"}), &mut w).unwrap();
        assert_eq!(t.input_schema["type"], "object");
        assert!(McpTool::from_value(&json!({"name": ""}), &mut w).is_err());
        assert_eq!(w.len(), 2);

        let t = McpTool::new("e", "", json!({"type": "object"}));
        assert_eq!(prepare_args(&t, "").unwrap(), json!({}));
        assert!(prepare_args(&t, "[1]").is_err());
        assert!(prepare_args(&t, "{bad").is_err());
    }

    #[test]
    fn names_fit_every_provider() {
        assert_eq!(exposed_name("fs", "read_file"), "mcp__fs__read_file");
        let a = exposed_name("my server", "admin.tools.list");
        let b = exposed_name("my_server", "admin_tools_list");
        assert_ne!(a, b);
        assert!(a.starts_with("mcp__my_server__admin_tools_list_"));
        let long = exposed_name("srv", &"x".repeat(100));
        assert_eq!(long.len(), 64);
        assert!(
            long.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        );
    }

    #[test]
    fn mcp_headers() {
        let mut t = McpTool::new(
            "q",
            "",
            json!({"type": "object", "properties": {
                "region": {"type": "string", "x-mcp-header": "Region"},
                "opts": {"type": "object", "properties": {"n": {"type": "integer", "x-mcp-header": "N"}}}
            }}),
        );
        t.read_headers().unwrap();
        let h = param_headers(&t, &json!({"region": "Hello, 世界", "opts": {"n": 4}}));
        assert_eq!(
            h,
            [
                (
                    "Mcp-Param-Region".to_owned(),
                    "=?base64?SGVsbG8sIOS4lueVjA==?=".to_owned()
                ),
                ("Mcp-Param-N".to_owned(), "4".to_owned()),
            ]
        );
        let mut bad = McpTool::new(
            "q",
            "",
            json!({"type": "object", "properties": {
                "xs": {"type": "array", "items": {"type": "string", "x-mcp-header": "X"}}
            }}),
        );
        assert!(bad.read_headers().is_err());
        let mut num = McpTool::new(
            "q",
            "",
            json!({"type": "object", "properties": {"f": {"type": "number", "x-mcp-header": "F"}}}),
        );
        assert!(num.read_headers().is_err());
        assert_eq!(header_value(" padded "), "=?base64?IHBhZGRlZCA=?=");
    }
}
