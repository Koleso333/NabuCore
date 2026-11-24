use std::fmt;

use serde_json::{Map, Value, json};

pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;
pub const HEADER_MISMATCH: i64 = -32020;
pub const MISSING_CLIENT_CAPABILITY: i64 = -32021;
pub const UNSUPPORTED_VERSION: i64 = -32022;

/// JSON-RPC error object.
#[derive(Debug, Clone, PartialEq)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

impl RpcError {
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    pub fn method_not_found(method: &str) -> Self {
        Self::new(METHOD_NOT_FOUND, format!("method not found: {method}"))
    }

    // Lenient: crooked servers send string codes or omit fields.
    fn from_value(v: &Value) -> Self {
        let code = match &v["code"] {
            Value::Number(n) => n.as_i64().unwrap_or(INTERNAL_ERROR),
            Value::String(s) => s.parse().unwrap_or(INTERNAL_ERROR),
            _ => INTERNAL_ERROR,
        };
        let message = match &v["message"] {
            Value::String(s) => s.clone(),
            Value::Null => match v {
                Value::String(s) => s.clone(),
                _ => "unknown error".into(),
            },
            other => other.to_string(),
        };
        Self {
            code,
            message,
            data: v.get("data").cloned().filter(|d| !d.is_null()),
        }
    }

    /// True for the errors only modern (2026-07-28+) servers send.
    pub fn is_modern(&self) -> bool {
        matches!(
            self.code,
            HEADER_MISMATCH | MISSING_CLIENT_CAPABILITY | UNSUPPORTED_VERSION
        )
    }

    /// Versions listed by an `UnsupportedProtocolVersionError`.
    pub fn supported_versions(&self) -> Vec<String> {
        self.data
            .as_ref()
            .and_then(|d| d["supported"].as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn to_value(&self) -> Value {
        let mut o = Map::new();
        o.insert("code".into(), json!(self.code));
        o.insert("message".into(), json!(self.message));
        if let Some(d) = &self.data {
            o.insert("data".into(), d.clone());
        }
        Value::Object(o)
    }
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (code {})", self.message, self.code)
    }
}

/// Request id normalized so that `5` and `"5"` match; some servers echo ids as strings.
pub fn id_key(id: &Value) -> Option<String> {
    match id {
        Value::Number(n) => Some(n.to_string()),
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// One message received from a server.
#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    Response {
        id: String,
        result: Result<Value, RpcError>,
    },
    /// Server-to-client request (legacy protocol versions only).
    Request {
        id: Value,
        method: String,
        params: Value,
    },
    Notification {
        method: String,
        params: Value,
    },
}

/// Splits a received JSON value into messages; malformed entries become warnings.
pub fn parse(v: Value) -> (Vec<Incoming>, Vec<String>) {
    let mut out = Vec::new();
    let mut warnings = Vec::new();
    match v {
        Value::Array(items) => {
            for item in items {
                one(item, &mut out, &mut warnings);
            }
        }
        other => one(other, &mut out, &mut warnings),
    }
    (out, warnings)
}

fn one(v: Value, out: &mut Vec<Incoming>, warnings: &mut Vec<String>) {
    let Value::Object(mut o) = v else {
        warnings.push(format!(
            "ignored non-object message: {}",
            crate::content::clip(&v.to_string(), 200)
        ));
        return;
    };
    let id = o.remove("id");
    let params = o.remove("params").unwrap_or(Value::Null);
    if let Some(Value::String(method)) = o.remove("method") {
        match id {
            Some(id) if !id.is_null() => out.push(Incoming::Request { id, method, params }),
            _ => out.push(Incoming::Notification { method, params }),
        }
        return;
    }
    let error = o.remove("error").filter(|e| !e.is_null());
    let result = o.remove("result");
    let Some(key) = id.as_ref().and_then(id_key) else {
        match error {
            Some(e) => warnings.push(format!(
                "server error without id: {}",
                RpcError::from_value(&e)
            )),
            None => warnings.push("ignored message without method or id".into()),
        }
        return;
    };
    let result = match (error, result) {
        (Some(e), _) => Err(RpcError::from_value(&e)),
        (None, Some(r)) => Ok(r),
        (None, None) => Err(RpcError::new(
            INVALID_REQUEST,
            "response has neither result nor error",
        )),
    };
    out.push(Incoming::Response { id: key, result });
}

pub fn request(id: u64, method: &str, params: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
}

pub fn notification(method: &str, params: Value) -> Value {
    if params.is_null() {
        json!({"jsonrpc": "2.0", "method": method})
    } else {
        json!({"jsonrpc": "2.0", "method": method, "params": params})
    }
}

pub fn response(id: Value, result: Result<Value, RpcError>) -> Value {
    match result {
        Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
        Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": e.to_value()}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_crooked_messages() {
        let (m, w) = parse(json!([
            {"jsonrpc": "2.0", "id": "7", "result": {}},
            {"id": 8, "error": {"code": "-32601", "message": "nope"}},
            {"id": 9},
            {"method": "notifications/progress", "params": {"progress": 1}},
            {"id": 3, "method": "ping"},
            {"error": {"code": -32700, "message": "bad"}},
            42
        ]));
        assert_eq!(w.len(), 2);
        assert_eq!(m.len(), 5);
        assert!(matches!(&m[0], Incoming::Response { id, result: Ok(_) } if id == "7"));
        assert!(
            matches!(&m[1], Incoming::Response { result: Err(e), .. } if e.code == METHOD_NOT_FOUND)
        );
        assert!(
            matches!(&m[2], Incoming::Response { result: Err(e), .. } if e.code == INVALID_REQUEST)
        );
        assert!(matches!(&m[3], Incoming::Notification { .. }));
        assert!(matches!(&m[4], Incoming::Request { method, .. } if method == "ping"));
    }

    #[test]
    fn reads_supported_versions() {
        let e = RpcError::from_value(&json!({
            "code": UNSUPPORTED_VERSION,
            "message": "Unsupported protocol version",
            "data": {"supported": ["2026-07-28", "2025-11-25"], "requested": "1900-01-01"}
        }));
        assert!(e.is_modern());
        assert_eq!(e.supported_versions(), ["2026-07-28", "2025-11-25"]);
    }
}
