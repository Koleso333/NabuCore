use std::fmt;

use serde::Serialize;

const MAX_DEPTH: usize = 512;

/// Byte range of a JSON value inside a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    pub pos: usize,
    pub reason: &'static str,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "json: {} at byte {}", self.reason, self.pos)
    }
}

impl std::error::Error for JsonError {}

fn err<T>(pos: usize, reason: &'static str) -> Result<T, JsonError> {
    Err(JsonError { pos, reason })
}

/// RFC 6901 JSON Pointer; tokens are resolved as keys or indices at lookup time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonPath(Vec<String>);

impl JsonPath {
    /// Parses a pointer like `/messages/0/content`.
    pub fn parse(pointer: &str) -> Result<Self, JsonError> {
        if pointer.is_empty() {
            return Ok(Self(Vec::new()));
        }
        let Some(rest) = pointer.strip_prefix('/') else {
            return err(0, "pointer must start with '/'");
        };
        Ok(Self(
            rest.split('/')
                .map(|t| t.replace("~1", "/").replace("~0", "~"))
                .collect(),
        ))
    }

    pub fn tokens(&self) -> &[String] {
        &self.0
    }
}

/// Recursive-descent validator that reports value spans without building a tree.
struct Scanner<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Scanner<'a> {
    fn at(b: &'a [u8], pos: usize) -> Self {
        Self { b, pos }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, c: u8, reason: &'static str) -> Result<(), JsonError> {
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            err(self.pos, reason)
        }
    }

    fn value(&mut self, depth: usize) -> Result<Span, JsonError> {
        if depth > MAX_DEPTH {
            return err(self.pos, "nesting too deep");
        }
        let start = self.pos;
        match self.peek() {
            Some(b'{') => self.object(depth)?,
            Some(b'[') => self.array(depth)?,
            Some(b'"') => {
                self.string()?;
            }
            Some(b't') => self.literal(b"true")?,
            Some(b'f') => self.literal(b"false")?,
            Some(b'n') => self.literal(b"null")?,
            Some(b'-' | b'0'..=b'9') => self.number()?,
            Some(_) => return err(self.pos, "unexpected character"),
            None => return err(self.pos, "unexpected end"),
        }
        Ok(Span {
            start,
            end: self.pos,
        })
    }

    fn object(&mut self, depth: usize) -> Result<(), JsonError> {
        self.pos += 1;
        self.ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(());
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return err(self.pos, "expected object key");
            }
            self.string()?;
            self.ws();
            self.expect(b':', "expected ':'")?;
            self.ws();
            self.value(depth + 1)?;
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(());
                }
                _ => return err(self.pos, "expected ',' or '}'"),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<(), JsonError> {
        self.pos += 1;
        self.ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(());
        }
        loop {
            self.ws();
            self.value(depth + 1)?;
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(());
                }
                _ => return err(self.pos, "expected ',' or ']'"),
            }
        }
    }

    // Returns true when the string contains escapes.
    fn string(&mut self) -> Result<bool, JsonError> {
        self.pos += 1;
        let mut escaped = false;
        loop {
            match self.peek() {
                None => return err(self.pos, "unterminated string"),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(escaped);
                }
                Some(b'\\') => {
                    escaped = true;
                    self.pos += 1;
                    match self.peek() {
                        Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => {
                            self.pos += 1
                        }
                        Some(b'u') => {
                            self.pos += 1;
                            for _ in 0..4 {
                                if !matches!(self.peek(), Some(c) if c.is_ascii_hexdigit()) {
                                    return err(self.pos, "bad \\u escape");
                                }
                                self.pos += 1;
                            }
                        }
                        _ => return err(self.pos, "bad escape"),
                    }
                }
                Some(c) if c < 0x20 => return err(self.pos, "control character in string"),
                Some(_) => self.pos += 1,
            }
        }
    }

    fn literal(&mut self, lit: &[u8]) -> Result<(), JsonError> {
        if self.b[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(())
        } else {
            err(self.pos, "bad literal")
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        self.pos - start
    }

    fn number(&mut self) -> Result<(), JsonError> {
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return err(self.pos, "bad number"),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if self.digits() == 0 {
                return err(self.pos, "bad fraction");
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if self.digits() == 0 {
                return err(self.pos, "bad exponent");
            }
        }
        Ok(())
    }
}

/// Object member as key span (with quotes) and value span.
#[derive(Debug, Clone, Copy)]
struct Member {
    key: Span,
    value: Span,
}

/// Validates the whole document and returns the span of the root value.
pub fn validate(b: &[u8]) -> Result<Span, JsonError> {
    let mut s = Scanner::at(b, 0);
    s.ws();
    let span = s.value(0)?;
    s.ws();
    if s.pos != b.len() {
        return err(s.pos, "trailing characters");
    }
    Ok(span)
}

fn members(b: &[u8], obj: Span) -> Result<Vec<Member>, JsonError> {
    let mut out = Vec::new();
    let mut s = Scanner::at(b, obj.start + 1);
    s.ws();
    if s.peek() == Some(b'}') {
        return Ok(out);
    }
    loop {
        s.ws();
        let ks = s.pos;
        s.string()?;
        let key = Span {
            start: ks,
            end: s.pos,
        };
        s.ws();
        s.expect(b':', "expected ':'")?;
        s.ws();
        let value = s.value(0)?;
        out.push(Member { key, value });
        s.ws();
        match s.peek() {
            Some(b',') => s.pos += 1,
            _ => return Ok(out),
        }
    }
}

fn elements(b: &[u8], arr: Span) -> Result<Vec<Span>, JsonError> {
    let mut out = Vec::new();
    let mut s = Scanner::at(b, arr.start + 1);
    s.ws();
    if s.peek() == Some(b']') {
        return Ok(out);
    }
    loop {
        s.ws();
        out.push(s.value(0)?);
        s.ws();
        match s.peek() {
            Some(b',') => s.pos += 1,
            _ => return Ok(out),
        }
    }
}

fn key_eq(b: &[u8], key: Span, want: &str) -> bool {
    let raw = &b[key.start + 1..key.end - 1];
    if !raw.contains(&b'\\') {
        return raw == want.as_bytes();
    }
    serde_json::from_slice::<String>(&b[key.start..key.end]).is_ok_and(|k| k == want)
}

fn child(b: &[u8], parent: Span, token: &str) -> Result<Option<Span>, JsonError> {
    match b[parent.start] {
        b'{' => Ok(members(b, parent)?
            .into_iter()
            .rev()
            .find(|m| key_eq(b, m.key, token))
            .map(|m| m.value)),
        b'[' => {
            let Ok(i) = token.parse::<usize>() else {
                return Ok(None);
            };
            Ok(elements(b, parent)?.get(i).copied())
        }
        _ => Ok(None),
    }
}

/// Finds the span of the value at `path`; `None` if any segment is missing.
pub fn find(b: &[u8], path: &JsonPath) -> Result<Option<Span>, JsonError> {
    let mut cur = validate(b)?;
    for token in path.tokens() {
        match child(b, cur, token)? {
            Some(s) => cur = s,
            None => return Ok(None),
        }
    }
    Ok(Some(cur))
}

/// Point edit applied to raw JSON bytes without re-serializing the document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Patch {
    Set { path: JsonPath, value: Vec<u8> },
    Remove { path: JsonPath },
}

impl Patch {
    /// Sets `pointer` to the serialized `value`, inserting the key if absent.
    pub fn set(pointer: &str, value: impl Serialize) -> Result<Self, JsonError> {
        let value = serde_json::to_vec(&value).map_err(|_| JsonError {
            pos: 0,
            reason: "unserializable value",
        })?;
        Ok(Self::Set {
            path: JsonPath::parse(pointer)?,
            value,
        })
    }

    /// Sets `pointer` to already-encoded JSON.
    pub fn set_raw(pointer: &str, value: impl Into<Vec<u8>>) -> Result<Self, JsonError> {
        let value = value.into();
        validate(&value)?;
        Ok(Self::Set {
            path: JsonPath::parse(pointer)?,
            value,
        })
    }

    /// Removes the member or element at `pointer`; missing paths are ignored.
    pub fn remove(pointer: &str) -> Result<Self, JsonError> {
        Ok(Self::Remove {
            path: JsonPath::parse(pointer)?,
        })
    }
}

fn splice(b: &[u8], at: Span, with: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(b.len() - (at.end - at.start) + with.len());
    out.extend_from_slice(&b[..at.start]);
    out.extend_from_slice(with);
    out.extend_from_slice(&b[at.end..]);
    out
}

fn apply_one(b: &[u8], patch: &Patch) -> Result<Vec<u8>, JsonError> {
    let (path, value) = match patch {
        Patch::Set { path, value } => (path, Some(value.as_slice())),
        Patch::Remove { path } => (path, None),
    };
    let Some((last, parents)) = path.tokens().split_last() else {
        return match value {
            Some(v) => Ok(v.to_vec()),
            None => err(0, "cannot remove the root"),
        };
    };
    let parent_path = JsonPath(parents.to_vec());
    let Some(parent) = find(b, &parent_path)? else {
        return match value {
            Some(_) => err(0, "parent path not found"),
            None => Ok(b.to_vec()),
        };
    };
    match b[parent.start] {
        b'{' => {
            let ms = members(b, parent)?;
            let idx = ms.iter().rposition(|m| key_eq(b, m.key, last));
            match (idx, value) {
                (Some(i), Some(v)) => Ok(splice(b, ms[i].value, v)),
                (Some(i), None) => {
                    let range = removal_range(
                        &ms.iter()
                            .map(|m| Span {
                                start: m.key.start,
                                end: m.value.end,
                            })
                            .collect::<Vec<_>>(),
                        i,
                    );
                    Ok(splice(b, range, b""))
                }
                (None, Some(v)) => {
                    let mut ins = Vec::new();
                    let at = match ms.last() {
                        Some(m) => {
                            ins.push(b',');
                            m.value.end
                        }
                        None => parent.start + 1,
                    };
                    ins.extend_from_slice(&serde_json::to_vec(last).expect("string key"));
                    ins.push(b':');
                    ins.extend_from_slice(v);
                    Ok(splice(b, Span { start: at, end: at }, &ins))
                }
                (None, None) => Ok(b.to_vec()),
            }
        }
        b'[' => {
            let es = elements(b, parent)?;
            if last == "-" {
                let Some(v) = value else {
                    return Ok(b.to_vec());
                };
                let mut ins = Vec::new();
                let at = match es.last() {
                    Some(e) => {
                        ins.push(b',');
                        e.end
                    }
                    None => parent.start + 1,
                };
                ins.extend_from_slice(v);
                return Ok(splice(b, Span { start: at, end: at }, &ins));
            }
            let idx = last.parse::<usize>().ok().filter(|i| *i < es.len());
            match (idx, value) {
                (Some(i), Some(v)) => Ok(splice(b, es[i], v)),
                (Some(i), None) => Ok(splice(b, removal_range(&es, i), b"")),
                (None, Some(_)) => err(parent.start, "array index out of range"),
                (None, None) => Ok(b.to_vec()),
            }
        }
        _ => match value {
            Some(_) => err(parent.start, "parent is not a container"),
            None => Ok(b.to_vec()),
        },
    }
}

// Range covering item `i` and one adjacent comma so the container stays valid.
fn removal_range(items: &[Span], i: usize) -> Span {
    if items.len() == 1 {
        items[0]
    } else if i + 1 < items.len() {
        Span {
            start: items[i].start,
            end: items[i + 1].start,
        }
    } else {
        Span {
            start: items[i - 1].end,
            end: items[i].end,
        }
    }
}

/// Applies patches in order; bytes outside the edited spans stay untouched.
pub fn apply(b: &[u8], patches: &[Patch]) -> Result<Vec<u8>, JsonError> {
    let mut cur = b.to_vec();
    if patches.is_empty() {
        validate(&cur)?;
    }
    for p in patches {
        cur = apply_one(&cur, p)?;
    }
    Ok(cur)
}

/// Reads a string value at `pointer`, if present.
pub fn get_str(b: &[u8], pointer: &str) -> Option<String> {
    let span = find(b, &JsonPath::parse(pointer).ok()?).ok()??;
    serde_json::from_slice::<String>(&b[span.start..span.end]).ok()
}

/// Reads a boolean value at `pointer`, if present.
pub fn get_bool(b: &[u8], pointer: &str) -> Option<bool> {
    let span = find(b, &JsonPath::parse(pointer).ok()?).ok()??;
    match &b[span.start..span.end] {
        b"true" => Some(true),
        b"false" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(b: &[u8], p: &str) -> Option<String> {
        find(b, &JsonPath::parse(p).unwrap())
            .unwrap()
            .map(|sp| String::from_utf8(b[sp.start..sp.end].to_vec()).unwrap())
    }

    #[test]
    fn finds_nested_values() {
        let b = br#" { "model" : "a", "messages": [ {"role":"user","content":[1, 2.5e3, "x"]} ], "a/b": {"~k": true} } "#;
        assert_eq!(s(b, "/model").as_deref(), Some(r#""a""#));
        assert_eq!(s(b, "/messages/0/content/1").as_deref(), Some("2.5e3"));
        assert_eq!(s(b, "/a~1b/~0k").as_deref(), Some("true"));
        assert_eq!(s(b, "/messages/3"), None);
        assert_eq!(s(b, "/nope"), None);
    }

    #[test]
    fn matches_escaped_keys() {
        let b = br#"{"model":"x"}"#;
        assert_eq!(s(b, "/model").as_deref(), Some(r#""x""#));
    }

    #[test]
    fn rejects_invalid_json() {
        for bad in [
            &b"{"[..],
            b"{\"a\":}",
            b"[1,]",
            b"01",
            b"\"\x01\"",
            b"{} x",
            b"tru",
        ] {
            assert!(validate(bad).is_err(), "{:?}", String::from_utf8_lossy(bad));
        }
    }

    #[test]
    fn set_replaces_only_the_span() {
        let b = br#"{"model":  "old" ,"x":1.0,"stream":true}"#;
        let out = apply(b, &[Patch::set("/model", "new").unwrap()]).unwrap();
        assert_eq!(out, br#"{"model":  "new" ,"x":1.0,"stream":true}"#);
    }

    #[test]
    fn set_inserts_missing_key() {
        let out = apply(br#"{"a":1}"#, &[Patch::set("/b", 2).unwrap()]).unwrap();
        assert_eq!(out, br#"{"a":1,"b":2}"#);
        let out = apply(b"{ }", &[Patch::set("/b", 2).unwrap()]).unwrap();
        assert_eq!(out, br#"{"b":2 }"#);
        let out = apply(br#"{"a":[1]}"#, &[Patch::set("/a/-", 2).unwrap()]).unwrap();
        assert_eq!(out, br#"{"a":[1,2]}"#);
    }

    #[test]
    fn remove_keeps_document_valid() {
        let b = br#"{"a":1, "b":2, "c":3}"#;
        for (p, want) in [
            ("/a", r#"{"b":2, "c":3}"#),
            ("/b", r#"{"a":1, "c":3}"#),
            ("/c", r#"{"a":1, "b":2}"#),
        ] {
            let out = apply(b, &[Patch::remove(p).unwrap()]).unwrap();
            assert_eq!(String::from_utf8(out).unwrap(), want);
        }
        let out = apply(br#"{"a":[1,2,3]}"#, &[Patch::remove("/a/1").unwrap()]).unwrap();
        assert_eq!(out, br#"{"a":[1,3]}"#);
        let out = apply(br#"{"a":1}"#, &[Patch::remove("/zz").unwrap()]).unwrap();
        assert_eq!(out, br#"{"a":1}"#);
    }

    #[test]
    fn set_with_missing_parent_fails() {
        assert!(apply(br#"{"a":1}"#, &[Patch::set("/x/y", 1).unwrap()]).is_err());
    }
}
