use std::ops::Range;

use bytes::{Bytes, BytesMut};

/// One SSE block with its exact raw bytes, including the terminating blank line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub raw: Bytes,
    pub event: Option<String>,
    pub data: String,
    pub id: Option<String>,
    pub retry: Option<u64>,
    data_lines: usize,
    data_span: Option<Range<usize>>,
}

impl SseEvent {
    /// False for comment-only blocks, which the SSE spec does not dispatch.
    pub fn has_data(&self) -> bool {
        self.data_lines > 0
    }

    /// Returns the raw block with its data replaced, keeping other bytes when possible.
    pub fn with_data(&self, data: &str) -> Bytes {
        if let (1, Some(span), false) = (
            self.data_lines,
            &self.data_span,
            data.contains(['\n', '\r']),
        ) {
            let mut out = BytesMut::with_capacity(self.raw.len() + data.len());
            out.extend_from_slice(&self.raw[..span.start]);
            out.extend_from_slice(data.as_bytes());
            out.extend_from_slice(&self.raw[span.end..]);
            return out.freeze();
        }
        encode(self.event.as_deref(), self.id.as_deref(), data)
    }

    /// Replaces the data in place, keeping the event consistent with its new raw bytes.
    pub fn set_data(&mut self, data: &str) {
        let raw = self.with_data(data);
        if let Some(ev) = SseParser::new().feed(&raw).pop() {
            *self = ev;
        }
    }
}

/// Encodes a single SSE event.
pub fn encode(event: Option<&str>, id: Option<&str>, data: &str) -> Bytes {
    let mut out = String::new();
    if let Some(e) = event {
        out.push_str("event: ");
        out.push_str(e);
        out.push('\n');
    }
    if let Some(i) = id {
        out.push_str("id: ");
        out.push_str(i);
        out.push('\n');
    }
    for line in data.split('\n') {
        out.push_str("data: ");
        out.push_str(line.strip_suffix('\r').unwrap_or(line));
        out.push('\n');
    }
    out.push('\n');
    Bytes::from(out)
}

/// Incremental WHATWG-style SSE parser that keeps raw bytes for byte-exact forwarding.
#[derive(Debug, Default)]
pub struct SseParser {
    buf: BytesMut,
    line_start: usize,
    started: bool,
    event: Option<String>,
    data: Vec<u8>,
    data_lines: usize,
    data_span: Option<Range<usize>>,
    id: Option<String>,
    retry: Option<u64>,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds a chunk and returns every block it completed.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some((content_end, next)) = self.next_line(false) {
            self.handle_line(content_end, next, &mut out);
        }
        out
    }

    /// Ends the stream: returns bytes of an unterminated trailing block, which is not dispatched.
    pub fn finish(&mut self) -> Bytes {
        while let Some((content_end, next)) = self.next_line(true) {
            let mut sink = Vec::new();
            self.handle_line(content_end, next, &mut sink);
            if let Some(ev) = sink.pop() {
                return ev.raw;
            }
        }
        self.reset_fields();
        self.line_start = 0;
        self.buf.split().freeze()
    }

    // Finds the next line end; a lone trailing CR waits for more input unless at EOF.
    fn next_line(&self, eof: bool) -> Option<(usize, usize)> {
        let b = &self.buf[..];
        let mut i = self.line_start;
        while i < b.len() {
            match b[i] {
                b'\n' => return Some((i, i + 1)),
                b'\r' if i + 1 < b.len() => {
                    return Some((i, if b[i + 1] == b'\n' { i + 2 } else { i + 1 }));
                }
                b'\r' if eof => return Some((i, i + 1)),
                b'\r' => return None,
                _ => i += 1,
            }
        }
        None
    }

    fn handle_line(&mut self, content_end: usize, next: usize, out: &mut Vec<SseEvent>) {
        let mut start = self.line_start;
        if !self.started {
            self.started = true;
            if self.buf[start..content_end].starts_with(b"\xEF\xBB\xBF") {
                start += 3;
            }
        }
        self.line_start = next;
        if start == content_end {
            let raw = self.buf.split_to(next).freeze();
            self.line_start = 0;
            out.push(SseEvent {
                raw,
                event: self.event.take(),
                data: String::from_utf8_lossy(&std::mem::take(&mut self.data)).into_owned(),
                id: self.id.take(),
                retry: self.retry.take(),
                data_lines: std::mem::take(&mut self.data_lines),
                data_span: self.data_span.take(),
            });
            return;
        }
        let line = &self.buf[start..content_end];
        if line[0] == b':' {
            return;
        }
        let (name, value_start) = match line.iter().position(|&c| c == b':') {
            Some(p) => {
                let vs = start + p + 1;
                let vs = if self.buf.get(vs) == Some(&b' ') && vs < content_end {
                    vs + 1
                } else {
                    vs
                };
                (&line[..p], vs)
            }
            None => (line, content_end),
        };
        let value = &self.buf[value_start..content_end];
        match name {
            b"event" => self.event = Some(String::from_utf8_lossy(value).into_owned()),
            b"data" => {
                if self.data_lines > 0 {
                    self.data.push(b'\n');
                }
                self.data.extend_from_slice(value);
                self.data_lines += 1;
                self.data_span = Some(value_start..content_end);
            }
            b"id" if !value.contains(&0) => {
                self.id = Some(String::from_utf8_lossy(value).into_owned())
            }
            b"retry" => {
                if let Some(n) = std::str::from_utf8(value).ok().and_then(|v| v.parse().ok()) {
                    self.retry = Some(n);
                }
            }
            _ => {}
        }
    }

    fn reset_fields(&mut self) {
        self.event = None;
        self.data.clear();
        self.data_lines = 0;
        self.data_span = None;
        self.id = None;
        self.retry = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fields_and_keeps_raw() {
        let input = b"event: message_start\ndata: {\"a\":1}\n\n: ping\n\ndata: x\ndata: y\r\n\r\n";
        let mut p = SseParser::new();
        let evs = p.feed(input);
        assert_eq!(evs.len(), 3);
        assert_eq!(evs[0].event.as_deref(), Some("message_start"));
        assert_eq!(evs[0].data, "{\"a\":1}");
        assert!(!evs[1].has_data());
        assert_eq!(evs[2].data, "x\ny");
        let joined: Vec<u8> = evs.iter().flat_map(|e| e.raw.to_vec()).collect();
        assert_eq!(joined, input);
        assert!(p.finish().is_empty());
    }

    #[test]
    fn handles_cr_split_across_chunks() {
        let mut p = SseParser::new();
        assert!(p.feed(b"data: a\r").is_empty());
        assert!(p.feed(b"\n\r").is_empty());
        let evs = p.feed(b"\ndata: b\n\n");
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].data, "a");
        assert_eq!(evs[0].raw.as_ref(), b"data: a\r\n\r\n");
        assert_eq!(evs[1].data, "b");
    }

    #[test]
    fn finish_returns_unterminated_tail() {
        let mut p = SseParser::new();
        assert!(p.feed(b"data: done\n").is_empty());
        assert_eq!(p.finish().as_ref(), b"data: done\n");
    }

    #[test]
    fn with_data_splices_single_line() {
        let mut p = SseParser::new();
        let ev = p.feed(b"event: e\ndata:  {\"m\":\"a\"}\n\n").pop().unwrap();
        assert_eq!(ev.data, " {\"m\":\"a\"}");
        assert_eq!(
            ev.with_data("{\"m\":\"b\"}").as_ref(),
            b"event: e\ndata: {\"m\":\"b\"}\n\n"
        );
    }
}
