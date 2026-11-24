use ncore_core::json::{self, JsonPath, Patch};
use ncore_core::{Format, SseParser, StreamTap};
use proptest::prelude::*;
use serde_json::Value;

const FIXTURES: [(&str, Format); 4] = [
    ("anthropic.sse", Format::AnthropicMessages),
    ("openai_chat.sse", Format::OpenAiChat),
    ("openai_responses.sse", Format::OpenAiResponses),
    ("gemini.sse", Format::Gemini),
];

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

fn tap_all(format: Format, input: &[u8], chunk: usize) -> (Vec<u8>, ncore_core::Observation) {
    let mut parser = SseParser::new();
    let mut tap = StreamTap::new(format);
    let mut out = Vec::new();
    for c in input.chunks(chunk.max(1)) {
        for ev in parser.feed(c) {
            tap.on_event(&ev);
            out.extend_from_slice(&ev.raw);
        }
    }
    out.extend_from_slice(&parser.finish());
    (out, tap.observation().clone())
}

#[test]
fn fixtures_yield_usage_model_and_stop() {
    let expect = [
        (Some(125), Some(15), "claude-opus-5-5", "end_turn"),
        (Some(50), Some(9), "gpt-x", "stop"),
        (Some(40), Some(12), "gpt-x", "completed"),
        (Some(20), Some(38), "gemini-x", "STOP"),
    ];
    for ((name, format), (input, output, model, stop)) in FIXTURES.iter().zip(expect) {
        let (_, obs) = tap_all(*format, &fixture(name), 4096);
        assert_eq!(obs.usage.input_total, input, "{name}");
        assert_eq!(obs.usage.output_total, output, "{name}");
        assert_eq!(obs.model.as_deref(), Some(model), "{name}");
        assert_eq!(obs.stop_reason.as_deref(), Some(stop), "{name}");
        assert!(obs.error.is_none(), "{name}");
    }
}

#[test]
fn mid_stream_errors_are_observed() {
    let input = b"event: error\ndata: {\"type\": \"error\", \"error\": {\"type\": \"overloaded_error\", \"message\": \"Overloaded\"}}\n\n";
    let (_, obs) = tap_all(Format::AnthropicMessages, input, 7);
    assert_eq!(
        obs.error.unwrap().kind,
        ncore_core::ErrorKind::Overloaded
    );

    let input = b"event: error\ndata: {\"type\":\"error\",\"code\":\"server_error\",\"message\":\"boom\",\"param\":null}\n\n";
    let (_, obs) = tap_all(Format::OpenAiResponses, input, 5);
    assert_eq!(obs.error.unwrap().message, "boom");
}

proptest! {
    #[test]
    fn stream_bytes_survive_any_chunking(idx in 0usize..4, chunk in 1usize..64) {
        let (name, format) = FIXTURES[idx];
        let input = fixture(name);
        let (out, obs) = tap_all(format, &input, chunk);
        prop_assert_eq!(&out, &input);
        let (_, whole) = tap_all(format, &input, input.len());
        prop_assert_eq!(obs, whole);
    }

    #[test]
    fn find_agrees_with_serde(v in arb_json(), pretty in any::<bool>()) {
        let text = if pretty { serde_json::to_vec_pretty(&v).unwrap() } else { serde_json::to_vec(&v).unwrap() };
        for (pointer, expected) in pointers(&v, String::new()) {
            let span = json::find(&text, &JsonPath::parse(&pointer).unwrap()).unwrap().unwrap();
            let got: Value = serde_json::from_slice(&text[span.start..span.end]).unwrap();
            prop_assert_eq!(got, expected);
        }
    }

    #[test]
    fn set_changes_only_target(v in arb_json(), new in arb_leaf()) {
        let text = serde_json::to_vec_pretty(&v).unwrap();
        for (pointer, _) in pointers(&v, String::new()).into_iter().skip(1) {
            let out = json::apply(&text, &[Patch::set(&pointer, &new).unwrap()]).unwrap();
            let mut want = v.clone();
            *want.pointer_mut(&pointer).unwrap() = new.clone();
            let got: Value = serde_json::from_slice(&out).unwrap();
            prop_assert_eq!(got, want);
        }
    }

    #[test]
    fn remove_matches_serde(v in arb_json()) {
        let text = serde_json::to_vec(&v).unwrap();
        for (pointer, _) in pointers(&v, String::new()).into_iter().skip(1) {
            let out = json::apply(&text, &[Patch::remove(&pointer).unwrap()]).unwrap();
            let mut want = v.clone();
            let (parent, last) = pointer.rsplit_once('/').unwrap();
            let last = last.replace("~1", "/").replace("~0", "~");
            match want.pointer_mut(parent).unwrap() {
                Value::Object(m) => { m.remove(&last); }
                Value::Array(a) => { a.remove(last.parse::<usize>().unwrap()); }
                _ => unreachable!(),
            }
            let got: Value = serde_json::from_slice(&out).unwrap();
            prop_assert_eq!(got, want);
        }
    }

    #[test]
    fn scanner_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..256)) {
        let _ = json::validate(&bytes);
        let mut p = SseParser::new();
        let _ = p.feed(&bytes);
        let _ = p.finish();
    }
}

fn arb_leaf() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(Value::from),
        (-1e9f64..1e9).prop_map(Value::from),
        ".{0,8}".prop_map(Value::String),
    ]
}

fn arb_json() -> impl Strategy<Value = Value> {
    arb_leaf().prop_recursive(4, 32, 5, |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 0..5).prop_map(Value::Array),
            proptest::collection::btree_map("[a-z/~\"\\\\]{0,4}", inner, 0..5)
                .prop_map(|m| Value::Object(m.into_iter().collect())),
        ]
    })
}

fn pointers(v: &Value, prefix: String) -> Vec<(String, Value)> {
    let mut out = vec![(prefix.clone(), v.clone())];
    match v {
        Value::Object(m) => {
            for (k, c) in m {
                let tok = k.replace('~', "~0").replace('/', "~1");
                out.extend(pointers(c, format!("{prefix}/{tok}")));
            }
        }
        Value::Array(a) => {
            for (i, c) in a.iter().enumerate() {
                out.extend(pointers(c, format!("{prefix}/{i}")));
            }
        }
        _ => {}
    }
    out
}
