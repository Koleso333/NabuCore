use ncore_core::codec::{self, EncodeOptions, StreamDecoder, StreamEncoder};
use ncore_core::ir::*;
use ncore_core::{Format, SseEvent, SseParser, sse};
use serde_json::{Value, json};

const F: Format = Format::OpenAiChat;

fn details() -> Value {
    json!([
        {"type": "reasoning.text", "text": "План 🌡", "index": 0, "id": "r0", "format": "native-v1", "extra": {"keep": true}},
        {"type": "reasoning.summary", "summary": "Check weather.", "index": 1, "id": "r1", "format": "native-v1"},
        {"type": "reasoning.encrypted", "data": "ABC/+=XYZ", "index": 2, "id": "r2", "format": "native-v1", "future": 42},
    ])
}

fn full_response(message: Value) -> Response {
    codec::chat::decode_response(json!({
        "id": "chat-1", "model": "model-1", "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8}, "extra": true,
    }))
    .unwrap()
}

fn request(response: &Response) -> Request {
    Request {
        model: "model-1".into(),
        messages: vec![Message::new(Role::Assistant, response.content.clone())],
        ..Default::default()
    }
}

fn encoded_response(response: &Response) -> Value {
    let (bytes, losses) = codec::encode_response(F, response);
    assert!(losses.is_empty(), "{losses:?}");
    serde_json::from_slice(&bytes).unwrap()
}

fn encoded_request(request: &Request, options: &EncodeOptions) -> Value {
    let encoded = codec::encode_request(F, request, options).unwrap();
    assert!(encoded.losses.is_empty(), "{:?}", encoded.losses);
    serde_json::from_slice(&encoded.body).unwrap()
}

fn payload(response: &Response) -> &Value {
    response
        .content
        .iter()
        .find_map(|block| match block {
            Block::Thinking(thinking) => thinking
                .sig
                .as_ref()
                .and_then(|sig| sig.for_format(F))
                .and_then(|value| value.get("reasoning_details")),
            _ => None,
        })
        .expect("structured reasoning")
}

fn thoughts(response: &Response) -> String {
    response
        .content
        .iter()
        .filter_map(|block| match block {
            Block::Thinking(thinking) => Some(thinking.text.as_str()),
            _ => None,
        })
        .collect::<String>()
}

fn event(value: Value) -> SseEvent {
    SseParser::new()
        .feed(&sse::encode(None, None, &value.to_string()))
        .pop()
        .unwrap()
}

fn chunk(delta: Value, finish: Value) -> Value {
    json!({
        "id": "chat-1", "model": "model-1", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
}

fn stream(chunks: &[Value], done: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    for chunk in chunks {
        bytes.extend_from_slice(&sse::encode(None, None, &chunk.to_string()));
    }
    if done {
        bytes.extend_from_slice(&sse::encode(None, None, "[DONE]"));
    }
    bytes
}

fn events(bytes: &[u8], size: usize) -> Vec<StreamEvent> {
    let mut parser = SseParser::new();
    let mut decoder = codec::chat::Decoder::default();
    let mut events = Vec::new();
    for bytes in bytes.chunks(size) {
        for event in parser.feed(bytes) {
            events.extend(decoder.feed(&event).unwrap());
        }
    }
    assert!(parser.finish().is_empty());
    events.extend(decoder.finish());
    events
}

fn accumulate(events: &[StreamEvent]) -> Response {
    let mut accumulator = Accumulator::new();
    for event in events {
        accumulator.push(event);
    }
    assert!(accumulator.is_done());
    assert!(accumulator.error().is_none());
    accumulator.finish()
}

fn assert_lifecycle(events: &[StreamEvent]) {
    let mut open = None;
    let mut next = 0;
    let mut ends = 0;
    for event in events {
        match event {
            StreamEvent::BlockStart { index, .. } => {
                assert!(open.is_none());
                assert_eq!(*index, next);
                next += 1;
                open = Some(*index);
            }
            StreamEvent::Delta { index, .. } => assert_eq!(open, Some(*index)),
            StreamEvent::BlockEnd { index } => assert_eq!(open.take(), Some(*index)),
            StreamEvent::End { .. } => {
                assert!(open.is_none());
                ends += 1;
            }
            _ => {}
        }
    }
    assert_eq!(ends, 1);
    assert!(open.is_none());
}

fn reasoning_chunks() -> Vec<Value> {
    vec![
        chunk(
            json!({"reasoning_details": [
                {"type": "reasoning.text", "text": "План ", "index": 0, "id": "r0", "format": "native-v1", "extra": {"keep": true}},
                {"type": "reasoning.summary", "summary": "Check ", "index": 1, "id": "r1", "format": "native-v1"},
                {"type": "reasoning.encrypted", "data": "ABC/", "index": 2, "id": "r2", "format": "native-v1"},
            ]}),
            Value::Null,
        ),
        chunk(
            json!({"reasoning_details": [
            {"index": 2, "data": "+=XYZ", "future": 42},
            {"index": 0, "text": "🌡"},
            {"index": 1, "summary": "weather."},
        ], "content": "Ready."}),
            Value::Null,
        ),
        chunk(json!({}), json!("stop")),
        json!({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 8, "completion_tokens_details": {"reasoning_tokens": 5}}}),
    ]
}

#[test]
fn structured_reasoning_round_trips_response_and_next_turn() {
    let response = full_response(json!({"content": "Ready.", "reasoning_details": details()}));
    assert_eq!(payload(&response), &details());
    assert_eq!(thoughts(&response), "План 🌡\n\nCheck weather.");
    let Block::Thinking(thinking) = &response.content[0] else {
        panic!()
    };
    assert_eq!(thinking.sig.as_ref().unwrap().namespace, None);

    let encoded = encoded_response(&response);
    assert_eq!(
        encoded["choices"][0]["message"]["reasoning_details"],
        details()
    );
    assert!(
        encoded["choices"][0]["message"]
            .get("reasoning_content")
            .is_none()
    );
    assert_eq!(encoded["extra"], true);
    let decoded = codec::chat::decode_response(encoded).unwrap();
    assert_eq!(response.content, decoded.content);

    for options in [
        EncodeOptions::default(),
        EncodeOptions {
            chat_reasoning_field: Some("reasoning_content".into()),
            ..Default::default()
        },
    ] {
        let encoded = encoded_request(&request(&response), &options);
        assert_eq!(encoded["messages"][0]["reasoning_details"], details());
        assert!(encoded["messages"][0].get("reasoning_content").is_none());
        assert_eq!(encoded["messages"][0]["content"], "Ready.");
    }
}

#[test]
fn plain_reasoning_is_not_duplicated_by_details() {
    let options = EncodeOptions {
        chat_reasoning_field: Some("reasoning_content".into()),
        ..Default::default()
    };
    let response = full_response(
        json!({"reasoning_content": "План 🌡", "reasoning_details": details(), "content": "Ready."}),
    );
    assert_eq!(thoughts(&response), "План 🌡");
    assert_eq!(payload(&response), &details());
    assert_eq!(
        encoded_response(&response)["choices"][0]["message"]["reasoning_content"],
        "План 🌡"
    );
    let encoded = encoded_request(&request(&response), &options);
    assert_eq!(encoded["messages"][0]["reasoning_content"], "План 🌡");
    assert_eq!(encoded["messages"][0]["reasoning_details"], details());
}

#[test]
fn request_details_leave_message_extensions_and_unknown_fields_intact() {
    let input = json!({"model": "model-1", "future": true, "messages": [
        {"role": "assistant", "content": "Ready.", "reasoning_details": details(), "vendor": {"keep": 7}},
    ]});
    let decoded = codec::chat::decode_request(input.clone()).unwrap();
    assert!(
        decoded.messages[0]
            .ext
            .fields
            .get("reasoning_details")
            .is_none()
    );
    assert_eq!(decoded.messages[0].ext.fields["vendor"], json!({"keep": 7}));
    assert_eq!(encoded_request(&decoded, &EncodeOptions::default()), input);
}

#[test]
fn encrypted_only_and_future_detail_shapes_are_preserved() {
    for details in [
        json!([{"type": "reasoning.encrypted", "data": "secret", "index": 0, "extra": true}]),
        json!([{"type": "future", "payload": {"secret": 1}}]),
        json!({"future": "shape"}),
        json!([]),
        Value::Null,
    ] {
        let response = full_response(json!({"content": null, "reasoning_details": details}));
        assert_eq!(thoughts(&response), "");
        assert_eq!(payload(&response), &details);
        assert_eq!(
            encoded_response(&response)["choices"][0]["message"]["reasoning_details"],
            details
        );
        assert_eq!(
            encoded_request(&request(&response), &EncodeOptions::default())["messages"][0]["reasoning_details"],
            details
        );
    }
}

#[test]
fn cross_format_requests_report_loss_and_do_not_leak_encrypted_details() {
    let response = full_response(json!({"content": "Ready.", "reasoning_details": details()}));
    for format in [
        Format::AnthropicMessages,
        Format::OpenAiResponses,
        Format::Gemini,
    ] {
        let encoded = codec::encode_request(
            format,
            &request(&response),
            &EncodeOptions {
                default_max_tokens: Some(1024),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!encoded.losses.is_empty(), "{format:?}");
        let body = String::from_utf8(encoded.body.to_vec()).unwrap();
        assert!(!body.contains("ABC/+=XYZ"), "{format:?}: {body}");
        assert!(!body.contains("reasoning_details"), "{format:?}: {body}");
    }
}

#[test]
fn namespaced_details_are_not_sent_as_native_chat_reasoning() {
    let mut response = full_response(json!({"content": "Ready.", "reasoning_details": details()}));
    let Block::Thinking(thinking) = &mut response.content[0] else {
        panic!()
    };
    thinking.sig.as_mut().unwrap().namespace = Some("another-account".into());
    let encoded = codec::encode_request(
        F,
        &request(&response),
        &EncodeOptions {
            chat_reasoning_field: Some("reasoning_content".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!encoded.losses.is_empty());
    let value: Value = serde_json::from_slice(&encoded.body).unwrap();
    assert!(value["messages"][0].get("reasoning_details").is_none());
    assert!(
        !String::from_utf8(encoded.body.to_vec())
            .unwrap()
            .contains("ABC/+=XYZ")
    );
}

#[test]
fn details_use_the_existing_signature_tunnel_contract() {
    let response = full_response(json!({"content": "Ready.", "reasoning_details": details()}));
    for format in [Format::AnthropicMessages, Format::OpenAiResponses] {
        let mut tunneled = response.clone();
        let Block::Thinking(thinking) = &mut tunneled.content[0] else {
            panic!()
        };
        thinking.sig = Some(thinking.sig.as_ref().unwrap().tunnel("account", format));
        let (bytes, _) = codec::encode_response(format, &tunneled);
        let mut restored = codec::decode_response(format, &bytes).unwrap();
        let Block::Thinking(thinking) = &mut restored.content[0] else {
            panic!()
        };
        let sig = thinking.sig.as_mut().unwrap();
        assert_eq!(sig.format, F);
        assert_eq!(sig.namespace.as_deref(), Some("account"));
        assert_eq!(sig.data["reasoning_details"], details());
        sig.namespace = None;
        assert_eq!(
            encoded_request(&request(&restored), &EncodeOptions::default())["messages"][0]["reasoning_details"],
            details()
        );
    }
}

#[test]
fn split_reasoning_details_survive_any_byte_chunking() {
    let bytes = stream(&reasoning_chunks(), true);
    for size in 1..=64 {
        let events = events(&bytes, size);
        assert_lifecycle(&events);
        let response = accumulate(&events);
        assert_eq!(payload(&response), &details(), "size={size}");
        assert_eq!(thoughts(&response), "План 🌡\n\nCheck weather.");
        assert_eq!(plain_text(&response.content), "Ready.");
        assert_eq!(response.usage.input_total, Some(12));
        assert_eq!(response.usage.output_total, Some(8));
        assert_eq!(response.usage.reasoning, Some(5));
        assert_eq!(
            encoded_response(&response)["choices"][0]["message"]["reasoning_details"],
            details()
        );
        assert_eq!(
            encoded_request(&request(&response), &EncodeOptions::default())["messages"][0]["reasoning_details"],
            details()
        );
    }
}

#[test]
fn plain_and_structured_streams_do_not_duplicate_thoughts() {
    let mut chunks = reasoning_chunks();
    chunks.insert(
        0,
        chunk(json!({"reasoning_content": "Plain "}), Value::Null),
    );
    chunks[1]["choices"][0]["delta"]["reasoning_content"] = json!("thought.");
    let response = accumulate(&events(&stream(&chunks, true), 3));
    assert_eq!(thoughts(&response), "Plain thought.");
    assert_eq!(payload(&response), &details());
    assert_eq!(
        encoded_response(&response)["choices"][0]["message"]["reasoning_content"],
        "Plain thought."
    );
}

#[test]
fn detail_identity_supports_shared_indices_and_id_only_fragments() {
    let chunks = vec![
        chunk(
            json!({"reasoning_details": [
                {"type": "reasoning.summary", "index": 0, "id": "r", "summary": "A"},
                {"type": "reasoning.encrypted", "index": 0, "id": "r", "data": "X"},
                {"type": "reasoning.text", "id": "text", "text": "One"},
            ]}),
            Value::Null,
        ),
        chunk(
            json!({"reasoning_details": [
                {"index": 0, "id": "r", "data": "Y"},
                {"index": 0, "id": "r", "summary": "B"},
                {"id": "text", "text": " two"},
            ]}),
            json!("stop"),
        ),
    ];
    let response = accumulate(&events(&stream(&chunks, true), 7));
    assert_eq!(
        payload(&response),
        &json!([
            {"type": "reasoning.summary", "index": 0, "id": "r", "summary": "AB"},
            {"type": "reasoning.encrypted", "index": 0, "id": "r", "data": "XY"},
            {"type": "reasoning.text", "id": "text", "text": "One two"},
        ])
    );
}

#[test]
fn interleaved_calls_accumulate_names_ids_and_raw_arguments() {
    let chunks = vec![
        chunk(
            json!({"content": "Checking.", "tool_calls": [
                {"index": 1, "id": "call_", "function": {"name": "get_", "arguments": "{\"tz\":"}},
                {"index": 0, "id": "weather", "function": {"name": "get_weather", "arguments": "{\"city\":"}},
                {"index": 2, "function": {"name": "empty", "arguments": ""}},
            ]}),
            Value::Null,
        ),
        chunk(
            json!({"tool_calls": [
                {"index": 0, "function": {"arguments": "\"Paris\""}},
                {"index": 1, "id": "time", "function": {"name": "time", "arguments": "\"CET\""}},
            ]}),
            Value::Null,
        ),
        chunk(
            json!({"tool_calls": [
                {"index": 1, "function": {"arguments": "}"}},
                {"index": 0, "function": {"arguments": "}"}},
            ]}),
            json!("tool_calls"),
        ),
        json!({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 8}}),
    ];
    for size in 1..=64 {
        let events = events(&stream(&chunks, true), size);
        assert_lifecycle(&events);
        let response = accumulate(&events);
        let calls: Vec<_> = response
            .content
            .iter()
            .filter_map(|block| match block {
                Block::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 3);
        assert_eq!(
            (&calls[0].id, &calls[0].name, &calls[0].args),
            (
                &"weather".into(),
                &"get_weather".into(),
                &"{\"city\":\"Paris\"}".into()
            )
        );
        assert_eq!(
            (&calls[1].id, &calls[1].name, &calls[1].args),
            (
                &"call_time".into(),
                &"get_time".into(),
                &"{\"tz\":\"CET\"}".into()
            )
        );
        assert_eq!(calls[2].id, "call_0_3");
        assert_eq!(calls[2].args, "{}");
        assert_eq!(plain_text(&response.content), "Checking.");
        assert_eq!(response.stop, Some(StopReason::ToolUse));
        assert_eq!(response.usage.output_total, Some(8));
        let encoded = encoded_request(&request(&response), &EncodeOptions::default());
        assert_eq!(
            encoded["messages"][0]["tool_calls"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
    }
}

#[test]
fn structured_reasoning_and_interleaved_calls_share_a_valid_lifecycle() {
    let mut chunks = reasoning_chunks();
    chunks[1]["choices"][0]["delta"]["tool_calls"] = json!([
        {"index": 0, "id": "a", "function": {"name": "first", "arguments": "{\"a\":"}},
        {"index": 1, "id": "b", "function": {"name": "second", "arguments": "{\"b\":"}},
    ]);
    chunks[2] = chunk(
        json!({"tool_calls": [
            {"index": 1, "function": {"arguments": "2}"}},
            {"index": 0, "function": {"arguments": "1}"}},
        ]}),
        json!("tool_calls"),
    );
    let input = events(&stream(&chunks, true), 1);
    assert_lifecycle(&input);
    let response = accumulate(&input);
    assert_eq!(payload(&response), &details());
    let calls: Vec<_> = response
        .content
        .iter()
        .filter_map(|block| match block {
            Block::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].args, "{\"a\":1}");
    assert_eq!(calls[1].args, "{\"b\":2}");
    let mut encoder = codec::chat::Encoder::default();
    let bytes: Vec<_> = input
        .iter()
        .flat_map(|event| encoder.encode(event))
        .flatten()
        .collect();
    let back = accumulate(&events(&bytes, 3));
    assert_eq!(back.content, response.content);
    assert_eq!(back.stop, Some(StopReason::ToolUse));
}

#[test]
fn finish_reason_waits_for_usage_and_eof_flushes_without_done() {
    let mut decoder = codec::chat::Decoder::default();
    let mut accumulator = Accumulator::new();
    for chunk in reasoning_chunks() {
        for event in decoder.feed(&event(chunk)).unwrap() {
            assert!(!matches!(event, StreamEvent::End { .. }));
            accumulator.push(&event);
        }
        assert!(!accumulator.is_done());
    }
    let last = decoder.finish();
    assert_eq!(last.len(), 1);
    accumulator.push(&last[0]);
    assert!(accumulator.is_done());
    assert_eq!(accumulator.response().usage.reasoning, Some(5));
    assert!(decoder.finish().is_empty());

    let chunks = [chunk(
        json!({"tool_calls": [{"index": 0, "id": "call", "function": {"name": "tool", "arguments": "{bad"}}]}),
        Value::Null,
    )];
    let events = events(&stream(&chunks, false), 2);
    assert_lifecycle(&events);
    let response = accumulate(&events);
    let Block::ToolCall(call) = &response.content[0] else {
        panic!()
    };
    assert_eq!(call.args, "{bad");
}

#[test]
fn chat_stream_encoding_preserves_details_without_synthetic_plain_reasoning() {
    let input = events(&stream(&reasoning_chunks(), true), 7);
    let mut encoder = codec::chat::Encoder::default();
    let mut bytes = Vec::new();
    for event in &input {
        for chunk in encoder.encode(event) {
            bytes.extend_from_slice(&chunk);
        }
    }
    assert!(encoder.finish().is_empty());
    let text = String::from_utf8(bytes.clone()).unwrap();
    assert!(!text.contains("reasoning_content"));
    assert_eq!(text.matches("reasoning_details").count(), 1);
    let back = accumulate(&events(&bytes, 1));
    let original = accumulate(&input);
    assert_eq!(back.content, original.content);
    assert_eq!(back.usage.input_total, original.usage.input_total);
    assert_eq!(back.usage.output_total, original.usage.output_total);
    assert_eq!(back.usage.reasoning, original.usage.reasoning);
}

#[test]
fn plain_reasoning_remains_live_and_encodes_on_implicit_block_end() {
    for field in ["reasoning_content", "reasoning"] {
        let mut decoder = codec::chat::Decoder::default();
        let mut delta = json!({});
        delta[field] = json!("Think.");
        let first = decoder.feed(&event(chunk(delta, Value::Null))).unwrap();
        assert!(first.iter().any(|event| matches!(event, StreamEvent::Delta { delta: Delta::Thinking(text), .. } if text == "Think.")));
        let mut encoder = codec::chat::Encoder::default();
        let mut bytes = Vec::new();
        for event in &first {
            for chunk in encoder.encode(event) {
                bytes.extend_from_slice(&chunk);
            }
        }
        for chunk in encoder.finish() {
            bytes.extend_from_slice(&chunk);
        }
        assert_eq!(thoughts(&accumulate(&events(&bytes, 1))), "Think.");
    }
}

#[test]
fn malformed_frames_and_provider_errors_keep_existing_behavior() {
    assert!(
        codec::chat::decode_response(json!({"choices": []}))
            .unwrap_err()
            .0
            .contains("no choices")
    );
    let mut decoder = codec::chat::Decoder::default();
    let malformed = SseParser::new().feed(b"data: {bad\n\n").pop().unwrap();
    assert!(
        decoder
            .feed(&malformed)
            .unwrap_err()
            .0
            .contains("invalid JSON")
    );
    let error =
        json!({"error": {"message": "quota", "type": "rate_limit_error", "code": "too_many"}});
    let result = decoder.feed(&event(error.clone())).unwrap();
    let [StreamEvent::Error(error)] = result.as_slice() else {
        panic!()
    };
    assert_eq!(error.message, "quota");
    assert_eq!(error.code.as_deref(), Some("rate_limit_error"));
    assert_eq!(
        error.raw,
        serde_json::to_vec(
            &json!({"error": {"message": "quota", "type": "rate_limit_error", "code": "too_many"}})
        )
        .unwrap()
    );
}
