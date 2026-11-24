use ncore_core::codec::{self, EncodeOptions};
use ncore_core::ir::*;
use ncore_core::{Format, SseParser};
use serde_json::{Value, json};

const ALL: [Format; 4] = [
    Format::AnthropicMessages,
    Format::OpenAiChat,
    Format::OpenAiResponses,
    Format::Gemini,
];

fn opts() -> EncodeOptions {
    EncodeOptions {
        default_max_tokens: Some(4096),
        ..Default::default()
    }
}

// Claude Code style turn: cached system, tools, signed thinking, a tool round trip.
fn anthropic_request() -> Value {
    json!({
        "model": "claude-x",
        "max_tokens": 2048,
        "system": [{"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Need the weather.", "signature": "sig-anthropic-1"},
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"},
                {"type": "text", "text": "Thanks", "cache_control": {"type": "ephemeral"}}
            ]}
        ],
        "tools": [{"name": "get_weather", "description": "Weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
        "tool_choice": {"type": "auto"},
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "metadata": {"user_id": "u1"},
        "stream": true
    })
}

fn decode(format: Format, path: &str, v: &Value) -> Request {
    codec::decode_request(format, path, &serde_json::to_vec(v).unwrap()).unwrap()
}

fn encode(format: Format, req: &Request) -> (Value, codec::Encoded) {
    let e = codec::encode_request(format, req, &opts()).unwrap();
    (serde_json::from_slice(&e.body).unwrap(), e)
}

#[test]
fn anthropic_round_trip_is_exact() {
    let src = anthropic_request();
    let req = decode(Format::AnthropicMessages, "/v1/messages", &src);
    let (out, e) = encode(Format::AnthropicMessages, &req);
    assert!(e.losses.is_empty(), "{:?}", e.losses);
    let mut expected = src.clone();
    expected["messages"][0]["content"] = json!([{"type": "text", "text": "Weather in Paris?"}]);
    assert_eq!(out, expected);
}

#[test]
fn anthropic_to_chat() {
    let req = decode(
        Format::AnthropicMessages,
        "/v1/messages",
        &anthropic_request(),
    );
    let (out, e) = encode(Format::OpenAiChat, &req);
    assert_eq!(
        out,
        json!({
            "model": "claude-x",
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": "Checking.", "tool_calls": [
                    {"id": "toolu_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "toolu_1", "content": "18C"},
                {"role": "user", "content": "Thanks"}
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
            "tool_choice": "auto",
            "max_completion_tokens": 2048,
            "reasoning_effort": "low",
            "stream": true,
            "stream_options": {"include_usage": true}
        })
    );
    let paths: Vec<&str> = e.losses.0.iter().map(|l| l.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "/cache",
            "/messages/1/content/0",
            "/reasoning_effort",
            "/metadata"
        ]
    );
}

#[test]
fn chat_cache_breakpoints_when_enabled() {
    let req = decode(
        Format::AnthropicMessages,
        "/v1/messages",
        &anthropic_request(),
    );
    let o = EncodeOptions {
        openai_cache_breakpoints: true,
        ..opts()
    };
    let e = codec::encode_request(Format::OpenAiChat, &req, &o).unwrap();
    let out: Value = serde_json::from_slice(&e.body).unwrap();
    assert_eq!(
        out["messages"][0]["content"][0]["prompt_cache_breakpoint"],
        json!({"mode": "explicit"})
    );
    assert_eq!(
        out["messages"][4]["content"][0]["prompt_cache_breakpoint"],
        json!({"mode": "explicit"})
    );
}

#[test]
fn anthropic_to_responses() {
    let req = decode(
        Format::AnthropicMessages,
        "/v1/messages",
        &anthropic_request(),
    );
    let (out, _) = encode(Format::OpenAiResponses, &req);
    assert_eq!(out["instructions"], json!("You are terse."));
    assert_eq!(
        out["input"],
        json!([
            {"type": "message", "role": "user", "content": "Weather in Paris?"},
            {"type": "message", "role": "assistant", "content": "Checking."},
            {"type": "function_call", "call_id": "toolu_1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call_output", "call_id": "toolu_1", "output": "18C"},
            {"type": "message", "role": "user", "content": "Thanks"}
        ])
    );
    assert_eq!(out["tools"][0]["strict"], json!(false));
    assert_eq!(out["store"], json!(false));
    assert_eq!(out["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(out["max_output_tokens"], json!(2048));
}

#[test]
fn anthropic_to_gemini() {
    let req = decode(
        Format::AnthropicMessages,
        "/v1/messages",
        &anthropic_request(),
    );
    let (out, e) = encode(Format::Gemini, &req);
    assert_eq!(
        e.path,
        "/v1beta/models/claude-x:streamGenerateContent?alt=sse"
    );
    assert_eq!(
        out["contents"],
        json!([
            {"role": "user", "parts": [{"text": "Weather in Paris?"}]},
            {"role": "model", "parts": [
                {"text": "Checking."},
                {"functionCall": {"id": "toolu_1", "name": "get_weather", "args": {"city": "Paris"}}, "thoughtSignature": "skip_thought_signature_validator"}
            ]},
            {"role": "user", "parts": [
                {"functionResponse": {"id": "toolu_1", "name": "get_weather", "response": {"output": "18C"}}},
                {"text": "Thanks"}
            ]}
        ])
    );
    assert_eq!(
        out["systemInstruction"],
        json!({"parts": [{"text": "You are terse."}]})
    );
    assert_eq!(
        out["generationConfig"],
        json!({"maxOutputTokens": 2048, "thinkingConfig": {"thinkingBudget": 1024}})
    );
    assert_eq!(
        out["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]["required"],
        json!(["city"])
    );
}

#[test]
fn every_format_round_trips_semantically() {
    let base = decode(
        Format::AnthropicMessages,
        "/v1/messages",
        &anthropic_request(),
    );
    for f in ALL {
        let (first, _) = encode(f, &base);
        let path = codec::encode_request(f, &base, &opts()).unwrap().path;
        let ir = decode(f, &path, &first);
        let (second, e) = encode(f, &ir);
        assert_eq!(first, second, "{f:?} is not stable");
        assert!(
            e.losses.0.iter().all(|l| l.path == "/cache"),
            "{f:?}: {:?}",
            e.losses
        );
    }
}

#[test]
fn chat_request_to_anthropic() {
    let chat = json!({
        "model": "gpt-x",
        "messages": [
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "Look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}}]},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{\"x\":1}"}},
                {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "not json"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "one"},
            {"role": "tool", "tool_call_id": "c2", "content": "two"},
            {"role": "system", "content": "Now answer."}
        ],
        "temperature": 0.2,
        "stop": "END",
        "parallel_tool_calls": false,
        "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object"}}}
    });
    let req = decode(Format::OpenAiChat, "/v1/chat/completions", &chat);
    let (out, e) = encode(Format::AnthropicMessages, &req);
    assert_eq!(
        out,
        json!({
            "model": "gpt-x",
            "max_tokens": 4096,
            "system": "Be brief.",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Look"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
                ]},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "c1", "name": "a", "input": {"x": 1}},
                    {"type": "tool_use", "id": "c2", "name": "b", "input": {"raw": "not json"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "c1", "content": "one"},
                    {"type": "tool_result", "tool_use_id": "c2", "content": "two"}
                ]},
                {"role": "user", "content": [{"type": "text", "text": "Now answer."}]}
            ],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
            "stop_sequences": ["END"],
            "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}}
        })
    );
    let paths: Vec<&str> = e.losses.0.iter().map(|l| l.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "/messages/0/content/1/detail",
            "/messages/1/content/1/input",
            "/messages/3",
            "/temperature"
        ]
    );
}

#[test]
fn gemini_request_ids_pair_up() {
    let g = json!({
        "contents": [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [
                {"functionCall": {"name": "f", "args": {"a": 1}}, "thoughtSignature": "sig"},
                {"functionCall": {"name": "g", "args": {}}}
            ]},
            {"role": "user", "parts": [
                {"functionResponse": {"name": "f", "response": {"output": "ok"}}},
                {"functionResponse": {"name": "g", "response": {"error": "boom"}}}
            ]}
        ],
        "generationConfig": {"thinkingConfig": {"thinkingLevel": "LOW", "includeThoughts": true}, "candidateCount": 1}
    });
    let req = decode(
        Format::Gemini,
        "/v1beta/models/gemini-x:generateContent",
        &g,
    );
    assert_eq!(req.model, "gemini-x");
    assert!(!req.stream);
    let (out, _) = encode(Format::AnthropicMessages, &req);
    assert_eq!(out["messages"][1]["content"][0]["id"], "call_1_0");
    assert_eq!(out["messages"][2]["content"][0]["tool_use_id"], "call_1_0");
    assert_eq!(out["messages"][2]["content"][1]["tool_use_id"], "call_1_1");
    assert_eq!(out["messages"][2]["content"][1]["is_error"], true);
    assert_eq!(out["output_config"]["effort"], "low");
    let (back, _) = encode(Format::Gemini, &req);
    assert_eq!(back["contents"][1]["parts"][0]["thoughtSignature"], "sig");
    assert_eq!(back["generationConfig"]["candidateCount"], 1);
}

// ---- streams ----

fn events(format: Format, sse: &[u8]) -> Vec<StreamEvent> {
    let mut parser = SseParser::new();
    let mut dec = codec::stream_decoder(format);
    let mut out = Vec::new();
    for chunk in sse.chunks(23) {
        for ev in parser.feed(chunk) {
            out.extend(dec.feed(&ev).unwrap());
        }
    }
    out.extend(dec.finish());
    out
}

fn accumulate(evs: &[StreamEvent]) -> Response {
    let mut acc = Accumulator::new();
    for e in evs {
        acc.push(e);
    }
    acc.finish()
}

fn fixture(format: Format) -> &'static [u8] {
    match format {
        Format::AnthropicMessages => include_bytes!("fixtures/translate/anthropic_tools.sse"),
        Format::OpenAiChat => include_bytes!("fixtures/translate/chat_tools.sse"),
        Format::OpenAiResponses => include_bytes!("fixtures/translate/responses_tools.sse"),
        Format::Gemini => include_bytes!("fixtures/translate/gemini_tools.sse"),
    }
}

// What survives any hop: reasoning text, text, tool names and arguments.
fn essence(r: &Response) -> Vec<String> {
    r.content
        .iter()
        .map(|b| match b {
            Block::Text { text, .. } => format!("text:{text}"),
            Block::Thinking(t) => format!("think:{}", t.text),
            Block::ToolCall(c) => {
                let args: Value = serde_json::from_str(&c.args).unwrap();
                format!("call:{}:{args}", c.name)
            }
            other => format!("{other:?}"),
        })
        .collect()
}

#[test]
fn fixture_streams_decode() {
    let weather = "call:get_weather:{\"city\":\"Paris\"}";
    for f in ALL {
        let r = accumulate(&events(f, fixture(f)));
        let mut want = vec!["think:Need the weather.", "text:Checking.", weather];
        if f == Format::OpenAiChat {
            want.push("call:get_time:{\"tz\":\"CET\"}");
        }
        assert_eq!(essence(&r), want, "{f:?}");
        assert_eq!(r.stop, Some(StopReason::ToolUse), "{f:?}");
        assert_eq!(r.usage.cache_read, Some(100), "{f:?}");
        assert_eq!(r.usage.input_total, Some(112), "{f:?}");
        assert_eq!(r.usage.output_total, Some(40), "{f:?}");
    }
    let a = accumulate(&events(
        Format::AnthropicMessages,
        fixture(Format::AnthropicMessages),
    ));
    let Block::Thinking(t) = &a.content[0] else {
        panic!()
    };
    assert_eq!(
        t.sig,
        Some(Opaque::new(
            Format::AnthropicMessages,
            json!("sig-anthropic-1")
        ))
    );
    let g = accumulate(&events(Format::Gemini, fixture(Format::Gemini)));
    let Block::ToolCall(c) = &g.content[2] else {
        panic!()
    };
    assert_eq!(
        c.sig,
        Some(Opaque::new(Format::Gemini, json!("sig-gemini-1")))
    );
    assert_eq!(c.id, "r-t1_2");
}

#[test]
fn every_stream_pair_preserves_content() {
    for src in ALL {
        let evs = events(src, fixture(src));
        let want = essence(&accumulate(&evs));
        for dst in ALL {
            let mut enc = codec::stream_encoder(dst);
            let mut bytes = Vec::new();
            for e in &evs {
                for b in enc.encode(e) {
                    bytes.extend_from_slice(&b);
                }
            }
            for b in enc.finish() {
                bytes.extend_from_slice(&b);
            }
            let back = accumulate(&events(dst, &bytes));
            let mut got = essence(&back);
            let mut want = want.clone();
            got.retain(|s| s != "text:");
            want.retain(|s| s != "text:");
            assert_eq!(
                got,
                want,
                "{src:?} -> {dst:?}\n{}",
                String::from_utf8_lossy(&bytes)
            );
            assert_eq!(back.stop, Some(StopReason::ToolUse), "{src:?} -> {dst:?}");
            assert_eq!(back.usage.input_total, Some(112), "{src:?} -> {dst:?}");
            assert_eq!(back.usage.cache_read, Some(100), "{src:?} -> {dst:?}");
            assert_eq!(back.usage.output_total, Some(40), "{src:?} -> {dst:?}");
        }
    }
}

#[test]
fn stream_and_full_response_agree() {
    for f in ALL {
        let streamed = accumulate(&events(f, fixture(f)));
        let (body, _) = codec::encode_response(f, &streamed);
        let full = codec::decode_response(f, &body).unwrap();
        assert_eq!(essence(&full), essence(&streamed), "{f:?}");
        assert_eq!(full.stop, streamed.stop, "{f:?}");
        assert_eq!(full.usage.input_total, streamed.usage.input_total, "{f:?}");
    }
}

#[test]
fn encoded_streams_are_well_formed() {
    let evs = events(
        Format::AnthropicMessages,
        fixture(Format::AnthropicMessages),
    );
    let mut enc = codec::stream_encoder(Format::OpenAiResponses);
    let mut seqs = Vec::new();
    let mut types = Vec::new();
    for e in &evs {
        for b in enc.encode(e) {
            for ev in SseParser::new().feed(&b) {
                let v: Value = serde_json::from_str(&ev.data).unwrap();
                assert_eq!(ev.event.as_deref(), v["type"].as_str());
                seqs.push(v["sequence_number"].as_u64().unwrap());
                types.push(v["type"].as_str().unwrap().to_owned());
            }
        }
    }
    assert_eq!(seqs, (0..seqs.len() as u64).collect::<Vec<_>>());
    assert_eq!(types.first().map(String::as_str), Some("response.created"));
    assert_eq!(types.last().map(String::as_str), Some("response.completed"));

    let mut enc = codec::stream_encoder(Format::AnthropicMessages);
    let chat = events(Format::OpenAiChat, fixture(Format::OpenAiChat));
    let mut names = Vec::new();
    for e in &chat {
        for b in enc.encode(e) {
            for ev in SseParser::new().feed(&b) {
                names.push(ev.event.unwrap());
            }
        }
    }
    assert_eq!(names.first().map(String::as_str), Some("message_start"));
    assert_eq!(names.last().map(String::as_str), Some("message_stop"));
    let starts = names.iter().filter(|n| *n == "content_block_start").count();
    let stops = names.iter().filter(|n| *n == "content_block_stop").count();
    assert_eq!((starts, stops), (4, 4));
}

// MCP screenshot tool: text plus an image in one tool result.
fn image_tool_result() -> Value {
    json!({
        "model": "claude-x",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Take a screenshot"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "shot", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [
                    {"type": "text", "text": "done"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
                ]},
                {"type": "text", "text": "What is on it?"}
            ]}
        ]
    })
}

#[test]
fn tool_result_images_reach_every_format() {
    let req = decode(
        Format::AnthropicMessages,
        "/v1/messages",
        &image_tool_result(),
    );
    let (chat, e) = encode(Format::OpenAiChat, &req);
    let m = chat["messages"].as_array().unwrap();
    assert_eq!(
        m[2],
        json!({"role": "tool", "tool_call_id": "t1", "content": "done"})
    );
    assert_eq!(m[3]["role"], "user");
    let parts = m[3]["content"].as_array().unwrap();
    assert_eq!(parts[0]["text"], "[attachments returned by tool call t1]");
    assert_eq!(
        parts[1]["image_url"]["url"],
        "data:image/png;base64,iVBORw0KGgo="
    );
    assert_eq!(parts[2]["text"], "What is on it?");
    assert!(
        e.losses.0.iter().all(|l| l.kind == LossKind::Approximated),
        "{:?}",
        e.losses
    );

    let (gem, e) = encode(Format::Gemini, &req);
    let parts = gem["contents"][2]["parts"].as_array().unwrap();
    assert_eq!(parts[0]["functionResponse"]["response"]["output"], "done");
    assert_eq!(parts[1]["text"], "[attachments returned by tool call t1]");
    assert_eq!(parts[2]["inlineData"]["mimeType"], "image/png");
    assert_eq!(parts[3]["text"], "What is on it?");
    assert!(
        e.losses.0.iter().all(|l| l.kind == LossKind::Approximated),
        "{:?}",
        e.losses
    );

    let (resp, e) = encode(Format::OpenAiResponses, &req);
    let out = resp["input"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["type"] == "function_call_output")
        .unwrap();
    assert_eq!(out["output"][1]["type"], "input_image");
    assert!(e.losses.is_empty(), "{:?}", e.losses);
}
