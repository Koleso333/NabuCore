use std::collections::HashMap;

use ncore_core::Format;
use ncore_core::Usage;
use ncore_core::cache::{
    BreakReason, CachePolicy, CacheTracker, Fingerprint, fingerprint, plan,
};
use ncore_core::codec::{EncodeOptions, encode_request};
use ncore_core::ir::{
    Block, CacheMark, Losses, Message, Opaque, Request, Role, Thinking, Tool, ToolCall, ToolChoice,
    ToolDef, ToolResult,
};
use proptest::prelude::*;
use serde_json::{Value, json};

const ALL: [Format; 4] = [
    Format::AnthropicMessages,
    Format::OpenAiChat,
    Format::OpenAiResponses,
    Format::Gemini,
];

fn opts() -> EncodeOptions {
    EncodeOptions {
        default_max_tokens: Some(1000),
        ..Default::default()
    }
}

fn tool(name: &str) -> ToolDef {
    ToolDef::Function(Tool {
        name: name.into(),
        description: Some(format!("{name} tool")),
        schema: json!({"type": "object", "properties": {"x": {"type": "string"}}}),
        strict: None,
        cache: None,
        ext: Default::default(),
    })
}

fn text(t: &str) -> Block {
    Block::text(t)
}

fn marked_text(t: &str, ttl: Option<&str>) -> Block {
    Block::Text {
        text: t.into(),
        cache: Some(CacheMark {
            ttl: ttl.map(str::to_owned),
        }),
        sig: None,
        ext: Default::default(),
    }
}

fn call(id: &str) -> Block {
    Block::ToolCall(ToolCall {
        id: id.into(),
        name: "a".into(),
        args: r#"{"x":"1"}"#.into(),
        sig: None,
    })
}

fn result(id: &str) -> Block {
    Block::ToolResult(ToolResult {
        call_id: id.into(),
        name: Some("a".into()),
        content: vec![text("ok")],
        is_error: false,
        cache: None,
    })
}

fn thinking() -> Block {
    Block::Thinking(Thinking {
        text: "hmm".into(),
        sig: Some(Opaque::new(Format::AnthropicMessages, json!("sig"))),
        redacted: false,
    })
}

fn user(content: Vec<Block>) -> Message {
    Message::new(Role::User, content)
}

fn assistant(content: Vec<Block>) -> Message {
    Message::new(Role::Assistant, content)
}

fn request(tools: usize, system: &[&str], messages: Vec<Message>) -> Request {
    Request {
        model: "m".into(),
        system: system.iter().map(|s| text(s)).collect(),
        tools: (0..tools).map(|i| tool(&format!("t{i}"))).collect(),
        messages,
        max_tokens: Some(100),
        ..Default::default()
    }
}

fn planned(format: Format, req: &Request, policy: &CachePolicy) -> (Request, Vec<String>, Losses) {
    let mut r = req.clone();
    let mut losses = Losses::default();
    let p = plan(format, &mut r, policy, &opts(), &mut losses);
    (r, p.added, losses)
}

fn body(format: Format, req: &Request) -> Value {
    serde_json::from_slice(&encode_request(format, req, &opts()).unwrap().body).unwrap()
}

/// Paths of `cache_control` marks in an encoded Anthropic body.
fn marks(body: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let mut visit = |path: String, v: &Value| {
        if v.get("cache_control").is_some() {
            out.push(path);
        }
    };
    for (i, t) in body["tools"].as_array().into_iter().flatten().enumerate() {
        visit(format!("/tools/{i}"), t);
    }
    for (i, b) in body["system"].as_array().into_iter().flatten().enumerate() {
        visit(format!("/system/{i}"), b);
    }
    for (m, msg) in body["messages"].as_array().unwrap().iter().enumerate() {
        for (i, b) in msg["content"].as_array().into_iter().flatten().enumerate() {
            visit(format!("/messages/{m}/content/{i}"), b);
        }
    }
    out
}

#[test]
fn default_plan_marks_tools_system_and_tail() {
    let req = request(
        3,
        &["base", "project"],
        vec![
            user(vec![text("go")]),
            assistant(vec![thinking(), text("calling"), call("a"), call("b")]),
            user(vec![result("a"), result("b")]),
        ],
    );
    let (r, added, losses) = planned(Format::AnthropicMessages, &req, &CachePolicy::default());
    assert!(losses.is_empty());
    // The tail of a tool_result run is its last result.
    assert_eq!(added, ["/tools/2", "/system/1", "/messages/2/content/1"]);
    let b = body(Format::AnthropicMessages, &r);
    assert_eq!(marks(&b), added);
    assert_eq!(
        b["system"][1]["cache_control"],
        json!({"type": "ephemeral"})
    );

    // Planning twice changes nothing.
    let (again, added2, _) = planned(Format::AnthropicMessages, &r, &CachePolicy::default());
    assert!(added2.is_empty());
    assert_eq!(again, r);

    // Without a system prompt the last tool is the stable prefix.
    let req = request(2, &[], vec![user(vec![text("go")])]);
    let (_, added, _) = planned(Format::AnthropicMessages, &req, &CachePolicy::default());
    assert_eq!(added, ["/tools/1", "/messages/0/content/0"]);

    // Nothing cacheable at the end: the tail walks back past thinking and tool_use.
    let req = request(
        0,
        &[],
        vec![
            user(vec![text("go"), text("")]),
            assistant(vec![thinking(), call("a")]),
        ],
    );
    let (_, added, _) = planned(Format::AnthropicMessages, &req, &CachePolicy::default());
    assert_eq!(added, ["/messages/0/content/0"]);
}

#[test]
fn host_marks_are_kept_and_capped() {
    let mut req = request(
        1,
        &["s"],
        vec![
            user(vec![marked_text("a", None), marked_text("b", None)]),
            assistant(vec![marked_text("c", None)]),
            user(vec![marked_text("d", None), marked_text("e", None)]),
        ],
    );
    let (r, added, losses) = planned(Format::AnthropicMessages, &req, &CachePolicy::default());
    assert!(added.is_empty());
    assert_eq!(losses.0.len(), 1);
    assert_eq!(losses.0[0].path, "/messages/2/content/0/cache");
    // The first three and the last stay.
    assert_eq!(
        marks(&body(Format::AnthropicMessages, &r)),
        [
            "/messages/0/content/0",
            "/messages/0/content/1",
            "/messages/1/content/0",
            "/messages/2/content/1"
        ]
    );

    // A mark on an empty block never reaches the provider and takes no slot.
    req.messages = vec![user(vec![text("x"), marked_text("", None)])];
    let (_, added, losses) = planned(Format::AnthropicMessages, &req, &CachePolicy::default());
    assert!(losses.is_empty());
    assert_eq!(added, ["/tools/0", "/system/0", "/messages/0/content/0"]);

    // OpenAI keeps three explicit marks next to its implicit one; Gemini adds nothing.
    let req = request(
        1,
        &["s"],
        vec![user(
            (0..5).map(|i| marked_text(&i.to_string(), None)).collect(),
        )],
    );
    let mut on = opts();
    on.openai_cache_breakpoints = true;
    let mut r = req.clone();
    let mut losses = Losses::default();
    let p = plan(
        Format::OpenAiChat,
        &mut r,
        &CachePolicy::default(),
        &on,
        &mut losses,
    );
    assert!(p.added.is_empty());
    assert_eq!(losses.0.len(), 2);
    let (r, added, losses) = planned(
        Format::Gemini,
        &request(2, &["s"], vec![user(vec![text("x")])]),
        &CachePolicy::default(),
    );
    assert!(added.is_empty() && losses.is_empty());
    assert!(r.messages[0].content[0].cache().is_none());
}

#[test]
fn ttls_stay_in_provider_order() {
    let policy = CachePolicy {
        ttl: Some("1h".into()),
        ..CachePolicy::default()
    };
    let req = request(
        1,
        &["s"],
        vec![
            user(vec![text("q")]),
            assistant(vec![text("a")]),
            user(vec![marked_text("x", None), text("y")]),
        ],
    );
    let (r, added, losses) = planned(Format::AnthropicMessages, &req, &policy);
    assert_eq!(added, ["/tools/0", "/system/0", "/messages/2/content/1"]);
    let b = body(Format::AnthropicMessages, &r);
    assert_eq!(b["tools"][0]["cache_control"]["ttl"], "1h");
    assert_eq!(b["system"][0]["cache_control"]["ttl"], "1h");
    // A 1h mark after the host's 5m one would be rejected.
    assert!(
        b["messages"][2]["content"][1]["cache_control"]
            .get("ttl")
            .is_none()
    );
    assert_eq!(losses.0.len(), 1);
    assert_eq!(losses.0[0].path, "/messages/2/content/1/cache/ttl");
}

#[test]
fn bridges_keep_long_histories_inside_the_window() {
    // One position per message after the system block at position 0.
    let messages = (0..50)
        .map(|i| {
            if i % 2 == 0 {
                user(vec![text(&format!("u{i}"))])
            } else {
                assistant(vec![text(&format!("a{i}"))])
            }
        })
        .collect();
    let req = request(0, &["s"], messages);
    let (_, added, _) = planned(Format::AnthropicMessages, &req, &CachePolicy::default());
    assert_eq!(
        added,
        [
            "/system/0",
            "/messages/9/content/0",
            "/messages/29/content/0",
            "/messages/49/content/0"
        ]
    );

    // A run of tool calls or results is one position, so no bridge is needed.
    let calls: Vec<Block> = (0..30).map(|i| call(&i.to_string())).collect();
    let results: Vec<Block> = (0..30).map(|i| result(&i.to_string())).collect();
    let req = request(
        0,
        &["s"],
        vec![user(vec![text("go")]), assistant(calls), user(results)],
    );
    let (_, added, _) = planned(Format::AnthropicMessages, &req, &CachePolicy::default());
    assert_eq!(added, ["/system/0", "/messages/2/content/29"]);
}

// ---- Anthropic cache simulation ----

/// Lookback positions of an encoded Anthropic body: (hash of the prefix ending there, marked).
fn sim_positions(body: &Value) -> Vec<(u64, bool)> {
    let mut out: Vec<(u64, bool)> = Vec::new();
    // Rolling FNV-1a over the unmarked blocks, so equal prefixes hash alike.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut push = |v: &Value, merge: bool, out: &mut Vec<(u64, bool)>| {
        let mut clean = v.clone();
        clean.as_object_mut().unwrap().remove("cache_control");
        for b in serde_json::to_vec(&clean).unwrap() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        let marked = v.get("cache_control").is_some();
        match out.last_mut() {
            Some(last) if merge => *last = (h, marked),
            _ => out.push((h, marked)),
        }
    };
    for t in body["tools"].as_array().into_iter().flatten() {
        push(t, false, &mut out);
    }
    for b in body["system"].as_array().into_iter().flatten() {
        push(b, false, &mut out);
    }
    let mut run = "";
    for msg in body["messages"].as_array().unwrap() {
        for b in msg["content"].as_array().unwrap() {
            let ty = b["type"].as_str().unwrap();
            let merge = (ty == "tool_use" || ty == "tool_result") && ty == run;
            run = if ty == "tool_use" || ty == "tool_result" {
                ty
            } else {
                ""
            };
            push(b, merge, &mut out);
        }
    }
    out
}

/// Reads then writes as Anthropic documents; returns the longest cached prefix found.
fn sim_request(entries: &mut HashMap<usize, u64>, body: &Value) -> Option<usize> {
    let ps = sim_positions(body);
    let marks: Vec<usize> = (0..ps.len()).filter(|&i| ps[i].1).collect();
    assert!(marks.len() <= 4, "too many breakpoints: {marks:?}");
    let mut hit = None;
    for &m in &marks {
        for p in (m.saturating_sub(19)..=m).rev() {
            if entries.get(&p) == Some(&ps[p].0) {
                hit = hit.max(Some(p));
                break;
            }
        }
    }
    for &m in &marks {
        entries.insert(m, ps[m].0);
    }
    hit
}

#[derive(Debug, Clone)]
enum Turn {
    Chat,
    Tools(usize),
    Thinking(usize),
}

fn append(msgs: &mut Vec<Message>, turn: &Turn, n: usize) {
    match turn {
        Turn::Chat => {
            msgs.push(assistant(vec![text(&format!("answer {n}"))]));
            msgs.push(user(vec![text(&format!("question {n}"))]));
        }
        Turn::Tools(k) | Turn::Thinking(k) => {
            let mut a = Vec::new();
            if matches!(turn, Turn::Thinking(_)) {
                a.push(thinking());
            }
            a.extend((0..*k).map(|i| call(&format!("{n}_{i}"))));
            msgs.push(assistant(a));
            msgs.push(user((0..*k).map(|i| result(&format!("{n}_{i}"))).collect()));
        }
    }
}

fn turn() -> impl Strategy<Value = Turn> {
    prop_oneof![
        Just(Turn::Chat),
        (1..4usize).prop_map(Turn::Tools),
        (1..3usize).prop_map(Turn::Thinking),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Every request finds the previous request's tail while the history grows by fewer
    /// positions than the tail and one bridge can reach.
    #[test]
    fn cache_hits_follow_the_conversation(
        tools in 0..3usize,
        system in 0..3usize,
        turns in proptest::collection::vec(proptest::collection::vec(turn(), 1..12), 1..8),
    ) {
        let sys: Vec<String> = (0..system).map(|i| format!("system {i}")).collect();
        let sys: Vec<&str> = sys.iter().map(String::as_str).collect();
        let mut msgs = vec![user(vec![text("start")])];
        let mut entries = HashMap::new();
        let mut prev: Option<usize> = None;
        let mut n = 0;
        for batch in turns {
            let before = request(tools, &sys, msgs.clone());
            for t in &batch {
                n += 1;
                append(&mut msgs, t, n);
            }
            let req = request(tools, &sys, msgs.clone());
            let (r, _, losses) = planned(Format::AnthropicMessages, &req, &CachePolicy::default());
            prop_assert!(losses.is_empty());
            let b = body(Format::AnthropicMessages, &r);
            let ps = sim_positions(&b);
            let grown = ps.len() - sim_positions(&body(Format::AnthropicMessages, &before)).len();
            let hit = sim_request(&mut entries, &b);
            if let Some(prev) = prev
                && grown < 40
            {
                prop_assert!(hit >= Some(prev), "grew {grown}: hit {hit:?}, previous tail {prev}");
            }
            prev = (0..ps.len()).rev().find(|&i| ps[i].1);
        }
    }

    /// render(H ++ X) starts with render(H) up to cache markers, in every format.
    #[test]
    fn rendering_is_prefix_stable(
        tools in 0..3usize,
        system in 0..3usize,
        head in proptest::collection::vec(turn(), 0..6),
        tail in proptest::collection::vec(turn(), 1..6),
    ) {
        let sys: Vec<String> = (0..system).map(|i| format!("system {i}")).collect();
        let sys: Vec<&str> = sys.iter().map(String::as_str).collect();
        let mut msgs = vec![user(vec![text("start")])];
        for (n, t) in head.iter().enumerate() {
            append(&mut msgs, t, n);
        }
        let h = request(tools, &sys, msgs.clone());
        for (n, t) in tail.iter().enumerate() {
            append(&mut msgs, t, 100 + n);
        }
        let hx = request(tools, &sys, msgs);
        for format in ALL {
            let print = |req: &Request| -> Fingerprint {
                let (r, _, _) = planned(format, req, &CachePolicy::default());
                let enc = encode_request(format, &r, &opts()).unwrap();
                fingerprint(format, "m", &enc.body).unwrap()
            };
            let mut tracker = CacheTracker::new();
            tracker.commit(print(&h), &Usage::default());
            let brk = tracker.check(&print(&hx));
            prop_assert!(brk.is_none(), "{format:?}: {brk:?}");
        }
    }
}

type Edit = fn(&mut Request);

#[test]
fn tracker_names_what_broke_the_cache() {
    let base = request(
        2,
        &["s"],
        vec![
            user(vec![text("q")]),
            assistant(vec![text("a")]),
            user(vec![text("q2")]),
        ],
    );
    let print = |format: Format, req: &Request| {
        let (r, _, _) = planned(format, req, &CachePolicy::default());
        fingerprint(
            format,
            &r.model,
            &encode_request(format, &r, &opts()).unwrap().body,
        )
        .unwrap()
    };
    for format in ALL {
        let mut t = CacheTracker::new();
        assert!(t.check(&print(format, &base)).is_none());
        t.commit(print(format, &base), &Usage::default());

        let mut longer = base.clone();
        longer.messages.push(assistant(vec![text("a2")]));
        longer.messages.push(user(vec![text("q3")]));
        assert_eq!(t.check(&print(format, &longer)), None, "{format:?}");

        let cases: Vec<(Edit, BreakReason)> = vec![
            (|r| r.tools.push(tool("new")), BreakReason::Tools),
            (|r| r.tools.swap(0, 1), BreakReason::Tools),
            (|r| r.system[0] = text("other"), BreakReason::System),
            (
                |r| r.messages[0].content[0] = text("edited"),
                BreakReason::History,
            ),
            (
                |r| {
                    r.messages.pop();
                },
                BreakReason::History,
            ),
            (
                |r| r.tool_choice = Some(ToolChoice::Required),
                BreakReason::Settings,
            ),
            (|r| r.model = "other".into(), BreakReason::Model),
        ];
        for (edit, reason) in cases {
            let mut r = base.clone();
            edit(&mut r);
            let got = t.check(&print(format, &r)).map(|b| b.reason);
            assert_eq!(got, Some(reason), "{format:?}");
        }
        // Sampling settings do not touch the cache.
        let mut r = base.clone();
        r.temperature = Some(0.3);
        r.max_tokens = Some(7);
        assert_eq!(t.check(&print(format, &r)), None, "{format:?}");
    }

    let mut t = CacheTracker::new();
    t.commit(print(Format::OpenAiChat, &base), &Usage::default());
    let brk = t.check(&print(Format::AnthropicMessages, &base)).unwrap();
    assert_eq!(brk.reason, BreakReason::Format);
}

#[test]
fn tracker_reports_unexpected_misses() {
    let fp = |n: usize| {
        let mut msgs = vec![user(vec![text("q")])];
        for i in 0..n {
            append(&mut msgs, &Turn::Chat, i);
        }
        let r = request(1, &["s"], msgs);
        fingerprint(
            Format::AnthropicMessages,
            "m",
            &encode_request(Format::AnthropicMessages, &r, &opts())
                .unwrap()
                .body,
        )
        .unwrap()
    };
    let usage = |read, write| Usage {
        cache_read: Some(read),
        cache_write: Some(write),
        ..Usage::default()
    };
    let mut t = CacheTracker::new();
    // A first request has nothing to read.
    assert_eq!(t.commit(fp(0), &usage(0, 2000)), None);
    assert_eq!(t.commit(fp(1), &usage(2000, 100)), None);
    let miss = t.commit(fp(2), &usage(0, 2200)).unwrap();
    assert_eq!(miss.shared, fp(1).segments.len());
    // After a break a cold read is expected.
    let mut other = fp(3);
    other.segments[0].hash ^= 1;
    assert_eq!(t.commit(other, &usage(0, 2400)), None);
    // Too short to cache at all: never warm, never a miss.
    let mut t = CacheTracker::new();
    t.commit(fp(0), &usage(0, 0));
    assert_eq!(t.commit(fp(1), &usage(0, 0)), None);

    // State survives a round trip through serde.
    let json = serde_json::to_string(&t).unwrap();
    let back: CacheTracker = serde_json::from_str(&json).unwrap();
    assert_eq!(back, t);
}
