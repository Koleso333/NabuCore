# NabuCore

An engine for working with LLM providers, in Rust.

## What it does

- **Four API formats.** Anthropic Messages, OpenAI Chat Completions, OpenAI Responses and Gemini. Requests and streams are translated between any of them: the client speaks one format, the provider answers in another.
- **Lossless proxying.** A request can go through as is, byte for byte, with point edits to the fields you need and token usage and errors read from the stream on the way.
- **Agent loop.** The model calls tools, the loop runs them and carries on, with retries, cancellation and hooks at every step. The history stays with the calling code.
- **MCP client.** Over stdio and Streamable HTTP, in both shapes of the protocol: with sessions and without.
- **Prompt cache.** Places Anthropic cache breakpoints by itself and shows where the cache broke.
- **Plugins.** Hooks on requests, responses and stream events, plus plugins at the HTTP level. Providers with their own sign-in plug in this way; the providers of [NabuCode](https://github.com/Koleso333/NabuCode) are built like that.
- **Core without a runtime.** The bottom layer is plain functions with no network and no async. The transport plugs in from outside; a ready one runs on tokio, hyper and rustls, with HTTP and SOCKS5 proxies.

## Crates

- `ncore-core` - formats, translation, SSE, cache; no network
- `ncore` - the engine: generation, agent loop, retries, hooks
- `ncore-tokio` - the tokio transport
- `ncore-mcp` - the MCP client

## Usage

The crates are not on crates.io; add them from git:

```toml
[dependencies]
ncore = { git = "https://github.com/Koleso333/NabuCore" }
ncore-tokio = { git = "https://github.com/Koleso333/NabuCore" }
tokio = { version = "1", features = ["full"] }
futures = "0.3"
```

Stream a reply from Claude:

```rust
use futures::StreamExt;
use ncore::core::ir::{Block, Delta, Message, Request, Role, StreamEvent};
use ncore::{Auth, CancelToken, Endpoint, Format, GenerateOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let core = ncore_tokio::core(None);
    let key = std::env::var("ANTHROPIC_API_KEY")?;
    let endpoint = Endpoint::new(Format::AnthropicMessages, "https://api.anthropic.com")
        .with_auth(Auth::Header { name: "x-api-key".into(), value: key });

    let request = Request {
        model: "claude-opus-5-5".into(),
        messages: vec![Message::new(Role::User, vec![Block::text("Hi!")])],
        max_tokens: Some(1024),
        stream: true,
        ..Default::default()
    };

    let mut reply = core.generate(&endpoint, &request, &GenerateOptions::default(), &CancelToken::new()).await?;
    while let Some(event) = reply.next().await {
        if let StreamEvent::Delta { delta: Delta::Text(text), .. } = event? {
            print!("{text}");
        }
    }
    Ok(())
}
```

The same `Request` goes to any other format: change the `Format`, URL and key of the endpoint.

## License

[MIT](LICENSE).
