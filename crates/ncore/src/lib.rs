pub mod agent;
pub mod cancel;
pub mod endpoint;
pub mod error;
pub mod generate;
pub mod hooks;
pub mod passthrough;
pub mod retry;
pub mod rt;
pub mod translate;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub use agent::{
    Agent, AgentHook, LoopEvent, Run, RunOptions, RunOutput, RunStop, ToolDecision, Tools,
};
pub use cancel::CancelToken;
pub use endpoint::{Auth, Endpoint};
pub use error::Error;
pub use generate::{GenerateOptions, Generation};
pub use hooks::{
    EventAction, Hook, Interests, Middleware, Next, Plugins, RequestCtx, ResponseHead,
};
pub use ncore_core as core;
pub use ncore_core::cache::{CacheBreak, CacheMiss, CachePolicy, CacheTracker};
pub use ncore_core::{Format, Observation, Observed, Usage};
pub use passthrough::{
    InboundRequest, PassthroughBody, PassthroughOptions, PassthroughResponse, PassthroughStream,
};
pub use retry::RetryPolicy;
pub use rt::{
    ChildProcess, Clock, HttpRequest, HttpResponse, ProcessControl, ProcessLauncher, ProcessSpec,
    ProcessStdin, Spawner, Transport, TransportError,
};
pub use translate::{
    LossPolicy, TranslateBody, TranslateOptions, TranslateResponse, TranslateStream,
};

/// Entry point holding the runtime pieces and plugins supplied by the host.
pub struct Core {
    transport: Arc<dyn Transport>,
    plugins: Plugins,
    retry: Option<(RetryPolicy, Arc<dyn Clock>)>,
    next_id: AtomicU64,
}

impl Core {
    pub fn builder(transport: Arc<dyn Transport>) -> CoreBuilder {
        CoreBuilder {
            core: Core {
                transport,
                plugins: Plugins::default(),
                retry: None,
                next_id: AtomicU64::new(1),
            },
        }
    }

    pub(crate) fn request_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }
}

pub struct CoreBuilder {
    core: Core,
}

impl CoreBuilder {
    /// Enables retries; the clock provides the backoff sleeps.
    pub fn retry(mut self, policy: RetryPolicy, clock: Arc<dyn Clock>) -> Self {
        self.core.retry = Some((policy, clock));
        self
    }

    /// Hook applied to every request; runs outside endpoint hooks.
    pub fn hook(mut self, hook: Arc<dyn Hook>) -> Self {
        self.core.plugins.hook(hook);
        self
    }

    /// Middleware applied to every request; wraps endpoint middleware.
    pub fn middleware(mut self, m: Arc<dyn Middleware>) -> Self {
        self.core.plugins.middleware(m);
        self
    }

    /// Registers a plugin that is both a hook and a middleware.
    pub fn plugin<P: Hook + Middleware + 'static>(self, p: Arc<P>) -> Self {
        self.hook(p.clone()).middleware(p)
    }

    pub fn build(self) -> Core {
        self.core
    }
}
