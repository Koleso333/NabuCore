use std::collections::{HashSet, VecDeque};
use std::pin::{Pin, pin};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use futures_core::Stream;
use futures_util::StreamExt;
use futures_util::future::{Either, join_all, select};
use ncore_core::Usage;
use ncore_core::cache::{CacheBreak, CacheMiss, CachePolicy, CacheTracker, fingerprint};
use ncore_core::ir::{
    Accumulator, Block, Loss, Message, Request, Response, Role, StopReason, StreamEvent, ToolCall,
    ToolDef, ToolResult,
};

use crate::Core;
use crate::cancel::{CancelToken, or_cancel};
use crate::endpoint::Endpoint;
use crate::error::Error;
use crate::generate::GenerateOptions;
use crate::rt::{BoxFuture, Clock, MaybeSend, MaybeSync, boxed};

/// Executes tool calls for the loop; `ncore-mcp`'s `Toolbox` implements it.
pub trait Tools: MaybeSend + MaybeSync {
    /// Definitions sent to the model, in a stable order; read before every step, so a step can add tools.
    fn definitions(&self) -> Vec<ToolDef>;

    /// Runs one call; failures should come back as error results the model can read.
    fn call<'a>(&'a self, call: &'a ToolCall, cancel: &'a CancelToken)
    -> BoxFuture<'a, ToolResult>;
}

/// Host verdict on a tool call before it runs.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolDecision {
    Allow,
    /// Skip the call and tell the model why.
    Deny(String),
    /// Skip the call and use this result instead.
    Replace(ToolResult),
}

/// Host control points inside the loop; hooks run in registration order.
pub trait AgentHook: MaybeSend + MaybeSync {
    /// Edits the request of a step before it is encoded.
    fn before_step<'a>(
        &'a self,
        _step: u32,
        _req: &'a mut Request,
    ) -> BoxFuture<'a, Result<(), String>> {
        boxed(async { Ok(()) })
    }

    /// Decides whether a call runs; may wait for user approval and is cut short by cancellation.
    fn on_tool_call<'a>(&'a self, _call: &'a ToolCall) -> BoxFuture<'a, ToolDecision> {
        boxed(async { ToolDecision::Allow })
    }

    /// Edits a result before it enters the history.
    fn on_tool_result<'a>(
        &'a self,
        _call: &'a ToolCall,
        _result: &'a mut ToolResult,
    ) -> BoxFuture<'a, ()> {
        boxed(async {})
    }
}

#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Model requests allowed in this run.
    pub max_steps: u32,
    /// When false, tool calls end the run and are returned to the host unexecuted.
    pub execute_tools: bool,
    /// Calls of one step running at once; `None` runs them all together.
    pub max_parallel_tools: Option<usize>,
    /// Limit for any single call, on top of the tool's own timeouts.
    pub tool_timeout: Option<Duration>,
    /// How long a cancelled call may take to wind down before it is abandoned.
    pub cancel_grace: Duration,
    /// Re-send a step whose stream broke, using the core retry policy.
    pub restart_streams: bool,
    /// Request options; by default cache breakpoints are planned for every step.
    pub generate: GenerateOptions,
    pub cancel: CancelToken,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            max_steps: 50,
            execute_tools: true,
            max_parallel_tools: None,
            tool_timeout: None,
            cancel_grace: Duration::from_secs(2),
            restart_streams: true,
            generate: GenerateOptions {
                cache: Some(CachePolicy::default()),
                ..GenerateOptions::default()
            },
            cancel: CancelToken::new(),
        }
    }
}

/// Why a run ended; `messages` in the output are a valid history continuation in every case.
#[derive(Debug, Clone)]
pub enum RunStop {
    /// The model ended its turn; the reason tells end turn from max tokens, refusal and the like.
    Finished(Option<StopReason>),
    /// Tool calls the host must answer before the next run (`execute_tools: false`).
    ToolCalls(Vec<ToolCall>),
    MaxSteps,
    Cancelled,
    Failed(Error),
}

#[derive(Debug, Clone)]
pub struct RunOutput {
    /// Messages to append to the history.
    pub messages: Vec<Message>,
    pub stop: RunStop,
    /// Sum over all steps.
    pub usage: Usage,
    /// Model requests made.
    pub steps: u32,
    /// Output of a model turn that never completed; not part of `messages`.
    pub partial: Option<Message>,
}

#[derive(Debug, Clone)]
pub enum LoopEvent {
    StepStart {
        step: u32,
    },
    /// Conversion losses of the step request.
    Losses {
        step: u32,
        losses: Vec<Loss>,
    },
    Stream {
        step: u32,
        event: StreamEvent,
    },
    /// The step's stream broke and is sent again; discard what it streamed so far.
    StreamRestart {
        step: u32,
        attempt: u32,
        error: Error,
    },
    ToolStart {
        step: u32,
        call: ToolCall,
    },
    ToolEnd {
        step: u32,
        result: ToolResult,
    },
    /// The step request does not extend the previous one, so part of the prompt cache is lost.
    CacheBreak {
        step: u32,
        info: CacheBreak,
    },
    /// The prefix was unchanged, yet nothing was read from cache (expired or another node).
    CacheMiss {
        step: u32,
        info: CacheMiss,
    },
    /// Messages this step added to the history.
    StepEnd {
        step: u32,
        messages: Vec<Message>,
        usage: Usage,
        stop: Option<StopReason>,
    },
    Done(RunOutput),
}

/// Agent loop over one endpoint; history stays with the host.
pub struct Agent<'a> {
    core: &'a Core,
    endpoint: &'a Endpoint,
    clock: Arc<dyn Clock>,
    tools: Option<&'a dyn Tools>,
    hooks: Vec<Arc<dyn AgentHook>>,
    tracker: Option<Arc<Mutex<CacheTracker>>>,
}

type Emit = Arc<Mutex<VecDeque<LoopEvent>>>;

impl<'a> Agent<'a> {
    /// The clock drives tool timeouts, cancellation grace and stream restarts.
    pub fn new(core: &'a Core, endpoint: &'a Endpoint, clock: Arc<dyn Clock>) -> Self {
        Self {
            core,
            endpoint,
            clock,
            tools: None,
            hooks: Vec::new(),
            tracker: None,
        }
    }

    pub fn tools(mut self, tools: &'a dyn Tools) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn hook(mut self, hook: Arc<dyn AgentHook>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Cache state of the conversation, kept across runs; without it each run starts fresh.
    pub fn tracker(mut self, tracker: Arc<Mutex<CacheTracker>>) -> Self {
        self.tracker = Some(tracker);
        self
    }

    /// Runs until the model stops calling tools, a limit is hit, or the run is cancelled.
    pub fn run(&self, request: Request, opts: RunOptions) -> Run<'_> {
        let emit: Emit = Arc::default();
        let fut = boxed(
            Driver {
                agent: self,
                opts,
                emit: emit.clone(),
                tracker: self.tracker.clone().unwrap_or_default(),
            }
            .drive(request),
        );
        Run {
            fut: Some(fut),
            emit,
            output: None,
        }
    }

    /// One model request plus its tool calls.
    pub fn step(&self, request: Request, mut opts: RunOptions) -> Run<'_> {
        opts.max_steps = 1;
        self.run(request, opts)
    }
}

/// Events of a running loop; the last one is `Done`.
pub struct Run<'a> {
    fut: Option<BoxFuture<'a, RunOutput>>,
    emit: Emit,
    output: Option<RunOutput>,
}

impl Run<'_> {
    /// Drives the run to the end, dropping intermediate events.
    pub async fn complete(mut self) -> RunOutput {
        while let Some(ev) = self.next().await {
            if let LoopEvent::Done(out) = ev {
                return out;
            }
        }
        unreachable!("a run always ends with Done")
    }
}

impl Stream for Run<'_> {
    type Item = LoopEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<LoopEvent>> {
        let this = &mut *self;
        loop {
            if let Some(ev) = this.emit.lock().unwrap().pop_front() {
                return Poll::Ready(Some(ev));
            }
            // Events come only from the driver future, so an empty queue means polling it.
            let Some(fut) = this.fut.as_mut() else {
                return Poll::Ready(this.output.take().map(LoopEvent::Done));
            };
            match fut.as_mut().poll(cx) {
                Poll::Ready(out) => {
                    this.fut = None;
                    this.output = Some(out);
                }
                Poll::Pending => {
                    if this.emit.lock().unwrap().is_empty() {
                        return Poll::Pending;
                    }
                }
            }
        }
    }
}

struct Driver<'r, 'a> {
    agent: &'r Agent<'a>,
    opts: RunOptions,
    emit: Emit,
    tracker: Arc<Mutex<CacheTracker>>,
}

// Model turn failure with whatever was streamed before it.
struct TurnError {
    error: Error,
    partial: Option<Message>,
}

impl Driver<'_, '_> {
    fn emit(&self, ev: LoopEvent) {
        self.emit.lock().unwrap().push_back(ev);
    }

    async fn drive(self, mut req: Request) -> RunOutput {
        let mut out = RunOutput {
            messages: Vec::new(),
            stop: RunStop::MaxSteps,
            usage: Usage::default(),
            steps: 0,
            partial: None,
        };

        // A history ending in unanswered calls would be rejected by every provider.
        let pending = unanswered(&req.messages);
        if !pending.is_empty() {
            if !self.can_execute() {
                out.stop = RunStop::ToolCalls(pending);
                return out;
            }
            let results = self.run_tools(0, &pending, false).await;
            let msg = Message::new(Role::User, results);
            self.emit(LoopEvent::StepEnd {
                step: 0,
                messages: vec![msg.clone()],
                usage: Usage::default(),
                stop: None,
            });
            req.messages.push(msg.clone());
            out.messages.push(msg);
        }

        out.stop = loop {
            if self.opts.cancel.is_cancelled() {
                break RunStop::Cancelled;
            }
            if out.steps >= self.opts.max_steps {
                break RunStop::MaxSteps;
            }
            out.steps += 1;
            let step = out.steps;
            self.emit(LoopEvent::StepStart { step });

            let mut step_req = req.clone();
            if let Some(tools) = self.agent.tools {
                step_req.tools.extend(tools.definitions());
            }
            if let Err(stop) = self.before_step(step, &mut step_req).await {
                break stop;
            }
            let resp = match self.model_turn(step, &step_req).await {
                Ok(r) => r,
                Err(e) => {
                    let TurnError { error, partial } = *e;
                    out.partial = partial;
                    break match error {
                        Error::Cancelled => RunStop::Cancelled,
                        e => RunStop::Failed(e),
                    };
                }
            };
            out.usage.add(&resp.usage);
            let usage = resp.usage.clone();
            let stop = resp.stop.clone();
            let assistant = assistant_message(resp, step, &req.messages);
            let calls: Vec<ToolCall> = assistant
                .content
                .iter()
                .filter_map(|b| match b {
                    Block::ToolCall(c) => Some(c.clone()),
                    _ => None,
                })
                .collect();

            // Most providers reject an empty assistant message in later requests.
            if assistant.content.is_empty() {
                self.step_end(step, Vec::new(), &usage, &stop);
                break RunStop::Finished(stop);
            }
            req.messages.push(assistant.clone());
            out.messages.push(assistant.clone());

            if calls.is_empty() {
                self.step_end(step, vec![assistant], &usage, &stop);
                if stop == Some(StopReason::Pause) {
                    continue;
                }
                break RunStop::Finished(stop);
            }
            if !self.can_execute() {
                self.step_end(step, vec![assistant], &usage, &stop);
                break RunStop::ToolCalls(calls);
            }
            let truncated = stop == Some(StopReason::MaxTokens);
            let results = self.run_tools(step, &calls, truncated).await;
            let msg = Message::new(Role::User, results);
            req.messages.push(msg.clone());
            out.messages.push(msg.clone());
            self.step_end(step, vec![assistant, msg], &usage, &stop);
        };
        out
    }

    fn can_execute(&self) -> bool {
        self.opts.execute_tools && self.agent.tools.is_some()
    }

    fn step_end(
        &self,
        step: u32,
        messages: Vec<Message>,
        usage: &Usage,
        stop: &Option<StopReason>,
    ) {
        self.emit(LoopEvent::StepEnd {
            step,
            messages,
            usage: usage.clone(),
            stop: stop.clone(),
        });
    }

    async fn before_step(&self, step: u32, req: &mut Request) -> Result<(), RunStop> {
        for h in &self.agent.hooks {
            match or_cancel(&self.opts.cancel, h.before_step(step, req)).await {
                None => return Err(RunStop::Cancelled),
                Some(Err(message)) => {
                    return Err(RunStop::Failed(Error::Plugin {
                        name: "agent hook".into(),
                        message,
                    }));
                }
                Some(Ok(())) => {}
            }
        }
        Ok(())
    }

    /// One model request, re-sent when its stream breaks and the retry policy allows.
    async fn model_turn(&self, step: u32, req: &Request) -> Result<Response, Box<TurnError>> {
        let core = self.agent.core;
        let cancel = &self.opts.cancel;
        let mut attempt = 0;
        loop {
            attempt += 1;
            let mut gen_ = core
                .generate(self.agent.endpoint, req, &self.opts.generate, cancel)
                .await
                .map_err(|error| {
                    Box::new(TurnError {
                        error,
                        partial: None,
                    })
                })?;
            if attempt == 1 && !gen_.losses.is_empty() {
                self.emit(LoopEvent::Losses {
                    step,
                    losses: gen_.losses.clone(),
                });
            }
            // A restarted step sends the same body, so it is checked once.
            let print = fingerprint(self.agent.endpoint.format, &req.model, &gen_.request).ok();
            if attempt == 1
                && let Some(info) = print
                    .as_ref()
                    .and_then(|p| self.tracker.lock().unwrap().check(p))
            {
                self.emit(LoopEvent::CacheBreak { step, info });
            }
            let mut acc = Accumulator::new();
            let failure = loop {
                match gen_.next().await {
                    Some(Ok(event)) => {
                        acc.push(&event);
                        self.emit(LoopEvent::Stream { step, event });
                    }
                    Some(Err(e)) => break Some(e),
                    None => break None,
                }
            };
            let Some(error) = failure else {
                let resp = gen_.take_response().unwrap_or_else(|| acc.finish());
                if let Some(p) = print
                    && let Some(info) = self.tracker.lock().unwrap().commit(p, &resp.usage)
                {
                    self.emit(LoopEvent::CacheMiss { step, info });
                }
                return Ok(resp);
            };
            let retry = core.retry.as_ref().filter(|(p, _)| {
                self.agent.endpoint.allow_replays
                    && self.opts.restart_streams
                    && attempt < p.max_attempts
                    && restartable(&error)
            });
            let Some((policy, clock)) = retry else {
                let partial = acc.finish();
                return Err(Box::new(TurnError {
                    error,
                    partial: (!partial.content.is_empty())
                        .then(|| Message::new(Role::Assistant, partial.content)),
                }));
            };
            self.emit(LoopEvent::StreamRestart {
                step,
                attempt,
                error,
            });
            if or_cancel(cancel, clock.sleep(policy.delay(attempt, None)))
                .await
                .is_none()
            {
                return Err(Box::new(TurnError {
                    error: Error::Cancelled,
                    partial: None,
                }));
            }
        }
    }

    /// Results for every call, in call order.
    async fn run_tools(&self, step: u32, calls: &[ToolCall], truncated: bool) -> Vec<Block> {
        // Built eagerly: a lazy `map` closure inside the future trips higher-ranked Send checks.
        let mut futs = Vec::with_capacity(calls.len());
        for c in calls {
            futs.push(self.run_tool(step, c, truncated));
        }
        let results = match self.opts.max_parallel_tools {
            Some(n) => {
                futures_util::stream::iter(futs)
                    .buffered(n.max(1))
                    .collect::<Vec<_>>()
                    .await
            }
            None => join_all(futs).await,
        };
        results.into_iter().map(Block::ToolResult).collect()
    }

    async fn run_tool(&self, step: u32, call: &ToolCall, truncated: bool) -> ToolResult {
        self.emit(LoopEvent::ToolStart {
            step,
            call: call.clone(),
        });
        let mut result = self.decide_and_call(call, truncated).await;
        result.call_id = call.id.clone();
        for h in &self.agent.hooks {
            h.on_tool_result(call, &mut result).await;
        }
        // Hooks must not detach a result from its call.
        result.call_id = call.id.clone();
        if result.name.is_none() {
            result.name = Some(call.name.clone());
        }
        self.emit(LoopEvent::ToolEnd {
            step,
            result: result.clone(),
        });
        result
    }

    async fn decide_and_call(&self, call: &ToolCall, truncated: bool) -> ToolResult {
        let cancel = &self.opts.cancel;
        if cancel.is_cancelled() {
            return error_result(call, CANCELLED);
        }
        if truncated && serde_json::from_str::<serde_json::Value>(&call.args).is_err() {
            return error_result(
                call,
                "the call was not run: its arguments were cut off by the output token limit",
            );
        }
        for h in &self.agent.hooks {
            match or_cancel(cancel, h.on_tool_call(call)).await {
                None => return error_result(call, CANCELLED),
                Some(ToolDecision::Allow) => {}
                Some(ToolDecision::Deny(reason)) => {
                    return error_result(call, &format!("the call was denied: {reason}"));
                }
                Some(ToolDecision::Replace(r)) => return r,
            }
        }
        let tools = self.agent.tools.expect("checked by can_execute");
        self.guarded(tools, call).await
    }

    /// Applies the run's cancellation and the per-call timeout to any `Tools` implementation.
    async fn guarded(&self, tools: &dyn Tools, call: &ToolCall) -> ToolResult {
        let clock = &self.agent.clock;
        let token = CancelToken::new();
        let mut fut = pin!(tools.call(call, &token));
        let stop = async {
            let timeout = async {
                match self.opts.tool_timeout {
                    Some(d) => clock.sleep(d).await,
                    None => std::future::pending().await,
                }
            };
            match select(pin!(timeout), self.opts.cancel.cancelled()).await {
                Either::Left(_) => format!(
                    "the call timed out after {}s",
                    self.opts.tool_timeout.unwrap_or_default().as_secs_f64()
                ),
                Either::Right(_) => CANCELLED.to_owned(),
            }
        };
        let reason = match select(fut.as_mut(), pin!(stop)).await {
            Either::Left((r, _)) => return r,
            Either::Right((reason, _)) => reason,
        };
        // Let the tool wind down (MCP sends notifications/cancelled), then give up on it.
        token.cancel();
        match select(fut, clock.sleep(self.opts.cancel_grace)).await {
            Either::Left((r, _)) if !r.is_error => r,
            _ => error_result(call, &reason),
        }
    }
}

const CANCELLED: &str = "the call was cancelled by the user";

fn error_result(call: &ToolCall, message: &str) -> ToolResult {
    ToolResult {
        call_id: call.id.clone(),
        name: Some(call.name.clone()),
        content: vec![Block::text(message)],
        is_error: true,
        cache: None,
    }
}

/// Breaks seen while reading a stream that a fresh request may not hit.
fn restartable(e: &Error) -> bool {
    match e {
        Error::Transport(_) => true,
        // Status errors were already retried before the stream began.
        Error::Provider(p) => p.status.is_none() && p.kind.is_transient(),
        _ => false,
    }
}

/// Calls of a trailing assistant message that have no results yet.
fn unanswered(messages: &[Message]) -> Vec<ToolCall> {
    let Some(last) = messages.last().filter(|m| m.role == Role::Assistant) else {
        return Vec::new();
    };
    last.content
        .iter()
        .filter_map(|b| match b {
            Block::ToolCall(c) => Some(c.clone()),
            _ => None,
        })
        .collect()
}

/// Turns a response into a history message with unique, non-empty call ids.
fn assistant_message(resp: Response, step: u32, history: &[Message]) -> Message {
    let mut seen: HashSet<String> = history
        .iter()
        .flat_map(|m| &m.content)
        .filter_map(|b| match b {
            Block::ToolCall(c) => Some(c.id.clone()),
            _ => None,
        })
        .collect();
    let mut content = Vec::with_capacity(resp.content.len());
    for (i, mut b) in resp.content.into_iter().enumerate() {
        match &mut b {
            // A model cannot produce tool results; drop anything that claims to be one.
            Block::ToolResult(_) => continue,
            Block::ToolCall(c) => {
                if c.id.trim().is_empty() {
                    c.id = format!("call_{step}_{i}");
                }
                if seen.contains(&c.id) {
                    let base = c.id.clone();
                    c.id = (2..)
                        .map(|n| format!("{base}_{n}"))
                        .find(|id| !seen.contains(id))
                        .expect("free id");
                }
                seen.insert(c.id.clone());
            }
            _ => {}
        }
        content.push(b);
    }
    Message::new(Role::Assistant, content)
}
