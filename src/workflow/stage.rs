// Copyright 2026 The Sashiko Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Declarative stage definitions and executable stage implementations.

use anyhow::Result;
use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::ai::{
    AiMessage, AiProvider, AiResponse, AiResponseFormat, AiTool, ErrorAction, LlmSession,
    SessionRunner, ToolCall, ValidationError,
};
use crate::toolbox::ToolBox;

use super::events::WorkflowEvent;
use super::output::OutputFormat;
use super::policy::{RecitationPolicy, StagePolicy, ToolScope};
use super::prompt::PromptTemplate;

/// Outcome and metrics from executing a single stage.
#[derive(Debug, Clone, Default)]
pub struct StageOutcome {
    pub tokens_in: u32,
    pub tokens_out: u32,
    pub tokens_cached: u32,
    pub history: Vec<AiMessage>,
}

/// Execution environment provided to stages during workflow runs.
pub struct WorkflowEnv<'a> {
    pub provider: Arc<dyn AiProvider>,
    pub tools: Arc<ToolBox>,
    pub base_dir: &'a std::path::Path,
    pub context_tag: Option<String>,
}

/// A deferred state mutation function returned after isolated stage execution.
pub type StateMutation<S> = Box<dyn FnOnce(&mut S) + Send>;

/// A type-erased executable stage that operates on workflow state `S`.
#[async_trait]
pub trait ExecutableStage<S: Send + Sync>: Send + Sync {
    /// Returns the human-readable name of the stage.
    fn name(&self) -> &'static str;

    /// Executes the stage in isolation against immutable state `&S`, returning
    /// execution metrics and a deferred reducer mutation to apply to `&mut S`.
    async fn execute_isolated(
        &self,
        env: &WorkflowEnv<'_>,
        state: &S,
        event_cb: Option<&(dyn Fn(WorkflowEvent) + Send + Sync)>,
    ) -> Result<(StageOutcome, StateMutation<S>)>;

    /// Executes the stage and immediately applies its mutation to `&mut S`.
    async fn execute(
        &self,
        env: &WorkflowEnv<'_>,
        state: &mut S,
        event_cb: Option<&(dyn Fn(WorkflowEvent) + Send + Sync)>,
    ) -> Result<StageOutcome> {
        let (outcome, mutation) = self.execute_isolated(env, state, event_cb).await?;
        mutation(state);
        Ok(outcome)
    }
}

/// Reducer function applying stage output `T` to mutable workflow state `S`.
pub type StageReducer<S, T> = Arc<dyn Fn(&mut S, T) + Send + Sync>;

/// Conditional predicate determining whether a stage should be evaluated.
pub type StageCondition<S> = Arc<dyn Fn(&S) -> bool + Send + Sync>;

/// A declarative workflow stage with typed output `T` and state reduction.
pub struct Stage<S, T> {
    pub name: &'static str,
    pub system_prompt: Option<PromptTemplate<S>>,
    pub user_prompt: PromptTemplate<S>,
    pub output_format: OutputFormat<S, T>,
    pub policy: StagePolicy,
    pub reducer: StageReducer<S, T>,
    pub skip_if: Option<StageCondition<S>>,
}

impl<S: 'static, T: 'static> Stage<S, T> {
    /// Creates a builder for a stage with a given name.
    pub fn builder(name: &'static str) -> StageBuilder<S, T> {
        StageBuilder::new(name)
    }
}

/// Builder for constructing [`Stage`] instances.
pub struct StageBuilder<S, T> {
    name: &'static str,
    system_prompt: Option<PromptTemplate<S>>,
    user_prompt: Option<PromptTemplate<S>>,
    output_format: Option<OutputFormat<S, T>>,
    policy: StagePolicy,
    reducer: Option<StageReducer<S, T>>,
    skip_if: Option<StageCondition<S>>,
}

impl<S: 'static, T: 'static> StageBuilder<S, T> {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            system_prompt: None,
            user_prompt: None,
            output_format: None,
            policy: StagePolicy::default(),
            reducer: None,
            skip_if: None,
        }
    }

    /// Sets the system prompt template.
    pub fn system_prompt(mut self, template: PromptTemplate<S>) -> Self {
        self.system_prompt = Some(template);
        self
    }

    /// Sets the user prompt template.
    pub fn user_prompt(mut self, template: PromptTemplate<S>) -> Self {
        self.user_prompt = Some(template);
        self
    }

    /// Sets the expected output format.
    pub fn output_format(mut self, fmt: OutputFormat<S, T>) -> Self {
        self.output_format = Some(fmt);
        self
    }

    /// Sets the stage execution policy.
    pub fn policy(mut self, policy: StagePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Configures the tool availability scope for this stage.
    pub fn tools(mut self, scope: ToolScope) -> Self {
        self.policy.tools = scope;
        self
    }

    /// Sets sampling temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.policy.temperature = temp;
        self
    }

    /// Sets max conversational turns.
    pub fn max_turns(mut self, turns: usize) -> Self {
        self.policy.max_turns = turns;
        self
    }

    /// Configures the recitation error policy.
    pub fn on_recitation(mut self, policy: RecitationPolicy) -> Self {
        self.policy.recitation_policy = policy;
        self
    }

    /// Defines how the stage output `T` mutates the workflow state `&mut S`.
    pub fn reduce<F>(mut self, reducer: F) -> Self
    where
        F: Fn(&mut S, T) + Send + Sync + 'static,
    {
        self.reducer = Some(Arc::new(reducer));
        self
    }

    /// Skips execution of this stage if the predicate returns true.
    pub fn skip_if<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&S) -> bool + Send + Sync + 'static,
    {
        self.skip_if = Some(Arc::new(predicate));
        self
    }

    /// Builds the stage.
    pub fn build(self) -> Stage<S, T> {
        let user_prompt = self
            .user_prompt
            .expect("user_prompt must be configured for stage");
        let output_format = self
            .output_format
            .expect("output_format must be configured for stage");
        let reducer = self.reducer.unwrap_or_else(|| Arc::new(|_, _| {}));

        Stage {
            name: self.name,
            system_prompt: self.system_prompt,
            user_prompt,
            output_format,
            policy: self.policy,
            reducer,
            skip_if: self.skip_if,
        }
    }
}

/// Dynamic adapter bridging [`Stage`] to [`LlmSession`].
struct StageSession<'a, S, T> {
    stage: &'a Stage<S, T>,
    state: &'a S,
    tools: Arc<ToolBox>,
    system_prompt: String,
    user_prompt: String,
    log_user_prompt: String,
    context_tag: Option<String>,
    recitation_fallback_active: bool,
    /// Name and arguments of the last call run, for the duplicate guard below.
    last_tool_call: Option<(String, Value)>,
    /// Every read call this stage has run and what it returned, so a repeat
    /// can be answered from here. Ordered by call number; short enough that a
    /// scan costs nothing next to the calls themselves.
    seen_tool_calls: Vec<SeenToolCall>,
    /// How many calls this stage has run, for numbering in the replay note.
    tool_calls_run: usize,
}

/// The result of an earlier call, marked as the repeat of that call.
fn replayed(result: &Value, call_number: usize) -> Value {
    let mut replay = result.clone();
    // Objects carry the note inline. Anything else is handed back unchanged
    // rather than reshaped, since the shape is the tool's.
    if let Some(obj) = replay.as_object_mut() {
        obj.insert(
            "note".to_string(),
            json!(format!(
                "This is the unchanged result of call {call_number} in this stage, \
                 which is already above in the conversation. The tree has not \
                 changed since. Use it and move on."
            )),
        );
    }
    replay
}

/// What a repeat is told when there is no result to hand back.
fn no_result_to_replay() -> Value {
    json!({
        "error": "Duplicate tool call blocked. Please change parameters or use a different tool."
    })
}

/// One completed tool call and its result.
struct SeenToolCall {
    call_number: usize,
    name: String,
    args: Value,
    result: Value,
}

/// Upper bound on remembered results. Each is already held in the
/// conversation, so remembering it roughly doubles that much memory; past
/// this point new calls simply are not remembered and the consecutive guard
/// below still applies.
const MAX_REMEMBERED_TOOL_CALLS: usize = 256;

impl<'a, S: Send + Sync + 'static, T: DeserializeOwned + Send + 'static> StageSession<'a, S, T> {
    /// The result of an earlier call in this stage that this one repeats, or
    /// None when there is no such result to hand back.
    fn replay_remembered(&self, name: &str, args: &Value) -> Option<Value> {
        if let Some(seen) = self
            .seen_tool_calls
            .iter()
            .find(|seen| seen.name == name && seen.args == *args)
        {
            tracing::warn!(
                "Repeat of call {}: {} with args {:?}; replaying its result",
                seen.call_number,
                name,
                args
            );
            return Some(replayed(&seen.result, seen.call_number));
        }
        None
    }

    /// Whether this call repeats the one immediately before it, which is what
    /// the guard has always blocked.
    fn repeats_last_call(&self, name: &str, args: &Value) -> bool {
        self.last_tool_call
            .as_ref()
            .is_some_and(|last| last.0 == name && last.1 == *args)
    }

    /// Records a call that is about to run, for the guard above. Returns its
    /// number within the stage.
    fn record_tool_call(&mut self, name: &str, args: &Value) -> usize {
        self.last_tool_call = Some((name.to_string(), args.clone()));
        self.tool_calls_run += 1;
        self.tool_calls_run
    }

    /// Remembers a completed call so a later repeat of it can be answered
    /// from the result. Failures are not remembered: the error is not what
    /// the model asked for, and the call may well succeed on a retry.
    fn remember_tool_call(
        &mut self,
        call_number: usize,
        name: String,
        args: Value,
        result: &Value,
    ) {
        if self.seen_tool_calls.len() >= MAX_REMEMBERED_TOOL_CALLS {
            return;
        }
        if result.get("error").is_some() {
            return;
        }
        self.seen_tool_calls.push(SeenToolCall {
            call_number,
            name,
            args,
            result: result.clone(),
        });
    }
}

#[async_trait]
impl<'a, S: Send + Sync + 'static, T: DeserializeOwned + Send + 'static> LlmSession
    for StageSession<'a, S, T>
{
    type Output = T;

    fn system_prompt(&self) -> String {
        self.system_prompt.clone()
    }

    fn initial_user_prompt(&self) -> String {
        self.user_prompt.clone()
    }

    fn log_user_prompt(&self) -> String {
        self.log_user_prompt.clone()
    }

    fn format_validation_feedback(&self, violation: &str) -> String {
        self.stage.output_format.format_feedback(violation)
    }

    fn tools(&self) -> Option<Vec<AiTool>> {
        match &self.stage.policy.tools {
            ToolScope::None => None,
            ToolScope::All => Some(self.tools.get_declarations_generic()),
            ToolScope::Selected(names) => {
                let all = self.tools.get_declarations_generic();
                Some(
                    all.into_iter()
                        .filter(|t| names.contains(&t.name))
                        .collect(),
                )
            }
        }
    }

    fn temperature(&self) -> Option<f32> {
        Some(self.stage.policy.temperature)
    }

    fn context_tag(&self) -> Option<String> {
        self.context_tag.clone()
    }

    fn response_format(&self) -> Option<AiResponseFormat> {
        if self.recitation_fallback_active {
            return None;
        }
        self.stage
            .output_format
            .schema()
            .map(|s| AiResponseFormat::Json {
                schema: Some(s.clone()),
            })
    }

    async fn call_tool(&mut self, name: &str, args: Value) -> Result<Value> {
        if let Some(reply) = self.replay_remembered(name, &args) {
            return Ok(reply);
        }
        // A repeat whose result was not remembered, because it failed or
        // because the cap was reached. Nothing to hand back, so say so.
        if self.repeats_last_call(name, &args) {
            tracing::warn!("Blocked duplicate tool call: {} with args {:?}", name, args);
            return Ok(no_result_to_replay());
        }
        let call_number = self.record_tool_call(name, &args);
        let result = match self.tools.call(name, args.clone()).await {
            Ok(v) => v,
            Err(e) => json!({ "error": e.to_string() }),
        };
        self.remember_tool_call(call_number, name.to_string(), args, &result);
        Ok(result)
    }

    async fn call_tools(&mut self, calls: Vec<ToolCall>) -> Result<Vec<(String, Value)>> {
        let mut results: Vec<Option<(String, Value)>> = vec![None; calls.len()];
        let mut to_run: Vec<(usize, ToolCall, usize)> = Vec::new();
        // Repeats of a call queued earlier in this same batch, as (this call,
        // the one it repeats, its id, that call's number).
        let mut sharing: Vec<(usize, usize, String, usize)> = Vec::new();

        for (idx, call) in calls.into_iter().enumerate() {
            if let Some(reply) = self.replay_remembered(&call.function_name, &call.arguments) {
                results[idx] = Some((call.id, reply));
                continue;
            }
            if self.repeats_last_call(&call.function_name, &call.arguments) {
                tracing::warn!(
                    "Blocked duplicate tool call: {} with args {:?}",
                    call.function_name,
                    call.arguments
                );
                // The call it repeats is queued in this same batch and has not
                // run yet, so there is nothing remembered to replay. Wait for
                // it and share its result rather than refusing outright.
                match to_run.last() {
                    Some((first_idx, queued, call_number))
                        if queued.function_name == call.function_name
                            && queued.arguments == call.arguments =>
                    {
                        sharing.push((idx, *first_idx, call.id, *call_number));
                    }
                    _ => results[idx] = Some((call.id, no_result_to_replay())),
                }
                continue;
            }
            let call_number = self.record_tool_call(&call.function_name, &call.arguments);
            to_run.push((idx, call, call_number));
        }

        let futures = to_run.into_iter().map(|(idx, call, call_number)| {
            let tools = self.tools.clone();
            let name = call.function_name.clone();
            let args = call.arguments.clone();
            async move {
                // A rejected call is the model's to read and correct. The
                // trait default propagates the error instead, which ends the
                // stage and with it the review.
                let res = match tools.call(&call.function_name, call.arguments).await {
                    Ok(v) => v,
                    Err(e) => json!({ "error": e.to_string() }),
                };
                (idx, call_number, name, args, (call.id, res))
            }
        });
        for (idx, call_number, name, args, res) in futures::future::join_all(futures).await {
            self.remember_tool_call(call_number, name, args, &res.1);
            results[idx] = Some(res);
        }

        for (idx, first_idx, id, call_number) in sharing {
            let reply = match results[first_idx].as_ref() {
                Some((_, value)) if value.get("error").is_none() => replayed(value, call_number),
                // The call it repeats failed, so there is still nothing to
                // hand back.
                _ => no_result_to_replay(),
            };
            results[idx] = Some((id, reply));
        }

        Ok(results.into_iter().flatten().collect())
    }

    fn validate(&mut self, response: &AiResponse) -> Result<Self::Output, ValidationError> {
        let text = response.content.as_deref().unwrap_or("");
        match self.stage.output_format.validate(text, self.state) {
            Ok(parsed) => Ok(parsed),
            Err(violation) => Err(ValidationError::FormatViolation(violation)),
        }
    }

    fn handle_provider_error(&mut self, error: &anyhow::Error, _attempt: usize) -> ErrorAction {
        let err_str = error.to_string();
        if err_str.contains("RECITATION") || err_str.contains("blocked") {
            match &self.stage.policy.recitation_policy {
                RecitationPolicy::Fail => ErrorAction::Fail,
                RecitationPolicy::RetryWithReminder(reminder) => {
                    ErrorAction::RetryWithFeedback(reminder.clone())
                }
                RecitationPolicy::FallbackToFreeForm { reminder } => {
                    self.recitation_fallback_active = true;
                    ErrorAction::RetryWithFeedback(reminder.clone())
                }
            }
        } else {
            ErrorAction::Fail
        }
    }
}

#[async_trait]
impl<S: Send + Sync + 'static, T: DeserializeOwned + Send + 'static> ExecutableStage<S>
    for Stage<S, T>
{
    fn name(&self) -> &'static str {
        self.name
    }

    async fn execute_isolated(
        &self,
        env: &WorkflowEnv<'_>,
        state: &S,
        event_cb: Option<&(dyn Fn(WorkflowEvent) + Send + Sync)>,
    ) -> Result<(StageOutcome, StateMutation<S>)> {
        if let Some(ref predicate) = self.skip_if
            && predicate(state)
        {
            return Ok((
                StageOutcome {
                    history: Vec::new(),
                    tokens_in: 0,
                    tokens_out: 0,
                    tokens_cached: 0,
                },
                Box::new(|_| {}),
            ));
        }

        if let Some(cb) = event_cb {
            cb(WorkflowEvent::StageStarted {
                stage_name: self.name,
            });
        }

        let system_prompt = if let Some(sys) = &self.system_prompt {
            sys.render_for_model(state, env.base_dir).await?
        } else {
            String::new()
        };

        let user_prompt = self
            .user_prompt
            .render_for_model(state, env.base_dir)
            .await?;
        let log_user_prompt = self.user_prompt.render_for_log(state);

        let stage_name = self.name;
        let result = {
            let mut session = StageSession {
                stage: self,
                state,
                tools: env.tools.clone(),
                system_prompt,
                user_prompt,
                log_user_prompt,
                context_tag: env.context_tag.clone(),
                recitation_fallback_active: false,
                last_tool_call: None,
                seen_tool_calls: Vec::new(),
                tool_calls_run: 0,
            };

            let runner = SessionRunner::new(env.provider.as_ref())
                .with_max_turns(self.policy.max_turns)
                .with_max_validation_attempts(self.policy.max_validation_attempts)
                .with_turn_callback(move |turn, max_turns| {
                    if let Some(cb) = event_cb {
                        cb(WorkflowEvent::StageTurn {
                            stage_name,
                            turn,
                            max_turns,
                        });
                    }
                });

            runner.run(&mut session).await?
        };

        let tokens_in = result.usage.prompt_tokens as u32;
        let tokens_out = result.usage.completion_tokens as u32;
        let tokens_cached = result.usage.cached_tokens.unwrap_or(0) as u32;

        if let Some(cb) = event_cb {
            cb(WorkflowEvent::StageFinished {
                stage_name: self.name,
                tokens_in,
                tokens_out,
                tokens_cached,
            });
        }

        let reducer = self.reducer.clone();
        let mutation: StateMutation<S> = Box::new(move |s: &mut S| {
            reducer(s, result.output);
        });

        let outcome = StageOutcome {
            tokens_in,
            tokens_out,
            tokens_cached,
            history: result.history,
        };

        Ok((outcome, mutation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{AiRequest, AiRole, ProviderCapabilities};
    use crate::workflow::output::OutputFormat;
    use crate::workflow::prompt::PromptTemplate;
    use std::sync::Mutex;

    #[derive(Default, Clone)]
    struct EmptyState;

    /// Calls a tool on its first turn, then answers. Records every request so a
    /// test can check what the model was told about the tool call.
    struct ToolCallingProvider {
        turn: Mutex<usize>,
        seen: Mutex<Vec<AiRequest>>,
        calls: Vec<ToolCall>,
        calling_turns: usize,
    }

    impl ToolCallingProvider {
        /// One call to git_read_files per path, each missing the required
        /// revision argument, so every one of them is rejected.
        fn rejecting(paths: &[&str]) -> Self {
            Self {
                turn: Mutex::new(0),
                seen: Mutex::new(Vec::new()),
                calls: paths
                    .iter()
                    .enumerate()
                    .map(|(i, path)| ToolCall {
                        id: format!("call_{i}"),
                        function_name: "git_read_files".to_string(),
                        arguments: json!({ "files": [{ "path": path }] }),
                        thought_signature: None,
                    })
                    .collect(),
                calling_turns: 1,
            }
        }

        /// One call to the concurrency probe per index, so the batch carries
        /// distinct arguments and neither the cache nor the duplicate guard
        /// collapses it.
        fn probing(count: usize) -> Self {
            Self {
                turn: Mutex::new(0),
                seen: Mutex::new(Vec::new()),
                calls: (0..count)
                    .map(|i| ToolCall {
                        id: format!("call_{i}"),
                        function_name: "concurrency_probe".to_string(),
                        arguments: json!({ "n": i }),
                        thought_signature: None,
                    })
                    .collect(),
                calling_turns: 1,
            }
        }

        /// One call to the counting read tool per path. These succeed, so a
        /// test can tell a replayed result from a fresh run.
        fn reading(paths: &[&str]) -> Self {
            Self {
                turn: Mutex::new(0),
                seen: Mutex::new(Vec::new()),
                calls: paths
                    .iter()
                    .enumerate()
                    .map(|(i, path)| ToolCall {
                        id: format!("call_{i}"),
                        function_name: "counting_read".to_string(),
                        arguments: json!({ "path": path }),
                        thought_signature: None,
                    })
                    .collect(),
                calling_turns: 1,
            }
        }

        /// Emit the same batch again on the next turn, so a test can reach the
        /// duplicate guard across two call_tools invocations.
        fn repeated_next_turn(mut self) -> Self {
            self.calling_turns = 2;
            self
        }
    }

    #[async_trait]
    impl AiProvider for ToolCallingProvider {
        async fn generate_content(&self, request: AiRequest) -> Result<AiResponse> {
            self.seen.lock().unwrap().push(request);
            let mut turn = self.turn.lock().unwrap();
            *turn += 1;
            if *turn <= self.calling_turns {
                return Ok(AiResponse {
                    content: None,
                    thought: None,
                    thought_signature: None,
                    tool_calls: Some(self.calls.clone()),
                    usage: None,
                    truncated: false,
                });
            }
            Ok(AiResponse {
                content: Some("done".to_string()),
                thought: None,
                thought_signature: None,
                tool_calls: None,
                usage: None,
                truncated: false,
            })
        }

        fn estimate_tokens(&self, _request: &AiRequest) -> usize {
            0
        }

        fn get_capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                model_name: "mock".to_string(),
                context_window_size: 100_000,
            }
        }
    }

    /// Succeeds, and counts how many times it was actually run, so a test can
    /// tell a replayed result from a fresh one.
    struct CountingRead {
        runs: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl crate::toolbox::framework::LlmTool<crate::toolbox::SashikoToolContext> for CountingRead {
        fn name(&self) -> &'static str {
            "counting_read"
        }

        fn description(&self) -> &'static str {
            "Test tool that succeeds and counts its runs."
        }

        fn parameters(&self) -> Value {
            json!({ "type": "object", "properties": { "path": { "type": "string" } } })
        }

        async fn call(
            &self,
            args: Value,
            _context: &crate::toolbox::SashikoToolContext,
        ) -> Result<Value> {
            self.runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(json!({ "contents": format!("body of {}", args["path"]) }))
        }
    }

    /// Blocks until every call in the batch is in flight. Under a sequential
    /// implementation the first call never returns.
    struct ConcurrencyProbe {
        barrier: Arc<tokio::sync::Barrier>,
    }

    #[async_trait]
    impl crate::toolbox::framework::LlmTool<crate::toolbox::SashikoToolContext> for ConcurrencyProbe {
        fn name(&self) -> &'static str {
            "concurrency_probe"
        }

        fn description(&self) -> &'static str {
            "Test tool that waits for the rest of its batch."
        }

        fn parameters(&self) -> Value {
            json!({ "type": "object", "properties": { "n": { "type": "integer" } } })
        }

        async fn call(
            &self,
            _args: Value,
            _context: &crate::toolbox::SashikoToolContext,
        ) -> Result<Value> {
            self.barrier.wait().await;
            Ok(json!({ "ok": true }))
        }
    }

    #[tokio::test]
    async fn test_a_batch_of_tool_calls_runs_concurrently() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = Arc::new(ToolCallingProvider::probing(3));
        let mut toolbox = ToolBox::new(tmp.path().to_path_buf(), None);
        toolbox.register_tool(ConcurrencyProbe {
            barrier: Arc::new(tokio::sync::Barrier::new(3)),
        });
        let env = WorkflowEnv {
            provider: provider.clone(),
            tools: Arc::new(toolbox),
            base_dir: tmp.path(),
            context_tag: None,
        };

        let stage: Stage<EmptyState, String> = Stage::builder("tool_concurrent")
            .user_prompt(PromptTemplate::new("go"))
            .output_format(OutputFormat::text())
            .reduce(|_: &mut EmptyState, _: String| {})
            .build();

        let run = stage.execute_isolated(&env, &EmptyState, None);
        let (_outcome, _mutation) = tokio::time::timeout(std::time::Duration::from_secs(10), run)
            .await
            .expect("the batch must run concurrently; a sequential loop never clears the barrier")
            .expect("the stage must not end");
    }

    #[tokio::test]
    async fn test_consecutive_duplicate_tool_call_is_blocked() {
        // Loop prevention: the same call twice in a row is answered with a
        // synthetic error rather than run again.
        let tmp = tempfile::tempdir().unwrap();
        let provider = Arc::new(ToolCallingProvider::rejecting(&["a", "a"]));
        let env = WorkflowEnv {
            provider: provider.clone(),
            tools: Arc::new(ToolBox::new(tmp.path().to_path_buf(), None)),
            base_dir: tmp.path(),
            context_tag: None,
        };

        let stage: Stage<EmptyState, String> = Stage::builder("tool_dup")
            .user_prompt(PromptTemplate::new("go"))
            .output_format(OutputFormat::text())
            .reduce(|_: &mut EmptyState, _: String| {})
            .build();

        let (_outcome, _mutation) = stage
            .execute_isolated(&env, &EmptyState, None)
            .await
            .expect("a duplicate call must not end the stage");

        let seen = provider.seen.lock().unwrap();
        let replies: Vec<String> = seen[1]
            .messages
            .iter()
            .filter(|m| m.role == AiRole::Tool)
            .map(|m| m.content.clone().unwrap_or_default())
            .collect();
        assert_eq!(replies.len(), 2);
        assert!(
            !replies[0].contains("Duplicate tool call blocked"),
            "the first of the pair runs: {}",
            replies[0]
        );
        assert!(
            replies[1].contains("Duplicate tool call blocked"),
            "the repeat is blocked: {}",
            replies[1]
        );
    }

    #[tokio::test]
    async fn test_repeat_of_a_successful_call_replays_its_result() {
        // The model asked for a file, then asked for the same file again on
        // the next turn. What it wanted both times is the file.
        let tmp = tempfile::tempdir().unwrap();
        let provider = Arc::new(ToolCallingProvider::reading(&["a"]).repeated_next_turn());
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut toolbox = ToolBox::new(tmp.path().to_path_buf(), None);
        toolbox.register_tool(CountingRead { runs: runs.clone() });
        let env = WorkflowEnv {
            provider: provider.clone(),
            tools: Arc::new(toolbox),
            base_dir: tmp.path(),
            context_tag: None,
        };

        let stage: Stage<EmptyState, String> = Stage::builder("tool_replay_turns")
            .user_prompt(PromptTemplate::new("go"))
            .output_format(OutputFormat::text())
            .reduce(|_: &mut EmptyState, _: String| {})
            .build();

        let (_outcome, _mutation) = stage
            .execute_isolated(&env, &EmptyState, None)
            .await
            .expect("a repeat must not end the stage");

        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the repeat is answered from the first run, not by running again"
        );

        let seen = provider.seen.lock().unwrap();
        let replies: Vec<String> = seen[2]
            .messages
            .iter()
            .filter(|m| m.role == AiRole::Tool)
            .map(|m| m.content.clone().unwrap_or_default())
            .collect();
        assert_eq!(replies.len(), 2);
        assert!(
            replies[1].contains("body of"),
            "the repeat gets the file it asked for: {}",
            replies[1]
        );
        assert!(
            replies[1].contains("call 1"),
            "and is told which earlier call it came from: {}",
            replies[1]
        );
        assert!(
            !replies[1].contains("Duplicate tool call blocked"),
            "nothing is refused: {}",
            replies[1]
        );
    }

    #[tokio::test]
    async fn test_repeat_inside_one_batch_replays_the_result() {
        // Both halves of the guard have to agree, so the same repeat arriving
        // inside a single batch is answered the same way.
        let tmp = tempfile::tempdir().unwrap();
        let provider = Arc::new(ToolCallingProvider::reading(&["a", "a"]));
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut toolbox = ToolBox::new(tmp.path().to_path_buf(), None);
        toolbox.register_tool(CountingRead { runs: runs.clone() });
        let env = WorkflowEnv {
            provider: provider.clone(),
            tools: Arc::new(toolbox),
            base_dir: tmp.path(),
            context_tag: None,
        };

        let stage: Stage<EmptyState, String> = Stage::builder("tool_replay_batch")
            .user_prompt(PromptTemplate::new("go"))
            .output_format(OutputFormat::text())
            .reduce(|_: &mut EmptyState, _: String| {})
            .build();

        let (_outcome, _mutation) = stage
            .execute_isolated(&env, &EmptyState, None)
            .await
            .expect("a repeat must not end the stage");

        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the second call in the batch does not run"
        );

        let seen = provider.seen.lock().unwrap();
        let replies: Vec<String> = seen[1]
            .messages
            .iter()
            .filter(|m| m.role == AiRole::Tool)
            .map(|m| m.content.clone().unwrap_or_default())
            .collect();
        assert_eq!(replies.len(), 2);
        assert!(
            replies.iter().all(|r| r.contains("body of")),
            "both halves get the file: {:?}",
            replies
        );
    }

    #[tokio::test]
    async fn test_repeat_of_a_failed_call_is_still_refused() {
        // A failure is not the answer to the question, and the call may well
        // succeed on a retry, so nothing is remembered to replay. The repeat
        // falls through to the refusal, which is what the tests above and
        // below this one already pin.
        let tmp = tempfile::tempdir().unwrap();
        let provider = Arc::new(ToolCallingProvider::rejecting(&["a", "a"]));
        let env = WorkflowEnv {
            provider: provider.clone(),
            tools: Arc::new(ToolBox::new(tmp.path().to_path_buf(), None)),
            base_dir: tmp.path(),
            context_tag: None,
        };

        let stage: Stage<EmptyState, String> = Stage::builder("tool_replay_failed")
            .user_prompt(PromptTemplate::new("go"))
            .output_format(OutputFormat::text())
            .reduce(|_: &mut EmptyState, _: String| {})
            .build();

        let (_outcome, _mutation) = stage
            .execute_isolated(&env, &EmptyState, None)
            .await
            .expect("a repeat must not end the stage");

        let seen = provider.seen.lock().unwrap();
        let replies: Vec<String> = seen[1]
            .messages
            .iter()
            .filter(|m| m.role == AiRole::Tool)
            .map(|m| m.content.clone().unwrap_or_default())
            .collect();
        assert_eq!(replies.len(), 2);
        assert!(
            replies[1].contains("Duplicate tool call blocked"),
            "a failure is not replayed as if it were an answer: {}",
            replies[1]
        );
    }

    #[tokio::test]
    async fn test_duplicate_tool_call_is_blocked_across_turns() {
        // The guard is per session, not per batch: the repeat here arrives on
        // the turn after the call it repeats.
        let tmp = tempfile::tempdir().unwrap();
        let provider = Arc::new(ToolCallingProvider::rejecting(&["a"]).repeated_next_turn());
        let env = WorkflowEnv {
            provider: provider.clone(),
            tools: Arc::new(ToolBox::new(tmp.path().to_path_buf(), None)),
            base_dir: tmp.path(),
            context_tag: None,
        };

        let stage: Stage<EmptyState, String> = Stage::builder("tool_dup_turns")
            .user_prompt(PromptTemplate::new("go"))
            .output_format(OutputFormat::text())
            .reduce(|_: &mut EmptyState, _: String| {})
            .build();

        let (_outcome, _mutation) = stage
            .execute_isolated(&env, &EmptyState, None)
            .await
            .expect("a duplicate call must not end the stage");

        let seen = provider.seen.lock().unwrap();
        let replies: Vec<String> = seen[2]
            .messages
            .iter()
            .filter(|m| m.role == AiRole::Tool)
            .map(|m| m.content.clone().unwrap_or_default())
            .collect();
        assert_eq!(replies.len(), 2);
        assert!(
            !replies[0].contains("Duplicate tool call blocked"),
            "the first turn's call runs: {}",
            replies[0]
        );
        assert!(
            replies[1].contains("Duplicate tool call blocked"),
            "the next turn's repeat is blocked: {}",
            replies[1]
        );
    }

    #[tokio::test]
    async fn test_non_consecutive_duplicate_tool_call_runs() {
        // Only a consecutive repeat is blocked, so the second "a" runs because
        // "b" separates it from the first.
        let tmp = tempfile::tempdir().unwrap();
        let provider = Arc::new(ToolCallingProvider::rejecting(&["a", "b", "a"]));
        let env = WorkflowEnv {
            provider: provider.clone(),
            tools: Arc::new(ToolBox::new(tmp.path().to_path_buf(), None)),
            base_dir: tmp.path(),
            context_tag: None,
        };

        let stage: Stage<EmptyState, String> = Stage::builder("tool_dup_gap")
            .user_prompt(PromptTemplate::new("go"))
            .output_format(OutputFormat::text())
            .reduce(|_: &mut EmptyState, _: String| {})
            .build();

        let (_outcome, _mutation) = stage
            .execute_isolated(&env, &EmptyState, None)
            .await
            .expect("a non-consecutive repeat must not end the stage");

        let seen = provider.seen.lock().unwrap();
        let replies: Vec<String> = seen[1]
            .messages
            .iter()
            .filter(|m| m.role == AiRole::Tool)
            .map(|m| m.content.clone().unwrap_or_default())
            .collect();
        assert_eq!(replies.len(), 3);
        assert!(
            replies
                .iter()
                .all(|r| !r.contains("Duplicate tool call blocked")),
            "none of the three is blocked: {:?}",
            replies
        );
    }

    #[tokio::test]
    async fn test_batched_tool_results_keep_their_call_order() {
        // join_all preserves the input order, which is what keeps a tool
        // result next to its call on the Gemini path, where the result
        // carries only the function name and not the call id.
        let tmp = tempfile::tempdir().unwrap();
        let provider = Arc::new(ToolCallingProvider::rejecting(&["a", "b", "c"]));
        let env = WorkflowEnv {
            provider: provider.clone(),
            tools: Arc::new(ToolBox::new(tmp.path().to_path_buf(), None)),
            base_dir: tmp.path(),
            context_tag: None,
        };

        let stage: Stage<EmptyState, String> = Stage::builder("tool_batch")
            .user_prompt(PromptTemplate::new("go"))
            .output_format(OutputFormat::text())
            .reduce(|_: &mut EmptyState, _: String| {})
            .build();

        let (_outcome, _mutation) = stage
            .execute_isolated(&env, &EmptyState, None)
            .await
            .expect("a batch of rejected calls must not end the stage");

        let seen = provider.seen.lock().unwrap();
        let ids: Vec<_> = seen[1]
            .messages
            .iter()
            .filter(|m| m.role == AiRole::Tool)
            .map(|m| m.tool_call_id.clone().unwrap_or_default())
            .collect();
        assert_eq!(ids, vec!["call_0", "call_1", "call_2"]);
    }

    #[tokio::test]
    async fn test_rejected_tool_call_is_reported_to_the_model_not_fatal() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = Arc::new(ToolCallingProvider::rejecting(&["x"]));
        let env = WorkflowEnv {
            provider: provider.clone(),
            tools: Arc::new(ToolBox::new(tmp.path().to_path_buf(), None)),
            base_dir: tmp.path(),
            context_tag: None,
        };

        let stage: Stage<EmptyState, String> = Stage::builder("tool_error")
            .user_prompt(PromptTemplate::new("go"))
            .output_format(OutputFormat::text())
            .reduce(|_: &mut EmptyState, _: String| {})
            .build();

        let (_outcome, _mutation) = stage
            .execute_isolated(&env, &EmptyState, None)
            .await
            .expect("a rejected tool call must not end the stage");

        // The model must have been handed the error and given another turn.
        let seen = provider.seen.lock().unwrap();
        assert_eq!(seen.len(), 2, "the model should get a second turn");
        let tool_reply = seen[1]
            .messages
            .iter()
            .find(|m| m.role == AiRole::Tool)
            .expect("the second request should carry the tool result");
        assert!(
            tool_reply
                .content
                .as_deref()
                .unwrap_or("")
                .contains("error"),
            "the model should see the tool's error: {:?}",
            tool_reply.content
        );
    }
}
