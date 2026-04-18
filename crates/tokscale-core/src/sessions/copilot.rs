//! GitHub Copilot CLI OTEL parser
//!
//! Parses file-exported OpenTelemetry JSONL emitted by Copilot CLI monitoring.
//! Phase 1 only turns `chat` spans into token usage rows; tool spans and metrics
//! are intentionally ignored.

use super::utils::file_modified_timestamp_ms;
use super::UnifiedMessage;
use crate::normalize_model_for_grouping;
use crate::provider_identity::inferred_provider_from_model;
use crate::TokenBreakdown;
use serde_json::{Map, Value};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub fn parse_copilot_file(path: &Path) -> Vec<UnifiedMessage> {
    if is_vscode_chat_session_path(path) {
        return parse_copilot_vscode_chat_session(path);
    }

    parse_copilot_otel_file(path)
}

fn parse_copilot_otel_file(path: &Path) -> Vec<UnifiedMessage> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return Vec::new(),
    };

    let fallback_timestamp = file_modified_timestamp_ms(path);
    let reader = BufReader::new(file);
    let mut messages = Vec::new();

    for line in reader.lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => continue,
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let span = match serde_json::from_str::<Value>(trimmed) {
            Ok(value) => value,
            Err(_) => continue,
        };

        if !is_chat_span(&span) {
            continue;
        }

        let attributes = match span.get("attributes").and_then(Value::as_object) {
            Some(attributes) => attributes,
            None => continue,
        };

        let input = attr_i64(attributes, "gen_ai.usage.input_tokens");
        let output = attr_i64(attributes, "gen_ai.usage.output_tokens");
        let cache_read = attr_i64(attributes, "gen_ai.usage.cache_read.input_tokens");
        let cache_write = attr_i64(attributes, "gen_ai.usage.cache_write.input_tokens");
        let reasoning = attr_i64(attributes, "gen_ai.usage.reasoning.output_tokens");

        let model = first_non_empty_attr(
            attributes,
            &["gen_ai.response.model", "gen_ai.request.model"],
        )
        .unwrap_or("unknown")
        .to_string();

        let provider_id = inferred_provider_from_model(&model)
            .unwrap_or("github-copilot")
            .to_string();

        let trace_id = span
            .get("traceId")
            .and_then(Value::as_str)
            .unwrap_or("unknown-trace");
        let span_id = span
            .get("spanId")
            .and_then(Value::as_str)
            .unwrap_or("unknown-span");
        let dedup_key = format!("{trace_id}:{span_id}");

        let session_id = first_non_empty_attr(
            attributes,
            &[
                "gen_ai.conversation.id",
                "github.copilot.interaction_id",
                "gen_ai.response.id",
            ],
        )
        .unwrap_or(trace_id)
        .to_string();

        let timestamp_ms = span
            .get("endTime")
            .and_then(timestamp_ms_from_value)
            .or_else(|| span.get("startTime").and_then(timestamp_ms_from_value))
            .unwrap_or(fallback_timestamp);

        let tokens = normalize_input_tokens(input, output, cache_read, cache_write, reasoning);
        if tokens.total() == 0 {
            continue;
        }

        messages.push(UnifiedMessage::new_with_dedup(
            "copilot",
            model,
            provider_id,
            session_id,
            timestamp_ms,
            tokens,
            0.0,
            Some(dedup_key),
        ));
    }

    messages
}

fn parse_copilot_vscode_chat_session(path: &Path) -> Vec<UnifiedMessage> {
    let fallback_timestamp = file_modified_timestamp_ms(path);
    let content = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(_) => return Vec::new(),
    };

    if let Ok(snapshot) = serde_json::from_str::<Value>(&content) {
        if let Some(requests) = snapshot.get("requests").and_then(Value::as_array) {
            return requests
                .iter()
                .filter_map(|request| {
                    parse_vscode_request_snapshot(request, path, fallback_timestamp)
                })
                .collect();
        }
    }

    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return Vec::new(),
    };
    let reader = BufReader::new(file);
    let mut requests: Vec<VsCodeRequestState> = Vec::new();
    let mut messages = Vec::new();

    for line in reader.lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => continue,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let entry = match serde_json::from_str::<Value>(trimmed) {
            Ok(entry) => entry,
            Err(_) => continue,
        };

        let keys = match entry.get("k").and_then(Value::as_array) {
            Some(keys) => keys,
            None => continue,
        };

        match entry.get("kind").and_then(Value::as_i64) {
            Some(2) if is_requests_root_patch(keys) => {
                if let Some(appended_requests) = entry.get("v").and_then(Value::as_array) {
                    requests.extend(
                        appended_requests
                            .iter()
                            .map(VsCodeRequestState::from_request),
                    );
                }
            }
            Some(1) if is_request_result_patch(keys) => {
                let Some(index) = keys
                    .get(1)
                    .and_then(Value::as_u64)
                    .and_then(|index| usize::try_from(index).ok())
                else {
                    continue;
                };
                let Some(request) = requests.get(index) else {
                    continue;
                };
                let Some(result) = entry.get("v") else {
                    continue;
                };
                if let Some(message) =
                    parse_vscode_result(result, request, path, fallback_timestamp)
                {
                    messages.push(message);
                }
            }
            _ => {}
        }
    }

    messages
}

fn is_chat_span(value: &Value) -> bool {
    if value.get("type").and_then(Value::as_str) != Some("span") {
        return false;
    }

    if value
        .get("attributes")
        .and_then(Value::as_object)
        .and_then(|attributes| attributes.get("gen_ai.operation.name"))
        .and_then(Value::as_str)
        == Some("chat")
    {
        return true;
    }

    value
        .get("name")
        .and_then(Value::as_str)
        .is_some_and(|name| name.starts_with("chat "))
}

fn is_vscode_chat_session_path(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str() == "chatSessions")
}

#[derive(Clone, Default)]
struct VsCodeRequestState {
    request_id: String,
    response_id: Option<String>,
    timestamp_ms: Option<i64>,
    model_id: String,
}

impl VsCodeRequestState {
    fn from_request(request: &Value) -> Self {
        Self {
            request_id: request
                .get("requestId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            response_id: request
                .get("responseId")
                .and_then(Value::as_str)
                .map(str::to_string),
            timestamp_ms: request.get("timestamp").and_then(value_as_i64),
            model_id: normalize_copilot_model_id(
                request
                    .get("modelId")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown"),
            ),
        }
    }
}

fn parse_vscode_request_snapshot(
    request: &Value,
    path: &Path,
    fallback_timestamp: i64,
) -> Option<UnifiedMessage> {
    let request_state = VsCodeRequestState::from_request(request);
    let result = request.get("result")?;
    parse_vscode_result(result, &request_state, path, fallback_timestamp)
}

fn parse_vscode_result(
    result: &Value,
    request: &VsCodeRequestState,
    path: &Path,
    fallback_timestamp: i64,
) -> Option<UnifiedMessage> {
    let (input, output) = exact_prompt_output_tokens(result)?;
    if input == 0 && output == 0 {
        return None;
    }

    let model_id = normalize_copilot_model_id(
        result
            .get("resolvedModel")
            .and_then(Value::as_str)
            .unwrap_or(&request.model_id),
    );
    let provider_id = inferred_provider_from_model(&model_id)
        .unwrap_or("github-copilot")
        .to_string();
    let session_id = result
        .get("sessionId")
        .and_then(Value::as_str)
        .filter(|session_id| !session_id.trim().is_empty())
        .map(str::to_string)
        .or_else(|| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "unknown-session".to_string());
    let dedup_source = result
        .get("responseId")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| request.response_id.clone())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| request.request_id.clone());

    Some(UnifiedMessage::new_with_dedup(
        "copilot",
        model_id,
        provider_id,
        session_id.clone(),
        request.timestamp_ms.unwrap_or(fallback_timestamp),
        TokenBreakdown {
            input,
            output,
            cache_read: 0,
            cache_write: 0,
            reasoning: 0,
        },
        0.0,
        Some(format!("copilot-vscode:{session_id}:{dedup_source}")),
    ))
}

fn exact_prompt_output_tokens(result: &Value) -> Option<(i64, i64)> {
    let metadata = result.get("metadata");
    let prompt = metadata
        .and_then(|metadata| metadata.get("promptTokens"))
        .and_then(value_as_i64)
        .or_else(|| result.get("promptTokens").and_then(value_as_i64))?;
    let output = metadata
        .and_then(|metadata| metadata.get("outputTokens"))
        .and_then(value_as_i64)
        .or_else(|| result.get("outputTokens").and_then(value_as_i64))?;
    Some((prompt.max(0), output.max(0)))
}

fn is_requests_root_patch(keys: &[Value]) -> bool {
    matches!(keys, [key] if key.as_str() == Some("requests"))
}

fn is_request_result_patch(keys: &[Value]) -> bool {
    matches!(
        keys,
        [root, index, tail]
            if root.as_str() == Some("requests")
                && index.as_u64().is_some()
                && tail.as_str() == Some("result")
    )
}

fn normalize_copilot_model_id(model: &str) -> String {
    let stripped = model
        .trim()
        .strip_prefix("copilot/")
        .unwrap_or(model.trim());
    if stripped.is_empty() {
        return "unknown".to_string();
    }
    normalize_model_for_grouping(stripped)
}

fn attr_i64(attributes: &Map<String, Value>, key: &str) -> i64 {
    attributes
        .get(key)
        .and_then(value_as_i64)
        .unwrap_or(0)
        .max(0)
}

fn normalize_input_tokens(
    input: i64,
    output: i64,
    cache_read: i64,
    cache_write: i64,
    reasoning: i64,
) -> TokenBreakdown {
    // OTEL reports input_tokens inclusive of cache reads. Normalize only the
    // cached-read portion out of input, but preserve the reported cache buckets
    // intact because pricing totals account for them separately.
    let cache_read_for_input = cache_read.max(0).min(input.max(0));

    TokenBreakdown {
        input: input.saturating_sub(cache_read_for_input).max(0),
        output: output.max(0),
        cache_read: cache_read.max(0),
        cache_write: cache_write.max(0),
        reasoning: reasoning.max(0),
    }
}

fn first_non_empty_attr<'a>(attributes: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .filter_map(|key| attributes.get(*key).and_then(Value::as_str))
        .find(|value| !value.trim().is_empty())
}

fn value_as_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
        .or_else(|| value.as_f64().map(|value| value as i64))
        .or_else(|| value.as_str().and_then(|value| value.parse::<i64>().ok()))
}

fn timestamp_ms_from_value(value: &Value) -> Option<i64> {
    let parts = value.as_array()?;
    let seconds = parts.first().and_then(value_as_i64)?;
    let nanos = parts.get(1).and_then(value_as_i64)?;
    Some(seconds.saturating_mul(1000) + nanos / 1_000_000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    fn create_test_file(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();
        file
    }

    fn create_vscode_chat_session_file(filename: &str, content: &str) -> std::path::PathBuf {
        let dir = TempDir::new().unwrap();
        let sessions_dir = dir.path().join("workspaceStorage/workspace-1/chatSessions");
        fs::create_dir_all(&sessions_dir).unwrap();
        let file_path = sessions_dir.join(filename);
        fs::write(&file_path, content).unwrap();
        // Leak the tempdir for the duration of the test process so the path stays valid.
        std::mem::forget(dir);
        file_path
    }

    #[test]
    fn test_parse_copilot_chat_span() {
        let content = r#"{"type":"metric","name":"gen_ai.client.token.usage"}
{"type":"span","traceId":"trace-1","spanId":"span-1","name":"chat claude-sonnet-4","startTime":[1775934260,133000000],"endTime":[1775934264,967317833],"attributes":{"gen_ai.operation.name":"chat","gen_ai.request.model":"claude-sonnet-4","gen_ai.response.model":"claude-sonnet-4","gen_ai.conversation.id":"conv-1","gen_ai.usage.input_tokens":19452,"gen_ai.usage.output_tokens":281,"gen_ai.usage.cache_read.input_tokens":123,"gen_ai.usage.reasoning.output_tokens":128,"github.copilot.interaction_id":"interaction-1"}}"#;
        let file = create_test_file(content);

        let messages = parse_copilot_file(file.path());

        assert_eq!(messages.len(), 1);
        let message = &messages[0];
        assert_eq!(message.client, "copilot");
        assert_eq!(message.model_id, "claude-sonnet-4");
        assert_eq!(message.provider_id, "anthropic");
        assert_eq!(message.session_id, "conv-1");
        assert_eq!(message.tokens.input, 19_329);
        assert_eq!(message.tokens.output, 281);
        assert_eq!(message.tokens.cache_read, 123);
        assert_eq!(message.tokens.reasoning, 128);
        assert_eq!(message.timestamp, 1_775_934_264_967);
        assert_eq!(message.dedup_key.as_deref(), Some("trace-1:span-1"));
    }

    #[test]
    fn test_parse_copilot_ignores_non_chat_spans() {
        let content = r#"{"type":"span","traceId":"trace-1","spanId":"tool-1","name":"execute_tool rg","attributes":{"gen_ai.operation.name":"execute_tool","gen_ai.tool.name":"rg"}}
{"type":"span","traceId":"trace-1","spanId":"invoke-1","name":"invoke_agent","attributes":{"gen_ai.operation.name":"invoke_agent","gen_ai.usage.input_tokens":999,"gen_ai.usage.output_tokens":111}}
{"type":"span","traceId":"trace-1","spanId":"chat-1","name":"chat gpt-5.4-mini","endTime":[1775934264,967317833],"attributes":{"gen_ai.operation.name":"chat","gen_ai.response.model":"gpt-5.4-mini","gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":5}}"#;
        let file = create_test_file(content);

        let messages = parse_copilot_file(file.path());

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].dedup_key.as_deref(), Some("trace-1:chat-1"));
        assert_eq!(messages[0].tokens.input, 10);
        assert_eq!(messages[0].tokens.output, 5);
    }

    #[test]
    fn test_parse_copilot_falls_back_to_trace_and_provider() {
        let content = r#"{"type":"span","traceId":"trace-fallback","spanId":"span-fallback","name":"chat custom-model","attributes":{"gen_ai.operation.name":"chat","gen_ai.request.model":"custom-model","gen_ai.usage.input_tokens":"7","gen_ai.usage.output_tokens":"9"}}"#;
        let file = create_test_file(content);

        let messages = parse_copilot_file(file.path());

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].provider_id, "github-copilot");
        assert_eq!(messages[0].session_id, "trace-fallback");
        assert_eq!(messages[0].tokens.input, 7);
        assert_eq!(messages[0].tokens.output, 9);
    }

    #[test]
    fn test_parse_copilot_normalizes_only_cache_read_from_input() {
        let content = r#"{"type":"span","traceId":"trace-cache","spanId":"span-cache","name":"chat gpt-5.4","endTime":[1775934264,967317833],"attributes":{"gen_ai.operation.name":"chat","gen_ai.response.model":"gpt-5.4","gen_ai.usage.input_tokens":1000,"gen_ai.usage.output_tokens":20,"gen_ai.usage.cache_read.input_tokens":200,"gen_ai.usage.cache_write.input_tokens":50}}"#;
        let file = create_test_file(content);

        let messages = parse_copilot_file(file.path());

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].tokens.input, 800);
        assert_eq!(messages[0].tokens.output, 20);
        assert_eq!(messages[0].tokens.cache_read, 200);
        assert_eq!(messages[0].tokens.cache_write, 50);
    }

    #[test]
    fn test_parse_copilot_clamps_only_cache_read_to_input() {
        let content = r#"{"type":"span","traceId":"trace-clamp","spanId":"span-clamp","name":"chat gpt-5.4-mini","endTime":[1775934264,967317833],"attributes":{"gen_ai.operation.name":"chat","gen_ai.response.model":"gpt-5.4-mini","gen_ai.usage.input_tokens":100,"gen_ai.usage.output_tokens":5,"gen_ai.usage.cache_read.input_tokens":90,"gen_ai.usage.cache_write.input_tokens":20}}"#;
        let file = create_test_file(content);

        let messages = parse_copilot_file(file.path());

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].tokens.input, 10);
        assert_eq!(messages[0].tokens.cache_read, 90);
        assert_eq!(messages[0].tokens.cache_write, 20);
    }

    #[test]
    fn test_parse_copilot_keeps_cache_only_message() {
        let content = r#"{"type":"span","traceId":"trace-zero","spanId":"span-zero","name":"chat gpt-5.4-mini","endTime":[1775934264,967317833],"attributes":{"gen_ai.operation.name":"chat","gen_ai.response.model":"gpt-5.4-mini","gen_ai.usage.input_tokens":0,"gen_ai.usage.cache_read.input_tokens":50,"gen_ai.usage.cache_write.input_tokens":20}}"#;
        let file = create_test_file(content);

        let messages = parse_copilot_file(file.path());

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].tokens.input, 0);
        assert_eq!(messages[0].tokens.cache_read, 50);
        assert_eq!(messages[0].tokens.cache_write, 20);
    }

    #[test]
    fn test_parse_copilot_keeps_cache_read_when_input_is_missing() {
        let content = r#"{"type":"span","traceId":"trace-cache-read","spanId":"span-cache-read","name":"chat gpt-5.4-mini","endTime":[1775934264,967317833],"attributes":{"gen_ai.operation.name":"chat","gen_ai.response.model":"gpt-5.4-mini","gen_ai.usage.cache_read.input_tokens":50}}"#;
        let file = create_test_file(content);

        let messages = parse_copilot_file(file.path());

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].tokens.input, 0);
        assert_eq!(messages[0].tokens.cache_read, 50);
        assert_eq!(messages[0].tokens.cache_write, 0);
    }

    #[test]
    fn test_parse_copilot_vscode_result_patch_with_exact_tokens() {
        let content = r#"{"kind":2,"k":["requests"],"v":[{"requestId":"request-1","timestamp":1775150932502,"modelId":"copilot/claude-opus-4.6","responseId":"response-1"}]}
{"kind":1,"k":["requests",0,"result"],"v":{"metadata":{"promptTokens":30059,"outputTokens":632},"resolvedModel":"claude-opus-4.6","responseId":"response-1","sessionId":"session-1"}}"#;
        let path = create_vscode_chat_session_file("session.jsonl", content);

        let messages = parse_copilot_file(&path);

        assert_eq!(messages.len(), 1);
        let message = &messages[0];
        assert_eq!(message.client, "copilot");
        assert_eq!(message.model_id, "claude-opus-4-6");
        assert_eq!(message.provider_id, "anthropic");
        assert_eq!(message.session_id, "session-1");
        assert_eq!(message.timestamp, 1_775_150_932_502);
        assert_eq!(message.tokens.input, 30_059);
        assert_eq!(message.tokens.output, 632);
        assert_eq!(
            message.dedup_key.as_deref(),
            Some("copilot-vscode:session-1:response-1")
        );
    }

    #[test]
    fn test_parse_copilot_vscode_snapshot_with_exact_tokens() {
        let content = r#"{
  "version": 3,
  "requests": [
    {
      "requestId": "request-2",
      "timestamp": 1775150932600,
      "modelId": "copilot/gpt-5.4",
      "result": {
        "metadata": {
          "promptTokens": 42,
          "outputTokens": 9
        }
      }
    }
  ]
}"#;
        let path = create_vscode_chat_session_file("session.json", content);

        let messages = parse_copilot_file(&path);

        assert_eq!(messages.len(), 1);
        let message = &messages[0];
        assert_eq!(message.model_id, "gpt-5.4");
        assert_eq!(message.provider_id, "openai");
        assert_eq!(message.tokens.input, 42);
        assert_eq!(message.tokens.output, 9);
        assert_eq!(
            message.dedup_key.as_deref(),
            Some("copilot-vscode:session:request-2")
        );
    }
}
