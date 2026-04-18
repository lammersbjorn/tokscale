//! Claude local-agent-mode audit parser
//!
//! Parses audit logs from:
//! `~/Library/Application Support/Claude/local-agent-mode-sessions/**/audit.jsonl`
//!
//! These audit files expose per-run `result.modelUsage`, which is the most
//! reliable token source because it already breaks a completed run down by model.

use super::utils::{file_modified_timestamp_ms, parse_timestamp_value};
use super::UnifiedMessage;
use crate::provider_identity::inferred_provider_from_model;
use crate::TokenBreakdown;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Deserialize)]
struct LocalAgentModeAuditRecord {
    #[serde(rename = "type")]
    entry_type: Option<String>,
    subtype: Option<String>,
    session_id: Option<String>,
    uuid: Option<String>,
    timestamp: Option<Value>,
    #[serde(rename = "_audit_timestamp")]
    audit_timestamp: Option<Value>,
    #[serde(rename = "modelUsage")]
    model_usage: Option<BTreeMap<String, LocalAgentModeModelUsage>>,
}

#[derive(Debug, Deserialize)]
struct LocalAgentModeModelUsage {
    #[serde(rename = "inputTokens")]
    input_tokens: Option<i64>,
    #[serde(rename = "outputTokens")]
    output_tokens: Option<i64>,
    #[serde(rename = "cacheReadInputTokens")]
    cache_read_input_tokens: Option<i64>,
    #[serde(rename = "cacheCreationInputTokens")]
    cache_creation_input_tokens: Option<i64>,
}

pub fn is_claude_local_agent_audit_path(path: &Path) -> bool {
    path.file_name().and_then(|name| name.to_str()) == Some("audit.jsonl")
        && path.components().any(|component| {
            component
                .as_os_str()
                .to_string_lossy()
                .eq_ignore_ascii_case("local-agent-mode-sessions")
        })
}

pub fn parse_claude_local_agent_audit(path: &Path) -> Vec<UnifiedMessage> {
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

        let record = match serde_json::from_str::<LocalAgentModeAuditRecord>(trimmed) {
            Ok(record) => record,
            Err(_) => continue,
        };

        if record.entry_type.as_deref() != Some("result") {
            continue;
        }
        if record
            .subtype
            .as_deref()
            .is_some_and(|subtype| subtype != "success")
        {
            continue;
        }

        let Some(model_usage) = record.model_usage else {
            continue;
        };
        if model_usage.is_empty() {
            continue;
        }

        let timestamp_ms = record
            .audit_timestamp
            .as_ref()
            .and_then(parse_timestamp_value)
            .or_else(|| record.timestamp.as_ref().and_then(parse_timestamp_value))
            .unwrap_or(fallback_timestamp);
        let session_id = record
            .session_id
            .clone()
            .unwrap_or_else(|| path.to_string_lossy().to_string());
        let dedup_base = record
            .uuid
            .unwrap_or_else(|| format!("{}:{}", session_id, path.to_string_lossy()));

        for (model_id, usage) in model_usage {
            let tokens = TokenBreakdown {
                input: usage.input_tokens.unwrap_or(0).max(0),
                output: usage.output_tokens.unwrap_or(0).max(0),
                cache_read: usage.cache_read_input_tokens.unwrap_or(0).max(0),
                cache_write: usage.cache_creation_input_tokens.unwrap_or(0).max(0),
                reasoning: 0,
            };

            if tokens.total() == 0 {
                continue;
            }

            let provider_id = inferred_provider_from_model(&model_id)
                .unwrap_or("anthropic")
                .to_string();
            let mut message = UnifiedMessage::new_with_dedup(
                "claude",
                model_id.clone(),
                provider_id,
                session_id.clone(),
                timestamp_ms,
                tokens,
                0.0,
                Some(format!("claude-local-agent:{dedup_base}:{model_id}")),
            );
            message.agent = Some("local-agent-mode".to_string());
            messages.push(message);
        }
    }

    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_is_claude_local_agent_audit_path() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(
            "Library/Application Support/Claude/local-agent-mode-sessions/session/local/audit.jsonl",
        );
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, "").unwrap();

        assert!(is_claude_local_agent_audit_path(&path));
        assert!(!is_claude_local_agent_audit_path(
            &dir.path().join("audit.jsonl")
        ));
    }

    #[test]
    fn test_parse_claude_local_agent_result_model_usage() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(
            "Library/Application Support/Claude/local-agent-mode-sessions/session/local/audit.jsonl",
        );
        fs::create_dir_all(path.parent().unwrap()).unwrap();

        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            "{{\"type\":\"result\",\"subtype\":\"success\",\"session_id\":\"session-1\",\"uuid\":\"result-1\",\"_audit_timestamp\":\"2026-04-06T23:41:54.681Z\",\"modelUsage\":{{\"claude-opus-4-6\":{{\"inputTokens\":9,\"outputTokens\":10240,\"cacheReadInputTokens\":693199,\"cacheCreationInputTokens\":123962}},\"claude-haiku-4-5-20251001\":{{\"inputTokens\":43590,\"outputTokens\":11639,\"cacheReadInputTokens\":645420,\"cacheCreationInputTokens\":264847}}}}}}"
        )
        .unwrap();

        let messages = parse_claude_local_agent_audit(&path);

        assert_eq!(messages.len(), 2);
        let by_model: BTreeMap<_, _> = messages
            .iter()
            .map(|message| (message.model_id.as_str(), message))
            .collect();

        let opus = by_model.get("claude-opus-4-6").unwrap();
        assert_eq!(opus.client, "claude");
        assert_eq!(opus.provider_id, "anthropic");
        assert_eq!(opus.agent.as_deref(), Some("local-agent-mode"));
        assert_eq!(opus.session_id, "session-1");
        assert_eq!(opus.timestamp, 1_775_518_914_681);
        assert_eq!(opus.tokens.input, 9);
        assert_eq!(opus.tokens.output, 10_240);
        assert_eq!(opus.tokens.cache_read, 693_199);
        assert_eq!(opus.tokens.cache_write, 123_962);
        assert_eq!(
            opus.dedup_key.as_deref(),
            Some("claude-local-agent:result-1:claude-opus-4-6")
        );

        let haiku = by_model.get("claude-haiku-4-5-20251001").unwrap();
        assert_eq!(haiku.tokens.input, 43_590);
        assert_eq!(haiku.tokens.output, 11_639);
        assert_eq!(haiku.tokens.cache_read, 645_420);
        assert_eq!(haiku.tokens.cache_write, 264_847);
    }

    #[test]
    fn test_parse_claude_local_agent_ignores_non_result_and_empty_usage() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(
            "Library/Application Support/Claude/local-agent-mode-sessions/session/local/audit.jsonl",
        );
        fs::create_dir_all(path.parent().unwrap()).unwrap();

        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            "{{\"type\":\"assistant\",\"message\":{{\"model\":\"claude-opus-4-6\"}}}}"
        )
        .unwrap();
        writeln!(
            file,
            "{{\"type\":\"result\",\"subtype\":\"success\",\"session_id\":\"session-1\",\"modelUsage\":{{\"claude-opus-4-6\":{{\"inputTokens\":0,\"outputTokens\":0,\"cacheReadInputTokens\":0,\"cacheCreationInputTokens\":0}}}}}}"
        )
        .unwrap();

        let messages = parse_claude_local_agent_audit(&path);
        assert!(messages.is_empty());
    }
}
