//! Kilo CLI session parser
//!
//! Parses messages from:
//! - JSON message files: ~/.local/share/kilo/storage/message/
//! - Legacy SQLite database: ~/.local/share/kilo/kilo.db
//!
//! Kilo CLI uses a SQLite database similar to OpenCode.

use super::utils::{file_modified_timestamp_ms, open_readonly_sqlite, read_file_or_none};
use super::UnifiedMessage;
use crate::{provider_identity, TokenBreakdown};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct KiloMessage {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(alias = "sessionID")]
    pub session_id: Option<String>,
    pub role: String,
    #[serde(rename = "modelID", default)]
    pub model_id: Option<String>,
    #[serde(rename = "providerID", default)]
    pub provider_id: Option<String>,
    pub cost: Option<f64>,
    pub tokens: Option<KiloTokens>,
    pub time: Option<KiloTime>,
    pub agent: Option<String>,
    pub mode: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct KiloTokens {
    pub input: i64,
    pub output: i64,
    #[serde(default)]
    pub reasoning: Option<i64>,
    pub cache: KiloCache,
}

#[derive(Debug, Deserialize)]
pub struct KiloCache {
    pub read: i64,
    pub write: i64,
}

#[derive(Debug, Deserialize)]
pub struct KiloTime {
    pub created: f64,
    pub completed: Option<f64>,
}

pub fn parse_kilo_file(path: &Path) -> Option<UnifiedMessage> {
    let data = read_file_or_none(path)?;
    let fallback_timestamp = file_modified_timestamp_ms(path);
    let mut bytes = data;
    let msg: KiloMessage = simd_json::from_slice(&mut bytes).ok()?;
    kilo_message_to_unified(
        msg,
        fallback_timestamp,
        path.file_stem().and_then(|s| s.to_str()),
    )
}

pub fn parse_kilo_sqlite(db_path: &Path) -> Vec<UnifiedMessage> {
    let fallback_timestamp = file_modified_timestamp_ms(db_path);
    parse_kilo_sqlite_with_fallback(db_path, fallback_timestamp)
}

pub fn parse_kilo_sqlite_with_fallback(
    db_path: &Path,
    fallback_timestamp: i64,
) -> Vec<UnifiedMessage> {
    let Some(conn) = open_readonly_sqlite(db_path) else {
        return Vec::new();
    };

    let query = r#"
        SELECT m.id, m.data
        FROM message m
        WHERE json_extract(m.data, '$.role') = 'assistant'
          AND json_extract(m.data, '$.tokens') IS NOT NULL
    "#;

    let mut stmt = match conn.prepare(query) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    let rows = match stmt.query_map([], |row| {
        let id: String = row.get(0)?;
        let data_json: String = row.get(1)?;
        Ok((id, data_json))
    }) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    let mut messages = Vec::new();

    for row_result in rows {
        let (_id, data_json) = match row_result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let mut bytes = data_json.into_bytes();
        let msg: KiloMessage = match simd_json::from_slice(&mut bytes) {
            Ok(m) => m,
            Err(_) => continue,
        };

        if let Some(unified) = kilo_message_to_unified(msg, fallback_timestamp, Some(&_id)) {
            messages.push(unified);
        }
    }

    messages
}

fn kilo_message_to_unified(
    msg: KiloMessage,
    fallback_timestamp: i64,
    fallback_dedup_key: Option<&str>,
) -> Option<UnifiedMessage> {
    if msg.role != "assistant" {
        return None;
    }

    let tokens = msg.tokens?;
    let model_id = msg.model_id?;

    let agent = msg.agent.or(msg.mode);
    let session_id = msg.session_id.unwrap_or_else(|| "unknown".to_string());
    let timestamp = msg
        .time
        .map(|t| t.created as i64)
        .unwrap_or(fallback_timestamp);

    let provider = msg
        .provider_id
        .as_deref()
        .or_else(|| provider_identity::inferred_provider_from_model(&model_id))
        .unwrap_or("kilo")
        .to_string();

    let mut unified = UnifiedMessage::new_with_agent(
        "kilo",
        model_id,
        provider,
        session_id,
        timestamp,
        TokenBreakdown {
            input: tokens.input.max(0),
            output: tokens.output.max(0),
            cache_read: tokens.cache.read.max(0),
            cache_write: tokens.cache.write.max(0),
            reasoning: tokens.reasoning.unwrap_or(0).max(0),
        },
        msg.cost.unwrap_or(0.0).max(0.0),
        agent,
    );
    unified.dedup_key = msg
        .id
        .or_else(|| fallback_dedup_key.map(|key| key.to_string()));
    Some(unified)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_parse_kilo_message_structure() {
        let json = r#"{
            "id": "msg-123",
            "sessionID": "sess-456",
            "role": "assistant",
            "modelID": "minimax/m2.5",
            "providerID": "kilo",
            "cost": 0.15,
            "tokens": {
                "input": 1000,
                "output": 200,
                "cache": {"read": 500, "write": 100}
            },
            "time": {"created": 1700000000000}
        }"#;

        let mut bytes = json.as_bytes().to_vec();
        let msg: KiloMessage = simd_json::from_slice(&mut bytes).unwrap();
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.cost, Some(0.15));
        assert_eq!(msg.model_id, Some("minimax/m2.5".to_string()));
        assert_eq!(msg.session_id, Some("sess-456".to_string()));
    }

    #[test]
    fn test_parse_kilo_file_current_message_format() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("msg-1.json");
        fs::write(
            &path,
            r#"{
                "id": "msg_c4ff0c1d8001ZppSF1FMxtR6o6",
                "sessionID": "ses_3b0275b1cffe04x8u3gF1Ma4jZ",
                "role": "assistant",
                "time": {
                  "created": 1770867704280,
                  "completed": 1770867734619
                },
                "modelID": "minimax/minimax-m2.1:free",
                "providerID": "kilo",
                "mode": "code",
                "agent": "code",
                "cost": 0.00467358,
                "tokens": {
                  "input": 3693,
                  "output": 1451,
                  "reasoning": 680,
                  "cache": {
                    "read": 60816,
                    "write": 0
                  }
                }
            }"#,
        )
        .unwrap();

        let message = parse_kilo_file(&path).unwrap();
        assert_eq!(message.client, "kilo");
        assert_eq!(message.session_id, "ses_3b0275b1cffe04x8u3gF1Ma4jZ");
        assert_eq!(message.model_id, "minimax/minimax-m2.1:free");
        assert_eq!(message.provider_id, "kilo");
        assert_eq!(message.tokens.input, 3693);
        assert_eq!(message.tokens.output, 1451);
        assert_eq!(message.tokens.cache_read, 60816);
        assert_eq!(message.tokens.reasoning, 680);
        assert_eq!(
            message.dedup_key,
            Some("msg_c4ff0c1d8001ZppSF1FMxtR6o6".to_string())
        );
    }
}
