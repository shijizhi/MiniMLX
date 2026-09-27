use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum Request {
    Load {
        model_dir: String,
    },
    Generate {
        prompt: String,
        max_tokens: u32,
        #[serde(default)]
        context_tokens: Option<u32>,
        #[serde(default)]
        session_id: Option<String>,
        #[serde(default = "default_prefill_step_size")]
        prefill_step_size: u32,
    },
    ClearSession {
        session_id: String,
    },
    ProfilePrefill {
        prompt: String,
        context_tokens: u32,
        #[serde(default = "default_profile_warmup")]
        warmup: u32,
        #[serde(default = "default_profile_prefill_step_size")]
        prefill_step_size: u32,
    },
    ProfileProjectionFusion {
        prompt: String,
        context_tokens: u32,
        #[serde(default = "default_profile_warmup")]
        warmup: u32,
        #[serde(default = "default_profile_repeats")]
        repeats: u32,
    },
}

fn default_profile_warmup() -> u32 {
    1
}

fn default_profile_prefill_step_size() -> u32 {
    1024
}

fn default_prefill_step_size() -> u32 {
    2048
}

fn default_profile_repeats() -> u32 {
    5
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    pub ok: bool,
    #[serde(flatten)]
    pub payload: ResponsePayload,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsePayload {
    Generate {
        command: String,
        text: String,
        prompt_tokens: Vec<u32>,
        generated_tokens: Vec<u32>,
        prefill_ms: f64,
        decode_ms: f64,
        decode_tokens_per_second: f64,
    },
    ProfilePrefill {
        command: String,
        prompt_tokens: u32,
        total_ms: f64,
        linear_attention_ms: f64,
        full_attention_ms: f64,
        unattributed_ms: f64,
        embedding_ms: f64,
        backbone_ms: f64,
        lm_head_ms: f64,
        actual_prefill_ms: f64,
        actual_chunk_ms: f64,
        actual_final_token_ms: f64,
        actual_chunks: Vec<f64>,
        groups: Vec<ProfileGroup>,
        #[serde(default)]
        projection_stages: Vec<ProfileStage>,
    },
    ProfileProjectionFusion {
        command: String,
        supported: bool,
        reason: Option<String>,
        direct_ms: Option<f64>,
        fused_ms: Option<f64>,
        speedup: Option<f64>,
        max_abs_error: Option<f64>,
        input_shape: Option<Vec<u32>>,
        qkv_output: Option<u32>,
        z_output: Option<u32>,
    },
    Load {
        command: String,
    },
    Error {
        error: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProfileGroup {
    pub start_layer: u32,
    pub end_layer: u32,
    pub kind: String,
    pub milliseconds: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProfileStage {
    pub name: String,
    pub layer: u32,
    pub milliseconds: f64,
    pub path: String,
    pub threshold: u32,
    pub eligible: bool,
}

#[cfg(test)]
mod tests {
    use super::{Response, ResponsePayload};

    #[test]
    fn generate_response_is_not_parsed_as_load() {
        let response: Response = serde_json::from_str(
            r#"{
                "ok": true,
                "command": "generate",
                "text": "hello",
                "prompt_tokens": [1],
                "generated_tokens": [2],
                "prefill_ms": 1.0,
                "decode_ms": 2.0,
                "decode_tokens_per_second": 500.0
            }"#,
        )
        .expect("valid generate response");
        assert!(matches!(response.payload, ResponsePayload::Generate { .. }));
    }

    #[test]
    fn prefill_profile_response_is_not_parsed_as_load() {
        let response: Response = serde_json::from_str(
            r#"{
                "ok": true,
                "command": "profile_prefill",
                "prompt_tokens": 4096,
                "total_ms": 100.0,
                "linear_attention_ms": 70.0,
                "full_attention_ms": 20.0,
                "unattributed_ms": 10.0,
                "embedding_ms": 1.0,
                "backbone_ms": 90.0,
                "lm_head_ms": 9.0,
                "actual_prefill_ms": 91.0,
                "actual_chunk_ms": 80.0,
                "actual_final_token_ms": 11.0,
                "actual_chunks": [80.0],
                "groups": []
            }"#,
        )
        .expect("valid prefill profile response");
        assert!(matches!(
            response.payload,
            super::ResponsePayload::ProfilePrefill { .. }
        ));
    }

    #[test]
    fn generate_request_session_id_is_optional_and_round_trips() {
        let request = super::Request::Generate {
            prompt: "hello".to_string(),
            max_tokens: 4,
            context_tokens: Some(4096),
            session_id: Some("conversation-1".to_string()),
            prefill_step_size: 1024,
        };
        let encoded = serde_json::to_string(&request).expect("serialize request");
        let decoded: super::Request = serde_json::from_str(&encoded).expect("deserialize request");
        assert!(matches!(
            decoded,
            super::Request::Generate {
                session_id: Some(id),
                ..
            } if id == "conversation-1"
        ));
    }
}
