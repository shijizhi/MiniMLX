use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum Request {
    Load {
        model_dir: String,
    },
    Unload,
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
    Infer {
        #[serde(default = "default_protocol_version")]
        schema_version: u16,
        task: ModelTask,
        messages: Vec<InferenceMessage>,
        #[serde(default)]
        accelerations: Vec<Acceleration>,
        #[serde(default = "empty_object")]
        options: Value,
        #[serde(default)]
        stream: bool,
        #[serde(default)]
        session_id: Option<String>,
    },
    InspectModel {
        #[serde(default = "default_protocol_version")]
        schema_version: u16,
        model_dir: String,
        task: ModelTask,
        messages: Vec<InferenceMessage>,
        #[serde(default)]
        accelerations: Vec<Acceleration>,
        #[serde(default = "empty_object")]
        options: Value,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTask {
    Generation,
    Embedding,
    Reranking,
    SpeechToText,
    TextToSpeech,
    SpeechToSpeech,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Acceleration {
    Mtp,
    Dflash,
    Speculative,
    QuantizedKv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeErrorCode {
    InvalidRequest,
    Unsupported,
    Ambiguous,
    MissingComponent,
    DependencyBlocked,
    HardwareBlocked,
    ModelLoadFailed,
    InferenceFailed,
    InternalError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMessage {
    pub role: String,
    pub content: Vec<ContentPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoFrame {
    pub timestamp_seconds: f64,
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default)]
    pub data_base64: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default = "empty_object")]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "modality", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
        #[serde(default = "empty_object")]
        metadata: Value,
    },
    Structured {
        data: serde_json::Map<String, Value>,
        #[serde(default = "empty_object")]
        metadata: Value,
    },
    Image {
        #[serde(default)]
        uri: Option<String>,
        #[serde(default)]
        data_base64: Option<String>,
        #[serde(default)]
        mime_type: Option<String>,
        #[serde(default = "empty_object")]
        metadata: Value,
    },
    Audio {
        #[serde(default)]
        uri: Option<String>,
        #[serde(default)]
        data_base64: Option<String>,
        #[serde(default)]
        mime_type: Option<String>,
        #[serde(default = "empty_object")]
        metadata: Value,
    },
    Video {
        #[serde(default)]
        uri: Option<String>,
        #[serde(default)]
        data_base64: Option<String>,
        #[serde(default)]
        mime_type: Option<String>,
        #[serde(default)]
        frames: Vec<VideoFrame>,
        #[serde(default = "empty_object")]
        metadata: Value,
    },
}

fn empty_object() -> Value {
    Value::Object(Default::default())
}

fn default_protocol_version() -> u16 {
    1
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
    InferenceStream {
        command: String,
        event: String,
        #[serde(default)]
        delta: Option<String>,
        #[serde(default)]
        text: Option<String>,
        #[serde(default)]
        token_ids: Option<Vec<u32>>,
        #[serde(default)]
        metrics: Option<Value>,
    },
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
    Inference {
        command: String,
        #[serde(default = "default_protocol_version")]
        schema_version: u16,
        task: ModelTask,
        payload_type: String,
        payload: Value,
        #[serde(default)]
        usage: Value,
        #[serde(default)]
        metrics: Value,
        #[serde(default)]
        metadata: Value,
    },
    InspectModel {
        command: String,
        #[serde(default = "default_protocol_version")]
        schema_version: u16,
        descriptor: Value,
        decision: Value,
        plan: Option<Value>,
    },
    Load {
        command: String,
        loaded: bool,
    },
    Unload {
        command: String,
        released: bool,
    },
    Error {
        error: String,
        #[serde(default)]
        command: Option<String>,
        #[serde(default)]
        error_code: Option<RuntimeErrorCode>,
        #[serde(default)]
        retryable: bool,
        #[serde(default = "empty_object")]
        details: Value,
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
    use super::{
        Acceleration, ContentPart, InferenceMessage, ModelTask, Request, Response, ResponsePayload,
        RuntimeErrorCode, VideoFrame,
    };

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

    #[test]
    fn multimodal_inference_request_round_trips_without_backend_types() {
        let request = Request::Infer {
            schema_version: 1,
            task: ModelTask::Generation,
            messages: vec![InferenceMessage {
                role: "user".to_string(),
                content: vec![
                    ContentPart::Text {
                        text: "describe this".to_string(),
                        metadata: serde_json::json!({"source": "user"}),
                    },
                    ContentPart::Image {
                        uri: Some("file:///tmp/image.png".to_string()),
                        data_base64: None,
                        mime_type: Some("image/png".to_string()),
                        metadata: serde_json::json!({}),
                    },
                    ContentPart::Structured {
                        data: serde_json::json!({
                            "query": "blue sky",
                            "documents": ["one", "two"]
                        })
                        .as_object()
                        .expect("structured payload object")
                        .clone(),
                        metadata: serde_json::json!({}),
                    },
                    ContentPart::Video {
                        uri: None,
                        data_base64: None,
                        mime_type: None,
                        frames: vec![VideoFrame {
                            timestamp_seconds: 1.25,
                            uri: Some("file:///tmp/frame.png".to_string()),
                            data_base64: None,
                            mime_type: Some("image/png".to_string()),
                            metadata: serde_json::json!({"source": "frame-list"}),
                        }],
                        metadata: serde_json::json!({}),
                    },
                ],
            }],
            accelerations: vec![Acceleration::Mtp],
            options: serde_json::json!({"max_tokens": 8}),
            stream: false,
            session_id: None,
        };
        let encoded = serde_json::to_string(&request).expect("serialize inference request");
        let decoded: Request = serde_json::from_str(&encoded).expect("decode inference request");
        assert!(matches!(
            decoded,
            Request::Infer { messages, accelerations, .. }
                if matches!(messages[0].content[1], ContentPart::Image { .. })
                    && accelerations == vec![Acceleration::Mtp]
                    && matches!(
                        &messages[0].content[0],
                        ContentPart::Text { metadata, .. } if metadata["source"] == "user"
                    )
                    && matches!(
                        &messages[0].content[2],
                        ContentPart::Structured { data, .. }
                            if data["documents"][0] == "one"
                    )
                    && matches!(
                        &messages[0].content[3],
                        ContentPart::Video { frames, .. }
                            if frames[0].timestamp_seconds == 1.25
                                && frames[0].metadata["source"] == "frame-list"
                    )
        ));
    }

    #[test]
    fn inference_response_is_tagged_and_round_trips() {
        let value = serde_json::json!({
            "ok": true,
            "command": "infer",
            "task": "embedding",
            "payload_type": "EmbeddingPayload",
            "payload": {"vectors": [[0.1, 0.2]]},
            "usage": {"prompt_tokens": 2},
            "metrics": {},
            "metadata": {}
        });
        let response: Response = serde_json::from_value(value.clone()).expect("decode inference");
        assert!(matches!(
            response.payload,
            ResponsePayload::Inference { .. }
        ));
        let encoded = serde_json::to_value(response).expect("encode inference response");
        assert_eq!(encoded["task"], "embedding");
    }

    #[test]
    fn inference_stream_event_round_trips() {
        let value = serde_json::json!({
            "ok": true,
            "command": "infer",
            "event": "delta",
            "delta": "Swift",
            "token_ids": [1, 2]
        });
        let response: Response = serde_json::from_value(value).expect("decode stream event");
        assert!(matches!(
            response.payload,
            ResponsePayload::InferenceStream { event, .. } if event == "delta"
        ));
    }

    #[test]
    fn inspect_model_request_round_trips_without_loading_weights() {
        let request = Request::InspectModel {
            schema_version: 1,
            model_dir: "/models/qwen".to_string(),
            task: ModelTask::Generation,
            messages: vec![InferenceMessage {
                role: "user".to_string(),
                content: vec![ContentPart::Text {
                    text: "hello".to_string(),
                    metadata: serde_json::json!({}),
                }],
            }],
            accelerations: vec![],
            options: serde_json::json!({}),
        };
        let encoded = serde_json::to_string(&request).expect("serialize inspect request");
        let decoded: Request = serde_json::from_str(&encoded).expect("decode inspect request");
        assert!(
            matches!(decoded, Request::InspectModel { task, .. } if task == ModelTask::Generation)
        );
    }

    #[test]
    fn infer_rejects_unknown_task_names() {
        let value = serde_json::json!({
            "command": "infer",
            "task": "image_generation",
            "messages": [],
            "options": {}
        });
        assert!(serde_json::from_value::<Request>(value).is_err());
    }

    #[test]
    fn infer_accepts_legacy_payload_without_acceleration_field() {
        let value = serde_json::json!({
            "command": "infer",
            "schema_version": 1,
            "task": "generation",
            "messages": [{
                "role": "user",
                "content": [{"modality": "text", "text": "hello"}]
            }],
            "options": {}
        });
        let request: Request = serde_json::from_value(value).expect("legacy infer request");
        assert!(matches!(
            request,
            Request::Infer { accelerations, .. } if accelerations.is_empty()
        ));
    }

    #[test]
    fn inspect_model_response_is_not_misparsed_as_load() {
        let response: Response = serde_json::from_str(
            r#"{
                "ok": true,
                "command": "inspect_model",
                "descriptor": {"model_type": "unknown"},
                "decision": {"status": "ambiguous"},
                "plan": null
            }"#,
        )
        .expect("valid inspection response");
        assert!(matches!(
            response.payload,
            ResponsePayload::InspectModel { .. }
        ));
    }

    #[test]
    fn unload_request_and_response_round_trip() {
        let load_response: Response =
            serde_json::from_str(r#"{"ok":true,"command":"load","loaded":true}"#)
                .expect("decode load response");
        assert!(matches!(
            load_response.payload,
            ResponsePayload::Load { .. }
        ));

        let request = Request::Unload;
        let encoded = serde_json::to_string(&request).expect("serialize unload request");
        assert_eq!(encoded, r#"{"command":"unload"}"#);
        let decoded: Request = serde_json::from_str(&encoded).expect("decode unload request");
        assert!(matches!(decoded, Request::Unload));

        let response: Response =
            serde_json::from_str(r#"{"ok":true,"command":"unload","released":true}"#)
                .expect("decode unload response");
        assert!(matches!(response.payload, ResponsePayload::Unload { .. }));
    }

    #[test]
    fn structured_error_response_and_legacy_error_both_decode() {
        let structured: Response = serde_json::from_str(
            r#"{
                "ok": false,
                "command": "infer",
                "error": "RuntimeError: dependency unavailable",
                "error_code": "dependency_blocked",
                "retryable": true,
                "details": {"missing_dependencies": ["mlx-vlm"]}
            }"#,
        )
        .expect("decode structured runtime error");
        match structured.payload {
            ResponsePayload::Error {
                error_code,
                retryable,
                details,
                ..
            } => {
                assert_eq!(error_code, Some(RuntimeErrorCode::DependencyBlocked));
                assert!(retryable);
                assert_eq!(details["missing_dependencies"][0], "mlx-vlm");
            }
            _ => panic!("expected error response"),
        }

        let legacy: Response =
            serde_json::from_str(r#"{"ok":false,"error":"ValueError: invalid request"}"#)
                .expect("decode legacy string-only error");
        assert!(matches!(legacy.payload, ResponsePayload::Error { .. }));
    }
}
