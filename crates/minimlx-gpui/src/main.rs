//! MiniMLX desktop entry point.
//!
//! GPUI integration is intentionally kept at this boundary. This executable
//! is already a RustRover-runnable end-to-end smoke test for the Python MLX
//! worker; the window is the next layer on the same application boundary.

use std::path::PathBuf;

use minimlx_protocol::{ProfileGroup, Request, ResponsePayload};
use minimlx_supervisor::PythonWorker;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct AppConfig {
    model_dir: String,
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    prompt_file: Option<String>,
    context_tokens: Option<u32>,
    session_id: Option<String>,
    #[serde(default = "default_warmup")]
    warmup: u32,
    #[serde(default = "default_profile_warmup")]
    profile_warmup: u32,
    #[serde(default = "default_repeats")]
    repeats: u32,
    #[serde(default = "default_prefill_step_size")]
    prefill_step_size: u32,
    #[serde(default)]
    disable_q4_prefill_dense: bool,
    #[serde(default)]
    disable_q4_mlp_fusion: bool,
    #[serde(default)]
    enable_q4_native_prefill: bool,
    #[serde(default)]
    enable_native_fa256_prefill: bool,
    #[serde(default)]
    greedy_skip_logsumexp: bool,
    #[serde(default)]
    full_attention_decode_force_fused: bool,
    #[serde(default)]
    full_attention_prefill_force_fused: bool,
    #[serde(default)]
    full_attention_kv_cache_step: u32,
    #[serde(default)]
    prefill_keep_allocator_cache: bool,
    python: Option<String>,
}

fn default_mode() -> String {
    "generate".to_string()
}

fn default_prompt() -> String {
    "Hello".to_string()
}

fn default_max_tokens() -> u32 {
    16
}

fn default_warmup() -> u32 {
    3
}

fn default_profile_warmup() -> u32 {
    3
}

fn default_repeats() -> u32 {
    5
}

fn default_prefill_step_size() -> u32 {
    2048
}

fn usage() -> ! {
    eprintln!(
        "usage: minimlx-gpui [--model-dir PATH] [--prompt TEXT] [--max-tokens N] [--context-tokens N] [--session-id ID] [--prefill-step-size N] [--python PATH]\n       --prefill-step-size 0 enables length-based auto scheduling"
    );
    std::process::exit(2);
}

fn option(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn request_once(
    worker: &PythonWorker,
    prompt: &str,
    max_tokens: u32,
    context_tokens: Option<u32>,
    session_id: Option<&str>,
    prefill_step_size: u32,
) -> minimlx_protocol::ResponsePayload {
    worker
        .request(&Request::Generate {
            prompt: prompt.to_string(),
            max_tokens,
            context_tokens,
            session_id: session_id.map(str::to_owned),
            prefill_step_size,
        })
        .expect("Python worker request failed")
        .payload
}

fn load_config(root: &std::path::Path) -> AppConfig {
    let path = std::env::var_os("MINIMLX_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("config/minimlx.local.json"));
    let contents = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "cannot read {path:?}: {error}; create config/minimlx.local.json or set MINIMLX_CONFIG"
        )
    });
    serde_json::from_str(&contents)
        .unwrap_or_else(|error| panic!("invalid MiniMLX config {path:?}: {error}"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        usage();
    }
    let root = std::env::current_dir().expect("current directory");
    let config = load_config(&root);
    let model_dir = option(&args, "--model-dir").unwrap_or(config.model_dir);
    let mode = option(&args, "--mode").unwrap_or(config.mode);
    let prompt = option(&args, "--prompt")
        .or_else(|| {
            config.prompt_file.as_ref().map(|path| {
                std::fs::read_to_string(root.join(path)).expect("cannot read prompt_file")
            })
        })
        .unwrap_or(config.prompt);
    let context_tokens = std::env::var("MINIMLX_CONTEXT_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .or_else(|| option(&args, "--context-tokens").and_then(|value| value.parse().ok()))
        .or(config.context_tokens);
    let session_id = option(&args, "--session-id").or(config.session_id);
    let prefill_step_size = option(&args, "--prefill-step-size")
        .and_then(|value| value.parse().ok())
        .unwrap_or(config.prefill_step_size);
    let max_tokens = option(&args, "--max-tokens")
        .and_then(|value| value.parse().ok())
        .unwrap_or(config.max_tokens);
    let full_attention_kv_cache_step = if config.full_attention_kv_cache_step >= 256 {
        config.full_attention_kv_cache_step
    } else if context_tokens.unwrap_or_default() >= 32768 {
        1024
    } else {
        256
    };
    let full_attention_kv_cache_step_value = full_attention_kv_cache_step.to_string();
    let python = option(&args, "--python")
        .or(config.python)
        .unwrap_or_else(|| root.join(".venv/bin/python").display().to_string());
    let socket = PathBuf::from(format!("/tmp/minimlx-runtime-{}.sock", std::process::id()));
    let worker_environment = if config.disable_q4_prefill_dense {
        [
            ("MINIMLX_DISABLE_Q4_PREFILL_DENSE", "1"),
            (
                "MINIMLX_DISABLE_Q4_MLP_FUSION",
                if config.disable_q4_mlp_fusion { "1" } else { "0" },
            ),
            (
                "MINIMLX_ENABLE_Q4_NATIVE_PREFILL",
                if config.enable_q4_native_prefill { "1" } else { "0" },
            ),
            (
                "MINIMLX_ENABLE_NATIVE_FA256_PREFILL",
                if config.enable_native_fa256_prefill { "1" } else { "0" },
            ),
            (
                "MINIMLX_GREEDY_SKIP_LOGSUMEXP",
                if config.greedy_skip_logsumexp {
                    "1"
                } else {
                    "0"
                },
            ),
            (
                "MINIMLX_FULL_ATTENTION_DECODE_FORCE_FUSED",
                if config.full_attention_decode_force_fused {
                    "1"
                } else {
                    "0"
                },
            ),
            (
                "MINIMLX_FULL_ATTENTION_PREFILL_FORCE_FUSED",
                if config.full_attention_prefill_force_fused {
                    "1"
                } else {
                    "0"
                },
            ),
            (
                "MINIMLX_FULL_ATTENTION_KV_CACHE_STEP",
                full_attention_kv_cache_step_value.as_str(),
            ),
            (
                "MINIMLX_PREFILL_KEEP_ALLOCATOR_CACHE",
                if config.prefill_keep_allocator_cache {
                    "1"
                } else {
                    "0"
                },
            ),
        ]
    } else {
        [
            ("MINIMLX_DISABLE_Q4_PREFILL_DENSE", "0"),
            (
                "MINIMLX_DISABLE_Q4_MLP_FUSION",
                if config.disable_q4_mlp_fusion { "1" } else { "0" },
            ),
            (
                "MINIMLX_ENABLE_Q4_NATIVE_PREFILL",
                if config.enable_q4_native_prefill { "1" } else { "0" },
            ),
            (
                "MINIMLX_ENABLE_NATIVE_FA256_PREFILL",
                if config.enable_native_fa256_prefill { "1" } else { "0" },
            ),
            (
                "MINIMLX_GREEDY_SKIP_LOGSUMEXP",
                if config.greedy_skip_logsumexp {
                    "1"
                } else {
                    "0"
                },
            ),
            (
                "MINIMLX_FULL_ATTENTION_DECODE_FORCE_FUSED",
                if config.full_attention_decode_force_fused {
                    "1"
                } else {
                    "0"
                },
            ),
            (
                "MINIMLX_FULL_ATTENTION_PREFILL_FORCE_FUSED",
                if config.full_attention_prefill_force_fused {
                    "1"
                } else {
                    "0"
                },
            ),
            (
                "MINIMLX_FULL_ATTENTION_KV_CACHE_STEP",
                full_attention_kv_cache_step_value.as_str(),
            ),
            (
                "MINIMLX_PREFILL_KEEP_ALLOCATOR_CACHE",
                if config.prefill_keep_allocator_cache {
                    "1"
                } else {
                    "0"
                },
            ),
        ]
    };
    let worker =
        PythonWorker::start_with_env(&python, &root, model_dir, &socket, &worker_environment)
            .expect("failed to start MiniMLX Python worker");
    match mode.as_str() {
        "generate" => match request_once(
            &worker,
            &prompt,
            max_tokens,
            context_tokens,
            session_id.as_deref(),
            prefill_step_size,
        ) {
            ResponsePayload::Generate {
                text,
                decode_tokens_per_second,
                ..
            } => {
                println!("{text}");
                println!("decode={decode_tokens_per_second:.2} tok/s");
            }
            ResponsePayload::Error { error } => eprintln!("worker error: {error}"),
            other => println!("worker response: {other:?}"),
        },
        "benchmark" => {
            println!(
                "warm-up: context={} tokens, decode={} tokens",
                context_tokens.unwrap_or(0),
                config.warmup
            );
            // Quick/Full in the oMLX UI is a warm-up token budget, not a
            // repeat count for the complete benchmark request.
            let _ = request_once(
                &worker,
                &prompt,
                config.warmup,
                context_tokens,
                session_id.as_deref(),
                prefill_step_size,
            );
            println!(
                "measurement: context={} tokens, decode={} tokens",
                context_tokens.unwrap_or(0),
                max_tokens
            );
            let mut decode_ms = Vec::new();
            let mut prefill_ms = Vec::new();
            for _ in 0..config.repeats {
                match request_once(
                    &worker,
                    &prompt,
                    max_tokens,
                    context_tokens,
                    session_id.as_deref(),
                    prefill_step_size,
                ) {
                    ResponsePayload::Generate {
                        prefill_ms: prefill,
                        decode_ms: decode,
                        prompt_tokens,
                        generated_tokens,
                        ..
                    } => {
                        prefill_ms.push(prefill);
                        decode_ms.push(decode);
                        if generated_tokens.len() != max_tokens as usize {
                            panic!(
                                "generation stopped early: expected {}, got {}",
                                max_tokens,
                                generated_tokens.len()
                            );
                        }
                        if prompt_tokens.len()
                            != context_tokens.unwrap_or(prompt_tokens.len() as u32) as usize
                        {
                            panic!(
                                "prompt token mismatch: expected {}, got {}",
                                context_tokens.unwrap_or(prompt_tokens.len() as u32),
                                prompt_tokens.len()
                            );
                        }
                    }
                    ResponsePayload::Error { error } => panic!("worker error: {error}"),
                    other => panic!("unexpected worker response: {other:?}"),
                }
            }
            decode_ms.sort_by(f64::total_cmp);
            prefill_ms.sort_by(f64::total_cmp);
            let middle = config.repeats as usize / 2;
            let decode = decode_ms[middle];
            let prefill = prefill_ms[middle];
            println!(
                "MiniMLX benchmark (warmup={}, repeats={}, statistic=median)",
                config.warmup, config.repeats
            );
            println!(
                "prefill_step_size={}{}",
                prefill_step_size,
                if prefill_step_size == 0 {
                    " (auto)"
                } else {
                    ""
                }
            );
            println!(
                "prompt_tokens={}, generated_tokens={}, TTFT={prefill:.1} ms, TPOT={:.2} ms, TG TPS={:.2}, E2E={:.1} ms",
                context_tokens.unwrap_or(0),
                max_tokens,
                decode / max_tokens as f64,
                max_tokens as f64 * 1000.0 / decode,
                prefill + decode
            );
        }
        "prefill_profile" => {
            let response = worker
                .request(&Request::ProfilePrefill {
                    prompt: prompt.to_string(),
                    context_tokens: context_tokens
                        .expect("prefill_profile requires context_tokens"),
                    warmup: config.profile_warmup,
                    prefill_step_size,
                })
                .expect("Python worker request failed")
                .payload;
            match response {
                ResponsePayload::ProfilePrefill {
                    prompt_tokens,
                    total_ms,
                    linear_attention_ms,
                    full_attention_ms,
                    unattributed_ms,
                    embedding_ms,
                    backbone_ms,
                    lm_head_ms,
                    actual_prefill_ms,
                    actual_chunk_ms,
                    actual_final_token_ms,
                    actual_chunks,
                    groups,
                    projection_stages,
                    ..
                } => {
                    println!(
                        "MiniMLX prefill profile (context={} tokens, warmup={}, grouped boundaries)",
                        prompt_tokens, config.profile_warmup
                    );
                    println!("total={total_ms:.1} ms");
                    println!("linear_attention={linear_attention_ms:.1} ms");
                    println!("full_attention={full_attention_ms:.1} ms");
                    println!("unattributed={unattributed_ms:.1} ms");
                    println!("boundary_embedding={embedding_ms:.1} ms");
                    println!("boundary_backbone_plus_final_norm={backbone_ms:.1} ms");
                    println!("boundary_lm_head={lm_head_ms:.1} ms");
                    println!("actual_prefill={actual_prefill_ms:.1} ms");
                    println!("actual_prefill_chunks={actual_chunk_ms:.1} ms");
                    println!("actual_final_token={actual_final_token_ms:.1} ms");
                    for (index, milliseconds) in actual_chunks.iter().enumerate() {
                        println!("actual_chunk_{index}={milliseconds:.1} ms");
                    }
                    for ProfileGroup {
                        start_layer,
                        end_layer,
                        kind,
                        milliseconds,
                    } in groups
                    {
                        println!("L{start_layer}..{end_layer} {kind}={milliseconds:.1} ms");
                    }
                    for stage in projection_stages {
                        println!(
                            "projection L{} {}={:.3} ms path={} eligible={} threshold={}",
                            stage.layer,
                            stage.name,
                            stage.milliseconds,
                            stage.path,
                            stage.eligible,
                            stage.threshold
                        );
                    }
                }
                ResponsePayload::Error { error } => eprintln!("worker error: {error}"),
                other => println!("worker response: {other:?}"),
            }
        }
        "projection_profile" => {
            let response = worker
                .request(&Request::ProfileProjectionFusion {
                    prompt: prompt.to_string(),
                    context_tokens: context_tokens
                        .expect("projection_profile requires context_tokens"),
                    warmup: 1,
                    repeats: 5,
                })
                .expect("Python worker request failed")
                .payload;
            match response {
                ResponsePayload::ProfileProjectionFusion {
                    supported,
                    reason,
                    direct_ms,
                    fused_ms,
                    speedup,
                    max_abs_error,
                    input_shape,
                    qkv_output,
                    z_output,
                    ..
                } => {
                    println!(
                        "MiniMLX projection fusion profile (context={} tokens)",
                        context_tokens.unwrap()
                    );
                    println!("supported={supported}");
                    if let Some(reason) = reason {
                        println!("reason={reason}");
                    }
                    if let (Some(direct), Some(fused), Some(speedup), Some(error)) =
                        (direct_ms, fused_ms, speedup, max_abs_error)
                    {
                        println!("input_shape={input_shape:?}, qkv_output={qkv_output:?}, z_output={z_output:?}");
                        println!("direct={direct:.3} ms");
                        println!("fused={fused:.3} ms");
                        println!("speedup={speedup:.3}x");
                        println!("max_abs_error={error:.6e}");
                    }
                }
                ResponsePayload::Error { error } => eprintln!("worker error: {error}"),
                other => println!("worker response: {other:?}"),
            }
        }
        other => panic!("unsupported mode in config: {other}"),
    }
}
