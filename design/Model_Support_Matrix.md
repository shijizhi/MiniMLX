# MiniMLX model support matrix

> This file preserves family-level validation records. The authoritative
> task/modality/provider architecture and support-state schema are in
> [`Universal_Model_Support_Architecture.md`](Universal_Model_Support_Architecture.md)
> and the machine-readable [`model_support_record.schema.json`](model_support_record.schema.json).
> Entries below are execution evidence, not a declaration that every task for
> a family is supported.

The implementation support-state key is `(task, input modalities, output
modalities, family, format, backend, hardware profile)`. Runtime-inspectable
states and provider routing are covered by the architecture tests; the historical
model-specific records below are not expanded into claims for untested tasks.

The broader support and performance process is defined in
[`Model_Support_and_Performance_Plan.md`](Model_Support_and_Performance_Plan.md).
Historical and future MiniMLX/oMLX numbers are maintained in
[`Model_Benchmark_Comparison.md`](Model_Benchmark_Comparison.md).
Older records include results from a MacBook Air M4 with 24 GB unified memory.
Each new model record identifies the actual device used; 32K results remain
conditional on that device's available memory.

This matrix tracks model-family compatibility separately from performance
optimization. Adapter/provider names in older records are the route used when
the measurement was collected; they do not define an optimization layer or
override the baseline-plus-overlay architecture. Qwen3.8's shared upstream
`qwen3_5` config identity establishes loader compatibility only; it does not
place Qwen3.8 in the Qwen3.5 optimization scope. A model is not considered
supported from `config.json` alone:
it must load through its adapter, execute a short prefill and decode, and pass
the family-specific shape checks.

## Initial Qwen3.5 coverage

This is the tested subset, not the family completion claim. The canonical
family configurations and quantization coverage are tracked in
[`Qwen35_Family_Coverage.md`](Qwen35_Family_Coverage.md).

| Model | Format | Architecture | Adapter | Short generation | Status |
|---|---|---|---|---|---|
| Qwen3.5-0.8B-MLX-4bit | affine Q4, group 64 | `qwen3_5` | Qwen35Adapter | previously validated | Supported |
| Qwen3.5-0.8B-MLX-bf16 | BF16 | `qwen3_5` | Qwen35Adapter | PASS: 4 tokens generated | Supported |
| Qwen3.5-4B-MLX-4bit | affine Q4, group 64 | `qwen3_5` | Qwen35Adapter | PASS: 4 tokens generated | Supported |
| Qwen3.5-4B-MLX-bf16 | BF16 | `qwen3_5` | Qwen35Adapter | PASS: 4 tokens generated | Supported (1K/4K/16K; 7.899 GiB MLX peak) |
| Qwen3.5-9B-4bit | affine Q4, group 64 | `qwen3_5` | Qwen35Adapter | PASS: `Hello! How can` | Supported for text generation; 1K/4K/8K/16K matrix recorded separately |
| Qwen3.5-27B-4bit | affine Q4, group 64 | `qwen3_5` | Qwen35Adapter | PASS: 4-token text smoke; image OCR + vision-cache reuse | Context-pass through 16K; 18.465 GiB suite peak; 1K EOS at 111/128 |
| Qwen3.5-35B-A3B-OptiQ-4bit-REAP-19B | mixed Q4/Q8, group 64 | `qwen3_5_moe` | Qwen35Adapter | PASS: 4 tokens at 1K/4K/16K; 11.586 GiB smoke peak | Context-pass through 16K; per-context peak memory unreported |

## Priority deep validation: Qwen3.8-27B

Checkpoint: `mlx-community/Qwen3.8-27B-4bit`, local path
`/Users/jizhishi/.minimlx/models/mlx-community--Qwen3.8-27B-4bit`. It is a
complete three-shard affine Q4/group-64 checkpoint. Its config declares
`qwen3_5` / `Qwen3_5ForConditionalGeneration`, a 64-layer hybrid text stack, a
vision tower, 262,144 max positions, and a video preprocessor. The installed
environment is `mlx-vlm 0.7.2`, MLX 0.32.2, and mlx-lm 0.31.3. Historical
functional checks loaded it through the shared upstream `mlx-vlm` implementation
using the checkpoint's `qwen3_5` architecture declaration. This is loader
compatibility evidence, not evidence that Qwen3.8 uses or inherits a Qwen3.5
MiniMLX optimization. The current resolver excludes Qwen3.8 from Qwen3.5
providers and routes its text and media requests through the shared upstream
MLX-LM and MLX-VLM implementations respectively; device-side regression of this
new route remains to be run.

Functional checks on the actual checkpoint:

- Eight known-answer text/image cases, repeated twice: 16/16 passed; all
  repeated outputs matched. Includes two-line OCR through base64 and data URI,
  object counting/presence, spatial color/shape, ordered two-image comparison,
  multi-turn image follow-up, and a text-only control.
- Image OCR streaming passed. Existing local-file image and ordered multi-image
  regressions each passed twice.
- Text reasoning smoke (`1 + 1`), exact arithmetic (`17 * 19`), strict JSON
  serialization, and generated Python code executed against a known input all
  passed.
- Text prefill/decode completed at 1K, 4K, 8K, 16K, and 32K requested context
  lengths. The 8K and 16K runs on the truncated code-corpus prompt ended with
  EOS and decoded to empty text; these are execution-boundary checks, not
  long-context answer-quality passes.

The shared VLM path now routes video for checkpoints that declare a video
processor and carries `chat_template_kwargs` into the native processor template.
On Qwen3.8, a four-frame timestamped synthetic clip completed actual Metal
inference and correctly described the red-to-blue transition. Template rendering
with `enable_thinking=false` and `true` now produces different prompt token IDs;
the false form suppresses the thinking block, while the true form includes the
template's default xhigh reasoning instruction. This verifies option plumbing,
not reasoning-quality parity or every template option.

- Video frame-list inference: PASS on the actual checkpoint. Direct video-file
  decoding is supported by the shared `mlx-vlm` input path but remains a
  separate end-to-end case to validate.
- A single request mixing still images and video frames is not supported by
  `mlx-vlm 0.7.2`'s shared `qwen3_5` model implementation: its image-feature
  merge counts both token families but consumes only one feature tensor. MiniMLX
  now rejects this combination before inference with an explicit explanation;
  separate image and video requests remain supported.
- `chat_template_kwargs`: template-level effect verified on the actual checkpoint;
  additional options and API-client integration still need broader tests.
- No separate Qwen3.8 MTP draft checkpoint was installed and the selected plan
  enabled no acceleration; MTP acceleration is untested.

Conclusion: text, image and timestamped-frame video generation are now
functionally compatible on this checkpoint through the shared Qwen VLM path.
Broader video file/API coverage, option-by-option template behavior and MTP
remain follow-up validation rather than blockers for the implemented path.

## Horizontal expansion: Qwen2.5

The first cross-family checkpoint is the locally available
`mlx-community/Qwen2.5-0.5B-Instruct-4bit`. Its descriptor resolves to the
shared standard MLX-LM baseline; Qwen2.5 is the tested family identity, not a
Qwen2-only performance adapter. The fixed short prompt
passed twice with identical tokens; the 1K/4K/16K context matrix passed with
five measured runs per context. This evidence applies to this checkpoint, not
every Qwen2 conversion or task.

| Model | Format | Architecture | Adapter | Short generation | Status |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct-4bit | MLX affine Q4, group 64 | `qwen2` / `Qwen2ForCausalLM` | Qwen2Adapter | PASS twice: `Hello! How can` (4 tokens) | context-pass 1K/4K/16K; five-run benchmark below |
| Qwen2.5-7B-Instruct-4bit | MLX affine Q4, group 64 | `qwen2` / `Qwen2ForCausalLM` | Qwen2Adapter | PASS twice with identical tokens: `Hello! How can` (4 tokens) | context-pass 1K/4K/16K; 32K boundary-pass; 6.297 GiB peak |
| Qwen2.5-Coder-7B-Instruct-4bit | MLX affine Q4, group 64 | `qwen2` / `Qwen2ForCausalLM` | Qwen2Adapter | PASS twice with identical tokens; code-generation smoke returned Python | 1K/4K/8K/16K matrix completed; EOS before 128 at 4K/16K; 32K boundary-pass; 6.297 GiB peak |
| Qwen2.5-VL-7B-Instruct-4bit | MLX affine Q4, group 64 | `qwen2_5_vl` / `Qwen2_5_VLForConditionalGeneration` | Shared `MLXVLMAdapter` / `mlx-vlm` | PASS: 8 correctness cases x2, streaming, local-file image and ordered multi-image | Functional smoke/regression PASS; visual robustness, video, long-context and broad quality benchmark pending |

### Qwen2.5-VL-7B-Instruct-4bit

The local checkpoint is complete (two weight shards, index, tokenizer, and
processor metadata; 5.3 GiB on disk). Its descriptor reports `qwen2_5_vl`,
Q4/group 64, 28 layers, hidden size 3,584, and a vision configuration. MiniMLX
now discovers its vision components and resolves it to the family-neutral
`mlx-vlm-generation` route and shared `MLXVLMAdapter`; there is no Qwen2.5-
specific VLM branch. Qwen3.5 retains its higher-priority route but uses the
same VLM backend base, with its optional vision-feature cache layered on top.

On the Mac's Metal device with `mlx-vlm 0.7.2` (MLX 0.32.2, mlx-lm 0.31.3),
MiniMLX itself passed a deterministic, project-authored correctness suite against
the local checkpoint. Eight case types were each run twice with greedy decoding;
all 16 answers met their explicit expected-answer checks, and each pair of
repeated outputs matched:

- Text-only control, exact two-line OCR, and OCR via both base64 image content and
  a `data:` URI.
- Visual counting (three red circles), negative object-presence check (no yellow
  star), and center-object shape/color identification.
- Ordered two-image comparison and a multi-turn image follow-up.
- OCR through streaming deltas.
- Existing regression cases using `Benchmarks/fixtures/swift-t-banner.png`:
  local-file image URI and ordered two-image input, each run twice and passed.

The repeatable in-memory suite is `tools/evaluate_vlm_correctness.py`; it emits
per-case answers and timing diagnostics as JSON Lines. These are controlled
functional probes with known answers, not a standardized visual benchmark and
not evidence of broad real-world visual reasoning quality. The result validates
the shared MiniMLX route and these image/request forms on Qwen2.5-VL, not every
architecture supported by upstream `mlx-vlm`. Aspect-ratio/resolution stress,
video, long-context, broader conversation/API/UI integration, and Qwen3.5
cache-regression checks remain pending.

Qwen2.5-7B test environment: MacBook Pro M5 Pro, 64 GB, macOS 27.0.1,
Python 3.13.12, MLX 0.32.2, mlx-lm 0.31.3. Model files occupy 4.0 GiB.
The runtime resolved `qwen2-mlx-lm` / `qwen2-runtime`; shape: 28 layers,
hidden size 3,584, intermediate size 18,944, 28 attention heads, 4 KV heads,
32,768 max positions, vocabulary 152,064, affine Q4 group size 64. Two
independent fixed-prompt smoke runs generated identical token IDs
`[9707, 0, 2585, 646]` (`Hello! How can`).

The 1K/4K/16K matrix used warmup 32, maximum 128 generated tokens, five
measured runs per context, and a 1,024-token prefill step. All 15 measured runs
generated 128/128 tokens. Values below are medians; raw samples are listed in
run order as `TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms`.

| Context | Median TTFT | Median TPOT | Median PP TPS | Median TG TPS | Generated | Median E2E | Raw samples (TTFT / TPOT / PP TPS / TG TPS / E2E) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 534.9 ms | 16.07 ms | 1,914.4 | 62.73 | 128/128 | 2,900.6 ms | 545.9/15.99/1876.0/63.01/2904.8; 537.8/16.13/1904.1/62.48/2900.6; 534.9/16.07/1914.4/62.73/2880.0; 532.6/16.25/1922.6/62.00/2905.2; 531.2/16.05/1927.6/62.81/2880.2 |
| 4,096 | 2,137.2 ms | 16.83 ms | 1,916.5 | 59.90 | 128/128 | 4,594.2 ms | 2141.5/16.78/1912.6/60.05/4601.8; 2137.2/16.83/1916.5/59.90/4593.8; 2135.4/16.84/1918.1/59.84/4591.6; 2135.2/16.86/1918.3/59.77/4594.2; 2142.9/16.80/1911.4/59.99/4594.7 |
| 16,384 | 10,807.7 ms | 20.00 ms | 1,516.0 | 50.40 | 128/128 | 13,702.6 ms | 10331.0/19.93/1585.9/50.57/13181.5; 10807.7/20.07/1516.0/50.22/13702.6; 10874.7/19.94/1506.6/50.54/13742.1; 10906.0/20.00/1502.3/50.40/13787.9; 10778.6/20.09/1520.1/50.18/13667.9 |

The 1K/4K/16K suite peak MLX memory was 5.483 GiB. The maximum-position
boundary check used 32,767 prompt tokens plus one generated token (one run):
PASS, TTFT 24,451.3 ms, PP 1,340.1 TPS, E2E 24,773.7 ms, peak MLX memory
6.297 GiB. It does not claim a 128-token generation at a 32K prompt because
that would exceed this checkpoint's configured 32,768-position window.
The tokenizer warned that the full source corpus tokenizes to 315,515 tokens;
the harness then sliced and verified each measured prompt at its requested
exact length before inference.

### Qwen2.5-Coder-7B-Instruct-4bit

The local checkpoint resolves to `qwen2` / `Qwen2ForCausalLM`, using the same
`qwen2-mlx-lm` family adapter and `qwen2-runtime` backend as Qwen2.5-7B
Instruct. It is MLX affine Q4, group size 64, with 28 layers, hidden size
3,584, intermediate size 18,944, 28 attention heads, 4 KV heads, and 32,768
maximum positions. The local model directory occupies 4.0 GiB. Two independent
`Hello` smoke runs produced identical token IDs `[9707, 0, 2585, 646]`
(`Hello! How can`). A Coder-specific request to write a documented Fibonacci
function returned Python code for the first 64 generated tokens; the output
was cut off by the requested token limit, so this checks the code-generation
path, not execution or full completion correctness.

The performance matrix used warmup 32, maximum 128 generated tokens, five
measured runs per context, and a 1,024-token prefill step. Values are medians;
raw samples are `TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms`.

| Context | Median TTFT | Median TPOT | Median PP TPS | Median TG TPS | Median generated | Median E2E | Raw samples (TTFT / TPOT / PP TPS / TG TPS / E2E) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 533.1 ms | 15.80 ms | 1,920.8 | 63.80 | 128/128 | 2,849.1 ms | 533.1/15.80/1920.8/63.80/2888.1; 539.2/15.74/1899.0/64.02/2845.6; 528.3/15.90/1938.3/63.38/2849.1; 531.4/15.79/1927.1/63.84/2845.6; 533.2/15.82/1920.4/63.70/2862.9 |
| 4,096 | 2,129.1 ms | 16.66 ms | 1,923.8 | 60.92 | 69/128 | 3,579.7 ms | 2129.1/16.66/1923.8/60.92/3566.9; 2130.8/16.57/1922.3/61.23/3576.5; 2132.3/16.53/1920.9/61.38/3580.7; 2120.9/16.72/1931.3/60.70/3579.7; 2123.0/16.70/1929.4/60.78/3582.9 |
| 8,192 | 4,629.6 ms | 17.98 ms | 1,769.5 | 56.05 | 128/128 | 7,293.7 ms | 4472.7/17.93/1831.6/56.22/7069.2; 4495.6/17.90/1822.2/56.30/7091.6; 4629.6/18.43/1769.5/54.68/7293.7; 4763.4/17.98/1719.8/56.05/7370.8; 4784.7/18.05/1712.1/55.85/7403.9 |
| 16,384 | 10,640.1 ms | 19.83 ms | 1,539.8 | 51.20 | 67/128 | 12,275.2 ms | 10642.2/19.83/1539.5/51.20/12275.2; 10661.1/19.95/1536.8/50.88/12301.5; 10640.1/19.90/1539.8/51.02/12307.1; 10564.3/19.80/1550.9/51.27/12190.6; 10550.5/19.67/1552.9/51.61/12185.8 |

At 4K and 16K, generation stopped early on EOS (median 69 and 67 tokens);
these rows are valid runtime observations but are not directly comparable to
128-token runs. The 32K boundary check used 32,767 prompt tokens plus one
generated token and passed: TTFT 24,002.8 ms, PP 1,365.1 TPS, E2E 24,322.0
ms, peak MLX memory 6.297 GiB. The 1K/4K/8K/16K matrix peak was 5.483 GiB.
The tokenizer emitted the same long-corpus warning as for Qwen2.5-7B
Instruct; exact prompt lengths were sliced and verified by the harness.

Code-generation quality smoke: `tools/evaluate_codegen_cases.py` ran the
eight fixed tasks in `Benchmarks/qwen25_coder_cases.json` at greedy decoding.
Pass@1 was 8/8 tasks, with all 28 unit assertions passing. The harness also
compared each MiniMLX prompt-token sequence and generated-token sequence with
direct `mlx-lm` generation from the same checkpoint; all 8/8 matched exactly.
This parity is specifically a wrapper/template/option regression check—MiniMLX
itself uses `mlx-lm`'s `stream_generate`, so it is not an independent check of
the underlying MLX kernels or a general code-quality score. Generated code is
run in a child interpreter with an AST allowlist, a restricted builtin set, no
imports, and a timeout. This is a small project-authored smoke set, not a
standardized HumanEval/MBPP score.

Commands: `PYTHONPATH=python .venv/bin/python tools/verify_model_support.py
<model-dir>` (run twice) and `PYTHONPATH=python
.venv/bin/python tools/benchmark_context_matrix.py <model-dir>
Resources/BenchmarkCorpus/code_python.txt --warmup 32 --max-tokens 128
--repeats 5 --step 1024 --contexts 1024 4096 16384`. The boundary run used
`--warmup 1 --max-tokens 1 --repeats 1 --step 1024 --contexts 32767`.

Device: MacBook Pro M5 Pro, 64 GB; Python 3.13, MLX 0.32.2, mlx-lm 0.31.3.
Smoke route: `qwen2-mlx-lm` / `qwen2-runtime`. Shape: 24 layers, hidden size
896, 14 attention heads, 2 KV heads, 32,768 max positions. Peak MLX memory
during the context suite was 1.229 GiB.

| Context | Median TTFT | Median TPOT | Median PP TPS | Median TG TPS | Generated | Median E2E |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 46.7 ms | 2.44 ms | 21,925.5 | 412.72 | 128/128 | 672.0 ms |
| 4,096 | 181.2 ms | 2.95 ms | 22,607.2 | 341.82 | 128/128 | 879.2 ms |
| 16,384 | 1,023.2 ms | 4.96 ms | 16,012.9 | 203.11 | 128/128 | 1,984.5 ms |

Command: `PYTHONPATH=python .venv/bin/python tools/benchmark_context_matrix.py
<model-dir> Resources/BenchmarkCorpus/code_python.txt --warmup 32 --max-tokens
128 --repeats 5 --step 1024 --contexts 1024 4096 16384`.

## Vision-language route

Qwen3.5 checkpoints that declare a vision configuration and contain processor,
vision-encoder, tokenizer, and text-weight components now resolve to the
`qwen35-vlm` provider. Image and text parts are converted at the adapter
boundary and executed through `mlx-vlm`; text-only requests on the same loaded
VLM instance use the same backend. The resolver reports `dependency_blocked`
when `mlx-vlm` is unavailable and `missing_component` when the checkpoint
manifest is incomplete. The native `mlx-vlm` model embedding path is used for
prefill, and projected feature reuse is delegated to
`mlx-vlm.VisionFeatureCache`.

### Device smoke evidence

| Model | Input | Result | Evidence |
|---|---|---|---|
| Qwen3.5-0.8B-MLX-4bit | local PNG + image prompt | PASS | Metal load, image prefill, 4-token generation; second request reused one projected vision feature and one session prompt-cache state |
| Qwen3.5-4B-MLX-4bit | local PNG + image prompt | PASS | Metal load, image prefill, 4-token generation; vision cache metrics reported one miss/store and one entry |
| Qwen3.5-9B-4bit | `Benchmarks/fixtures/swift-t-banner.png` + OCR prompt | PASS | Output contained `Swift` and `Transformers`; 4,161 prompt tokens, 5 completion tokens |

This is a device smoke result, not yet a quality/golden OCR benchmark. Fixed
quality/golden samples are still required before claiming task correctness
across all image understanding tasks.

Image inputs are bounded by default to 16 images per request, 16 MiB per image,
and 40 million pixels per image. Local files, data URIs, and base64 payloads are
accepted; remote HTTP(S) images require explicit `MINIMLX_ALLOW_REMOTE_IMAGES=1`.
The repeatable smoke harness is `tools/verify_vlm_support.py --repeat 2`, which
also reports vision feature cache metrics.

The checked-in regression manifest is `Benchmarks/vlm_cases.json`; run it with
`tools/run_vlm_cases.py` to verify single-image, multi-image, expected text, and
cache reuse behavior.

The BF16 model intentionally skips Q4-only MLP prefill paths. It should still
use the common Qwen3.5 GDN, full-attention, cache, tokenizer, and generation
logic. No BF16-specific performance optimization is being added in this phase.

## BF16 verification result

- 24 layers: 18 linear-attention/GDN and 6 full-attention
- Q4 MLP layers detected: 0
- Prompt: token `[9419]`
- Generated tokens: `[11, 271, 40, 1044]`
- Text: `",\\n\\nI am"`
- Prefill: 107.695 ms
- Decode: 49.560 ms
- Result: `compatibility=PASS`

## BF16 context matrix

Configuration: `warmup=32`, `repeats=5`, median, generated tokens `128`,
prefill step `1024`, corpus `code_python.txt`.

| Context | TTFT | TPOT | TG TPS | E2E |
|---:|---:|---:|---:|---:|
| 1K | 448.8 ms | 16.68 ms | 59.97 | 2583.3 ms |
| 2K | 871.4 ms | 16.75 ms | 59.70 | 3015.6 ms |
| 4K | 1752.8 ms | 17.10 ms | 58.50 | 3941.0 ms |
| 8K | 3829.2 ms | 17.54 ms | 57.02 | 6074.0 ms |
| 16K | 8939.7 ms | 18.73 ms | 53.40 | 11336.9 ms |

Result: all tested context lengths passed. The higher BF16 TPOT relative to
the 4-bit model is expected from the larger unquantized weight bandwidth and
does not indicate a compatibility failure.

## Qwen3.5-4B BF16 smoke result

- Architecture: `Qwen3_5ForConditionalGeneration`
- Layers: 32 total, 24 linear-attention/GDN and 8 full-attention
- Q4 MLP layers: 0 (expected for BF16)
- Prompt tokens: `[9419]`
- Generated tokens: `[11, 353, 2688, 4313]`
- Text: `", I'm trying"`
- Prefill: 867.459 ms
- Decode: 279.811 ms
- Result: `compatibility=PASS`
- Context checks: 1K, 4K, and 16K; each completed with 4 generated tokens.
- Context-check TTFT: 2572.2 ms / 10071.8 ms / 40317.6 ms.
- Context-check E2E: 2833.7 ms / 10336.6 ms / 40609.2 ms.
- Smoke peak MLX memory: 7.899 GiB on the 24 GB MacBook Air M4.
- These single-run short-generation
  timings are compatibility evidence, not a performance comparison.

## BF16 MiniMLX / oMLX comparison

Assuming the screenshot uses the same model, prompt corpus, 128 generated
tokens, warm-up, repeats, and single-request settings:

| Context | MiniMLX TTFT | oMLX TTFT | MiniMLX TPOT | oMLX TPOT | MiniMLX TG TPS | oMLX TG TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 448.8 ms | 501.4 ms | 16.68 ms | 18.7 ms | 59.97 | 53.8 |
| 4K | 1752.8 ms | 2506.1 ms | 17.10 ms | 19.0 ms | 58.50 | 53.1 |
| 8K | 3829.2 ms | 4798.6 ms | 17.54 ms | 19.7 ms | 57.02 | 51.2 |
| 16K | 8939.7 ms | 10992.6 ms | 18.73 ms | 21.2 ms | 53.40 | 47.5 |

MiniMLX is faster in both Prefill and Decode at all four lengths. Relative to
oMLX, TTFT is lower by approximately 10.5%, 30.0%, 20.2%, and 18.7%; TPOT is
lower by approximately 10.8%, 10.0%, 11.0%, and 11.7%. This is a performance
comparison, not part of the compatibility acceptance gate.

## Verification command

```bash
cd /Users/jizhishi/mycode/ai/minimlx
./.venv/bin/python tools/verify_model_support.py \
  /Users/jizhishi/.omlx/models/mlx-community/Qwen3.5-0.8B-MLX-bf16
```

## Acceptance criteria

1. Descriptor resolves to the Qwen3.5 family adapter.
2. Model loads without quantization-only assumptions.
3. Layer counts and linear/full-attention split are reported.
4. Short prefill and greedy decode complete.
5. Generated output is non-empty and the process exits cleanly.

## Llama 3.2 3B Instruct MLX 4-bit

- Checkpoint: `mlx-community/Llama-3.2-3B-Instruct-4bit`
- Architecture: `llama` / `LlamaForCausalLM`; 28 layers, hidden size 3072,
  24 attention heads, 8 KV heads, max positions 131,072.
- Quantization: MLX affine Q4, group size 64; checkpoint size 1.807 GB.
- Route: `mlx-lm-standard` baseline / `mlx-lm-standard-runtime`; no acceleration
  and no optimization provider. This validates the generic route for this
  checkpoint only.
- Compatibility: two independent Metal loads and 16-token greedy generations
  passed. Both runs produced the same token sequence
  `[37046, 21043, 48044, 17792, 49792, 118034, 103129, 46034, 3922, 113961,
  104908, 28469, 5486, 126427, 34208, 122903]` and text
  `我是一个人工智能助手，能够提供信息、答案和建议`.
- Hardware: MacBook Pro, Apple M5 Pro, 64 GB unified memory; macOS 27.0.1.
- Runtime: Python 3.13, MLX 0.32.2, mlx-lm 0.31.3, Transformers 5.17.0,
  Tokenizers 0.23.2. Source checkout HEAD at test time: `188ed46` plus the
  uncommitted architecture/overlay changes in this worktree.
- Context matrix: `warmup=32`, `max_tokens=128`, five measured runs per
  context, prefill step 1024, `Resources/BenchmarkCorpus/code_python.txt`,
  EOS enabled. Every measured run generated 128/128 tokens.

| Context | Median TTFT | Median TPOT | Median PP TPS | Median TG TPS | Generated | Median E2E | Peak MLX memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 259.9 ms | 8.88 ms | 3,940.3 | 113.52 | 128/128 | 1,691.6 ms | 2.460 GiB |
| 4K | 1,083.2 ms | 10.42 ms | 3,781.2 | 96.74 | 128/128 | 2,709.1 ms | 2.691 GiB |
| 8K | 2,289.6 ms | 11.29 ms | 3,577.9 | 89.31 | 128/128 | 3,988.4 ms | 3.181 GiB |
| 16K | 5,998.9 ms | 16.63 ms | 2,731.2 | 60.62 | 128/128 | 8,394.9 ms | 3.986 GiB |

Five raw samples per context (TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms):

- 1K: `259.9/8.84/3940.3/114.05/1654.4`; `259.7/9.12/3942.4/110.57/1705.7`;
  `259.8/8.88/3940.9/113.52/1691.6`; `260.1/8.87/3937.5/113.60/1665.4`;
  `261.4/8.97/3916.6/112.41/1708.9`.
- 4K: `1080.9/10.47/3789.4/96.25/2709.1`; `1083.2/10.37/3781.2/97.14/2682.6`;
  `1097.9/10.42/3730.6/96.74/2717.1`; `1082.5/10.50/3783.8/96.03/2697.8`;
  `1089.1/10.41/3760.8/96.80/2714.6`.
- 8K: `2581.1/13.46/3173.8/74.88/4574.0`; `2582.8/13.99/3171.8/72.04/4651.7`;
  `2236.6/11.29/3662.8/89.31/3982.3`; `2239.0/11.13/3658.8/90.59/3926.7`;
  `2289.6/11.20/3577.9/89.95/3988.4`.
- 16K: `5751.7/16.56/2848.6/60.87/8141.8`; `5818.4/16.68/2815.9/60.44/8223.5`;
  `5998.9/16.60/2731.2/60.72/8394.9`; `6151.3/16.83/2663.5/59.89/8583.8`;
  `6198.0/16.63/2643.4/60.62/8606.7`.

- 32K boundary: 32,767 prompt tokens + 1 generated token passed; TTFT 14,864.2 ms,
  PP 2,204.4 TPS, E2E 15,155.1 ms, suite peak 5.726 GiB. Single-token TPOT/TG TPS
  are not meaningful performance measurements.
- The benchmark tokenizer emitted a long-corpus length warning because it
  tokenizes the full 313,611-token source before slicing to each requested
  context. Exact requested contexts were processed and all generated tokens
  completed; the warning did not indicate a model context failure.
- Commands: `tools/verify_model_support.py <model-dir> --prompt '用一句话介绍你自己。' --max-tokens 16`
  (run twice); `tools/benchmark_context_matrix.py <model-dir>
  Resources/BenchmarkCorpus/code_python.txt --warmup 32 --max-tokens 128
  --repeats 5 --step 1024 --contexts 1024 4096 8192 16384`; and the 32K
  boundary variant with `--warmup 1 --max-tokens 1 --repeats 1 --contexts 32767`.

Result: supported for text generation on this checkpoint under the project's
load/smoke/context gate; repeated greedy output was token-identical and context
passed through 32K. This is not a separate language-quality evaluation or a
benchmark comparison against another runtime. No Llama-specific optimization
was added.

## Gemma 4 12B IT MLX 4-bit

- Checkpoint: `mlx-community/gemma-4-12B-it-4bit`; local directory
  `~/.minimlx/models/mlx-community--gemma-4-12B-it-4bit`.
- Architecture: `gemma4_unified` / `Gemma4UnifiedForConditionalGeneration`;
  48 text layers, hidden size 3840, 16 attention heads, 8 KV heads, maximum
  position length 262,144. The checkpoint also contains vision and audio
  components; this run validates text generation only.
- Quantization: MLX affine Q4, group size 64; checkpoint size 6.3 GiB on disk.
- Route: family-neutral `mlx-vlm-generation` baseline / `mlx-vlm-runtime`;
  no Gemma-specific adapter, acceleration, or optimization provider.
- Compatibility: two independent Metal loads and 8-token greedy generations
  passed. Both produced the identical token sequence
  `[236776, 2028, 25537, 563, 496, 1944, 1076, 529]` and text
  `A model adapter is a small set of`.
- Context matrix: warmup 32 tokens at 2K context; five measured runs per
  context; 128 output tokens; prefill step 1024; EOS enabled. All 15 runs
  generated 128/128 tokens.
- Hardware: MacBook Pro, Apple M5 Pro, 64 GB unified memory; macOS 27.0.1.
- Runtime: Python 3.13, MLX 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.7.2.

| Context | Median TTFT | Median TPOT | Median PP TPS | Median TG TPS | Generated | Median E2E | Peak MLX memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 948.7 ms | 28.12 ms | 1,079.3 | 35.84 | 128/128 | 4,922.1 ms | 7.347 GiB |
| 4K | 3,850.0 ms | 28.71 ms | 1,063.9 | 35.11 | 128/128 | 7,919.8 ms | 7.771 GiB |
| 8K | 8,309.6 ms | 29.18 ms | 985.9 | 34.54 | 128/128 | 12,444.2 ms | 7.985 GiB |
| 16K | 17,704.0 ms | 29.52 ms | 925.4 | 34.14 | 128/128 | 21,887.7 ms | 8.442 GiB |

- Route resolution initially selected `mlx-lm-standard` because the broad
  `gemma` family token matched `gemma4_unified`. The family-neutral VLM route
  now ranks above broad text-family matches when required vision components
  are present; Qwen3.5's more specific provider keeps higher priority. The
  corrected plan was confirmed by both inspect and actual model loading.
- Commands: `tools/verify_model_support.py <model-dir> --prompt 'Write a short
  sentence explaining what a model adapter does.' --max-tokens 8` (run twice);
  `tools/benchmark_context_matrix.py <model-dir>
  Resources/BenchmarkCorpus/code_python.txt --warmup 32 --max-tokens 128
  --repeats 5 --step 1024 --contexts 1024 4096 8192 16384`. The 8K context
  was measured in a separate follow-up run with the same settings.

Result: `loadable`, `smoke-pass`, and text-only `context-pass` through 16K for
this Gemma 4 checkpoint. Image and audio input are not validated by these runs;
do not treat the text results as multimodal acceptance or a cross-runtime
performance comparison.

## Mistral Nemo Instruct 2407 MLX 4-bit

- Checkpoint: `mlx-community/Mistral-Nemo-Instruct-2407-4bit`; local directory
  `~/.minimlx/models/mlx-community--Mistral-Nemo-Instruct-2407-4bit`.
- Architecture: `MistralForCausalLM` / `mistral`, 40 layers, hidden size 5120;
  maximum position length 1,024,000. Quantization is MLX affine Q4, group size
  64.
- Route: shared `StandardTextAdapter` / `mlx-lm-standard-runtime`; this is the
  generic MLX-LM baseline, with no Mistral-specific adapter or performance
  overlay.
- Tokenizer: the checkpoint carries the known Mistral regex issue. The shared
  adapter now enables upstream `fix_mistral_regex=True` for `model_type=mistral`;
  the subsequent smoke and benchmark loads completed without that warning.
- Smoke: PASS; exact-response prompt generated `Mistral test passed` (6 tokens).
- Context matrix: code corpus, warmup 32 tokens at context=2048, output cap 128,
  five measurements per context, prefill step 1024, EOS enabled. All 20 measured
  runs generated 128/128 tokens.
- Hardware: MacBook Pro, Apple M5 Pro, 64 GB unified memory; test ran in the
  interactive macOS Metal session.
- Runtime kernels: Q4 native prefill and native FA256 enabled; Q4 dense prefill
  disabled; Q4 MLP fusion disabled; KV quantization off; autoregressive decode.

| Context | Median TTFT | Median TPOT | Median PP TPS | Median TG TPS | Generated | Median E2E | Peak MLX memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 1,253.0 ms | 38.42 ms | 817.3 | 26.23 | 128/128 | 6,442.1 ms | 7.290 GiB |
| 4K | 5,222.1 ms | 41.58 ms | 784.4 | 24.24 | 128/128 | 10,830.7 ms | 7.592 GiB |
| 8K | 10,885.2 ms | 45.34 ms | 752.6 | 22.23 | 128/128 | 16,965.1 ms | 8.264 GiB |
| 16K | 24,561.7 ms | 53.60 ms | 667.1 | 18.80 | 128/128 | 31,706.4 ms | 9.469 GiB |

Result: supported for text generation on this checkpoint under the generic
MLX-LM baseline and validated through 16K context. This test did not assess
instruction-following quality, multimodal tasks, or performance against oMLX.
No Mistral-specific optimization layer was added.
