# Qwen3.5 family coverage plan

This is the Qwen3.5 family completion checklist. Local checkpoints are test
fixtures; their presence or absence does not define whether a family variant is
in scope. Qwen3.8 and later Qwen generations remain queued until this matrix is
resolved.

## Canonical model configurations

The official Qwen3.5 release includes dense 0.8B, 2B, 4B, 9B, and 27B models,
plus MoE 35B-A3B, 122B-A10B, and 397B-A17B models. Record each canonical
configuration independently from its quantized or community-converted files.

| Canonical configuration | Architecture | Text | Vision | Provider route | Runtime evidence |
|---|---|---:|---:|---|---|
| Qwen3.5-0.8B | Dense | Yes | Yes | Static text/VLM route pass | Q4 image/text; BF16 text |
| Qwen3.5-2B | Dense | Yes | Yes | Static text/VLM route pass | MLX Q8 text/image/context pass; official BF16 not run |
| Qwen3.5-4B | Dense | Yes | Yes | Static text/VLM route pass | Q4 image/text; BF16 text/context |
| Qwen3.5-9B | Dense | Yes | Yes | Static text/VLM route pass | `mlx-community/Qwen3.5-9B-4bit`: text, image and 1K–16K MiniMLX matrix run; see ledger caveats |
| Qwen3.5-27B | Dense | Yes | Yes | Static text/VLM route pass | `mlx-community/Qwen3.5-27B-4bit`: text, image/cache and 1K–16K MiniMLX matrix pass; see ledger |
| Qwen3.5-35B-A3B | MoE | Yes | Yes | Static text/VLM route pass | Related REAP-19B derivative tested; canonical checkpoint not run |
| Qwen3.5-122B-A10B | MoE | Yes | Yes | Static text/VLM route pass | Not run |
| Qwen3.5-397B-A17B | MoE | Yes | Yes | Static text/VLM route pass | Not run |

The official release list is maintained in the [Qwen model repository](https://github.com/QwenLM/Qwen3.8).
Its MLX guidance currently identifies `mlx-lm` for text and `mlx-vlm` for
vision-language execution of Qwen3.5; MiniMLX must still validate its own
provider route and version compatibility.

## Quantization coverage

The conversion ecosystem is open-ended, so coverage is defined by weight
format, not every user fine-tune or every repackaged repository.

| Weight class | Required coverage | Acceptance |
|---|---|---|
| MLX-format BF16/FP16 checkpoints (when available) | Representative Dense and MoE architecture classes | Load, greedy correctness, context safety; raw Transformers-format imports are a separate conversion workflow |
| MLX affine Q4 | Dense and MoE; small, medium, and large configurations where artifacts exist | Quant metadata/shape validation, load, generation correctness |
| MLX affine Q8 | Dense and MoE representative configurations | Same checks as Q4; no Q4-only kernel assumptions |
| Other MLX quantization layouts (including mixed-bit/group-size variants) | One fixture per distinct layout/loader contract | Explicitly supported or a structured incompatibility reason |
| MTP draft weights | Paired target/drafter combinations | Existing opt-in MTP correctness contract; never treated as standalone models |

### Evidence currently available

| Format / configuration | Evidence | Status |
|---|---|---|
| BF16 Qwen3.5-0.8B and 4B | Text runtime; 4B also has context evidence | Partial runtime validation |
| Affine Q4, group size 64, Qwen3.5-0.8B and 4B | Text and image runtime | Partial runtime validation; does not generalize to all Q4 artifacts |
| Qwen3.5-2B MLX affine Q8, group size 64 | Text generation matched the expected output; single-image and two-image regression cases passed; 4,101-token prompt generated 4 tokens; peak memory 2.576 GiB | Runtime pass for this converted checkpoint; not proof for every Q8 layout |
| Mixed per-layer Q4/Q8 derivative based on Qwen3.5-35B-A3B | Descriptor preserves mixed quantization metadata; a community REAP-19B derivative has runtime evidence | Derivative only; canonical 35B and general mixed-bit loader coverage remain unverified |
| Standalone affine Q8 | Qwen3.5-2B MLX conversion, revision `562fb8ea19bbe7565a3ddf5b7d49899ea2c88d2b` | One Dense size/conversion passes; MoE Q8 remains untested |

The Qwen3.5-2B Q8 runtime run used MiniMLX's `mlx-vlm 0.7.2`, `mlx 0.32.2`,
and `mlx-lm 0.31.3`. The validation script now reports quantized Dense MLP
layers by bit width; this run confirmed `{"8": 24}` and zero Q4 Dense layers.

Discovery tests now assert that nested per-layer quantization overrides survive
descriptor creation unchanged. This is metadata preservation only: it does not
assert that the backend can load or execute every such layout.

## Test ledger and measurement protocol (source of truth)

The user-approved scope is **Qwen first, then other mainstream model families**.
For this phase, select artifacts from `mlx-community` where available. Popularity
and family/architecture coverage determine priority; checkpoint size is recorded
as a practical runtime/download constraint, never used to exclude or demote a
model. Do not switch to another family while Qwen3.5 items below remain open.

### What we measure for each checkpoint

1. Identity: exact repository and revision, architecture (Dense/MoE), quantization
   bits/group size/layout, shard manifest, processor/tokenizer and dependency versions.
2. Compatibility: discovery/descriptor, provider route, model load, deterministic
   short text generation, and image input for vision-capable checkpoints.
3. Context/performance: use the same code corpus and benchmark procedure for each
   comparable run: warm-up 32 tokens; max generation 128 tokens; contexts 1K, 4K,
   8K and 16K; step 1024. Run serially, without concurrent model downloads or
   inference workloads. Record each run separately; headline figures use the
   median of five runs and retain raw samples to expose variance.
4. Report, per context: TTFT, TPOT, prompt-processing tokens/s (PP TPS),
   generation tokens/s (TG TPS), end-to-end latency, aggregate throughput,
   peak memory, and actual generated-token count. EOS before `max_tokens` is
   recorded, not treated as a benchmark failure.
5. Keep correctness, context safety, and performance as separate outcomes.
   Mark hardware-limited cases as such; do not label them software-unsupported.

Performance testing here is **MiniMLX's own benchmark and report**. We do not
run oMLX benchmarks as part of this plan. External comparisons can be made later
from separately collected results using the recorded protocol and hardware.

### Completed and observed tests

| Checkpoint | Status | Evidence / caveat |
|---|---|---|
| `mlx-community/Qwen3.5-2B-8bit` | Performance matrix complete; text, image and context checks pass | Three manually repeated context runs were pasted in the task. Full raw performance observations are recorded below; one run had an anomalous 1K TTFT. |
| `mlx-community/Qwen3.5-4B-MLX-4bit` | Tested; do not queue again | User confirmed it has already been tested. Preserve any existing detailed result in the benchmark report; no new numbers are inferred here. |
| Qwen3.5-0.8B Q4/BF16; Qwen3.5-4B BF16/Q4 | Runtime evidence exists | See canonical/evidence tables above. These entries do not substitute for a missing performance report. |
| Qwen3.5-35B-A3B-based REAP-19B derivative | Derivative tested, canonical model still open | Not equivalent to testing the canonical 35B-A3B checkpoint. |
| `mlx-community/Qwen3.5-9B-8bit` | Attempt cancelled before weight transfer | Log showed metadata requests and a `vocab.json` timeout/retry; no safetensors GET was observed. Not a completed test. |
| `mlx-community/Qwen3.5-9B-4bit` | Download complete; text, image and performance matrix completed | Exact local artifact validated as `Qwen3_5ForConditionalGeneration`, affine Q4/group 64, 32 layers (24 linear-attention, 8 full-attention). Text smoke and image OCR pass. Performance matrix uses 3 runs/context; early EOS yielded only 29 tokens at 1K and 16 at 16K, so compare those TG/E2E values cautiously. Revision not yet captured. |
| `mlx-community/Qwen3.5-27B-4bit` | Complete: descriptor, text smoke, image/cache smoke, and five-run context matrix | Dense Q4 affine/group 64; 64 layers; all 3 shards present. Text context matrix passed at 1K/4K/8K/16K; 1K naturally stopped at 111/128 tokens. Image OCR returned `Swift / Transformers`; two-request vision cache miss then hit. Peak memory 18.465 GiB. Cached revision `45797d2985a12c55e6473686e9ea91b95e959553`. Full raw measurements below. |

#### Raw Qwen3.5-2B-8bit context observations

The user ran the same command three times with one measured pass per invocation.
Keep these raw runs rather than averaging away the observed variance.

| Run | Context | TTFT ms | TPOT ms | PP TPS | TG TPS | Generated | E2E ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,024 | 164.3 | 8.51 | 6,231.2 | 118.42 | 124/128 | 1,494.0 |
| 1 | 4,096 | 578.3 | 8.64 | 7,083.1 | 116.87 | 105/128 | 1,748.0 |
| 1 | 8,192 | 1,162.9 | 8.75 | 7,044.3 | 115.18 | 128/128 | 2,552.8 |
| 1 | 16,384 | 2,430.2 | 9.01 | 6,741.7 | 111.91 | 128/128 | 3,848.3 |
| 2 | 1,024 | 523.1 | 8.66 | 1,957.4 | 116.43 | 124/128 | 1,886.5 |
| 2 | 4,096 | 664.6 | 8.65 | 6,162.8 | 116.70 | 105/128 | 1,852.4 |
| 2 | 8,192 | 1,157.1 | 8.90 | 7,079.8 | 113.19 | 128/128 | 2,573.2 |
| 2 | 16,384 | 2,446.0 | 9.18 | 6,698.4 | 109.80 | 128/128 | 3,899.7 |
| 3 | 1,024 | 164.1 | 8.43 | 6,238.7 | 119.56 | 124/128 | 1,487.1 |
| 3 | 4,096 | 577.2 | 8.55 | 7,096.6 | 118.02 | 105/128 | 1,743.5 |
| 3 | 8,192 | 1,158.4 | 8.77 | 7,071.5 | 114.89 | 128/128 | 2,548.5 |
| 3 | 16,384 | 2,434.4 | 9.15 | 6,730.2 | 110.09 | 128/128 | 3,878.4 |

Run 2's 1K result is a clear outlier relative to runs 1 and 3; report it, but
do not use that single observation as the typical 1K performance. The captured
terminal output does not include peak memory for these three runs.
The benchmark UI screenshot identified the machine as M5 Pro / 64 GB / 20 GPU
cores, while the shell prompt hostname was `MacBook-Air`; because the hostname
does not prove the hardware, attach this run to a specific hardware row only
after confirming the actual system report. The historical terminal output also
does not include the model revision or peak-memory values; future reports must.

### Ordered Qwen3.5 queue

Queue order reflects Qwen3.5 popularity and the remaining family/architecture
coverage, not a preference for smaller files. Exact repository IDs are recorded
so the next recommendation cannot drift between similarly named conversions.

| Order | Exact `mlx-community` checkpoint | Purpose / state |
|---:|---|---|
| 1 | [`mlx-community/Qwen3.5-9B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) | Completed: text/image correctness and full MiniMLX matrix; see results below. |
| 2 | [`mlx-community/Qwen3.5-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | Completed: text, image/cache and full MiniMLX matrix; see results below. |
| 3 | [`mlx-community/Qwen3.5-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) | Next: canonical MoE runtime/performance check; REAP-19B derivative is not a substitute. |
| 4 | Qwen3.5-9B Q8 from `mlx-community`, if a canonical non-finetune artifact is available | Same-size quantization comparison against Q4; verify exact repo/config before acquisition. |
| 5 | Qwen3.5-27B additional supported quantization(s) from `mlx-community` | Quantization/layout coverage; select by actual repository metadata, not name alone. |
| 6 | Qwen3.5-122B-A10B and Qwen3.5-397B-A17B `mlx-community` artifacts, if available | Preserve as in-scope MoE configurations. Assess runtime/hardware feasibility when scheduled; do not remove them from scope based on size. |

#### `mlx-community/Qwen3.5-27B-4bit` completed run

- Local artifact: `/path/to/Qwen3.5-27B-4bit`
- Cached revision: `45797d2985a12c55e6473686e9ea91b95e959553`
- Descriptor: `Qwen3_5ForConditionalGeneration`, `qwen3_5`, Dense, affine Q4/group 64, 64 layers; all three indexed safetensor shards present.
- Environment: Apple M5 Pro, 64 GiB unified memory; Python 3.13.12, MLX 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.7.2; MiniMLX git revision `dbad46b` (working tree had other changes).
- Text smoke: PASS, 4/4 generated; peak 16.830 GiB.
- Vision: PASS, `Benchmarks/fixtures/swift-t-banner.png`; two requests each recognized `Swift / Transformers` (4,155 prompt tokens, 5 completion tokens). Feature cache changed from 1 miss/store to 1 hit; entry count remained 1.
- Performance: MiniMLX-only, autoregressive; warm-up 32, max 128, step 1024, contexts 1024/4096/8192/16384, five measured runs each. Suite peak 18.465 GiB. No oMLX benchmark was run. The 1K request stopped at 111/128; its TG/E2E are retained as observed values, not a full-length decode comparison.
- Command: `PYTHONPATH=python ./.venv/bin/python tools/benchmark_context_matrix.py /path/to/Qwen3.5-27B-4bit Resources/BenchmarkCorpus/code_python.txt --warmup 32 --max-tokens 128 --repeats 5 --step 1024 --contexts 1024 4096 8192 16384`

| Context | Median TTFT ms | Median TPOT ms | Median PP TPS | Median TG TPS | Generated | Median E2E ms | Five raw samples (TTFT / TPOT / PP / TG / E2E) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 2,182.0 | 57.23 | 469.3 | 17.63 | 111/128 | 8,813.1 | 2190.1/57.67/467.6/17.50/8864.0; 2174.1/57.29/471.0/17.62/8813.1; 2174.8/57.23/470.8/17.63/8778.2; 2182.0/57.13/469.3/17.66/8780.5; 2251.2/57.12/454.9/17.67/8844.3 |
| 4,096 | 9,161.0 | 58.31 | 447.1 | 17.28 | 128/128 | 16,875.4 | 9161.0/58.28/447.1/17.29/16875.4; 9290.1/58.36/440.9/17.27/17040.0; 9187.7/58.28/445.8/17.29/16917.2; 9052.1/58.31/452.5/17.28/16788.6; 8974.8/58.36/456.4/17.27/16703.8 |
| 8,192 | 22,013.1 | 69.24 | 372.1 | 14.56 | 128/128 | 31,245.3 | 18504.9/66.34/442.7/15.19/27254.8; 19921.5/64.30/411.2/15.67/28430.1; 22111.4/69.24/370.5/14.56/31245.3; 22013.1/70.85/372.1/14.22/31371.5; 23400.1/78.10/350.1/12.91/33666.9 |
| 16,384 | 42,153.1 | 63.91 | 388.7 | 15.77 | 128/128 | 50,584.8 | 48468.2/73.45/338.0/13.72/58150.9; 46336.0/66.51/353.6/15.15/55115.4; 42153.1/63.91/388.7/15.77/50584.8; 41351.9/63.87/396.2/15.78/49773.5; 40902.2/62.69/400.6/16.08/49178.6 |

##### One-pass MiniMLX sample for comparison with the user-supplied oMLX screenshot

This is a separate run with one measurement per context, not a replacement for
the five-run median above. It was run after the five-run suite and therefore
does not establish a fully cold-machine baseline. Suite peak was 18.464 GiB;
1K again ended naturally at 111/128 tokens.

Command: `PYTHONUNBUFFERED=1 PYTHONPATH=python ./.venv/bin/python tools/benchmark_context_matrix.py /path/to/Qwen3.5-27B-4bit Resources/BenchmarkCorpus/code_python.txt --warmup 32 --max-tokens 128 --repeats 1 --step 1024 --contexts 1024 4096 8192 16384`

| Context | TTFT ms | TPOT ms | PP TPS | TG TPS | Generated | E2E ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 2,176.0 | 57.20 | 470.6 | 17.64 | 111/128 | 8,792.9 |
| 4,096 | 8,378.5 | 58.04 | 488.9 | 17.36 | 128/128 | 16,057.0 |
| 8,192 | 17,513.0 | 59.28 | 467.8 | 17.00 | 128/128 | 25,352.2 |
| 16,384 | 37,967.9 | 61.78 | 431.5 | 16.31 | 128/128 | 46,134.3 |

#### `mlx-community/Qwen3.5-9B-4bit` completed run

- Local artifact: `/path/to/Qwen3.5-9B-4bit`
- Descriptor: `Qwen3_5ForConditionalGeneration`; affine Q4, group size 64; 32 layers; `{"4": 32}` quantized Dense MLP layers; autoregressive decode.
- Text smoke: PASS; output `Hello! How can`; 5.656 GiB peak.
- Vision: PASS on `Benchmarks/fixtures/swift-t-banner.png`; output `Swift / Transformers`; prompt 4,161 tokens, completion 5 tokens.
- Performance: MiniMLX `benchmark_context_matrix.py`, warm-up 32, max 128, step 2048, contexts 1024/4096/8192/16384, 3 measured runs each; autoregressive; suite peak 7.680 GiB. Values below are medians with the three raw samples retained. Sample order is TTFT ms / TPOT ms / PP TPS / TG TPS / E2E ms.

| Context | Median TTFT ms | Median TPOT ms | Median PP TPS | Median TG TPS | Generated | Median E2E ms | Three samples (TTFT / TPOT / PP / TG / E2E) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 620.7 | 18.14 | 1,649.8 | 57.11 | 29/128 | 1,429.6 | 620.7/18.05/1649.8/57.37/1431.6; 618.7/18.14/1655.0/57.11/1426.1; 627.7/18.23/1631.4/56.81/1429.6 |
| 4,096 | 2,387.1 | 18.55 | 1,715.9 | 54.34 | 128/128 | 5,035.6 | 2383.9/18.60/1718.2/54.20/5035.6; 2388.2/18.53/1715.1/54.39/5030.5; 2387.1/18.55/1715.9/54.34/5040.5 |
| 8,192 | 4,835.0 | 19.12 | 1,694.3 | 52.72 | 128/128 | 7,584.4 | 4834.9/19.12/1694.4/52.72/7566.5; 4835.0/19.18/1694.3/52.55/7584.4; 4855.7/19.03/1687.1/52.95/7585.1 |
| 16,384 | 10,321.5 | 20.34 | 1,587.4 | 52.43 | 16/128 | 10,944.0 | 10101.9/20.34/1621.9/52.43/10711.7; 10321.5/20.81/1587.4/51.27/10944.0; 10406.5/20.22/1574.4/52.77/11034.6 |

The 1K and 16K requests stopped before 128 generated tokens; those observed values
are retained without treating early EOS as a correctness failure. Hardware was
not independently recorded in this execution log, so do not attach an inferred
device claim. This is a MiniMLX-only measurement; no oMLX benchmark was run.

The 2B-8bit and 4B-4bit checkpoints are completed items, not future queue entries.
Do not re-download or re-test them unless a specific missing measurement is
identified. MTP draft weights remain a separate paired-target test, not a model
size entry. Qwen3.8 and other families remain out of sequence until the Qwen3.5
queue and required coverage rows have explicit outcomes.

The raw Transformers-format repository [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B)
is deferred from this queue: oMLX's documented model downloader searches MLX
models, and its serving directory expects MLX-format checkpoints. Test raw HF
conversion/import only if MiniMLX intentionally adds that workflow or an
oMLX-compatible MLX BF16 fixture becomes available.

The canonical size list follows Qwen's official family release. MLX-community
repositories are practical MLX-converted fixtures, not the canonical source of
the model definition. If a conversion exposes a loader issue, compare its
configuration and conversion provenance with the official checkpoint before
classifying the architecture as unsupported.

Do not claim that all Q4 files are interchangeable: group size, affine
parameters, mixed-bit tensors, shard layout, tokenizer/processor files, and
architecture metadata are part of the compatibility key.

## Validation levels

For every canonical model configuration and weight class, record the following
independently:

1. Static discovery: architecture, Dense/MoE structure, modalities, quantization
   metadata, artifact manifest, and selected provider.
2. Provider compatibility: whether the pinned `mlx-lm` / `mlx-vlm` versions
   expose the required architecture and load path.
3. Runtime conformance when a fixture is available: load, short greedy output,
   text-only boundary, and image path for vision checkpoints.
4. Context/performance evidence: 1K/4K/16K where hardware permits; larger
   contexts and large models receive `hardware-limited` evidence rather than an
   unsupported label when the blocker is memory.
5. Regression record: exact model revision, backend versions, hardware, test
   commands, and failure classification.

All published variants need a clear support state. A missing local fixture is
`not-tested`, not `unsupported`; hardware limits are separate from software
compatibility. Community conversions can extend the matrix when they introduce
a new architecture, quantization layout, shard convention, or processor
contract.

The headless provider-routing test constructs metadata fixtures for all eight
configurations and resolves their text and image requests. It proves that Dense
and MoE sizes share the intended routes; it does not prove that a checkpoint
loads or that every quantization layout works.

## Current completion state

The family is **not yet fully validated**. The current evidence covers only a
subset of model sizes and formats. Static routing is now checked for all eight
canonical configurations. Next work should close runtime conformance and
representative quantization classes. Model acquisition is needed only where a
runtime conformance result cannot be established from upstream implementation
and metadata alone.
