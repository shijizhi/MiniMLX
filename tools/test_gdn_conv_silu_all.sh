#!/usr/bin/env bash
# One-shot correctness + single-request A/B test for the opt-in GDN conv+SiLU path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${1:-/path/to/Qwen3.8-27B-4bit}"
PROMPT_FILE="${2:-Resources/BenchmarkCorpus/code_python.txt}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  echo "Set PYTHON=/path/to/python or create the project .venv first." >&2
  exit 2
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 2
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

echo "========== 1/4 Numerical correctness: BF16 =========="
MINIMLX_GDN_DECODE_FUSED_CONV_SILU=0 "$PYTHON" tools/verify_gdn_conv_silu.py \
  --channels 10240 --dtype bfloat16 --atol 0.0

echo "========== 2/4 Numerical correctness: FP16 =========="
MINIMLX_GDN_DECODE_FUSED_CONV_SILU=0 "$PYTHON" tools/verify_gdn_conv_silu.py \
  --channels 10240 --dtype float16 --atol 0.0

COMMON_ARGS=(
  "$MODEL_DIR" "$PROMPT_FILE"
  --contexts 1024 4096 --step 1024
  --max-tokens 128 --warmup 1 --repeats 3
  --profile-context 4096 --profile-tokens 8 --ignore-eos
)

echo "========== 3/4 Single-request baseline: fusion OFF =========="
MINIMLX_GDN_DECODE_FUSED_CONV_SILU=0 "$PYTHON" \
  tools/benchmark_qwen_gdn_single_request.py "${COMMON_ARGS[@]}"

echo "========== 4/4 Single-request candidate: fusion ON =========="
MINIMLX_GDN_DECODE_FUSED_CONV_SILU=1 "$PYTHON" \
  tools/benchmark_qwen_gdn_single_request.py "${COMMON_ARGS[@]}"

echo "========== ALL CHECKS COMPLETED =========="
echo "Paste the complete output back. The baseline and candidate each load the model once."
