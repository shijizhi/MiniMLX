#!/bin/zsh
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi

PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
.venv/bin/python -m pip install --index-url "$PIP_INDEX_URL" --upgrade pip
.venv/bin/python -m pip install --index-url "$PIP_INDEX_URL" -r requirements-runtime.txt
echo "MiniMLX runtime environment ready: $ROOT/.venv"
