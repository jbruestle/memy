#!/usr/bin/env bash
# One-time setup of a RunPod pod for memy v2. Everything lands on the network
# volume ($WS, mounted at /workspace) so the next pod on the same volume
# skips it. Idempotent: re-running only fills in what is missing.
#
#   REPO_URL=https://<token>@github.com/jbruestle/memy.git bash bootstrap.sh
# or rsync the repo to $WS/memy first and run without REPO_URL.
set -euo pipefail
WS=${WORKSPACE:-/workspace}
export HF_HOME=$WS/hf
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$HF_HOME" "$WS/gguf"

echo "== code"
if [ ! -d "$WS/memy/.git" ]; then
  git clone "${REPO_URL:?set REPO_URL or rsync the repo to $WS/memy}" "$WS/memy"
else
  (cd "$WS/memy" && git pull --ff-only || true)
fi

echo "== python env ($WS/venv)"
if [ ! -x "$WS/venv/bin/python" ]; then
  python3 -m venv "$WS/venv"
fi
# shellcheck disable=SC1091
source "$WS/venv/bin/activate"
pip install -q --upgrade pip
pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -q transformers==5.2.0 peft==0.18.1 datasets==4.3.0 accelerate==1.13.0 \
    flash-linear-attention==0.5.2 safetensors huggingface_hub runpod

echo "== llama.cpp (teacher server)"
if [ ! -x "$WS/llama.cpp/build/bin/llama-server" ]; then
  apt-get update -qq && apt-get install -y -qq cmake build-essential libcurl4-openssl-dev > /dev/null
  [ -d "$WS/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$WS/llama.cpp"
  cmake -S "$WS/llama.cpp" -B "$WS/llama.cpp/build" -DGGML_CUDA=ON -DLLAMA_CURL=ON \
        -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build "$WS/llama.cpp/build" --target llama-server -j "$(nproc)" > /dev/null
fi

echo "== model + teacher weights"
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
import os
snapshot_download("Qwen/Qwen3.5-4B")
hf_hub_download("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q8_0.gguf",
                local_dir=os.path.join(os.environ["WORKSPACE"] if "WORKSPACE" in os.environ else "/workspace", "gguf"))
PY

echo "== datasets (one sample per source forces the download)"
cd "$WS/memy"
for src in ultrachat wildchat musique qasper triviaqa copy; do
  CUDA_VISIBLE_DEVICES="" python peek_source.py "$src" --n 1 > /dev/null 2>&1 && echo "  $src ok" || echo "  $src FAILED"
done

echo "== engine test (GPU)"
PYTORCH_ALLOC_CONF=expandable_segments:True python test_engine.py 2>&1 | grep -E "loss|ok|PASS|Error|assert" || true
echo "bootstrap done"
