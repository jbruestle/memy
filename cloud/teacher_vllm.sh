#!/usr/bin/env bash
# Start a vLLM teacher server (alternative to teacher.sh / llama.cpp) on one
# GPU. Env: GPU (index, default: last), CTX (max model len, default 8192),
# SEQS (max concurrent sequences, default 64), MEM (fraction of GPU memory,
# default 0.85; lower it when sharing the GPU with a trainer), PORT (8080).
# Needs /workspace/venv-vllm (pip install vllm). Logs to $WS/teacher-vllm.log.
set -euo pipefail
WS=${WORKSPACE:-/workspace}
NGPU=$(nvidia-smi -L | wc -l)
GPU=${GPU:-$((NGPU - 1))}
CTX=${CTX:-8192}
SEQS=${SEQS:-64}
MEM=${MEM:-0.85}
PORT=${PORT:-8080}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
export HF_HOME=$WS/hf HF_HUB_DISABLE_PROGRESS_BARS=1
if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "something already listens on :$PORT"; exit 0
fi
# shellcheck disable=SC1091
source "$WS/venv-vllm/bin/activate"
CUDA_VISIBLE_DEVICES=$GPU nohup vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization "$MEM" \
  --return-tokens-as-token-ids --dtype bfloat16 > "$WS/teacher-vllm.log" 2>&1 < /dev/null &
echo "vllm teacher starting on GPU $GPU (ctx $CTX, $SEQS seqs, mem $MEM), pid $!"
for _ in $(seq 1 120); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "vllm teacher healthy"; exit 0; fi
  sleep 5
done
echo "vllm teacher failed to start; see $WS/teacher-vllm.log"; exit 1
