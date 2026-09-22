#!/usr/bin/env bash
# Start the llama.cpp teacher server on one GPU. Env: GPU (index, default:
# last), SLOTS (parallel sequences, default 32), CTX (tokens per slot,
# default 8192), PORT (8080). Logs to $WS/teacher.log; prints when healthy.
set -euo pipefail
WS=${WORKSPACE:-/workspace}
NGPU=$(nvidia-smi -L | wc -l)
GPU=${GPU:-$((NGPU - 1))}
SLOTS=${SLOTS:-32}
CTX=${CTX:-8192}
PORT=${PORT:-8080}
MODEL=${MODEL:-$WS/gguf/Qwen3.5-4B-Q8_0.gguf}
if curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok; then
  echo "teacher already up on :$PORT"; exit 0
fi
CUDA_VISIBLE_DEVICES=$GPU nohup "$WS/llama.cpp/build/bin/llama-server" -m "$MODEL" \
  --host 127.0.0.1 --port "$PORT" -ngl 99 -fa on -c $((SLOTS * CTX)) -np "$SLOTS" \
  -b 4096 -ub 1024 --no-mmproj > "$WS/teacher.log" 2>&1 < /dev/null &
echo "teacher starting on GPU $GPU ($SLOTS slots x $CTX ctx), pid $!"
for _ in $(seq 1 120); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok; then echo "teacher healthy"; exit 0; fi
  sleep 5
done
echo "teacher failed to start; see $WS/teacher.log"; exit 1
