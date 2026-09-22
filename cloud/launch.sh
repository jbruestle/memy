#!/usr/bin/env bash
# Launch a training run on a pod: teacher server on the last GPU, torchrun
# data-parallel trainers on the rest (all GPUs share one on a 1-GPU pod).
# All train_v2.py flags pass through; --resume auto is added so a relaunch
# on the same run name continues from the last checkpoint.
#
#   bash cloud/launch.sh --run-name v2-main --sources wildchat:0.4,musique:0.2,triviaqa:0.2,copy:0.2 \
#        --batch 8 --max-chunk-tokens 8192 --turn-tags --probes bindings,copy --max-steps 25000
set -euo pipefail
WS=${WORKSPACE:-/workspace}
cd "$WS/memy"
# shellcheck disable=SC1091
source "$WS/venv/bin/activate"
export HF_HOME=$WS/hf HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_ALLOC_CONF=expandable_segments:True
NGPU=$(nvidia-smi -L | wc -l)
PORT=${PORT:-8080}
bash cloud/teacher.sh
if [ "$NGPU" -gt 1 ]; then
  NTRAIN=$((NGPU - 1))
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NTRAIN - 1)))
else
  NTRAIN=1
fi
RUN=$(printf '%s\n' "$@" | grep -A1 -x -- '--run-name' | tail -1 || echo run)
echo "trainers: $NTRAIN GPU(s) [$CUDA_VISIBLE_DEVICES]; teacher :$PORT; log runs/$RUN-console.log"
mkdir -p runs
nohup torchrun --standalone --nproc_per_node "$NTRAIN" train_v2.py "$@" \
  --teacher-url "http://127.0.0.1:$PORT" --teacher-prefetch "${PREFETCH:-4}" --resume auto \
  > "runs/$RUN-console.log" 2>&1 &
echo "torchrun pid $!  (tail -f runs/$RUN-console.log)"
