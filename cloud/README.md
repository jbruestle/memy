# Running memy v2 on RunPod

Layout on a pod: one **network volume** at `/workspace` holds everything
that is slow to recreate (venv, Qwen weights, the Q8 GGUF, datasets,
llama.cpp, and `memy/runs/` with checkpoints and caches). Pods are
disposable; the volume is not. On an N-GPU pod the last GPU serves the
teacher (llama.cpp) and the other N−1 train data-parallel via `torchrun`.

## State (2026-09-22)

- API key: `source ~/.config/memy/runpod.env` on the Seattle box (not in git).
- Network volume **`uwl0aoa2aa`**, 150 GB, **US-NE-1** (chosen 2026-09-22:
  the only North-American data center with standard volumes and H100 SXM
  stock in two availability snapshots; stock is volatile, check
  `python cloud/pod.py dcs` before creating a pod; no volume-capable DC
  showed B200 that day — the 27B run may need a second volume elsewhere).
- Jeremy's SSH key is registered in RunPod and GitHub; the repo is pushed.
- Code onto the pod: the pod has no GitHub credentials, so rsync from the
  Seattle box (`rsync -av --exclude runs --exclude .git --exclude __pycache__
  ~/memy/ root@<ip>:/workspace/memy/ -e "ssh -p <port>"`), then run
  `bootstrap.sh` without `REPO_URL`. A read token in `REPO_URL` works too.
- `pod.py create` defaults: secure cloud, `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
  (verify the tag still exists), 50 GB container disk, ports 22 + 8080.
- 2026-09-22: first pod (`memy-calib`, 1×H100 SXM, US-NE-1, $3.49/h) bootstrapped;
  the volume now holds venv (+ causal-conv1d, triton 3.7.1), `venv-vllm`
  (vllm 0.30.0, own torch), Qwen weights, the Q8 GGUF, llama.cpp, all six
  datasets, and `runs/calib-b8`, `calib-b16` (100 steps each). Gotchas hit:
  the Seattle box has no rsync (use `tar cz | ssh tar xz`); fla 0.5.2 refuses
  triton 3.4–3.7.0 on Hopper (bootstrap pins 3.7.1, after causal-conv1d which
  drags it back to 3.6.0); `pkill -f llama-server` from an ssh one-liner kills
  the ssh session itself.
- **Teacher backend: vLLM** (`cloud/teacher_vllm.sh`, `TEACHER=vllm` default in
  `launch.sh`; engine auto-detects it via `/v1/models`). Measured on an idle
  H100 with 1.5k-token prompts × 512 generated (`cloud/teacher_bench.py`):
  llama.cpp Q8 547 tok/s at 32 streams (42 ms/token/stream — the hybrid GDN
  layers batch poorly there); vLLM bf16 4,030 tok/s at 32 and 6,100 at 64.
  One batch-8 trainer consumes ~250 generated tok/s, so 7 trainers need
  ~1,750: llama.cpp would have been a 3× bottleneck, vLLM is not.
  vLLM takes ~10 min to start the first time (compile + CUDA graphs).
  `runs/smoke-vllm` (20 steps, batch 8, vLLM at MEM=0.3 on the same GPU):
  teacher wait 1% of wall-clock (was 42% with llama.cpp), wildchat steps
  7.9 s vs 30 s, eval + probes clean, teacher probe ceilings unchanged
  (bindings 1.0, copy EM 0.84) — token-id round trip verified.

## One-time (done)

1. API key (console → Settings) — stored as above.
2. Network volume — created via `python cloud/pod.py volume create --dc US-NE-1 --size 150`.
3. SSH public key in the console.
4. Repo pushed.

## First pod (1×H100): bootstrap + calibration

    source ~/.config/memy/runpod.env
    python cloud/pod.py create --gpu "NVIDIA H100 80GB HBM3" --count 1 --volume uwl0aoa2aa --dc US-NE-1
    python cloud/pod.py ssh <pod-id>          # prints the ssh command (wait until the pod shows a public port)
    rsync -av --exclude runs --exclude .git --exclude __pycache__ ~/memy/ root@<ip>:/workspace/memy/ -e "ssh -p <port>"
    # on the pod:
    bash /workspace/memy/cloud/bootstrap.sh    # ~15 min first time: env, llama.cpp build, ~20 GB downloads, engine test
    cd /workspace/memy && bash cloud/launch.sh --run-name calib --sources wildchat:0.4,musique:0.2,triviaqa:0.2,copy:0.2 \
        --batch 8 --max-chunk-tokens 8192 --turn-tags --probes bindings,copy --max-steps 100 --eval-interval 100
    tail -f runs/calib-console.log             # sec, teacher_wait, peak_gb per step

On a 1-GPU pod the teacher shares the GPU (llama.cpp ~15 GB at 32×8k
slots; vLLM takes `MEM`×80 GB, 0.3 by default there). `teacher_wait` near 0
means the teacher keeps up; `peak_gb` tells whether the batch fits.

Calibration results (2026-09-22, mixture wildchat:0.5,copy:0.25,musique:0.15,
triviaqa:0.1, `--token-budget 32768 --max-gen 1024`, llama.cpp teacher on
the same GPU; tok/s and samples/s exclude teacher wait, which was ~42% of
wall-clock in both runs):

| source | b8 tok/s | b16 tok/s | b8 samples/s | b16 samples/s | b16 n/batch |
|---|---|---|---|---|---|
| wildchat | 475 | 382 | 0.60 | 0.47 | 16 |
| copy | 1160 | 1650 | 1.20 | 1.71 | 16 |
| musique | 1820 | 2460 | 1.24 | 1.69 | 16 |
| triviaqa | 2800 | 4210 | 0.22 | 0.34 | 2 |

Peak 38.7 GB (b8) / 60.8 GB (b16); batch 32 OOMs on the shared GPU (63 GB
+ teacher; the failing 7.6 GB allocation is the full-vocab logits of a
32-sample target chunk). Overall 0.64 vs 0.65 samples/s: the QA sources gain
~1.4× from batch 16 but wildchat, which dominates, gets slower — its student
time is superlinear in batch (depth-3 batches 9.9 s → 33.6 s), presumably
padding across samples in the turn passes / bank plus allocator pressure
near 73 GB. Batch 8 for chat; a per-source batch cap would let QA use 16.
Note the sampler weights act per *sample* (a not-ready source pulls one
sample and redraws), so a 2-sample triviaqa batch gets 4× the batches per
unit weight that an 8-sample source does.
`python cloud/pod.py terminate <pod-id>` when done (billing stops; the
volume keeps everything).

## Real run (8×H100)

    python cloud/pod.py create --gpu H100 --count 8 --volume <vol-id> --dc <dc> --name memy-8
    # on the pod (bootstrap is a no-op now except git pull + test):
    bash /workspace/memy/cloud/bootstrap.sh
    cd /workspace/memy && bash cloud/launch.sh --run-name v2-topk8 \
        --sources wildchat:0.5,copy:0.25,musique:0.15,triviaqa:0.1 --batch 8 --token-budget 32768 \
        --max-chunk-tokens 8192 --max-gen 1024 --turn-tags --probes bindings,copy \
        --eval-interval 250 --max-steps 12000 --read-top-k 8

`--read-top-k 8` from step 0: on ultrachat it trained ~3× faster than the
full softmax and plateaued higher (DESIGN log 2026-09-23); the training
read then goes through `MemoryReader._gathered` (K winners gathered, dense
softmax over them), which is also the deployment read.

`launch.sh` starts the teacher on GPU 7 (vLLM, 64 seqs × 8k context), then
`torchrun --nproc_per_node 7`. Each rank logs its own steps; rank 0 writes
`runs/<name>/log.jsonl`, evals, probes and checkpoints. A relaunch with the
same `--run-name` resumes from the last checkpoint (`--resume auto`).
Stop with `pkill -f train_v2.py`; the teacher can stay up.

## Costs (RunPod on-demand, Sept 2026)

H100 80GB ≈ $2.0–2.7/h, B200 ≈ $5.9/h, network volume ≈ $0.07/GB/month.
The 4B run at 200k samples is ~15–20 H100-hours ≈ $50; on 8 GPUs ≈ 3 h.
Terminate pods when idle — a stopped pod still bills its container disk.

## Notes

- `DEFAULT_IMAGE` in `pod.py` is a RunPod PyTorch devel image (needs nvcc
  for the llama.cpp build); verify the tag exists in the console's
  template list and adjust if RunPod has rotated it.
- The venv lives on the volume and is tied to the image's Python; if the
  image changes, delete `/workspace/venv` and re-run bootstrap.
- Multi-GPU has not been exercised before the first 8-GPU pod: watch the
  first steps of all ranks (`runs/<name>-console.log` interleaves them)
  and the `exhausted`/barrier behaviour at the end.
