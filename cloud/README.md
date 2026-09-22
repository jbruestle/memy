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

On a 1-GPU pod the teacher shares the GPU (~7 GB). `teacher_wait` near 0
means the teacher keeps up; `peak_gb` tells whether batch 8 fits.
`python cloud/pod.py terminate <pod-id>` when done (billing stops; the
volume keeps everything).

## Real run (8×H100)

    python cloud/pod.py create --gpu H100 --count 8 --volume <vol-id> --dc <dc> --name memy-8
    # on the pod (bootstrap is a no-op now except git pull + test):
    bash /workspace/memy/cloud/bootstrap.sh
    cd /workspace/memy && bash cloud/launch.sh --run-name v2-main --sources ... --batch 8 ... --max-steps 25000

`launch.sh` starts the teacher on GPU 7 with 32 slots × 8k context, then
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
