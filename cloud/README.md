# Running memy v2 on RunPod

Layout on a pod: one **network volume** at `/workspace` holds everything
that is slow to recreate (venv, Qwen weights, the Q8 GGUF, datasets,
llama.cpp, and `memy/runs/` with checkpoints and caches). Pods are
disposable; the volume is not. On an N-GPU pod the last GPU serves the
teacher (llama.cpp) and the other N−1 train data-parallel via `torchrun`.

## One-time

1. RunPod console → Settings → API key. Locally: `export RUNPOD_API_KEY=...`
   (the `runpod` package is installed in the `finetune` env).
2. Console → Storage → new **network volume**, 150 GB, in a data center
   that lists H100 80GB (and B200 for later). Note its id and data center.
3. Console → Settings → add your SSH public key.
4. Push the repo (private is fine): `git push origin main`. For the pod to
   clone it, make a fine-grained GitHub token with read access to the repo
   and use `REPO_URL=https://<token>@github.com/jbruestle/memy.git`.
   Alternative without a token: `rsync -av --exclude runs --exclude .git
   ~/memy/ root@<pod>:/workspace/memy/`.

## First pod (1×H100): bootstrap + calibration

    python cloud/pod.py create --gpu H100 --count 1 --volume <vol-id> --dc <dc> \
        --repo-url "https://<token>@github.com/jbruestle/memy.git"
    python cloud/pod.py ssh <pod-id>          # prints the ssh command
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
