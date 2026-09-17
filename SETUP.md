# Environment setup (tested: 4090/sm_89; works on 5090/sm_120)

Mirrors the `finetune` conda env this was developed in (python 3.13,
torch 2.10.0+cu128 — arch list includes sm_120, so Blackwell is native).

```bash
conda create -n memy python=3.13 -y
conda activate memy
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.2.0 peft==0.18.1 datasets==4.3.0 \
    accelerate==1.13.0 flash-linear-attention==0.5.2 safetensors
```

Optional GDN fast path (needs local CUDA toolkit >= 12.8 for nvcc; measured
benefit was only ~5% on the 4090, safe to skip — pure-torch fallback is
numerically fine):

```bash
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
    pip install causal-conv1d --no-build-isolation
```

First run downloads Qwen/Qwen3.5-4B (~10GB) and ultrachat_200k (~1.6GB) into
`~/.cache/huggingface` automatically; to skip, rsync those two subdirs of the
cache from the other machine.

## Artifacts NOT in git — sync as needed

- `runs/<name>/ckpt-*/` — checkpoints (adapter + readers.pt), needed for the
  post-hoc suite (`probe_mech.py`).
- `runs/eval-targets.json` — cached teacher eval generations. COPY THIS if you
  want eval_kl numbers comparable across machines/arms; it is created by the
  first arm that evals if absent.

## Verify

```bash
python test_step0.py   # must print "identity check: PASS ... 0.000e+00"
```

## Run

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1 \
    python -u train.py --arm L2 --batch 8 --run-name <name> 2>&1 | tee runs/<name>-console.log
```

Batch 8 peaks ~15.2GB on the 4090; the 5090's 32GB allows batch 12-16
(untested — verify peak with diag_mem.py before committing to a long run).
