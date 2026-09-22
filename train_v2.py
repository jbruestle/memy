"""Memy v2 training loop: multi-source, multi-chunk distillation. See
TRAINING_V2.md. Sources and probes are plugins (`sources/`, `probes/`);
this file owns batching, the step, eval, probes, checkpoints, resume, DDP.

Single GPU:
  python train_v2.py --sources ultrachat --batch 8 --run-name L2-v2-ultrachat
Data parallel (one process per GPU, manual grad all-reduce over trainables):
  torchrun --nproc_per_node 4 train_v2.py ...
Resume:  --resume auto   (latest ckpt in the run dir) or --resume <ckpt dir>
"""

import argparse
import ast
import glob
import json
import os
import random
import time

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

import probes as probe_registry
import sources as source_registry
from engine import (Encoder, Teacher, build_model, student_generate, student_loss,
                    teacher_logits)


# ----------------------------------------------------------------- config

def parse_weighted(spec):
    """'ultrachat,wildchat:0.5' -> [(name, weight)] (default weight 1)."""
    out = []
    for item in spec.split(","):
        if not item:
            continue
        name, _, w = item.partition(":")
        out.append((name, float(w) if w else 1.0))
    return out


def parse_cfg(items):
    """['ultrachat.max_user_tokens=320', ...] -> {'ultrachat': {...}}."""
    out = {}
    for it in items or []:
        key, _, val = it.partition("=")
        mod, _, field = key.partition(".")
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
        out.setdefault(mod, {})[field] = val
    return out


def module_cfg(module, overrides, tok):
    return {**module.DEFAULTS, **overrides.get(module.NAME, {}), "tok": tok}


# ----------------------------------------------------------------- sampler

class Sampler:
    """Streams every source, buckets encoded samples by depth, and emits
    batches homogeneous in (source, depth). A bucket is ready at `batch`
    samples or at `token_budget` tokens (if set). State is checkpointable:
    per-source epoch, stream position, queued raw samples, and the mixture
    RNG."""

    def __init__(self, sources, encoder, args, seed, shard):
        self.sources = sources            # [(name, weight, module, cfg)]
        self.encoder, self.args, self.seed, self.shard = encoder, args, seed, shard
        self.rng = random.Random(seed + 1000 * shard[0])
        self.state = {n: {"epoch": 0, "pos": 0, "queue": {}, "done": False}
                      for n, *_ in sources}
        self.encoded = {n: {} for n, *_ in sources}   # depth -> [Encoded] (parallel to queue)
        self.iters = {}
        self.dropped = 0
        self.pending = []     # [(name, [Encoded])] restored from a checkpoint; emitted first

    def state_dict(self, pending=()):
        """pending: [(name, [Encoded])] pulled but not yet trained on (prefetch);
        stored raw so a resume trains on them first, in order."""
        return {"state": self.state, "rng": self.rng.getstate(),
                "pending": [(n, [e.sample for e in encs]) for n, encs in pending]}

    def load_state_dict(self, sd):
        self.state = sd["state"]
        self.rng.setstate(sd["rng"])
        self.pending = [(n, [self.encoder.encode(x) for x in samples])
                        for n, samples in sd.get("pending", [])]
        for name, st in self.state.items():
            self.encoded[name] = {int(d): [self.encoder.encode(s) for s in q]
                                  for d, q in st["queue"].items()}
            st["queue"] = {int(d): q for d, q in st["queue"].items()}

    def _open(self, name):
        _, _, module, cfg = next(s for s in self.sources if s[0] == name)
        st = self.state[name]
        self.iters[name] = module.train(cfg, self.seed, st["epoch"], start=st["pos"],
                                        shard=self.shard)

    def _ready(self, name, force=False):
        st, enc = self.state[name], self.encoded[name]
        for depth in sorted(st["queue"], key=lambda d: -len(st["queue"][d])):
            q, e = st["queue"][depth], enc[depth]
            if not q:
                continue
            tokens = sum(x.tokens for x in e)
            if force or len(q) >= self.args.batch or \
                    (self.args.token_budget and tokens >= self.args.token_budget):
                n, tot = 0, 0
                for x in e:
                    if n >= self.args.batch or \
                            (n and self.args.token_budget and tot + x.tokens > self.args.token_budget):
                        break
                    n += 1; tot += x.tokens
                out = e[:n]
                del q[:n]; del e[:n]
                return out
        return None

    def next_batch(self):
        """Returns (source name, [Encoded]) or None when every source is done."""
        if self.pending:
            return self.pending.pop(0)
        while True:
            live = [(n, w) for n, w, *_ in self.sources if not self.state[n]["done"]]
            if not live:
                return None
            name = self.rng.choices([n for n, _ in live], [w for _, w in live])[0]
            batch = self._ready(name)
            if batch is not None:
                return name, batch
            st = self.state[name]
            if name not in self.iters:
                self._open(name)
            try:
                pos, sample = next(self.iters[name])
            except StopIteration:
                batch = self._ready(name, force=True)   # flush partial buckets
                if batch is not None:
                    return name, batch
                st["epoch"] += 1; st["pos"] = 0
                if self.args.max_epochs and st["epoch"] >= self.args.max_epochs:
                    st["done"] = True
                    continue
                self._open(name)
                continue
            st["pos"] = pos
            enc = self.encoder.encode(sample)
            if enc is None:
                self.dropped += 1
                continue
            st["queue"].setdefault(enc.depth, []).append(sample)
            self.encoded[name].setdefault(enc.depth, []).append(enc)


# ----------------------------------------------------------------- eval / probes

def load_cache(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def import_v1_eval_targets(cache, samples, path="runs/eval-targets.json"):
    """v1 cached its 64 ultrachat eval targets by position; map them by id so
    eval_kl stays comparable with runs/L2-v1/log.jsonl."""
    if not os.path.exists(path) or all(s["id"] in cache for s in samples):
        return
    old = json.load(open(path))
    if len(old) == len(samples):
        for s, g in zip(samples, old):
            cache.setdefault(s["id"], g)


def run_eval(model, ctx, encoder, teacher, sources, args, cache_path):
    cache = load_cache(cache_path)
    out = {}
    for name, _, module, cfg in sources:
        samples = module.eval(cfg)
        if name == "ultrachat":
            import_v1_eval_targets(cache, samples)
        encs = [e for e in (encoder.encode(s) for s in samples) if e is not None]
        by_depth = {}
        for e in encs:
            by_depth.setdefault(e.depth, []).append(e)
        tot, n_tot = 0.0, 0
        for depth, group in by_depth.items():
            for j in range(0, len(group), args.batch):
                chunk = group[j:j + args.batch]
                missing = [e for e in chunk if e.sample["id"] not in cache]
                if missing:
                    gens = teacher.generate([e.prompt for e in missing])
                    for e, g in zip(missing, gens):
                        cache[e.sample["id"]] = g
                    json.dump(cache, open(cache_path, "w"))
                gens = [cache[e.sample["id"]] for e in chunk]
                t_logits = teacher_logits(model, ctx, [e.prompt for e in chunk], gens,
                                          encoder.pad_id)
                model.eval()
                loss, n_tok, _, _ = student_loss(model, ctx, encoder, chunk, gens, t_logits,
                                                 args, want_stats=False, grad_ok=False)
                model.train()
                tot += float(loss) * n_tok; n_tot += n_tok
        out[name] = tot / max(n_tot, 1)
    return out


def run_probes(model, ctx, encoder, teacher, probes, args, step, outdir, cache_path):
    cache = load_cache(cache_path)
    logs = {}
    for module, cfg in probes:
        interval = cfg.get("interval") or args.eval_interval
        if step % interval:
            continue
        samples = module.samples(cfg, step)
        encs = [encoder.encode(s) for s in samples]
        results = []
        for j in range(0, len(encs), args.batch):
            chunk = encs[j:j + args.batch]
            missing = [e for e in chunk if e.sample["id"] not in cache]
            if missing:
                gens = teacher.generate([e.prompt for e in missing], greedy=True)
                for e, g in zip(missing, gens):
                    cache[e.sample["id"]] = encoder.tok.decode(g, skip_special_tokens=True)
                json.dump(cache, open(cache_path, "w"))
            gens, ent = student_generate(model, ctx, encoder, chunk, args, args.max_gen)
            for e, g in zip(chunk, gens):
                results.append({"sample": e.sample,
                                "student": encoder.tok.decode(g, skip_special_tokens=True),
                                "teacher": cache[e.sample["id"]], "read_entropy": ent})
        blob = module.score(cfg, results)
        logs[module.NAME] = blob
        with open(os.path.join(outdir, f"probe-{module.NAME}-{step}.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    return logs


# ----------------------------------------------------------------- checkpoints

def save_ckpt(outdir, step, model, opt, sched, sampler, args, rank, pending=()):
    ckdir = os.path.join(outdir, f"ckpt-{step}")
    os.makedirs(ckdir, exist_ok=True)
    if rank == 0:
        if not args.no_lora:
            model.save_pretrained(ckdir)               # adapter (probe_mech / chat compat)
        torch.save({k: v for k, v in model.state_dict().items() if "mem_" in k},
                   os.path.join(ckdir, "readers.pt"))
        torch.save({"trainable": {n: p.detach().cpu() for n, p in model.named_parameters()
                                  if p.requires_grad},
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "args": vars(args)}, os.path.join(ckdir, "state.pt"))
    torch.save({"sampler": sampler.state_dict(pending),
                "rng": {"py": random.getstate(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state()}},
               os.path.join(ckdir, f"sampler-rank{rank}.pt"))
    return ckdir


def load_ckpt(ckdir, model, opt, sched, sampler, rank):
    st = torch.load(os.path.join(ckdir, "state.pt"), map_location="cpu", weights_only=False)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in st["trainable"].items():
            params[n].copy_(v.to(params[n].dtype))
    opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    rs = torch.load(os.path.join(ckdir, f"sampler-rank{rank}.pt"), map_location="cpu",
                    weights_only=False)
    sampler.load_state_dict(rs["sampler"])
    random.setstate(rs["rng"]["py"]); torch.set_rng_state(rs["rng"]["torch"])
    torch.cuda.set_rng_state(rs["rng"]["cuda"])
    return st["step"]


def latest_ckpt(outdir):
    cks = glob.glob(os.path.join(outdir, "ckpt-*"))
    cks = [c for c in cks if os.path.exists(os.path.join(c, "state.pt"))]
    return max(cks, key=lambda c: int(c.rsplit("-", 1)[1])) if cks else None


# ----------------------------------------------------------------- DDP helpers

def allreduce_grads(trainable, world):
    grads = [p.grad for p in trainable if p.grad is not None]
    if not grads:
        return
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat /= world
    for g, s in zip(grads, _unflatten_dense_tensors(flat, grads)):
        g.copy_(s)


def sync_flag(value, world, device):
    if world == 1:
        return value
    t = torch.tensor([int(value)], device=device)
    dist.all_reduce(t)
    return bool(t.item())


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="ultrachat", help="name[:weight],...")
    ap.add_argument("--probes", default="bindings", help="name,... ('' = none)")
    ap.add_argument("--cfg", action="append", help="module.field=value (sources and probes)")
    ap.add_argument("--turn-tags", action="store_true")
    ap.add_argument("--batch", type=int, default=8, help="max samples per batch")
    ap.add_argument("--token-budget", type=int, default=0, help="max chunk tokens per batch (0 = off)")
    ap.add_argument("--bg-tokens", type=int, default=16384, help="background micro-batch padded tokens")
    ap.add_argument("--max-gen", type=int, default=300, help="per-turn teacher generation cap")
    ap.add_argument("--teacher-url", default=None,
                    help="llama.cpp server for teacher generation; default: inline HF generate")
    ap.add_argument("--teacher-prefetch", type=int, default=1,
                    help="batches requested ahead of training (remote teacher only)")
    ap.add_argument("--max-chunk-tokens", type=int, default=0, help="drop samples with a longer chunk (0 = off)")
    ap.add_argument("--max-prompt-tokens", type=int, default=0, help="drop samples with a longer teacher transcript")
    ap.add_argument("--distractor-grad-k", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--read-heads", type=int, default=4)
    ap.add_argument("--read-rank", type=int, default=128)
    ap.add_argument("--write-layer", type=int, default=19)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.set_defaults(grad_ckpt=True)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = until sources are exhausted")
    ap.add_argument("--max-epochs", type=int, default=1, help="per source; 0 = unlimited")
    ap.add_argument("--detach-writes", action="store_true")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None, help="'auto' or a ckpt dir")
    args = ap.parse_args()
    if args.no_lora:
        args.detach_writes = True

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        # Long timeout: rank 0 runs eval + probes while the others wait at the barrier.
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(hours=3))
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cuda")
    torch.manual_seed(args.seed + rank); random.seed(args.seed + rank)

    run = args.run_name or f"v2-{time.strftime('%m%d-%H%M')}"
    outdir = os.path.join("runs", run)
    os.makedirs(outdir, exist_ok=True)
    logf = open(os.path.join(outdir, "log.jsonl"), "a") if rank == 0 else None

    def log(rec):
        if rank == 0:
            print(rec, flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()

    tok, model, ctx = build_model(args)
    encoder = Encoder(tok, args.turn_tags, args.max_chunk_tokens, args.max_prompt_tokens)
    overrides = parse_cfg(args.cfg)
    sources = []
    for name, w in parse_weighted(args.sources):
        m = source_registry.get(name)
        sources.append((name, w, m, module_cfg(m, overrides, tok)))
    probes = []
    for name, _ in parse_weighted(args.probes):
        m = probe_registry.get(name)
        probes.append((m, module_cfg(m, overrides, tok)))

    trainable = [p for p in model.parameters() if p.requires_grad]
    log({"config": vars(args), "trainable_M": sum(p.numel() for p in trainable) / 1e6,
         "world": world})
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)))
    sampler = Sampler(sources, encoder, args, args.seed, (rank, world))
    step = 0
    if args.resume:
        ck = latest_ckpt(outdir) if args.resume == "auto" else args.resume
        if ck:
            step = load_ckpt(ck, model, opt, sched, sampler, rank)
            log({"resumed": ck, "step": step})
    # Teacher targets depend on the generation cap; keep caches per cap.
    eval_cache = os.path.join("runs", f"eval-targets-v2-g{args.max_gen}.json")
    probe_cache = os.path.join("runs", f"probe-teacher-v2-g{args.max_gen}.json")
    teacher = Teacher(model, ctx, encoder, args)

    def pull():
        nb = sampler.next_batch()
        return (nb[0], nb[1], teacher.submit([e.prompt for e in nb[1]])) if nb else None

    depth = max(args.teacher_prefetch, 1) if args.teacher_url else 1
    pending, exhausted = [], False   # remote: generations run while earlier batches train

    def fill():
        nonlocal exhausted
        while not exhausted and len(pending) < depth:
            nb = pull()
            if nb is None:
                exhausted = True
            else:
                pending.append(nb)

    fill()
    while True:
        if args.max_steps and step >= args.max_steps:
            break
        if sync_flag(not pending, world, device):
            log({"exhausted": True, "step": step})
            break
        name, encs, fut = pending.pop(0)
        t0 = time.time()
        gens = fut.result()
        t_wait = time.time() - t0
        fill()
        t_logits = teacher_logits(model, ctx, [e.prompt for e in encs], gens, encoder.pad_id)
        t1 = time.time()
        loss, n_tok, stats, info = student_loss(model, ctx, encoder, encs, gens, t_logits,
                                                args, want_stats=(step % 50 == 0))
        loss.backward()
        if world > 1:
            allreduce_grads(trainable, world)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        rec = {"step": step, "src": name, "kl": float(loss.detach()), "tokens": n_tok,
               "n": len(encs), "chunk_tokens": sum(e.tokens for e in encs), **info,
               "sec": round(time.time() - t0, 2), "teacher_sec": round(t1 - t0, 2),
               "teacher_wait": round(t_wait, 2),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "teacher_local": teacher.n_local}
        if stats:
            rec["read_entropy"] = round(sum(s["entropy"] for s in stats) / len(stats), 3)
        log(rec)
        del loss, t_logits

        if step % args.eval_interval == 0:
            if rank == 0:
                ev = run_eval(model, ctx, encoder, teacher, sources, args, eval_cache)
                pr = run_probes(model, ctx, encoder, teacher, probes, args, step, outdir,
                                probe_cache)
                rec = {"step": step, "eval_kl": ev, "probe": pr, "dropped": sampler.dropped}
                print({**rec, "probe": {k: {kk: vv for kk, vv in v.items() if kk != "sample"}
                                        for k, v in pr.items()}}, flush=True)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
            if world > 1:
                dist.barrier()
        if step % args.save_interval == 0:
            save_ckpt(outdir, step, model, opt, sched, sampler, args, rank,
                      pending=[(n, e) for n, e, _ in pending])
            if world > 1:
                dist.barrier()
    if step % args.save_interval:
        save_ckpt(outdir, step, model, opt, sched, sampler, args, rank,
                  pending=[(n, e) for n, e, _ in pending])
    log({"done": True, "step": step})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
