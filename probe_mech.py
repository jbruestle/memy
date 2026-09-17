"""Post-hoc mechanism battery for an L2 checkpoint. See DESIGN.md.

Runs the held-out probes under memory transforms that bound WHAT the reads
consume (no training involved):

  full      - untouched per-token bank (reference)
  meanpool  - bank replaced by its single mean vector (catches diffuse gist)
  lasttoken - bank replaced by final token's memory (catches EOT-summary gist)
  rank<k>   - bank replaced by rank-k SVD approximation (effective dimension)
  del-<key> - memories at that binding's token positions zeroed out
              (binding-specific recall drop => token-targeted retrieval)

Usage (needs the GPU free — do not run alongside training):
  python probe_mech.py --ckpt runs/L2-v1/ckpt-1000
  python probe_mech.py --ckpt runs/L2-v1/ckpt-1000 --modes full,meanpool,rank4
  python probe_mech.py --ckpt runs/L2-v1/ckpt-1000 --readmap 0   # dump maps for probe 0

Results: printed table + <ckpt>/mech-<mode>.jsonl generation dumps.
"""

import argparse
import json
import os
from types import SimpleNamespace

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from data import make_probes, score_probe
from model import install_memory
from train import MODEL, encode, pad_batch

BINDING_KEYS = ["name", "relname", "amount", "year", "item"]


def load_checkpoint(ckpt, args):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = True
    ctx, _ = install_memory(model, write_layer=args.write_layer,
                            n_heads=args.read_heads, rank=args.read_rank)
    model = PeftModel.from_pretrained(model, ckpt, is_trainable=False)
    readers = torch.load(os.path.join(ckpt, "readers.pt"),
                         map_location="cuda", weights_only=True)
    missing, unexpected = model.load_state_dict(readers, strict=False)
    assert not unexpected, f"unmatched reader keys: {unexpected[:3]}"
    n_loaded = sum("mem_" in k for k in readers)
    print(f"loaded adapter + {n_loaded} reader tensors from {ckpt}")
    model.eval()
    return tok, model, ctx


def binding_spans(tok, question, bindings):
    """Token index spans (within the templated user chunk) for each binding."""
    msgs = [{"role": "user", "content": question}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    spans = {}
    for key in BINDING_KEYS:
        val = bindings[key]
        c0 = rendered.find(val)
        if c0 < 0:
            continue
        c1 = c0 + len(val)
        spans[key] = [i for i, (a, b) in enumerate(enc["offset_mapping"])
                      if a < c1 and b > c0]
    return enc["input_ids"], spans


def transform_memory(mem, mask, mode, spans=None):
    """mem (1,N,H) fp against a single sample; returns transformed (mem, mask)."""
    n = int(mask.sum())
    valid = mem[:, :n]
    if mode == "full":
        return mem, mask
    if mode == "meanpool":
        return valid.mean(1, keepdim=True), torch.ones(1, 1, dtype=mask.dtype, device=mask.device)
    if mode == "lasttoken":
        return valid[:, -1:], torch.ones(1, 1, dtype=mask.dtype, device=mask.device)
    if mode.startswith("rank"):
        k = int(mode[4:])
        u, s, vh = torch.linalg.svd(valid[0].float(), full_matrices=False)
        approx = (u[:, :k] * s[:k]) @ vh[:k]
        out = mem.clone()
        out[:, :n] = approx.to(mem.dtype)
        return out, mask
    if mode.startswith("del-"):
        key = mode[4:]
        out_mask = mask.clone()
        for i in spans.get(key, []):
            out_mask[0, i] = 0
        return mem, out_mask
    raise ValueError(mode)


@torch.no_grad()
def run_mode(model, tok, ctx, args, probes, mode):
    pad_id = tok.pad_token_id
    device = next(model.parameters()).device
    per_key, n, gens = {}, 0, []
    for p in probes:  # batch 1: transforms are per-sample anyway
        user_ids, open_ids = encode(tok, [p["question"]])
        _, spans = binding_spans(tok, p["question"], p["bindings"])
        uids, umask = pad_batch(user_ids, pad_id, "right", device)
        ctx.clear()
        ctx.collect_writes, ctx.reads_enabled = True, False
        model(input_ids=uids, attention_mask=umask, use_cache=False)
        ctx.collect_writes = False
        mem, mmask = transform_memory(ctx.written, umask, mode, spans)
        ctx.memory, ctx.memory_mask = mem, mmask
        ctx.reads_enabled = True
        oids, omask = pad_batch([open_ids], pad_id, "left", device)
        out = model.generate(input_ids=oids, attention_mask=omask,
                             max_new_tokens=args.max_gen, do_sample=False,
                             use_cache=True, pad_token_id=pad_id)
        txt = tok.decode(out[0, oids.shape[1]:], skip_special_tokens=True)
        hits = score_probe(txt, p["bindings"])
        for k, h in hits.items():
            per_key[k] = per_key.get(k, 0) + int(h)
        gens.append({"bindings": p["bindings"], "hits": hits, "text": txt})
        n += 1
        ctx.clear()
    scores = {k: v / n for k, v in per_key.items()}
    scores["recall"] = sum(scores.values()) / len(scores)
    return scores, gens


@torch.no_grad()
def read_map_report(model, tok, ctx, args, probe):
    """Teacher-force the model's own full-memory generation and report, for each
    binding, the max softmax mass any head puts on that binding's memories."""
    pad_id = tok.pad_token_id
    device = next(model.parameters()).device
    user_ids, open_ids = encode(tok, [probe["question"]])
    _, spans = binding_spans(tok, probe["question"], probe["bindings"])
    uids, umask = pad_batch(user_ids, pad_id, "right", device)
    ctx.clear()
    ctx.collect_writes, ctx.reads_enabled = True, False
    model(input_ids=uids, attention_mask=umask, use_cache=False)
    ctx.collect_writes = False
    ctx.memory, ctx.memory_mask = ctx.written, umask
    ctx.reads_enabled = True
    oids, omask = pad_batch([open_ids], pad_id, "left", device)
    out = model.generate(input_ids=oids, attention_mask=omask,
                         max_new_tokens=args.max_gen, do_sample=False,
                         use_cache=True, pad_token_id=pad_id)
    full = out[:, oids.shape[1]:]
    ctx.maps = []
    ctx.capture_maps = True
    model(input_ids=torch.cat([oids, full], 1),
          attention_mask=torch.ones_like(torch.cat([oids, full], 1)),
          use_cache=False)
    ctx.capture_maps = False
    print("generation:", tok.decode(full[0], skip_special_tokens=True)[:300])
    n_open = oids.shape[1]
    for key, span in spans.items():
        best = 0.0, None
        for m in ctx.maps:
            # mass on this binding's memories, max over heads and answer positions
            mass = m["p"][0, :, n_open:, span].sum(-1).max()
            if float(mass) > best[0]:
                best = float(mass), m["layer"]
        print(f"  binding {key:8s} ({probe['bindings'][key]}): "
              f"max head mass {best[0]:.3f} at layer {best[1]}")
    ctx.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--modes", default="full,meanpool,lasttoken,rank1,rank2,rank4,"
                    "rank8,rank16,del-name,del-relname,del-amount,del-year,del-item")
    ap.add_argument("--n-probes", type=int, default=32)
    ap.add_argument("--max-gen", type=int, default=300)
    ap.add_argument("--readmap", type=int, default=-1, help="probe idx for map dump")
    ap.add_argument("--read-heads", type=int, default=4)
    ap.add_argument("--read-rank", type=int, default=128)
    ap.add_argument("--write-layer", type=int, default=19)
    args = ap.parse_args()

    tok, model, ctx = load_checkpoint(args.ckpt, args)
    probes = make_probes(n=args.n_probes, path=os.path.join(args.ckpt, "mech-probes.jsonl"))

    if args.readmap >= 0:
        read_map_report(model, tok, ctx, args, probes[args.readmap])
        return

    results = {}
    for mode in args.modes.split(","):
        scores, gens = run_mode(model, tok, ctx, args, probes, mode)
        results[mode] = scores
        with open(os.path.join(args.ckpt, f"mech-{mode}.jsonl"), "w") as f:
            for g in gens:
                f.write(json.dumps(g) + "\n")
        print(f"{mode:12s} " + " ".join(f"{k}={scores.get(k, 0):.2f}"
                                        for k in BINDING_KEYS + ["recall"]), flush=True)
    with open(os.path.join(args.ckpt, "mech-summary.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
