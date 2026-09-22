"""Round 2: think-block, chunked ingestion, and mission-capture rescue.

  think   - scale-one targeted query, bare vs instructed recall-dump-first
            ("think block" without training). Scores the name->age pairing in
            the output and the final stated number separately.
  chunked - probe late in ~1300 tokens of filler: one long chunk vs split
            into ~300-token separately-ingested chunks (positions from 0).
  rescue  - probe + 8 distractor turns: probe-first (known 0.00), probe-last,
            probe-first + top-K 4, and a final redirect turn referencing the
            probe by name vs by topic.

Usage: python explore/explore2.py --ckpt runs/L2-v1/ckpt-14000
"""

import argparse
import json
import random
import re

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for data/train/model imports
from data import make_probes, score_probe, user_questions
from explore import NAMES, filler_text, generate, ingest, user_chunk_ids
from probe_mech import load_checkpoint
from train import encode

os.makedirs("explore_out", exist_ok=True)
OUT = open("explore_out/round2.jsonl", "a", buffering=1)


def exp_think(model, tok, ctx, device, open_ids, args):
    rng = random.Random(33)
    print("=== think: recall-dump-before-answer vs bare ===", flush=True)
    for M in [8, 16]:
        tally = {"bare": [0, 0], "think": [0, 0]}  # [pair-hit, final-number-hit]
        total = 0
        for trial in range(2):
            names = rng.sample(NAMES, M)
            ages = rng.sample(range(18, 98), M)
            data = " ".join(f"{n} is {a} years old." for n, a in zip(names, ages))
            for qi in [0, M // 2, M - 1]:
                total += 1
                asks = {"bare": "Please answer with just the number.",
                        "think": ("Before answering, first list every person "
                                  "mentioned and their age, then clearly state "
                                  "the final answer.")}
                for style, suffix in asks.items():
                    q = f"{data} How old is {names[qi]}? {suffix}"
                    bank = [ingest(model, tok, ctx, user_chunk_ids(tok, q), device)]
                    txt, _ = generate(model, tok, ctx, bank, open_ids, device,
                                      max_gen=(60 if style == "bare"
                                               else 100 + 14 * M))
                    pair = bool(re.search(
                        rf"\b{names[qi]}\b\D{{0,24}}\b{ages[qi]}\b", txt))
                    nums = re.findall(r"\b\d+\b", txt)
                    final = bool(nums) and nums[-1] == str(ages[qi])
                    tally[style][0] += int(pair)
                    tally[style][1] += int(final)
                    OUT.write(json.dumps({"exp": "think", "M": M, "style": style,
                                          "qi": qi, "pair": pair, "final": final,
                                          "text": txt[:400]}) + "\n")
        for style, (p, f) in tally.items():
            print(f"  M={M:2d} {style:5s} pair={p}/{total} final-number={f}/{total}",
                  flush=True)


def exp_chunked(model, tok, ctx, device, open_ids, args):
    rng = random.Random(41)
    probes = make_probes(n=args.n_probes, path="/tmp/explore-probes.jsonl")
    print("=== chunked: 1300-token late-placement turn, single vs 300-tok chunks ===",
          flush=True)
    for mode in ["single", "chunked"]:
        hits_sum, n = {}, 0
        for p in probes[:args.n_probes]:
            text = ("Some notes from my journal: " + filler_text(tok, 1300, rng) +
                    " Anyway, on to my request. " + p["question"])
            if mode == "single":
                bank = [ingest(model, tok, ctx, user_chunk_ids(tok, text), device)]
            else:
                raw = tok(text, add_special_tokens=False).input_ids
                bank = []
                for i in range(0, len(raw), 300):
                    piece = tok.decode(raw[i:i + 300])
                    bank.append(ingest(model, tok, ctx,
                                       user_chunk_ids(tok, piece), device))
            txt, ent = generate(model, tok, ctx, bank, open_ids, device)
            hits = score_probe(txt, p["bindings"])
            for k, h in hits.items():
                hits_sum[k] = hits_sum.get(k, 0) + int(h)
            n += 1
            OUT.write(json.dumps({"exp": "chunked", "mode": mode, "hits": hits,
                                  "entropy": round(ent, 3),
                                  "text": txt[:200]}) + "\n")
        sc = {k: v / n for k, v in hits_sum.items()}
        print(f"  {mode:8s} recall={sum(sc.values())/len(sc):.2f}  " +
              " ".join(f"{k}={v:.2f}" for k, v in sc.items()), flush=True)


def exp_rescue(model, tok, ctx, device, open_ids, args):
    probes = make_probes(n=args.n_probes, path="/tmp/explore-probes.jsonl")
    stream = user_questions(tok, 320)
    dbanks = [ingest(model, tok, ctx, user_chunk_ids(tok, next(stream)), device)
              for _ in range(8)]
    print("=== rescue: probe + 8 distractors, capture mitigations ===", flush=True)
    for cond in ["first", "last", "first-topk4", "first-refname", "first-refsem"]:
        hits_sum, n = {}, 0
        for p in probes[:args.n_probes]:
            b = p["bindings"]
            w = ingest(model, tok, ctx, user_chunk_ids(tok, p["question"]), device)
            bank = [w] + dbanks if cond != "last" else dbanks + [w]
            if cond == "first-refname":
                redirect = (f"Please now handle the request from {b['name']}: "
                            f"write the apology note they asked for.")
                bank.append(ingest(model, tok, ctx,
                                   user_chunk_ids(tok, redirect), device))
            elif cond == "first-refsem":
                redirect = ("Please now handle the request about the unpaid "
                            "loan: write the apology note that was asked for.")
                bank.append(ingest(model, tok, ctx,
                                   user_chunk_ids(tok, redirect), device))
            ctx.top_k = 4 if cond == "first-topk4" else None
            txt, ent = generate(model, tok, ctx, bank, open_ids, device)
            ctx.top_k = None
            hits = score_probe(txt, b)
            for k, h in hits.items():
                hits_sum[k] = hits_sum.get(k, 0) + int(h)
            n += 1
            OUT.write(json.dumps({"exp": "rescue", "cond": cond, "hits": hits,
                                  "entropy": round(ent, 3),
                                  "text": txt[:200]}) + "\n")
        sc = {k: v / n for k, v in hits_sum.items()}
        print(f"  {cond:14s} recall={sum(sc.values())/len(sc):.2f}  " +
              " ".join(f"{k}={v:.2f}" for k, v in sc.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-probes", type=int, default=6)
    ap.add_argument("--read-heads", type=int, default=4)
    ap.add_argument("--read-rank", type=int, default=128)
    ap.add_argument("--write-layer", type=int, default=19)
    args = ap.parse_args()
    tok, model, ctx = load_checkpoint(args.ckpt, args)
    device = next(model.parameters()).device
    open_ids = encode(tok, ["x"])[1]
    exp_think(model, tok, ctx, device, open_ids, args)
    exp_chunked(model, tok, ctx, device, open_ids, args)
    exp_rescue(model, tok, ctx, device, open_ids, args)


if __name__ == "__main__":
    main()
