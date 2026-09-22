"""Scripted exploration of a memy checkpoint (machine-usable REPL).

Each experiment builds memory banks by ingesting chunks separately (positions
from 0 per chunk, like chat.py) and generates from memory alone. Results go to
explore_out/<exp>.jsonl plus a printed summary.

Experiments:
  scale     - M name-age pairs in one turn; targeted single-pair query at
              first/middle/last position, plus list-all. Recall + confusion vs M.
  length    - fixed 5-binding probe embedded early or late in filler text of
              growing total length (position OOD vs bank-size interference).
  multiturn - decompose the multi-turn failure: split data/question across
              chunks, position_ids offset, assistant-turn pollution,
              read-on-ingest contamination, no-data control.
  distract  - probe turn + N unrelated ultrachat turns in the bank; recall and
              read entropy vs N, optionally with --top-k.

Usage:
  python explore/explore.py --ckpt runs/L2-v1/ckpt-14000 --exp scale
  python explore/explore.py --ckpt runs/L2-v1/ckpt-14000 --exp all --top-k 0
"""

import argparse
import json
import os
import random
import re

import torch

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for data/train/model imports
from data import FIRST, make_probes, score_probe
from probe_mech import load_checkpoint
from train import encode, pad_batch

MORE_FIRST = ["Beatrix", "Caspian", "Delphine", "Emeric", "Fintan", "Greta", "Horacio",
              "Ingrid", "Jasper", "Katinka", "Leopold", "Maud", "Nikolai", "Odette",
              "Percival", "Rosalind"]
NAMES = FIRST + MORE_FIRST  # 32 unique

FILLER = ["The afternoon light settled slowly over the quiet harbor while gulls "
          "drifted above the water.",
          "A gentle breeze moved through the olive trees and carried the smell of "
          "rain from the distant hills.",
          "The old kitchen smelled of bread and rosemary, and a kettle murmured "
          "softly on the back burner.",
          "Clouds gathered along the ridge by evening, casting long violet shadows "
          "across the terraced fields.",
          "Somewhere beyond the garden wall a dog barked twice and then the lane "
          "fell silent again.",
          "The tide withdrew from the flats, leaving ribbons of kelp and the clean "
          "smell of salt behind.",
          "Morning fog clung to the pines until the sun finally burned through and "
          "warmed the wooden porch.",
          "Rain tapped against the window for most of the night, steady and "
          "unhurried, like a patient drummer.",
          "The market stalls were folding up for the day, canvas awnings snapping "
          "gently in the wind.",
          "A slow river slid past the meadow, dark and glassy, folding the willows "
          "into its surface."]


def filler_text(tok, n_tokens, rng):
    out, count = [], 0
    while count < n_tokens:
        s = rng.choice(FILLER)
        out.append(s)
        count += len(tok(s, add_special_tokens=False).input_ids)
    return " ".join(out)


def ingest(model, tok, ctx, ids, device, position_offset=0, reads_from=None):
    """Forward one chunk, return its (T, H) writes. reads_from: optional bank
    (list of (T,H)) visible during ingestion (the REPL's read-on-ingest)."""
    x = torch.tensor([ids], device=device)
    mask = torch.ones_like(x)
    if reads_from:
        mem = torch.cat(reads_from, dim=0)[None]
        ctx.memory = mem
        ctx.memory_mask = torch.ones(1, mem.shape[1], dtype=torch.long, device=device)
        ctx.reads_enabled = True
    else:
        ctx.memory, ctx.memory_mask, ctx.reads_enabled = None, None, False
    ctx.collect_writes = True
    kw = {}
    if position_offset:
        kw["position_ids"] = torch.arange(position_offset, position_offset + x.shape[1],
                                          device=device)[None]
    with torch.no_grad():
        model(input_ids=x, attention_mask=mask, use_cache=False, **kw)
    ctx.collect_writes = False
    w = ctx.written[0]
    ctx.written = None
    ctx.memory, ctx.memory_mask, ctx.reads_enabled = None, None, False
    return w


def generate(model, tok, ctx, bank, open_ids, device, max_gen=300):
    """Greedy generation from memory alone. Returns (text, mean_norm_entropy)."""
    mem = torch.cat(bank, dim=0)[None]
    ctx.memory = mem
    ctx.memory_mask = torch.ones(1, mem.shape[1], dtype=torch.long, device=device)
    ctx.reads_enabled = True
    ctx.log_stats = True
    ctx.stats = []
    oids, omask = pad_batch([open_ids], tok.pad_token_id, "left", device)
    with torch.no_grad():
        out = model.generate(input_ids=oids, attention_mask=omask,
                             max_new_tokens=max_gen, do_sample=False,
                             use_cache=True, pad_token_id=tok.pad_token_id)
    ctx.log_stats = False
    ent = sum(s["entropy"] for s in ctx.stats) / max(len(ctx.stats), 1)
    ctx.memory, ctx.memory_mask, ctx.reads_enabled = None, None, False
    ctx.stats = []
    return tok.decode(out[0, oids.shape[1]:], skip_special_tokens=True), ent


def user_chunk_ids(tok, text):
    return encode(tok, [text])[0][0]


def assistant_chunk_ids(tok, text, open_ids):
    return open_ids + tok(text, add_special_tokens=False).input_ids + [tok.eos_token_id]


# ---------------------------------------------------------------- scale

def exp_scale(model, tok, ctx, device, out, args):
    rng = random.Random(7)
    open_ids = encode(tok, ["x"])[1]
    print("\n=== scale: M name-age pairs, targeted + list-all queries ===")
    for M in [2, 4, 8, 16, 24, 32]:
        for trial in range(args.trials):
            names = rng.sample(NAMES, M)
            ages = rng.sample(range(18, 98), M)
            data = " ".join(f"{n} is {a} years old." for n, a in zip(names, ages))
            for pos_label, qi in [("first", 0), ("mid", M // 2), ("last", M - 1)]:
                q = (f"{data} How old is {names[qi]}? Please answer with just "
                     f"the number.")
                bank = [ingest(model, tok, ctx, user_chunk_ids(tok, q), device)]
                txt, ent = generate(model, tok, ctx, bank, open_ids, device, max_gen=120)
                hit = bool(re.search(rf"\b{ages[qi]}\b", txt))
                confused = [a for j, a in enumerate(ages)
                            if j != qi and re.search(rf"\b{a}\b", txt)]
                rec = {"exp": "scale-one", "M": M, "trial": trial, "pos": pos_label,
                       "bank": len(bank[0]), "hit": hit, "n_confused": len(confused),
                       "entropy": round(ent, 3), "text": txt[:120]}
                out.write(json.dumps(rec) + "\n")
                print(f"  M={M:2d} t{trial} {pos_label:5s} hit={int(hit)} "
                      f"conf={len(confused)} ent={ent:.2f}  {txt[:60]!r}")
            # list-all
            q = f"{data} Please list every person mentioned and their age."
            bank = [ingest(model, tok, ctx, user_chunk_ids(tok, q), device)]
            txt, ent = generate(model, tok, ctx, bank, open_ids, device,
                                max_gen=min(60 + 12 * M, 500))
            frac = sum(bool(re.search(rf"\b{a}\b", txt)) for a in ages) / M
            namefrac = sum(bool(re.search(rf"\b{n}\b", txt)) for n in names) / M
            rec = {"exp": "scale-all", "M": M, "trial": trial, "age_frac": frac,
                   "name_frac": namefrac, "entropy": round(ent, 3), "text": txt[:400]}
            out.write(json.dumps(rec) + "\n")
            print(f"  M={M:2d} t{trial} ALL   ages={frac:.2f} names={namefrac:.2f} "
                  f"ent={ent:.2f}")


# ---------------------------------------------------------------- length

def exp_length(model, tok, ctx, device, out, args):
    rng = random.Random(11)
    probes = make_probes(n=args.n_probes, path="/tmp/explore-probes.jsonl")
    open_ids = encode(tok, ["x"])[1]
    print("\n=== length: probe early/late in filler, growing user-turn length ===")
    for total in [0, 300, 700, 1300]:
        for place in (["early", "late"] if total else ["-"]):
            hits_sum, n = {}, 0
            for p in probes[:args.n_probes]:
                fill = filler_text(tok, total, rng) if total else ""
                if place == "early":
                    q = p["question"] + " By the way, some notes from my journal: " + fill
                elif place == "late":
                    q = ("Some notes from my journal: " + fill +
                         " Anyway, on to my request. " + p["question"])
                else:
                    q = p["question"]
                ids = user_chunk_ids(tok, q)
                bank = [ingest(model, tok, ctx, ids, device)]
                txt, ent = generate(model, tok, ctx, bank, open_ids, device)
                hits = score_probe(txt, p["bindings"])
                for k, h in hits.items():
                    hits_sum[k] = hits_sum.get(k, 0) + int(h)
                n += 1
                out.write(json.dumps({"exp": "length", "filler": total, "place": place,
                                      "chunk_tokens": len(ids), "hits": hits,
                                      "entropy": round(ent, 3),
                                      "text": txt[:200]}) + "\n")
            sc = {k: v / n for k, v in hits_sum.items()}
            rec = sum(sc.values()) / len(sc)
            print(f"  filler={total:5d} place={place:5s} tokens~{len(ids):4d} "
                  f"recall={rec:.2f}  " +
                  " ".join(f"{k}={v:.2f}" for k, v in sc.items()))


# ---------------------------------------------------------------- multiturn

def exp_multiturn(model, tok, ctx, device, out, args):
    probes = make_probes(n=args.n_probes, path="/tmp/explore-probes.jsonl")
    open_ids = encode(tok, ["x"])[1]
    print("\n=== multiturn: decompose the split-chunk failure ===")

    def split_probe(p):
        b = p["bindings"]
        data = (f"Hi! My name is {b['name']} and I live in {b['city']}. My "
                f"{[w for w in p['question'].split() if w in ('sister','brother','cousin','neighbor','landlord','coworker')][0]} "
                f"{b['relname']} lent me {b['amount']} dollars in {b['year']} to buy "
                f"a used {b['item']}, and I still have not paid it back.")
        ask = (f"Please write a short apology note from me to {b['relname']} that "
               f"mentions my name, the exact amount, the year, and what I bought.")
        return data, ask

    variants = ["single", "split", "split-posoff", "split-ingestread",
                "split-bot-between", "q-only"]
    for variant in variants:
        hits_sum, n, ents = {}, 0, []
        for p in probes[:args.n_probes]:
            data, ask = split_probe(p)
            if variant == "single":
                bank = [ingest(model, tok, ctx, user_chunk_ids(tok, data + " " + ask),
                               device)]
            elif variant == "split":
                w1 = ingest(model, tok, ctx, user_chunk_ids(tok, data), device)
                w2 = ingest(model, tok, ctx, user_chunk_ids(tok, ask), device)
                bank = [w1, w2]
            elif variant == "split-posoff":
                ids1 = user_chunk_ids(tok, data)
                w1 = ingest(model, tok, ctx, ids1, device)
                w2 = ingest(model, tok, ctx, user_chunk_ids(tok, ask), device,
                            position_offset=len(ids1))
                bank = [w1, w2]
            elif variant == "split-ingestread":
                w1 = ingest(model, tok, ctx, user_chunk_ids(tok, data), device)
                w2 = ingest(model, tok, ctx, user_chunk_ids(tok, ask), device,
                            reads_from=[w1])
                bank = [w1, w2]
            elif variant == "split-bot-between":
                w1 = ingest(model, tok, ctx, user_chunk_ids(tok, data), device)
                bot = assistant_chunk_ids(tok, "I understand. I have noted all of "
                                          "that. How can I help you further?", open_ids)
                wb = ingest(model, tok, ctx, bot, device)
                w2 = ingest(model, tok, ctx, user_chunk_ids(tok, ask), device)
                bank = [w1, wb, w2]
            elif variant == "q-only":
                bank = [ingest(model, tok, ctx, user_chunk_ids(tok, ask), device)]
            txt, ent = generate(model, tok, ctx, bank, open_ids, device)
            hits = score_probe(txt, p["bindings"])
            for k, h in hits.items():
                hits_sum[k] = hits_sum.get(k, 0) + int(h)
            n += 1
            ents.append(ent)
            out.write(json.dumps({"exp": "multiturn", "variant": variant,
                                  "hits": hits, "entropy": round(ent, 3),
                                  "text": txt[:200]}) + "\n")
        sc = {k: v / n for k, v in hits_sum.items()}
        rec = sum(sc.values()) / len(sc)
        print(f"  {variant:18s} recall={rec:.2f} ent={sum(ents)/len(ents):.2f}  " +
              " ".join(f"{k}={v:.2f}" for k, v in sc.items()))


# ---------------------------------------------------------------- distract

def exp_distract(model, tok, ctx, device, out, args):
    from data import user_questions
    probes = make_probes(n=args.n_probes, path="/tmp/explore-probes.jsonl")
    open_ids = encode(tok, ["x"])[1]
    stream = user_questions(tok, 320)
    distractor_pool = [next(stream) for _ in range(24)]
    dbanks = [ingest(model, tok, ctx, user_chunk_ids(tok, d), device)
              for d in distractor_pool]
    print(f"\n=== distract: probe + N unrelated ultrachat turns in bank "
          f"(top_k={args.top_k or 'off'}) ===")
    ctx.top_k = args.top_k or None
    for nd in [0, 2, 4, 8, 16, 24]:
        hits_sum, n, ents, bank_n = {}, 0, [], 0
        for p in probes[:args.n_probes]:
            w = ingest(model, tok, ctx, user_chunk_ids(tok, p["question"]), device)
            bank = [w] + dbanks[:nd]   # probe FIRST, distractors after
            ctx.top_k = args.top_k or None
            txt, ent = generate(model, tok, ctx, bank, open_ids, device)
            hits = score_probe(txt, p["bindings"])
            for k, h in hits.items():
                hits_sum[k] = hits_sum.get(k, 0) + int(h)
            n += 1
            ents.append(ent)
            bank_n = sum(len(b) for b in bank)
            out.write(json.dumps({"exp": "distract", "n_distract": nd,
                                  "bank": bank_n, "top_k": args.top_k, "hits": hits,
                                  "entropy": round(ent, 3), "text": txt[:200]}) + "\n")
        sc = {k: v / n for k, v in hits_sum.items()}
        rec = sum(sc.values()) / len(sc)
        print(f"  nd={nd:2d} bank~{bank_n:5d} recall={rec:.2f} "
              f"ent={sum(ents)/len(ents):.2f}  " +
              " ".join(f"{k}={v:.2f}" for k, v in sc.items()))
    ctx.top_k = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--exp", default="all",
                    choices=["all", "scale", "length", "multiturn", "distract"])
    ap.add_argument("--n-probes", type=int, default=8)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=0, help="distract: read top-K (0=off)")
    ap.add_argument("--read-heads", type=int, default=4)
    ap.add_argument("--read-rank", type=int, default=128)
    ap.add_argument("--write-layer", type=int, default=19)
    args = ap.parse_args()

    tok, model, ctx = load_checkpoint(args.ckpt, args)
    device = next(model.parameters()).device
    os.makedirs("explore_out", exist_ok=True)

    exps = {"multiturn": exp_multiturn, "scale": exp_scale,
            "length": exp_length, "distract": exp_distract}
    todo = list(exps) if args.exp == "all" else [args.exp]
    for name in todo:
        with open(os.path.join("explore_out", f"{name}.jsonl"), "a") as out:
            exps[name](model, tok, ctx, device, out, args)


if __name__ == "__main__":
    main()
