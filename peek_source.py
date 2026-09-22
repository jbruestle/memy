"""Poke a source plugin without the model: print Samples as the engine would
encode them (chunk token counts, teacher prompt length), or length stats.

  python peek_source.py wildchat                       # 3 train samples
  python peek_source.py musique --split eval --n 5
  python peek_source.py qasper --stats 100 --tags      # length distribution
  python peek_source.py wildchat --cfg max_turn_tokens=2048 --json > out.jsonl
"""

import argparse
import json
import statistics

from transformers import AutoTokenizer

import sources
from engine import MODEL, Encoder
from train_v2 import module_cfg, parse_cfg


def short(t, n=160):
    t = t.replace("\n", "\\n")
    return t if len(t) <= n else t[:n] + f"... [{len(t)} chars]"


def describe(enc, tok, s, e):
    print(f"--- {s['id']}  depth={e.depth} gold={len(s['gold'])} distractors={len(s['distractors'])} "
          f"chunk_tokens={e.tokens} prompt_tokens={len(e.prompt)}")
    for k, (p, ids) in enumerate(zip(s["gold"] + s["distractors"], e.bg)):
        tag = "gold" if k < len(s["gold"]) else "dis "
        print(f"  {tag} [{len(ids):5d} tok] {short(p)}")
    if "teacher_gold" in s:
        for p in s["teacher_gold"]:
            print(f"  tgold [{len(tok(p, add_special_tokens=False).input_ids):5d} tok] {short(p)}")
    for (role, text), ids in zip(s["turns"], e.turns):
        print(f"  {role:9s} [{len(ids):5d} tok] {short(text)}")
    if s.get("meta"):
        print("  meta:", short(json.dumps(s["meta"]), 300))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--split", choices=["train", "eval"], default="train")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--start", type=int, default=0, help="train stream position")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--cfg", action="append", help="field=value overrides for this source")
    ap.add_argument("--tags", action="store_true", help="encode with turn tags")
    ap.add_argument("--max-chunk-tokens", type=int, default=0)
    ap.add_argument("--max-prompt-tokens", type=int, default=0)
    ap.add_argument("--stats", type=int, default=0, help="length stats over this many samples")
    ap.add_argument("--json", action="store_true", help="dump raw samples as JSONL instead")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = sources.get(args.source)
    overrides = parse_cfg([f"{args.source}.{c}" for c in (args.cfg or [])])
    cfg = module_cfg(mod, overrides, tok)
    enc = Encoder(tok, args.tags, args.max_chunk_tokens, args.max_prompt_tokens)
    print("cfg:", {k: v for k, v in cfg.items() if k != "tok"})

    if args.split == "eval":
        stream = ((None, s) for s in mod.eval(cfg))
    else:
        stream = mod.train(cfg, args.seed, args.epoch, start=args.start)

    if args.stats:
        rows, dropped = [], 0
        for _, s in stream:
            e = enc.encode(s)
            if e is None:
                dropped += 1
                continue
            rows.append({"tokens": e.tokens, "prompt": len(e.prompt), "depth": e.depth,
                         "bg": len(e.bg), "max_chunk": max(len(c) for c in e.bg + e.turns),
                         "turn_max": max(len(c) for c in e.turns)})
            if len(rows) >= args.stats:
                break
        print(f"{len(rows)} samples ({dropped} dropped by caps)")
        for k in ("tokens", "prompt", "depth", "bg", "max_chunk", "turn_max"):
            xs = [r[k] for r in rows]
            print(f"  {k:10s} mean {statistics.mean(xs):8.1f}  p50 {pct(xs, .5):6d}  "
                  f"p90 {pct(xs, .9):6d}  max {max(xs):6d}")
        from collections import Counter
        print("  depth histogram:", dict(sorted(Counter(r["depth"] for r in rows).items())))
        return

    for k, (_, s) in enumerate(stream):
        if k >= args.n:
            break
        if args.json:
            print(json.dumps(s))
            continue
        e = enc.encode(s)
        if e is None:
            print(f"--- {s['id']} DROPPED by caps")
            continue
        describe(enc, tok, s, e)


if __name__ == "__main__":
    main()
