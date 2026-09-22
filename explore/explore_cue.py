"""Follow-up: is targeted-query failure a 'cold query' problem?

Same M name-age pairs, three query styles:
  bare     - "Answer with just the number."            (cold: age emitted first)
  sentence - "Answer with a complete sentence that starts with the person's
              name."                                    (name cue precedes age)
  forced   - teacher-force the prefix "<Name> is" and let it continue.

Usage: python explore/explore_cue.py --ckpt runs/L2-v1/ckpt-14000
"""

import argparse
import json
import random
import re

import torch

from explore import NAMES, ingest, generate, user_chunk_ids
from probe_mech import load_checkpoint
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for data/train/model imports
from train import encode, pad_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--read-heads", type=int, default=4)
    ap.add_argument("--read-rank", type=int, default=128)
    ap.add_argument("--write-layer", type=int, default=19)
    args = ap.parse_args()
    tok, model, ctx = load_checkpoint(args.ckpt, args)
    device = next(model.parameters()).device
    open_ids = encode(tok, ["x"])[1]
    rng = random.Random(21)
    out = open("explore_out/cue.jsonl", "a", buffering=1)

    for M in [4, 8, 16]:
        tally = {"bare": 0, "sentence": 0, "forced": 0}
        total = 0
        for trial in range(args.trials):
            names = rng.sample(NAMES, M)
            ages = rng.sample(range(18, 98), M)
            data = " ".join(f"{n} is {a} years old." for n, a in zip(names, ages))
            for qi in [0, M // 2, M - 1]:
                total += 1
                styles = {
                    "bare": (f"{data} How old is {names[qi]}? Please answer with "
                             f"just the number.", None),
                    "sentence": (f"{data} How old is {names[qi]}? Please answer "
                                 f"with a complete sentence that starts with the "
                                 f"person's name.", None),
                    "forced": (f"{data} How old is {names[qi]}?",
                               f"{names[qi]} is"),
                }
                for style, (q, prefix) in styles.items():
                    bank = [ingest(model, tok, ctx, user_chunk_ids(tok, q), device)]
                    if prefix is None:
                        txt, _ = generate(model, tok, ctx, bank, open_ids, device,
                                          max_gen=60)
                    else:
                        pids = open_ids + tok(prefix, add_special_tokens=False).input_ids
                        txt, _ = generate(model, tok, ctx, bank, pids, device,
                                          max_gen=20)
                        txt = prefix + " " + txt
                    hit = bool(re.search(rf"\b{ages[qi]}\b", txt))
                    tally[style] += int(hit)
                    out.write(json.dumps({"M": M, "trial": trial, "qi": qi,
                                          "style": style, "hit": hit,
                                          "text": txt[:100]}) + "\n")
        print(f"M={M:2d}  " + "  ".join(f"{s}={tally[s]}/{total}" for s in tally),
              flush=True)
    out.close()


if __name__ == "__main__":
    main()
