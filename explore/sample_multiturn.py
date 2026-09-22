"""Dump N random ultrachat_200k conversations with >= 4 messages (first 4 shown)."""
import argparse, json, random, textwrap
from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=20)
p.add_argument("--min-msgs", type=int, default=4)
p.add_argument("--show", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="explore_out/multiturn_samples.jsonl")
a = p.parse_args()

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
rng = random.Random(a.seed)
idx = [i for i in rng.sample(range(len(ds)), 2000) if len(ds[i]["messages"]) >= a.min_msgs][: a.n]
lens = [len(m["messages"]) for m in ds.select(range(0, len(ds), 50))]
print(f"dataset rows: {len(ds)}; msg-count histogram on 1/50 subsample: "
      f"{ {k: lens.count(k) for k in sorted(set(lens))} }\n")

with open(a.out, "w") as f:
    for k, i in enumerate(idx):
        msgs = ds[i]["messages"]
        f.write(json.dumps({"idx": i, "n_msgs": len(msgs), "messages": msgs[: a.show]}) + "\n")
        print(f"{'='*100}\n### SAMPLE {k+1}  (row {i}, {len(msgs)} messages total)\n")
        for m in msgs[: a.show]:
            print(f"--- {m['role'].upper()} ({len(m['content'].split())} words) ---")
            print(textwrap.fill(m["content"], 100, replace_whitespace=False))
            print()
