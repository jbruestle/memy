"""Data: ultrachat_200k first user turns + synthetic multi-binding probes."""

import json
import random
import re

from datasets import load_dataset


def user_questions(tokenizer, max_user_tokens: int = 320, split: str = "train_sft"):
    """Yield first-turn user questions, length-filtered. Single pass, no shuffle
    beyond the dataset's own order (we never repeat examples anyway)."""
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    for row in ds:
        msgs = row["messages"]
        if not msgs or msgs[0]["role"] != "user":
            continue
        q = msgs[0]["content"].strip()
        if not q:
            continue
        if len(tokenizer(q, add_special_tokens=False).input_ids) > max_user_tokens:
            continue
        yield q


FIRST = ["Karst", "Mireille", "Obadiah", "Tsuneo", "Waverly", "Ilsa", "Dmitri", "Yolanda",
         "Ferdinand", "Anouk", "Ravi", "Solveig", "Quentin", "Zelda", "Hamish", "Petra"]
CITY = ["Tarnow", "Ostrava", "Bujumbura", "Fresno", "Ulm", "Kagoshima", "Recife", "Tromso",
        "Davao", "Windhoek", "Cuenca", "Galway", "Bandung", "Split", "Regina", "Matera"]
ITEM = ["kayak", "theremin", "microscope", "espresso machine", "unicycle", "loom",
        "telescope", "accordion", "beehive", "kiln", "drone", "banjo"]
REL = ["sister", "brother", "cousin", "neighbor", "landlord", "coworker"]


def make_probes(n: int = 200, seed: int = 1234, path: str | None = "probes.jsonl"):
    """Questions with several independent arbitrary bindings that cannot be
    compressed into one gist vector. Bindings are recorded for exact-match
    scoring of free generations."""
    rng = random.Random(seed)
    probes = []
    for _ in range(n):
        name, city = rng.choice(FIRST), rng.choice(CITY)
        rel, relname = rng.choice(REL), rng.choice(FIRST)
        item = rng.choice(ITEM)
        owed = rng.randint(11, 989)
        year = rng.randint(1961, 2019)
        q = (f"Hi! My name is {name} and I live in {city}. My {rel} {relname} lent me "
             f"{owed} dollars in {year} to buy a used {item}, and I still have not paid "
             f"it back. Please write a short apology note from me to {relname} that "
             f"mentions my name, the exact amount, the year, and what I bought.")
        probes.append({"question": q,
                       "bindings": {"name": name, "city": city, "relname": relname,
                                    "amount": str(owed), "year": str(year), "item": item}})
    if path:
        with open(path, "w") as f:
            for p in probes:
                f.write(json.dumps(p) + "\n")
    return probes


def score_probe(generation: str, bindings: dict) -> dict:
    """Word-boundary exact recall per binding (city excluded: not requested in
    the note). Boundary matters for numbers: '46' must not match '460'."""
    keys = ["name", "relname", "amount", "year", "item"]
    return {k: bool(re.search(r"\b" + re.escape(bindings[k]) + r"\b", generation))
            for k in keys}


if __name__ == "__main__":
    make_probes()
    print("wrote probes.jsonl")
