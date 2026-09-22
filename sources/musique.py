"""Source: MuSiQue-Full multi-hop QA (2-4 hops, 20 paragraphs per item).

Sample = gold (supporting paragraphs) + N distractors from the item's own
non-supporting paragraphs, one user turn = the question. N is uniform in
`distractors` per (seed, epoch, id). Unanswerable twins that keep >= 1
supporting paragraph are kept, subsampled so they make up `unanswerable_frac`
of the stream; twins with no supporting paragraph are dropped (nothing for
the teacher to see). Eval = first n_eval eligible items of the validation
split (distractor counts fixed by seed 0)."""

from datasets import load_dataset

from ._util import parse_range, rng_for

NAME = "musique"
DEFAULTS = dict(distractors="0:18", unanswerable_frac=0.15, n_eval=64,
                split="train", eval_split="validation")
_cache = {}


def _ds(split):
    if split not in _cache:
        _cache[split] = load_dataset("bdsaglam/musique", split=split)  # default config = full
    return _cache[split]


def _keep_prob(cfg, split):
    """Probability of keeping an unanswerable-with-support item so that the
    kept unanswerables are `unanswerable_frac` of all kept items."""
    key = ("keep", split)
    if key not in _cache:
        a = u = 0
        for row in _ds(split):
            sup = sum(p["is_supporting"] for p in row["paragraphs"])
            if row["answerable"]:
                a += 1
            elif sup:
                u += 1
        f = cfg["unanswerable_frac"]
        _cache[key] = min(1.0, (f / (1 - f)) * a / max(u, 1)) if f > 0 else 0.0
    return _cache[key]


def _passage(p):
    return f"{p['title']}\n\n{p['paragraph_text']}"


def _sample(cfg, row, rng):
    sup = [p for p in row["paragraphs"] if p["is_supporting"]]
    if not sup:
        return None
    if not row["answerable"] and rng.random() >= _keep_prob(cfg, cfg["split"]):
        return None
    rest = [p for p in row["paragraphs"] if not p["is_supporting"]]
    lo, hi = parse_range(cfg["distractors"])
    n = rng.randint(lo, min(hi, len(rest)))
    dis = rng.sample(rest, n)
    return {"id": f"{NAME}:{row['id']}", "source": NAME,
            "gold": [_passage(p) for p in sup], "distractors": [_passage(p) for p in dis],
            "turns": [("user", row["question"])],
            "meta": {"answer": row["answer"], "aliases": list(row.get("answer_aliases") or []),
                     "answerable": bool(row["answerable"]), "n_gold": len(sup),
                     "n_distractors": n,
                     "hops": len(row.get("question_decomposition") or [])}}


def eval(cfg):
    key = ("eval", cfg["eval_split"], cfg["n_eval"])
    if key not in _cache:
        out = []
        for i, row in enumerate(_ds(cfg["eval_split"])):
            s = _sample(cfg, row, rng_for(0, 0, row["id"]))
            if s is not None:
                out.append(s)
                if len(out) >= cfg["n_eval"]:
                    break
        _cache[key] = out
    return _cache[key]


def train(cfg, seed, epoch, start=0, shard=(0, 1)):
    rank, world = shard
    for i, row in enumerate(_ds(cfg["split"])):
        if i < start or i % world != rank:
            continue
        s = _sample(cfg, row, rng_for(seed, epoch, row["id"]))
        if s is not None:
            yield i + 1, s
