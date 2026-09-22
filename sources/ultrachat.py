"""Source: ultrachat_200k first user turns (the v1 data).

Sample = one user turn; no gold, no distractors. Eval = the first `n_eval`
filtered rows (identical to v1's held-out slice); train = the rest, in
dataset order (v1 never shuffled). Positions are raw row indices, so
resume and sharding (row % world == rank) need no rescans."""

from datasets import load_dataset

NAME = "ultrachat"
DEFAULTS = dict(max_user_tokens=320, n_eval=64, split="train_sft")
_cache = {}


def _ds(cfg):
    key = cfg["split"]
    if key not in _cache:
        _cache[key] = load_dataset("HuggingFaceH4/ultrachat_200k", split=key)
    return _cache[key]


def _question(cfg, row):
    msgs = row["messages"]
    if not msgs or msgs[0]["role"] != "user":
        return None
    q = msgs[0]["content"].strip()
    if not q:
        return None
    if len(cfg["tok"](q, add_special_tokens=False).input_ids) > cfg["max_user_tokens"]:
        return None
    return q


def _sample(i, q):
    return {"id": f"{NAME}:{i}", "source": NAME, "gold": [], "distractors": [],
            "turns": [("user", q)], "meta": {}}


def _eval_rows(cfg):
    key = ("eval", cfg["split"], cfg["n_eval"], cfg["max_user_tokens"])
    if key not in _cache:
        rows = []
        for i, row in enumerate(_ds(cfg)):
            q = _question(cfg, row)
            if q is not None:
                rows.append((i, q))
                if len(rows) >= cfg["n_eval"]:
                    break
        _cache[key] = rows
    return _cache[key]


def eval(cfg):
    return [_sample(i, q) for i, q in _eval_rows(cfg)]


def train(cfg, seed, epoch, start=0, shard=(0, 1)):
    skip = {i for i, _ in _eval_rows(cfg)}
    rank, world = shard
    for i, row in enumerate(_ds(cfg)):
        if i < start or i % world != rank or i in skip:
            continue
        q = _question(cfg, row)
        if q is None:
            continue
        yield i + 1, _sample(i, q)
