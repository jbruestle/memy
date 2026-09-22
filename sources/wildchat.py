"""Source: WildChat-1M multi-turn chats, as-is history with a random cutoff.

Sample = turns[:2t+1] of a conversation (user0, assistant0, ..., user_t),
target = the dataset's assistant turn t is NOT used (the teacher regenerates
it); cutoff t is drawn uniformly over the conversation's assistant turns
per (seed, epoch, row). No gold, no distractors. Filters: language,
non-toxic, >= min_rounds rounds, every turn <= max_turn_tokens, history
<= max_history_tokens. Eval = first n_eval eligible rows with t fixed by
seed 0; train skips those rows."""

from datasets import load_dataset

from ._util import n_tokens, rng_for

NAME = "wildchat"
DEFAULTS = dict(language="English", min_rounds=2, max_turn_tokens=4096,
                max_history_tokens=16384, n_eval=64, split="train")
_cache = {}


def _ds(cfg):
    key = cfg["split"]
    if key not in _cache:
        _cache[key] = load_dataset("allenai/WildChat-1M", split=key)
    return _cache[key]


def _turns(cfg, row):
    """Cleaned (role, text) list or None if the row fails a filter."""
    if row["language"] != cfg["language"] or row["toxic"] or row["turn"] < cfg["min_rounds"]:
        return None
    conv = row["conversation"]
    turns = []
    for i, m in enumerate(conv):
        want = "user" if i % 2 == 0 else "assistant"
        if m["role"] != want or not m["content"] or not m["content"].strip():
            return None
        turns.append((want, m["content"].strip()))
    if len(turns) < 2 * cfg["min_rounds"] or len(turns) % 2:
        return None
    tok, cap = cfg["tok"], cfg["max_turn_tokens"]
    lens = []
    for _, t in turns:
        if len(t) > 6 * cap:          # cheap pre-check before tokenizing
            return None
        n = n_tokens(tok, t)
        if n > cap:
            return None
        lens.append(n)
    return turns, lens


def _sample(cfg, i, turns, lens, t):
    hist = turns[:2 * t + 1]
    if sum(lens[:2 * t + 1]) > cfg["max_history_tokens"]:
        return None
    return {"id": f"{NAME}:{i}:{t}", "source": NAME, "gold": [], "distractors": [],
            "turns": hist, "meta": {"cutoff": t, "rounds": len(turns) // 2,
                                    "reference": turns[2 * t + 1][1]}}


def _eval_rows(cfg):
    key = ("eval", cfg["split"], cfg["n_eval"])
    if key not in _cache:
        rows = []
        for i, row in enumerate(_ds(cfg)):
            r = _turns(cfg, row)
            if r is None:
                continue
            turns, lens = r
            t = rng_for(0, 0, i).randrange(len(turns) // 2)
            s = _sample(cfg, i, turns, lens, t)
            if s is not None:
                rows.append((i, s))
                if len(rows) >= cfg["n_eval"]:
                    break
        _cache[key] = rows
    return _cache[key]


def eval(cfg):
    return [s for _, s in _eval_rows(cfg)]


def train(cfg, seed, epoch, start=0, shard=(0, 1)):
    skip = {i for i, _ in _eval_rows(cfg)}
    rank, world = shard
    for i, row in enumerate(_ds(cfg)):
        if i < start or i % world != rank or i in skip:
            continue
        r = _turns(cfg, row)
        if r is None:
            continue
        turns, lens = r
        t = rng_for(seed, epoch, i).randrange(len(turns) // 2)
        s = _sample(cfg, i, turns, lens, t)
        if s is not None:
            yield i + 1, s
