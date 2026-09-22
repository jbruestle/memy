"""The v1 five-binding probe: a single user turn with several arbitrary
bindings (name, relative, amount, year, item); the note must restate them.
Same generator and seed as v1 (`data.make_probes`), so recall is comparable
with the L2-v1 log."""

from data import make_probes, score_probe

NAME = "bindings"
DEFAULTS = dict(n=32, seed=1234)
KEYS = ["name", "relname", "amount", "year", "item"]


def samples(cfg, step):
    probes = make_probes(n=cfg["n"], seed=cfg["seed"], path=None)
    return [{"id": f"{NAME}:{cfg['seed']}:{i}", "source": NAME, "gold": [], "distractors": [],
             "turns": [("user", p["question"])], "meta": p["bindings"]}
            for i, p in enumerate(probes)]


def _recall(results, field):
    per = {k: 0 for k in KEYS}
    for r in results:
        hits = score_probe(r[field], r["sample"]["meta"])
        for k in KEYS:
            per[k] += int(hits[k])
    n = max(len(results), 1)
    per = {k: v / n for k, v in per.items()}
    per["recall"] = sum(per.values()) / len(per)
    return per


def score(cfg, results):
    st, te = _recall(results, "student"), _recall(results, "teacher")
    return {**st, "teacher_recall": te["recall"], "teacher": {k: te[k] for k in KEYS},
            "sample": results[0]["student"][:400]}
