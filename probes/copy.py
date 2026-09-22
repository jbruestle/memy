"""Copy probe: verbatim-copy samples (same generator as `sources/copy`),
scored on free generation by exact match and by longest-correct-prefix
fraction (token level) — the direct measure of prefix-correct /
tail-lossy retrieval. Reported overall, per selection kind, and per target
length bucket; teacher scores alongside."""

import re

from sources import copy as copysrc
from sources._util import rng_for

NAME = "copy"
DEFAULTS = dict(n=32, seed=7, split="validation", **{k: v for k, v in copysrc.DEFAULTS.items()
                                                       if k not in ("n_eval", "split", "eval_split")})


def samples(cfg, step):
    out, i = [], 0
    n_pages = len(copysrc._pages(cfg["split"]))
    rng = rng_for(cfg["seed"], 0, "copy-probe")
    while len(out) < cfg["n"] and i < 5000:
        pi = rng.randrange(n_pages)
        s = copysrc.make(cfg, rng_for(cfg["seed"], 0, f"copy-probe:{i}"), cfg["split"], pi)
        i += 1
        if s is not None:
            s["id"] = f"{NAME}:{cfg['seed']}:{len(out)}"
            out.append(s)
    return out


def _norm(t):
    return re.sub(r"\s+", " ", t.strip().strip('"').strip())


def _prefix_frac(tok, target, text):
    a, b = tok(_norm(target), add_special_tokens=False).input_ids, \
        tok(_norm(text), add_special_tokens=False).input_ids
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k / max(len(a), 1)


def _score(tok, results, field):
    em = [float(_norm(r[field]) == _norm(r["sample"]["meta"]["target"])) for r in results]
    pf = [_prefix_frac(tok, r["sample"]["meta"]["target"], r[field]) for r in results]
    n = max(len(results), 1)
    out = {"em": sum(em) / n, "prefix": sum(pf) / n}
    by = {}
    for r, e, p in zip(results, em, pf):
        m = r["sample"]["meta"]
        for key in (m["kind"], "len<=20" if m["target_tokens"] <= 20 else
                    "len<=60" if m["target_tokens"] <= 60 else "len>60"):
            by.setdefault(key, []).append((e, p))
    out["by"] = {k: {"em": sum(e for e, _ in v) / len(v), "prefix": sum(p for _, p in v) / len(v),
                     "n": len(v)} for k, v in by.items()}
    return out


def score(cfg, results):
    tok = cfg["tok"]
    st, te = _score(tok, results, "student"), _score(tok, results, "teacher")
    return {**st, "teacher_em": te["em"], "teacher_prefix": te["prefix"], "teacher_by": te["by"],
            "sample": results[0]["student"][:300], "sample_target": results[0]["sample"]["meta"]["target"][:300]}
