"""Source: TriviaQA (rc.wikipedia) long-document QA.

Sample = the question's Wikipedia entity pages as gold + D pages of other
questions as distractors; one user turn = the question. Pages longer than
`chunk_tokens` are split at paragraph boundaries into several bank
passages (each prefixed with the page title), so a long relevant page is
never dropped; samples whose gold exceeds `max_gold_tokens` in total are
dropped instead. TriviaQA has no evidence annotation, so teacher_gold is
distant supervision: each page's lead paragraph plus up to
`evidence_paras` paragraphs containing an answer alias; samples with no
alias hit are dropped (unanswerable from the bank). Eval = validation."""

import re

from datasets import load_dataset

from ._util import n_tokens, parse_range, rng_for

NAME = "triviaqa"
DEFAULTS = dict(distractors="0:2", chunk_tokens=4096, max_gold_tokens=16384,
                evidence_paras=3, n_eval=64, split="train", eval_split="validation")
_cache = {}


def _ds(split):
    if split not in _cache:
        _cache[split] = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split=split)
    return _cache[split]


def _paras(ctx):
    return [p.strip() for p in re.split(r"\n\s*\n", ctx) if p.strip()]


def _chunks(cfg, title, ctx):
    """Split a page into passages of <= chunk_tokens at paragraph boundaries
    (a single oversized paragraph is truncated). Returns [(text, n_tokens)]."""
    tok, cap = cfg["tok"], cfg["chunk_tokens"]
    head = f"{title}\n\n"
    n_head = n_tokens(tok, head)
    out, cur, cur_n = [], [], n_head
    for p in _paras(ctx):
        n = n_tokens(tok, p)
        if n > cap - n_head:
            ids = tok(p, add_special_tokens=False).input_ids[:cap - n_head]
            p, n = tok.decode(ids), len(ids)
        if cur and cur_n + n > cap:
            out.append((head + "\n\n".join(cur), cur_n))
            cur, cur_n = [], n_head
        cur.append(p); cur_n += n
    if cur:
        out.append((head + "\n\n".join(cur), cur_n))
    return out


def _alias_re(row):
    aliases = {a.strip().lower() for a in row["answer"]["aliases"] + [row["answer"]["value"]]
               if len(a.strip()) >= 2}
    return re.compile(r"\b(" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
                      + r")\b", re.IGNORECASE)


def _evidence(cfg, title, ctx, pat):
    paras = _paras(ctx)
    if not paras:
        return None
    hits = [p for p in paras[1:] if pat.search(p)][:cfg["evidence_paras"]]
    if not hits and not pat.search(paras[0]):
        return None
    return f"{title}\n\n" + "\n\n".join([paras[0]] + hits)


def _distractor(cfg, split, rng, exclude):
    """A random page of another question (as chunks), within max_gold_tokens."""
    ds = _ds(split)
    for _ in range(50):
        j = rng.randrange(len(ds))
        if j in exclude:
            continue
        ep = ds[j]["entity_pages"]
        if not ep["title"]:
            continue
        k = rng.randrange(len(ep["title"]))
        ch = _chunks(cfg, ep["title"][k], ep["wiki_context"][k])
        if ch and sum(n for _, n in ch) <= cfg["max_gold_tokens"]:
            return [t for t, _ in ch]
    return []


def _sample(cfg, split, i, row, rng):
    ep = row["entity_pages"]
    pat = _alias_re(row)
    gold, tgold, n_gold = [], [], 0
    for title, ctx in zip(ep["title"], ep["wiki_context"]):
        ch = _chunks(cfg, title, ctx)
        gold += [t for t, _ in ch]
        n_gold += sum(n for _, n in ch)
        ev = _evidence(cfg, title, ctx, pat)
        if ev:
            tgold.append(ev)
    if not gold or not tgold or n_gold > cfg["max_gold_tokens"]:
        return None
    lo, hi = parse_range(cfg["distractors"])
    d = rng.randint(lo, hi)
    dis = []
    for _ in range(d):
        dis += _distractor(cfg, split, rng, {i})
    return {"id": f"{NAME}:{row['question_id']}", "source": NAME, "gold": gold,
            "distractors": dis, "teacher_gold": tgold, "turns": [("user", row["question"])],
            "meta": {"answer": row["answer"]["value"], "aliases": row["answer"]["aliases"],
                     "n_pages": len(ep["title"]), "n_gold_chunks": len(gold),
                     "gold_tokens": n_gold, "n_distractor_pages": d,
                     "n_distractor_chunks": len(dis)}}


def eval(cfg):
    key = ("eval", cfg["eval_split"], cfg["n_eval"])
    if key not in _cache:
        out, split = [], cfg["eval_split"]
        for i, row in enumerate(_ds(split)):
            s = _sample(cfg, split, i, row, rng_for(0, 0, row["question_id"]))
            if s is not None:
                out.append(s)
                if len(out) >= cfg["n_eval"]:
                    break
        _cache[key] = out
    return _cache[key]


def train(cfg, seed, epoch, start=0, shard=(0, 1)):
    rank, world = shard
    split = cfg["split"]
    ds = _ds(split)
    for i in range(start, len(ds)):
        if i % world != rank:
            continue
        row = ds[i]
        s = _sample(cfg, split, i, row, rng_for(seed, epoch, row["question_id"]))
        if s is not None:
            yield i + 1, s
