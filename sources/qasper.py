"""Source: Qasper long-document QA over NLP papers.

Sample = the whole paper as ONE bank passage (never sectioned) + D whole
distractor papers from the same split; teacher_gold = the annotated
evidence paragraphs only (the teacher never needs the long context); one
user turn = the question. D uniform in `distractors` per (seed, epoch,
question). Unanswerable questions and answers without evidence are
dropped; papers over max_paper_tokens are dropped. Positions are paper
rows (one paper yields several questions). Eval = validation split."""

from datasets import load_dataset

from ._util import n_tokens, parse_range, rng_for

NAME = "qasper"
DEFAULTS = dict(distractors="0:4", max_paper_tokens=8192, n_eval=64,
                split="train", eval_split="validation")
_cache = {}


def _ds(split):
    if split not in _cache:
        _cache[split] = load_dataset("allenai/qasper", split=split,
                                     revision="refs/convert/parquet")
    return _cache[split]


def paper_text(row):
    parts = [row["title"].strip(), row["abstract"].strip()]
    ft = row["full_text"]
    for name, paras in zip(ft["section_name"], ft["paragraphs"]):
        body = "\n\n".join(p.strip() for p in paras if p and p.strip())
        if body:
            parts.append(f"## {name.strip()}\n\n{body}" if name else body)
    return "\n\n".join(p for p in parts if p)


def _papers(cfg, split):
    """[(row index, text)] of papers within the token cap."""
    key = ("papers", split, cfg["max_paper_tokens"])
    if key not in _cache:
        out = []
        for i, row in enumerate(_ds(split)):
            text = paper_text(row)
            if n_tokens(cfg["tok"], text) <= cfg["max_paper_tokens"]:
                out.append((i, text))
        _cache[key] = out
    return _cache[key]


def _answer(a):
    if a["unanswerable"]:
        return None, None
    ev = [e for e in a["evidence"] if e and not e.startswith("FLOAT SELECTED")]
    if not ev:
        return None, None
    if a["extractive_spans"]:
        ans = "; ".join(a["extractive_spans"])
    elif a["yes_no"] is not None:
        ans = "Yes" if a["yes_no"] else "No"
    else:
        ans = (a["free_form_answer"] or "").strip()
    return (ans or None), ev


def _questions(cfg, split, i, row):
    text = dict(_papers(cfg, split)).get(i)
    if text is None:
        return []
    qas = row["qas"]
    out = []
    for q, qid, answers in zip(qas["question"], qas["question_id"], qas["answers"]):
        for a in answers["answer"]:                 # several annotators; first usable wins
            ans, ev = _answer(a)
            if ans is not None:
                out.append((qid, q, ans, ev))
                break
    return text, out


def _sample(cfg, split, i, text, qid, q, ans, ev, rng):
    others = [(j, t) for j, t in _papers(cfg, split) if j != i]
    lo, hi = parse_range(cfg["distractors"])
    d = rng.randint(lo, min(hi, len(others)))
    dis = [t for _, t in rng.sample(others, d)]
    return {"id": f"{NAME}:{qid}", "source": NAME, "gold": [text], "distractors": dis,
            "teacher_gold": ev, "turns": [("user", q)],
            "meta": {"answer": ans, "paper": i, "n_evidence": len(ev), "n_distractors": d}}


def eval(cfg):
    key = ("eval", cfg["eval_split"], cfg["n_eval"])
    if key not in _cache:
        out, split = [], cfg["eval_split"]
        for i, row in enumerate(_ds(split)):
            r = _questions(cfg, split, i, row)
            if not r:
                continue
            text, qs = r
            for qid, q, ans, ev in qs:
                out.append(_sample(cfg, split, i, text, qid, q, ans, ev, rng_for(0, 0, qid)))
                if len(out) >= cfg["n_eval"]:
                    break
            if len(out) >= cfg["n_eval"]:
                break
        _cache[key] = out
    return _cache[key]


def train(cfg, seed, epoch, start=0, shard=(0, 1)):
    rank, world = shard
    split = cfg["split"]
    for i, row in enumerate(_ds(split)):
        if i < start or i % world != rank:
            continue
        r = _questions(cfg, split, i, row)
        if not r:
            continue
        text, qs = r
        for qid, q, ans, ev in qs:
            yield i + 1, _sample(cfg, split, i, text, qid, q, ans, ev, rng_for(seed, epoch, qid))
