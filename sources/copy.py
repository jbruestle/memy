"""Source: verbatim copy with a content-based selection criterion.

Bank = one real passage (a window of Wikipedia paragraphs from the TriviaQA
pages) + D distractor passages, optionally including a NEAR-DUPLICATE of
the gold passage with the cue words replaced and a few other words
perturbed (hard negative). One user turn asks to copy a span verbatim,
selected by content:

  word       the sentence containing the word "X"
  next       the sentence immediately after the one containing "X"
  begins     the paragraph that begins with "<first words>"
  between    the text starting with "<A>" and ending with "<B>"
  ordinal    the N-th sentence of the passage (minority; never with a
             near-duplicate in the bank)

Trains content-addressed lookup followed by sequential readout (the
induction-head analogue over the bank) and the answer-first format. The
correct span is known (`meta.target`), so `probes/copy.py` scores exact
match and longest-correct-prefix on free generation. Eval / probe passages
come from the TriviaQA validation pages, train from its train pages."""

import re

from . import triviaqa
from ._util import n_tokens, parse_range, rng_for

NAME = "copy"
DEFAULTS = dict(passage_tokens="150:600", distractors="0:2", near_dup_frac=0.3,
                perturb_frac=0.05, kinds="word,next,begins,between,ordinal",
                ordinal_frac=0.1, n_eval=64, split="train", eval_split="validation")

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
_WORD = re.compile(r"[A-Za-z][A-Za-z-]{4,}")
TEMPLATES = {
    "word": 'From the text you were given, copy verbatim the sentence that contains the word "{x}".',
    "next": 'From the text you were given, copy verbatim the sentence that immediately follows the sentence containing the word "{x}".',
    "begins": 'From the text you were given, copy verbatim the paragraph that begins with "{x}".',
    "between": 'From the text you were given, copy verbatim the passage that starts with "{a}" and ends with "{b}" (inclusive).',
    "ordinal": "From the text you were given, copy verbatim the {n} sentence of the passage about {t}.",
}
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
SUFFIX = " Reply with only the copied text, nothing else."


def _pages(split):
    return triviaqa._ds(split)


def _paragraphs(page_ctx):
    """Paragraphs with >= 2 sentences; returns [(paragraph, [sentences])]."""
    out = []
    for p in triviaqa._paras(page_ctx):
        if len(p) < 80 or "\n" in p:
            continue
        sents = [s.strip() for s in _SENT.split(p) if s.strip()]
        if len(sents) >= 2 and all(len(s) >= 15 for s in sents):
            out.append((p, sents))
    return out


def _window(cfg, rng, paras):
    """A contiguous run of paragraphs totalling ~passage_tokens tokens."""
    lo, hi = parse_range(cfg["passage_tokens"])
    want = rng.randint(lo, hi)
    start = rng.randrange(len(paras))
    got, n = [], 0
    for p, sents in paras[start:]:
        t = n_tokens(cfg["tok"], p)
        if got and n + t > hi:
            break
        got.append((p, sents)); n += t
        if n >= want:
            break
    return got, n


def _passage_text(win):
    return "\n\n".join(p for p, _ in win)


def _unique_word(rng, sentence, passage_text, others):
    """A word (>= 5 letters) whose stem occurs once in the passage (case-
    insensitive, prefix match: "president" also counts "Presidents") and
    never in the other passages, so the cue is unambiguous."""
    cands = [w for w in _WORD.findall(sentence)]
    rng.shuffle(cands)
    for w in cands:
        pat = re.compile(r"\b" + re.escape(w), re.IGNORECASE)
        if len(pat.findall(passage_text)) == 1 and not any(pat.search(o) for o in others):
            return w
    return None


def _first_words(text, k):
    return " ".join(text.split()[:k])


def _perturb(rng, text, frac, pool, cues):
    """Near-duplicate: cue strings removed (replaced by pool words), plus a
    fraction of other long words swapped for pool words."""
    for c in cues:
        text = re.sub(r"\b" + re.escape(c) + r"[A-Za-z]*", lambda m: rng.choice(pool), text,
                      flags=re.IGNORECASE)
    def swap(m):
        return rng.choice(pool) if rng.random() < frac else m.group(0)
    return _WORD.sub(swap, text)


def make(cfg, rng, split, page_idx):
    """Build one Sample from page `page_idx` of the split, or None."""
    ds = _pages(split)
    row = ds[page_idx]
    ep = row["entity_pages"]
    if not ep["title"]:
        return None
    k = rng.randrange(len(ep["title"]))
    title, ctx = ep["title"][k], ep["wiki_context"][k]
    paras = _paragraphs(ctx)
    if len(paras) < 2:
        return None
    win, n_tok = _window(cfg, rng, paras)
    passage = _passage_text(win)

    # distractor passages (other pages)
    lo, hi = parse_range(cfg["distractors"])
    dis = []
    for _ in range(rng.randint(lo, hi)):
        for _try in range(20):
            j = rng.randrange(len(ds))
            e2 = ds[j]["entity_pages"]
            if j == page_idx or not e2["title"]:
                continue
            k2 = rng.randrange(len(e2["title"]))
            p2 = _paragraphs(e2["wiki_context"][k2])
            if len(p2) < 2:
                continue
            w2, _ = _window(cfg, rng, p2)
            dis.append(_passage_text(w2))
            break
    pool = [w for d in dis for w in _WORD.findall(d)] or ["thing", "place", "other"]

    kinds = [x for x in cfg["kinds"].split(",") if x]
    kind = "ordinal" if ("ordinal" in kinds and rng.random() < cfg["ordinal_frac"]) \
        else rng.choice([x for x in kinds if x != "ordinal"] or kinds)
    pi = rng.randrange(len(win))
    para, sents = win[pi]
    cues, target = [], None
    if kind == "word":
        si = rng.randrange(len(sents))
        x = _unique_word(rng, sents[si], passage, dis)
        if x is None:
            return None
        cues, target, instr = [x], sents[si], TEMPLATES["word"].format(x=x)
    elif kind == "next":
        if len(sents) < 2:
            return None
        si = rng.randrange(len(sents) - 1)
        x = _unique_word(rng, sents[si], passage, dis)
        if x is None:
            return None
        cues, target, instr = [x], sents[si + 1], TEMPLATES["next"].format(x=x)
    elif kind == "begins":
        x = _first_words(para, rng.randint(3, 5))
        if sum(p.startswith(x) for p, _ in win) != 1 or any(x in d for d in dis):
            return None
        cues, target, instr = [x], para, TEMPLATES["begins"].format(x=x)
    elif kind == "between":
        if len(sents) < 2:
            return None
        a_i = rng.randrange(len(sents) - 1)
        b_i = rng.randint(a_i + 1, min(len(sents) - 1, a_i + 3))
        a, b = _first_words(sents[a_i], 3), " ".join(sents[b_i].split()[-3:])
        if passage.count(a) != 1 or passage.count(b) != 1 or any(a in d or b in d for d in dis):
            return None
        cues, target = [a, b], " ".join(sents[a_i:b_i + 1])
        instr = TEMPLATES["between"].format(a=a, b=b)
    else:  # ordinal over the whole passage's sentences
        allsents = [s for _, ss in win for s in ss]
        if len(allsents) < 2 or len(allsents) > len(ORD):
            return None
        si = rng.randrange(len(allsents))
        target, instr = allsents[si], TEMPLATES["ordinal"].format(n=ORD[si], t=title)
    if kind != "ordinal" and rng.random() < cfg["near_dup_frac"]:
        dis.append(_perturb(rng, passage, cfg["perturb_frac"], pool, cues))
    rng.shuffle(dis)
    return {"id": f"{NAME}:{split}:{page_idx}:{rng.random():.6f}", "source": NAME,
            "gold": [passage], "distractors": dis, "turns": [("user", instr + SUFFIX)],
            "meta": {"target": target, "kind": kind, "passage_tokens": n_tok,
                     "target_tokens": n_tokens(cfg["tok"], target),
                     "near_dup": len(dis) > 0 and dis and any(d.startswith(passage[:40]) for d in dis)}}


def eval(cfg):
    key = ("copy-eval", cfg["eval_split"], cfg["n_eval"])   # own key: triviaqa.eval shares this dict
    if key not in triviaqa._cache:
        out, split = [], cfg["eval_split"]
        for i in range(len(_pages(split))):
            s = make(cfg, rng_for(0, 0, f"copy:{i}"), split, i)
            if s is not None:
                out.append(s)
                if len(out) >= cfg["n_eval"]:
                    break
        triviaqa._cache[key] = out
    return triviaqa._cache[key]


def train(cfg, seed, epoch, start=0, shard=(0, 1)):
    rank, world = shard
    split = cfg["split"]
    n = len(_pages(split))
    for i in range(start, n):
        if i % world != rank:
            continue
        s = make(cfg, rng_for(seed, epoch, f"copy:{i}"), split, i)
        if s is not None:
            yield i + 1, s
