"""Helpers shared by source plugins."""

import random


def rng_for(seed, epoch, key):
    """Deterministic per-sample RNG: same (seed, epoch, key) -> same draws,
    so a resumed stream reproduces its cutoffs / distractor picks."""
    return random.Random(f"{seed}:{epoch}:{key}")


def parse_range(spec):
    """'0:18' -> (0, 18); '4' -> (4, 4)."""
    if isinstance(spec, int):
        return spec, spec
    lo, _, hi = str(spec).partition(":")
    return int(lo), int(hi or lo)


def n_tokens(tok, text):
    return len(tok(text, add_special_tokens=False).input_ids)
