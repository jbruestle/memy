"""Probe plugins. Each module exposes NAME, DEFAULTS, samples(cfg, step) ->
[Sample] and score(cfg, results) -> dict (the JSON blob to log). A result is
{"sample", "student", "teacher", "read_entropy"} with texts decoded."""

from . import bindings

REGISTRY = {m.NAME: m for m in [bindings]}


def get(name):
    return REGISTRY[name]
