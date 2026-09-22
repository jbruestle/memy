"""Dataset plugins. Each module exposes NAME, DEFAULTS, train(cfg, seed, epoch,
start, shard) yielding (next_start, Sample), and eval(cfg) -> [Sample].
cfg always carries the tokenizer under "tok". Named `sources/` because
`datasets/` would shadow the HuggingFace package."""

from . import ultrachat

REGISTRY = {m.NAME: m for m in [ultrachat]}


def get(name):
    return REGISTRY[name]
