"""Dataset plugins. Each module exposes NAME, DEFAULTS, train(cfg, seed, epoch,
start, shard) yielding (next_start, Sample), and eval(cfg) -> [Sample].
cfg always carries the tokenizer under "tok". Named `sources/` because
`datasets/` would shadow the HuggingFace package. Poke one with
`python peek_source.py <name>`."""

from . import musique, qasper, triviaqa, ultrachat, wildchat

REGISTRY = {m.NAME: m for m in [ultrachat, wildchat, musique, qasper, triviaqa]}


def get(name):
    return REGISTRY[name]
