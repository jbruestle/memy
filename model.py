"""Memory surgery for Qwen3.5-4B.

Wraps the 8 full-attention decoder layers with memory read heads and captures
the post-layer-19 residual stream as the (parameter-free, RMS-normalized)
memory write stream. See DESIGN.md.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def pure_rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Parameter-free RMS normalization (the write op must stay unlearned)."""
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


class MemoryContext:
    """Shared mutable state threaded past the HF forward signatures.

    The training harness sets fields before each forward call; wrappers and
    readers consult it. Not an nn.Module on purpose: holds no parameters.
    """

    def __init__(self):
        self.reads_enabled = False      # False => read heads are bypassed entirely (teacher / L0 / pass 1)
        self.collect_writes = False     # True during the user-turn pass
        self.written = None             # (B, T, H) normalized write-site states, set by the write wrapper
        self.memory = None              # (B, N, H) memories visible to readers
        self.memory_mask = None         # (B, N) 1 = real memory, 0 = padding
        self.log_stats = False
        self.stats = []                 # per read site: dict(layer=, entropy=, argmax_hist=)
        self.capture_maps = False       # store full read softmaxes (diagnostics only)
        self.maps = []                  # per read call: dict(layer=, p=(B,heads,T,N) cpu)

    def clear(self):
        self.written = None
        self.memory = None
        self.memory_mask = None
        self.stats = []
        self.maps = []


class MemoryReader(nn.Module):
    """Low-rank cross-attention over the memory bank.

    score = (A h) . (B m) / sqrt(r); out = W_o(concat_heads(softmax @ (W_d m))).
    W_o is zero-initialized so the model is bit-identical to base at step 0.
    Module names deliberately avoid PEFT target patterns (q_proj etc.).
    """

    def __init__(self, hidden: int, n_heads: int = 4, rank: int = 128, layer_idx: int = -1):
        super().__init__()
        self.n_heads, self.rank, self.layer_idx = n_heads, rank, layer_idx
        self.mem_q = nn.Linear(hidden, n_heads * rank, bias=False)
        self.mem_k = nn.Linear(hidden, n_heads * rank, bias=False)
        self.mem_v = nn.Linear(hidden, n_heads * rank, bias=False)
        self.mem_o = nn.Linear(n_heads * rank, hidden, bias=False)
        nn.init.zeros_(self.mem_o.weight)

    def forward(self, h: torch.Tensor, ctx: MemoryContext) -> torch.Tensor:
        mem, mask = ctx.memory, ctx.memory_mask
        B, T, _ = h.shape
        N = mem.shape[1]
        q = self.mem_q(h).view(B, T, self.n_heads, self.rank)
        k = self.mem_k(mem).view(B, N, self.n_heads, self.rank)
        v = self.mem_v(mem).view(B, N, self.n_heads, self.rank)
        scores = torch.einsum("bthr,bnhr->bhtn", q, k) / math.sqrt(self.rank)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool()[:, None, None, :], torch.finfo(scores.dtype).min)
        p = F.softmax(scores.float(), dim=-1).to(v.dtype)  # (B, heads, T, N)
        if ctx.log_stats:
            with torch.no_grad():
                pf = p.float()
                ent = -(pf * (pf + 1e-9).log()).sum(-1).mean(dim=(1, 2))  # (B,)
                n = mask.float().sum(-1).clamp(min=2) if mask is not None \
                    else torch.full_like(ent, N)
                # normalized: 1.0 = uniform over the sample's N memories, ->0 = sharp
                ctx.stats.append({"layer": self.layer_idx,
                                  "entropy": (ent / n.log()).mean().item(),
                                  "argmax_mean_pos": pf.argmax(-1).float().mean().item()})
        if ctx.capture_maps:
            ctx.maps.append({"layer": self.layer_idx, "p": p.detach().float().cpu()})
        mix = torch.einsum("bhtn,bnhr->bthr", p, v).reshape(B, T, self.n_heads * self.rank)
        return self.mem_o(mix)


class MemoryLayerWrapper(nn.Module):
    """Transparent wrapper around a Qwen3_5DecoderLayer.

    Captures writes at the write site and adds read output into the residual
    stream at read sites. The outer model loop only touches `.layer_type`.
    """

    def __init__(self, inner: nn.Module, ctx: MemoryContext,
                 reader: MemoryReader | None = None, is_write_site: bool = False):
        super().__init__()
        self.inner = inner
        self.reader = reader
        self.is_write_site = is_write_site
        object.__setattr__(self, "ctx", ctx)  # plain attr: contexts hold tensors, not params

    @property
    def layer_type(self):
        return self.inner.layer_type

    def forward(self, hidden_states, *args, **kwargs):
        h_in = hidden_states
        out = self.inner(hidden_states, *args, **kwargs)
        ctx = self.ctx
        if self.is_write_site and ctx.collect_writes:
            ctx.written = pure_rmsnorm(out)
        if self.reader is not None and ctx.reads_enabled and ctx.memory is not None:
            out = out + self.reader(h_in, ctx)
        return out


def install_memory(model, write_layer: int = 19, n_heads: int = 4, rank: int = 128):
    """Wrap the text model's layers in-place. Returns (ctx, readers ModuleList).

    `model` is Qwen3_5ForConditionalGeneration; text layers live at
    model.model.language_model.layers (falls back to model.model.layers).
    Read sites = all full_attention layers.
    """
    lm = model.model
    text = getattr(lm, "language_model", lm)
    ctx = MemoryContext()
    readers = []
    hidden = text.config.hidden_size
    p = next(text.parameters())
    for i, layer in enumerate(text.layers):
        reader = None
        if layer.layer_type == "full_attention":
            reader = MemoryReader(hidden, n_heads, rank, layer_idx=i).to(device=p.device, dtype=p.dtype)
            readers.append(reader)
        text.layers[i] = MemoryLayerWrapper(layer, ctx, reader=reader,
                                            is_write_site=(i == write_layer))
    return ctx, nn.ModuleList(readers)


def reader_parameters(model):
    for m in model.modules():
        if isinstance(m, MemoryReader):
            yield from m.parameters()
