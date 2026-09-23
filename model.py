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
        self.collect_writes = False     # True during a write pass
        self.write_only = False         # True => layers above the write site are skipped (write passes)
        self.top_k = None               # int => each read keeps only its top-K scoring memories
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
    Default path is F.scaled_dot_product_attention (fused, no T x N score
    matrix in memory); the explicit path is used only for diagnostics.
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
        q = self.mem_q(h).view(B, T, self.n_heads, self.rank).transpose(1, 2)    # (B, heads, T, r)
        k = self.mem_k(mem).view(B, N, self.n_heads, self.rank).transpose(1, 2)  # (B, heads, N, r)
        v = self.mem_v(mem).view(B, N, self.n_heads, self.rank).transpose(1, 2)
        attn_mask = mask.bool()[:, None, None, :] if mask is not None else None   # True = attend
        if ctx.top_k is not None and ctx.top_k < N:
            if ctx.capture_maps:
                # Readmaps need the full (B, heads, T, N) softmax.
                mix = self._explicit(q, k, v, attn_mask, ctx, N)
            else:
                # Training/eval top-K: gather the K winners, dense read over them.
                mix = self._gathered(q, k, v, attn_mask, ctx, N, ctx.top_k)
        else:
            # Fused kernel: never materializes the (B, heads, T, N) score matrix,
            # which at v2 sizes (T~5k, N~25k) would be tens of GB per sample.
            mix = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            if ctx.log_stats or ctx.capture_maps:
                with torch.no_grad():
                    self._explicit(q, k, v, attn_mask, ctx, N)
        mix = mix.transpose(1, 2).reshape(B, T, self.n_heads * self.rank)
        return self.mem_o(mix)

    def _gathered(self, q, k, v, attn_mask, ctx, N, K, chunk=512):
        """Top-K read without the full score matrix: scores are computed in
        query chunks under no_grad only to pick the K indices per query; the
        differentiable read is a dense softmax over the gathered K keys/values
        (exactly the deployment MIPS read). Memory is O(T*K*r) per site
        instead of O(T*N), so it scales to v2 banks (T~8k, N~25k)."""
        B, H, T, r = q.shape
        scale = 1.0 / math.sqrt(r)
        mb = attn_mask[:, 0, 0, :] if attn_mask is not None else None       # (B, N) bool
        with torch.no_grad():
            kt = k.transpose(-1, -2)
            idx = torch.empty(B, H, T, K, dtype=torch.long, device=q.device)
            for t0 in range(0, T, chunk):
                s = torch.matmul(q[:, :, t0:t0 + chunk], kt) * scale         # (B, H, c, N)
                if mb is not None:
                    s = s.masked_fill(~mb[:, None, None, :], torch.finfo(s.dtype).min)
                idx[:, :, t0:t0 + chunk] = s.topk(K, dim=-1).indices
        # Gather K keys/values per query: flat index into (B*H*N, r).
        base = (torch.arange(B * H, device=q.device) * N).view(B, H, 1, 1)
        flat = (idx + base).reshape(-1)
        kk = k.reshape(B * H * N, r).index_select(0, flat).view(B, H, T, K, r)
        vv = v.reshape(B * H * N, r).index_select(0, flat).view(B, H, T, K, r)
        s = (q.unsqueeze(3) * kk).sum(-1) * scale                            # (B, H, T, K)
        if mb is not None:
            mg = mb.gather(1, idx.reshape(B, -1)).view(B, H, T, K)
            s = s.masked_fill(~mg, torch.finfo(s.dtype).min)
        p = F.softmax(s.float(), dim=-1).to(v.dtype)
        if ctx.log_stats:
            with torch.no_grad():
                pf = p.float()
                ent = -(pf * (pf + 1e-9).log()).sum(-1).mean(dim=(1, 2))    # (B,)
                n = mb.float().sum(-1).clamp(min=2) if mb is not None \
                    else torch.full_like(ent, N)
                am = idx.gather(-1, pf.argmax(-1, keepdim=True)).float()
                ctx.stats.append({"layer": self.layer_idx,
                                  "entropy": (ent / n.log()).mean().item(),
                                  "argmax_mean_pos": am.mean().item()})
        return (p.unsqueeze(-1) * vv).sum(3)                                 # (B, H, T, r)

    def _explicit(self, q, k, v, attn_mask, ctx, N):
        """Reference read with the full softmax materialized (stats, maps, top-K)."""
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.rank)  # (B, heads, T, N)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        if ctx.top_k is not None and ctx.top_k < N:
            kth = scores.topk(ctx.top_k, dim=-1).values[..., -1:]
            scores = scores.masked_fill(scores < kth, torch.finfo(scores.dtype).min)
        p = F.softmax(scores.float(), dim=-1).to(v.dtype)  # (B, heads, T, N)
        if ctx.log_stats:
            with torch.no_grad():
                pf = p.float()
                ent = -(pf * (pf + 1e-9).log()).sum(-1).mean(dim=(1, 2))  # (B,)
                n = attn_mask[:, 0, 0, :].float().sum(-1).clamp(min=2) if attn_mask is not None \
                    else torch.full_like(ent, N)
                # normalized: 1.0 = uniform over the sample's N memories, ->0 = sharp
                ctx.stats.append({"layer": self.layer_idx,
                                  "entropy": (ent / n.log()).mean().item(),
                                  "argmax_mean_pos": pf.argmax(-1).float().mean().item()})
        if ctx.capture_maps:
            ctx.maps.append({"layer": self.layer_idx, "p": p.detach().float().cpu()})
        return torch.matmul(p, v)  # (B, heads, T, r)


class MemoryLayerWrapper(nn.Module):
    """Transparent wrapper around a Qwen3_5DecoderLayer.

    Captures writes at the write site and adds read output into the residual
    stream at read sites. The outer model loop only touches `.layer_type`.
    """

    def __init__(self, inner: nn.Module, ctx: MemoryContext,
                 reader: MemoryReader | None = None, is_write_site: bool = False,
                 above_write_site: bool = False):
        super().__init__()
        self.inner = inner
        self.reader = reader
        self.is_write_site = is_write_site
        self.above_write_site = above_write_site
        object.__setattr__(self, "ctx", ctx)  # plain attr: contexts hold tensors, not params

    @property
    def layer_type(self):
        return self.inner.layer_type

    def forward(self, hidden_states, *args, **kwargs):
        ctx = self.ctx
        if ctx.write_only and self.above_write_site:
            return hidden_states  # nothing above the write site affects the writes
        h_in = hidden_states
        out = self.inner(hidden_states, *args, **kwargs)
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
                                            is_write_site=(i == write_layer),
                                            above_write_site=(i > write_layer))
    return ctx, nn.ModuleList(readers)


def reader_parameters(model):
    for m in model.modules():
        if isinstance(m, MemoryReader):
            yield from m.parameters()
