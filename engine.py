"""Memy v2 engine: source-agnostic execution of Samples. See TRAINING_V2.md
("Harness architecture").

A Sample is a dict:
    id          stable key (eval-target cache, resume, logs)
    source      dataset / probe name
    gold        [passage, ...]   in the bank AND in the teacher context
    distractors [passage, ...]   in the bank only
    teacher_gold optional [passage, ...]: what the teacher sees INSTEAD of
                gold (e.g. Qasper: bank holds the whole paper, the teacher
                gets only the annotated evidence paragraphs)
    turns       [(role, text), ...]  alternating, ending on the target's user turn
    meta        anything the source / probe wants back (answer, bindings, ...)

The engine turns a Sample into chunks (background passages, tagged history
turns, target), runs them as separate forward passes with the bank as the
only cross-chunk channel, and derives the teacher transcript (gold folded
into the first user turn, untagged, ordinary attention).
"""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from model import install_memory, reader_parameters

MODEL = "Qwen/Qwen3.5-4B"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "out_proj"]
READ_WRAP = "Please read the following:\n"
GEN_SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)


# ----------------------------------------------------------------- model

def build_model(args):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    ctx, _readers = install_memory(model, write_layer=args.write_layer,
                                   n_heads=args.read_heads, rank=args.read_rank)
    if args.no_lora:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in reader_parameters(model):
            p.requires_grad_(True)
    else:
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                          target_modules=LORA_TARGETS, bias="none")
        model = get_peft_model(model, lora)
        for p in reader_parameters(model):
            p.requires_grad_(True)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if not args.no_lora:
            model.enable_input_require_grads()
    model.train()  # required: HF only checkpoints in train mode (all dropouts are 0)
    return tok, model, ctx


def base_of(model):
    """The Qwen3_5ForConditionalGeneration under an optional PEFT wrapper."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


@contextmanager
def teacher_mode(model, ctx):
    """LoRA off, reads off, eval mode (generation needs use_cache), no grad."""
    prev = ctx.reads_enabled
    was_training = model.training
    ctx.reads_enabled = False
    model.eval()
    adapter_off = model.disable_adapter() if hasattr(model, "disable_adapter") \
        else nullcontext()
    try:
        with adapter_off, torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()
        ctx.reads_enabled = prev


@contextmanager
def eval_mode(model):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()


def pad_batch(seqs, pad_id, side, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        t = torch.tensor(s, dtype=torch.long)
        if side == "left":
            ids[i, L - len(s):] = t; mask[i, L - len(s):] = 1
        else:
            ids[i, :len(s)] = t; mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


def trim_eos(gen, eos):
    """(B, Tg) generated ids -> per-sample lists cut after the first eos."""
    out = []
    for i in range(gen.shape[0]):
        hits = (gen[i] == eos).nonzero()
        g = int(hits[0]) + 1 if len(hits) else gen.shape[1]
        out.append(gen[i, :g].tolist())
    return out


# ----------------------------------------------------------------- loss

class _KLSum(torch.autograd.Function):
    """sum_t KL(p_t || p_s), exact full-vocab, fp32 math chunked over tokens.
    Analytic backward (grad_s = p_s - p_t) recomputed from bf16 logits, so
    autograd never retains fp32 log-softmax buffers."""

    CHUNK = 64

    @staticmethod
    def forward(ctx, s_logits, t_logits):
        total = torch.zeros((), dtype=torch.float32, device=s_logits.device)
        for i in range(0, s_logits.shape[0], _KLSum.CHUNK):
            tl = F.log_softmax(t_logits[i:i + _KLSum.CHUNK].float(), dim=-1)
            sl = F.log_softmax(s_logits[i:i + _KLSum.CHUNK].float(), dim=-1)
            total += (tl.exp() * (tl - sl)).sum()
        ctx.save_for_backward(s_logits, t_logits)
        return total

    @staticmethod
    def backward(ctx, grad_out):
        s_logits, t_logits = ctx.saved_tensors
        grad = torch.empty_like(s_logits)
        for i in range(0, s_logits.shape[0], _KLSum.CHUNK):
            ps = F.softmax(s_logits[i:i + _KLSum.CHUNK].float(), dim=-1)
            pt = F.softmax(t_logits[i:i + _KLSum.CHUNK].float(), dim=-1)
            grad[i:i + _KLSum.CHUNK] = ((ps - pt) * grad_out).to(s_logits.dtype)
        return grad, None


def kl_sum(t_logits, s_logits):
    return _KLSum.apply(s_logits, t_logits)


# ----------------------------------------------------------------- encoding

@dataclass
class Encoded:
    sample: dict
    bg: list        # background chunk ids: gold first, then distractors
    n_gold: int
    turns: list     # history chunk ids: user0, agent0, ..., user_t (tagged if enabled)
    prefix: list    # target-chunk ids after the assistant opening (tag or empty)
    prompt: list    # teacher transcript ids, ends with the assistant opening
    tokens: int     # total student chunk tokens (batching cost)
    depth: int      # number of history chunks


class Encoder:
    def __init__(self, tok, turn_tags: bool, max_chunk_tokens: int = 0,
                 max_prompt_tokens: int = 0):
        self.tok, self.turn_tags = tok, turn_tags
        self.max_chunk, self.max_prompt = max_chunk_tokens, max_prompt_tokens
        u = self.user_chunk("x")
        full = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=True,
                                       add_generation_prompt=True,
                                       enable_thinking=False)["input_ids"]
        assert full[:len(u)] == u, "chat template lost prefix property"
        self.open_ids = full[len(u):]          # <|im_start|>assistant\n<think>\n\n</think>\n\n
        self.close_ids = tok("<|im_end|>\n", add_special_tokens=False).input_ids
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.eos_id = tok.eos_token_id

    def user_chunk(self, text):
        return self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                            add_generation_prompt=False)["input_ids"]

    def agent_chunk(self, text):
        return self.open_ids + self.tok(text, add_special_tokens=False).input_ids + self.close_ids

    def encode(self, sample) -> Encoded | None:
        """None => sample exceeds a cap (dropped)."""
        turns = sample["turns"]
        assert turns and turns[-1][0] == "user", "turns must end on the target's user turn"
        gold, dis = list(sample.get("gold", [])), list(sample.get("distractors", []))
        bg = [self.user_chunk(READ_WRAP + p) for p in gold + dis]
        hist, nu, na = [], 0, 0
        for role, text in turns:
            if role == "user":
                t = f"[user turn {nu}]:\n{text}" if self.turn_tags else text
                hist.append(self.user_chunk(t)); nu += 1
            else:
                t = f"[agent turn {na}]:\n{text}" if self.turn_tags else text
                hist.append(self.agent_chunk(t)); na += 1
        prefix = self.tok(f"[agent turn {na}]:\n", add_special_tokens=False).input_ids \
            if self.turn_tags else []
        msgs = [{"role": r, "content": t} for r, t in turns]
        tgold = sample.get("teacher_gold", gold)
        if tgold:
            msgs[0]["content"] = READ_WRAP + "\n\n".join(tgold) + "\n\n" + msgs[0]["content"]
        prompt = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                              enable_thinking=False)["input_ids"]
        chunks = bg + hist
        if self.max_chunk and max(len(c) for c in chunks) > self.max_chunk:
            return None
        if self.max_prompt and len(prompt) > self.max_prompt:
            return None
        return Encoded(sample, bg, len(gold), hist, prefix, prompt,
                       sum(len(c) for c in chunks), len(hist))


# ----------------------------------------------------------------- passes

def stack_banks(banks):
    """banks: per-sample list of (T,H) write tensors. Returns padded
    (mem (B,N,H), mask (B,N)) or (None, None) if every bank is empty. An
    empty bank among non-empty ones gets one zero memory: its read output
    is exactly zero (softmax over one element times mem_v(0))."""
    cats = [torch.cat(b, 0) if b else None for b in banks]
    if all(c is None for c in cats):
        return None, None
    ref = next(c for c in cats if c is not None)
    cats = [c if c is not None else torch.zeros(1, ref.shape[-1], dtype=ref.dtype,
                                                device=ref.device) for c in cats]
    mem = pad_sequence(cats, batch_first=True)
    mask = pad_sequence([torch.ones(c.shape[0], dtype=torch.long, device=ref.device)
                         for c in cats], batch_first=True)
    return mem, mask


def write_pass(model, ctx, chunks, banks, pad_id, grad):
    """Forward chunks (token-id lists) with fresh state, reading over `banks`
    (per-chunk list of tensors, or None), and return each chunk's (T,H)
    writes. Layers above the write site are skipped; no LM head."""
    device = next(model.parameters()).device
    ids, mask = pad_batch(chunks, pad_id, "right", device)
    mem, mmask = stack_banks(banks) if banks is not None else (None, None)
    ctx.memory, ctx.memory_mask = mem, mmask
    ctx.reads_enabled = mem is not None
    ctx.collect_writes, ctx.write_only = True, True
    try:
        with (nullcontext() if grad else torch.no_grad()):
            base_of(model).model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        ctx.collect_writes, ctx.write_only = False, False
        ctx.memory, ctx.memory_mask, ctx.reads_enabled = None, None, False
    w = ctx.written
    ctx.written = None
    return [w[i, :len(c)] for i, c in enumerate(chunks)]


def _pack(items, budget):
    """Greedy micro-batches: items (sorted longest first) of (.., ids, ..)
    such that count * max_len <= budget (always at least one item)."""
    out, cur = [], []
    for it in items:
        L = len(it[2])
        if cur and (len(cur) + 1) * len(cur[0][2]) > budget:
            out.append(cur); cur = []
        cur.append(it)
    if cur:
        out.append(cur)
    return out


def run_background(model, ctx, enc_list, args, pad_id, grad_ok):
    """All background chunks of the batch against an empty bank, micro-batched
    by token count. Gold + the first `distractor_grad_k` distractors of each
    sample keep the graph (if grad_ok); the rest run under no_grad.
    Returns per-sample lists of writes (in bg order)."""
    items = [(si, ci, ids, ci < e.n_gold + args.distractor_grad_k)
             for si, e in enumerate(enc_list) for ci, ids in enumerate(e.bg)]
    out = {}
    for grad in (True, False):
        sel = sorted((it for it in items if it[3] == grad), key=lambda it: -len(it[2]))
        for mb in _pack(sel, args.bg_tokens):
            ws = write_pass(model, ctx, [it[2] for it in mb], None, pad_id,
                            grad and grad_ok and not args.detach_writes)
            for it, w in zip(mb, ws):
                out[(it[0], it[1])] = w
    return [[out[(si, ci)] for ci in range(len(e.bg))] for si, e in enumerate(enc_list)]


def build_banks(model, ctx, enc_list, args, pad_id, grad_ok):
    """Background passes, then the history turns in order. Returns per-sample
    banks (lists of (T,H) tensors)."""
    banks = run_background(model, ctx, enc_list, args, pad_id, grad_ok)
    depth = enc_list[0].depth
    assert all(e.depth == depth for e in enc_list), "batch must be depth-homogeneous"
    for j in range(depth):
        chunks = [e.turns[j] for e in enc_list]
        ws = write_pass(model, ctx, chunks, banks if any(banks) else None, pad_id,
                        grad_ok and not args.detach_writes)
        for b, w in zip(banks, ws):
            b.append(w)
    return banks


# ----------------------------------------------------------------- teacher

def teacher_generate(model, ctx, prompts, pad_id, eos_id, max_gen, greedy=False):
    device = next(model.parameters()).device
    ids, mask = pad_batch(prompts, pad_id, "left", device)
    with teacher_mode(model, ctx):
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_gen,
                             use_cache=True, pad_token_id=pad_id,
                             **(dict(do_sample=False) if greedy else GEN_SAMPLING))
    return trim_eos(out[:, ids.shape[1]:], eos_id)


def teacher_logits(model, ctx, prompts, gen_list, pad_id):
    """Per-sample (g, V) bf16 logits at the generated positions only."""
    device = next(model.parameters()).device
    full = [p + g for p, g in zip(prompts, gen_list)]
    ids, mask = pad_batch(full, pad_id, "right", device)
    base = base_of(model)
    with teacher_mode(model, ctx):
        h = base.model(input_ids=ids, attention_mask=mask, use_cache=False)[0]
        outs = [base.lm_head(h[i, len(p) - 1:len(p) - 1 + len(g)])
                for i, (p, g) in enumerate(zip(prompts, gen_list))]
    del h
    return outs


# ----------------------------------------------------------------- student

def student_loss(model, ctx, encoder, enc_list, gen_list, t_logits, args, want_stats,
                 grad_ok=True):
    """Bank construction + teacher-forced target chunk + exact KL.
    Returns (loss, n_tok, stats, info)."""
    device = next(model.parameters()).device
    pad_id = encoder.pad_id
    ctx.clear()
    banks = build_banks(model, ctx, enc_list, args, pad_id, grad_ok)
    tgt = [encoder.open_ids + e.prefix + g for e, g in zip(enc_list, gen_list)]
    ids, mask = pad_batch(tgt, pad_id, "right", device)
    mem, mmask = stack_banks(banks)
    ctx.memory, ctx.memory_mask = mem, mmask
    ctx.reads_enabled = mem is not None
    ctx.log_stats = want_stats
    base = base_of(model)
    with (nullcontext() if grad_ok else torch.no_grad()):
        h = base.model(input_ids=ids, attention_mask=mask, use_cache=False)[0]
        ctx.log_stats = False
        n0 = [len(encoder.open_ids) + len(e.prefix) for e in enc_list]
        sel = torch.cat([h[i, n0[i] - 1:n0[i] - 1 + len(g)] for i, g in enumerate(gen_list)], 0)
        logits = base.lm_head(sel)                       # (sum g, V)
        loss_sum, off, n_tok = None, 0, 0
        for i, g in enumerate(gen_list):
            k = kl_sum(t_logits[i], logits[off:off + len(g)])
            loss_sum = k if loss_sum is None else loss_sum + k
            off += len(g); n_tok += len(g)
        loss = loss_sum / max(n_tok, 1)
    stats = list(ctx.stats)
    info = {"bank": int(mmask.sum(1).float().mean()) if mmask is not None else 0,
            "depth": enc_list[0].depth,
            "bg": sum(len(e.bg) for e in enc_list) / len(enc_list)}
    ctx.clear()
    return loss, n_tok, stats, info


def student_generate(model, ctx, encoder, enc_list, args, max_gen):
    """Free generation from memory alone (greedy). Returns (token lists,
    mean normalized read entropy over the decode)."""
    device = next(model.parameters()).device
    pad_id = encoder.pad_id
    with eval_mode(model):
        ctx.clear()
        banks = build_banks(model, ctx, enc_list, args, pad_id, grad_ok=False)
        mem, mmask = stack_banks(banks)
        ctx.memory, ctx.memory_mask = mem, mmask
        ctx.reads_enabled = mem is not None
        opens = [encoder.open_ids + e.prefix for e in enc_list]
        ids, mask = pad_batch(opens, pad_id, "left", device)
        ctx.log_stats, ctx.stats = True, []
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_gen,
                             do_sample=False, use_cache=True, pad_token_id=pad_id)
        ctx.log_stats = False
        ent = sum(s["entropy"] for s in ctx.stats) / max(len(ctx.stats), 1)
        ctx.clear()
    return trim_eos(out[:, ids.shape[1]:], encoder.eos_id), ent


# ----------------------------------------------------------------- remote teacher

class RemoteTeacher:
    """Teacher generation on a llama.cpp server (`llama-server`), native
    /completion endpoint: the prompt goes in as token ids and the generated
    token ids come back, so no re-tokenization anywhere. Requests of one
    batch are sent concurrently; the server does continuous batching."""

    def __init__(self, url, eos_id, max_conc=16, timeout=600):
        import concurrent.futures as cf
        self.url, self.eos_id, self.timeout = url.rstrip("/"), eos_id, timeout
        self.pool = cf.ThreadPoolExecutor(max_conc)

    def _post(self, path, payload, retry_for=1800):
        """POST with retries: a server restart (weights swap, more slots) must
        not kill a training run. Retries for up to `retry_for` seconds."""
        import json as _json
        import time as _time
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.url + path, data=_json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = _time.time()
        while True:
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return _json.loads(r.read())
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                if _time.time() - t0 > retry_for:
                    raise
                print(f"[remote teacher] {e}; retrying", flush=True)
                _time.sleep(5)

    def tokenize(self, text):
        return self._post("/tokenize", {"content": text})["tokens"]

    def _one(self, prompt, max_gen, greedy):
        payload = {"prompt": prompt, "n_predict": max_gen, "return_tokens": True,
                   "cache_prompt": False, "min_p": 0.0, "repeat_penalty": 1.0}
        if greedy:
            payload.update(temperature=0.0, top_k=1)
        else:
            payload.update(temperature=GEN_SAMPLING["temperature"], top_p=GEN_SAMPLING["top_p"],
                           top_k=GEN_SAMPLING["top_k"])
        r = self._post("/completion", payload)
        toks = list(r["tokens"])
        stopped_eos = r.get("stop_type") == "eos" or r.get("stopped_eos", False)
        if stopped_eos and (not toks or toks[-1] != self.eos_id):
            toks.append(self.eos_id)          # match HF: the target includes <|im_end|>
        return toks[:max_gen]

    def generate(self, prompts, max_gen, greedy=False):
        futs = [self.pool.submit(self._one, p, max_gen, greedy) for p in prompts]
        return [f.result() for f in futs]

    def submit(self, prompts, max_gen, greedy=False):
        """Non-blocking: returns a future whose .result() is the token lists."""
        return self.pool.submit(self.generate, prompts, max_gen, greedy)


class _Done:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class Teacher:
    """Uniform front for local (inline HF generate) or remote generation.
    `submit` is non-blocking for the remote teacher (prefetch), immediate
    for the local one."""

    def __init__(self, model, ctx, encoder, args):
        self.model, self.ctx, self.encoder, self.args = model, ctx, encoder, args
        self.remote = RemoteTeacher(args.teacher_url, encoder.eos_id) if args.teacher_url else None

    def generate(self, prompts, greedy=False):
        if self.remote:
            return self.remote.generate(prompts, self.args.max_gen, greedy)
        return teacher_generate(self.model, self.ctx, prompts, self.encoder.pad_id,
                                self.encoder.eos_id, self.args.max_gen, greedy)

    def submit(self, prompts, greedy=False):
        if self.remote:
            return self.remote.submit(prompts, self.args.max_gen, greedy)
        return _Done(self.generate(prompts, greedy))
