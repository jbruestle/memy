"""Memy v1 training loop. See DESIGN.md.

Per batch:
  1. Teacher (LoRA disabled, reads bypassed) GENERATES the agent turn from the
     user question with normal attention; raw per-step logits captured.
  2. Student pass 1: user-turn chunk alone; post-layer-19 normalized residuals
     become the memory (gradients flow through).
  3. Student pass 2: fresh state, assistant opening + teacher tokens forced;
     reads attend over memory. Loss = exact full-vocab KL(teacher || student)
     over the teacher's generated tokens.

Arms: L0 = no memory (pass 1 skipped, reads off), L1 = last-token memory only,
L2 = full per-token memory.
"""

import argparse
import json
import os
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from data import make_probes, score_probe, user_questions
from model import install_memory, reader_parameters

MODEL = "Qwen/Qwen3.5-4B"

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "out_proj"]


def build_model(args):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    ctx, _readers = install_memory(model, write_layer=args.write_layer,
                                   n_heads=args.read_heads, rank=args.read_rank)
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                      target_modules=LORA_TARGETS, bias="none")
    model = get_peft_model(model, lora)
    for p in reader_parameters(model):
        p.requires_grad_(True)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()  # required: HF only checkpoints in train mode (all dropouts are 0)
    return tok, model, ctx


@contextmanager
def teacher_mode(model, ctx):
    prev = ctx.reads_enabled
    was_training = model.training
    ctx.reads_enabled = False
    model.eval()  # train mode + grad ckpt forces use_cache=False, breaking generate
    try:
        with model.disable_adapter(), torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()
        ctx.reads_enabled = prev


def encode(tok, questions):
    """Returns per-sample user-chunk ids and the (shared) assistant opening ids."""
    user_ids, open_ids = [], None
    for q in questions:
        msgs = [{"role": "user", "content": q}]
        u = tok.apply_chat_template(msgs, tokenize=True,
                                    add_generation_prompt=False)["input_ids"]
        full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       enable_thinking=False)["input_ids"]
        assert full[:len(u)] == u, "chat template lost prefix property"
        user_ids.append(u)
        open_ids = full[len(u):]  # identical for every sample by construction
    return user_ids, open_ids


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


class _KLSum(torch.autograd.Function):
    """sum_t KL(p_t || p_s), exact full-vocab, fp32 math chunked over tokens.

    Analytic backward (grad_s = p_s - p_t) recomputed from bf16 logits, so
    autograd never retains fp32 log-softmax buffers (~1.3GB saved per step).
    """

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


def kl_chunked(t_logits, s_logits):
    return _KLSum.apply(s_logits, t_logits)


def training_step(model, tok, ctx, args, questions, want_stats=False, fixed_gen=None):
    """fixed_gen: optional per-question target token lists (skips teacher
    generation; used for deterministic, cross-arm-comparable eval)."""
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    user_ids, open_ids = encode(tok, questions)
    prompt = [u + open_ids for u in user_ids]

    # 1. Teacher generation (normal attention over the full prompt).
    if fixed_gen is not None:
        gen_list = fixed_gen
    else:
        pids, pmask = pad_batch(prompt, pad_id, "left", device)
        with teacher_mode(model, ctx):
            out = model.generate(input_ids=pids, attention_mask=pmask,
                                 max_new_tokens=args.max_gen, do_sample=True,
                                 temperature=0.7, top_p=0.8, top_k=20,
                                 use_cache=True, pad_token_id=pad_id)
        gen = out[:, pids.shape[1]:]                            # (B, Tg)
        del out
        eos = tok.eos_token_id
        gen_list = []
        for i in range(gen.shape[0]):
            hits = (gen[i] == eos).nonzero()
            g = int(hits[0]) + 1 if len(hits) else gen.shape[1]
            gen_list.append(gen[i, :g].tolist())

    # Teacher logits from one forced forward (bf16, right-padded, no_grad) —
    # far cheaper in memory than accumulating per-step logits during generation.
    t_full = [prompt[i] + gen_list[i] for i in range(len(prompt))]
    tids, tmask = pad_batch(t_full, pad_id, "right", device)
    with teacher_mode(model, ctx):
        t_out = model(input_ids=tids, attention_mask=tmask, use_cache=False).logits
    t_logits = [t_out[i, len(prompt[i]) - 1:len(prompt[i]) - 1 + len(gen_list[i])].clone()
                for i in range(len(prompt))]                    # per-sample (g, V)
    del t_out
    torch.cuda.empty_cache()

    # 2. Student pass 1: write memory (skipped for L0).
    ctx.clear()
    if args.arm != "L0":
        uids, umask = pad_batch(user_ids, pad_id, "right", device)
        ctx.collect_writes, ctx.reads_enabled = True, False
        model(input_ids=uids, attention_mask=umask, use_cache=False)
        ctx.collect_writes = False
        mem, mem_mask = ctx.written, umask
        if args.arm == "L1":  # last real token only
            idx = (umask.sum(1) - 1).view(-1, 1, 1).expand(-1, 1, mem.shape[-1])
            mem = mem.gather(1, idx)
            mem_mask = torch.ones(mem.shape[:2], dtype=torch.long, device=device)
        ctx.memory, ctx.memory_mask = mem, mem_mask

    # 3. Student pass 2: teacher-forced agent turn with fresh state.
    forced = [open_ids + gen_list[i] for i in range(len(user_ids))]
    sids, smask = pad_batch(forced, pad_id, "right", device)
    ctx.reads_enabled = args.arm != "L0"
    ctx.log_stats = want_stats
    s_out = model(input_ids=sids, attention_mask=smask, use_cache=False)
    ctx.log_stats = False

    # 4. Exact KL over each sample's generated tokens.
    n_open = len(open_ids)
    loss_sum, n_tok = None, 0
    for i in range(len(user_ids)):
        g = len(gen_list[i])
        s_slice = s_out.logits[i, n_open - 1:n_open - 1 + g]
        k = kl_chunked(t_logits[i], s_slice)
        loss_sum = k if loss_sum is None else loss_sum + k
        n_tok += g
    loss = loss_sum / max(n_tok, 1)
    stats = list(ctx.stats)
    ctx.clear()
    return loss, n_tok, stats, gen_list


@torch.no_grad()
def run_probes(model, tok, ctx, args, probes):
    """Free generation on probe questions through the memory bottleneck."""
    if args.arm == "L0":
        return {"recall": 0.0, "note": "L0 has no memory; probes trivially fail"}, []
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    per_key, n, generations = {}, 0, []
    for i in range(0, len(probes), args.batch):
        chunk = probes[i:i + args.batch]
        user_ids, open_ids = encode(tok, [p["question"] for p in chunk])
        ctx.clear()
        uids, umask = pad_batch(user_ids, pad_id, "right", device)
        ctx.collect_writes, ctx.reads_enabled = True, False
        model(input_ids=uids, attention_mask=umask, use_cache=False)
        ctx.collect_writes = False
        ctx.memory, ctx.memory_mask = ctx.written, umask
        ctx.reads_enabled = True
        oids, omask = pad_batch([open_ids] * len(chunk), pad_id, "left", device)
        out = model.generate(input_ids=oids, attention_mask=omask,
                             max_new_tokens=args.max_gen, do_sample=False,
                             use_cache=True, pad_token_id=pad_id)
        texts = tok.batch_decode(out[:, oids.shape[1]:], skip_special_tokens=True)
        for p, txt in zip(chunk, texts):
            hits = score_probe(txt, p["bindings"])
            for k, hit in hits.items():
                per_key[k] = per_key.get(k, 0) + int(hit)
            generations.append({"bindings": p["bindings"], "hits": hits, "text": txt})
            n += 1
        ctx.clear()
    scores = {k: v / n for k, v in per_key.items()}
    scores["recall"] = sum(scores.values()) / len(scores)
    scores["sample"] = generations[0]["text"][:400]
    return scores, generations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["L0", "L1", "L2"], default="L2")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-user-tokens", type=int, default=320)
    ap.add_argument("--max-gen", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--read-heads", type=int, default=4)
    ap.add_argument("--read-rank", type=int, default=128)
    ap.add_argument("--write-layer", type=int, default=19)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.set_defaults(grad_ckpt=True)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--n-probes", type=int, default=32)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = one full epoch")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    run = args.run_name or f"{args.arm}-{time.strftime('%m%d-%H%M')}"
    outdir = os.path.join("runs", run)
    os.makedirs(outdir, exist_ok=True)
    logf = open(os.path.join(outdir, "log.jsonl"), "a")

    tok, model, ctx = build_model(args)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)))

    probes = make_probes(n=args.n_probes, path=os.path.join(outdir, "probes.jsonl"))
    stream = user_questions(tok, args.max_user_tokens)
    eval_qs = [next(stream) for _ in range(args.n_eval)]  # held out from training

    # Eval targets are teacher generations cached ONCE, shared across arms so
    # every arm's eval_kl is scored on identical token sequences.
    tgt_path = os.path.join("runs", "eval-targets.json")
    eval_gen = json.load(open(tgt_path)) if os.path.exists(tgt_path) else None

    step, batch_qs = 0, []
    for q in stream:
        batch_qs.append(q)
        if len(batch_qs) < args.batch:
            continue
        t0 = time.time()
        want_stats = step % 50 == 0
        loss, n_tok, stats, _ = training_step(model, tok, ctx, args, batch_qs, want_stats)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        rec = {"step": step, "kl": float(loss.detach()), "tokens": n_tok,
               "sec": round(time.time() - t0, 2)}
        if stats:
            rec["read_entropy"] = round(sum(s["entropy"] for s in stats) / len(stats), 3)
        print(rec, flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()

        if step > 0 and step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                ev, ev_tok = 0.0, 0
                first_time, collected = eval_gen is None, []
                for j in range(0, len(eval_qs), args.batch):
                    fg = None if first_time else eval_gen[j:j + args.batch]
                    l, nt, _, gl = training_step(model, tok, ctx, args,
                                                 eval_qs[j:j + args.batch],
                                                 fixed_gen=fg)
                    collected += gl
                    ev += float(l) * nt; ev_tok += nt
                if first_time:
                    eval_gen = collected
                    json.dump(eval_gen, open(tgt_path, "w"))
            pr, gens = run_probes(model, tok, ctx, args, probes)
            if gens:
                with open(os.path.join(outdir, f"probe_gen-{step}.jsonl"), "w") as pf:
                    for g in gens:
                        pf.write(json.dumps(g) + "\n")
            model.train()
            rec = {"step": step, "eval_kl": ev / max(ev_tok, 1),
                   "probe": {k: v for k, v in pr.items() if k != "sample"}}
            print(rec, flush=True); print("probe sample:", pr.get("sample", "")[:200], flush=True)
            logf.write(json.dumps({**rec, "probe_sample": pr.get("sample", "")}) + "\n")
            logf.flush()

        if step > 0 and step % args.save_interval == 0:
            model.save_pretrained(os.path.join(outdir, f"ckpt-{step}"))
            torch.save({k: v for k, v in model.state_dict().items() if "mem_" in k},
                       os.path.join(outdir, f"ckpt-{step}", "readers.pt"))
        step += 1
        batch_qs = []
        if args.max_steps and step >= args.max_steps:
            break
    print("done")


if __name__ == "__main__":
    main()
