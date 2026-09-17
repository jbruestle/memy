"""One instrumented training step: memory watermark per phase."""

import sys
import torch
from types import SimpleNamespace

import train as T


def mark(tag):
    torch.cuda.synchronize()
    print(f"{tag:24s} alloc {torch.cuda.memory_allocated()/1e9:6.2f} GB  "
          f"peak {torch.cuda.max_memory_allocated()/1e9:6.2f} GB")
    torch.cuda.reset_peak_memory_stats()


def main():
    args = SimpleNamespace(arm="L2", batch=4, max_user_tokens=320, max_gen=300,
                           lora_r=16, read_heads=4, read_rank=128, write_layer=19,
                           grad_ckpt=True)
    tok, model, ctx = T.build_model(args)
    mark("model loaded")
    stream = T.user_questions(tok, args.max_user_tokens)
    qs = [next(stream) for _ in range(args.batch)]

    device = next(model.parameters()).device
    pad_id = tok.pad_token_id
    user_ids, open_ids = T.encode(tok, qs)
    prompt = [u + open_ids for u in user_ids]
    pids, pmask = T.pad_batch(prompt, pad_id, "left", device)
    with T.teacher_mode(model, ctx):
        out = model.generate(input_ids=pids, attention_mask=pmask,
                             max_new_tokens=args.max_gen, do_sample=True,
                             temperature=0.7, top_p=0.8, top_k=20,
                             use_cache=True, pad_token_id=pad_id)
    mark("teacher generate")
    gen = out[:, pids.shape[1]:]
    del out
    eos = tok.eos_token_id
    gen_len = [int((gen[i] == eos).nonzero()[0]) + 1 if (gen[i] == eos).any()
               else gen.shape[1] for i in range(gen.shape[0])]
    t_full = [prompt[i] + gen[i, :gen_len[i]].tolist() for i in range(len(prompt))]
    tids, tmask = T.pad_batch(t_full, pad_id, "right", device)
    with T.teacher_mode(model, ctx):
        t_out = model(input_ids=tids, attention_mask=tmask, use_cache=False).logits
    mark("teacher forward")
    print("   t_out dtype:", t_out.dtype, "shape:", tuple(t_out.shape))
    t_logits = [t_out[i, len(prompt[i]) - 1:len(prompt[i]) - 1 + gen_len[i]].clone()
                for i in range(len(prompt))]
    del t_out
    torch.cuda.empty_cache()
    mark("t_logits sliced")

    uids, umask = T.pad_batch(user_ids, pad_id, "right", device)
    ctx.clear()
    ctx.collect_writes, ctx.reads_enabled = True, False
    model(input_ids=uids, attention_mask=umask, use_cache=False)
    ctx.collect_writes = False
    ctx.memory, ctx.memory_mask = ctx.written, umask
    mark("student pass 1")

    forced = [open_ids + gen[i, :gen_len[i]].tolist() for i in range(len(user_ids))]
    sids, smask = T.pad_batch(forced, pad_id, "right", device)
    ctx.reads_enabled = True
    s_out = model(input_ids=sids, attention_mask=smask, use_cache=False)
    mark("student pass 2")
    print("   s_logits dtype:", s_out.logits.dtype, "shape:", tuple(s_out.logits.shape))

    n_open = len(open_ids)
    loss_sum, n_tok = None, 0
    for i in range(len(user_ids)):
        g = gen_len[i]
        k = T.kl_chunked(t_logits[i], s_out.logits[i, n_open - 1:n_open - 1 + g])
        loss_sum = k if loss_sum is None else loss_sum + k
        n_tok += g
    loss = loss_sum / max(n_tok, 1)
    mark("loss computed")
    loss.backward()
    mark("backward")
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    opt.step()
    mark("optimizer step")


if __name__ == "__main__":
    main()
