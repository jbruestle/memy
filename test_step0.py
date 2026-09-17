"""Step-0 sanity: surgery + PEFT must leave the model bit-identical to base.

Checks, on a fixed prompt:
  1. teacher path (adapters disabled, reads bypassed) == student path with
     adapters enabled, reads enabled, and RANDOM memory installed — because
     LoRA B and mem_o are zero-initialized, these must match exactly.
  2. write capture produces one normalized vector per token.
  3. chat template prefix property + assistant opening look as expected.
"""

import torch
from types import SimpleNamespace

from train import build_model, encode, pad_batch, teacher_mode

ARGS = SimpleNamespace(write_layer=19, read_heads=4, read_rank=128,
                       lora_r=16, grad_ckpt=False)


def main():
    tok, model, ctx = build_model(ARGS)
    model.eval()
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    qs = ["What is the capital of France, and why is it famous?",
          "Explain photosynthesis to a five year old."]
    user_ids, open_ids = encode(tok, qs)
    print("assistant opening:", repr(tok.decode(open_ids)))
    ids, mask = pad_batch([u + open_ids for u in user_ids], pad_id, "right", device)

    with teacher_mode(model, ctx), torch.no_grad():
        ref = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.clone()

    # Student path: adapters ON, reads ON, random memory — must be identical.
    ctx.clear()
    ctx.memory = torch.randn(2, 37, model.config.text_config.hidden_size,
                             dtype=torch.bfloat16, device=device)
    ctx.memory_mask = torch.ones(2, 37, dtype=torch.long, device=device)
    ctx.reads_enabled = True
    with torch.no_grad():
        stu = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    ctx.clear()
    same = torch.equal(ref, stu)
    print(f"identity check: {'PASS' if same else 'FAIL'} "
          f"(max abs diff {(ref - stu).abs().max().item():.3e})")

    # Write capture.
    uids, umask = pad_batch(user_ids, pad_id, "right", device)
    ctx.collect_writes = True
    with torch.no_grad():
        model(input_ids=uids, attention_mask=umask, use_cache=False)
    ctx.collect_writes = False
    w = ctx.written
    rms = w.float().pow(2).mean(-1).sqrt()
    print(f"writes: shape {tuple(w.shape)}, rms mean {rms.mean():.4f} "
          f"(want ~1.0), {'PASS' if w.shape[:2] == uids.shape else 'FAIL'}")
    ctx.clear()

    n_read = sum(1 for m in model.modules() if type(m).__name__ == "MemoryReader")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"read sites: {n_read} (want 8); trainable {n_trainable/1e6:.1f}M")
    assert same, "logits diverged at step 0"


if __name__ == "__main__":
    main()
