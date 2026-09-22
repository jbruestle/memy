"""Engine smoke test (needs the GPU):
  1. v1 `train.training_step` vs v2 `engine.student_loss` on the same batch
     and teacher tokens, tags off: losses must agree to bf16 noise.
  2. A synthetic multi-chunk sample (gold + distractors + 3-turn history),
     tags on: forward/backward runs, gradients reach the read heads and the
     background-chunk writes, and student_generate works.
"""

from types import SimpleNamespace

import torch

import train as v1
from engine import Encoder, build_model, student_generate, student_loss, teacher_logits

ARGS = SimpleNamespace(write_layer=19, read_heads=4, read_rank=128, lora_r=16, grad_ckpt=True,
                       no_lora=False, detach_writes=False, distractor_grad_k=2, bg_tokens=4096,
                       arm="L2", max_gen=32, batch=2)


def main():
    tok, model, ctx = build_model(ARGS)
    with torch.no_grad():  # make reads non-trivial so the comparison is meaningful
        for n, p in model.named_parameters():
            if "mem_o" in n:
                p.normal_(0, 0.02)
    enc = Encoder(tok, turn_tags=False)
    qs = ["My cat is named Bartholomew and he is 7. What should I feed him?",
          "Explain photosynthesis to a five year old."]
    samples = [{"id": f"t{i}", "source": "test", "gold": [], "distractors": [],
                "turns": [("user", q)], "meta": {}} for i, q in enumerate(qs)]
    encs = [enc.encode(s) for s in samples]
    gens = [tok("Sure! Here is a short answer for you.", add_special_tokens=False).input_ids + [tok.eos_token_id],
            tok("Plants eat sunlight.", add_special_tokens=False).input_ids + [tok.eos_token_id]]

    # --- 1. v1 vs v2 loss
    l1, n1, _, _ = v1.training_step(model, tok, ctx, ARGS, qs, fixed_gen=gens)
    l1 = float(l1.detach())
    tl = teacher_logits(model, ctx, [e.prompt for e in encs], gens, enc.pad_id)
    l2, n2, _, _ = student_loss(model, ctx, enc, encs, gens, tl, ARGS, want_stats=False)
    l2 = float(l2.detach())
    print(f"v1 loss {l1:.5f} ({n1} tok)  v2 loss {l2:.5f} ({n2} tok)  "
          f"rel diff {abs(l1 - l2) / max(l1, 1e-9):.2e}")
    assert n1 == n2 and abs(l1 - l2) / max(l1, 1e-9) < 2e-2, "v1/v2 loss mismatch"

    # --- 2. multi-chunk, tags on
    enc = Encoder(tok, turn_tags=True)
    sample = {"id": "mc", "source": "test",
              "gold": ["The Zorblatt festival is held every March in Tarnow."],
              "distractors": ["Kagoshima is known for its active volcano.",
                              "The theremin was invented in 1920.",
                              "Ostrava has a long industrial history."],
              "turns": [("user", "I'm planning a trip. Any ideas?"),
                        ("assistant", "Sure, where are you thinking of going?"),
                        ("user", "When is the Zorblatt festival held, and where?")],
              "meta": {}}
    e = enc.encode(sample)
    print("chunks: bg", len(e.bg), "turns", len(e.turns), "prefix", tok.decode(e.prefix).encode(),
          "tokens", e.tokens)
    print("history turn 1:", repr(tok.decode(e.turns[1])))
    gens = [tok("It is held every March in Tarnow.", add_special_tokens=False).input_ids + [tok.eos_token_id]]
    tl = teacher_logits(model, ctx, [e.prompt], gens, enc.pad_id)
    loss, n, stats, info = student_loss(model, ctx, enc, [e], gens, tl, ARGS, want_stats=True)
    loss.backward()
    print(f"multi-chunk loss {float(loss):.4f} info {info} entropy "
          f"{sum(s['entropy'] for s in stats) / len(stats):.3f}")
    gq = [p.grad for n_, p in model.named_parameters() if "mem_q" in n_ and p.grad is not None]
    gl = [p.grad for n_, p in model.named_parameters() if "lora_B" in n_ and p.grad is not None]
    assert gq and any(g.abs().sum() > 0 for g in gq), "no gradient reached read queries"
    assert gl and any(g.abs().sum() > 0 for g in gl), "no gradient reached LoRA"
    model.zero_grad(set_to_none=True)
    out, ent = student_generate(model, ctx, enc, [e], ARGS, max_gen=24)
    print("student generation:", repr(tok.decode(out[0], skip_special_tokens=True)), f"ent {ent:.2f}")

    # --- 3. generate() must not poison later forwards of a different batch size
    # (Qwen3.5 caches rope_deltas per batch; engine clears it).
    from engine import write_pass
    ws = write_pass(model, ctx, [e.bg[0], e.bg[1], e.turns[0]], None, enc.pad_id, grad=False)
    assert len(ws) == 3 and all(w.shape[-1] == ws[0].shape[-1] for w in ws)
    print("write pass of 3 chunks after a batch-1 generate: ok")
    print("PASS")


if __name__ == "__main__":
    main()
