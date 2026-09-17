"""Interactive REPL over a memy checkpoint. Memory is the ONLY cross-turn
channel: every turn is an independent forward pass (positions from 0), and the
bank accumulates normalized write-site states.

Untrained-but-interesting behaviors, on by default:
  * assistant self-writes (its own replies enter the bank)  --no-self-write
  * reads during user-turn ingestion (cross-turn coref)     --no-read-on-ingest

Commands: /bank /clear /quit
Usage: python chat.py --ckpt runs/L2-v1/ckpt-5000
"""

import argparse

import torch

from probe_mech import load_checkpoint
from train import encode, pad_batch


def ingest(model, ctx, ids, bank, read_on_ingest, device):
    """Forward a chunk, collect its writes, append to bank."""
    x = torch.tensor([ids], device=device)
    mask = torch.ones_like(x)
    set_bank(ctx, bank, device, enabled=read_on_ingest and len(bank) > 0)
    ctx.collect_writes = True
    with torch.no_grad():
        model(input_ids=x, attention_mask=mask, use_cache=False)
    ctx.collect_writes = False
    written = ctx.written[0]  # (T, H)
    ctx.written = None
    return written


def set_bank(ctx, bank, device, enabled=True):
    if enabled and bank:
        mem = torch.cat([b for _, b in bank], dim=0)[None]  # (1, N, H)
        ctx.memory = mem
        ctx.memory_mask = torch.ones(1, mem.shape[1], dtype=torch.long, device=device)
        ctx.reads_enabled = True
    else:
        ctx.memory, ctx.memory_mask, ctx.reads_enabled = None, None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-gen", type=int, default=400)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--no-self-write", dest="self_write", action="store_false")
    ap.add_argument("--no-read-on-ingest", dest="read_on_ingest", action="store_false")
    ap.add_argument("--read-heads", type=int, default=4)
    ap.add_argument("--read-rank", type=int, default=128)
    ap.add_argument("--write-layer", type=int, default=19)
    ap.set_defaults(self_write=True, read_on_ingest=True)
    args = ap.parse_args()

    tok, model, ctx = load_checkpoint(args.ckpt, args)
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id

    bank = []  # list of (label, (T, H) tensor)
    print(f"\nmemy chat — self_write={args.self_write} read_on_ingest={args.read_on_ingest}"
          f"\n/bank /clear /quit\n")
    turn = 0
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/clear":
            bank = []
            print("[bank cleared]")
            continue
        if user == "/bank":
            for label, m in bank:
                print(f"  {label}: {m.shape[0]} memories")
            print(f"  total: {sum(m.shape[0] for _, m in bank)}")
            continue

        turn += 1
        # 1. Ingest the user turn (writes memories; reads see prior bank).
        user_ids, open_ids = encode(tok, [user])
        w = ingest(model, ctx, user_ids[0], bank, args.read_on_ingest, device)
        bank.append((f"turn{turn}-user", w))

        # 2. Generate the reply: fresh state, memory-only context.
        set_bank(ctx, bank, device)
        ctx.log_stats = True
        oids, omask = pad_batch([open_ids], pad_id, "left", device)
        with torch.no_grad():
            out = model.generate(input_ids=oids, attention_mask=omask,
                                 max_new_tokens=args.max_gen,
                                 do_sample=not args.greedy,
                                 temperature=args.temp, top_p=0.8, top_k=20,
                                 use_cache=True, pad_token_id=pad_id)
        ctx.log_stats = False
        gen = out[0, oids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True).strip()
        print(f"\nbot> {text}\n")
        if ctx.stats:
            ent = sum(s["entropy"] for s in ctx.stats) / len(ctx.stats)
            print(f"     [bank {sum(m.shape[0] for _, m in bank)} | read entropy {ent:.2f}]")
        ctx.stats = []

        # 3. Optionally write the assistant's own turn into the bank.
        if args.self_write:
            reply_ids = open_ids + gen.tolist()
            w = ingest(model, ctx, reply_ids, bank, args.read_on_ingest, device)
            bank.append((f"turn{turn}-bot", w))
        set_bank(ctx, [], device, enabled=False)


if __name__ == "__main__":
    main()
