# Training v2: multi-chunk, multi-source distillation

Status: plan drafted and reviewed 2026-09-21; harness architecture agreed
(see the section at the end). Supersedes the earlier TRAINING_V2.md draft
(moved out of the repo). Companion to `DESIGN.md`,
which remains the source of truth for the architecture and decision log.

## Goal

Train the memory system in the regimes that v1 never saw and that the
exploration battery showed to be its failure modes: banks holding more than
one chunk, assistant self-writes, distractor content in the bank, chunks up
to several thousand tokens, and answers that must combine several memories.
Target use case: turns of a few hundred tokens with a skewed tail up to ~5k.

Loss is unchanged: per-token KL(teacher ‖ student) on a teacher-generated
assistant turn, teacher-forced into the student. Nothing else is optimized.

## Data sources

Three sources. Each batch is drawn from one source, chosen at random with
configurable mixture weights; batches are homogeneous by source. Starting
weights: **50% WildChat, 30% MuSiQue, 20% Qasper**.

| Source | Role | Bank contents | Target |
|---|---|---|---|
| WildChat-1M | multi-turn chat | as-is prior user + assistant turns | assistant turn t |
| MuSiQue-Full | multi-doc QA | gold + N distractor paragraphs | answer to the question |
| Qasper | long-doc QA | full paper + D distractor papers | answer to the question |

Why these three (decided 2026-09-21; see DESIGN decision log):

- **WildChat** over ultrachat: real user follow-ups that reference the prior
  answer ("shorter", "no, the other one"); GPT-4-era assistant turns are
  stylistically closer to Qwen than ultrachat's GPT-3.5 text. ~41% of the
  1M conversations are multi-turn (mean 2.5 rounds, 3.7% beyond 10).
- **MuSiQue** over HotpotQA: HotpotQA is largely answerable from one gold
  paragraph, which would let the student win by reading one memory.
  MuSiQue is 2–4 hop with each hop depending on the previous answer,
  20 paragraphs per question with chain-similar distractors, ~20k train.
  MuSiQue-Full adds minimally altered unanswerable twins (abstention):
  train is 19,938 answerable + 19,938 unanswerable; the unanswerable
  twins are built by removing supporting paragraphs, and 13,617 of them
  (68%) still retain 1–3 supporting paragraphs (insufficient gold), while
  6,289 retain none.
- **Qasper** over QuALITY: QuALITY has 285 articles total (118 train) and is
  multiple choice; a distractor bank would recycle the same stories.
  Qasper has 1,585 NLP papers at roughly the 5k-token budget, 5,049
  free-form questions, annotated evidence paragraphs (which tell us which
  memories reads *should* land on — plugs into readmap / targeted-deletion
  diagnostics), same-domain distractors as natural hard negatives.
  Train split: 888 papers, 2,593 questions (dev adds 281 papers). 271
  questions (10.5%) are unanswerable and carry no evidence at all.

Filtering:

- WildChat: English, non-toxic subset, ≥ 2 rounds, every turn ≤ chunk cap,
  total history ≤ teacher cap. Follow-ups that reference entities invented
  in a prior assistant turn are fine here because history is used as-is.
- MuSiQue: keep answerable items and the unanswerable twins that retain
  ≥ 1 supporting paragraph; drop the 0-supporting twins (nothing for the
  teacher to see). Unanswerables are subsampled to **15% of MuSiQue
  samples** so abstention is present but does not dominate.
- Qasper: drop unanswerable questions (no evidence ⇒ empty teacher context)
  and the 78 answerable questions with no evidence annotation.
- All sources: any single chunk > 5k tokens → drop the sample. Any teacher
  transcript > 64k tokens → drop (Qwen3.5-4B native context is 262,144, so
  this is a cost cap, not a model limit). In practice only deep WildChat
  histories can hit the teacher cap; QA teacher transcripts are short.

Held-out eval: a fixed slice of each source, plus the existing synthetic
probes. Candidate external evals (not for training): QuALITY hard subset,
FRAMES, ThoughtTrace, LongMemEval / LoCoMo.

## Student encoding

A sample is an ordered list of **chunks**. Each chunk is a separate forward
pass with fresh KV / GDN state and position ids restarting at 0. Reads see
the bank accumulated from earlier chunks; the chunk's own per-token
memories are appended to the bank after the pass. Nothing else crosses a
chunk boundary. **Every chunk of every type writes memories**; the only
chunk that need not write is the target turn (nothing reads after it).

Chunk types:

- **Background chunk.** One passage, wrapped as a user message:
  `Please read the following:\n<passage>`. No turn tag. No loss. Ingested
  against an **empty bank**, independently of other background chunks, so
  all background chunks of a batch run together in one (micro-batched)
  pass. Passage order is irrelevant, matching the permutation-invariant
  bank.
- **User turn k.** User message: `[user turn k]:\n<text>`. Reads over
  background + all turns < k. Writes. No loss.
- **Agent turn k, k < t.** Assistant message: `[agent turn k]:\n<as-is
  dataset assistant text>`. Reads over everything before it. Writes. No
  loss. These are the off-distribution self-writes; at deployment the
  bank holds the model's own turns, so this mismatch is accepted rather
  than corrected (see Alternatives).
- **Agent turn t (target).** Chat template opens the assistant turn; the
  prefix `[agent turn t]:\n` is force-fed and excluded from the loss; the
  teacher-generated tokens are then teacher-forced and the KL is taken
  over them. Reads over everything. Does not need to write.

Tags end in a newline, not a space, so the tokens after the tag are the
same tokens the teacher produced after `assistant\n` (a trailing space
would re-tokenize the first word). Tags are a flag (`--turn-tags`) so the
v1-shaped regression run can be exact; the flag may be dropped later.

Ordering across chunks comes **only from the turn tags**. Position ids
restart per chunk (decided: tags are the trained-in order signal;
position continuation helped an untrained model in exploration but puts
deep histories at OOD positions).

Per-source layout:

- WildChat: [user 0, agent 0, user 1, agent 1, …, user t, **agent t**].
  Cutoff t is sampled uniformly over the conversation's assistant turns,
  so every history depth is covered at one teacher generation per sample.
  No distractor conversations in the bank for this run.
- MuSiQue: [gold + N distractor paragraphs as background] then
  [user 0 = question, **agent 0**]. N is drawn per sample uniformly from
  0..(20 − #gold), distractors taken from the item's own 20-paragraph
  context.
- Qasper: [paper + D distractor papers as background] then
  [user 0 = question, **agent 0**]. D is drawn per sample uniformly from
  0..4, distractor papers sampled from the rest of the split. Papers are
  never split into sections: they are in the mix precisely because they
  are large.

Distractor "curriculum" is therefore no schedule at all: per-sample
uniform sampling of the count covers the whole range at every step, needs
no tuning, and yields an eval curve of KL vs distractor count for free.
Ranges are configurable (`--musique-distractors 0:18`,
`--qasper-distractors 0:4`).

Gradients: flow through all writes (per the 2026-09-21 ablation, write
gradients matter for the recall plateau), **except** that distractor
background chunks beyond the first 2 per sample run under no-grad. I.e.
gold chunks and up to 2 distractors get full backward; further distractors
are forward-only. Rationale: keep enough distractor gradient to let the
writer specialize against negatives, without paying (1+D)×5k tokens of
backward per Qasper sample. `--distractor-grad-k` (default 2).

## Teacher encoding

Same weights, LoRA disabled, reads bypassed, ordinary attention over a
single plain chat transcript. **No turn tags** in the teacher transcript:
tags are a student-side addressing device and the teacher should show
ordinary assistant behaviour. The teacher generates the target turn
(non-thinking sampling params), and its logits are captured in a no-grad
forward afterwards, as in v1. Per-turn generation cap is a configurable
soft cap (`--max-gen`, default ~1k for the cloud run, v1's 300 for local
validation): a batch waits for its longest generation and the target
chunk's logits scale with it, so an uncapped turn is a cost and memory
hazard rather than a capability limit. The 64k transcript cap stays.

- WildChat: transcript = the as-is history (untagged), open assistant turn.
  Teacher and student see the same text, differing only in access mode.
- MuSiQue: one user turn = `Please read the following:` + **only the
  supporting paragraphs** + the question. Teacher never sees distractors.
  For unanswerable twins this is the 1–3 remaining supporting paragraphs,
  so the teacher answers from insufficient context and (hopefully)
  abstains; the student must reproduce that from a bank whose gold is
  equally insufficient — a negative memory hit is part of the target.
- Qasper: one user turn = `Please read the following:` + **only the
  annotated evidence paragraphs** + the question.

So in the QA sources the teacher's single turn contains gold context +
question while the student's user turn 0 contains only the question and
everything else lives in memory. The KL target is "the answer given the
right short context". No brevity instruction: chatty teacher answers are
accepted. Principle unchanged from v2 roadmap: the teacher never needs
long context, only the right short context.

Teacher text is generated inline per batch (no precompute). With one
random cutoff per conversation there is no prefix sharing between samples
to exploit, and inline keeps encoding changes rerun-free. Fallback if
generation dominates step time: tokens-only vLLM pregeneration keyed by
(conversation id, t), which survives all student-side encoding changes;
logits stay inline.

## Batching and cost

- Batches are homogeneous in source **and** in cutoff depth t. WildChat is
  bucketed by t at load time; the sampler picks a bucket with probability
  proportional to its size. QA sources are always depth 0. This keeps the
  t+1 sequential turn passes free of whole-chunk padding.
- Background chunks of a batch are flattened into one empty-bank pass,
  micro-batched by token count. Turn passes are batched across samples at
  the same turn index, padded to the longest chunk.
- Per-sample student cost ≈ (background tokens, grad on gold + 2
  distractors) + Σ turn-chunk tokens with full backward. Approximate
  tokens per sample: WildChat ~1–5k, MuSiQue ~3k, Qasper ~5k grad + D×5k
  forward-only. Versus ~600 in v1, so expect 5–10× the per-sample cost;
  the 27B budget model in DESIGN.md applies with ctx = tokens/sample.
- Bank size: up to ~(1+D)×5k memories per Qasper sample; read cost is
  negligible, but bf16 storage is ~13MB per 1k memories per sample. Watch
  peak VRAM at batch 8 with D = 4.

## Compute

This run is expected to exceed what the single local 5090 can do in a
reasonable time (v1 was ~1.4 s/sample at 600 tokens; v2 samples are
5–10× larger). Plan: spin up cloud GPUs for this run, both for throughput
and to exercise cloud infra before the 27B target. Single-GPU H100 80GB
keeps the harness unchanged (see DESIGN.md scaling section); multi-GPU
data parallel is the natural next step since samples are independent.

## Diagnostics to keep

Everything from v1 (eval KL per source, read entropy / argmax histograms
per head, probe recall), plus:

- Per-source eval KL and, for MuSiQue/Qasper, exact-match / F1 of the
  student's free generation against the dataset answer.
- Readmaps on QA samples: mass on gold vs distractor memories at the step
  where the answer is emitted (Qasper evidence and MuSiQue supporting
  paragraphs give ground truth).
- Depth curve: WildChat eval KL as a function of t.
- Abstention: student behaviour on unanswerable items (MuSiQue-Full,
  Qasper) vs teacher.

## Alternatives considered

- **Regenerating prior assistant turns with the teacher** (fully
  on-distribution self-writes). A sample of 20 ultrachat conversations
  showed ~15% of follow-ups name entities invented in the prior assistant
  turn, so naive replay breaks those; a cheap filter (capitalized names /
  quoted titles in U2 present in A1 but not U1) would make it workable.
  Deferred: as-is history is simpler, unbiased, and the deployment bank
  is Qwen-style anyway. Keep as a later arm to price self-write mismatch.
- **Position-id continuation across chunks.** Rejected for this run in
  favour of turn tags (see above).
- **Distractor conversations in the WildChat bank.** Skipped for this use
  case; the mission-capture failure remains open and would need it.
- **Precomputed teacher text.** Rejected; no prefix reuse, and inline is
  more flexible. vLLM fallback noted above.
- **HotpotQA / QuALITY.** Replaced by MuSiQue / Qasper for the reasons
  given under Data sources; QuALITY hard kept as an eval.

## Resolved 2026-09-21 (Jeremy review)

- Mixture weights: configurable, start 50/30/20 (WildChat/MuSiQue/Qasper).
- Distractor counts: per-sample uniform over a configurable range, no
  schedule (MuSiQue 0..18, Qasper 0..4).
- Unanswerables: MuSiQue-Full twins with ≥ 1 supporting paragraph kept at
  15% of MuSiQue samples; Qasper unanswerables dropped (no evidence).
- Teacher generation: configurable per-turn soft cap (amended from "no
  cap" after the harness discussion) plus the 64k transcript cap.
- Qasper papers are single chunks, never sectioned.

## Harness architecture (agreed 2026-09-21)

Everything below is source-agnostic: the engine never knows what a paper
or a conversation is. Datasets and probes are plugins under `sources/`
and `probes/` (plain modules; a registry by name, not dynamic loading).
`sources/` rather than `datasets/` because the latter shadows the
HuggingFace package. Built 2026-09-21: `engine.py`, `train_v2.py`,
`sources/ultrachat.py`, `probes/bindings.py`, `test_engine.py`.

### Sample schema

One schema for training data, eval data, and probes:

```python
Sample = {
  "id": str,                     # stable key: eval-target cache, resume cursor, logs
  "source": str,                 # dataset / probe name
  "gold": [str, ...],            # passages in the bank AND in the teacher context
  "distractors": [str, ...],     # passages in the bank only
  "turns": [("user", str), ("assistant", str), ..., ("user", str)],  # ends on the target's user turn
  "meta": {...},                 # optional: answer (EM/F1), evidence (readmaps), anything the probe wants back
  "teacher_gold": [str, ...],    # optional: what the TEACHER sees instead of gold (Qasper: evidence
}                                #   paragraphs, while the bank holds the whole paper)
```

The teacher transcript is derived by the engine, never by the dataset:
gold passages (if any) are prepended to the first user turn under the
`Please read the following:` wrapper, followed by the turns as-is and
untagged, with the assistant turn open. This one rule covers WildChat
(no gold), MuSiQue and long-doc QA (gold + one question turn), and
ultrachat (no gold, one turn — the v1 degenerate case).

### Dataset plugin

`python peek_source.py <name> [--split eval] [--stats N] [--tags] [--cfg k=v]`
prints Samples as the engine encodes them (per-chunk token counts,
teacher prompt length) or length statistics, without loading the model —
sources are built and debugged in isolation from training.
`sources/<name>.py` exposes `train(cfg, seed, epoch, start, shard)` (an iterator of
Samples in a deterministic per-seed-per-epoch permutation, so resume is an
index), `eval(cfg)` (a fixed slice), and its own knobs (distractor range,
length filters). Source-specific filtering and the gold/distractor split
live entirely inside the module. The first module is `ultrachat` (v1
data, `turns=[("user", q)]`), used to regress the new engine against the
`runs/L2-v1/log.jsonl` trajectory with tags off.

### Probe plugin

`probes/<name>.py` exposes `samples(step) -> list[Sample]` (fixed seed,
so curves are comparable across steps and runs), `interval` (steps), and
`score(results) -> dict` (the JSON blob logged; the probe may also write
its own files). For each sample the engine builds the bank (gold +
distractors + prior turns), free-generates the **student** from memory,
and free-generates the **teacher** from the derived transcript with gold
in attention. The teacher never changes, so teacher completions are
computed once per sample id and cached: the teacher ceiling on every
probe comes for free (the 4B base may itself fall short of 100% on some
probes, which this exposes). Each result carries both texts, per-sample
read stats (entropy, argmax), and readmaps when the probe asks for them
(per-chunk read mass; gold/evidence chunks are known from the sample).
Per-source QA eval (MuSiQue / long-doc EM-F1 against `meta.answer`) is a
probe over that dataset's eval slice; the v1 five-binding probe, the
name-age scale test, distractor capture and the multi-turn split each
become a probe module. Only the KL eval is engine-native.

### Engine responsibilities

- Chunk construction from a Sample: background chunks (each gold and
  distractor passage, empty bank), turn chunks (tagged when
  `--turn-tags`), target chunk (assistant opening + tag + teacher tokens).
- Two background passes per batch: grad for gold + the first
  `--distractor-grad-k` distractors, no-grad for the rest (a single
  forward cannot mix the two). Then sequential turn passes with the bank
  growing by concatenation (padded bank + mask across samples).
- Teacher generate (inline, sampled, `--max-gen`) and teacher logits from
  a forced forward; **logits only at the generated positions** (left-pad
  the transcript and use `logits_to_keep`, or run the LM head on sliced
  hidden states). Same for the student target chunk. The v1 code
  computed full-transcript logits and sliced; at 8×5k×248k that is ~20GB.
- Exact full-vocab KL as in v1 (`_KLSum`).
- Batching by a **token budget** within homogeneous (source, depth)
  buckets, with per-bucket queues so the data stays streaming; the
  mixture weight picks the source, then a bucket proportional to size.
- Eval KL per source on the fixed slices, teacher targets cached by
  sample id (replaces the position-keyed `runs/eval-targets.json`).
- Probes on their intervals; all diagnostics from v1 (read entropy,
  argmax histograms) retained.
- Checkpoint + resume: adapter, readers, optimizer, scheduler, step,
  python/torch/cuda RNG, per-dataset (and per-rank) cursors, epoch. A
  killed cloud run resumes exactly.
- Data parallel: plain DDP over the trainable parameters only (~70M at
  4B; at 27B each GPU still holds a full bf16 base copy, so DDP suffices
  and FSDP is not needed). Each rank runs its own inline teacher
  generation on its own shard of the stream; step time varies by rank
  (buckets differ) and that is accepted. Written through one code path;
  world size 1 is tested locally, multi-GPU first on the rented box.
- Size knobs all configurable (`--max-gen`, chunk cap, transcript cap,
  token budget, distractor ranges, grad-k, mixture) so the pipeline is
  validated locally at v1 sizes and scaled on cloud hardware unchanged.

### Validation plan

1. Reader through the fused SDPA kernel — DONE 2026-09-21 (see DESIGN
   decision log: ckpt-14000 recall 0.94 unchanged, step-0 identity exact,
   fwd+bwd peak 0.41GB at T=4k over N=24k).
2. Engine + `sources/ultrachat`, tags off, v1 sizes on the 4090:
   eval_kl / recall trajectory over the first ~1k steps must track
   `runs/L2-v1/log.jsonl` (same data order, batch 8). Step-level parity
   is already exact (`test_engine.py`: v1 `training_step` and v2
   `student_loss` give the identical loss on the same batch and teacher
   tokens). Trajectory run `runs/L2-v2-regress` PASSED 2026-09-22: eval_kl
   0.360 / 0.308 / 0.281 / 0.255 vs v1 0.342 / 0.292 / 0.263 / 0.244 at
   steps 250 / 500 / 750 / 1000; recall 0.15 vs 0.13 at 1000 (see
   DESIGN decision log).
3. Resume: DONE at smoke scale (4 steps, checkpoint, resume for 2 more:
   sampler cursor, optimizer, RNG restored). Kill-and-restart on the long
   run still to be exercised.
4. Same run with tags on — DONE 2026-09-22 (`runs/L2-v2-tags`): no
   measurable cost (eval_kl 0.254 vs 0.255 at step 1000).
5. Per-source smoke runs — DONE 2026-09-22 (`runs/smoke-<source>`, 20
   steps each, batch 2, tags on, chunk cap 8192, remote teacher, eval +
   probes exercised; all exit 0). Measured on the 4090:

   | source | s/step (batch 2) | chunk tokens/step | bank max | peak VRAM |
   |---|---|---|---|---|
   | wildchat | 5.2 | 1.5k | 2.5k | 12.6 GB |
   | musique | 4.2 | 3.1k | 2.5k | 13.0 GB |
   | qasper | 11.0 | 29k | 23.5k | 18.4 GB |
   | triviaqa | 7.5 | 25k | 23.0k | 20.2 GB |

   So ~5 s per 15k-token long-doc sample on the 4090 with the teacher
   hidden; the long-doc sources need batch ≤ 2 on 24GB and are fine at
   batch 8 on 80GB. WildChat prompts beyond the teacher server's slot
   context (2048 here) fall back to local generation automatically
   (6 of 40 in the smoke run); a bigger `-c` on the server removes that.

### Implementation notes (2026-09-21/22)

Decisions made while building, beyond the contract above:

- **Remote teacher** (`--teacher-url`, llama.cpp `llama-server`): prompt
  token ids in, generated ids out via the native `/completion` endpoint
  (tokenizer parity with HF verified exactly), `--teacher-prefetch N`
  batches requested ahead so the server's continuous batching stays
  saturated independent of the trainer's batch; prefetched batches are
  stored raw in the checkpoint and replayed on resume. Prompts that do
  not fit a server slot (prompt + max_gen > n_ctx, read from `/props`)
  are generated locally; 4xx errors raise, connection loss / 5xx retry
  for 30 min (server restarts survive). On the shared 5090 the server
  gave ~200–250 tok/s (single-stream decode 58 ms/step = time-sliced
  with another training job), no speedup over inline; the design needs
  an uncontended card to hide the teacher entirely.

- **Write passes stop at the write site.** `MemoryContext.write_only`
  makes every wrapper above layer 19 return its input unchanged, and the
  pass goes through the inner model only (no LM head, no full-vocab
  logits). Nothing above the write site can affect the writes, so this is
  exact and saves ~37% of write-pass compute plus the v1 pass-1 logits
  (8×5k×248k would have been ~20GB).
- **Logits at generated positions only**, for both teacher and target
  pass: hidden states are gathered per sample and the LM head is applied
  to the gathered rows. Not `logits_to_keep` (it is a single slice shared
  across the batch); right padding is kept everywhere, so the GDN layers
  see exactly v1's padding.
- **Mixed empty/non-empty banks** in one batch (a MuSiQue sample with
  zero distractors is still non-empty; the only all-empty case is turn 0
  without background): an empty bank gets one zero memory, whose read
  output is exactly zero. An all-empty batch runs with reads disabled.
- **Teacher caches are per generation cap** (`runs/eval-targets-v2-g{max_gen}.json`,
  `runs/probe-teacher-v2-g{max_gen}.json`), keyed by sample id. The v1
  position-keyed `runs/eval-targets.json` is imported by order for the
  ultrachat eval slice, so eval_kl stays comparable with L2-v1.
- **Data parallel** is a manual flat all-reduce of trainable gradients
  after backward rather than the DDP wrapper: a step makes several
  forward calls plus a `generate`, which the wrapper's reducer handles
  badly. Streams are sharded by row index (`row % world == rank`), the
  checkpoint holds one `sampler-rank{r}.pt` (cursor + RNG) per rank next
  to a single `state.pt`, and source exhaustion is all-reduced each step
  so ranks stop together. Untested beyond world size 1.
- **Sampler**: one queue per (source, depth); a batch is emitted when a
  queue reaches `--batch` samples or `--token-budget` chunk tokens;
  partial queues are flushed at epoch end; queued samples are stored raw
  in the checkpoint and re-encoded on resume. Samples over
  `--max-chunk-tokens` / `--max-prompt-tokens` are counted as dropped.
- **Checkpoint layout**: `ckpt-N/` holds the PEFT adapter and
  `readers.pt` (so `probe_mech.py` still loads it) plus `state.pt`
  (trainable params, optimizer, scheduler, step) and the per-rank sampler
  files. `--resume auto` picks the latest complete one.
- **Probes** take `interval` from their cfg (default: eval interval),
  and every result set is dumped to `probe-<name>-<step>.jsonl`.
- **Measured at v1 sizes** (batch 8, 300-token cap, 4090): 10.1 s/step of
  which teacher generation is 8.8 s. Generation is ~85% of the step, not
  "roughly doubles" as v1 estimated; at scale the tokens-only vLLM
  pregeneration is the first optimization to make, not a fallback.

### Sources built (2026-09-21)

All four planned sources exist and were checked through `peek_source.py`
(CPU only). Measured on 200–300 training samples each, default cfg:

| source | items | tokens/sample (mean, p90) | max chunk (p90) | bank chunks | notes |
|---|---|---|---|---|---|
| ultrachat | 200k | ~300 | 320 | 1 | v1 data |
| wildchat | 838k rows (English non-toxic multi-turn subset) | 0.9k, 2.6k | 1.1k | 2t+1 (depth p50 3, p90 9, max 31) | cutoff t per (seed, epoch, row); ~1/3 of samples are depth 1 |
| musique | 39.9k train / 4.8k dev (full) | 1.4k, 2.5k | 355 | 1–20 | unanswerables 15.1% realized (keep-prob computed from split counts); MuSiQue's supporting-paragraph annotations are occasionally noisy (answer not in gold) |
| qasper | 888 papers / 2.6k q train, 281 / 1k dev | 15.3k, 23k | 8.0k | 1–5 | papers > `max_paper_tokens` (8192) dropped; teacher sees evidence only via `teacher_gold`; a run must set `--max-chunk-tokens` ≥ 8192. Demoted: small weight or eval-only (see below) |
| triviaqa | 77.6k q train / 7.9k dev (rc.wikipedia) | 11.6k, 21.9k | 4.1k | 1–11 | pages split at paragraph boundaries into ≤ `chunk_tokens` (4096) passages; rows with gold > `max_gold_tokens` (16384) or no alias hit dropped (75% kept); `teacher_gold` = lead paragraph + ≤3 alias-hit paragraphs per page (distant supervision) |

Details: `bdsaglam/musique` (default config = MuSiQue-Full), `allenai/qasper`
via its `refs/convert/parquet` revision (the script loader is unsupported
by datasets 4.x), `allenai/WildChat-1M` (non-toxic release, 3.4GB parquet).
Eval slices: ultrachat/wildchat = first `n_eval` eligible train rows
(skipped by train); musique/qasper = the validation split. Per-sample
randomness (cutoff, distractor draws) is seeded by (seed, epoch, id) so a
resumed stream reproduces its draws.

### Long-document source: TriviaQA replaces Qasper (2026-09-21)

Qasper is small (2,593 train questions ⇒ ~15 epochs at a 20% slice over
200k samples) and single-subject. Survey of alternatives with working
loaders: TriviaQA rc.wikipedia (77.6k questions, 1–3 whole Wikipedia
pages each, answer aliases, no evidence annotation), Natural Questions
(best content fit — annotated long-answer paragraph — but the full train
set is 41GB and the parquet conversion holds only 18k train rows; HTML
must be stripped; kept as a later upgrade), SQuAD reassembled into ~5k-token
articles (442 articles, 87k questions, exact evidence paragraph — cheap
secondary candidate, not built), NarrativeQA (only the ~1.2k-token
summaries are usable), QuAC (multi-turn QA over ~700-token Wikipedia
sections; loader unverified). Decision: `sources/triviaqa.py` built and
takes the long-doc slot; Qasper stays available at a small weight or as
eval. TriviaQA design note: the first version dropped whole pages over a
cap, which preferentially removed the relevant page (e.g. the Angola
question kept only "Nation state" because "Portugal" appears in it);
pages are now split into capped chunks and the cap applies to total gold
tokens instead. Page lengths (652 sampled): p50 6.3k, p90 14.7k tokens;
36% ≤ 4k, 62% ≤ 8k, 93% ≤ 16k.
