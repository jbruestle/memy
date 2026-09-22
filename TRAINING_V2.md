# Training v2: multi-chunk, multi-source distillation

Status: plan drafted 2026-09-21, pending review. Supersedes the earlier
TRAINING_V2.md draft (moved out of the repo). Companion to `DESIGN.md`,
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
- **User turn k.** User message: `[user turn k]: <text>`. Reads over
  background + all turns < k. Writes. No loss.
- **Agent turn k, k < t.** Assistant message: `[agent turn k]: <as-is
  dataset assistant text>`. Reads over everything before it. Writes. No
  loss. These are the off-distribution self-writes; at deployment the
  bank holds the model's own turns, so this mismatch is accepted rather
  than corrected (see Alternatives).
- **Agent turn t (target).** Chat template opens the assistant turn; the
  prefix `[agent turn t]: ` is force-fed and excluded from the loss; the
  teacher-generated tokens are then teacher-forced and the KL is taken
  over them. Reads over everything. Does not need to write.

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
forward afterwards, as in v1. No per-turn generation cap beyond the
overall 64k transcript cap.

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
- Teacher generation: no per-turn cap, only the 64k transcript cap.
- Qasper papers are single chunks, never sectioned.
