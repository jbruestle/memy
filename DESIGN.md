# Memy: memory as the only cross-chunk channel

Status (2026-09-21): v1 answered positive (L2-v1 stopped at 14k steps,
eval_kl 0.113, probe recall 0.93; readers-only and writer-gradient
ablations done, see decision log). v2 plan settled in `TRAINING_V2.md`;
v2 harness built (`engine.py`, `train_v2.py`, plugin `sources/` and
`probes/`), step-level parity with v1 exact, 1k-step trajectory
regression `runs/L2-v2-regress` in progress. Next: tagged run, then
WildChat / MuSiQue / long-doc sources, then cloud.

Files: `model.py` (surgery), `engine.py` (v2 executor: Sample -> chunks,
passes, teacher, KL), `train_v2.py` (v2 loop: sampler, eval, probes,
resume, DDP), `sources/` and `probes/` (plugins; see `TRAINING_V2.md`),
`test_engine.py` (v1/v2 loss parity + multi-chunk smoke), `data.py`
(v1 ultrachat + probe generator), `train.py` (v1 loop, kept for parity
checks), `test_step0.py` (identity test), `diag_mem.py` (per-phase VRAM
watermarks), `probe_mech.py` (post-hoc mechanism battery), `chat.py`
(REPL; v1 encoding, untagged), `explore/` (v1 exploration scripts; run
from repo root). Run with the `finetune` conda env and
`PYTORCH_ALLOC_CONF=expandable_segments:True`.
This doc is the cross-session source of truth. Update the decision log when anything changes.

## Concept

Long-term memory for LLMs where:

- **Write**: one memory per token, *no learnable parameters* — the residual-stream
  hidden state at a chosen layer is stored as-is. The write "policy" is entirely
  (a) which layer we tap and (b) whatever the LoRA does to shape those states.
- **Read**: learned memory heads at several layers. Low-rank query/key: score =
  (A·h)·(B·m); softmax over memories; value = W_out(W_down · weighted-mix), added
  to the residual stream. B is applied at read time, so it does not violate the
  parameter-free write constraint, and folds into a single-index MIPS search at
  inference (Unlimiformer trick: q' = Bᵀ(A·h)).
- **Chunking**: chunk = chat turn. Memories written in chunk n are visible from
  chunk n+1. **No cross-chunk attention at all** — normal (and linear) attention
  operates only within a chunk; all cross-chunk information flows through memory.
  v2 adds *background chunks* (passages ingested against an empty bank) and
  `[user turn k]:` / `[agent turn k]:` text tags as the only cross-chunk
  order signal; see `TRAINING_V2.md`.
- Theory: per-token cross-boundary information is small and already contextualized
  by mid-network; syntax/agreement is resolved in-chunk, so processed per-token
  states should suffice.

Prior art positioning: write-time compressors (gist tokens, AutoCompressors, RMT,
Infini-attention) bottleneck the past into few/fixed vectors and lose detail; we
store everything per-token and put the bottleneck at *read bandwidth*, which is
query-conditioned. Unlearned writes validated by Memorizing Transformers; single
shared per-token store serving many layers validated by YOCO. The untested cell:
zero cross-chunk attention + growing unlearned per-token store + sparse learned
reads. That is the contribution.

## V1 experiment (smoke test)

**Question**: can multiple learned reads per token recover enough state from a
single unlearned write per token to answer a question with no attention access
to it?

- **Base model**: Qwen3.5-4B (hybrid: repeating 3× GatedDeltaNet + 1× gated full
  attention; 32 layers; hidden 2560; full-attention layers at 0-indexed
  3, 7, 11, 15, 19, 23, 27, 31; vocab 248,320 padded; text-only usage,
  `enable_thinking: False`).
- **Write site**: post-layer-19 residual stream (~60% depth, coincidentally a
  full-attention layer), RMS-normalized before storage. Gradients DO flow through
  writes (both passes stay in the graph).
- **Read sites**: 4 heads at each of the 8 full-attention layers = 32 heads,
  rank 128, ~40M params. Output projections zero-initialized so student ==
  teacher at step 0.
- **LoRA** over the base for everything else; student starts bit-identical to base.
- **Student passes** (two separate forward calls — separate passes implement
  "no cross-chunk attention/state" by construction, no masking surgery):
  1. User turn: writes per-token memories; reads see empty memory (return 0).
  2. Agent turn: fresh KV/GDN state, chat template opens directly at assistant
     turn; reads attend over the user-turn memories; teacher-forced on the
     dataset assistant response.
- **Teacher**: the same weights with LoRA disabled (`disable_adapter()`) and read
  heads bypassed; *generates* the agent turn inline per batch (no_grad, normal
  attention over user+response, recommended non-thinking sampling params, capped
  at ~300 tokens). Per-step logits are captured during generation, so generation
  is the entire teacher cost — no separate logit pass, no precompute (see
  decision log).
- **Loss**: per-token KL(teacher ‖ student) over the teacher-generated response
  tokens, teacher-forced into the student. On-policy teacher text: ultrachat
  supplies only the user questions, never the responses (GPT-3.5-flavored
  responses are off-distribution for a 4B Qwen and would distort the KL).
- **Data**: ultrachat_200k first exchanges, both sides length-filtered to a few
  hundred tokens; at most one epoch; fixed held-out eval slice; plus a synthetic
  probe set (held out from training entirely).
- **Probes**: questions with multiple independent arbitrary bindings that cannot
  compress into one gist vector ("my name is X, my sister is Y, I owe her Z...").
  Free-generation recall on these is the qualitative headline metric; KL is the
  training signal.

### The ladder (all arms share one harness, config-flag apart)

- **L0 — no memory**: reads forced to zero. Floor: what LoRA buys from
  "teacher-flavored generic answers" + empty-context distribution repair.
- **L1 — single vector**: memory truncated to the final user-turn token's state.
  Tests the "network smuggles one gist vector" degenerate solution.
- **L2 — full per-token memory**: the real system.

Result of interest: L2 < L1 < L0 in eval KL with meaningful gaps, and L2 ≫ L1 on
probe recall. If L1 ≈ L2, retrieval isn't earning its keep at this scale.

### Cheap post-hoc mechanism battery (no training required)

Run on any checkpoint; no single test suffices (last-token truncation alone
misses averaging-style gist, per Jeremy). Together these bound the mechanism:

- Mean-pool: bank -> its mean vector. Unchanged output => reads use only the
  average (catches diffuse-softmax gist).
- Last-token: bank -> final token's memory (catches EOT-summary gist).
- Rank-k: bank -> rank-k SVD approx, sweep k. Effective dimensionality of
  what reads consume; survives k=1-2 => gist, needs k~20+ => distributed.
- Targeted deletion: on probes, delete the memories at binding-token
  positions; binding-specific recall drop is near-conclusive evidence of
  token-targeted retrieval.
- Read maps: dump per-head softmax over memories at the step where a binding
  is emitted; retrieval should show mass on the binding's memory.

Run at 1k/5k/10k-step checkpoints to timeline when (if) retrieval starts
mattering. The trained L1 arm remains the separate *performance* ablation.
(Motivated by step-250: task semantics + placeholder slots crossed the
divide early — "Dear [Uncle's Name] ... in [Location]" — consistent with
the value pathway learning gist transport before query sharpening.)

### Diagnostics to wire in from day one

- Read softmax entropy and argmax-memory histograms per head/layer (detects
  collapse onto last-token memory; needed to diagnose ordering failures).
- Eval KL on the held-out slice at fixed intervals (train KL alone confounds).
- Step-0 sanity test: with reads zeroed and LoRA at init, student logits must
  equal base-model logits exactly.

## v2 roadmap (drafted 2026-09-17, L2 at step ~7200, recall 86%)

L1/L0 arms deprioritized (single-vector hypothesis untenable given ordered-
digit recall). Priorities:

1. **Writer-gradient ladder** — DONE 2026-09-21: (i) end-to-end beats
   (ii) detached and (iii) frozen writer at convergence; precomputable-
   memories / reader-only branch deprioritized. v2 keeps full write
   gradients (distractor chunks beyond the first 2 excepted, for cost).
2. **Cheap post-hoc suite on existing checkpoints** (second machine ok):
   top-K sweep (predict K=8 lossless at entropy 1.6); site/head pruning map
   (predict sub-write-layer sites 3-15 matter least); read-rank SVD knee;
   readmaps; length generalization (600-1300 tok questions); contradiction
   + two-binding interference probes; empty-bank regression vs base.
3. **Chat REPL** — DONE (`chat.py`, findings 2026-09-17).
4. **Multi-turn training** — SUPERSEDED by `TRAINING_V2.md`: WildChat
   (not ultrachat) as-is histories with a random cutoff turn.
5. **Passage-bank training** — SUPERSEDED by `TRAINING_V2.md`: MuSiQue-Full
   and Qasper supply real questions and gold annotations, passages are
   written online with gradients (no frozen writer, no precompute), and the
   distractor count is sampled per sample rather than scheduled. The core
   principle survives unchanged: the teacher never needs long context, only
   the right short context. Sliding-window LM distillation remains an
   unexplored alternative objective.

v2 training procedure spec (problem families, generation processes, eval
sets, transcript structures): see `TRAINING_V2.md` (rewritten 2026-09-21).

## Deferred until the smoke test shows promise

- ANN / metric-tree top-K search (Jeremy has existing O(log N) infrastructure);
  v1 banks were N≈300 and v2 banks reach ~25k (Qasper, D=4), still cheap
  for exact softmax; top-K sweep on ckpt-14000 showed K=1–8 lossless.
- Sparsity/low-entropy regularization on read softmax (note: in tension with
  decay-mixing semantics — keep an unregularized arm when we get there).
- Update semantics: per-query learned exponential decay over matches.
- Explicit position/order features in memories (v1 relies on states being
  causal-contextual). v2 supplies order only via turn tags in the chunk
  text; learned order features / recency stay deferred (needed for the
  mission-capture failure, which v2 does not target).
- Ablations: write layer (60% vs final vs multi-layer concat), MLPs around the
  read head, detached-writes arm (measures how much LoRA "learns to write"),
  number of read heads/sites.
- Read-head init escalation, ONLY if queries stay inert (entropy pinned at
  uniform while eval KL moves). Note first: with mem_o zero-init, mem_q/k get
  exactly zero gradient at step 0 and only learn as mem_o grows (LoRA-style
  bootstrap) — slow early query movement is expected, not failure. Ladder:
  (1) separate higher lr on mem_o; (2) tiny random mem_o init (std ~1e-3,
  sacrifices bit-exact step-0 test); (3) symmetric init mem_q = mem_k (PSD
  "retrieve similar states" prior, position-free); (4) warm-start Q/K from
  existing attention heads — considered and rejected for v1: per-site space
  mismatch vs layer-19 memories, RoPE-contaminated metrics, and risk of
  landing in position-based attention basins that are meaningless over the
  memory bank. If ever tried, restrict to sites 19/23 where spaces align.
- ~~Multi-chunk histories (v1 is exactly two chunks).~~ In v2.

## Scaling to 27B: budget model (drafted 2026-09-21)

Long-term target: Qwen 3.8 27B. Compute is LINEAR in total tokens for any
context we will use: the dense term is ~6N FLOPs/token (fwd+bwd with grad
checkpointing; full backward is required because LoRA is everywhere and
write-gradients matter) and the attention term 4·L_attn·d·ctx only matches
it at ctx ≈ N/(2·L_attn·d) ≈ 160k tokens for a 27B hybrid. The bank read is
bank×ctx×rank×heads — quadratic in history but ~1e-4 of the dense term.
Sanity check vs the 4B run: 6·4B·600 = 14.4 TF/sample (doc: 14), teacher
2N·600 = 4.8 (doc: 5); achieved ~14 TF/s ≈ 9% of 4090 peak — MFU is the
dominant uncertainty.

    FLOPs/token ≈ 8N   (6N student + 2N teacher logits; ~4N if writes frozen)
    GPU-s       = samples × ctx × 8N / (MFU × peak)
    $           = GPU-s / 3600 × $/GPU-h

27B on one H100 (990 dense bf16 TF, 20% MFU, $2.5/h): ~900 tok/s, ~3.3M
tok/GPU-h, ≈ $0.75 per million training tokens (band $0.3–3). 200k×1k-tok
samples ≈ 60 GPU-h ≈ $150 (≈ $340 at today's 9% MFU); 1M×1k ≈ $750;
1M×4k ≈ $3k. Anchor: 4B→27B is ~7× compute and 4090→H100 ~6× peak, so a
27B run on one H100 feels like the current 4B run on the 4090 (~1–1.5
s/sample, ~3 days per 200k). Teacher decode is bandwidth-bound and does not
scale with tokens: pregenerate response tokens with vLLM (~$20 per 200k
responses) and keep only the 2N logit forward inline. Card choice is
memory-gated (54GB bf16 base + optimizer state + fp32 248k-vocab logits
≈ 8GB at batch 8×1k): H100 80GB (default, single-GPU keeps harness
unchanged), RTX Pro 6000 96GB (price it), H200/B200 only if within ~1.5×
of H100 per FLOP. Consumer 24–32GB cards need a quantized base — avoid.

## Hardware / environment

- Two local machines: this box has a 4090 (24.5GB), 31GB RAM, 16 cores; a
  second box has a 5090 (used for the readers-only arm). v1 fits on the
  4090: ~10GB bf16 frozen weights + ~70M trainable (LoRA + read heads,
  AdamW fp32 states ~0.9GB) + activations 1–2GB with checkpointing at
  batch 4–8 → ~13GB. No cloud was needed for v1; 4B stays (no need to drop
  to 2B). v2 goes to cloud for throughput, not memory.
- Throughput: student fwd+bwd ~14 TFLOPs/sample + teacher forward ~5 → ~0.5–1.5
  s/sample; full epoch 1–2 days, expect signal within the first 20–50k samples.
  These are v1 numbers (~600 tokens/sample). v2 samples are 1–25k tokens,
  so the v2 run is planned for cloud GPUs (see `TRAINING_V2.md`, Compute).
- Conda env: `finetune` (torch 2.10, transformers 5.2.0 — has native
  `Qwen3_5ForCausalLM` — peft 0.18.1, accelerate, datasets). No new env unless
  we hit a dependency conflict.

## Decision log

- **2026-09-16** — Initial design settled (all of the above). Notable calls:
  - Qwen3.5-4B over dense Qwen3-4B: closer to the target architecture; the 8
    full-attention layers are natural read sites. (4B is dense-FFN per model
    card; the MoE variants are the larger siblings.)
  - Normalize stored memories (massive-activation / attention-sink outliers
    would otherwise dominate the read softmax).
  - Gradients through writes for v1 (maximizes success odds, cheap at ~600 tok).
  - Teacher runs inline per batch, no precompute: logprob storage is expensive
    (~45GB), teacher weights are resident anyway via disable_adapter, and with
    a strict single pass over the data there is no reuse for precompute to
    amortize. Teacher GENERATES its response inline (ultrachat responses are
    off-distribution for a 4B Qwen; forced off-policy tokens make the KL
    measure the wrong thing). Logits captured during generation ⇒ no separate
    teacher logit pass. Roughly doubles step time; epoch ≈ 2–4 days, signal
    same-day. Fallback if generation throughput hurts: tokens-only vLLM
    pregeneration (~240MB), logits still inline — an optimization, not a
    design change.
  - Low-rank read heads (r=128) rather than full hidden×hidden (which would
    have been ~1.9B params).
- **2026-09-16 (implementation)** — harness built and smoke-tested. Measured:
  69.9M trainable params; step-0 identity test passes bit-exactly; batch 8 =
  ~1.4 s/sample at 15.2GB peak (batch 4 = 2.4 s/sample, 12.4GB). Epoch ≈ 3.5
  days; 20–50k-sample signal ≈ same-day. Untrained probe baseline: student
  answers "Hello! How can I help you today?" — fully blind, recall 0.0.
  Gotchas encoded in the code, do not regress:
  - `model.train()` is REQUIRED for HF gradient checkpointing to engage
    (eval-mode passes silently retain ~10GB of activations → OOM).
  - Generation must run in eval mode (train mode + grad ckpt forces
    use_cache=False and breaks decoding) — `teacher_mode()` handles both.
  - transformers v5 `apply_chat_template(tokenize=True)` returns a
    BatchEncoding, not an id list.
  - Teacher logits come from a forced no-grad forward AFTER generation
    (bf16, per-sample clones), not `output_logits=True` (fp32 per-step
    accumulation). KL uses a custom autograd Function with analytic
    backward (softmax(s) − softmax(t)) to avoid retaining fp32 buffers.
  - GDN fast path enabled (causal-conv1d 1.7.0 built against CUDA 13.1 on
    2026-09-16). Measured speedup over the torch fallback was minor at this
    scale (~10.5 vs ~11 s/step at batch 8) — decode is dominated by the
    248k-vocab head, attention layers, and generate() overhead, not GDN.
- **2026-09-16 (step 1000, L2)** — eval_kl 0.342→0.244; raw read entropy
  ~4.4→2.27 (~10 effective memories); probe recall online: item 0.53, year
  0.06, name/relname 0.03, amount 0.00. Acquisition order matches the
  semantic-content-first gradient; probes show confabulation replacing
  placeholders ("Mr. Moseley" for Obadiah). NOTE: Qwen3.5 tokenizes numbers
  per digit — amount/year recall requires ordered multi-token retrieval,
  the hardest case; amount>0 will be the cleanest retrieval evidence.
  `probe_mech.py` battery written (modes: meanpool/lasttoken/rank-k/
  del-binding/readmap); needs a free GPU — run on checkpoints in the gap
  before launching L0/L1. Training has NO resume; don't kill L2 casually.
- **2026-09-16 (step 1750, L2)** — eval_kl 0.186, recall 0.40 (item 0.97,
  relname 0.66, year 0.34, name 0.03, amount 0.00). Probe errors are
  prefix-correct/tail-confabulated ("Dear Obie" for Obadiah; "1973" for
  1971) — signature of partially-converged ordered multi-token retrieval,
  inconsistent with pure gist transport. relname≫name asymmetry: appears
  2x in probe + strong "Dear ___" retrieval cue — evidence redundant
  writes help (relevant to future update/decay design). Read entropy
  ~1.6 nats, sharper than ~90% of base-model attention rows at similar
  context length (measured: attn mean 2.90/median 3.02 nats) — sharpness
  emerging WITHOUT sparsity regularization, de-risking top-K scaling.
- **2026-09-18 (L2-readers @5000, 5090)** — FROZEN BASE + readers-only
  (--no-lora, 42M params) works: recall 0.60 and climbing (vs v1 LoRA arm
  0.875@4500; ~half speed). Gap is concentrated in arbitrary-symbol
  bindings (name 0.22, amount 0.06 vs v1 0.875/0.75) while semantic/
  structural bindings match (item 1.0, year 0.94, relname 0.78) ⇒ LoRA's
  main contribution = making token IDENTITIES linearly retrievable from
  layer-19 states. Same developmental sequence as v1 (placeholder →
  typed confabulation → retrieval), stretched ~2x. Next: (a) let it run,
  watch for plateau vs convergence; (b) detach arm is now the key
  discriminator (directed write-gradients vs mere co-adaptation);
  (c) tighten probe scoring to word-boundary before quoting numbers.
- **2026-09-17 (REPL, ckpt ~7k)** — single-turn: REASONING OVER RETRIEVED
  BINDINGS works: 4 name-age pairs retrieved with zero cross-binding
  confusion, then compared correctly (youngest). Normalized read entropy
  0.46 over a 43-memory bank (~5-6 effective memories/read). Multi-turn:
  incoherent, as expected — fully OOD (bank never held >1 chunk nor
  self-writes in training). Conclusion: the gap is distributional, not
  architectural → multi-turn training rises in v2 priority.
- **2026-09-17 (step 4500, L2)** — eval_kl 0.137, recall 0.875 (name 0.875,
  relname 0.75, AMOUNT 0.75, year 1.0, item 1.0). Ordered-digit recall
  through the bottleneck confirmed — the v1 core question is answered
  positive at this scale. Residual errors still prefix-correct/tail-lossy
  ("$34.00" for 344). City retrieved unprompted. Remaining for v1: the
  L0/L1 arms (train to matched step count) + mech battery on checkpoints.
- **2026-09-18 (L2-v1 stopped @14k; top-K + exploration on ckpt-14000)** —
  Stopped L2-v1 at 14k steps (eval_kl 0.113, recall 0.93); further gains
  marginal given the readers-only arm's trajectory. Machine freed for
  post-hoc work on ckpt-14000:
  - **Top-K works with margin** (`probe_mech.py --modes topkN`; ctx.top_k
    masks scores before softmax): full=0.94, topk8=0.93, topk4=0.94,
    topk2=0.93, topk1=0.90. Only `amount` degrades at K=1 (0.88→0.69) —
    multi-digit retrieval plausibly wants mass on several adjacent digit
    memories. Sparse/ANN reads are viable; exact K to be retuned on the
    readers-only model (higher read entropy). Sweep stopped after topk8;
    meanpool/lasttoken/rank-k still not run.
  - **Exploration harness** (`explore/explore.py`, machine-usable REPL;
    `explore_out/*.jsonl`). Findings, 8 probes/condition unless noted:
    - *multiturn*: pure data/question chunk split costs ~20pt (0.97→0.78);
      loss concentrates in bindings with no cue in the question chunk
      (name 1.0→0.38, amount 0.88→0.50) while year/item stay 1.0.
      position_ids continuation recovers about half (0.85) ⇒ position
      collision is real but secondary. Read-on-ingest is NOT the problem
      (0.85). An interposed assistant turn is the worst poison (0.57,
      year 1.0→0.38). q-only control 0.23 (= relname leaking from the ask).
    - *scale* (M name-age pairs): targeted "answer with just the number"
      queries fail even at M=2-4 (~25%) while list-all is near-perfect at
      M≤8 (1.0/0.88) and degrades by M=16-32 (ages 0.2-0.6, names 0.1 —
      typed confabulation: "Yola"/"Pere"/"Hans"). Wrong answers are digit
      blends not other pairs' ages (conf=0 throughout) — prefix/tail-lossy,
      not cross-binding confusion. See cue-style follow-up below.
    - *length* (probe early vs late in neutral filler): early placement
      degrades slowly (0.95@400tok, 0.90@800, 0.80@1400); late placement
      drops immediately past the 320-token training cap (0.82@400,
      0.75@800, 0.72@1400). Write-position OOD, not bank size, is the
      primary length bottleneck; name/year/item barely care, relname/amount
      carry the entire drop.
    - *cue follow-up* (`explore/explore_cue.py`): targeted queries fail because the
      READ QUERY IS COLD, not because the memory is gone. Same M name-age
      bank, "answer with just the number" vs "start your answer with the
      person's name" vs teacher-forcing "<Name> is": bare 4/9→2/9→0/9 at
      M=4/8/16, but sentence 6/6/5 and forced 7/6/5. Retrieval is driven
      by the generated prefix (the "Dear ___"→relname effect, now causal):
      once the name token is in the stream, the paired age is retrievable
      even at M=16. Answer-first formats were never trained (chatty KL
      targets always restate context before answering). Implication for
      v2: either train on short-answer formats, or rely on the model
      restating cues — and note wrong answers are digit blends, so a
      confidence/abstention signal may fall out of read sharpness.
    - *distract* (probe + N ultrachat turns in bank): total collapse at
      N=2 (recall 0.00) — the model coherently answers ONE distractor's
      prompt (same answer regardless of which probe is present). Mission
      selection from a multi-prompt bank is untrained and fails before
      retrieval does. Directly motivates the v2 "question references the
      target" / passage-bank training designs; top-K will not fix this.
- **2026-09-18 (round 2, `explore/explore2.py`)** — zero-training mitigation tests
  on ckpt-14000, all essentially NEGATIVE — the failures are distributional
  and need training, not inference tricks:
  - *think block* (instructed recall-dump before answering, M=8/16 pairs):
    does NOT rescue targeted queries (pair 2/6 vs bare 1/6) and the final
    stated answer gets WORSE (1/6 and 0/6 vs bare 3/6): the model produces
    a fluent, partially-confabulated list ("Nietzsche", "Pepper",
    "Gertrude" for NAMES entries) and then answers self-consistently from
    its own wrong list; it also drifts the question intent (answered "who
    is oldest" instead of "how old is X"). An untrained think block is
    bounded by list-recall accuracy AND compounds its errors. A trained
    recall-preamble (teacher-forced from constructed transcripts, not
    distilled teacher CoT) remains the interesting version.
  - *chunked ingestion* (1300-token late-placement turn split into ~300-tok
    separately-templated chunks): recall 0.17 vs 0.77 single-chunk. The
    multi-chunk regime hurts far more than write-position OOD; intent
    partially survives (it attempts the apology note) but binds to wrong
    chunks' content (weaves journal filler into the note). No deployment
    workaround via chunking until multi-chunk training exists.
  - *mission capture* (probe + 8 distractors): probe-last 0.00 — expected
    in hindsight, the read softmax is permutation-invariant over the bank,
    so CROSS-CHUNK ORDER DOES NOT EXIST in this architecture (recency is
    unrepresentable until order features are added; relevant to v2
    update-semantics). topk4 0.03 — sparse reads don't fix capture. A
    final redirect turn referencing the probe by name or by topic: 0.00
    both; both conditions answer the SAME distractor (deterministic
    salience capture, redirect ignored). USER_PROMPT_ALPHA-style
    referencing has no untrained foothold — it must be trained in.
    Strongest variant: teacher-forcing "Dear {relname}," under 8
    distractors still only reaches 0.27 (name 0.17, amount 0.17) — the
    model writes to the CORRECT addressee about the CAPTURED distractor's
    topic ("Dear Obadiah, ... regarding the SEIU picket line"). A local
    cue anchors local retrieval, but nothing gives generation a
    chunk-level coherence prior, so the salient chunk keeps supplying the
    mission. Hard-negative / distractor-bank training (v2 passage-bank
    curriculum) is precisely the missing pressure.
- **2026-09-16 (launch)** — L2 run (`runs/L2-v1`, batch 8) started. Step KL
  fluctuates ~0.4–0.6 (sample-dependent); judge progress by `eval_kl` on the
  held-out slice (every 250 steps ≈ 45 min), read_entropy vs its ~3.6
  near-uniform start, and probe recall. ~1.33 s/sample → 200k epoch ≈ 3 days.
- **2026-09-21** — Further ablations (Jeremy): gradients through writes DO
  matter for the recall plateau (end-to-end beats detached/frozen writer at
  convergence, not just in speed), LoRA matters, and MORE LoRA rank is
  valuable. Consequence: the "precomputable memories / reader-only" branch
  of the v2 ladder is deprioritized; budget for the 27B target assumes full
  backward (8N FLOPs/token) — see the scaling section. Still open: training
  set work (TRAINING_V2.md) and remaining ablations before scaling up.
- **2026-09-21 (v2 training plan)** — `TRAINING_V2.md` rewritten. Sources:
  WildChat (multi-turn, as-is history, random cutoff t), MuSiQue (multi-doc,
  chosen over HotpotQA: no single-paragraph shortcut), Qasper (long-doc,
  chosen over QuALITY: 1,585 papers at ~5k tokens with evidence
  annotations). Encoding: chunks tagged `[user turn k]:` / `[agent turn k]:`
  (student only; teacher transcript untagged), background passages
  ingested against an empty bank, positions restart per chunk, every chunk
  writes, write-gradients kept for gold + first 2 distractors only.
  Batches homogeneous in source and depth. Teacher inline, gold-context
  only. Expected 5–10× v1 per-sample cost → cloud GPUs for this run.
- **2026-09-21 (v2 harness prep)** — `MemoryReader` now runs the read
  through `F.scaled_dot_product_attention` (fused; the explicit
  (B,heads,T,N) softmax is built only for stats/maps/top-K, under no_grad
  unless top-K is on). Motivation: the explicit path stored the full score
  matrix for backward (reader sits outside HF's checkpointed region); at
  v2 sizes (T≈4k, N≈25k) that is ~2.4GB per site per sample vs 0.41GB
  measured for the fused fwd+bwd. Agreement with the old path is bf16-level
  (rel 5e-3 fwd and grads); ckpt-14000 32-probe recall 0.94 full / 0.94
  topk4, identical to the explicit run; step-0 identity test still exact.
  Harness v2 architecture agreed (Jeremy): dataset plugins yield one schema
  (`id, source, gold[], distractors[], turns[]` + optional meta), the
  engine derives the teacher transcript (gold prepended to the first user
  turn), owns chunking/tags/batching/eval/resume/DDP; probes are plugins
  too (supply samples in the same schema, receive student AND teacher
  completions + read stats, return a JSON blob to log). Turn tags are a
  flag so the ultrachat-only module can regress against the L2-v1 log
  exactly. Build order: engine + `datasets/ultrachat` → untagged regression
  → resume + DDP → tagged run → WildChat/MuSiQue/long-doc source (Qasper
  likely swapped for NQ-with-Wikipedia-pages or demoted to a small weight).
- **2026-09-21 (v2 harness built)** — `engine.py` + `train_v2.py` with
  plugin `sources/` (ultrachat) and `probes/` (bindings); contract and
  implementation notes in `TRAINING_V2.md`. Verified: v1 `training_step`
  and v2 `student_loss` produce the identical loss on one batch
  (`test_engine.py`); multi-chunk tagged sample runs fwd/bwd/generate;
  4-step smoke + resume clean. Write passes now stop at the write site
  (exact, ~37% cheaper) and logits are computed only at generated
  positions. Measured: teacher generation is ~85% of a v1-sized step
  (8.8 of 10.1 s at batch 8) — vLLM pregeneration moves up the list for
  the cloud run. 1k-step regression vs L2-v1 running.
- **2026-09-21 (LoRA rank ablation, 5090, Jeremy)** — v1-style run at
  LoRA r=64 (vs 16). Early trajectory is slightly SLOWER per step to a
  given loss / recall than r=16; running to convergence to see whether
  it ends at a higher recall plateau (the 2026-09-21 ablation note said
  more rank is valuable; this checks the endpoint rather than the speed).
  Result pending. The 5090 also hosts the llama.cpp teacher server for
  the local v2 runs (`--teacher-url`, Q8_0 GGUF, 16 slots).
- **2026-09-22 (v2 harness regression PASSED)** — `runs/L2-v2-regress`:
  new engine, ultrachat only, tags off, batch 8, teacher = llama.cpp Q8_0
  on dev-5090 (`--teacher-url`, prefetch 4). Trajectory vs L2-v1
  (eval_kl / probe recall): step 250 0.360/0.00 vs 0.342/0.00; 500
  0.308/0.08 vs 0.292/0.06; 750 0.281/0.07 vs 0.263/0.09; 1000
  0.255/0.15 vs 0.244/0.13. Same shape, same binding-acquisition order
  (item first), eval_kl ~4% above v1 throughout — consistent with fresh
  sampled eval targets (the v1 target file is not on this machine) and
  v1's later sampling/entropy changes; qualitative parity is the agreed
  bar. Teacher ceiling on the bindings probe: 1.0 on all five bindings.
  A local-teacher run (`L2-v2-regress-local`, stopped at 315) gave
  0.358 at step 250, i.e. the Q8 llama.cpp tokens are an equivalent
  target. Remote teacher on the shared 5090 gave no speedup (~200–250
  tok/s; single-stream decode 58 ms/step = time-sliced with the r=64
  ablation); expect >1k tok/s on an idle card. Next: `L2-v2-tags`
  (same run, `--turn-tags`) auto-launched to measure the tag cost.
- **2026-09-22 (turn-tag cost: none)** — `runs/L2-v2-tags` (identical
  to L2-v2-regress but `--turn-tags`): eval_kl 0.358 / 0.314 / 0.282 /
  0.254 vs untagged 0.360 / 0.308 / 0.281 / 0.255 at 250 / 500 / 750 /
  1000; recall 0.13 vs 0.15 at 1000. Tags are free at depth 1, as
  expected (their value is cross-chunk ordering, untested here). Both
  runs saved ckpt-500/1000. Tags stay on by default for the multi-source
  run.
- **2026-09-22 (all v2 sources smoke-tested on GPU)** — wildchat,
  musique, qasper, triviaqa each ran 20 steps at batch 2 with tags,
  chunk cap 8192, remote teacher, eval + probes: all clean. Long-doc
  sources: ~25–29k chunk tokens/step, banks up to 23.5k memories, peak
  18–20 GB at batch 2 on the 4090 (table in TRAINING_V2.md). Remote
  teacher client fixed: 4xx (prompt over the server's slot context) no
  longer retried; such prompts fall back to local generation. Harness is
  ready for the multi-source run; remaining before cloud: server `-c`
  large enough for WildChat prompts, DDP on >1 GPU, mixture weights.
- **2026-09-22 (copy source; mrope cache bug)** — Added the verbatim-copy
  source/probe pair (`sources/copy.py`, `probes/copy.py`; design in
  TRAINING_V2.md) to train content-addressed lookup + sequential readout.
  Found and fixed an engine bug on the way: Qwen3.5 `generate()` caches a
  per-batch `rope_deltas` on the inner model and later plain forwards
  reuse it (silently when batch sizes divide, a shape error otherwise —
  it crashed the first background pass of a different batch size after a
  probe generate). `engine.inner_model()` now clears it before every
  forward; regression added to `test_engine.py`. Next per Jeremy: remote
  training infrastructure, then the large run once the LoRA-rank
  ablation settles the rank.
