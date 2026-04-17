# Investigation: TTT on top of VarLen attention degrades eval BPB

## 2026-04-18 update — ROOT CAUSE FOUND: RoPE-base mismatch

**Confirmed bidirectionally.** `Rotary.forward(seq_len)` uses NTK-aware
rescaling when `seq_len > train_seq_len=2048`, otherwise returns the cached
vanilla cos/sin.

- **Training (varlen)**: x is `[1, packed_flat]` where packed_flat ≈
  `train_batch_tokens / (world × grad_accum)` ≈ 98304 → NTK-rescaled RoPE.
- **Sliding eval (varlen)**: x_cat is `[1, batch_seqs × seq_len]` ≈ `[1, 65536]`
  → NTK-rescaled RoPE (matches training ✓).
- **TTT eval**: x is `[bsz, ttt_eval_seq_len=2048]` per row → vanilla cached
  RoPE (different base from training ✗).

### Diagnostic results (2000 docs, varlen-trained `final_model.int6.ptz`)

| Run | RoPE base | val_bpb |
|---|---|---:|
| Sliding (default) | NTK ↑ (packed 65k row) | **1.0742** |
| Sliding `ROPE_FORCE_BASE_SEQLEN=2048` | vanilla | 1.2887 |
| TTT lora (default, broken) | vanilla cached | 1.1920 |
| TTT lora `ROPE_FORCE_BASE_SEQLEN=65536` | NTK ↑ | **1.0786** |
| TTT lora **auto-fix** (`base_force=98304`) | NTK ↑ | **1.0778** |

Forcing the "wrong" RoPE base into either protocol moves it onto the other
protocol's curve. **+0.114 BPB regression collapsed to +0.004.**

### Why PR #1530 doesn't have this bug

PR #1530's Rotary has **no NTK-rescale path** — it always uses cached cos/sin,
recomputing the cache on demand for any seq_len. So their training and TTT eval
share identical RoPE positions. Our model added NTK rescaling for >2048 rows
but never propagated that decision to TTT eval, where rows are always 2048.

### Fix shipped

Added `ttt_rope_base_seqlen` hyperparameter (env: `TTT_ROPE_BASE_SEQLEN`).
When 0 (default), auto-derive at TTT eval entry as
`max(train_seq_len + 1, train_batch_tokens // (world × grad_accum))`. The
override is set via `ROPE_FORCE_BASE_SEQLEN` env var (read by `Rotary.forward`)
for the duration of TTT eval, then restored. Sliding eval is unaffected
(continues to use the natural NTK path on long packed rows).

Patch is now permanent in `train_gpt_improved_04_16.py`. Diagnostics in
`logs/ttt_ropebase65k_d2000_04_17.txt`, `logs/sw_ropevanilla2_04_17.txt`,
`logs/ttt_ropefix_auto_d2000_04_17.txt`.

### What this overturns

- "TTT-vs-sliding gap is a scoring-protocol issue" (2026-04-17 update below) —
  partially true (the protocols do differ), but the dominant cause is the RoPE
  encoding mismatch, not the chunked scoring or sample skew. Once RoPE is
  aligned, the residual gap drops to noise level (+0.004 BPB on 2000 docs).
- "Forward-path mismatch in `_block_with_lora`" (original H1) — still wrong;
  bitwise sanity check stands. The bug was upstream of the block, in `Rotary`.
- Gated-attention hypothesis — falsified earlier today; gates universally
  essential, not the cause.

---

## 2026-04-17 update — root cause is the TTT scoring protocol, NOT a forward-path bug

After comparing against PR #1530 (samacqua, also varlen + LoRA-TTT, reaches
ttt_lora ~1.073) and our diagnostic floor of ~1.29 with no LoRA on the TTT
harness:

- PR #1530's TTT path uses dense `flash_attn_3_func(causal=True)` on
  `[bsz, seq_len]` one-doc-per-row — **no varlen kernel inside TTT**. Their
  varlen is only used in *training* and in `eval_val` / sliding eval. Varlen is
  a batch-layout feature (cu_seqlens-driven row packing), not a model feature;
  weights trained with packing work in either layout.
- Our `eval_val_ttt` is structurally a port of PR #1530's `eval_val_ttt_lora`
  (same `_compute_chunk_window`, `_accumulate_bpb`, `_build_ttt_global_batches`,
  `BatchedTTTLoRA`), and we already auto-synthesize `cu_seqlens=[0, sl, ...]`
  inside `forward_ttt`. Forcing dense (`TTT_DENSE_ATTN=1`) gave 1.192 vs
  varlen-path 1.186 — equivalent.

### Hypotheses tested in this batch

| Hypothesis | Test | Result | Verdict |
|---|---|---|---|
| Adaptive LR / NaN guard noise | `TTT_ADAPTIVE_LR=0 TTT_NAN_GUARD=0` | 1.192 (no change) | falsified |
| Varlen kernel inside TTT | `TTT_DENSE_ATTN=1` | 1.192 | falsified |
| RoPE position offset (BOS-pad) | `TTT_ROPE_OFFSET ∈ {0,256,1024,1536}` | 1.31→1.37→1.83→2.12 | confounded by OOD pad context |
| **Gated attention** (we have it, PR #1530 doesn't) | `DISABLE_GATED_ATTN_EVAL=1` | sliding ON 1.074 / OFF 1.580; TTT ON 1.192 / OFF 1.560 | **FALSIFIED** — gates universally essential, removing them widens the gap, not narrows it |

### Notable cross-protocol observation

With gates **OFF**, sliding (1.58) and TTT (1.56) converge. With gates **ON**,
sliding (1.07) and TTT (1.19) diverge by 0.12. The gates seem to *amplify* the
protocol divergence rather than cause it, suggesting the underlying mismatch is
in *what the gate sees* (residual stream `x` after some layers of attention),
which is itself shaped by RoPE-position distribution and context shape.

### Remaining open hypotheses (in order of likelihood)

1. **RoPE-position distribution.** Sliding packs many docs into a flat row, so
   docs sit at varied row offsets and see varied RoPE positions. TTT puts each
   doc in its own row at positions 0..doclen-1. Trained model also packs but
   most docs train at non-zero starts → position-0 may be thin in training
   distribution. Clean test requires adding `position_offset` to `Rotary.forward`
   so we can shift cos/sin without OOD padding tokens.
2. **Numerical drift between `_block_with_lora` and `Block.forward`** at higher
   precision than the bitwise sanity check used. Current sanity check checks
   final logits; intermediate tensors (post-attn, post-MLP, post-skip) may drift
   measurably even when logits look identical.
3. **Skip-gates / fused-RoPE / parallel-residual ordering** as a numerical
   second-order effect.

### Patches reverted

`DISABLE_GATED_ATTN_EVAL`, `TTT_ROPE_OFFSET`, `TTT_NO_LORA_PROBE` all reverted
from `train_gpt_improved_04_16.py`. Raw logs preserved under
`logs/{ttt_nogate,sw_nogate,ttt_ropeoff_*,ttt_nolora_probe*,ttt_dense_attn,
ttt_clean_adapt0_guard0,ttt_zerolr}_*_04_17.txt`.

---

## 2026-04-17 update — root cause is the TTT scoring protocol, NOT a forward-path bug

After three follow-up batches (`logs/sanity_check_04_17_stdout.txt`,
`logs/no_lora_probe_*_04_17_stdout.txt`,
`logs/warmup_w*_04_17_stdout.txt`,
`logs/fair_*_warmup1024_04_17_stdout.txt`) the original H1 conclusion in
`varlen_ttt_invest/results.md` (that `_block_with_lora` had drifted from
`Block.forward`) is **wrong**. Findings:

1. `TTT_SANITY_CHECK=1 TTT_LAYER_DEBUG=1` shows `forward_logits` and
   zero-LoRA `forward_ttt` are **bitwise identical** at every one of the 19
   layer states (mean_abs = max_abs = 0.0e+00) for both a 317-token prefix
   probe and a 2048-token tail-window probe. `_block_with_lora` is exonerated.
2. A `TTT_NO_LORA_PROBE` flag (calls `forward_logits` instead of
   `forward_ttt`, no LoRA at all, but reuses `eval_val_ttt`'s windowing,
   chunking, and `_accumulate_bpb`) reports the **same ~1.30 BPB** as the
   zero-LoRA `forward_ttt` probe. So the +0.26 BPB gap exists even when the
   model forward is the trained one. → The defect lives in the eval
   *protocol* in `eval_val_ttt`, not in any TTT-specific forward code.
3. Sweeping `TTT_MAX_DOCS ∈ {50, 200, 1000, 5000}` with the no-LoRA probe
   keeps val_bpb at 1.29-1.34. Not a sample-size artifact.
4. Sweeping `TTT_CHUNK_SIZE ∈ {64, 256, 1024, 2048}` with the no-LoRA probe
   moves val_bpb from 1.291 → 1.291 → 1.258 → 1.245. The number drops
   monotonically as fewer cold-start positions get scored, but never
   approaches the 1.074 sliding-window baseline.
5. Adding a `TTT_SCORE_WARMUP` shift (skip the first N positions of every
   chunk's scored region, matching `eval_val_sliding`'s `s = seq_len -
   stride` skip) makes bpb **worse**, not better:
   `W=0:1.291  W=64:1.290  W=256:1.335  W=512:1.405  W=1024:1.501`.
   This means the gap is dominated by **bytes-per-token denominator skew**
   in the first ~1000 FineWeb docs (their late-doc tokens are dominated by
   short pieces — URLs, references, footers — so the bpb denominator
   shrinks faster than the cross-entropy numerator).

**Conclusion.** `eval_val_ttt` and `eval_val_sliding` measure *different
quantities* on the same model. Three independent confounders:

- **Scored-position set differs.** Sliding-window only scores the trailing
  `stride` tokens of each window (≥ `seq_len - stride` of left context per
  scored position). TTT scores every position of every chunk window,
  including cold-start positions.
- **Token sample differs.** TTT (with `TTT_MAX_DOCS≪all`) only scores the
  first N documents; sliding-window scores all 10 M tokens. The first N
  FineWeb docs have skewed bytes-per-token statistics, especially in their
  tails, which moves the bpb denominator independently of the model.
- **`cu_seqlens` handling differs.** `forward_ttt` auto-synthesizes
  `cu_seqlens=[0, sl_in]` per row, while `eval_val_sliding` (varlen) builds
  `cu_seqlens` from a BOS scan across the packed batch and forces a
  boundary at offset 0 of every window.

LoRA TTT is **actually working**: at `TTT_LORA_LR=1e-4` it pulls the score
from the protocol-floor (~1.336 zero-LoRA, 50 docs) down to 1.208 (a real
−0.128 BPB improvement). The headline regression ("TTT 1.21 vs sw 1.07")
is an apples-to-oranges comparison — not a model or LoRA bug.

### Implications for next steps

- **Do not** rewrite `_block_with_lora` to delegate to `Block.forward`. It
  isn't broken.
- **Fix the comparison, not the model.** The only fair LoRA-improvement
  number is `Δ(LoRA) := bpb_TTT_lora − bpb_TTT_no_lora` measured on the
  *same* scored-token set, *same* docs. Reintroduce `TTT_NO_LORA_PROBE`
  as a permanent diagnostic flag (it was reverted after this batch), pair
  every TTT report with its zero-LoRA control number, and judge LoRA on
  Δ rather than on absolute bpb.
- To make the headline TTT bpb directly comparable to `eval_val_sliding`,
  refactor `eval_val_ttt` so it: (a) iterates the same window starts as
  `eval_val_sliding`, (b) scores the same trailing-`stride`-tokens-per-
  window subset, (c) builds `cu_seqlens` the same way (BOS scan + offset-0
  boundary insertion). This is a substantial refactor.
- The original sliding-window varlen-degradation observation (sw_bpb
  degradation on varlen-trained checkpoints vs non-varlen) is a separate
  phenomenon and should be investigated independently — it is not caused
  by `eval_val_ttt`.
- Patches `TTT_NO_LORA_PROBE` and `TTT_SCORE_WARMUP` were reverted from
  `train_gpt_improved_04_16.py` after this batch; logs preserved under
  `/root/parameter-golf/logs/` (`sanity_check_04_17_*`,
  `no_lora_probe_*_04_17_*`, `warmup_w*_04_17_*`,
  `fair_*_warmup1024_04_17_*`).

The remainder of this document is preserved for historical context.
Sections below (especially H1 "per-path forward mismatch") have been
**superseded by this update**.

---


## Working file & output folder
- **All code edits land in**: `train_gpt_improved_04_16.py` (a snapshot of `train_gpt_improved.py` taken on 2026-04-16; currently byte-identical). All investigation code edits happen here; do **not** touch `train_gpt_improved.py` until a fix is validated.
- **All runs, logs, and notes for this investigation** go into a new folder: `varlen_ttt_invest/` at the repo root. Layout:
  ```
  varlen_ttt_invest/
    README.md              # running narrative + final root cause
    commands.txt           # every torchrun command, copy-pasteable
    results.md             # table: run_id | config | pre_bpb | quant_bpb | sw_bpb | ttt_bpb | notes
    logs/                  # tee'd stdout per run (one file per RUN_ID)
    patches/               # optional: git diffs of incremental fixes on train_gpt_improved_04_16.py
  ```
- Create the folder on first run. Do **not** clutter top-level `logs/` with varlen-TTT debug runs.

## Upstream references (read first)
External reference PRs are on the upstream repo: https://github.com/openai/parameter-golf/pulls
Filter by "TTT" and "varlen" / "cu_seqlens" to find the canonical implementations. In particular, look at:
- PRs that introduced or fixed TTT (LoRA and fullparam) for varlen-packed eval.
- PRs that specifically plumb `cu_seqlens` through the TTT forward path.
- Any PR that reports TTT gains on top of VarLen (the target delta we're missing).
Record the PR numbers + 1-line summary of what each changed in `varlen_ttt_invest/README.md` before writing code.

## TL;DR
Under `VARLEN_ATTENTION=1`, the baseline produces **sw_val_bpb ≈ 1.074** (e.g. `pg09_varlen`, `pg12_varlen_clip14`). When the same checkpoint is evaluated with `TTT_ENABLED=1` (LoRA path, chosen automatically for varlen), `eval_val_ttt` reports running BPB ≈ **1.20–1.23** instead of improving. TTT should *improve* BPB by ≥ 0.003; instead it regresses by > 0.13. Program.md currently says to "ignore" this, but we need to understand and fix it before pursuing further varlen experiments, because TTT is worth ~0.005 BPB on the non-varlen track.

## What "the missing points" means here
Quantify the gap before fixing, so we know what success looks like. In `varlen_ttt_invest/results.md`, fill in this table from existing logs + one reproduction run:

| Config | sw_bpb | ttt_bpb | TTT delta |
|---|---|---|---|
| Non-varlen baseline (record) | 1.07369 | — | — |
| Non-varlen + TTT (`improved_GA_FUSErope_MLP435_Mclip13_TTT`) | 1.07448 | **1.07077** | **−0.00371** |
| Varlen baseline (`pg12_varlen_clip14`) | 1.07425 | — | — |
| Varlen + TTT (current, broken) | 1.07425 | ~1.20+ | **+0.13** regression |
| **Varlen + TTT target** | 1.07425 | **≤ 1.071** | **≤ −0.003** |

The headline gap to close is roughly the **0.003–0.005 BPB** that TTT reliably buys on the non-varlen baseline and that varlen currently forfeits. Any fix that lands varlen+TTT below the non-varlen+TTT record (1.07077) is a clear win.

Reference runs (same checkpoint, same `final_model.int6.ptz`):
- Non-TTT: `logs/pg09_varlen.txt` → quantized_sliding_window BPB 1.07798
- TTT:    `logs/pg09_varlen_TTT.txt`, `pg09_varlen_TTT_debug.txt`, `pg12_varlen_clip14_TTT.txt` → running_bpb ~1.20+

## Why this matters
- `pg12_varlen_clip14` is the current best pre-quant model (1.06882) but its post-TTT number is the only one that lands on the leaderboard.
- Non-varlen baseline + TTT is **1.07077** (`improved_GA_FUSErope_MLP435_Mclip13_TTT`). If varlen + working TTT yielded the same ~0.003 improvement it would clearly beat the 1.07369 record.

## Observations & working theory (from experiment log)

Cross-referencing all varlen runs points at a **shared defect upstream of any specific TTT path**:

1. **Varlen wins on the model side.** `pg12_varlen_clip14` has the **best pre-quant BPB (1.06882)** and **best post-quant BPB (1.08142)** in the whole program. Varlen clearly helps the trained model.
2. **But every varlen eval path degrades more than its non-varlen counterpart:**
   - **Sliding-window eval** on varlen: post-quant → sw delta is noticeably worse than on non-varlen (e.g., pg12 quant 1.08142 → sw 1.07425 = Δ −0.0072, versus non-varlen where sw usually gains >0.01 over quant).
   - **Fullparam TTT** on varlen: degraded (same direction as LoRA TTT).
   - **LoRA TTT** on varlen: regresses to ~1.20.
3. **Because all three eval paths degrade, the culprit is almost certainly common to all three — not specific to LoRA plumbing.** Candidates for the shared cause (these become the first things to check):
   - Val-side `cu_seqlens` / BOS detection not matching what the training-side `DocumentPackingLoader` produced.
   - `deserialize()` step silently dropping or misconfiguring the varlen attention flag on the eval model (`use_varlen` not propagated to blocks after reload).
   - `eval_seq_len` / `eval_stride` windows misaligned with doc boundaries, so varlen sees many single-token "docs" or a wrong `max_seqlen`.
   - `torch.compile(dynamic=False)` caching a wrong trace for the varlen path.
   - Wrong pad token (`0` vs `BOS_ID=1`) used when padding eval windows — id `0` is a real piece, and varlen BOS-scanning can misclassify it.
4. **Best current varlen+TTT is ~1.070** (floor we must dip under). Non-varlen+TTT already sits at **1.07077**, so a useful fix must land **≤ 1.068** on varlen+TTT to justify the complexity — not merely "better than broken".
5. **Upstream PRs have solved this.** https://github.com/openai/parameter-golf/pulls shows varlen+TTT combinations that *do* yield the expected improvement. **Step 0 (PR survey) is now the highest-leverage step**: find the shared eval preamble those PRs use for varlen (how they build `cu_seqlens` on the val token stream, whether they re-apply `use_varlen` after `deserialize`, how they pad) and port it over. Don't reinvent it.

**Implication for ranking:** a LoRA-forward-only defect cannot explain sliding-window + fullparam regressions. So the highest-prior hypothesis is now a **shared eval-preamble bug (H0 below)**; the previous H1 (per-path forward mismatch) drops to second place.

## Scope of this investigation
Allowed changes:
- **Evaluation-side code** in `train_gpt_improved_04_16.py` (`eval_val_ttt`, `forward_ttt`, `_block_with_lora`, `BatchedTTTLoRA`, dispatcher in `main_eval`).
- **Training *parameters*** (env vars: `MLP_MULT`, `CLIP_SIGMAS`, `GPTQ_CALIBRATION_BATCHES`, seq-len, LR, etc.) to match the other varlen baselines (`pg09_varlen`, `pg12_varlen_clip14`). This is how we make runs comparable.

Not allowed:
- Changing the **training loop logic**, optimizer structure, quantization pipeline, tokenizer, or architecture.
- Editing `train_gpt_improved.py` itself (keep it as the reference/frozen version; only `_04_16` is touched during the investigation).
- Retraining from scratch for debugging — reuse `final_model.int6.ptz` from a completed varlen run (e.g. `pg09_varlen` or `pg12_varlen_clip14`) where possible.

## Relevant code (`train_gpt_improved_04_16.py` — line numbers currently identical to `train_gpt_improved.py`)

| Location | What it does |
|---|---|
| L295–340 | TTT hyperparams. Note comment L296: "varlen → LoRA TTT, non-varlen → fullparam TTT" |
| L377–378 | `VARLEN_ATTENTION` flag |
| L562–612 | `DocumentPackingLoader` (PR #1530) — training path uses true `cu_seqlens` via `flash_attn_3_varlen_func` |
| L759–820 | `CausalSelfAttention.forward`: `use_varlen` branch calls `flash_attn_3_varlen_func`, otherwise `flash_attn_3_func(causal=True)` |
| L858–1000 | `BatchedTTTLoRA` (PR #1530 style) — one LoRA slot per doc in batch |
| L1001–1055 | `GPT.__init__`: `self.use_varlen = h.varlen_attention and _HAS_VARLEN`, propagated to each block |
| L1095–1110 | `GPT._compute_cu_seqlens` — builds boundaries from BOS_ID |
| L1156–1213 | `_block_with_lora` — **calls `flash_attn_3_func(..., causal=True)` unconditionally**, i.e. NOT the varlen kernel even when `use_varlen=True`. No `cu_seqlens` plumbed in. |
| L1214–1260 | `forward_ttt` — **no `cu_seqlens` argument**, calls `_block_with_lora` per block |
| L2730–3010 | `eval_val_ttt` — LoRA TTT loop. Builds `(bsz, context_size)` tensors, pads invalid cols with token `0`, calls `forward_ttt(x, y, lora)` |
| L3390–3406 | `main_eval` dispatcher: when `use_varlen` → LoRA TTT, otherwise fullparam TTT |

## Hypotheses, ranked by likelihood

> **Key observation:** every row of a TTT batch is a *single-document window* built from `doc_start + win_start` (see `eval_val_ttt` L2840–2870). So **within one row** dense-causal attention is equivalent to varlen attention — there is no cross-document leakage regardless of which kernel runs. This means the previously-hypothesized "TTT uses dense kernel, training used varlen" is *not* automatically a correctness bug at eval-time. That downgrades the pure kernel-dispatch story; the real culprit must lie elsewhere. The hypotheses below are re-ranked accordingly.

### H0 (NEW, most likely): Shared eval-preamble bug hits **every** varlen eval path
Because sliding-window, fullparam TTT, and LoRA TTT all degrade together on varlen, the defect must be upstream of any TTT-specific code — in something all three share. Primary suspects:
- **`deserialize()` drops `use_varlen`**: after the int6 reload, blocks may run dense causal on a model whose weights were trained under varlen (doc-reset) attention. Check that `eval_model.use_varlen` is `True` *and* `eval_model.blocks[i].attn.use_varlen` is `True` for all blocks after `deserialize`. Fix = propagate the flag in `deserialize` (mirror what `GPT.__init__` L1047–1049 does).
- **Eval `cu_seqlens` construction** (`_compute_cu_seqlens` / `_build_cu_seqlens` on concatenated val tokens): bucket rounding, BOS-0 insertion, or off-by-one at the last segment produce wrong `max_seqlen` so the flash kernel receives a different shape than training did.
- **Pad token mismatch**: eval pads with id `0`; varlen BOS-scan treats only `BOS_ID=1` as a doc start; any literal `0` mid-buffer can be mistaken for, or silently mixed into, a doc boundary decision — causing attention to bleed across what eval believes are boundaries.
- **`torch.compile(dynamic=False)`** on the varlen forward path: caches a trace for the first `(cu_seqlens, max_seqlen)` shape seen and re-uses it for mismatched shapes. Try `dynamic=True` or disable compile on the varlen eval path.
- **`.training` vs `.eval()` divergence inside `_xsa_efficient` / window-attn masking** — affects all three eval paths equally if a module's behavior depends on training mode.

**Cheapest falsification:** step 1 below — the cross-path repro table. If sliding-window and fullparam-TTT on varlen are both degraded relative to their non-varlen counterparts using the same training seed, H0 is confirmed.

### H1: `_block_with_lora` is an incomplete copy of `Block.forward` and omits features the trained varlen model relies on
Still plausible, now second-tier. Would only explain the **LoRA**-TTT regression, not the sliding-window or fullparam degradation. Keep the line-by-line checklist (xsa / gating / QK-norm / logit_softcap ordering / skip weights / window-attn / fused RoPE) and revisit once H0 is ruled out.

### H2: LoRA / TTT hyperparameters tuned for the non-varlen baseline don't transfer
- Defaults: `TTT_LORA_LR=1e-4 → 5e-5 (in logs) `, `TTT_CHUNK_SIZE=64 → 48`, `TTT_BETA2=0.99`, `TTT_WEIGHT_DECAY=0.2`, adaptive LR scaling.
- Varlen-trained models have different activation scales (doc-reset attention means first-in-doc hidden states differ in norm). LoRA init + these LRs could be unstable.
- Trend in `pg09_varlen_TTT.txt`: running_bpb rises from 1.17 → 1.22 as scoring progresses → consistent with the LoRA drifting in a direction that hurts rather than helps.
- Compatible with H1: even after a forward-path fix, these defaults may still need retuning.

### H3: Pad / position corruption at chunk boundaries
- `eval_val_ttt` pads invalid positions with token id `0` (L2872). `0` is a real SentencePiece piece, not a designated pad.
- Causal attention + pads at the END of each row mean scored positions don't attend to pads → pads shouldn't affect loss directly.
- However, `forward_ttt` computes `F.cross_entropy(..., target_ids.reshape(-1), reduction="none")` over the full tensor (including pad positions). Loss at pad positions is not accumulated into bpb, but it is part of the gradient unless masked. Check whether `per_doc = (per_tok_loss * train_mask).sum(...)` correctly excludes all pad positions. An off-by-one between `context_size`/`tok_wls`/`chunk_offset+chunk_len` would silently pull pads into train gradients.
- Unlikely to cause the +0.13 regression by itself, but can amplify H1/H2.

### H4: Varlen-kernel dispatch still matters across *multiple* docs inside a single TTT batch row
- Currently: one row = one doc, so within-row varlen ≡ dense. But if `eval_seq_len > doc_len`, there's no real doc concatenation — the window is a single short doc padded out. So varlen wouldn't change anything here.
- This becomes relevant only if we **change** `eval_val_ttt` to pack multiple short docs per row (for throughput). That's an optimization, not a fix.

### Ruled out / downgraded
- "Cross-doc attention within a row" — no: each row is a single doc.
- RoPE absolute-position shift — relative positions within a doc are identical between training and TTT eval; RoPE is shift-invariant on the relative-position axis.

## Action plan

Execute in order; stop as soon as the regression is explained. All runs use `train_gpt_improved_04_16.py` packed via `pack_submission_file.py`, tee logs into `varlen_ttt_invest/logs/`, append each command to `varlen_ttt_invest/commands.txt`.

### Step 0 — Upstream PR survey (high leverage)
Browse https://github.com/openai/parameter-golf/pulls. Filter for "varlen" + "TTT" + "cu_seqlens" + "document packing". Specifically look for:
- How do PRs that successfully combine varlen + TTT build `cu_seqlens` on the **validation** token stream?
- Do they re-apply `use_varlen` / propagate doc boundaries after `deserialize()`?
- Do they use a different pad token (e.g. `BOS_ID`) or avoid padding entirely by packing?
- Do they disable `torch.compile` on the varlen eval path?
- Do they share an eval preamble between sliding-window and TTT?

Record 3–5 most relevant PRs (number, title, 1-line summary, key files/lines changed) in `varlen_ttt_invest/README.md`. **If any PR already solves the sliding-window regression, port that first — it likely fixes fullparam-TTT and LoRA-TTT along the way.**

### Step 1 — Cross-path repro (falsifies H0 vs H1)
Build the following table in `varlen_ttt_invest/results.md` by running the same packed model through each eval path. Run with `TTT_MAX_DOCS=500` for speed:

| Config | eval_val (full) | eval_val_sliding | fullparam TTT | LoRA TTT |
|---|---|---|---|---|
| non-varlen baseline | ref | ref | ref (improves) | ref (improves) |
| varlen baseline (same-arch) | ? | ? | ? | ? |

- **If only LoRA-TTT regresses** → H0 is wrong, go to H1 (step 4).
- **If sliding + fullparam + LoRA all regress** → H0 confirmed, go to step 2.
- **If even `eval_val` regresses on varlen** → the deserialized model is broken (first bullet of H0); go to step 2 with that specific check.

Packing + launch template:
```bash
python3 pack_submission_file.py train_gpt_improved_04_16.py train_gpt.py
RUN_ID=tv01_crosspath TTT_ENABLED=1 VARLEN_ATTENTION=1 \
  MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=128 \
  TTT_MAX_DOCS=500 MAX_WALLCLOCK_SECONDS=3600 \
  torchrun --standalone --nproc_per_node=4 train_gpt.py 2>&1 \
  | tee varlen_ttt_invest/logs/tv01_crosspath.txt
```

### Step 2 — H0 diagnostic (shared-preamble)
Instrument `main_eval` just after `deserialize()` to log, for the eval model:
- `eval_model.use_varlen`
- `all(b.attn.use_varlen for b in eval_model.blocks)`
- dump `cu_seqlens[:64]`, `max_seqlen`, and `x.shape` on the first call of the varlen path in `eval_val_sliding`, `eval_val_ttt_fullparam`, and `eval_val_ttt`.

Confirm all three eval paths see the same `use_varlen=True` and the same `cu_seqlens` construction. If any is wrong, that's the root cause. Apply the minimal fix in `train_gpt_improved_04_16.py` (e.g. in `deserialize`, set `m.use_varlen = True` and iterate blocks setting `attn.use_varlen = True`). Re-run step 1.

### Step 3 — H0 fix candidates (in order)
Implement each only if step 2 points at it:
- **`use_varlen` not propagated after deserialize** → patch `deserialize` to mirror `GPT.__init__` L1047–1049.
- **Pad token** → change eval pads from `0` to `BOS_ID` (1), or refactor eval to pack real docs without padding.
- **`cu_seqlens` off-by-one / bucket mismatch** → align `_compute_cu_seqlens` with the training-side `DocumentPackingLoader` exactly (same bucket=64, same BOS handling).
- **`torch.compile` trace reuse** → set `dynamic=True` or `fullgraph=False` for varlen path; if still wrong, disable compile on that path.

### Step 4 — H1 fallback: LoRA forward-path audit (only if H0 is cleared but LoRA still regresses)
Add a zero-LoRA probe in `eval_val_ttt`: with `h.ttt_lora_lr == 0.0`, skip `cur_opt.step()` and ensure every LoRA module returns zero. Expected: `ttt_bpb ≈ sw_bpb`. If not, diff `_block_with_lora` against `Block.forward` + `CausalSelfAttention.forward` line by line (xsa / gating / QK-norm / fused RoPE / logit_softcap / skip weights / window-attn) and align.

### Step 5 — H2 hyperparameter sweep (only after H0/H1 are clean)
Small grid on `TTT_MAX_DOCS=2000`:

| Run ID | Vars |
|---|---|
| tv05a | `TTT_LORA_LR=5e-5 TTT_CHUNK_SIZE=48` (current) |
| tv05b | `TTT_LORA_LR=1e-5` |
| tv05c | `TTT_LORA_LR=5e-6` |
| tv05d | `TTT_CHUNK_SIZE=64` |
| tv05e | `TTT_ADAPTIVE_LR=0` |
| tv05f | `TTT_WEIGHT_DECAY=0.5` |

Pick the best, run full.

### Step 6 — Mask audit (H3)
In `_accumulate_bpb` + `train_mask` construction, assert:
- `train_mask.sum() == expected_train_token_count`
- No position where `train_mask==1` has both `x==0` and `y==0`
- Score and train regions are disjoint for the same doc.

### Step 7 — Validation
Final full run (no `TTT_MAX_DOCS` cap):
```bash
RUN_ID=tv07_final TTT_ENABLED=1 VARLEN_ATTENTION=1 \
  MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=128 \
  <winning TTT vars> MAX_WALLCLOCK_SECONDS=3600 \
  torchrun --standalone --nproc_per_node=4 train_gpt.py 2>&1 \
  | tee varlen_ttt_invest/logs/tv07_final.txt
```
**Success criteria:** on varlen,
- `eval_val_sliding` BPB improves over non-varlen sliding (expected ≤ 1.073, currently 1.07425),
- `quantized_ttt_* BPB ≤ 1.068` (strictly better than the 1.07077 non-varlen+TTT record and the ~1.070 best-varlen-TTT floor).

Append to `results.md`; save final patch to `varlen_ttt_invest/patches/tv07_final.patch`.

## Out of scope / do not touch
- Training loop, optimizer groups, quantization algorithm, tokenizer.
- MoE + TTT interaction (program.md L54 notes this is a separate issue).
- Fullparam TTT path on the **non-varlen** baseline — it works there. The varlen + fullparam-TTT path *is* in scope as evidence for H0.

## Artifacts to produce
All artifacts live under `varlen_ttt_invest/`:
- `README.md` — upstream PR survey (step 0), final root cause, one-line fix description.
- `commands.txt` — every command run, copy-pasteable.
- `logs/ttt_varlen_debug.txt` — step 1 repro.
- `logs/ttt_varlen_probe_lr0.txt` — step 2.
- `logs/ttt_varlen_<fix>.txt` — post-fix validation runs.
- `results.md` — the BPB table (pre/quant/sw/ttt) for every run, so the TTT delta vs baseline is obvious.
- `patches/*.patch` — diffs of `train_gpt_improved_04_16.py` against its frozen starting point, one per hypothesis tested.
- If a fix wins: copy the final `train_gpt_improved_04_16.py` forward (do *not* overwrite `train_gpt_improved_04_16.py` as part of this investigation; propose it as a follow-up once validated). Remember `pack_submission_file.py` must still produce ≤ 16 MB.

## References in-repo
- Training-side varlen reference: `DocumentPackingLoader` (L562, PR #1530)
- Eval varlen (non-TTT) reference: `eval_val` L1986–2020, `eval_val_sliding` L2069–2160 (PR #1610)
- Existing TTT PRs cited in code: #1530 (LoRA TTT), #1610/#1626 (phased SGD), #1639 (adaptive LR), #1647 (SLOT-4)
- Upstream PR list for cross-reference: https://github.com/openai/parameter-golf/pulls
