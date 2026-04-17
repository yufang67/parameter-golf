# Investigation: TTT on top of VarLen attention degrades eval BPB

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
