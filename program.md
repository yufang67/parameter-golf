# Experiment Program

**Goal:** Lower `val_bpb` on `train_gpt_improved_04_16.py`. One variable at a time, record everything.

## Setup

| Item | Value |
|------|-------|
| GPUs | 4× A100 80GB PCIe |
| Env | `.venv` (activated) |
| Data | `./data` (sp8192) |
| Wallclock | `MAX_WALLCLOCK_SECONDS=3600` |

## Baselines (best runs)

### Non-varlen track

| Metric | 16MB-valid baseline | Best with TTT (over budget) |
|--------|---------------------|------------------------------|
| Run | `improved_GA_FUSErope` | `improved_GA_FUSErope_MLP435_Mclip13_TTT` |
| Sliding-window BPB | **1.07369** ✅ | **1.07077** ⚠ (16.83 MB) |
| Pre-quant BPB | 1.07873 | 1.07608 |
| Quant BPB | 1.09031 | 1.08737 |
| Total artifact | 15,989,467 B ✅ | 16,827,813 B ⚠ |
| Steps | 5,445 | 4,974 |
| Config | `GATED_ATTENTION=1 FUSED_ROPE=1` | `+ MLP_MULT=4.35 MATRIX_CLIP_SIGMAS=13 TTT_ENABLED=1` |

### Varlen track (post RoPE-base TTT fix)

| Metric | 16MB-valid baseline | Best pre-quant (over budget) | Best with TTT (over budget) |
|--------|---------------------|------------------------------|------------------------------|
| Run | `pg11_varlen_gptq192` | `pgm_loopat0_5` | `pgm_loopat0_5` (TTT on same ckpt) |
| Sliding-window BPB | **1.07722** ✅ | 1.07129 ⚠ | — (TTT path only) |
| TTT BPB (`quantized_ttt_lora`) | — | — | **1.06919** ⚠ |
| Pre-quant BPB | 1.07000 | **1.06630** | 1.06630 |
| Quant BPB | 1.08440 | 1.07897 | 1.07897 |
| Total artifact | 15,977,377 B ✅ | 16,525,791 B ⚠ | 16,525,791 B ⚠ |
| Steps | 4,883 | 5,716 | 5,716 |
| Config | `+ VARLEN_ATTENTION=1 GPTQ_CALIBRATION_BATCHES=192` | `+ VARLEN_ATTENTION=1 LOOP_EMBEDDINGS=1 ENABLE_LOOPING_AT=0.5` | `+ TTT_ENABLED=1` |

**Headline:** the best leaderboard-eligible result is still the non-varlen 1.07369. Varlen wins on the model side (best pre-quant **1.06630** via `pgm_loopat0_5`) and now wins post-TTT (**1.06919** ≪ varlen sliding 1.07129), but all varlen winners currently bust the 16 MB budget (~0.5MB over). Closing the artifact-size gap is the highest-leverage open task.

## Workflow

```bash
# 1. Pack
python3 pack_submission_file.py train_gpt_improved_04_16.py train_gpt.py

# 2. Run without TTT
RUN_ID=<name> <ENV_OVERRIDES> MAX_WALLCLOCK_SECONDS=3600 \
  torchrun --standalone --nproc_per_node=4 train_gpt.py 2>&1 | tee logs/<name>.txt

# 3. Record — append command to command.txt, update logs/results_summary.md

# 4. Rerun the best with TTT
```

## Rules

- Pack before every run.
- Always `tee` to `logs/`.
- One variable per experiment. Name `RUN_ID` after what changed.
- Sequential: finish → record → wait for next.

## Experiments Plan
-  baseline is 
```
MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=128 TTT_ENABLED=1 MAX_WALLCLOCK_SECONDS=3600 torchrun --standalone --nproc_per_node=4 train_gpt.py
```

### XSA depth coverage (`XSA_LAST_N`)
- **What it controls:** number of *final* attention layers that apply Excess-Subtracted Attention (XSA), which removes the `v_t` self-projection from `y_t` after FlashAttention. Cheap (one normalize + dot + subtract per token, GQA-aware), no extra parameters.
- **Current default:** `XSA_LAST_N=11` with `NUM_LAYERS=11` → applied to **all** layers.
- **Hypothesis / rationale:**
  - Cost is negligible vs. attention FLOPs, so downside is only a possible loss regression.
  - Classical argument for restricting to top-K (early layers want the self-copy to inject `V` into the residual stream) is weaker here: 11 layers is shallow, and `resid_mix` + U-Net skips already feed embeddings forward, reducing reliance on the attention self-copy.
  - Worth re-ablating whenever `NUM_LAYERS` changes, especially for deeper configs where sparing early layers may help.
- **Sweep:** `XSA_LAST_N ∈ {0, 4, 7, 9, 11}` at fixed `NUM_LAYERS=11`. Add `XSA_LAST_N ∈ {8, 12, 16}` if testing `NUM_LAYERS≥16`.
- **Decision rule:** keep current default unless a smaller K beats it by ≥0.002 BPB at matched wallclock; if so, lock new default and re-check interactions with `MLP_MULT` and looping (`NUM_LOOPS`).
- **Results (no LOOP_EMBEDDINGS, default loopat):**

| Run | `XSA_LAST_N` | pre_val_bpb | sw_val_bpb | ttt_val_bpb | steps |
|-----|---|---|---|---|---|
| pgm_xsa9  | 9  | **1.06803** | **1.07307** | **1.07102** 🏆 | 5,005 |
| pgm_xsa7  | 7  | 1.06896 | 1.07419 | 1.07208 | 5,052 |
| pgm_xsa0  | 0  | 1.07120 | 1.07659 | 1.07442 | 5,109 |
| pgm_xsa4  | 4  | 1.07125 | 1.07648 | 1.07430 | 5,034 |

  **Conclusion:** `XSA_LAST_N=9` is the best within this family (Δ=−0.0011 TTT BPB vs default 11 baseline, although the loopat winner uses 11; need a combined sweep). `XSA_LAST_N ≤ 4` clearly hurts. Consider locking **default to 9** and re-running the loopat curriculum + LOOP_EMBEDDINGS combo to confirm.

### Loop-pass conditioning (`LOOP_EMBEDDINGS`)
- **What it controls:** whether each pass through the looped mid-stack (`layers [LOOP_START..LOOP_END]`, repeated `NUM_LOOPS+1` times) adds a learnable per-pass bias vector `loop_embs[p] ∈ ℝ^{model_dim}` to the residual stream before the block call. Breaks input symmetry across otherwise-identical repeated calls.
- **Current default:** `LOOP_EMBEDDINGS=0` with `LOOP_START=3 LOOP_END=5 NUM_LOOPS=2` (3 passes, shared weights, no pass tag).
- **Hypothesis / rationale:**
  - Repeated application of the same block with the same input computes the same function. The residual stream does change between passes, so something is already learned, but a per-pass bias gives the block an explicit "which iteration am I on" signal — Universal-Transformer-style *timestep embedding*.
  - Parameter cost is tiny: `(NUM_LOOPS+1) * model_dim` floats (e.g. 3 × `model_dim` ≈ a handful of KB), negligible vs. the 16 MB budget and trivially quantizable.
  - Expected to matter most when loops repeat many times or when the mid-stack does iterative refinement; at `NUM_LOOPS=2` (3 passes) the signal may be small but should not hurt.
- **Sweep:**
  1. `LOOP_EMBEDDINGS ∈ {0, 1}` at current loop config (`NUM_LOOPS=2`).
  2. If (1) wins, re-test at deeper recurrence: `NUM_LOOPS ∈ {3, 4}` with `LOOP_EMBEDDINGS=1`, checking that the extra FLOPs still fit the 10-min budget.
  3. Optional: widen/shift the looped segment (`LOOP_START, LOOP_END`) with `LOOP_EMBEDDINGS=1` on, to see if pass-tagging enables more aggressive looping.
- **Decision rule:** enable `LOOP_EMBEDDINGS=1` if it beats the default by ≥0.002 BPB at matched wallclock with no artifact-size regression. If gain is within noise at `NUM_LOOPS=2` but positive at `NUM_LOOPS≥3`, prefer combining with more loops only if total wallclock fits.
- **Results (`LOOP_EMBEDDINGS=1`, default `ENABLE_LOOPING_AT=0.35`):**

| Run | `NUM_LOOPS` | pre_val_bpb | sw_val_bpb | ttt_val_bpb | steps |
|-----|---|---|---|---|---|
| pgm_loopemb1        | 2 | **1.06869** | **1.07384** | **1.07169** 🏆 | 4,958 |
| pgm_loopemb1_loops4 | 4 | 1.07190 | 1.07610 | 1.07394 | 4,196 |
| pgm_loopemb1_loops3 | 3 | 1.07218 | 1.07631 | 1.07419 | 4,325 |

  **Conclusion:** `LOOP_EMBEDDINGS=1` at `NUM_LOOPS=2` is the recipe; pushing loops to 3–4 hurts BPB at fixed wallclock (extra FLOPs steal training steps). Update default to **`LOOP_EMBEDDINGS=1` (keep `NUM_LOOPS=2`)**. The loopat sweep above is run on top of this combo.

### Per-group GPTQ quantization clip multipliers (`CLIP_MULT_*`)
- **What they control:** multipliers on the σ-based clipping threshold (`clip_sigmas`) used by **GPTQ int8 mixed quantization** (post-training). Applied per *block group*, scaling the group's effective `cs = matrix_clip_sigmas * CLIP_MULT_*`. Not related to gradient clipping.
- **Block → group mapping** (`_get_group_clip_mult` at [train_gpt_improved_04_16.py](train_gpt_improved_04_16.py#L1580-L1594)):
  - `idx ≤ 2` → `CLIP_MULT_EARLY`
  - `LOOP_START ≤ idx ≤ LOOP_END` → `CLIP_MULT_LOOP` (takes precedence over mid/late)
  - `idx ∈ {3, 6, 7}` → `CLIP_MULT_MID`
  - `idx ≥ 8` → `CLIP_MULT_LATE`
- **Current default:** all four = `1.0` (uniform quantization aggressiveness).
- **Hypothesis / rationale:**
  - Smaller `cs` → tighter int8 grid → less bulk error but more clipping of outliers. Larger `cs` → fewer clipped outliers but coarser grid.
  - Different block depths produce weights with different tail behavior:
    - **Early** blocks tend to have structured/low-entropy weights → tolerate tighter `cs`.
    - **Loop** blocks accumulate gradient over `NUM_LOOPS+1` call sites → fatter tails → usually need larger `cs`.
    - **Mid/late** blocks often drive logit-relevant features → sensitive to outlier clipping.
  - BPB is the metric that gets round-tripped through GPTQ, so per-group tuning directly attacks the quantization error floor.
- **Sweep (coarse → fine):**
  1. **Coarse per-group scan, one at a time** (hold others at 1.0): `{0.75, 1.0, 1.25, 1.5}` for each of `CLIP_MULT_EARLY`, `CLIP_MULT_LOOP`, `CLIP_MULT_MID`, `CLIP_MULT_LATE`. 16 runs total.
  2. **Combine winners:** take each group's best single-variable value and run the combined config; verify it beats every single-variable change individually.
  3. **Diagnostic first:** before the sweep, log per-layer `max(|w|) / std(w)` after training; groups with large ratios likely benefit from `>1.0`, groups with small ratios from `<1.0`. Start the coarse scan's center on that prior.
  4. **Interaction with `MATRIX_CLIP_SIGMAS`:** the multiplier scales this base; if the baseline `MATRIX_CLIP_SIGMAS` changes (currently 15), re-run step 1.
- **Decision rule:** accept per-group values only if the combined config beats the uniform 1.0 baseline by ≥0.002 pre-to-post-quant BPB delta (look at `val_bpb` after the round-trip reload), and artifact size stays ≤16 MB.
- **Correction:** an earlier entry described `CLIP_MULT_LOOP` as a *gradient*-clip multiplier — that was wrong. It affects GPTQ quantization only. Gradient clipping is global via `GRAD_CLIP_NORM` (no per-group form).
- **Results (2026-04-22, on `pgm_loopat0_5` baseline; `MATRIX_CLIP_SIGMAS=14`, `ENABLE_LOOPING_AT=0.5`, `LOOP_EMBEDDINGS=1`, `HESSIAN_CLIP_LAMBDA=0.3`, varlen+TTT). Skipped 1.5 to halve the budget; 0.75 vs 1.25 is the informative pair.**

  | Run | group | mult | sw_val_bpb | ttt_val_bpb | Δ vs baseline (1.06919) |
  |-----|-------|------|------------|-------------|-------------------------|
  | pgm_cliplate075 | LATE | 0.75 | **1.06955** | **1.06756** 🏆 | **−0.00163** |
  | pgm_cliploop075 | LOOP | 0.75 | 1.06969 | 1.06770 | −0.00149 |
  | pgm_clipearly075 | EARLY | 0.75 | 1.07049 | 1.06839 | −0.00080 |
  | pgm_clipmid075 | MID | 0.75 | 1.07051 | 1.06844 | −0.00075 |
  | pgm_clipearly125 | EARLY | 1.25 | 1.07238 | 1.07025 | +0.00106 |
  | pgm_clipmid125 | MID | 1.25 | 1.07296 | 1.07082 | +0.00163 |
  | pgm_cliploop125 | LOOP | 1.25 | 1.07353 | 1.07133 | +0.00214 |
  | pgm_cliplate125 | LATE | 1.25 | 1.07382 | 1.07163 | +0.00244 |

  **Conclusion:** every group prefers **tighter** clipping (0.75); every 1.25 regresses. Sensitivity ranking by (1.25 − 0.75) Δttt: **LATE 0.0041 > LOOP 0.0036 > MID 0.0024 > EARLY 0.0019**. New best absolute TTT BPB: **1.06756** (`pgm_cliplate075`, beats 1.06919 by 0.00163). The original hypothesis ("LOOP needs larger `cs`") is **inverted**. Next step: stack the two strongest signals (`CLIP_MULT_LATE=0.75 + CLIP_MULT_LOOP=0.75`) and probe sub-0.75 (e.g. 0.5 / 0.6) on LATE/LOOP since both still want tighter at the swept boundary.
- **Stacking + sub-0.75 follow-up (section 7, 2026-04-23):**

  | Run | Config | sw_val_bpb | ttt_val_bpb | Δ vs prior best (1.06756) |
  |-----|--------|------------|-------------|---------------------------|
  | **pgm_clip_lateloop05** | LATE=0.5 + LOOP=0.5 | **1.06559** | **1.06372** 🏆 | **−0.00385** |
  | pgm_clip_all075 | all 4 groups = 0.75 | 1.06614 | 1.06426 | −0.00330 |
  | pgm_clip_lateloop075 | LATE=0.75 + LOOP=0.75 | 1.06814 | 1.06626 | −0.00131 |
  | pgm_clip_late05 | LATE=0.5 | 1.06848 | 1.06650 | −0.00106 |
  | pgm_clip_late06 | LATE=0.6 | 1.06859 | 1.06660 | −0.00097 |
  | pgm_clip_loop05 | LOOP=0.5 | 1.06857 | 1.06664 | −0.00092 |
  | pgm_clip_loop06 | LOOP=0.6 | 1.06905 | 1.06708 | −0.00048 |

  **Conclusion:** stacking is **super-additive** — LATE=0.5 alone (−0.00106) + LOOP=0.5 alone (−0.00092) → combined **−0.00385** (much more than the sum). Sub-0.75 keeps winning: 0.5 > 0.6 > 0.75 ordering on both LATE and LOOP. **New record: 1.06372 TTT** (`pgm_clip_lateloop05`), beats prior 1.06919 by 0.00547. **Defaults updated:** `CLIP_MULT_LATE 1.0→0.5`, `CLIP_MULT_LOOP 1.0→0.5` in `train_gpt_improved_04_16.py`. Open follow-ups: probe sub-0.5 on LATE/LOOP (e.g. 0.35), all-groups=0.5, full 4-way stack.

- **Wallclock-budget sensitivity (`pgm_cliplate075_w3000`):** rerunning the prior best with `MAX_WALLCLOCK_SECONDS=3000` (vs 3600) gave pre-quant 1.07299 / sw 1.07561 / **ttt 1.07359** — a +0.00603 TTT regression for a 17% wallclock cut. The recipe is genuinely budget-hungry; a step-bounded re-run would isolate this from the throughput jitter discussed in the loopat0_5 reproduction note.

### Loop-activation curriculum (`ENABLE_LOOPING_AT`)
- **What it controls:** training-progress fraction at which `looping_active` flips from `False` to `True`. Before the switch the forward uses the non-looped 11-step graph; after, it uses the looped 17-step graph. `frac` is the same measure that drives `lr_mul`/`wd_mul` (elapsed-wallclock-based when `MAX_WALLCLOCK_SECONDS` is set).
- **Current default:** `ENABLE_LOOPING_AT=0.35`. With `WARMDOWN_FRAC=0.85` (warmdown starts at `frac=0.15`), looping turns on **during warmdown**, while LR is already decaying.
- **Hypothesis / rationale:**
  - Curriculum: train the cheaper non-looped graph first at high LR to stabilize all weights, then activate looping to squeeze extra effective depth from the same parameters during the lower-LR phase.
  - Flipping looping on is a sharp step-change in effective gradient magnitude for the looped blocks; doing it when LR is already decaying limits the disruption.
  - Earlier activation → more training steps benefit from 17-step depth, but more optimizer instability at the switch-over and less cheap stabilization beforehand. Later activation → less time to exploit looping, but cleaner switch.
- **Sweep:**
  1. `ENABLE_LOOPING_AT ∈ {0.0, 0.15, 0.35, 0.5, 0.7}` at fixed loop config and warmdown.
     - `0.0` = loop from step 0 (no curriculum).
     - `0.15` = switch coincides with warmdown start.
     - `0.5` = literal midpoint.
     - `0.7` = mostly non-looped training, brief looped fine-tune.
  2. Pair the best value with `CLIP_MULT_LOOP` sweep to catch interactions at the switch-over.
  3. Check the training log right after `layer_loop:enabled` for loss spikes or `grad_norm` jumps; if present, prefer a later switch or a tighter `CLIP_MULT_LOOP`.
- **Decision rule:** replace default only if BPB improves ≥0.002 at matched wallclock. If two settings tie on BPB, prefer the later switch (cheaper wallclock, more stable).
- **Results (with `LOOP_EMBEDDINGS=1`, loops=2):**

| Run | `ENABLE_LOOPING_AT` | pre_val_bpb | sw_val_bpb | ttt_val_bpb | steps |
|-----|---------------------|-------------|------------|-------------|-------|
| pgm_loopat0_5 | 0.5 | **1.06630** | **1.07129** | **1.06919** 🏆 | 5,716 |
| pgm_loopat0_5_repro | 0.5 (rerun 2026-04-22) | 1.06837 | 1.07323 | 1.07117 | 5,303 |
| pgm_loopat0_15 | 0.15 | 1.06881 | 1.07392 | 1.07180 | 4,647 |
| pgm_loopat0_7 | 0.7 | 1.06893 | 1.07368 | 1.07159 | 6,190 |
| pgm_loopat0_0 | 0.0 | 1.07246 | 1.07756 | 1.07537 | 4,353 |

  **Conclusion:** `ENABLE_LOOPING_AT=0.5` is the clear winner — **new best absolute TTT BPB (1.06919)** and best pre-quant (1.06630). Immediate looping (0.0) is worst; curriculum matters. **Update default from 0.35 → 0.5.**

  **Reproduction (2026-04-22):** re-ran `pgm_loopat0_5` with identical hyperparameters
  (`pgm_loopat0_5_repro`). Got 1.07117 TTT (+0.00198 vs 1.06919). Found two issues:
  (a) `HESSIAN_CLIP_LAMBDA` default had silently changed from 0.3→0.0 in
  `train_gpt_improved_04_16.py`; `run_program_md_section.sh` did not pin it. **Fixed**
  by adding `HESSIAN_CLIP_LAMBDA=0.3` to `COMMON_TRAIN_ENV`. (b) Repro hit only 5,303
  steps (−413, −7.2% throughput on this node) in the 3600s budget. The pre-quant Δ
  (+0.00207) propagates directly through quant/TTT — wallclock-bounded training is
  not bitwise-reproducible across node states. Recipe is otherwise faithful.

### Parallel residuals placement (`PARALLEL_RESIDUAL_START`)
- **What it controls:** index of the **first block** that switches from sequential (`x + attn; x + mlp(x)`) to parallel (`x + attn + mlp(x)`) residuals. Blocks `[PARALLEL_RESIDUAL_START..num_layers-1]` use the parallel layout; earlier blocks stay sequential. Set `≥ num_layers` to disable entirely; set `0` to make the whole stack parallel.
- **Current default:** `PARALLEL_RESIDUAL_START=7` with `NUM_LAYERS=11` → top 4 blocks are parallel, bottom 7 are sequential.
- **Hypothesis / rationale:**
  - Parallel residuals save a data dependency per block, letting `torch.compile` fuse attention and MLP more aggressively → lower wallclock per step at equal quality for late layers.
  - Early blocks benefit more from the attn→MLP dependency (MLP sees attention's contribution before next block). Late blocks are doing output-shaping, so parallel is usually a free speed win there.
  - Moving the threshold earlier trades a bit of expressive power for more wallclock → more training steps. At fixed `MAX_WALLCLOCK_SECONDS` this can net better BPB.
- **Sweep:**
  1. Suffix sweep: `PARALLEL_RESIDUAL_START ∈ {11 (disabled), 9, 7, 5, 3, 0 (all parallel)}` at `NUM_LAYERS=11`. Look for the sweet spot on BPB-at-matched-wallclock.
  2. Interaction with looping: re-test the best two values combined with `ENABLE_LOOPING_AT ∈ {0.35, 0.5}` — the looped mid-stack (`[3..5]`) is particularly sensitive to whether its blocks are sequential or parallel.
  3. Interaction with XSA: if `XSA_LAST_N` is reduced, re-verify the `PARALLEL_RESIDUAL_START` optimum, since both primarily affect the top of the stack.
- **Decision rule:** accept a new default if it improves BPB ≥0.002 at matched wallclock, or ties BPB while reducing wallclock ≥5% (letting us add steps).

### Interleaved parallel residuals (experimental, requires code change)
- **What it would control:** instead of a single prefix/suffix split, choose an **arbitrary set of blocks** to be parallel (e.g. even-indexed layers, or `{0, 3, 6, 9}`), keeping the rest sequential.
- **Status:** not currently wired — the code at [train_gpt_improved_04_16.py](train_gpt_improved_04_16.py#L1038-L1039) only supports the contiguous-suffix form. A small edit would add a `PARALLEL_RESIDUAL_LAYERS` env var (comma-separated indices or a pattern keyword like `even`/`odd`/`every3`) and replace the range loop with `for i in parallel_set: self.blocks[i].parallel = True`.
- **Hypothesis / rationale:**
  - Alternating sequential/parallel may preserve attn→MLP coupling in roughly half the blocks while still gaining compile/fusion benefits in the other half, potentially Pareto-beating the pure suffix pattern on BPB-at-wallclock.
  - PaLM/GPT-J used uniform-parallel stacks at large scale; modded-nanoGPT-scale models may prefer heterogeneous layouts because each layer matters more relatively.
- **Sweep (after enabling):**
  1. Patterns: `all-sequential`, `suffix(7)` (current), `every-other` (`0,2,4,...`), `every-3` (`0,3,6,9`), `all-parallel`.
  2. Ablate whether the **looped block range** (`[3..5]`) should be parallel or sequential independently from the global pattern.
- **Decision rule:** only implement if suffix sweep plateaus and a clear theoretical gap remains; otherwise stick with the contiguous suffix form.

### Window attention on alternating layers (`WINDOW_ATTN_SIZE`, `WINDOW_ATTN_LAYERS`)
- **What it controls:** enables FlashAttention-3 sliding-window causal attention on a chosen subset of layers, with half-window radius `WINDOW_ATTN_SIZE`. `WINDOW_ATTN_LAYERS` is a comma-separated index list; if empty, defaults to even-indexed layers (alternating local/global).
- **Asymmetric train/eval:** the window is applied **only during training** (`window_size > 0 and self.training` at [train_gpt_improved_04_16.py](train_gpt_improved_04_16.py#L805)); evaluation always uses full causal attention. So this is a *wallclock optimization*, not an inference architectural choice.
- **Current default:** `WINDOW_ATTN_SIZE=0` (disabled).
- **Hypothesis / rationale:**
  - Attention is `O(L²)` per layer; with `L=2048` and `W=512`, windowed layers are ~4× cheaper. Half the layers windowed → ~2× total attention speedup → more training steps in the 10-min cap.
  - Alternating local/global (à la Gemma-2) lets full-attention layers carry windowed layers' local features into global context, preserving most of the BPB.
  - Train-eval mismatch is benign in expectation: at eval, windowed layers receive *more* context than during training.
- **Sweep:**
  1. **Window radius:** `WINDOW_ATTN_SIZE ∈ {0, 256, 512, 1024}` with default alternating (even layers). Look for max `steps × quality` product at `MAX_WALLCLOCK_SECONDS=3600`.
  2. **Layer pattern:** at the best `WINDOW_ATTN_SIZE`, vary `WINDOW_ATTN_LAYERS`:
     - default (`even`): `0,2,4,6,8,10`
     - odd: `1,3,5,7,9`
     - bottom-half: `0,1,2,3,4`
     - top-half: `6,7,8,9,10`
     - skip-loop-range (avoid windowing the looped mid-stack): `0,2,8,10`
  3. **Interaction with looping:** verify that windowed layers inside `[LOOP_START..LOOP_END]` don't degrade BPB more than the wallclock saved buys back. Re-test with looping disabled vs. enabled.
  4. **Interaction with sequence length:** if `TRAIN_SEQ_LEN` increases, windowed layers' relative speedup grows — re-sweep when changing it.
- **Decision rule:** enable if BPB drops ≥0.002 at the same `MAX_WALLCLOCK_SECONDS`, or BPB ties with ≥10% wallclock saved (which lets us add steps). Disable if windowing the looped range causes a clear BPB regression.

### TTT (Test-Time Training) sweeps

> ✅ **RESOLVED (2026-04-18).** The varlen+TTT regression was a **RoPE base-seqlen mismatch**: training packs long rows that trigger NTK-aware RoPE rescaling, while TTT eval processes 2048-token rows and hit the vanilla cached RoPE → +0.12 BPB encoding mismatch. Fix lives in `train_gpt_improved_04_16.py`: a new `ROPE_FORCE_BASE_SEQLEN` env var (auto-set from `TTT_ROPE_BASE_SEQLEN`, defaulting to `train_batch_tokens / (world * grad_accum)`) forces `Rotary.forward` to recompute cos/sin under the training-equivalent NTK base whenever the LoRA-TTT eval path runs. See [invest.md](invest.md) for the full diagnosis.
>
> **Current best on `pg12_varlen_clip14`:** `pg12_ttt_r48_phased2` → **quantized_ttt_lora val_bpb = 1.07183** (vs. sliding-only baseline 1.07425; **−0.00242 BPB delta**). Non-varlen+TTT record (1.07077) is not yet beaten — primary residual gap is the varlen quant→sw step, not TTT itself.

The four TTT variants are: **LoRA TTT** (per-chunk online adapter, varlen path), **Full-param TTT** (per-chunk full-backbone fine-tune, non-varlen path), **SLOT** (per-window logit-bias delta), and **Phased global SGD** (periodic full-model SGD on already-scored prefix). All gated by `TTT_ENABLED=1`.

#### TTT sweep results (on `pg12_varlen_clip14`, sliding eval baseline 1.07425)

Top configs from `run_pg12_ttt_sweep.sh` / `_phase3` / `_phase4` (all use defaults except those listed; logs in `logs/pg12_sweep/`):

| Run | Vars (overrides) | quantized_ttt_lora val_bpb |
|---|---|---|
| `pg12_ttt_r48_phased2`        | `TTT_LORA_RANK=48 TTT_PHASES=2` | **1.07183** |
| `pg12_ttt_rank48`             | `TTT_LORA_RANK=48` | 1.07187 |
| `pg12_ttt_lr5e5`              | `TTT_LORA_LR=5e-5` | 1.07194 |
| `pg12_ttt_lr7e5`              | `TTT_LORA_LR=7e-5` | 1.07202 |
| `pg12_ttt_rank32`             | `TTT_LORA_RANK=32` | 1.07202 |
| `pg12_ttt_chunk128`           | `TTT_CHUNK_SIZE=128` | 1.07229 |
| `pg12_ttt_chunk96`            | `TTT_CHUNK_SIZE=96`  | 1.07232 |
| `pg12_ttt_phased2`            | `TTT_PHASES=2` (rank=96) | 1.07240 |
| `pg12_ttt_adaptive`           | `TTT_ADAPTIVE_LR=1`  | 1.07249 |
| `pg12_ttt_chunk32`            | `TTT_CHUNK_SIZE=32`  | 1.07282 |
| `pg12_ttt_lr3e4`              | `TTT_LORA_LR=3e-4` (too high) | 1.07850 |
| `pg12_ttt_rank192`            | `TTT_LORA_RANK=192` (too large, undertrained) | 1.07589 |
| `pg12_ttt_r192_lr3e4`         | `TTT_LORA_RANK=192 TTT_LORA_LR=3e-4` | 1.09354 |

**Takeaways:**
1. **Smaller rank wins.** Rank 48 ≈ rank 32 < rank 96 (default) ≪ rank 192. Default should drop to **`TTT_LORA_RANK=48`**.
2. **Default LR (1e-4) is near-optimal.** Lowering to 5e-5 / 7e-5 is within 0.0001 BPB; raising to 3e-4 collapses (−0.007 BPB).
3. **Phased SGD adds a small consistent win** at rank 48 (`r48_phased2` 1.07183 < `rank48` 1.07187), and is essentially free at `TTT_PHASES=2`.
4. **Larger chunks don't help** once rank is right; default `TTT_CHUNK_SIZE=64` ties with 96/128.
5. **Adaptive LR is neutral** at this scale (1.07249 vs. 1.07240 baseline phased2). Drop from priority sweeps.
6. **Recommended new default for varlen+TTT runs:** `TTT_LORA_RANK=48 TTT_PHASES=2 TTT_LORA_LR=1e-4 TTT_CHUNK_SIZE=64` (plus the auto RoPE-base fix, which is on by default once `TTT_ENABLED=1` and `VARLEN_ATTENTION=1`).

#### LoRA TTT (`TTT_LORA_*`, `TTT_CHUNK_SIZE`, `TTT_ADAPTIVE_LR`, `TTT_ROPE_BASE_SEQLEN`)
- **Current default (post-fix):** `TTT_LORA_RANK=48 TTT_LORA_LR=1e-4 TTT_CHUNK_SIZE=64 TTT_GRAD_STEPS=1 TTT_K_LORA=1 TTT_MLP_LORA=1 TTT_O_LORA=1 TTT_ADAPTIVE_LR=0 TTT_PHASES=2 TTT_ROPE_BASE_SEQLEN=0` (auto). The RoPE base fix is automatic on the varlen path; only override `TTT_ROPE_BASE_SEQLEN` if `train_batch_tokens / (world * grad_accum)` doesn't match what training actually saw.
- **Hypothesis / rationale:**
  - Rank trades adapter capacity vs. wallclock and per-batch memory. Empirically rank 48 beats both 32 and 96 on `pg12_varlen_clip14`; rank 192 underperforms (likely undertrained per chunk).
  - Adaptive LR (`TTT_ADAPTIVE_LR=1`, scales by `chunk_loss/EMA(chunk_loss)`) was neutral in the post-fix sweep — keep off by default.
  - Adapter placement (Q/K/MLP/O) interacts: removing MLP-LoRA cuts most of the params; removing K-LoRA loses most of the contextual adaptation. Not yet re-ablated post-fix.
- **Sweep (post-fix, residual exploration):**
  1. **Adapter placement ablation** (highest priority — never re-run after the RoPE fix): turn each of `TTT_K_LORA / TTT_MLP_LORA / TTT_O_LORA` off one at a time at the new default; pick the smallest set that keeps the gain.
  2. **Re-test rank** at the new lower band: `TTT_LORA_RANK ∈ {24, 32, 48, 64}` with `TTT_PHASES=2`.
  3. **Fine LR scan** around the new optimum: `TTT_LORA_LR ∈ {3e-5, 5e-5, 1e-4, 2e-4}` at `TTT_LORA_RANK=48 TTT_PHASES=2`.
  4. **`TTT_GRAD_STEPS=2`** at `TTT_LORA_RANK=48 TTT_LORA_LR=5e-5` — earlier `_steps2` runs were on rank 96 / pre-fix.
  5. **Adaptive LR re-test** at the new default: `TTT_ADAPTIVE_LR=1` with `TTT_ADAPT_POWER ∈ {0.5, 1.0}` and `TTT_ADAPT_EMA ∈ {0.95, 0.99}`.
- **Decision rule:** keep a setting if it improves TTT BPB ≥0.001 over the new `pg12_r48_phased3` floor (1.07173) without >10% wallclock blowup. Aim is to dip below the non-varlen+TTT record (**1.07077**).
- **Post-fix follow-up results (on top of `TTT_LORA_RANK=48 TTT_PHASES=2`):**

| Sweep axis | Run | val_bpb | Δ vs r48_phased2 (1.07183) |
|---|---|---|---|
| Rank | `pg12_r32_phased2` (rank 32) | 1.07199 | +0.00016 |
| Rank | `pg12_r64_phased2` (rank 64) | 1.07189 | +0.00006 |
| Rank | **`pg12_r48_phased3`** (phases=3) | **1.07173** | **−0.00010** |
| Adaptive | `pg12_adapt_ema80` / `ema99` / `pow05` / `pow15` / `tight` / `wide` | 1.07187–1.07190 | +0.00004 to +0.00007 |
| Adaptive | `pg12_r48_phased2_adapt_tight` | 1.07184 | +0.00001 |
| Min-doc-len | `pg12_r48_phased2_minlen256` | 1.07186 | +0.00003 |
| Min-doc-len | `pg12_r48_phased2_minlen512` | 1.07194 | +0.00011 |
| Chunk size | `pg12_r48_phased2_chunk80` | 1.07194 | +0.00011 |
| Min-doc-len standalone | `pg12_ttt_minlen{512,1024,2048,4096}` | 1.0720 → 1.0726 (monotone worse) | n/a |

  **Takeaways:**
  - **Rank 48 confirmed as the sweet spot** — both 32 and 64 regress slightly. Drop further rank sweeps from priority.
  - **`TTT_PHASES=3` is the new floor** at rank 48 (1.07173). Test `TTT_PHASES ∈ {4, 6}` next.
  - **Adaptive LR remains neutral** even with re-tuned EMA/power. Drop entirely from priority sweeps (was already on the chopping block).
  - **`TTT_MIN_DOC_LEN` strictly hurts** at every value tried (256/512/1024/2048/4096). Default `0` is correct; remove this from future sweeps.
  - **Chunk-size variants near 64 (80/96/128)** all neutral-to-slightly-worse. Default `64` is locked.
  - **Open priorities:** adapter placement ablation (K/MLP/O), fine LR scan around `1e-4`, `TTT_PHASES ∈ {4, 6}` at rank 48, and `TTT_GRAD_STEPS=2` at rank 48.

#### SLOT TTT (`SLOT_*`)
- **Current default:** `SLOT_ENABLED=0`. When on: `SLOT_STEPS=4 SLOT_LR=0.01 SLOT_WD=0.01 SLOT_TRAIN_MODE=context`.
- **Hypothesis / rationale:**
  - Per-window logit-bias delta is the cheapest TTT variant; should compose additively with LoRA TTT.
  - `SLOT_TRAIN_MODE=context` is the legal mode (only trains on already-scored prefix); `'all'` is for upper-bound diagnostics only.
- **Sweep:**
  1. `SLOT_ENABLED ∈ {0, 1}` at otherwise-default config.
  2. If on: `SLOT_STEPS ∈ {2, 4, 8}` × `SLOT_LR ∈ {0.003, 0.01, 0.03}`.
  3. Compose with best LoRA TTT config — verify additivity (gain ≈ sum of individual gains, not redundant).
- **Decision rule:** enable if total TTT BPB improves ≥0.001 (SLOT alone is small but ~free in wallclock).
- **Results (standalone SLOT, on `pg12_varlen_clip14`, sliding baseline 1.07425):**

| Run | `SLOT_STEPS` × `SLOT_LR` (other) | quantized_slot val_bpb |
|---|---|---|
| pg12_slot_wd001        | 4 × 0.01, `SLOT_WD=0.001`  | **1.07351** |
| pg12_slot_default      | 4 × 0.01                   | 1.07351 |
| pg12_slot_steps8       | 8 × 0.01                   | 1.07353 |
| pg12_slot_steps8_lr3e3 | 8 × 0.003                  | 1.07359 |
| pg12_slot_steps2       | 2 × 0.01                   | 1.07363 |
| pg12_slot_lr3e3        | 4 × 0.003                  | 1.07370 |
| pg12_slot_lr3e2        | 4 × 0.03                   | 1.07387 |

  **Conclusion:** standalone SLOT delivers a stable −0.0007 BPB vs the sliding-only 1.07425 baseline; defaults (`SLOT_STEPS=4 SLOT_LR=0.01`) are at the optimum. `SLOT_WD` is insensitive in [0.001, 0.01]. Strictly weaker than LoRA TTT (1.07183) but cheap enough to compose — see SLOT-in-TTT below.

#### SLOT-in-TTT composition (`SLOT_IN_TTT_*`)
- **What it does:** during LoRA TTT scoring, fits a per-chunk logit-bias delta on the already-scored within-window context, then scores the chunk with the corrected logits. TTT gradient updates still use the uncorrected loss (no double-dipping).
- **Results (composed on top of `TTT_LORA_RANK=48 TTT_PHASES=2`):**

| Run | Vars | quantized_ttt_lora val_bpb |
|---|---|---|
| pg12_slotttt_default     | defaults (`SLOT_IN_TTT_STEPS=4 LR=0.01 WD=0.001`) | **1.07176** |
| pg12_slotttt_wd0         | `WD=0.0`     | 1.07177 |
| pg12_slotttt_steps8_lr3e3| `STEPS=8 LR=0.003` | 1.07178 |
| pg12_slotttt_steps2      | `STEPS=2`    | 1.07180 |
| pg12_slotttt_lr3e3       | `LR=0.003`   | 1.07183 |
| pg12_slotttt_steps8      | `STEPS=8`    | 1.07191 |
| pg12_slotttt_lr3e2       | `LR=0.03`    | 1.07235 |

  **Conclusion:** SLOT-in-TTT shaves a further ~0.00007 BPB on top of the `r48_phased2` floor (1.07183 → 1.07176). Gain is small and well below the ≥0.001 decision rule, so **do not enable by default** — keep as a cheap composition to revisit once individual variants are pushed further. Defaults are at the optimum; high LR (0.03) hurts.

#### Phased global SGD TTT (`TTT_PHASES`, `TTT_PHASE_SGD_*`)
- **Current default:** `TTT_PHASES=2` (post-fix recommended; was effectively off at 1). When on: `TTT_PHASE_SGD_LR=5e-4 MOMENTUM=0.9 EPOCHS=1 SEQ_LEN=1024 BATCH=8 OPT=sgd WD=0`.
- **Empirically:** at `TTT_LORA_RANK=48`, going from `TTT_PHASES=1` to `TTT_PHASES=2` improved varlen+TTT BPB from 1.07187 → 1.07183 (small but consistent, near-free in wallclock at 2 phases). **`TTT_PHASES=3` at rank 48** (`pg12_r48_phased3`) is **1.07173** — a further small win, now the standalone-LoRA-TTT floor on this baseline. At rank 96, going 1→2 moves 1.07248 → 1.07240 and `TTT_PHASES=3` (`pg12_ttt_phased3`) is 1.07230. **Updated recommendation:** at rank 48, prefer `TTT_PHASES=3`; at higher ranks the gain shrinks.
- **Phase-SGD hyperparam re-tests (rank 48, phases=2):** `pg12_r48_phased2_sgd_ep2` (epochs=2) → 1.07181; `pg12_r48_phased2_sgd_lr1e3` (LR=1e-3) → 1.07181. Both essentially neutral vs default 1.07183 — keep `EPOCHS=1 LR=5e-4`.
- **Hypothesis / rationale:**
  - Per-window LoRA TTT captures local adaptation; phased SGD captures global drift across documents (e.g., topic shift in val data).
  - Score-before-update keeps it legal: SGD only sees already-scored docs.
  - Cost grows with `TTT_PHASES`; benefit appears to saturate by `TTT_PHASES=2` at this scale.
- **Sweep:**
  1. `TTT_PHASES ∈ {2, 4, 8}` at the new LoRA default (rank 48). 1 and 3 already tested.
  2. At best `TTT_PHASES`: `TTT_PHASE_SGD_LR ∈ {1e-4, 5e-4, 1e-3}` × `EPOCHS ∈ {1, 2}`. The earlier `pg12_ttt_phased2_lr3e4` (1.07847) shows phased SGD is sensitive to LR; do not exceed 5e-4 without an epoch cap.
  3. Try `TTT_PHASE_SGD_OPT='adam'` to see if the better-conditioned optimizer wins on the small phase batches.
  4. Compose with best LoRA + SLOT.
- **Decision rule:** enable additional phases if BPB improves ≥0.0005 at acceptable wallclock (each phase boundary adds substantial time).

#### Full-param TTT (`TTT_FP_*`)
- **Current default:** non-varlen fallback only. `TTT_FP_LR=0.001 TTT_FP_EPOCHS=3 TTT_FP_CHUNK_TOKENS=32768`.
- **Hypothesis / rationale:**
  - Strictly more expressive than LoRA, but expensive. Only relevant on non-varlen runs (varlen path uses LoRA).
  - Likely dominated by LoRA TTT + phased SGD on varlen once the blocker is resolved; mostly kept as a non-varlen safety net.
- **Sweep (low priority, run only if non-varlen track is being pursued):**
  1. `TTT_FP_LR ∈ {3e-4, 1e-3, 3e-3}` × `TTT_FP_EPOCHS ∈ {1, 2, 3}`.
  2. `TTT_FP_CHUNK_TOKENS ∈ {16384, 32768, 65536}` to find wallclock sweet spot.
- **Decision rule:** keep only if BPB ≥ varlen+TTT minus 0.002 at comparable wallclock; otherwise prefer the varlen+LoRA stack.

#### TTT composition order (after individual sweeps)
Once each variant is tuned individually, run the combinatorial step:
- `(LoRA only) → (LoRA + SLOT) → (LoRA + SLOT + Phased SGD)`
- Verify each addition is monotonic in BPB; drop any that doesn't pay for its wallclock.

## Keep/Discard

- **Keep:** BPB improves and artifact ≤ 16,000,000 B. Or BPB flat with better size/speed.
- **Discard:** BPB worsens without compensating gains.
- **Crash:** Log, diagnose, fix, re-run.