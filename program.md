# Experiment Program

**Goal:** Lower `val_bpb` on `train_gpt_improved.py`. One variable at a time, record everything.

## Setup

| Item | Value |
|------|-------|
| GPUs | 4× A100 80GB PCIe |
| Env | `.venv` (activated) |
| Data | `./data` (sp8192) |
| Wallclock | `MAX_WALLCLOCK_SECONDS=3600` |

## Baseline (best run)

| Metric | Value |
|--------|-------|
| Sliding-window BPB | **1.07369** |
| Pre-quant BPB | 1.07873 |
| Total artifact | 15,989,467 B (fits 16MB) |
| Steps | 5,445 |
| Config | `GATED_ATTENTION=1 FUSED_ROPE=1` |

## Workflow

```bash
# 1. Pack
python3 pack_submission_file.py train_gpt_improved.py train_gpt.py

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

> ⚠ **BLOCKER — DO NOT RUN UNTIL RESOLVED.**
> Under `VARLEN_ATTENTION=1`, **all three TTT eval paths regress** (LoRA TTT to ~1.20 BPB, fullparam degrades, even sliding-window eval shows worse quant→sw delta than non-varlen). Active investigation in [invest.md](invest.md) and the [varlen_ttt_invest/](varlen_ttt_invest) folder. Working theory: a shared defect upstream of any specific TTT path — likely val-side `cu_seqlens`/BOS mismatch, `use_varlen` not propagating after `deserialize()`, or `eval_seq_len`/`eval_stride` misalignment with doc boundaries.
>
> **Until that's fixed, the only TTT runs to do are the ones in [invest.md](invest.md) (`varlen_ttt_invest/` track).** Sweeps below are queued for *after* varlen+TTT lands ≤1.071 BPB.

The four TTT variants are: **LoRA TTT** (per-chunk online adapter, varlen path), **Full-param TTT** (per-chunk full-backbone fine-tune, non-varlen path), **SLOT** (per-window logit-bias delta), and **Phased global SGD** (periodic full-model SGD on already-scored prefix). All gated by `TTT_ENABLED=1`.

#### LoRA TTT (`TTT_LORA_*`, `TTT_CHUNK_SIZE`, `TTT_ADAPTIVE_LR`)
- **Current default:** `TTT_LORA_RANK=96 TTT_LORA_LR=1e-4 TTT_CHUNK_SIZE=64 TTT_GRAD_STEPS=1 TTT_K_LORA=1 TTT_MLP_LORA=1 TTT_O_LORA=1 TTT_ADAPTIVE_LR=0`.
- **Hypothesis / rationale:**
  - Rank trades adapter capacity vs. wallclock and per-batch memory; chunk size trades adaptation granularity vs. compute.
  - Adaptive LR (`TTT_ADAPTIVE_LR=1`, scales by `chunk_loss/EMA(chunk_loss)`) should help on heterogeneous validation chunks.
  - Adapter placement (Q/K/MLP/O) interacts: removing MLP-LoRA cuts most of the params; removing K-LoRA loses most of the contextual adaptation.
- **Sweep (after blocker resolved):**
  1. **Rank:** `TTT_LORA_RANK ∈ {32, 64, 96, 128}` at default LR.
  2. **Chunk size:** `TTT_CHUNK_SIZE ∈ {32, 48, 64, 96, 128}`.
  3. **LR:** `TTT_LORA_LR ∈ {3e-5, 1e-4, 3e-4, 1e-3}` paired with the best rank.
  4. **Adapter placement ablation:** turn each of `TTT_K_LORA / TTT_MLP_LORA / TTT_O_LORA` off one at a time; pick the smallest set that keeps the gain.
  5. **Adaptive LR:** `TTT_ADAPTIVE_LR=1` with `TTT_ADAPT_POWER ∈ {0.5, 1.0, 1.5}` and `TTT_ADAPT_EMA ∈ {0.9, 0.95, 0.99}`.
- **Decision rule:** keep a setting if it improves TTT BPB ≥0.002 over default with no wallclock blowup (>10%).

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

#### Phased global SGD TTT (`TTT_PHASES`, `TTT_PHASE_SGD_*`)
- **Current default:** `TTT_PHASES=1` (effectively off — no phase boundary). When on: `TTT_PHASE_SGD_LR=5e-4 MOMENTUM=0.9 EPOCHS=1 SEQ_LEN=1024 BATCH=8 OPT=sgd WD=0`.
- **Hypothesis / rationale:**
  - Per-window LoRA TTT captures local adaptation; phased SGD captures global drift across documents (e.g., topic shift in val data).
  - Score-before-update keeps it legal: SGD only sees already-scored docs.
  - Cost grows with `TTT_PHASES`; benefit may saturate.
- **Sweep:**
  1. `TTT_PHASES ∈ {1, 2, 4, 8}` at default SGD config.
  2. At best `TTT_PHASES`: `TTT_PHASE_SGD_LR ∈ {1e-4, 5e-4, 1e-3}` × `EPOCHS ∈ {1, 2}`.
  3. Try `TTT_PHASE_SGD_OPT='adam'` to see if the better-conditioned optimizer wins on the small phase batches.
  4. Compose with best LoRA + SLOT.
- **Decision rule:** enable if BPB improves ≥0.002 at acceptable wallclock (each phase boundary adds substantial time).

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