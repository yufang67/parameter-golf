# Journal

<!-- ============================================================
     APPEND-ONLY — the agent adds entries at the TOP of this file.
     Never delete or modify previous entries.

     Each entry should include:
     - What you tried and WHY (not just what changed)
     - The result (metric value)
     - What you learned from it
     - What this suggests for next steps

     Keep entries concise — 3-5 sentences, not paragraphs.
     ============================================================ -->

## Experiment 11 — WARMDOWN=0.85 + EMA_DECAY=0.998

**Hypothesis:** Slower EMA averaging (0.998 vs 0.9965) keeps weights closer to recent training, potentially capturing late-stage improvements better with the extended warmdown.
**Change:** EMA_DECAY 0.9965 → 0.998, with WARMDOWN=0.85.
**Result:** Pre-quant BPB=1.0816 (+0.0023), post-quant=1.0911 (+0.0005). Worse.
**Status:** discard
**Learned:** Slower EMA hurts — 0.9965 provides better weight averaging for this training duration. The faster averaging (lower decay) captures a broader window of training history, which helps smooth out noise. All major HP dimensions now explored; WARMDOWN=0.85 is the only confirmed improvement over SOTA defaults.

## Experiment 10 — WARMDOWN=0.85 + ENABLE_LOOPING_AT=0.20

**Hypothesis:** Enabling depth recurrence earlier (20% vs 35%) gives more training steps at full virtual depth.
**Change:** ENABLE_LOOPING_AT 0.35 → 0.20, with WARMDOWN=0.85.
**Result:** Pre-quant BPB=1.0800, post-quant=1.0913. Worse by +0.0007 across the board. Only 4641 steps (vs 4989) due to longer slower phase.
**Status:** discard
**Learned:** Earlier looping activation trades step count for virtual depth. Net negative — the optimal activation point of 0.35 balances "train fast with shallow model" vs "train slow with deep model." All SOTA hyperparameters (WD, LR, warmdown, loop timing) are well-tuned. The only confirmed improvement: WARMDOWN 0.72→0.85 (−0.0007 pre-quant BPB).

## Experiment 9 — WARMDOWN=0.85 + MUON_WD=0.06

**Hypothesis:** Lower Muon weight decay (0.06 vs 0.095) might let model learn more with the optimized warmdown schedule.
**Change:** MUON_WD 0.095 → 0.06, with WARMDOWN_FRAC=0.85.
**Result:** Pre-quant BPB=1.0802 (+0.0009), post-quant BPB=1.0927 (+0.0021). **Worse.** Mid-training convergence was faster (step 4500: 1.0952 vs 1.0980) but final quality worse.
**Status:** discard
**Learned:** Lower WD speeds mid-training convergence but hurts final quality — less regularization causes overfitting at the end. The SOTA value 0.095 is well-tuned. The full HP space (LR, WD, warmdown, EMA) is near-optimal. Need structural/architectural changes for further gains.

## Experiment 8 — WARMDOWN_FRAC=0.90

**Hypothesis:** Continue warmdown sweep to find peak. 0.85 was best so far.
**Change:** WARMDOWN_FRAC 0.72 → 0.90.
**Result:** Pre-quant BPB=1.0794, post-quant BPB=1.0908. Tied with 0.85 pre-quant, slightly worse post-quant.
**Status:** discard
**Learned:** Warmdown sweep complete: 0.85 is optimal. Full sweep: 0.50(1.0825) < 0.72(1.0800) < 0.80(1.0794) < 0.85(1.0793) ≈ 0.90(1.0794). Total gain from 0.72→0.85: −0.0007 pre-quant. Now move to architecture search.

## Experiment 7 — WARMDOWN_FRAC=0.85

**Hypothesis:** Warmdown trend: 0.50→0.72→0.80 shows monotonic improvement. Try 0.85.
**Change:** WARMDOWN_FRAC 0.72 → 0.85.
**Result:** Pre-quant BPB=**1.0793** (−0.0007 from baseline), post-quant=1.0906, sliding=**1.0805**. **Best pre-quant yet.** Improvement decelerating: 0.72→0.80 gained 0.0006, 0.80→0.85 gained 0.0001.
**Status:** keep
**Learned:** Warmdown trend continues but flattening. 0.85 is marginally better than 0.80. Try 0.90 to find the peak.

## Experiment 6 — WARMDOWN_FRAC=0.80

**Hypothesis:** Since 0.72 beats 0.50, more aggressive warmdown might help further. With 0.80, warmdown starts at 20% through training (~step 1000).
**Change:** WARMDOWN_FRAC 0.72 → 0.80.
**Result:** Pre-quant BPB=**1.0794** (−0.0006), post-quant BPB=1.0905 (−0.0003). **New best pre-quant.** Consistent improvement across all checkpoints.
**Status:** keep
**Learned:** Longer warmdown helps — model benefits from gradual LR decay. Trend: 0.50 (1.0825) < 0.72 (1.0800) < 0.80 (1.0794). Try 0.85 next.

## Experiment 5 — MATRIX_LR=0.030

**Hypothesis:** Higher matrix LR (0.030 vs 0.022) could allow faster convergence in our fixed step budget.
**Change:** MATRIX_LR 0.022 → 0.030.
**Result:** Pre-quant BPB=1.0803 (+0.0003), post-quant BPB=1.0902 (−0.0006). Within noise.
**Status:** discard
**Learned:** Matrix LR 0.022 is well-tuned. 0.030 trades marginal pre-quant quality for slightly better quantization resilience, but net effect is zero. The SOTA hyperparameters are near a local optimum for this architecture.

## Experiment 4 — WARMDOWN_FRAC=0.50

**Hypothesis:** Lower warmdown frac means warmdown starts later (50% vs 28% from end), allowing more training at full LR. Might help with our 5000-step budget.
**Change:** WARMDOWN_FRAC 0.72 → 0.50.
**Result:** Pre-quant BPB=1.0825, post-quant BPB=1.0924 — worse than baseline by +0.0025. Same step count (4986).
**Status:** discard
**Learned:** 0.72 warmdown is well-tuned. Earlier warmdown helps convergence even though it reduces peak LR duration. The model benefits from longer gradual cooldown rather than more time at full LR. Don't go below 0.72.

## Experiment 3 — 3 depth recurrence loops (NUM_LOOPS=3)

**Hypothesis:** More depth recurrence loops (3 vs 2) add virtual layers (20 vs 17) at zero parameter cost. Should help if compute-limited rather than parameter-limited.
**Change:** NUM_LOOPS=2 → 3. Encoder: [0,1,2,3,4,5,3,4,5,3], Decoder: [4,5,3,4,5,6,7,8,9,10].
**Result:** Pre-quant BPB=1.0820, post-quant BPB=1.0924. **Worse** than baseline by 0.002. Only 4554 steps vs 4990 (−9%) due to 13% slower tok/s (1.0M vs 1.15M). Peak memory 51.8 GiB (+7.5 GiB).
**Status:** discard
**Learned:** 3 loops trades too much throughput for virtual depth on our 3600s budget. Each extra loop adds ~15% compute per step but only marginal quality gain per step. The step count reduction dominates. Depth recurrence is already near-optimal at 2 loops for our budget. Focus elsewhere.

## Experiment 2 — QK_GAIN_INIT=7.0

**Hypothesis:** SOTA README notes "monotonic improvement from 4.0 to 5.25" for QK gain. Testing 7.0 to see if the trend continues. Higher gain amplifies query-key interactions, potentially sharpening attention.
**Change:** QK_GAIN_INIT 5.25 → 7.0, all else identical to gqa_baseline.
**Result:** Pre-quant BPB=1.0797, post-quant BPB=1.0905 — within 0.0003 of baseline. No improvement.
**Status:** discard
**Learned:** QK gain saturated at 5.25. Going to 7.0 gives essentially zero benefit. The learnable q_gain parameter adapts during training regardless of initialization, so the init value matters less once it's "large enough." Don't revisit QK gain.

## Experiment 1 — GQA baseline on 4xA100

**Hypothesis:** Reproduce SOTA-equivalent architecture (11L×512d, SP8192, GQA, depth recurrence, parallel residuals, QK_GAIN=5.25) on 4xA100 with 3600s wallclock to establish local baseline.
**Change:** Used `train_gpt_04_09.py` with `LINEAR_ATTN_RATIO=0` (pure GQA, no GatedDeltaNet). Patched FA3→SDPA for A100 compatibility. Skipped GatedDeltaNet experiment due to OOM (float32 chunk processing needs >80GB).
**Result:** COMPLETE — pre-quant val_bpb=**1.0800**, post-quant val_bpb=**1.0908**, 4990 steps in 59.8min, 44.3GiB peak, model 17.2MB compressed (over 16MB limit).
**Status:** keep (baseline)
**Learned:** 4xA100 at 3600s matches 8xH100 at 600s for pre-quant quality. Quantization degrades BPB by ~0.011. Compressed model is 17.2MB (SOTA gets 15.97MB via LZMA code wrapper saving ~54KB, but main gap is model compressibility). GatedDeltaNet OOMs on A100 80GB at default batch. torch.compile incompatible with GatedDeltaNet's Python loops. Next: try higher QK_GAIN (6.0+) since SOTA README notes monotonic improvement up to 5.25.

