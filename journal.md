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

## Experiment 28 — LOGIT_SOFTCAP=25

**Hypothesis:** Fill the gap between 20 and 30 to confirm 20 is the peak.
**Change:** LOGIT_SOFTCAP=25 with WARMDOWN=0.85.
**Result:** Pre-quant=1.0778, post-quant=1.0890. Worse than 20, better than 30.
**Status:** discard
**Learned:** Full softcap sweep complete: 15(1.0777) < **20(1.0772)** > 25(1.0778) > 30(1.0793) > 50(worse). Softcap=20 is the definitive optimal value, providing −0.0021 BPB improvement. The improvement is from stronger logit compression acting as a regularizer.

## Experiments 25-27 — SC20+ScalarLR, SC20+EMA, SC20+WD080

**SC20+ScalarLR=0.03:** prequant=1.0780, postquant=1.0892. Doesn't stack.
**SC20+EMA=0.997:** prequant=1.0775, postquant=**1.0883**. Tied with best (marginally better post-quant).
**SC20+WD=0.80:** prequant=1.0774, postquant=1.0885. Also tied.
**Conclusion:** LOGIT_SOFTCAP=20 is the dominant finding (−0.0021 pre-quant, −0.0022 post-quant). It's robust: all three WD/EMA variants give ~same result. The full softcap sweep: 15 < **20** > 30 > 50. Three independent combo attempts (rope32, gradclip05, scalarLR03) all failed to stack, confirming the screening winners were correlated.

## Experiments 22-24 — Combo runs and softcap sweep

**Combo: softcap20+rope32:** prequant=1.0779, postquant=1.0892. Rope32 doesn't stack (+0.0007 worse).
**Combo: softcap20+gradclip05:** prequant=1.0777, postquant=1.0889. GradClip doesn't stack (+0.0005 worse).
**Softcap=15:** prequant=1.0777, postquant=1.0890. Slightly worse than 20.
**Conclusion:** Softcap=20 is the optimal value. Combinations don't help — the screening winners are correlated (all improve attention/gradient dynamics), not independent.

## Experiments 29-36 — train_gpt_improved.py exploration

**Baseline (exp29):** Pre-quant=1.08027, post-quant=1.09196, artifact=**15.96MB (fits 16MB!)**. 35.9M params, 5400 steps. Faster than 04_09 (1.6M vs 1.4M tok/s) but worse BPB due to fewer params.

**MLP capacity sweep:**
- MLP=4.5 (exp30): **Best BPB ever** (1.07647/1.08763) but artifact=17.17MB ❌
- MLP=4.25 (exp31): 1.07767/1.08914, artifact=16.57MB ❌
- MLP=4.06 (exp32): 1.08064/1.09212, artifact=16.10MB ❌ (marginal params don't help BPB)

**Feature tests (all on default MLP=4.0):**
- WARMDOWN_WD_MULT=1.5 (exp33): Worse by +0.0014 pre-quant
- NORMUON=1 (exp34): Within noise
- ROPE_DIMS=16 (exp35): Worse by +0.0005 vs 32 default

**Key insight:** The artifact budget is the binding constraint. Best BPB (MLP=4.5) doesn't fit under 16MB. The improved baseline with default params fits (15.96MB) but has worse BPB than our 04_09 softcap20 run. Need either better compression or architectural improvements that don't add params.

## Experiment 29 — train_gpt_improved.py baseline

**Hypothesis:** The improved script has our best HPs baked in (softcap=20, WD=0.85) plus ROPE_DIMS=32, Triton fused kernels, cleaner architecture. Should match or beat train_gpt_04_09.py.
**Change:** Switched to train_gpt_improved.py with FUSED_ROPE=0 FUSED_MLP=0 (disabled fused kernels for compile compat). Added SDPA fallback for A100.
**Result:** Pre-quant=1.08027, post-quant=1.09196, **artifact=15.96MB (under 16MB!)**, 5400 steps, 35.9M params.
**Status:** keep (fits artifact budget, but worse BPB than 04_09)
**Learned:** Improved script is faster (1.6M tok/s, 5400 steps vs 4989) and much smaller (15.96MB vs 17.27MB), but 2.9M fewer params hurts BPB by +0.003. The artifact size win is critical — this is the only run that fits under 16MB. Need to recover BPB, likely by increasing model capacity.



**Method:** Ran 10 HP variants at 1800s each for fast screening, plus a 1800s baseline reference (prequant=1.1105). Then promoted best winner to full 3600s.

**Screening results (sorted by pre-quant BPB):**
| Run | Change | Δ Pre-quant | Δ Post-quant |
|-----|--------|-------------|--------------|
| softcap20 | LOGIT_SOFTCAP=20 | **−0.0014** | **−0.0012** |
| rope32 | ROPE_DIMS=32 | −0.0009 | −0.0010 |
| gradclip05 | GRAD_CLIP_NORM=0.5 | −0.0007 | −0.0007 |
| scalarLR03 | SCALAR_LR=0.03 | −0.0005 | −0.0006 |
| embedlr03 | EMBED_LR=0.3 | ≈0 | ≈0 |
| beta2_099 | MUON_BETA2=0.99 | ≈0 | ≈0 |
| tiedLR05 | TIED_EMBED_LR=0.05 | +0.0011 | +0.0018 |
| mlp35 | MLP_MULT=3.5 | +0.0017 | +0.0022 |
| parres5 | PARALLEL_RESIDUAL_START=5 | +0.0019 | +0.0019 |
| softcap50 | LOGIT_SOFTCAP=50 | +0.0019 | +0.0047 |

**Full 3600s run — SOFTCAP=20:** Pre-quant BPB=**1.07718** (−0.00282), Post-quant=**1.08844** (−0.00231). **BEST RUN.**

**Learned:** Lower logit softcap (20 vs 30) is a significant win. Stronger logit compression helps generalization. RoPE_DIMS=32 and GRAD_CLIP=0.5 also show promise. Next: combine softcap20 with rope32 and gradclip05.

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

