# Strategy

<!-- ============================================================
     AGENT-MAINTAINED — you (the agent) own this file.
     Update it after experiments that change your understanding.
     This is your persistent memory across context window resets.
     ============================================================ -->

## Current phase

Architecture search — SOTA hyperparameters are near-optimal; focus on structural changes

## Current best

- **Run ID:** warmdown_085
- **Pre-quant BPB:** 1.0793
- **Post-quant BPB:** 1.0906
- **Config:** 11L×512d, SP8192, GQA, QK_GAIN=5.25, depth recur, parallel resid, EMA=0.9965, WARMDOWN=0.85, 4xA100 3600s

## What's working

- SOTA architecture reproduces well on 4xA100: pre-quant BPB matches 8xH100 results
- train_gpt_04_09.py with LINEAR_ATTN_RATIO=0 is our reliable working script
- ~1.4M tok/s pre-recurrence, ~1.15M with depth recurrence active
- SOTA hyperparameters (LR, warmdown, WD, EMA) are near-optimal

## What's been tried and failed

- GatedDeltaNet (LINEAR_ATTN_RATIO=0.75): OOMs on A100 80GB
- NUM_LOOPS=3: fewer steps, worse BPB
- QK_GAIN_INIT=7.0: saturated
- WARMDOWN_FRAC=0.50: too late warmdown start
- WARMDOWN_FRAC=0.90: no better than 0.85
- MATRIX_LR=0.030: within noise
- MUON_WD=0.06: faster mid-training but worse final convergence
- ENABLE_LOOPING_AT=0.20: fewer steps, worse final
- **Conclusion:** SOTA HPs are near-optimal. Only confirmed gain: WARMDOWN 0.72→0.85 (−0.0007)

## Next hypotheses

1. **Muon WD=0.06 + WARMDOWN=0.85**: Combine best warmdown with lower regularization — 0.095 is quite aggressive
2. **EMA decay=0.998 + WARMDOWN=0.85**: Slower EMA averaging might pair well with longer warmdown
3. **Wider model (768d × 7L)**: Trade depth for width at similar param count
4. **train_gpt_new.py features**: N-gram hash embeddings, full attention residual from the experimental script
