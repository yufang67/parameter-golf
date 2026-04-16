# Strategy

<!-- ============================================================
     AGENT-MAINTAINED — you (the agent) own this file.
     Update it after experiments that change your understanding.
     This is your persistent memory across context window resets.
     ============================================================ -->

## Current phase

Architecture converged — 30 experiments across 3 batches (exp46-75) complete. All HP dimensions exhausted. Best compliant config is exp58_ttt (TTT=1.07045, 15.99MB). Further gains require fundamentally new approaches: heterogeneous layers, novel quantization, tokenizer co-design, or quantization-aware training.

## Current best

- **Run ID:** exp58_ttt
- **Pre-quant BPB:** 1.07409
- **Post-quant BPB:** 1.08890
- **TTT BPB:** 1.07045 (best eval metric)
- **Artifact:** 15.99MB ✅ (under 16MB, ~11KB to spare)
- **Config:** train_gpt_stripped.py (83KB), 11L×512d, SP8192, GATED_ATTENTION=1, MATRIX_CLIP_SIGMAS=15, MLP_MULT=4.35, SOFTCAP=20, WARMDOWN=0.85, ROPE_DIMS=32, GPTQ_CALIBRATION_BATCHES=128, TTT_ENABLED=1, 4xA100 3600s
- **Runner-up:** exp56_stripped_gptq128 (post-quant=1.08881, no TTT)
- **Improvement over gated_clip15_mlp435:** −0.00109 pre-quant, −0.00137 post-quant, TTT=1.07045

## What's working

- **LOGIT_SOFTCAP=20** — biggest single HP finding: −0.0021 BPB from 30→20
- **GATED_ATTENTION=1** — per-head sigmoid gates: −0.0015 BPB with only +45K params
- **MATRIX_CLIP_SIGMAS** — key tradeoff lever: lower clip = better post-quant BPB but larger artifact
- **MLP_MULT=4.35** — maximum capacity that fits 16MB with clip=15
- **GPTQ_CALIBRATION_BATCHES=128** — more calibration data improves post-quant by −0.0012 (1.09027→1.08905)
- **HESSIAN_CLIP_LAMBDA=0.3** — Hessian-aware per-row clipping also improves post-quant by −0.0008
- **LOOP_LAYER_BITS=8** — int8 for looped layers gives best-ever post-quant (1.08471) but +3MB artifact
- **TTT_ENABLED=1** — free eval-time BPB win: −0.00184 (1.07229→1.07045). No training or artifact cost.
- WARMDOWN=0.85 adds −0.0007 on top of SOTA's 0.72
- ROPE_DIMS=32 confirmed better than 16 on improved.py
- Stripped script (83KB vs 96KB) saves 13KB code bytes for more model budget

## What's been tried and failed

- HP tuning: QK_GAIN, MATRIX_LR, MUON_WD, EMA_DECAY, SCALAR_LR, EMBED_LR — all near-optimal in SOTA
- Combo stacking: rope32, gradclip05, scalarLR03 don't stack with softcap20
- LOOP_EMBEDDINGS, NORMUON, WARMDOWN_WD_MULT — marginal or worse
- GatedDeltaNet: OOMs on A100
- TRAIN_SEQ_LEN=4096: torch.compile crash
- LZMA code wrapper: incompatible with Triton @jit
- CLIP=13 with any MLP ≤ 4.20 — still doesn't fit 16MB (artifact ~16.57MB)
- CLIP=14 — slower throughput, worse on all metrics; doesn't fit even with MLP=4.25 + stripped code (exp45, exp65)
- MLP=4.40 with clip=15 — 290KB over budget
- Per-group clip tightening (early=0.85, loop=0.9) — 450KB over budget
- Window attention (layers 0-2, size=512) — severe quality loss (+0.015 pre-quant, +0.019 post-quant)
- Hessian + GPTQ128 combo — hessian makes weights less compressible, artifact 15KB over (exp57)
- MLP=4.30 — fits but worse BPB than 4.35 (exp59)
- GPTQ_CAL=256 — no improvement over 128 (exp61, exp73)
- EVAL_STRIDE=32 — same BPB, 2x eval time (exp62)
- EMA=0.997 — worse pre-quant than 0.9965 (exp63)
- 12L+MLP=3.85 — fewer steps, worse BPB than 11L (exp64, exp71)
- SOFTCAP=18 — worse than 20 (exp67)
- WARMDOWN=0.88 — within noise of 0.85, artifact doesn't fit (exp68)
- EMBED_BITS=6 — saves 1MB artifact but catastrophic post-quant BPB (+0.029, exp69, exp72)
- GRAD_CLIP=0.5 — worse on all metrics, doesn't stack (exp70)
- MUON_WD=0.08 — worse than 0.095 (exp74)
- QK_GAIN=6.0+SOFTCAP=18 — within noise, no benefit (exp75)
- Seed-dependent artifact size — exp66 (seed=42) was 16.01MB vs exp58 (seed=1337) 15.99MB

## Completed experiments

**Batch 3 (exp66-75) — ALL COMPLETE.** No compliant improvements. All HP dimensions exhausted.

## Next hypotheses

Requires fundamentally new approaches to improve beyond exp58_ttt (TTT=1.07045):

1. **Strip more code** — further reduce code bytes to give more model budget. Target: fit GPTQ256+TTT (exp73 was 16KB over) (done)
2. **Quantization-aware training (QAT)** — train with quantization noise to improve post-quant BPB
3. **Heterogeneous layers** — cheaper early layers (fewer KV heads, smaller MLP) + stronger late layers
4. **Tokenizer co-design** — SP4096 or SP16384 with matching model capacity
5. **Mixed-precision quantization** — int8 for critical layers, int4/int5 for less critical ones, optimize for total artifact budget
6. **Code golf** — minimize code size to squeeze more model capacity into 16MB (code can be compressed to around 30K separately. so its done)
