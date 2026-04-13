# Strategy

<!-- ============================================================
     AGENT-MAINTAINED — you (the agent) own this file.
     Update it after experiments that change your understanding.
     This is your persistent memory across context window resets.
     ============================================================ -->

## Current phase

Quantization/compression tuning — balancing post-quant BPB vs artifact size within 16MB budget

## Current best

- **Run ID:** gated_clip15_mlp435
- **Pre-quant BPB:** 1.07518
- **Post-quant BPB:** 1.09027
- **Artifact:** 15.98MB ✅ (under 16MB, 24KB to spare)
- **Config:** train_gpt_improved.py, 11L×512d, SP8192, GATED_ATTENTION=1, MATRIX_CLIP_SIGMAS=15, MLP_MULT=4.35, SOFTCAP=20, WARMDOWN=0.85, ROPE_DIMS=32, 4xA100 3600s
- **Improvement over original SOTA defaults:** −0.00482 pre-quant BPB

## What's working

- **LOGIT_SOFTCAP=20** — biggest single HP finding: −0.0021 BPB from 30→20
- **GATED_ATTENTION=1** — per-head sigmoid gates: −0.0015 BPB with only +45K params
- **MATRIX_CLIP_SIGMAS** — key tradeoff lever: lower clip = better post-quant BPB but larger artifact
- **MLP_MULT=4.35** — maximum capacity that fits 16MB with clip=15
- WARMDOWN=0.85 adds −0.0007 on top of SOTA's 0.72
- ROPE_DIMS=32 confirmed better than 16 on improved.py
- train_gpt_improved.py is faster than 04_09 (~1.5M vs ~1.4M tok/s pre-recurrence)

## What's been tried and failed

- HP tuning: QK_GAIN, MATRIX_LR, MUON_WD, EMA_DECAY, SCALAR_LR, EMBED_LR — all near-optimal in SOTA
- Combo stacking: rope32, gradclip05, scalarLR03 don't stack with softcap20
- LOOP_EMBEDDINGS, NORMUON, WARMDOWN_WD_MULT — marginal or worse
- GatedDeltaNet: OOMs on A100
- TRAIN_SEQ_LEN=4096: torch.compile crash
- LZMA code wrapper: incompatible with Triton @jit

## Next hypotheses

1. **CLIP=14 + MLP=4.35** — find the sweet spot between clip=13 (16.83MB, great post-quant) and clip=15 (15.98MB, fits)
2. **CLIP=13 + MLP=4.15** — reduce MLP to fit clip=13's better post-quant in 16MB
3. **GPTQ_CALIBRATION_BATCHES=128** — more calibration for better quantization
4. **Reproducibility run** — re-run best config with different seed
