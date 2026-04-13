# Strategy

<!-- ============================================================
     AGENT-MAINTAINED — you (the agent) own this file.
     Update it after experiments that change your understanding.
     This is your persistent memory across context window resets.
     ============================================================ -->

## Current phase

Architecture search complete for scalar HPs. Identified LOGIT_SOFTCAP=20 as main win.

## Current best

- **Run ID:** full_softcap20
- **Pre-quant BPB:** 1.07718
- **Post-quant BPB:** 1.08844
- **Config:** 11L×512d, SP8192, GQA, QK_GAIN=5.25, depth recur, parallel resid, EMA=0.9965, WARMDOWN=0.85, LOGIT_SOFTCAP=20, 4xA100 3600s
- **Improvement over original SOTA defaults:** −0.00282 pre-quant, −0.00231 post-quant

## What's working

- **LOGIT_SOFTCAP=20** is the biggest single finding: −0.0021 BPB from 30→20
- WARMDOWN=0.85 adds −0.0007 on top of SOTA's 0.72
- Softcap=20 is robust across WD (0.80, 0.85) and EMA (0.9965, 0.997) values
- 1800s screening runs are effective for ranking single-variable changes

## What's been tried and failed

- All previous HP tuning (see experiments 1-11)
- Combinations of screening winners: rope32, gradclip05, scalarLR03 don't stack with softcap20
- Softcap=15: worse than 20 (over-compressed)
- Softcap=50: much worse (under-compressed)
- GatedDeltaNet: OOMs on A100
- TRAIN_SEQ_LEN=4096: torch.compile fullgraph crash

## Next hypotheses
1. applied known working solution to `train_gpt_improved.py` if its not default.
2. explore any optimization on `train_gpt_improved.py`
3. **Model shape changes**: Need code changes to try 9L×576d or 12L×480d (requires custom LOOP_START/END, PARALLEL_RESIDUAL_START adjustments)
4. **Quantization improvements**: GPTQ params, different clip sigmas, more calibration batches
