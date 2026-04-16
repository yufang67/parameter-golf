# Strategy

<!-- ============================================================
     AGENT-MAINTAINED — you (the agent) own this file.
     Update it after experiments that change your understanding.
     This is your persistent memory across context window resets.
     ============================================================ -->

## Current phase

**Batch 5 complete (TTT on VarLen model):** Trained exp86 (same config as exp80) achieving a new sliding SOTA of **1.07405** (pre=1.0668, post-quant=1.0812). Then ran 10 TTT configs via new `EVAL_ONLY` mode. **All TTT variants hurt BPB by +0.027–0.031 vs the sliding baseline.** Fullparam TTT with exp58's proven SGD recipe (lr=5e-3, mom=0.9, clip=1.0) does NOT transfer to the varlen+improved model — damage is structural (first grad step degrades), not divergence. Found and fixed a critical bug: TTT scoring path needs cu_seqlens for varlen-trained models (else BPB≈1.73). Added NaN safeguards (never triggered — runs were stable but systematically worse). LoRA TTT at sl=4096 OOMs on 4×A100.

**Decision:** Drop TTT from the VarLen submission path. Sliding-window eval (1.07405) is the new SOTA for this script. Focus next on closing the artifact-size gap (exp80=16.06MB, exp86 checkpoint=15.9MB so exp86 may already fit — needs full size check).

## Current best

- **Run ID:** exp86_varlen_ttt_base (on train_gpt_improved_04_15.py, EVAL_ONLY reused by 10 TTT sweeps)
- **Pre-quant BPB:** 1.0668 (BEST EVER)
- **Post-quant BPB:** 1.0812 (BEST EVER)
- **Sliding BPB:** 1.07405 (BEST EVER, −0.00227 over exp80)
- **Compressed model:** 15.9MB (final_model.int6.ptz); full artifact size not yet measured
- **Config:** same as exp80 (VarLen+GPTQ128+hessian), seed/run differences only

Older:
- exp80_varlen: pre=1.06912 / post=1.08350 / sliding=1.07632, artifact=16.06MB ❌
- **Run ID:** exp80_varlen (on train_gpt_improved_04_15.py)
- **Pre-quant BPB:** 1.06912 (BEST EVER)
- **Post-quant BPB:** 1.08350 (BEST EVER)
- **Sliding BPB:** 1.07632
- **Artifact:** 16.06MB ❌ (improved script code too large — need stripped version)
- **Config:** train_gpt_improved_04_15.py, 11L×512d, SP8192, GATED_ATTENTION=1, MATRIX_CLIP_SIGMAS=15, MLP_MULT=4.35, VARLEN_ATTENTION=1, GPTQ_CALIBRATION_BATCHES=128, 4xA100 3600s
- **Runner-up (compliant):** exp58_ttt (TTT=1.07045, 15.99MB ✅)

## What's working

- **VARLEN_ATTENTION=1** — biggest single finding ever: −0.005 BPB pre-quant AND post-quant. Within-document attention prevents cross-doc leakage. No extra params.
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

- **Bigram hash embeddings** (1024×64, 2048×128) — no BPB improvement, adds 155-225KB artifact (exp77, exp78)
- **Trigram hash embeddings** (1024×64) — within noise, adds 155KB (exp79)
- **MoE MLP** (2 experts, top-1, 1 layer) — catastrophic: −0.009 BPB, slower training (exp83)
- **Clip=13 + hessian + MLP=4.0** — smaller MLP loses too much capacity (exp84)
- **Bigram + Trigram + MLP=4.0** — fits 15.52MB but BPB much worse (exp85)

## Completed experiments

**Batch 3 (exp66-75) — ALL COMPLETE.** No compliant improvements. All HP dimensions exhausted.

## Next hypotheses

**Priority 1 — Port VarLen to stripped script:**
1. Add flash_attn_varlen support to train_gpt_stripped.py (~20 lines of code, ~2KB)
2. Run with SOTA params + VARLEN_ATTENTION=1 + TTT_ENABLED=1
3. Expected: pre-quant ~1.069, post-quant ~1.083, artifact ~15.99MB ✅

**Priority 2 — VarLen + TTT combo:**
- VarLen improves training quality, TTT improves eval. Should stack.
- Expected TTT BPB: ~1.065 (−0.005 from VarLen + −0.002 from TTT)

**Priority 3 — VarLen + hessian clip or GPTQ optimizations:**
- Better quantization on top of VarLen-trained model
