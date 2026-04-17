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

## TTT forward-path fix (train_gpt_improved_04_16.py) — follow-up to varlen_ttt_invest

**Investigation output (`varlen_ttt_invest/results.md`):** 10 diagnostic runs on a single checkpoint found that zero-LoRA `forward_ttt` gives ttt_bpb=1.33587 vs sliding bpb=1.07407 — a ~+0.26 gap the report attributed to "a manual re-implementation of `Block.forward` in `_block_with_lora` that has drifted."

**Findings after code audit + standalone bitwise test:**
- Built a random-init 11L model (varlen, looping, xsa, parallel residuals, gated, ln_scale all ON) and compared per-layer states: **zero-LoRA `forward_ttt` == `forward_logits` bitwise (max_abs=0.0)** across every layer and on final per-token loss.
- So `_block_with_lora` is NOT mathematically drifted; the 0.26 BPB gap is a **protocol mismatch** (TTT scores only the first 50 docs via `TTT_MAX_DOCS=50`, where early-doc positions have far less context than the full-dataset sliding average).
- However, there IS a real training-distribution mismatch worth fixing: `_block_with_lora` always used the dense `flash_attn_3_func` kernel, even when the model was trained with `flash_attn_3_varlen_func`. Subtle kernel-level numerical differences (different FP reduction order, no per-doc boundary marking) can accumulate and are the kind of thing that erodes TTT's marginal gains.
- Bonus bug in `_collect_zero_lora_debug_states`: the `use_ttt=True` branch passed a fresh `{}` counter per layer to `_add_loop_emb` instead of the shared `lpc` counter — would cause false divergence reports in looping mode.

**Fixes applied to `train_gpt_improved_04_16.py`:**
1. `_block_with_lora` now accepts `(cu_seqlens, max_seqlen)` and dispatches to `flash_attn_3_varlen_func` when the model uses varlen. Non-varlen path unchanged.
2. `forward_ttt` now accepts `(cu_seqlens, max_seqlen)`. When `use_varlen=True` and caller omits them, auto-synthesizes `cu_seqlens = arange(0, B*T+1, T)` treating each batch row as one document (matches how `eval_val_ttt` builds its inputs: each row is a contiguous doc slice). This makes the TTT attention kernel identical to training.
3. `_collect_zero_lora_debug_states` bug fix (shared `lpc` in both `use_ttt` branches).

**Verification:** Standalone parity test passes bitwise (delta=0) for both:
- `forward_logits(x)` vs `forward_ttt(x,y,zero_lora)` (dense path)
- `forward_logits(x, cu, max_sl)` vs `forward_ttt(x,y,zero_lora, cu, max_sl)` (varlen path)

**Status:** forward path is now provably correct. The residual TTT-worse-than-sliding gap (confirmed in batch5 experiments) is structural (LoRA updates systematically degrade a well-trained varlen model, and TTT scores a biased subset of docs), not a forward-path bug. Keep TTT dropped from the submission path; the cu_seqlens threading is a correctness improvement available if anyone revisits TTT on a future checkpoint.

---

## Experiments 86, 97-106 — Batch 5: TTT on VarLen+Improved (Fix divergence, 10 configs)

**Motivation:** User-directed — run 10 experiments on TTT, warning it "may diverge or give worse results." Starting SOTA is exp80_varlen (sliding=1.07632) but artifact=16.06MB (over cap). Goal: establish whether TTT can help the VarLen-trained improved model and add divergence safeguards.

**Fixes added to `train_gpt_improved_04_15.py`:**
1. **LoRA TTT safeguards** (`eval_val_ttt`): NaN/Inf loss skip, `nan_to_num_` on grads, `clip_grad_norm_` (env `TTT_LORA_GRAD_CLIP`), post-step param-NaN detect → reset LoRA+optimizer.
2. **Fullparam TTT safeguards** (`eval_val_ttt_fullparam`): same NaN guards + state-dict snapshot/restore on NaN.
3. **VarLen attention in TTT paths:** `forward_ttt` and `_block_with_lora` now accept `cu_seqlens, max_seqlen`; auto-synthesizes cu_seqlens (per-row = one doc) when missing. **Critical bug fix** — scoring path in `eval_val_ttt_fullparam` previously fed no cu_seqlens → varlen-trained model saw cross-doc attention → BPB blew up to 1.73+. Fixed to build cu_seqlens from BOS like `eval_val_sliding`.
4. **Optimizer selector** `TTT_FP_OPTIMIZER={sgd,adamw}` (default SGD lr=0.005 mom=0.9, matching proven exp58 config; previous AdamW lr=1e-3 default caused instant blowup).
5. **EVAL_ONLY=1 mode** — loads `final_model.pt`+`.ptz` from disk and skips training, enabling fast (15-min) TTT sweeps on a frozen checkpoint.

**exp86 (train once, reuse):** VarLen+GPTQ+hessian, same config as exp80. Pre-quant=1.0668, post-quant=1.0812, sliding=**1.07405** (beats exp80's 1.07632). Checkpoint saved as `final_model.pt` + `.ptz` (15.9MB). Used as base for all EVAL_ONLY runs below.

**Fullparam TTT sweep (exp97–104)** — all EVAL_ONLY against exp86, reusing exp58's proven SGD protocol + new varlen cu_seqlens scoring:
| Exp | Config | TTT BPB | Δ vs sliding |
|---|---|---|---|
| 97 | SGD lr=5e-3 (exp58-like) | 1.10205 | +0.028 |
| 98 | SGD lr=3e-3 | 1.10158 | +0.028 |
| 99 | SGD lr=1e-2 | 1.10300 | +0.029 |
| 100 | SGD lr=5e-3 ep=1 | 1.10138 | +0.027 |
| 101 | SGD lr=5e-3 ep=5 | 1.10266 | +0.029 |
| 102 | SGD lr=5e-3 chunk=16k | 1.10377 | +0.030 |
| 103 | SGD lr=5e-3 chunk=64k | 1.10206 | +0.028 |
| 104 | AdamW lr=3e-5 | 1.10502 | +0.031 |

**LoRA TTT (exp105, exp106):** sl=4096 OOMed (48-step adapt × 4096-seq too heavy on 4×A100). sl=2048 had separate chunk/context protocol mismatch (initial chunks see only 2048-token context vs sliding's 4032).

**Result:** All 8 fullparam TTT variants give BPB=1.101–1.105, systematically **+0.027 to +0.031 worse** than the sliding baseline (1.07405). Variance between configs is <0.004 — the damage is structural, not a diverging-run issue. Single-epoch (ep=1) is no better than 2 or 5 epochs → damage happens on the very first grad step. No NaN safeguards were triggered in any run.

**Status:** discard TTT for VarLen submission path. Sliding-window eval (1.07405) remains best.

**Learned:**
- The proven stripped-script TTT recipe (SGD lr=0.005, mom=0.9, clip=1.0, cos LR decay over chunks) does NOT transfer to varlen-trained improved-script models. Hypothesis: the varlen+depth-recurrence+GQA+softcap model is more tightly optimized; bf16 SGD updates to the full model (momentum buffers in bf16, ~38M params) introduce noise that degrades rather than improves the per-chunk alignment. On stripped (exp58 smaller 23M model) the same protocol helped by −0.0019; here it hurts by +0.028.
- **Do not feed a varlen-trained model cross-doc attention at eval time.** `forward_logits` without cu_seqlens silently produces BPB≈1.73 (catastrophic). Sliding-window eval correctly builds per-window cu_seqlens from BOS tokens; TTT scoring now mirrors this.
- EVAL_ONLY mode is a 10× speedup for TTT sweeps — train once (~1h), then 8-10 hparam configs at 15-17min each.
- For LoRA TTT on 4×A100 with full varlen context, sl ≤ 2048 is the limit with 48 adapt steps × stride-64 scoring; can't extend to match sliding's 4032-token context without further OOM mitigation.
- **Next steps:** drop TTT from VarLen submission path. Focus remaining budget on (a) shrinking exp86's 15.9MB artifact to add headroom, or (b) other levers (better quantization, code-size reduction) to get exp80's 16.06MB artifact under the 16MB cap.

---

## Experiments 76-85 — Batch 4: New Features on train_gpt_improved_04_15.py

**Exp76 — Improved baseline (SOTA params):** Pre-quant=1.07396, post-quant=1.08888, sliding=1.07219, artifact=16.06MB ❌. Matches SOTA BPB within noise but improved script's larger code (145KB vs 83KB) makes artifact 60KB over budget.

**Exp77 — Bigram hash (small, 1024×64):** Pre-quant=1.07438, post-quant=1.08909, sliding=1.07248, artifact=16.22MB ❌. Bigram adds ~155KB artifact, BPB slightly worse. Not viable.

**Exp78 — Bigram hash (medium, 2048×128):** Pre-quant=1.07548, post-quant=1.09019, sliding=1.07368, artifact=16.28MB ❌. Larger bigram is worse across the board. More params don't help.

**Exp79 — Trigram hash (1024×64):** Pre-quant=1.07434, post-quant=1.08894, sliding=1.07235, artifact=16.21MB ❌. Within noise of baseline. N-gram hash embeddings don't help.

**Exp80 — VarLen attention:** Pre-quant=**1.06912**, post-quant=**1.08350**, sliding=1.07632, artifact=16.06MB ❌. **MASSIVE improvement!** −0.005 pre-quant, −0.005 post-quant vs baseline. Within-document attention prevents cross-doc leakage during training. No extra params. Slower throughput (~1.1M vs 1.5M tok/s) means fewer steps but much better quality. Artifact over only due to code size.

**Exp81 — Hessian clip 0.3:** Pre-quant=1.07372, post-quant=1.08887, sliding=1.07220, artifact=16.09MB ❌. Within noise of baseline. Hessian adds 30KB to artifact.

**Exp82 — Bigram + VarLen:** Pre-quant=1.06911, post-quant=1.08332, sliding=1.07607, artifact=16.21MB ❌. VarLen does all the work — bigram adds nothing, just +155KB artifact.

**Exp83 — MoE tiny (2 experts, 1 layer):** Pre-quant=1.08310, post-quant=1.09747, sliding=1.08091, artifact=16.04MB ❌. MoE is terrible — slower training, fewer steps, +0.009 worse BPB.

**Exp84 — Clip=13 + hessian + MLP=4.0:** Pre-quant=1.07777, post-quant=1.08947, sliding=1.07282, artifact=16.04MB ❌. Smaller MLP loses too much capacity. Clip=13 doesn't compensate.

**Exp85 — Bigram + Trigram (MLP=4.0):** Pre-quant=1.07731, post-quant=1.09264, sliding=1.07604, artifact=**15.52MB** ✅. Fits! But BPB worse than MLP=4.35 baseline. N-gram hashes don't compensate for reduced MLP capacity.

**Key learnings:**
- **VarLen attention is the clear winner** — biggest BPB improvement seen in this entire project (−0.005 pre-quant, −0.005 post-quant). It prevents cross-document attention leakage during training.
- **N-gram hash embeddings (bigram/trigram) don't help** — they add artifact size without improving BPB.
- **MoE is not viable** — slower training means fewer steps, worse BPB despite extra capacity.
- **Code size is the remaining blocker** — improved script is 145KB vs 83KB stripped, putting all runs ~60KB over. Need to strip the code or port VarLen to stripped script.
- **Next step:** Port VARLEN_ATTENTION=1 to train_gpt_stripped.py to get the BPB win within the 16MB artifact budget.

## Exp58 Repro — Apr 16 confirmation run

Re-ran the current best `exp58_ttt` configuration unchanged to verify that the best result is reproducible from the saved setup rather than a one-off. The repro landed at pre-quant `1.07415`, post-quant `1.08901`, sliding `1.07239`, and TTT `1.07058`, with total artifact size `15.989MB`, so it still fits comfortably under 16MB. This is effectively identical to the original `exp58_ttt` result within about `1e-4` BPB and a few bytes of artifact size. The main takeaway is that the best config is stable enough to treat as real, so further work should focus on genuine architectural changes rather than revalidating the baseline again.

## Experiments 66-75 — Batch 3: HP Refinement & Reproducibility

**Exp66 — Reproducibility (SEED=42 + TTT):** Pre-quant=1.07460, post-quant=1.08954, sliding=1.07298, TTT=**1.07112**, artifact=16.01MB ❌. Reproduces within noise of exp58 (TTT=1.07045 vs 1.07112, Δ=0.00067). Seed variance ~0.0005 pre-quant. Artifact 13KB over — tight margin is seed-dependent.

**Exp67 — SOFTCAP=18:** Pre-quant=1.07500, post-quant=1.08997, sliding=1.07337, artifact=16.02MB ❌. Worse than softcap=20 across the board. Confirms 20 is the sweet spot.

**Exp68 — WARMDOWN=0.88:** Pre-quant=1.07415, post-quant=**1.08931**, sliding=**1.07268**, artifact=16.01MB ❌. Slightly better pre-quant and post-quant than 0.85, but artifact 13KB over. The improvement is within noise. Best warmdown remains 0.85.

**Exp69 — EMBED_BITS=6:** Pre-quant=1.07425, post-quant=**1.11828**, artifact=**14.97MB** ✅✅. Fits with 1MB to spare! But post-quant BPB is catastrophic (+0.029). 6-bit embeddings destroy quality. Not viable.

**Exp70 — GRAD_CLIP=0.5:** Pre-quant=1.07545, post-quant=1.09044, sliding=1.07393, artifact=16.01MB ❌. Worse on all metrics. Confirms gradclip doesn't stack with softcap20 (as found in earlier screening).

**Exp71 — 12L + MLP=3.85 + TTT:** Pre-quant=1.07934, post-quant=1.09403, sliding=1.07740, TTT=1.07557, artifact=16.01MB ❌. 12L still worse than 11L — fewer steps (4852 vs 5229) kills it. Even TTT can't save it.

**Exp72 — EMBED_BITS=6 + MLP=4.45:** Pre-quant=1.07440, post-quant=**1.11709**, artifact=**15.25MB** ✅. Same story as exp69 — 6-bit embeddings ruin post-quant BPB despite huge size savings and better pre-quant from bigger MLP.

**Exp73 — GPTQ_CAL=256 + TTT:** Pre-quant=1.07421, post-quant=1.08931, sliding=1.07267, TTT=**1.07076**, artifact=16.02MB ❌. GPTQ256 gives slightly better post-quant than 128 (1.08931 vs 1.08890) but artifact 16KB over. TTT=1.07076 is close to exp58's 1.07045.

**Exp74 — MUON_WD=0.08:** Pre-quant=1.07462, post-quant=1.09062, sliding=1.07398, artifact=16.02MB ❌. Worse on all metrics. WD=0.095 remains optimal.

**Exp75 — QK_GAIN=6.0 + SOFTCAP=18:** Pre-quant=1.07475, post-quant=1.08954, sliding=1.07287, artifact=16.02MB ❌. Within noise of default (QK=5.25, SC=20). No benefit to this combo.

**Key learnings:**
- **No compliant improvements found.** exp58_ttt remains the best (TTT=1.07045, 15.99MB ✅).
- **Artifact size is seed-dependent.** exp66 (seed=42) was 16.01MB while exp58 (seed=1337) was 15.99MB — a 23KB difference from seed alone.
- **EMBED_BITS=6 saves 1MB** but costs +0.029 post-quant BPB — not viable with current architecture.
- **All HP tweaks (softcap, warmdown, gradclip, muon_wd, qk_gain)** are near-optimal. No further gains from tuning.
- **Architecture is converged.** Need fundamentally new approaches: different tokenizer, heterogeneous layers, or novel quantization to improve further.

## Experiments 59-65 — Batch 2: MLP/Architecture/Eval Sweep

**Exp59 — MLP=4.30 (safe fit):** Pre-quant=1.07468, post-quant=1.08959, sliding=1.07298, artifact=15.99MB ✅. 5231 steps. MLP=4.30 fits comfortably but worse BPB than 4.35 across the board. No benefit to smaller MLP.

**Exp60 — MLP=4.30 + LOOP_BITS=7 + hessian_clip=0.3:** Pre-quant=1.07445, post-quant=**1.08624**, sliding=**1.06955**, artifact=17.18MB ❌. 5228 steps. Best post-quant and sliding BPB ever, but 1.2MB over budget. 7-bit loop layers improve quant quality dramatically but add too much size.

**Exp61 — GPTQ_CALIBRATION=256:** Pre-quant=1.07435, post-quant=1.08916, sliding=1.07252, artifact=15.99MB ✅. 5230 steps. Diminishing returns: 256 batches ≈ identical to 128 (1.08916 vs 1.08881). Not worth the extra calibration time.

**Exp62 — EVAL_STRIDE=32:** Pre-quant=1.07413, post-quant=1.08892, sliding=1.07230, artifact=15.99MB ✅. 5232 steps. Stride=32 gives same sliding BPB as stride=64 (1.07230 vs 1.07229) but takes 2x eval time (890s vs 450s). No benefit.

**Exp63 — EMA=0.997:** Pre-quant=1.07475, post-quant=1.08896, sliding=1.07234, artifact=15.99MB ✅. 5229 steps. Worse pre-quant (+0.0007). 0.9965 remains optimal.

**Exp64 — 12L + MLP=3.85 (depth over width):** Pre-quant=1.07909, post-quant=1.09352, sliding=1.07687, artifact=15.98MB ✅. 4850 steps. Extra layer hurts: fewer steps (4850 vs 5228) and worse BPB everywhere. 11L is the sweet spot.

**Exp65 — CLIP=14 + MLP=4.25:** Pre-quant=1.07664, post-quant=1.08976, sliding=1.07317, artifact=16.11MB ❌. 5279 steps. Clip=14 doesn't fit even with MLP=4.25 and stripped code. Tighter clip needs even smaller MLP to compensate, but that hurts BPB.

**Key learnings:**
- Diminishing returns on quantization tuning: GPTQ_CAL=128 is sufficient, 256 doesn't help.
- EVAL_STRIDE=32, EMA=0.997 don't improve over current config.
- LOOP_BITS=7 is promising (best-ever sliding=1.06955) but needs artifact shrinkage.
- 12L architecture is strictly worse — 11L with bigger MLP is better use of params.
- Current config (exp58/56) appears near-optimal for this architecture class.

## Experiment 58 — TTT (Score-First Test-Time Training)

**Hypothesis:** TTT adapts model at eval time for free BPB gain (SOTA uses it).
**Result:** Pre-quant=1.07409, post-quant=1.08890, sliding=1.07229, **TTT=1.07045**. Artifact=15.99MB ✅. 5232 steps.
**Status:** keep — TTT gives −0.00184 BPB on top of sliding window (1.07229→1.07045)
**Learned:** TTT is a free eval-time win with no training or artifact cost. Same model as exp56 but with TTT_ENABLED=1. Should be enabled for all future submission candidates. TTT eval takes ~870s (14.5min) vs 450s for regular sliding window.

## Experiment 57 — Stripped + GPTQ128 + Hessian (completed)

**Hypothesis:** Combine two best quantization wins (GPTQ_CAL=128 + hessian_clip=0.3) on stripped script.
**Result:** Pre-quant=1.07437, post-quant=1.08949, artifact=**16.015MB** ❌ (15KB over!). 5231 steps.
**Status:** discard (over budget)
**Learned:** Hessian clipping makes weights less compressible, adding ~26KB to artifact vs exp56 (15.99→16.015MB). The two techniques don't combine well within budget. Exp56 (GPTQ128 alone, 15.99MB) remains best compliant.

## Experiments 56-57 — Stripped script + best config

**Exp56 — Stripped script + GPTQ_CAL=128:** Pre-quant=**1.07403**, post-quant=**1.08881**, artifact=**15.99MB** ✅. 5228 steps, 38.2M params. **NEW BEST COMPLIANT RUN!** The stripped script (83KB vs 88KB code) saves ~5KB, giving more room for model. Post-quant is our best compliant at 1.08881, beating exp48 (1.08905). Sliding window BPB=1.07215.

**Exp57 — Stripped + GPTQ_CAL=128 + hessian_clip=0.3:** CRASHED at step 2500 (~23min). Killed by signal before completion. Would have been interesting to see if hessian+GPTQ stacks with stripped script.

**Key takeaway:** Smaller code file = more model budget within 16MB. The stripped script approach is the right direction. Current best compliant: **exp56** (pre=1.07403, post=1.08881, 15.99MB).

## Experiment 54 — Window Attention (completed)

**Hypothesis:** Local attention (size=512) on layers 0-2 could be faster and save memory for more throughput.
**Change:** WINDOW_ATTN_SIZE=512, WINDOW_ATTN_LAYERS=0,1,2. CLIP=15, MLP=4.35.
**Result:** Pre-quant=1.08994, post-quant=**1.10862**, artifact=16.01MB. 5352 steps (more steps due to faster window attn).
**Status:** discard
**Learned:** Window attention on early layers severely hurts quality (+0.015 pre-quant, +0.019 post-quant). Despite more steps (5352 vs 5228), the restricted context in early layers damages representation learning. Full attention needed at all layers.

## Experiments 46-54 — Batch of 10: Compression & Architecture Sweep

**Exp46 — Clip=15/MLP=4.35 rerun (baseline verify):** CRASHED (killed by signal). No results.

**Exp47 — Hessian-aware clipping (hessian_clip_lambda=0.3):** Pre-quant=1.07428, post-quant=**1.08944**, artifact=16.04MB ✅. 5229 steps. Hessian clipping improves post-quant by −0.0008 vs gated_clip15_mlp435 (1.09027→1.08944) and still fits 16MB (60KB to spare). New best compliant post-quant!

**Exp48 — GPTQ_CALIBRATION_BATCHES=128:** Pre-quant=1.07430, post-quant=**1.08905**, artifact=16.01MB ✅. 5226 steps. More calibration data further improves post-quant to 1.08905 — best compliant post-quant! Only 12KB headroom though.

**Exp49 — LOOP_LAYER_BITS=8 (int8 for looped layers):** Pre-quant=1.07409, post-quant=**1.08471** (best ever!), artifact=19.01MB ❌. The int8 looped layers give fantastic post-quant BPB but add 3MB to artifact. Would need massive architectural shrink.

**Exp50 — Per-group clip tightening (early=0.85, loop=0.9):** Pre-quant=1.07400, post-quant=**1.08723**, artifact=16.45MB ❌. Great post-quant (+0.003 better than baseline) but 450KB over budget.

**Exp51 — CLIP=13/MLP=4.15:** Pre-quant=1.07662, post-quant=1.08790, artifact=16.57MB ❌. MLP=4.15 still too big for clip=13. 570KB over.

**Exp52 — CLIP=13/MLP=4.20:** Pre-quant=1.07641, post-quant=1.08780, artifact=16.57MB ❌. Same artifact as MLP=4.15 surprisingly — clip=13 dominates the size, MLP barely matters.

**Exp53 — CLIP=15/MLP=4.40:** Pre-quant=1.07433, post-quant=1.08897, artifact=16.29MB ❌. MLP=4.40 pushes capacity but adds 310KB vs MLP=4.35. Over budget.

**Exp54 — Window attention (layers 0-2, size=512):** INCOMPLETE — last seen at step 3500/20000 (35min). Faster tok/s (~1.5M) in early layers but training slowed after loop warmup.

**Key learnings:**
- **GPTQ_CALIBRATION_BATCHES=128 is the new best compliant config** (1.08905 post-quant, 16.01MB).
- Hessian clipping also helps and stacks, but barely fits.
- CLIP=13 doesn't fit even with MLP=4.15 — clip=13 is fundamentally incompatible with 16MB.
- LOOP_LAYER_BITS=8 shows int8 for shared layers is a goldmine for post-quant but needs smaller base model.
- Per-group clip tightening works but pushes artifact over limit.
- Next: try combining GPTQ_CALIBRATION=128 + hessian_clip. Consider MLP=4.30 as smaller compliant option.

## Experiment 45 — CLIP=14 + MLP=4.35

**Hypothesis:** Clip=14 is the midpoint between clip=13 (best post-quant 1.08670, artifact 16.83MB ❌) and clip=15 (compliant at 15.98MB). Should give better post-quant than clip=15 while hopefully fitting 16MB.
**Change:** MATRIX_CLIP_SIGMAS=14 with GATED_ATTENTION=1, MLP_MULT=4.35. Also using brotli compressor (new) and hessian_clip_lambda=0.3.
**Result:** Pre-quant=1.08086, post-quant=**1.09351**, artifact=16.39MB ❌. Only 4394 steps (vs 5122 for clip=15).
**Status:** discard
**Learned:** Clip=14 worse on every metric. Slower throughput (~0.98M tok/s at step 4000) yielded 700 fewer steps than clip=15 run. New code features (hessian_clip_lambda=0.3, brotli/ANS compressor) may have changed baseline. Need to re-run clip=15 with current code to verify old results still hold. ANS compression gave 16.30MB model — bigger than zlib was for clip=15.

## Experiment 44 — GATED + CLIP=13 + MLP=4.35

**Hypothesis:** Clip=13 should give better post-quant BPB than clip=15 while hopefully fitting under 16MB.
**Change:** MATRIX_CLIP_SIGMAS=13 with GATED_ATTENTION=1, MLP_MULT=4.35.
**Result:** Pre-quant=1.07522, post-quant=**1.08670** (best overall!), but artifact=16.83MB ❌.
**Status:** keep (best post-quant, over budget)
**Learned:** Clip=13 gives much better post-quant (1.08670 vs 1.09027) but +850KB artifact. Clip=15+MLP=4.35 (15.98MB) remains best compliant config.

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

**🎉 NEW BEST: gated_clip15_mlp435**
MLP=4.35 pushes capacity further while staying under 16MB (15.98MB, 24KB to spare):
- Pre-quant BPB = **1.07518** (best ever, −0.00071 vs MLP=4.25)
- Post-quant BPB = **1.09027** (−0.00077 vs MLP=4.25)
- 38.0M params, 5122 steps

The recipe: GATED_ATTENTION for BPB + loose CLIP_SIGMAS=15 for compression + max MLP capacity that fits.

## Experiments 37-42 — Compression tuning + Gated Attention combos

**Key discovery: MATRIX_CLIP_SIGMAS controls artifact size vs post-quant BPB tradeoff:**
| CLIP_SIGMAS | Post-quant BPB | Artifact |
|---|---|---|
| 10 (tight) | **1.08627** | 17.46MB ❌ |
| 12.85 (default) | 1.09031 | 16.05MB ❌ |
| 15 (loose) | 1.09427 | **15.19MB** ✅ |

**GATED_ATTENTION=1** improves BPB by −0.0015 with minimal extra params (+45K).
**LOOP_EMBEDDINGS=1** — marginal, not worth it.
**LZMA wrapper** — incompatible with Triton @jit (needs source file).

**🎉 BREAKTHROUGH: gated_clip15_mlp425**
Combining GATED_ATTENTION=1 + MATRIX_CLIP_SIGMAS=15 + MLP_MULT=4.25:
- Pre-quant BPB = **1.07589** (best ever, −0.00129 vs previous best)
- Post-quant BPB = 1.09104
- Artifact = **15.75MB** (fits 16MB with 250KB headroom!)
- 37.4M params, 5283 steps

This is our first run with both strong BPB AND artifact compliance.

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

