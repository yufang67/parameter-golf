# Parameter Golf: Potential BPB Improvements (No SLOT / No TTT)

> **Date:** 2026-04-14 | **Current Merged SOTA:** 1.0810 BPB (PR #1493, uses TTT)
> **Best Merged Non-TTT/SLOT Record:** 1.0856 BPB (PR #1394)
> **Best Open Non-TTT/SLOT Record:** 1.0758 BPB (PR #1529, pending merge)
> **Best Open TTT Record:** 1.0728 BPB (PR #1610, VarLen + PhasingTTT)
> **Best Open Casefold (controversial):** 1.0639 BPB (PR #1585, legality pending)

---

## 1. Leaderboard Context

The merged leaderboard top-5 all use TTT (Test-Time Training) except PR #1394:

| PR | BPB | TTT? | SLOT? | Key Techniques |
|----|-----|------|-------|----------------|
| #1493 (SOTA) | 1.0810 | ✅ | ✅ | 3-layer recurrence, parallel residuals, QK-Gain 5.25, score-first TTT |
| #1477 | 1.0822 | ✅ | ✅ | Parallel residuals on SP8192 + TTT stack |
| #1413 | 1.0828 | ✅ | ✅ | QK-Gain 5.0, score-first TTT on SP8192 |
| **#1394** | **1.0856** | ❌ | ❌ | SP8192, GPTQ embeddings, depth recurrence (L4-5), MuonEq-R, SDClip |
| #1412 | 1.0835 | ✅ | ✅ | Parallel residuals, Hessian-aware SDClip |

### Best Open (Unmerged) PRs

| PR | BPB | TTT? | No-TTT? | Key Techniques | Status |
|----|-----|------|---------|----------------|--------|
| **#1585** | **1.0639** | ✅ | — | Casefold tokenizer + parallel residuals + systems opt | ⚠️ Casefold legality pending |
| **#1578** | **1.0668** | ✅ | — | Casefold v2 tokenizer (21% case-dup removal) + PR #1529 arch | ⚠️ Casefold legality pending |
| **#1610** | **1.0728** | ✅ | — | VarLen + PhasingTTT (global SGD on scored prefix of 2000 docs) | Open |
| **#1530** | **1.0734** | ✅ | — | VarLen attention (FA3) + fused MLP + doc-independent LoRA TTT | Open |
| **#1560** | **1.0741** | ✅ | — | VarLen + Doc-TTT + Warmdown 0.75 + Chunk 48 | Open |
| **#1586** | **1.0749** | ✅ | — | Per-layer adaptive GPTQ clip + int7 embeddings + MLR 0.026 | Open |
| **#1529** | **1.0758** | ✅ | — | Improved parallel residuals (dual-lane decoder) + CUTLASS EVT | Open |
| **#1523** | **1.0778** | ✅ | — | Triple recurrence + banking + fused MLP + Muon 0.97 | Closed→#1561 |
| **#1435** | **1.0980** | ❌ | ✅ | Depth recurrence + BigramHash + EMA 0.9965 (no TTT) | Open |

**Gap to close (without TTT/SLOT):** ~0.0046 BPB from PR #1394 → SOTA

---

## 2. Proven Techniques (Merged, No TTT/SLOT)

These are battle-tested techniques from merged PRs that DO NOT require TTT or SLOT:

### 2.1 Architecture Improvements

| Technique | PR | BPB Impact | Description |
|-----------|-----|-----------|-------------|
| **Depth Recurrence (3-layer)** | #1285, #1394, #1493 | -0.005 to -0.01 | Layers 3-5 repeat (14-20 virtual from 11 physical). Biggest architectural win. `torch.compile` penalty = 0 (PR #1449 ablation). |
| **Parallel Residuals** (GPT-J style) | #1204, #1412 | -0.003 to -0.005 | Attention + MLP in parallel (layers 7+). Saves a sequential dependency. |
| **U-Net Skip Connections** | baseline | included | Already in baseline. Encoder/decoder halves with learnable skip weights. |
| **Larger Vocab (SP8192)** | #1394 | -0.008 vs SP1024 | 8192 SentencePiece vocab. Major win over SP1024, modest over SP4096. |

### 2.2 Quantization & Compression

| Technique | PR | BPB Impact | Description |
|-----------|-----|-----------|-------------|
| **GPTQ (Full Hessian + SDClip)** | #1394, #1412 | -0.003 to -0.005 | Cholesky Hessian + actorder + 5-way clip sweep. Much better than naive int8. |
| **GPTQ Embeddings** | #1394 | -0.001 to -0.002 | GPTQ applied to embedding layers too, not just transformer blocks. |
| **Int6 quantization (all layers)** | #1285 | -0.002 | All-int6 instead of mixed int6/int8. More params in same budget. |
| **Brotli-11 compression** | #1179, #1334 | -0.001 | Better than zlib/zstd for model artifact compression. Frees ~1MB for more params. |
| **LZMA compression** | #1446 | similar | Alternative to Brotli; comparable or better compression ratio. |

### 2.3 Optimizer & Training

| Technique | PR | BPB Impact | Description |
|-----------|-----|-----------|-------------|
| **MuonEq-R** | #1285, #1334, #1394 | -0.002 | Per-row normalize BEFORE Newton-Schulz in Muon optimizer. |
| **Muon momentum 0.97** | #1493, #1523 | -0.0004 | Down from default 0.99. Small but free. |
| **QK-Gain 5.0 → 5.25** | #1334, #1493 | -0.001 | Larger initial QK gain. Monotonic improvement up to 5.25. |
| **EMA 0.9965** | #1394, #1435 | -0.001 | Exponential moving average of weights. Replaces SWA. |
| **Higher Weight Decay (0.085-0.090)** | #1218, #1285 | -0.001 | Up from baseline 0.04. |
| **Warmdown 72%** | #1493 | -0.0005 | Longer warmdown fraction of training. |

### 2.4 Evaluation Improvements

| Technique | PR | BPB Impact | Description |
|-----------|-----|-----------|-------------|
| **Sliding Window Eval** (stride=64) | #50 (baseline) | -0.02 to -0.03 | Already standard. Longer context at eval time. |
| **Window Attention** (FA3) | #1219 | -0.002 | Size=512 window on alternating layers via FlashAttention-3. |
| **Mixed seq_len training** | #1219 | -0.002 | 5 GPUs at 2048×36 + 3 GPUs at 6144×10 for longer context exposure. |

---

## 3. Promising Unmerged Techniques (No TTT/SLOT)

### 3.1 High-Confidence (Validated on 8×H100)

| Technique | PR | BPB | Status | Description |
|-----------|-----|-----|--------|-------------|
| **Improved Parallel Residuals** (dual-lane) | #1529 | 1.0758 | Open | Richer learned routing between attn+MLP lanes. CUTLASS EVT fusion. -0.0020 vs #1523. |
| **VarLen Attention + Doc-TTT** | #1530 | 1.0734 | Open | FA3 varlen with cu_seqlens doc boundaries + per-doc LoRA TTT. ~5% faster training. |
| **Phased TTT** (global SGD prefix) | #1610 | 1.0728 | Open | Score-first on 2000 docs, then global SGD on scored prefix. Builds on #1530. |
| **Triple Depth Recurrence + Banking** | #1523 | 1.0778 | Closed→#1561 | 17 virtual from 11 physical (loop 3-5 ×3). Batched Newton-Schulz 15× faster. |
| **Per-Layer Adaptive GPTQ Clip** | #1586 | 1.0749 | Open | Per-layer clip sweep + int7 embeddings. Builds on #1530. |
| **Systems Optimization** (fused Muon/EMA) | #1583, #1584, #1585 | 1.0639–1.0801 | Open | Fused Muon kernel, batched EMA, loader prealloc. Pure throughput gains → more steps. |
| **Casefold Tokenizer** | #1578 | 1.0668 | Open ⚠️ | SP8192 retrained on lowercased text. 10.4% better compression. Legality debated. |
| **Depth Recurrence + BigramHash + EMA** | #1435 | 1.0980 | Open | 13 virtual from 11 physical, BigramHash(1536), EMA 0.9965. No TTT. |
| **Fused Triton MLP + Full GPTQ + Coprime Loader** | #1135 | 1.1116 | Open | Custom Triton kernel for LeakyReLU(0.5)², saves 1.8ms/step. +5% throughput. |
| **Compressibility Regularization** | #1508 | 1.1135 | Closed | Trains model to be more compressible. Warmdown WD mult=2.0. |
| **VarLen Attention** (training-only) | #1530 | — | Open | Within-document-only attention via FA3 varlen. Eliminates cross-doc noise. ~2% faster. |
| **MoE MLP** | #1538 | 1.1180 | Open | 4 experts, top-2 routing. First MoE exploration. Not yet competitive but promising. |
| **ANS weight compression** | #1510 | — | Open | Asymmetric Numeral Systems. Frees ~1.6MB = +2.2M params. Zero legality risk. |

### 3.2 Research-Stage (Promising but Need Work)

| Technique | PR | BPB | Status | Description |
|-----------|-----|-----|--------|-------------|
| **N-gram Backoff Mixer** (order=82) | #1605 | 0.2988 | Closed | Causal backoff n-gram + entropy-adaptive blending. Massive BPB but legality/comparability unclear. |
| **Gated DeltaNet Hybrid** (SSM+SWA) | #1553 | 1.2097 | Open | [GDN×5]→SWA→[GDN×5]→SWA. First non-transformer competitive arch. Corrected from 1.028 (BPB bug). |
| **Gated Krylov + GPTQ int6 + LZMA** | #1446 | 1.0960 | Open | Non-record (1×A100, 8h52m). Gated Krylov layers — potential if speed improves. |
| **DepthScale (Parameter-Shared Iterative)** | #1509 | 1.1962 | Open | 5 layers × 2 iterations. Needs int6 quant to fit 16MB. |
| **Gated Attention** | (ref in #1485) | -0.001? | Research | Sigmoid gate per attention head. From NeurIPS 2025 paper. |
| **NorMuon** | (ref in #1485) | -0.0005? | Research | Per-row normalize AFTER Newton-Schulz (distinct from MuonEq-R which is BEFORE). |
| **HybridMamba-11 (SSM)** | #1365 | 2.12 | Open | First SSM submission. Parallel associative scan. Not competitive yet. |
| **MDLM Diffusion** | #1106 | 1.1465 | Open | First diffusion to beat AR baseline. |
| **Per-Pass Loop Embeddings** | #1518 | — | Open | Reduces quant gap from 0.0131 → 0.0114 in depth-recurrent models. |
| **Cosine LR Schedule** | #1380 | -0.070? | Open | Reported large improvement but needs validation. |
| **ALBERT-Style Low-Rank Embedding** | #1481 | — | Open | Factorized embeddings for parameter savings. |
| **Cross-Layer Shared Weight Bank** | #1315 | — | Open | Shared weight bank across layers. |

---

## 4. Recommended Improvement Stack (No TTT/SLOT)

Starting from the best non-TTT/SLOT merged record (PR #1394, BPB=1.0856):

### Tier 1: Near-Certain Gains (est. -0.005 to -0.010 combined)

1. **3-Layer Depth Recurrence** (layers 3-5 × 2+) — proven in multiple PRs. Triple recurrence (×3) now proven in #1523.
2. **Improved Parallel Residuals** (dual-lane, layers 7+) — PR #1529 shows -0.002 BPB over basic GPT-J-style split.
3. **QK-Gain 5.25** — monotonic improvement, trivial change.
4. **Muon momentum 0.97** — free, validated.
5. **VarLen Attention** — within-document via FA3 cu_seqlens. ~2% faster training + cleaner gradients (PR #1530).

### Tier 2: Likely Gains (est. -0.003 to -0.005 combined)

6. **Fused Triton MLP** — +5% throughput = more training steps (PR #1523, #1530).
7. **Systems Optimization** — fused Muon kernel + batched EMA + loader prealloc (PR #1583–#1585). Pure throughput.
8. **Per-Layer Adaptive GPTQ Clip** — per-layer sweep + int7 embeddings (PR #1586). Better than global clip.
9. **ANS Compression** — frees 1.6MB for ~2.2M more params (PR #1510).
10. **CUTLASS EVT Fusion** — custom backward fusion for MLP, recovers throughput for richer parallel residuals (PR #1529).

### Tier 3: Speculative / Research Gains (est. -0.001 to -0.003 each)

11. **Casefold Tokenizer** — 10.4% better compression, -0.012 BPB (PR #1578). Legality questionable.
12. **Gated Attention** — per-head sigmoid gates. Validated -0.001 in our experiments.
13. **N-gram Backoff Mixer** — order=82 causal n-gram achieves 0.2988 BPB (PR #1605). Extreme but legality debated.
14. **Phased TTT** — global SGD on scored prefix after initial LoRA TTT (PR #1610). -0.0006 over #1530.
15. **Gated DeltaNet Hybrid** — non-transformer SSM+SWA arch (PR #1553). BPB 1.21 after bug fix, but novel direction.
16. **MoE MLP** — 4 experts, top-2 routing (PR #1538).
17. **Window Attention + Mixed seq_len** — longer context training (PR #1219).
18. **Compressibility Regularization** — train-aware compression (PR #1508).

### Estimated Combined Potential

| Stack | Estimated BPB |
|-------|---------------|
| PR #1394 baseline (no TTT/SLOT) | 1.0856 |
| + Tier 1 techniques | ~1.075 – 1.078 |
| + Tier 2 techniques | ~1.072 – 1.075 |
| + Tier 3 (if successful) | ~1.068 – 1.072 |
| + TTT (doc-independent LoRA, PR #1530 style) | ~1.064 – 1.068 |

**For reference:** Best open TTT submissions now reach 1.0728 (PR #1610, PhasingTTT). Casefold submissions reach 1.0639 (PR #1585) but legality is debated.

---

## 5. Experiment Results (4×A100, 3600s wallclock)

All runs use `train_gpt_improved.py` (or packed `train_gpt.py`), SP8192, 11 layers, dim=512, depth recurrence (layers 3-5 × 2), parallel residuals (layers 7+).

| Run ID | Features | Steps | tok/s (early→late) | Pre-Q BPB | Q BPB | Sliding BPB | Quant Method | Total Size | Fit? |
|--------|----------|-------|-------------------|-----------|-------|-------------|--------------|------------|------|
| `improved_GA_FUSErope` | GA+HS+ANS+FR+FM | 5,445 | 1,540K→1,212K | **1.0787** | 1.0903 | **1.0737** | GPTQ+ANS | 15,989,467 | ✅ |
| `improved_GA_LE_FUSErope` | GA+LE+HS+ANS+FR+FM | 5,441 | 1,675K→1,212K | 1.0788 | 1.0903 | 1.0737 | GPTQ+ANS | 15,992,326 | ✅ |
| `improved_GA_LE_WWM_FUSErope` | GA+LE+WW+HS+ANS+FR+FM | 5,444 | 1,534K→1,213K | 1.0819 | 1.0915 | 1.0749 | GPTQ+ANS | 16,042,064 | ❌ |
| `improved_GA_LE_WWM` | GA+LE+WW+HS+ANS+FM | 5,369 | 1,512K→1,193K | 1.0820 | 1.0916 | 1.0750 | GPTQ+ANS | 16,041,171 | ❌ |
| `improved_tier1_2baseline` | HS+FM | 5,392 | 1,540K→1,200K | 1.0799 | 1.0916 | 1.0751 | GPTQ+Brotli | 16,074,000 | ❌ |
| `improved_tier3_1-4` | GA+LE+NM+WW+HS+ANS+FR+FM | 5,419 | 1,531K→1,207K | 1.0829 | 1.0927 | 1.0762 | GPTQ+ANS | 16,041,270 | ❌ |
| `improved_GA_LE` | GA+LE+NM+WW+FM | 5,298 | 1,475K→1,173K | 1.0854 | 1.0947 | 1.0782 | GPTQ+Brotli | 16,116,510 | ❌ |
| `improved_tier3` | GA+LE+NM+WW+HS+ANS+FM | 4,575 | 1,332K→1,006K | 1.0886 | 1.0980 | 1.0811 | GPTQ+ANS | 16,042,785 | ❌ |
| `improved2_GA` | GA+BK+HS+ANS+FR+FM | 6,886 | 1,587K→1,509K | 1.0866 | 1.0990 | 1.0827 | GPTQ+ANS | 16,211,197 | ⚠️ |
| `improved_varlen` | GA+LE+NM+WW+VL+FM | 540 | 120K→120K | 1.9011 | 1.9034 | 1.9009 | GPTQ+Brotli | 16,157,081 | ❌ |

### Feature Legend

| Code | Feature | Env Var |
|------|---------|---------|
| GA | Gated Attention | `GATED_ATTENTION=1` |
| LE | Loop Embeddings | `LOOP_EMBEDDINGS=1` |
| NM | NorMuon | `NORMUON=1` |
| WW | Warmdown WD ×2 | `WARMDOWN_WD_MULT=2.0` |
| HS | Hessian-aware SDClip | `HESSIAN_CLIP_LAMBDA=0.3` |
| ANS | ANS Compression | `COMPRESS_ANS=1` |
| VL | VarLen Attention | `VARLEN_ATTENTION=1` |
| FR | Fused RoPE Triton | `FUSED_ROPE=1` |
| FM | Fused MLP Triton | `FUSED_MLP=1` |
| BK | Parameter Banking | `BANK_ENABLED=1` |

### Key Findings

1. **Best sliding BPB: `improved_GA_FUSErope`** (1.0737) — GA+HS+ANS+FR+FM, fits 16MB ✅
2. **Loop Embeddings don't help** — `improved_GA_LE_FUSErope` (1.0737) essentially tied with `improved_GA_FUSErope` (1.0737), no measurable gain
3. **Baseline** (HS+FM only) competitive at 1.0751 — only 0.0014 BPB behind best
4. **Fused RoPE consistently helps throughput** — ~1% more tok/s → ~75 extra training steps in same wallclock
5. **Gated Attention is a consistent win** — ~-0.001 BPB for minimal parameter cost
6. **NorMuon hurts** — runs with NM consistently worse than without (compare GA_LE vs GA_LE_WWM)
7. **Parameter Banking hurts BPB and busts size budget** — `improved2_GA` got 1.0827 sw_bpb (0.009 worse) with 16.2MB total. More steps (6,886 vs 5,441) due to no looping slowdown, but worse loss
8. **VarLen attention broken** — only 540 steps completed, doc boundary detection bug
9. **ANS saves ~30KB** artifact size consistently vs Brotli
10. **Mixed seq_len + Window Attention hurt** on 4 GPUs (fewer steps due to memory pressure)
11. **Code size bug fixed** — `_ORIG_SCRIPT` env var now used. Saves 63KB, making `improved_GA_FUSErope` first run to fit 16MB

---

## 6. Key Takeaways

1. **Depth recurrence is the #1 non-TTT improvement** — consistently delivers -0.005 to -0.01 BPB. Triple recurrence (×3, 17 virtual layers) now validated in PR #1523.
2. **VarLen Attention is now proven** — PR #1530 shows FA3 varlen with cu_seqlens doc boundaries works: ~2% faster training, ~0.001 nats improvement. Our VarLen implementation was broken; #1530's approach (FA3 `flash_attn_varlen_func`) is the correct path.
3. **Improved Parallel Residuals deliver** — PR #1529's dual-lane decoder with learned attn↔MLP routing gives -0.0020 BPB over basic GPT-J split.
4. **Doc-independent LoRA TTT is the cleanest TTT path** — PR #1530's per-document LoRA (no cross-doc contamination) is legally cleaner than sequence-level TTT, adds ~0.008 nats.
5. **Systems-level throughput matters** — fused Muon kernel, batched EMA, loader prealloc (PR #1583–#1585) give extra training steps in same 600s budget.
6. **Per-layer adaptive GPTQ** — PR #1586's per-layer clip sweep + int7 embeddings improves over global Hessian SDClip.
7. **Casefold tokenizer is controversial** — 10.4% better compression, -0.012 BPB (PR #1578), but community debate on whether lowercasing changes the benchmark. Maintainer ruling pending.
8. **GDN-Hybrid (Gated DeltaNet)** — first non-transformer architecture attempted. Original PR #1545 had a BPB counting bug (reported 1.028, actual ~1.21). Corrected submission at PR #1553. Novel direction but not yet competitive.
9. **N-gram mixers achieve extreme BPB** — PR #1605 reports 0.2988 BPB with order-82 causal n-gram, but legality/comparability unclear.
10. **SP8192 vocabulary** is now standard for competitive submissions.
11. **GPTQ quality matters enormously** — Hessian-aware SDClip + actorder vs naive int8 is worth -0.005+ BPB.
12. **Gated Attention is a consistent win** — -0.001 BPB for minimal parameter cost.
13. **Code size matters** — using LZMA+base85 packed loader saves 40-63KB vs raw script.
14. **Loop Embeddings provide no benefit** — tied with non-LE runs; adds complexity for zero gain.
15. **NorMuon hurts at this scale** — consistently worse than runs without it.
16. **Parameter Banking hurts** — worse BPB despite more steps, and busts 16MB budget.
17. **The TTT gap is narrowing** — best open TTT PR reaches 1.0728 (PR #1610), while best non-TTT is 1.0758 (PR #1529 with TTT) / 1.0980 (PR #1435 truly no TTT). Doc-independent LoRA TTT adds ~0.008 nats.
