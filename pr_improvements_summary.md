# Parameter Golf: Potential BPB Improvements (No SLOT / No TTT)

> **Date:** 2026-04-11 | **Current Merged SOTA:** 1.0810 BPB (PR #1493, uses TTT)
> **Best Merged Non-TTT/SLOT Record:** 1.0856 BPB (PR #1394)
> **Best Open Non-TTT/SLOT Record:** 1.0980 BPB (PR #1435, pending merge)

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
| **Depth Recurrence + BigramHash + EMA** | #1435 | 1.0980 | Open | 13 virtual from 11 physical, BigramHash(1536), EMA 0.9965. No TTT. |
| **Fused Triton MLP + Full GPTQ + Coprime Loader** | #1135 | 1.1116 | Open | Custom Triton kernel for LeakyReLU(0.5)², saves 1.8ms/step. +5% throughput. |
| **Compressibility Regularization** | #1508 | 1.1135 | Closed | Trains model to be more compressible. Warmdown WD mult=2.0. |
| **Fused Triton TMA MLP** | #1523 | — | Open | +5% throughput via TMA (Tensor Memory Accelerator). More training steps in same time. |
| **VarLen Attention** | #1530, #1536 | — | Open | Within-document-only attention. Eliminates cross-doc noise. |
| **MoE MLP** | #1538 | 1.1180 | Open | 4 experts, top-2 routing. First MoE exploration. Not yet competitive but promising. |
| **ANS weight compression** | #1510 | — | Open | Asymmetric Numeral Systems. Frees ~1.6MB = +2.2M params. Zero legality risk. |

### 3.2 Research-Stage (Promising but Need Work)

| Technique | PR | BPB | Status | Description |
|-----------|-----|-----|--------|-------------|
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

1. **3-Layer Depth Recurrence** (layers 3-5 × 2) — proven in multiple PRs
2. **Parallel Residuals** (layers 7+) — proven in PR #1412, #1204
3. **QK-Gain 5.25** — monotonic improvement, trivial change
4. **Muon momentum 0.97** — free, validated

### Tier 2: Likely Gains (est. -0.003 to -0.005 combined)

5. **Fused Triton MLP** — +5% throughput = more training steps (PR #1523)
6. **VarLen Attention** — within-document only, removes cross-doc noise (PR #1530)
7. **ANS Compression** — frees 1.6MB for ~2.2M more params (PR #1510)
8. **Better GPTQ** — Hessian-aware SDClip + full actorder (PR #1412)

### Tier 3: Speculative / Research Gains (est. -0.001 to -0.003 each)

9. **MoE MLP** — 4 experts, top-2 routing (PR #1538)
10. **Gated Attention** — per-head sigmoid gates (referenced in PR #1485)
11. **NorMuon** — post-NS normalization in Muon optimizer
12. **Per-Pass Loop Embeddings** — better depth recurrence (PR #1518)
13. **Window Attention + Mixed seq_len** — longer context training (PR #1219)
14. **Compressibility Regularization** — train-aware compression (PR #1508)

### Estimated Combined Potential

| Stack | Estimated BPB |
|-------|---------------|
| PR #1394 baseline (no TTT/SLOT) | 1.0856 |
| + Tier 1 techniques | ~1.078 – 1.080 |
| + Tier 2 techniques | ~1.074 – 1.077 |
| + Tier 3 (if successful) | ~1.070 – 1.074 |

**For reference:** Best TTT submissions reach ~1.058–1.068 (PR #1539, #1485), showing TTT adds roughly -0.010 to -0.020 BPB on top of architectural improvements.

---

## 5. Key Takeaways

1. **Depth recurrence is the #1 non-TTT improvement** — consistently delivers -0.005 to -0.01 BPB across multiple independent PRs.
2. **SP8192 vocabulary** is now standard for competitive submissions.
3. **GPTQ quality matters enormously** — Hessian-aware SDClip + actorder vs naive int8 is worth -0.005+ BPB.
4. **Throughput optimizations translate directly to BPB** — Fused Triton MLP (+5%), ANS compression (+2.2M params) give more training steps/params in the same budget.
5. **VarLen attention is underexplored** — within-document-only attention is a clean architectural win that doesn't add eval-time compute.
6. **MoE is barely explored** — PR #1538 is the first attempt. Potential for significant gains with proper tuning.
7. **The TTT gap (~0.010–0.020 BPB) may be partially closable** through VarLen attention + better depth recurrence + throughput gains, but TTT remains the single biggest lever available.
