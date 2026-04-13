# Improved Training Results Summary

**Base:** PR #1394 (SP8192 + GPTQ + Depth Recurrence + MuonEq-R + SDClip), val_bpb 1.0856

## Results Table

| run_id | features | pre_val_bpb | quant_val_bpb | sw_val_bpb | num_steps | tok/s (early→late) | quant_method | final_size |
|--------|----------|-------------|---------------|------------|-----------|-------------------|--------------|------------|
| improved_tier1_2baseline | HS+FM | **1.07992** | 1.09155 | 1.07508 | 5,392 | 1,540K→1,200K | GPTQ+Brotli | 16,074,000 |
| improved_GA_LE_WWM_FUSErope | GA+LE+WW+HS+ANS+FR+FM | 1.08190 | 1.09148 | **1.07492** | 5,444 | 1,534K→1,213K | GPTQ+ANS | 16,042,064 |
| improved_GA_LE_WWM | GA+LE+WW+HS+ANS+FM | 1.08201 | 1.09157 | 1.07499 | 5,369 | 1,512K→1,193K | GPTQ+ANS | 16,041,171 |
| improved_tier3_1-4 | GA+LE+NM+WW+HS+ANS+FR+FM | 1.08290 | 1.09273 | 1.07624 | 5,419 | 1,531K→1,207K | GPTQ+ANS | 16,041,270 |
| improved_GA_LE | GA+LE+NM+WW+FM | 1.08536 | 1.09472 | 1.07823 | 5,298 | 1,475K→1,173K | GPTQ+Brotli | 16,116,510 |
| improved_tier3 | GA+LE+NM+WW+HS+ANS+FM | 1.08857 | 1.09798 | 1.08113 | 4,575 | 1,332K→1,006K | GPTQ+ANS | 16,042,785 |
| improved_GA_FUSErope | GA+HS+ANS+FR+FM | **1.07873** | 1.09031 | **1.07369** | 5,445 | 1,540K→1,212K | GPTQ+ANS | 15,989,467** |
| improved_varlen | GA+LE+NM+WW+VL+FM | 1.90107 | 1.90335 | 1.90094 | 540 | 120K→120K | GPTQ+Brotli | 16,157,081 |

| improved_GA_LE_FUSErope | GA+LE+HS+ANS+FR+FM | ⏳ running | — | — | ~5 (early) | 1,675K→… | GPTQ+ANS | — |

\*\* `improved_GA_FUSErope` originally reported 16,052,040 due to code size bug (measured `_train.py` at 89KB instead of packed `train_gpt.py` at 26KB). Corrected total: model (15,962,852) + code (26,615) = **15,989,467 bytes** ✅ fits 16MB.

## Feature Legend

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

## Key Takeaways

1. **Best sliding BPB:** `improved_GA_FUSErope` (**1.07369**) — GA+HS+ANS+FR+FM, and the only run that fits 16MB after code size fix
2. **Baseline** (HS+FM only) competitive at 1.07508 — 0.0014 BPB behind best
3. **Fused RoPE consistently helps throughput** — ~1% more tok/s → ~75 extra training steps in same wallclock
4. **Gated Attention is a consistent win** — ~-0.001 BPB for minimal parameter cost
5. **NorMuon hurts** — runs with NM consistently worse than without (compare GA_LE vs GA_LE_WWM)
6. **VarLen is broken** — 1.90 BPB, doc boundary detection bug
7. **ANS saves ~30KB** artifact size consistently vs Brotli
8. **improved_tier3** had fewer steps (4,575) due to mixed_seq_len overhead, explaining worse BPB
9. **Code size bug fixed** — `_ORIG_SCRIPT` env var now used in `serialize()`. Saves 63KB, makes `improved_GA_FUSErope` the first run to actually fit 16MB budget
10. **improved_GA_LE_FUSErope** — currently running (GA+LE+FusedRoPE), expected to be best combo

---

## Earlier Runs

### Baseline (SP1024)

| run_id | params | best_bpb | quant_method | final_size |
|--------|--------|----------|--------------|------------|
| baseline_1311 | — | 1.1547 | Int8+Zlib | — |
| baseline_sp1024_algo_opt | 17,048,648 | 1.3255 | Int8+Zlib | — |
| baseline_sp1024_diff | 17,057,864 | 1.3918 | Int8+Zlib | — |
| baseline_sp1024_mla | 18,080,840 | 1.2583 | Int8+Zlib | 55,140,290 |
| baseline_sp1024_mla_256 | 18,818,120 | 1.3329 | Int8+Zlib | 56,614,850 |

### Comparison Runs

| run_id | params | best_bpb | sw_val_bpb | num_steps | tok/s | quant_method | final_size |
|--------|--------|----------|------------|-----------|-------|--------------|------------|
| compare_03_23_s2048 | 32,680,540 | 1.3258 | — | — | — | Int8+Zlib | 129,122,813 |
| compare_03_23_s2048_l4_d768_mpl6 | 36,476,193 | 1.2799 | — | — | — | Int8+Zlib | — |
| compare_04_05 | 35,925,080 | 1.1640 | — | 6,000 | 775K | GPTQ+Brotli | — |
| compare_04_09_mixLiGate | 38,927,000 | 1.1017 | 1.10176 | 2,669 | 592K | GPTQ+Brotli | 17,458,665 |
| compare_04_09_mixLiGate_l9 | 32,618,048 | 1.1057 | 1.10569 | 3,112 | 687K | GPTQ+Brotli | 14,757,163 |

### Architecture Exploration

| run_id | params | best_bpb | quant_method | final_size |
|--------|--------|----------|--------------|------------|
| s4096_l4_d768_mlp6_s8192 | 42,046,753 | 1.1233 | Int8+Zlib | 155,174,118 |
| s2048_l11_d512_mlp6_v8192 | 36,252,249 | 1.1421 | Int8+Zlib | 136,199,099 |
| s4096_l8_s8192 | 27,598,401 | 1.1488 | Int8+Zlib | 101,580,018 |
| s4096_l4_d768_mlp6 | 36,181,280 | 1.1784 | Int8+Zlib | 143,228,945 |
| s4096_l4_d768_mlp6_ema_lzma_pRope_LN_q6_decay_wp | 36,181,280 | 1.1876 | Int8+Zlib | 143,236,405 |
| s4096_l4_d768_mlp6_ema_lzma_pRope_LN_q6_decay_wp_BiTr_ve | 36,541,729 | 1.1985 | Int8+Zlib | 144,161,523 |
| s4096_l4_d768_mlp6_ema_lzma_pRope_LN | 36,181,280 | 1.2155 | Int8+Zlib | 143,233,949 |
| l6_d768_M6_8H100 | 53,879,088 | 1.2189 | Int8+Zlib | 214,015,591 |
| s4096_l18_d256_mlp3 | 13,247,376 | 1.3013 | Int8+Zlib | 52,578,915 |

*Note: Architecture exploration runs used `train_gpt_new.py` with Int8+Zlib (no GPTQ, no Brotli). Final sizes exceed 16MB budget — these were for architecture comparison only.*


### TODO
1. ~~The real optimization: PR #1523's approach — parameter banking where looped layers share a weight bank accessed via scatter/gather, avoiding repeated full-layer forward passes.~~ **Done** — implemented `BankedRecurrence` in `train_gpt_improved2.py` (`BANK_ENABLED=1`).
2. Wait for `improved_GA_LE_FUSErope` to complete — expect best BPB (GA + Loop Embeddings + Fused RoPE).
3. Re-run `improved_GA_FUSErope` with corrected code size to get official 16MB-valid results.
4. Debug VarLen attention (doc boundary detection).
5. Try parameter banking run: `BANK_ENABLED=1 BANK_SIZE=64 BANK_RANK=32`.
