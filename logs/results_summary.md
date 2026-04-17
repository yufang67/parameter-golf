# Improved Training Results Summary

**Base:** PR #1394 (SP8192 + GPTQ + Depth Recurrence + MuonEq-R + SDClip), val_bpb 1.0856

## Results Table

| run_id | features | pre_val_bpb | quant_val_bpb | sw_val_bpb | num_steps | tok/s (early→late) | quant_method | final_size |
|--------|----------|-------------|---------------|------------|-----------|-------------------|--------------|------------|
| improved_GA_FUSErope | GA+HS+ANS+FR+FM | 1.07873 | 1.09031 | **1.07369** | 5,445 | 1,676K→1,212K | GPTQ+ANS | 15,989,467 ✅ |
| improved_GA_LE_FUSErope | GA+LE+HS+ANS+FR+FM | 1.07878 | 1.09028 | 1.07370 | 5,441 | 1,675K→1,212K | GPTQ+ANS | 15,992,326 ✅ |
| improved_GA_FUSErope_MLP435_Mclip15_gptqCl_TTT | GA+HS+ANS+FR+FM+M435+C15 | 1.07611 | 1.09109 | 1.07448 | 4,985 | 1,607K→1,114K | GPTQ+ANS | 15,970,793 ✅ |
| improved_GA_LE_WWM_FUSErope | GA+LE+WW+HS+ANS+FR+FM | 1.08190 | 1.09148 | 1.07492 | 5,444 | 1,652K→1,213K | GPTQ+ANS | 16,042,064 |
| improved_GA_LE_WWM | GA+LE+WW+HS+ANS+FM | 1.08201 | 1.09157 | 1.07499 | 5,369 | 1,512K→1,193K | GPTQ+ANS | 16,041,171 |
| improved_tier1_2baseline | HS+FM | 1.07992 | 1.09155 | 1.07508 | 5,392 | 1,540K→1,200K | GPTQ+Brotli | 16,074,000 |
| improved_tier3_1-4 | GA+LE+NM+WW+HS+ANS+FR+FM | 1.08290 | 1.09273 | 1.07624 | 5,419 | 1,531K→1,207K | GPTQ+ANS | 16,041,270 |
| improved_GA_LE | GA+LE+NM+WW+FM | 1.08536 | 1.09472 | 1.07823 | 5,298 | 1,475K→1,173K | GPTQ+Brotli | 16,116,510 |
| gated_clip13_mlp435 | GA+HS+ANS+M435+C13 | **1.07522** | 1.08670 | **1.07015** | 5,132 | 1,545K→1,130K | GPTQ+ANS | 16,827,143 ⚠️ |
| gated_clip15_mlp435 | GA+HS+ANS+M435+C15 | **1.07518** | 1.09027 | *(no SW eval)* | 5,122 | 1,544K→1,127K | GPTQ+ANS | 15,976,015 ✅ |
| improved_GA_FUSErope_MLP435_Mclip13_TTT | GA+HS+ANS+FR+FM+M435+C13+TTT | 1.07608 | 1.08737 | 1.07077 | 4,974 | 1,601K→1,111K | GPTQ+ANS | 16,827,813 ⚠️ |
| improved_tier3 | GA+LE+NM+WW+HS+ANS+FM | 1.08857 | 1.09798 | 1.08113 | 4,575 | 1,332K→1,006K | GPTQ+ANS | 16,042,785 |
| improved2_GA | GA+BK+HS+ANS+FR+FM | 1.08664 | 1.09900 | 1.08273 | 6,886 | 1,587K→1,509K | GPTQ+ANS | 16,211,197 ⚠️ |
| improved_GA_FUSErope_WinATT | GA+HS+ANS+FR+FM+WA | 1.10673 | 1.12084 | 1.17560 | 5,405 | 1,763K→1,202K | GPTQ+ANS | 15,988,745 ✅ |
| improved_varlen | GA+LE+NM+WW+VL+FM | 1.90107 | 1.90335 | 1.90094 | 540 | 120K→120K | GPTQ+Brotli | 16,157,081 |

### N-gram & VarLen Experiments (4×A100, 3600s wallclock)

| run_id | features | pre_val_bpb | quant_val_bpb | sw_val_bpb | num_steps | tok/s (early→late) | params | final_size |
|--------|----------|-------------|---------------|------------|-----------|-------------------|--------|------------|
| pg01_base_m435_c15_g128 | GA+HS+ANS+FR+FM+M435+C15 | 1.07610 | 1.09104 | 1.07441 | 4,982 | 1,576K→1,113K | 38.2M | 15,978,314 ✅ |
| pg02_bigram512 | GA+HS+ANS+FR+FM+M435+C15+BG512 | 1.07641 | 1.09123 | 1.07464 | 4,948 | 1,601K→1,105K | 38.3M | 16,193,272 ⚠️ |
| pg05_bigram2048 | GA+HS+ANS+FR+FM+M435+C15+BG2048 | 1.07715 | *(incomplete)* | *(incomplete)* | 4,958 | 1,562K→1,107K | 38.5M | *(incomplete)* |
| pg05_bigram2048_dim512 | GA+HS+ANS+FR+FM+M435+C15+BG2048d512 | 1.07675 | 1.09144 | *(no SW)* | 4,952 | 1,444K→1,106K | 39.2M | 16,385,551 ⚠️ |
| pg08_trigram1536d256 | GA+HS+ANS+FR+FM+M435+C15+TG1536d256 | 1.07752 | 1.09232 | 1.07584 | 4,949 | 1,508K→1,105K | 38.7M | 16,181,698 ⚠️ |
| pg09_varlen | GA+HS+ANS+FR+FM+M435+C15+VL | 1.07090 | 1.08517 | 1.07798 | 4,778 | 1,074K→1,050K | 38.2M | 15,975,632 ✅ |
| pg11_varlen_gptq192 | GA+HS+ANS+FR+FM+M435+C15+VL+G192 | 1.07000 | 1.08440 | 1.07722 | 4,883 | 1,103K→1,077K | 38.2M | 15,977,377 ✅ |
| pg12_varlen_clip14 | GA+HS+ANS+FR+FM+M435+C14+VL | **1.06882** | **1.08142** | **1.07425** | 5,077 | 1,129K→1,115K | 38.2M | 16,388,003 ⚠️ |
| pg13_varlen_clip16 | GA+HS+ANS+FR+FM+M435+C16+VL | 1.06874 | 1.08520 | 1.07807 | 5,081 | 1,131K→1,116K | 38.2M | 15,591,834 ✅ |

### MoE Experiments (over 16MB — architectural exploration)

| run_id | features | pre_val_bpb | quant_val_bpb | sw_val_bpb | num_steps | tok/s (early→late) | params | final_size |
|--------|----------|-------------|---------------|------------|-----------|-------------------|--------|------------|
| improved_GA_MoE79_MLP4_TTT | GA+HS+ANS+FR+FM+MoE(7,9)+M4+C13 | **1.07411** | 1.08382 | **1.06740** | 3,713 | 1,103K→820K | 48.6M | 21,186,554 ⚠️ |
| pg14_moe9_e2_k1 | GA+HS+ANS+FR+FM+MoE(9;e2,k1)+M435+C15 | 1.07654 | 1.09122 | 1.07467 | 4,846 | 1,512K→1,076K | 40.4M | 16,885,243 ⚠️ |
| pg15_moe79_e2_k1 | GA+HS+ANS+FR+FM+MoE(7,9;e2,k1)+M435+C15 | 1.07814 | 1.09205 | 1.07551 | 4,554 | 1,400K→1,000K | 42.7M | 17,782,607 ⚠️ |
| improved_GA_MoE_MLP3_TTT | GA+HS+ANS+FR+FM+MoE(3,5,7,9)+M3+C13 | 1.08563 | *(incomplete)* | *(incomplete)* | 3,075 | 1,051K→678K | 49.1M | *(unknown)* |

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
| BK | Parameter Banking | `BANK_ENABLED=1` |
| M435 | MLP multiplier 4.35× | `MLP_MULT=4.35` |
| M4 | MLP multiplier 4.0× | `MLP_MULT=4.0` |
| M3 | MLP multiplier 3.0× | `MLP_MULT=3.0` |
| C13 | Clip sigmas 13.0 | `CLIP_SIGMAS=13.0` |
| C15 | Clip sigmas 15.0 | `CLIP_SIGMAS=15.0` |
| WA | Window Attention | `WINDOW_ATTN_LAYERS=...` |
| MoE | Mixture of Experts | `MOE_LAYERS=...` |
| TTT | Test-Time Training | `TTT_ENABLED=1` |
| BG512 | Bigram embed (V=512, D=128) | `BIGRAM_VOCAB_SIZE=512` |
| BG2048 | Bigram embed (V=2048, D=128) | `BIGRAM_VOCAB_SIZE=2048` |
| BG2048d512 | Bigram embed (V=2048, D=512) | `BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=512` |
| TG1536d256 | Trigram embed (V=1536, D=256) | `TRIGRAM_VOCAB_SIZE=1536 TRIGRAM_DIM=256` |

## Key Takeaways

1. **Best sliding BPB (16MB-valid):** `improved_GA_FUSErope` (**1.07369**) — GA+HS+ANS+FR+FM, 15.99MB ✅
2. **Best absolute BPB (over budget):** `improved_GA_MoE79_MLP4_TTT` (**1.06740** sw) — MoE is powerful but 21.2MB
3. **MLP 4.35× improves pre-quant BPB** — ~1.075 vs ~1.079 for 3.0×, but adds ~2.2M params
4. **Clip sigma 15.0 solves MLP4.35 size problem** — 15.97MB (vs 16.83MB at clip=13) with only 0.004 wider quant gap
5. **`pg01_base_m435_c15_g128` confirms M435+C15 on 4×A100** — 1.07441 sw_bpb, 15.98MB ✅ (matches 8-GPU result)
6. **N-gram embeddings don't help:** Bigram-512 (1.07464) and trigram-1536 (1.07584) are worse than baseline (1.07441), and all bust 16MB
7. **Loop Embeddings don't help:** `improved_GA_LE_FUSErope` (1.07370) essentially tied with `improved_GA_FUSErope` (1.07369)
8. **Gated Attention is a consistent win** — ~-0.001 BPB for minimal parameter cost
9. **NorMuon hurts** — runs with NM consistently worse than without
10. **Window Attention is catastrophic** — 1.176 sw_bpb (vs 1.074 baseline), fast throughput doesn't compensate
11. **Fullparam TTT is broken** — `pg08_trigram1536d256` TTT produced 2.824 bpb (worse than no TTT at 1.076)
12. **MoE achieves best raw BPB but models are 21MB+** — not viable for 16MB budget without dramatic compression
13. **Lightweight MoE still misses the target** — `pg14_moe9_e2_k1` and `pg15_moe79_e2_k1` are both over 16MB and slightly worse than the non-MoE `pg01` baseline on sliding BPB
14. **ANS saves ~30KB** artifact size consistently vs Brotli
15. **Fused kernels provide minimal throughput benefit** — ~1% difference per throughput_check
16. **Parameter Banking hurts BPB and busts size budget** — improved2_GA got 1.08273 sw_bpb, 16.2MB
17. **VarLen is now competitive but clip-sensitive** — `pg12_varlen_clip14` reached 1.07425 sliding, slightly beating dense `pg01` (1.07441), but it busts 16MB; `pg13_varlen_clip16` fits at 15.59MB but falls back to 1.07807 sliding

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

### Incomplete / Smoke Test Runs

| run_id | notes |
|--------|-------|
| improved_test | TTT LoRA smoke test (600s wallclock). ~370 steps, no meaningful eval. TTT with LoRA rank 96 for K/O/MLP projections. |
| throughput_check | Throughput comparison without fused kernels. ~500 steps, 1,615K→1,389K tok/s (vs ~1,600K with fused — minimal difference). |


### TODO
1. ~~The real optimization: PR #1523's approach — parameter banking where looped layers share a weight bank accessed via scatter/gather, avoiding repeated full-layer forward passes.~~ **Done** — implemented `BankedRecurrence` in `train_gpt_improved2.py` (`BANK_ENABLED=1`).
2. ~~Wait for `improved_GA_LE_FUSErope` to complete.~~ **Done** — 1.07370 sw_bpb, essentially identical to GA_FUSErope. Loop Embeddings provide no benefit.
3. ~~Try parameter banking run: `BANK_ENABLED=1 BANK_SIZE=64 BANK_RANK=32`.~~ **Done** — `improved2_GA` got 1.08273 sw_bpb, significantly worse than non-banked runs and over 16MB budget.
4. ~~Re-run `improved_GA_FUSErope` with corrected code size to get official 16MB-valid results.~~ **Done** — fits at 15,989,467 bytes.
5. ~~Debug VarLen attention (doc boundary detection).~~ Still broken (1.90 BPB).
6. ~~Fix code size bug in `train_gpt_improved2.py` (94KB → need `_ORIG_SCRIPT` fix).~~ Fixed via `_ORIG_SCRIPT` env var.
7. ~~Investigate why parameter banking hurts~~ — worse BPB and over budget, abandoned.
8. ~~Try MLP 4.35× with clip=13 and clip=15.~~ **Done** — clip=15 fits 16MB (15.98MB), clip=13 does not (16.83MB).
9. ~~Try MoE.~~ **Done** — best raw BPB (1.067) but 21MB+, not viable.
10. ~~Try Window Attention.~~ **Done** — catastrophic (1.176 sw_bpb), abandoned.
11. ~~Try TTT.~~ **Done** — broken (NaN in all runs).
12. Run sliding window eval on `gated_clip15_mlp435` — best candidate to beat current leader.
13. Debug TTT NaN issue if pursuing test-time training.
14. Explore MoE with aggressive quantization (6-bit?) to fit 16MB budget.

## Auto batch 2026-04-15

- Sequential 10-run sweep from program.md using packed `train_gpt_improved.py` with non-TTT baseline `MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=128`.
- `pg01_base_m435_c15_g128` | status=ok | pre=1.07609774 | quant=1.09104034 | sw=1.07440969 | steps=4982 | log=logs/pg01_base_m435_c15_g128.txt
- `pg02_bigram512` | status=ok | pre=1.07641493 | quant=1.09122775 | sw=1.07464374 | steps=4948 | log=logs/pg02_bigram512.txt
- `pg03_bigram1024` | status=exit=1 | pre=1.07646698 | steps=4949 | log=logs/pg03_bigram1024.txt
- `pg04_bigram1536` | status=exit=1 | pre=1.07673012 | steps=4947 | log=logs/pg04_bigram1536.txt
- `pg05_bigram2048` | status=exit=1 | pre=1.07760320 | steps=4952 | log=logs/pg05_bigram2048.txt
- `pg06_trigram512` | status=ok | pre=1.07648794 | quant=1.09123454 | sw=1.07462638 | steps=4949 | log=logs/pg06_trigram512.txt
- `pg07_trigram1024` | status=exit=1 | pre=1.07641117 | steps=4947 | log=logs/pg07_trigram1024.txt
- `pg08_trigram1536` | status=exit=1 | pre=1.07651455 | steps=4949 | log=logs/pg08_trigram1536.txt
- `pg09_varlen` | status=exit=1 | log=logs/pg09_varlen.txt
- `pg10_moe79` | status=exit=1 | log=logs/pg10_moe79.txt
