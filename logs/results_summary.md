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

### `program.md` Architecture Sweeps (4×A100, 3600s wallclock, full train+eval)

Built on the `pg12_varlen_clip14` baseline (VarLen+M435+C14, fixed RoPE). Each run reports
pre-quant val_bpb / sliding-window val_bpb / TTT-LoRA val_bpb.

| run_id | feature change | pre_val_bpb | sw_val_bpb | ttt_val_bpb | final_size |
|--------|---------------|-------------|------------|-------------|------------|
| pgm_xsa9 | xsa_last_n=9 (sliding-attn last 9 layers) | 1.08077 | 1.07307 | **1.07102** 🏆 | 16,523,099 ⚠️ |
| pgm_loopemb1 | LOOP_EMBEDDINGS=1, loops=2 | 1.08147 | 1.07384 | 1.07169 | 16,526,656 ⚠️ |
| pgm_xsa7 | xsa_last_n=7 | 1.08184 | 1.07419 | 1.07208 | 16,524,957 ⚠️ |
| pgm_xsa4 | xsa_last_n=4 | 1.08410 | 1.07648 | 1.07430 | 16,525,432 ⚠️ |
| pgm_xsa0 | xsa_last_n=0 (control, no XSA) | 1.08419 | 1.07659 | 1.07442 | 16,524,433 ⚠️ |
| pgm_loopemb1_loops3 | LOOP_EMBEDDINGS=1, loops=3 | 1.08401 | 1.07631 | 1.07419 | 16,533,337 ⚠️ |
| pgm_loopemb1_loops4 | LOOP_EMBEDDINGS=1, loops=4 | *(incomplete)* | — | — | — |

#### `ENABLE_LOOPING_AT` Sweep (LOOP_EMBEDDINGS=1, loops=2)

All on the `pg12_varlen_clip14` baseline with `LOOP_EMBEDDINGS=1`. Varies when looping activates
during training. Later activation → more steps (cheaper non-looped forward) but less time with
17-step depth.

| run_id | enable_looping_at | pre_val_bpb | sw_val_bpb | ttt_val_bpb | steps | final_size |
|--------|-------------------|-------------|------------|-------------|-------|------------|
| pgm_loopat0_5 | 0.5 | **1.06630** | **1.07129** | **1.06919** 🏆 | 5,716 | 16,525,791 ⚠️ |
| pgm_loopat0_5_repro | 0.5 (rerun 2026-04-22) | 1.06837 | 1.07323 | 1.07117 | 5,303 | — |
| pgm_loopat0_15 | 0.15 | 1.06881 | 1.07392 | 1.07180 | 4,647 | 16,528,431 ⚠️ |
| pgm_loopat0_7 | 0.7 | 1.06893 | 1.07368 | 1.07159 | 6,190 | 16,525,093 ⚠️ |
| pgm_loopat0_0 | 0.0 (no curriculum) | 1.07246 | 1.07756 | 1.07537 | 4,353 | 16,530,286 ⚠️ |

**Reproduction note (2026-04-22):** `pgm_loopat0_5_repro` re-ran with identical
hyperparameters (verified by diffing the `Hyperparameters` printout). The recipe
is config-equivalent but landed at **1.07117** TTT vs the original **1.06919**
(+0.00198). Two findings:
1. **Hidden config drift.** `HESSIAN_CLIP_LAMBDA` default in
   `train_gpt_improved_04_16.py` had silently changed from `0.3 → 0.0` since
   the original run. `run_program_md_section.sh` did not pin it, so a naïve
   re-run today picks up the new default. Fixed by adding
   `HESSIAN_CLIP_LAMBDA=0.3` to `COMMON_TRAIN_ENV` in
   `run_program_md_section.sh`. The repro above already includes this fix.
2. **Wallclock-bounded ≠ bitwise-reproducible.** Repro reached only **5,303**
   steps in 3600s vs **5,716** originally (−7.2% throughput on this node).
   The pre-quant Δ (+0.00207) propagates straight through quant (+0.00194 sw)
   and TTT (+0.00198), exactly matching the step shortfall. To reproduce
   bitwise across nodes, switch to step-bounded training.

**Takeaways:** `enable_looping_at=0.5` is the clear winner across all metrics (best pre-quant
1.06630, best sw 1.07129, best ttt 1.06919). Immediate looping (0.0) is worst — the curriculum
matters. The sweet spot is mid-training activation; too early (0.0/0.15) or too late (0.7)
underperforms 0.5. **`pgm_loopat0_5` holds the best absolute TTT BPB (1.06919)** across all
runs, but is ~0.5MB over 16MB.

All `pgm_*` runs are ~0.5MB over the 16MB budget — the baseline branch needs further compression
to be leaderboard-eligible. **XSA on the last 9 layers gives the largest single-feature gain
seen so far** (~0.0035 sw_bpb improvement, ~0.0034 ttt_bpb improvement vs xsa0 control).
**`enable_looping_at=0.5` with loop embeddings is the new best absolute BPB** (1.06919 ttt).

#### `CLIP_MULT_*` Per-Group GPTQ Sweep (built on `pgm_loopat0_5` recipe)

All on the `pgm_loopat0_5` recipe (`MATRIX_CLIP_SIGMAS=14`, `LOOP_EMBEDDINGS=1`,
`ENABLE_LOOPING_AT=0.5`, `HESSIAN_CLIP_LAMBDA=0.3`, varlen + TTT, 3600s wallclock,
4× A100). Each run flips one of `CLIP_MULT_{EARLY,LOOP,MID,LATE}` to 0.75 or 1.25
(others stay at 1.0). The 1.5 leg was skipped to halve cost; 0.75 vs 1.25 is the
informative pair.

| run_id | group | mult | sw_val_bpb | ttt_val_bpb | Δ vs baseline (1.06919) |
|--------|-------|------|------------|-------------|-------------------------|
| **pgm_cliplate075** | LATE | 0.75 | **1.06955** | **1.06756** 🏆 | **−0.00163** |
| pgm_cliploop075 | LOOP | 0.75 | 1.06969 | 1.06770 | −0.00149 |
| pgm_clipearly075 | EARLY | 0.75 | 1.07049 | 1.06839 | −0.00080 |
| pgm_clipmid075 | MID | 0.75 | 1.07051 | 1.06844 | −0.00075 |
| pgm_clipearly125 | EARLY | 1.25 | 1.07238 | 1.07025 | +0.00106 |
| pgm_clipmid125 | MID | 1.25 | 1.07296 | 1.07082 | +0.00163 |
| pgm_cliploop125 | LOOP | 1.25 | 1.07353 | 1.07133 | +0.00214 |
| pgm_cliplate125 | LATE | 1.25 | 1.07382 | 1.07163 | +0.00244 |

**Takeaways:** every group prefers **tighter** clipping (0.75); every 1.25 regresses.
Sensitivity ranking by (1.25 − 0.75) Δttt: **LATE 0.00407 > LOOP 0.00362 > MID 0.00238
> EARLY 0.00186**. The original "LOOP needs larger `cs`" hypothesis is **inverted**.
**New best absolute TTT BPB: 1.06756** (`pgm_cliplate075`), beats prior record 1.06919
by 0.00163. Natural follow-ups: stack `LATE=0.75 + LOOP=0.75` and probe sub-0.75
(0.5 / 0.6) on LATE/LOOP since both still want tighter at the swept boundary.

#### `CLIP_MULT_*` Stacking + Sub-0.75 Sweep (section 7, 2026-04-23)

Same recipe; tests stacking and sub-0.75 magnitudes for the two strongest groups
(LATE, LOOP) plus a uniform all-0.75 reference.

| run_id | config | sw_val_bpb | ttt_val_bpb | Δ vs prior best (1.06756) |
|--------|--------|------------|-------------|---------------------------|
| **pgm_clip_lateloop05** | LATE=0.5 + LOOP=0.5 | **1.06559** | **1.06372** 🏆 | **−0.00385** |
| pgm_clip_all075 | all 4 groups = 0.75 | 1.06614 | 1.06426 | −0.00330 |
| pgm_clip_lateloop075 | LATE=0.75 + LOOP=0.75 | 1.06814 | 1.06626 | −0.00131 |
| pgm_clip_late05 | LATE=0.5 | 1.06848 | 1.06650 | −0.00106 |
| pgm_clip_late06 | LATE=0.6 | 1.06859 | 1.06660 | −0.00097 |
| pgm_clip_loop05 | LOOP=0.5 | 1.06857 | 1.06664 | −0.00092 |
| pgm_clip_loop06 | LOOP=0.6 | 1.06905 | 1.06708 | −0.00048 |

**Takeaways:** stacking is **super-additive** — LATE=0.5 alone (−0.00106) + LOOP=0.5
alone (−0.00092) → combined **−0.00385** (~2× the linear sum). Sub-0.75 keeps winning:
0.5 > 0.6 > 0.75 ordering on both LATE and LOOP. **New absolute TTT record: 1.06372**
(`pgm_clip_lateloop05`), beats `pgm_loopat0_5` (1.06919) by **0.00547**. **Defaults
updated:** `CLIP_MULT_LATE 1.0→0.5`, `CLIP_MULT_LOOP 1.0→0.5` in
`train_gpt_improved_04_16.py`. Open follow-ups: sub-0.5 on LATE/LOOP, all-groups=0.5,
full 4-way stack.

#### Wallclock-budget Sensitivity (`pgm_cliplate075_w3000`)

Re-ran the section-3 winner with `MAX_WALLCLOCK_SECONDS=3000` (vs 3600).

| run_id | wallclock | pre_val_bpb | sw_val_bpb | ttt_val_bpb |
|--------|-----------|-------------|------------|-------------|
| pgm_cliplate075 | 3600s | 1.06834 | 1.06955 | 1.06756 |
| pgm_cliplate075_w3000 | 3000s | 1.07299 | 1.07561 | 1.07359 |

A 17% wallclock cut costs **+0.00603 TTT BPB** — the recipe is genuinely
budget-hungry, not just bottlenecked on TTT eval.

### `best2304` Baseline Sweeps (4×A100, 3000s wallclock, train_gpt_improved_04_23.py)

New baseline `best2304` re-runs `program.md` sections **3 / 5 / 6** with
`HESSIAN_CLIP_LAMBDA=0.0`, `MATRIX_CLIP_SIGMAS=14`, `MLP_MULT=4.35`,
`VARLEN_ATTENTION=1`, `MAX_WALLCLOCK_SECONDS=3000`. Driver:
`run_best2304_sweeps.sh` (`SECTION=program_s356`).

Baseline `best2304` reference: pre **1.06840** · sw **1.08000** ·
**ttt 1.07788** · 16,527,189 bytes ✅.

#### Section 3 — `CLIP_MULT_*` (12 runs)

Sorted by TTT BPB. ⚠️ = exceeds 16 MB submission budget.

| run_id | sw_val_bpb | ttt_val_bpb | total_bytes | Δ ttt vs base |
|---|---:|---:|---:|---:|
| best2304_s3_cliploop075 | 1.07490 | **1.07289** | 16,844,016 ⚠️ | −0.00499 |
| best2304_s3_cliplate075 | 1.07496 | **1.07293** | 16,843,607 ⚠️ | −0.00496 |
| best2304_s3_clipmid075  | 1.07569 | 1.07362 | 16,686,349 ⚠️ | −0.00426 |
| **best2304_s3_clipearly125** | **1.07762** | **1.07552** | **16,009,739** ✅ | **−0.00237** |
| best2304_s3_clipmid125  | 1.07783 | 1.07571 | 16,131,142 ✅ | −0.00217 |
| best2304_s3_clipearly075| 1.07819 | 1.07610 | 16,843,555 ⚠️ | −0.00178 |
| best2304_s3_cliploop125 | 1.07852 | 1.07633 | 16,009,646 ✅ | −0.00155 |
| best2304_s3_cliplate125 | 1.07882 | 1.07668 | 16,009,815 ✅ | −0.00120 |
| best2304_s3_clipmid15   | 1.07922 | 1.07706 | 15,936,762 ✅ | −0.00082 |
| best2304_s3_clipearly15 | 1.07931 | 1.07714 | 15,718,081 ✅ | −0.00074 |
| best2304_s3_cliploop15  | 1.08125 | 1.07886 | 15,718,303 ✅ | +0.00098 |
| best2304_s3_cliplate15  | 1.08161 | 1.07928 | 15,718,029 ✅ | +0.00140 |

Takeaways:
- All four groups prefer **smaller** `CLIP_MULT` (0.75) for BPB; **LATE** and
  **LOOP** are the most sensitive (each gives ≈ −0.005 TTT) but bust budget.
- Among under-budget configs, **EARLY=1.25** is the new winner (−0.00237 TTT,
  drops size by 0.5 MB).

#### Section 5 — `PARALLEL_RESIDUAL_START` (5 runs, baseline = 7)

| run_id | sw_val_bpb | ttt_val_bpb | total_bytes |
|---|---:|---:|---:|
| **best2304_s5_pr9** | **1.07830** | **1.07624** | 16,372,792 ✅ |
| best2304_s5_pr11 | 1.07874 | 1.07668 | 16,372,945 ✅ |
| best2304_s5_pr5  | 1.08146 | 1.07941 | 16,373,900 ✅ |
| best2304_s5_pr3  | 1.08516 | 1.08310 | 16,373,526 ✅ |
| best2304_s5_pr0  | 1.09096 | 1.08884 | 16,376,004 ✅ |

Takeaways:
- `PARALLEL_RESIDUAL_START=9` is a marginal win over default 7 (−0.0016 TTT).
- Anything ≤5 collapses fast (parallel-residual blocks dominate).
- Size essentially constant (~16.37 MB) — pure architectural lever.

#### Section 6 — `WINDOW_ATTN_SIZE` (3 runs, baseline = 0)

| run_id | sw_val_bpb | ttt_val_bpb | total_bytes |
|---|---:|---:|---:|
| **best2304_s6_wattn512**  | **1.07647** | **1.07449** 🏆 | 16,373,469 ✅ |
| best2304_s6_wattn1024 | 1.07656 | 1.07452 | 16,372,993 ✅ |
| best2304_s6_wattn256  | 1.07651 | 1.07486 | 16,373,365 ✅ |

Takeaways:
- `WINDOW_ATTN_SIZE=512` is the best run across **all 20 sweep entries** at
  TTT **1.07449** (−0.00339 vs `best2304` baseline), and it fits 16 MB.
- 512 vs 1024 within noise; both decisively beat default (no window).
- Size unchanged (parameter-free attention mask change).

#### Pending follow-ups

- (done) `best2304_no_ntk_3000` — see NTK section below.
- (done) MoE size scout — see MoE section below.
- Open question: stack `wattn=512` + `pr=9` + `clipearly=1.25` — each is an
  independent architectural / quant lever, expected to be ~additive.

#### NTK-RoPE Disabled (1 run, 3000s, full TTT eval)

| run_id | sw_val_bpb | ttt_val_bpb | total_bytes |
|---|---:|---:|---:|
| best2304_no_ntk_3000 | 1.07739 | **1.20355** ❌ | 16,372,652 ✅ |
| baseline `best2304`  | 1.08000 | **1.07788** | 16,527,189 ✅ |

Takeaways:
- `DISABLE_NTK_ROPE=1` (no eval-time NTK base override + skip TTT base override)
  marginally **helps SW** (−0.00261 BPB).
- It **destroys TTT-LoRA** (+0.12567 BPB; TTT becomes much worse than SW).
- Verdict: TTT-LoRA depends on the NTK base override; cannot be disabled
  without redesigning TTT.

#### MoE Size Scout (6 runs, 60s `SIZE_ONLY=1`)

All MoE variants exceed the 16 MB budget. Sorted by total bytes.

| run_id | layers | experts | top_k | total_bytes | over budget |
|---|---|---:|---:|---:|---:|
| best2304_moe9_e2_k1_size60   | 9      | 2 | 1 | 17,296,407 | +519 KB |
| best2304_moe79_e2_k2_size60  | 7,9    | 2 | 2 | 18,222,289 | +1.45 MB |
| best2304_moe79_e2_k1_size60  | 7,9    | 2 | 1 | 18,225,175 | +1.45 MB |
| best2304_moe579_e2_k1_size60 | 5,7,9  | 2 | 1 | 19,152,699 | +2.38 MB |
| best2304_moe79_e3_k2_size60  | 7,9    | 3 | 2 | 20,075,116 | +3.30 MB |
| best2304_moe79_e3_k1_size60  | 7,9    | 3 | 1 | 20,079,064 | +3.30 MB |

Takeaways:
- **MoE adds ~1 MB per added MoE layer** and ~1.85 MB per extra expert.
- The cheapest variant (single MoE layer at depth 9, 2 experts, top-1) is
  +519 KB over budget. To fit MoE you must combine it with size-recovery
  levers (e.g. `LOOP_LAYER_BITS=5`, smaller `CLIP_MULT_LATE/LOOP`).
- `top_k` is essentially free for size (k=1 vs k=2 differs by ≤4 KB);
  it changes routing/eval BPB only.

#### Stacked-Winners Sweep (`logs/best2304_stack/`, driver finished; one failed 2026-04-27T12:51Z)

Stack the three independent winners (`WINDOW_ATTN_SIZE=512`,
`PARALLEL_RESIDUAL_START=9`, `CLIP_MULT_EARLY=1.25`) on the best2304
baseline and sweep extra knobs. Per-variant: 60s `SIZE_ONLY=1` size scout
-> if `< 16 MB`, full 3000s TTT-only run. Driver:
`run_best2304_stack.sh`. Resumable plan: `logs/best2304_stack/PLAN.md`.

Common env: `VARLEN_ATTENTION=1 MATRIX_CLIP_SIGMAS=14 MLP_MULT=4.35
HESSIAN_CLIP_LAMBDA=0.0 WINDOW_ATTN_SIZE=512 PARALLEL_RESIDUAL_START=9
CLIP_MULT_EARLY=1.25`.

| variant         | TTT bpb       | bytes        | delta vs best2304 (1.07788) |
|-----------------|---------------|--------------|-----------------------------|
| stack_sigmas13  | **1.07329** best | 16,451,868 over +451,868 | -0.00459                    |
| stack_loop085   | 1.07418       | 16,275,179 over +275,179 | -0.00370                    |
| stack_late085   | 1.07427       | 16,274,621 over +274,621 | -0.00361                    |
| stack_late09    | 1.07451       | 16,181,099 over +181,099 | -0.00337                    |
| stack_loop09    | 1.07457       | 16,180,912 over +180,912 | -0.00331                    |
| stack_hcl02     | 1.07491       | 16,021,047 over +21,047 | -0.00297                    |
| stack_hcl01     | 1.07492       | 16,013,460 over +13,460 | -0.00296                    |
| stack_hcl045    | 1.07507       | 16,059,546 over +59,546 | -0.00281                    |
| stack_control   | 1.07521       | 16,009,495 over +9,495 | -0.00267                    |
| stack_sigmas15  | **1.07942** valid | 15,599,622 OK | +0.00154                    |
| stack_early2    | FAILED        | scout 15,278,966 OK | NCCL init timeout; no full result |

Observations: best completed variant by BPB is `stack_sigmas13` at **1.07329** TTT BPB, but it is **451,868 bytes over** the 16,000,000-byte limit. The best completed decimal-size-valid variant is `stack_sigmas15` at **1.07942** TTT BPB (15,599,622 bytes), which is worse than the `best2304` baseline by +0.00154.
The driver reached its final board at 2026-04-27T02:00:41Z and no torch jobs are currently running. `stack_early2` passed the size scout but the full run failed during distributed/NCCL initialization (`wait timeout after 600000ms`), so it has no valid TTT result and should be rerun if that variant is still needed.


### TTT-LoRA Sweep on `pg12_varlen_clip14` Checkpoint

All runs below reload the same trained checkpoint (`pg12_eval_verify` confirms sliding-window
baseline = **1.07383**) and apply test-time LoRA adaptation. No retraining; the deltas isolate
TTT hyperparameter effects.

| run_id | rank | lr | chunk | extra | ttt_val_bpb | Δ vs sw |
|--------|------|-----|-------|-------|-------------|---------|
| **pg12_r48_phased3** | 48 | 1e-4 | 64 | phased(3) | **1.07173** 🏆 | −0.00210 |
| pg12_slotttt_default | 48 | 1e-4 | 64 | + SLOT-in-TTT (lr=1e-2,s=4) | 1.07176 | −0.00207 |
| pg12_slotttt_wd0 | 48 | 1e-4 | 64 | + SLOT, wd=0 | 1.07177 | −0.00206 |
| pg12_slotttt_steps8_lr3e3 | 48 | 1e-4 | 64 | + SLOT (s=8, lr=3e-3) | 1.07178 | −0.00205 |
| pg12_slotttt_steps2 | 48 | 1e-4 | 64 | + SLOT (s=2) | 1.07180 | −0.00203 |
| pg12_r48_phased2_sgd_ep2 | 48 | 1e-4 | 64 | phased(2) + SGD ep=2 | 1.07181 | −0.00202 |
| pg12_r48_phased2_sgd_lr1e3 | 48 | 1e-4 | 64 | phased(2) + SGD lr=1e-3 | 1.07181 | −0.00202 |
| pg12_slotttt_lr3e3 | 48 | 1e-4 | 64 | + SLOT (lr=3e-3) | 1.07183 | −0.00200 |
| pg12_r48_phased2 | 48 | 1e-4 | 64 | phased(2) | 1.07183 | −0.00200 |
| pg12_r48_phased2_adapt_tight | 48 | 1e-4 | 64 | phased(2) + adapt tight | 1.07184 | −0.00199 |
| pg12_r48_phased2_ema99 | 48 | 1e-4 | 64 | phased(2) + ema=0.99 | 1.07185 | −0.00198 |
| pg12_r48_phased2_minlen256 | 48 | 1e-4 | 64 | phased(2) + minlen=256 | 1.07186 | −0.00197 |
| pg12_ttt_rank48 | 48 | 1e-4 | 64 | (canonical) | 1.07187 | −0.00196 |
| pg12_adapt_ema80 | 48 | 1e-4 | 64 | adapt ema=0.80 | 1.07188 | −0.00195 |
| pg12_adapt_pow05 | 48 | 1e-4 | 64 | adapt pow=0.5 | 1.07188 | −0.00195 |
| pg12_adapt_tight | 48 | 1e-4 | 64 | adapt tight | 1.07188 | −0.00195 |
| pg12_adapt_wide | 48 | 1e-4 | 64 | adapt wide | 1.07188 | −0.00195 |
| pg12_r64_phased2 | 64 | 1e-4 | 64 | phased(2) | 1.07189 | −0.00194 |
| pg12_adapt_ema99 | 48 | 1e-4 | 64 | adapt ema=0.99 | 1.07190 | −0.00193 |
| pg12_adapt_pow15 | 48 | 1e-4 | 64 | adapt pow=1.5 | 1.07189 | −0.00194 |
| pg12_r48_phased2_chunk80 | 48 | 1e-4 | 80 | phased(2) | 1.07194 | −0.00189 |
| pg12_ttt_lr5e5 | 48 | 5e-5 | 64 | — | 1.07194 | −0.00189 |
| pg12_r48_phased2_minlen512 | 48 | 1e-4 | 64 | phased(2) + minlen=512 | 1.07194 | −0.00189 |
| pg12_ttt_minlen512 | 48 | 1e-4 | 64 | minlen=512 | 1.07198 | −0.00185 |
| pg12_r32_phased2 | 32 | 1e-4 | 64 | phased(2) | 1.07199 | −0.00184 |
| pg12_ttt_lr7e5 | 48 | 7e-5 | 64 | — | 1.07202 | −0.00181 |
| pg12_ttt_rank32 | 32 | 1e-4 | 64 | — | 1.07202 | −0.00181 |
| pg12_ttt_r48_chunk96 | 48 | 1e-4 | 96 | — | 1.07206 | −0.00177 |
| pg12_ttt_minlen1024 | 48 | 1e-4 | 64 | minlen=1024 | 1.07216 | −0.00167 |
| pg12_ttt_r48_chunk128 | 48 | 1e-4 | 128 | — | 1.07220 | −0.00163 |
| pg12_ttt_r48_lr5e5 | 48 | 5e-5 | 64 | — | 1.07226 | −0.00157 |
| pg12_ttt_chunk128 | 48 | 1e-4 | 128 | — | 1.07229 | −0.00154 |
| pg12_ttt_phased3 | 48 | 1e-4 | 64 | phased(3) | 1.07230 | −0.00153 |
| pg12_ttt_chunk96 | 48 | 1e-4 | 96 | — | 1.07233 | −0.00150 |
| pg12_ttt_phased2 | 48 | 1e-4 | 64 | phased(2) | 1.07240 | −0.00143 |
| pg12_slotttt_lr3e2 | 48 | 1e-4 | 64 | + SLOT (lr=3e-2) | 1.07235 | −0.00148 |
| pg12_eval_verify (TTT-LoRA) | 48 | 1e-4 | 64 | recompute baseline | 1.07243 | −0.00140 |
| pg12_ttt_minlen2048 | 48 | 1e-4 | 64 | minlen=2048 | 1.07236 | −0.00147 |
| pg12_ttt_chunk32 | 48 | 1e-4 | 32 | — | 1.07282 | −0.00101 |
| pg12_ttt_chunk32_adaptive | 48 | 1e-4 | 32 | + adaptive | 1.07290 | −0.00093 |
| pg12_ttt_phased3_r192 | 192 | 1e-4 | 64 | phased(3) | 1.07577 | +0.00194 |
| pg12_ttt_rank192 | 192 | 1e-4 | 64 | — | 1.07589 | +0.00207 |
| pg12_ttt_r192_adaptive | 192 | 1e-4 | 64 | + adaptive | 1.07607 | +0.00224 |
| pg12_ttt_steps2 | 48 | 1e-4 | 64 | inner_steps=2 | 1.07655 | +0.00272 |
| pg12_ttt_lr3e4 | 48 | 3e-4 | 64 | — | 1.07850 | +0.00467 |
| pg12_ttt_phased2_lr3e4 | 48 | 3e-4 | 64 | phased(2) | 1.07847 | +0.00464 |
| pg12_ttt_chunk32_steps2 | 48 | 1e-4 | 32 | inner_steps=2 | 1.07910 | +0.00527 |
| pg12_ttt_r192_lr2e4_adaptive | 192 | 2e-4 | 64 | + adaptive | 1.08447 | +0.01064 |
| pg12_ttt_r192_lr3e4 | 192 | 3e-4 | 64 | — | 1.09354 | +0.01971 |
| pg12_ttt_r192_chunk32_steps2 | 192 | 1e-4 | 32 | inner_steps=2 | 1.09455 | +0.02072 |

### SLOT-Only Sweep (no TTT-LoRA, same checkpoint)

SLOT (Self-supervised LOss-based Test-time fine-tuning of decoder bias terms only).
Baseline sliding-window = 1.07383.

| run_id | slot lr | slot steps | slot wd | slot_val_bpb | Δ vs sw |
|--------|---------|------------|---------|--------------|---------|
| pg12_slot_wd001 | 1e-2 | 4 | 0.001 | **1.07351** | −0.00032 |
| pg12_slot_default | 1e-2 | 4 | 0.01 | 1.07351 | −0.00032 |
| pg12_slot_steps8 | 1e-2 | 8 | 0.01 | 1.07353 | −0.00030 |
| pg12_slot_steps8_lr3e3 | 3e-3 | 8 | 0.01 | 1.07359 | −0.00024 |
| pg12_slot_steps2 | 1e-2 | 2 | 0.01 | 1.07363 | −0.00020 |
| pg12_slot_lr3e3 | 3e-3 | 4 | 0.01 | 1.07370 | −0.00013 |
| pg12_slot_lr3e2 | 3e-2 | 4 | 0.01 | 1.07387 | +0.00004 |

### RoPE Eval-Stride Ablation (`pg12_varlen_clip14` checkpoint)

Sliding-window evaluation at smaller strides has near-zero effect on bpb.

| run_id | eval_stride | sw_val_bpb |
|--------|-------------|------------|
| ropeA_auto (default 64) | 64 | 1.07491 |
| ropeA_stride s=32 | 32 | 1.07488 |
| ropeA_stride s=16 | 16 | 1.07487 |
| ropeA_stride s=8  | 8  | 1.07486 |

### Misc

| run_id | notes | result |
|--------|-------|--------|
| pg12_varlen_clip14_TTT | VarLen+TTT, **no** RoPE seqlen fix | broken: pre=1.281, sw=1.273, ttt=1.183 |
| pg12_varlen_clip14_TTT_ropeFix | VarLen+TTT, RoPE NTK override | pre=1.08147, sw=1.07383, ttt=1.07243 |
| check_debug_04_16_varlen_TTT | varlen+TTT debug attempt | pre=1.08321, sw=1.07598, ttt=1.18141 (broken) |

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

1. **Best 16MB-valid leaderboard candidate:** `pg13_varlen_clip16` (**1.07807** sw_bpb, 15.59MB ✅) — still the strongest size-passing dense run.
2. **TTT-LoRA is the biggest single post-training gain measured to date:** on the `pg12_varlen_clip14` checkpoint it cuts bpb from 1.07383 (sw) to **1.07173** (`pg12_r48_phased3`), a ≈−0.0021 improvement at zero training cost. Sweet spot: rank 48, lr 1e-4, chunk 64, phased schedule.
3. **TTT-LoRA hyperparameter ranking** (from sweep): rank ≤ 64 wins decisively (rank 192 is +0.002 to +0.020 worse), lr 1e-4 is optimal (3e-4 is +0.005, 5e-5 is +0.0001 worse), chunk 64 ≥ chunk 96 ≥ chunk 128 > chunk 32, `inner_steps=2` always hurts.
4. **SLOT alone is a small but free win** (~−0.0003 bpb), best at lr=1e-2, steps=4. SLOT-in-TTT is essentially neutral on top of TTT-LoRA (best `slotttt_default` 1.07176 vs best pure TTT 1.07173).
5. **XSA (sliding-attn last N layers) is the most promising new architectural feature.** `pgm_xsa9` reached **1.07102** ttt-bpb, but the branch is 16.52MB and needs additional compression to be eligible.
6. **Loop-Embeddings depth-2 helps slightly** (`pgm_loopemb1` 1.07169 ttt vs `pgm_xsa0` control 1.07442), but loops=3/4 hurt and all variants bust 16MB.
7. **`ENABLE_LOOPING_AT=0.5` is the new best absolute BPB.** `pgm_loopat0_5` reached **1.06919** ttt-bpb and **1.06630** pre-quant (both new records). Curriculum matters: immediate looping (0.0) is worst. All `pgm_loopat0_*` are ~0.5MB over 16MB.
8. **Eval-stride finer than 64 is wasted compute** — strides 8/16/32/64 all within 0.00005 bpb.
9. **Best sliding BPB on a 16MB-valid dense run:** `improved_GA_FUSErope` (**1.07369**) — GA+HS+ANS+FR+FM, 15.99MB ✅
10. **Best absolute BPB (over budget):** `pgm_clip_lateloop05` (**1.06372** TTT, 1.06559 sw) ← new — `CLIP_MULT_LATE=0.5 + CLIP_MULT_LOOP=0.5` on top of the loopat0_5 recipe; beats prior record 1.06919 by 0.00547. `pgm_clip_all075` (1.06426 TTT) close behind with all 4 groups at 0.75. `improved_GA_MoE79_MLP4_TTT` (1.06740 sw) — MoE 21.2MB.
11. **GPTQ wants tighter, not looser, per-group clipping — and stacking is super-additive.** Section 3 found every group prefers 0.75 over 1.25 (sensitivity LATE > LOOP > MID > EARLY). Section 7 then showed LATE=0.5 + LOOP=0.5 stacks to −0.00385 vs the linear sum 0.00198. Defaults bumped: `CLIP_MULT_LATE 1.0→0.5`, `CLIP_MULT_LOOP 1.0→0.5`.
10. **MLP 4.35× improves pre-quant BPB** — ~1.075 vs ~1.079 for 3.0×, but adds ~2.2M params
11. **Clip sigma 15.0 solves MLP4.35 size problem** — 15.97MB (vs 16.83MB at clip=13) with only 0.004 wider quant gap
12. **N-gram embeddings don't help:** Bigram-512 (1.07464) and trigram-1536 (1.07584) are worse than baseline (1.07441), and all bust 16MB
13. **Gated Attention is a consistent win** — ~-0.001 BPB for minimal parameter cost
14. **NorMuon hurts** — runs with NM consistently worse than without
15. **Window Attention is catastrophic** — 1.176 sw_bpb (vs 1.074 baseline), fast throughput doesn't compensate
16. **MoE achieves best raw BPB but models are 21MB+** — not viable for 16MB budget without dramatic compression
17. **ANS saves ~30KB** artifact size consistently vs Brotli
18. **VarLen is competitive but clip-sensitive** — `pg12_varlen_clip14` reached 1.07425 sliding (over 16MB); `pg13_varlen_clip16` fits at 15.59MB with 1.07807 sliding. **VarLen+TTT requires the RoPE NTK seqlen override** — without it, eval blows up to 1.18+.
19. **Earlier broken VarLen kernel (`improved_varlen`) is unrelated to current pg09+ VarLen path.**

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
5. ~~Debug VarLen attention (doc boundary detection).~~ Working in pg09+ pipeline; old `improved_varlen` path still broken (1.90).
6. ~~Fix code size bug in `train_gpt_improved2.py` (94KB → need `_ORIG_SCRIPT` fix).~~ Fixed via `_ORIG_SCRIPT` env var.
7. ~~Investigate why parameter banking hurts~~ — worse BPB and over budget, abandoned.
8. ~~Try MLP 4.35× with clip=13 and clip=15.~~ **Done** — clip=15 fits 16MB (15.98MB), clip=13 does not (16.83MB).
9. ~~Try MoE.~~ **Done** — best raw BPB (1.067) but 21MB+, not viable.
10. ~~Try Window Attention.~~ **Done** — catastrophic (1.176 sw_bpb), abandoned.
11. ~~Try TTT.~~ **Done** — fully working with RoPE NTK fix; ~−0.0021 bpb on the pg12 checkpoint.
12. ~~Run sliding window eval on `gated_clip15_mlp435`.~~ Superseded — focus moved to TTT + XSA on the pg12/pgm baseline.
13. ~~Sweep TTT-LoRA hyperparameters.~~ **Done** — `pg12_r48_phased3` (rank 48, lr 1e-4, chunk 64, 3-phase) is best at 1.07173.
14. ~~Try SLOT (decoder-bias-only test-time fine-tuning).~~ **Done** — small standalone gain (~−0.0003), neutral when stacked on TTT-LoRA.
15. ~~Eval-stride sweep.~~ **Done** — strides 8/16/32/64 indistinguishable; keep stride 64.
16. **Pack `train_gpt_improved_04_16.py` with `pack_submission_file.py`** — the script is 173KB raw but packs to ~44KB with LZMA+base85. That ~129KB savings makes the `pgm_xsa9` branch (1.07102 ttt_bpb, currently 16.52MB) easily fit under 16MB without architectural changes. Use the packed `train_gpt.py` (which auto-sets `_ORIG_SCRIPT`) for all subsequent pgm/XSA/TTT runs so size accounting reflects the packed code.
17. **Train a 16MB-valid VarLen + XSA + TTT submission** combining the best architectural and post-training tricks.
18. **Investigate broken `check_debug_04_16_varlen_TTT`** (varlen+TTT produced 1.181 bpb) — appears to be missing the RoPE override that `pg12_varlen_clip14_TTT_ropeFix` applied.
19. Explore MoE with aggressive quantization (6-bit?) to fit 16MB budget.

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
