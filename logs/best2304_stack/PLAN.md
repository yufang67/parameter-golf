# best2304 Stacked-Winners Sweep — Resumable Plan

**Folder:** `logs/best2304_stack/`
**Driver:** `/root/parameter-golf/run_best2304_stack.sh`
**Outer log:** `logs/best2304_stack/_outer.log`
**Source:** `train_gpt_improved_04_23.py` → packed to `train_gpt.py`
**Size budget:** 16,777,216 bytes (16 MB)

## Goal
Stack the 3 independent best2304 baseline winners and search for further BPB
gains via clip/sigma/hessian sweeps. Pre 1.06840 / sw 1.08000 / **TTT 1.07788**
is the baseline to beat.

## Stacked baseline (COMMON_STACK_ENV)
```
VARLEN_ATTENTION=1
MATRIX_CLIP_SIGMAS=14
MLP_MULT=4.35
HESSIAN_CLIP_LAMBDA=0.0
WINDOW_ATTN_SIZE=512        # winner s6
PARALLEL_RESIDUAL_START=9   # winner s5
CLIP_MULT_EARLY=1.25        # winner s3
GPTQ_CALIBRATION_BATCHES=128
COMPRESS_ANS=1 COMPRESS_BROTLI=0 COMPRESS_LZMA=0
```

## Workflow per variant
1. **Scout** (≈6 min): `MAX_WALLCLOCK_SECONDS=60 TTT_ENABLED=0 SIZE_ONLY=1`
   → log `${name}_scout.log`, parse `Total submission size quantized+ans: N bytes`.
2. If `N < 16,777,216` → **Full** (≈75 min):
   `MAX_WALLCLOCK_SECONDS=3000 TTT_ENABLED=1 SLIDING_WINDOW_ENABLED=0`
   → log `${name}.log`.
3. Else → write `${name}.skip` and move on.

Sentinels: `${name}_scout.done`, `${name}.done`, `${name}.skip`.
The driver auto-skips any variant whose sentinel already exists, so the
script is **idempotent / resumable** — just re-run it after a restart.

## 11 Variants

| # | name           | extra env override                  |
|---|----------------|--------------------------------------|
| 1 | stack_control  | (none)                               |
| 2 | stack_late085  | CLIP_MULT_LATE=0.85                  |
| 3 | stack_late09   | CLIP_MULT_LATE=0.9                   |
| 4 | stack_loop085  | CLIP_MULT_LOOP=0.85                  |
| 5 | stack_loop09   | CLIP_MULT_LOOP=0.9                   |
| 6 | stack_sigmas13 | MATRIX_CLIP_SIGMAS=13                |
| 7 | stack_sigmas15 | MATRIX_CLIP_SIGMAS=15                |
| 8 | stack_hcl01    | HESSIAN_CLIP_LAMBDA=0.1              |
| 9 | stack_hcl02    | HESSIAN_CLIP_LAMBDA=0.2              |
|10 | stack_hcl045   | HESSIAN_CLIP_LAMBDA=0.45             |
|11 | stack_early2   | CLIP_MULT_EARLY=2.0                  |

## How to resume
```bash
# 1) make sure no torchrun is still alive
pgrep -af "torchrun.*train_gpt" || echo "no train procs"

# 2) re-launch driver — finished variants are auto-skipped via .done/.skip
cd /root/parameter-golf
nohup bash run_best2304_stack.sh > logs/best2304_stack/_outer.log 2>&1 & disown
```

To **force re-run** of a variant: delete its `.done`/`.skip`/`_scout.done`
sentinels under `logs/best2304_stack/`.

## Status board
```bash
ls logs/best2304_stack/*.done logs/best2304_stack/*.skip 2>/dev/null
tail -40 logs/best2304_stack/_outer.log
```

## Done condition
- All 11 variants have either a `.done` (full ran) or `.skip` (over budget) file.
- Driver prints final eval board to `_outer.log`.
- Update `logs/results_summary.md` with a "best2304 Stacked Winners" table.
