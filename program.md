# Experiment Program

**Goal:** Lower `val_bpb` on `train_gpt_improved.py`. One variable at a time, record everything.

## Setup

| Item | Value |
|------|-------|
| GPUs | 4× A100 80GB PCIe |
| Env | `.venv` (activated) |
| Data | `./data` (sp8192) |
| Wallclock | `MAX_WALLCLOCK_SECONDS=3600` |

## Baseline (best run)

| Metric | Value |
|--------|-------|
| Sliding-window BPB | **1.07369** |
| Pre-quant BPB | 1.07873 |
| Total artifact | 15,989,467 B (fits 16MB) |
| Steps | 5,445 |
| Config | `GATED_ATTENTION=1 FUSED_ROPE=1` |

## Workflow

```bash
# 1. Pack
python3 pack_submission_file.py train_gpt_improved.py train_gpt.py

# 2. Run without TTT
RUN_ID=<name> <ENV_OVERRIDES> MAX_WALLCLOCK_SECONDS=3600 \
  torchrun --standalone --nproc_per_node=4 train_gpt.py 2>&1 | tee logs/<name>.txt

# 3. Record — append command to command.txt, update logs/results_summary.md

# 4. Rerun the best with TTT
```

## Rules

- Pack before every run.
- Always `tee` to `logs/`.
- One variable per experiment. Name `RUN_ID` after what changed.
- Sequential: finish → record → wait for next.

## Experiments Plan
-  baseline is 
```
MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=128 TTT_ENABLED=1 MAX_WALLCLOCK_SECONDS=3600 torchrun --standalone --nproc_per_node=4 train_gpt.py
```
- experiment bigram and trigram (done)
- experiment varlen and try different parameters. Ignore TTT issue
- experiment MoE. The TTT could have issue with MoE.


## Keep/Discard

- **Keep:** BPB improves and artifact ≤ 16,000,000 B. Or BPB flat with better size/speed.
- **Discard:** BPB worsens without compensating gains.
- **Crash:** Log, diagnose, fix, re-run.