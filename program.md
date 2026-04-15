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

# 2. Run
RUN_ID=<name> <ENV_OVERRIDES> MAX_WALLCLOCK_SECONDS=3600 \
  torchrun --standalone --nproc_per_node=4 train_gpt.py 2>&1 | tee logs/<name>.txt

# 3. Record — append command to command.txt, update logs/results_summary.md
```

## Rules

- Pack before every run.
- Always `tee` to `logs/`.
- One variable per experiment. Name `RUN_ID` after what changed.
- Sequential: finish → record → wait for next.

## Feature Flags

| Flag | Default | Notes |
|------|---------|-------|
| `BIGRAM_VOCAB_SIZE` | 0 | Hash-bigram embedding |
| `TRIGRAM_VOCAB_SIZE` | 0 | Hash-trigram embedding |
| `MOE_ENABLED` | 0 | Mixture-of-experts MLP |
| `TTT_ENABLED` | 0 | Test-time training |
| `MLP_MULT` | 4 | MLP width multiplier |

## Keep/Discard

- **Keep:** BPB improves and artifact ≤ 16,000,000 B. Or BPB flat with better size/speed.
- **Discard:** BPB worsens without compensating gains.
- **Crash:** Log, diagnose, fix, re-run.