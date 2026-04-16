#!/bin/bash
# 5 experiments based on strategy.md next hypotheses
# Base config: train_gpt_improved.py defaults already include:
#   SOFTCAP=20, WARMDOWN=0.85, ROPE_DIMS=32, GATED_ATTENTION=1
# Current best: gated_clip15_mlp435 (pre=1.07518, post=1.09027, artifact=15.98MB)

set -e
cd /root/parameter-golf
source .venv/bin/activate

COMMON="DATA_DIR=./data2/ MAX_WALLCLOCK_SECONDS=3600"

echo "=== Experiment 45: CLIP=14 + MLP=4.35 ==="
echo "Hypothesis: Clip=14 interpolates clip=13 (post=1.08670,16.83MB) and clip=15 (post=1.09027,15.98MB)"
RUN_ID=exp45_clip14_mlp435 DATA_DIR=./data2/ MAX_WALLCLOCK_SECONDS=3600 \
  MLP_MULT=4.35 MATRIX_CLIP_SIGMAS=14 \
  torchrun --standalone --nproc_per_node=4 train_gpt_improved.py 2>&1 | tee logs/exp45_clip14_mlp435.log

echo "=== Experiment 46: CLIP=13 + MLP=4.15 ==="
echo "Hypothesis: Clip=13 better post-quant, reduce MLP to fit 16MB"
RUN_ID=exp46_clip13_mlp415 DATA_DIR=./data2/ MAX_WALLCLOCK_SECONDS=3600 \
  MLP_MULT=4.15 MATRIX_CLIP_SIGMAS=13 \
  torchrun --standalone --nproc_per_node=4 train_gpt_improved.py 2>&1 | tee logs/exp46_clip13_mlp415.log

echo "=== Experiment 47: CLIP=15 + MLP=4.35 + CALIB=128 ==="
echo "Hypothesis: More GPTQ calibration batches improve quantization"
RUN_ID=exp47_clip15_calib128 DATA_DIR=./data2/ MAX_WALLCLOCK_SECONDS=3600 \
  MLP_MULT=4.35 MATRIX_CLIP_SIGMAS=15 GPTQ_CALIBRATION_BATCHES=128 \
  torchrun --standalone --nproc_per_node=4 train_gpt_improved.py 2>&1 | tee logs/exp47_clip15_calib128.log

echo "=== Experiment 48: CLIP=14 + MLP=4.40 ==="
echo "Hypothesis: With clip=14 better compression, can push MLP higher"
RUN_ID=exp48_clip14_mlp440 DATA_DIR=./data2/ MAX_WALLCLOCK_SECONDS=3600 \
  MLP_MULT=4.40 MATRIX_CLIP_SIGMAS=14 \
  torchrun --standalone --nproc_per_node=4 train_gpt_improved.py 2>&1 | tee logs/exp48_clip14_mlp440.log

echo "=== Experiment 49: CLIP=15 + MLP=4.35 + SEED=42 (reproducibility) ==="
echo "Hypothesis: Verify current best is reproducible with different seed"
RUN_ID=exp49_repro_seed42 DATA_DIR=./data2/ MAX_WALLCLOCK_SECONDS=3600 \
  MLP_MULT=4.35 MATRIX_CLIP_SIGMAS=15 SEED=42 \
  torchrun --standalone --nproc_per_node=4 train_gpt_improved.py 2>&1 | tee logs/exp49_repro_seed42.log

echo "=== All 5 experiments complete ==="
