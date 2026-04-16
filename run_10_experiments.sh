#!/bin/bash
set -e

# Batch 2: 10 experiments — 2026-04-14
# Focus: fit under 16MB, combine winners, test TTT, architecture exploration
# Key insight: stripped code (83.7KB vs 96.3KB) saves 12.6KB for model budget

cd /root/parameter-golf
source .venv/bin/activate

COMMON="DATA_DIR=./data2/ FUSED_ROPE=0 MAX_WALLCLOCK_SECONDS=3600 GATED_ATTENTION=1 COMPRESS_ANS=1 COMPRESS_BROTLI=1"
NPROC=4
STRIPPED=train_gpt_stripped.py
FULL=train_gpt_improved.py
LOG_DIR=logs

run_exp() {
    local run_id="$1"
    local script="$2"
    shift 2
    echo ""
    echo "================================================================"
    echo "=== Starting $run_id at $(date -u) ==="
    echo "================================================================"
    local cmd="$COMMON $* RUN_ID=$run_id torchrun --standalone --nproc_per_node=$NPROC $script"
    echo "CMD: $cmd"
    eval "$cmd" 2>&1 | tee "${LOG_DIR}/${run_id}_stdout.txt"
    echo "=== Finished $run_id at $(date -u) ==="
    echo ""
}

# ── Exp 56: Stripped code + GPTQ128 ──
# The winning combo: stripped code saves 12.6KB, making exp48's model fit under 16MB
# exp48 was only 12KB over, stripped code saves 12.6KB → should fit with ~400 bytes spare
run_exp exp56_stripped_gptq128 $STRIPPED \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128

# ── Exp 57: Stripped + GPTQ128 + Hessian ──
# Combine the two best quantization techniques from batch 1
# GPTQ128 gave 1.08905 post-quant, hessian gave 1.08944 — together should be even better
run_exp exp57_stripped_combo $STRIPPED \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 HESSIAN_CLIP_LAMBDA=0.3 GPTQ_CALIBRATION_BATCHES=128

# ── Exp 58: Stripped + GPTQ128 + TTT ──
# Test-time training: SOTA uses TTT for ~0.002 BPB boost (1.083→1.081)
# Should give us a free quality boost at eval time
run_exp exp58_ttt $STRIPPED \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128 \
    TTT_ENABLED=1

# ── Exp 59: Stripped + MLP=4.30 (safe fit) ──
# MLP=4.30 gives more headroom under 16MB for combining techniques
# Slightly fewer params but more budget margin
run_exp exp59_mlp430 $STRIPPED \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.30 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128

# ── Exp 60: Stripped + MLP=4.30 + GPTQ128 + Hessian + LoopBits=7 ──
# MLP=4.30 frees budget; use some for loop layer int7 (between int6 and int8)
# Loop layers are reused 2x, so higher precision helps a lot
run_exp exp60_mlp430_loop7 $STRIPPED \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.30 HESSIAN_CLIP_LAMBDA=0.3 GPTQ_CALIBRATION_BATCHES=128 \
    LOOP_LAYER_BITS=7

# ── Exp 61: Stripped + GPTQ256 ──
# Push calibration batches further. 64→128 helped; maybe 256 helps more
run_exp exp61_gptq256 $STRIPPED \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=256

# ── Exp 62: Stripped + lower eval_stride ──
# EVAL_STRIDE=32 (vs 64) gives finer-grained sliding window — better BPB at cost of eval time
run_exp exp62_stride32 $STRIPPED \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128 \
    EVAL_STRIDE=32

# ── Exp 63: Stripped + EMA_DECAY=0.997 ──
# EMA=0.997 was a runner-up in earlier experiments (tied with best)
# With better quantization it might pull ahead
run_exp exp63_ema997 $STRIPPED \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128 \
    EMA_DECAY=0.997

# ── Exp 64: Stripped + NUM_LAYERS=12, MLP_MULT=3.85 ──
# More layers = more capacity per param (attention is cheap)
# Trade MLP width for depth. Same total params, different allocation.
run_exp exp64_12L_mlp385 $STRIPPED \
    NUM_LAYERS=12 MLP_MULT=3.85 MATRIX_CLIP_SIGMAS=15 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128

# ── Exp 65: Stripped + CLIP_SIGMAS=14 + MLP=4.25 ──
# Try clip=14 with smaller MLP to fit 16MB. Old exp45 used clip=14 with full code
# and higher MLP — now with stripped code and lower MLP it might fit
run_exp exp65_clip14_mlp425 $STRIPPED \
    MATRIX_CLIP_SIGMAS=14 MLP_MULT=4.25 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128

echo ""
echo "================================================================"
echo "=== ALL 10 EXPERIMENTS COMPLETE at $(date -u) ==="
echo "================================================================"
