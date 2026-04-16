#!/bin/bash
set -e

# Batch 4: 10 experiments — 2026-04-16
# Script: train_gpt_improved_04_15.py
# New features to test: bigram/trigram hash embeddings, VarLen attention,
# MoE MLP, mixed seq length training, hessian clip defaults
# Base: 4xA100, 3600s wallclock, SP8192

cd /root/parameter-golf
source .venv/bin/activate

# Common env: data, compression, wallclock, gated attention
COMMON="DATA_DIR=./data2/ MAX_WALLCLOCK_SECONDS=3600 GATED_ATTENTION=1 COMPRESS_ANS=1"
NPROC=4
SCRIPT=train_gpt_improved_04_15.py
LOG_DIR=logs

mkdir -p "$LOG_DIR"

run_exp() {
    local run_id="$1"
    shift
    echo ""
    echo "================================================================"
    echo "=== Starting $run_id at $(date -u) ==="
    echo "================================================================"
    local cmd="$COMMON $* RUN_ID=$run_id torchrun --standalone --nproc_per_node=$NPROC $SCRIPT"
    echo "CMD: $cmd"
    eval "$cmd" 2>&1 | tee "${LOG_DIR}/${run_id}_stdout.txt"
    echo "=== Finished $run_id at $(date -u) ==="
    echo ""
}

# ── Exp 76: Baseline — improved script with SOTA params ──
# Establish what improved_04_15.py achieves with SOTA-equivalent params.
# Key difference from stripped: this script may have different code paths.
run_exp exp76_improved_baseline \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128

# ── Exp 77: Bigram hash embeddings (small) ──
# BigramHash gives the model explicit bigram context via hash embeddings.
# Small config to minimize parameter overhead: 1024 vocab, 64 dim.
# Extra params: ~1024*64 + 64*512 ≈ 98K params ≈ ~74KB compressed.
run_exp exp77_bigram_small \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128 \
    BIGRAM_VOCAB_SIZE=1024 BIGRAM_DIM=64

# ── Exp 78: Bigram hash embeddings (medium) ──
# Larger bigram: 2048 vocab, 128 dim. More capacity for bigram patterns.
# Extra params: ~2048*128 + 128*512 ≈ 328K params ≈ ~246KB compressed.
run_exp exp78_bigram_medium \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128 \
    BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128

# ── Exp 79: Trigram hash embeddings ──
# TrigramHash captures 3-token context. Small config to test viability.
run_exp exp79_trigram \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128 \
    TRIGRAM_VOCAB_SIZE=1024 TRIGRAM_DIM=64

# ── Exp 80: VarLen attention (within-document) ──
# flash_attn_varlen restricts attention to within documents during training.
# No extra params — just better training signal by avoiding cross-doc attention.
run_exp exp80_varlen \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128 \
    VARLEN_ATTENTION=1

# ── Exp 81: Hessian clip 0.3 with SOTA params ──
# improved_04_15.py defaults to hessian=0.3. exp47 showed -0.0008 post-quant.
# Re-test with improved code (may interact with brotli compressor differently).
run_exp exp81_hessian03 \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 HESSIAN_CLIP_LAMBDA=0.3 GPTQ_CALIBRATION_BATCHES=128

# ── Exp 82: Bigram + VarLen combo ──
# Combine two low-cost features: bigram embeddings + within-doc attention.
run_exp exp82_bigram_varlen \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128 \
    BIGRAM_VOCAB_SIZE=1024 BIGRAM_DIM=64 VARLEN_ATTENTION=1

# ── Exp 83: MoE tiny — 2 experts, top-1, only layer 5 ──
# Minimal MoE: 2 experts with top-1 routing on a single layer.
# Adds only 1 extra MLP copy = ~2M extra params. MLP=4.0 to fit budget.
run_exp exp83_moe_tiny \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.0 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128 \
    MOE_ENABLED=1 MOE_NUM_EXPERTS=2 MOE_TOP_K=1 MOE_LAYERS=5

# ── Exp 84: Clip=13 + hessian (improved defaults) ──
# Test the improved script's default clip=13 + hessian=0.3.
# MLP=4.0 to give room for the larger artifact from tighter clip.
run_exp exp84_clip13_hessian \
    MATRIX_CLIP_SIGMAS=13 MLP_MULT=4.0 HESSIAN_CLIP_LAMBDA=0.3 GPTQ_CALIBRATION_BATCHES=128

# ── Exp 85: Bigram + Trigram combined ──
# Both n-gram hash embeddings at once. Captures 2-token and 3-token context.
# MLP=4.0 to leave artifact room for the extra embedding tables.
run_exp exp85_bigram_trigram \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.0 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128 \
    BIGRAM_VOCAB_SIZE=1024 BIGRAM_DIM=64 TRIGRAM_VOCAB_SIZE=1024 TRIGRAM_DIM=64

echo ""
echo "================================================================"
echo "=== ALL 10 EXPERIMENTS COMPLETE at $(date -u) ==="
echo "================================================================"
