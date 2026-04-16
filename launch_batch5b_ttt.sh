#!/bin/bash
set -e

# Batch 5b: TTT sweep on exp86 VarLen checkpoint — 2026-04-16
# All runs are EVAL_ONLY=1 reusing final_model.pt / .ptz from exp86_varlen_base.
#
# Diagnosis of exp87 fail: LoRA TTT with ttt_eval_seq_len=2048 + chunk_size=48
# scores every chunk including short-context ones (chunks 0..41 of each doc).
# Sliding eval only scores tokens past position 4032. So LoRA TTT inherently
# has worse BPB than sliding. Fullparam TTT uses the sliding scoring protocol
# and is what worked in exp58 (1.07045).
#
# Plan: default to fullparam TTT (works with varlen now via ttt_mode=fullparam),
# sweep LR/epochs/chunk-tokens. Also try a couple of LoRA variants with
# larger ttt_eval_seq_len to see if the short-context gap closes.

cd /root/parameter-golf
source .venv/bin/activate

# Sanity check
test -f final_model.pt
test -f final_model.int6.ptz

COMMON="DATA_DIR=./data2/ GATED_ATTENTION=1 COMPRESS_ANS=1 \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 HESSIAN_CLIP_LAMBDA=0 GPTQ_CALIBRATION_BATCHES=128 \
    VARLEN_ATTENTION=1 EVAL_ONLY=1 SLIDING_WINDOW_ENABLED=0 TTT_ENABLED=1"

NPROC=4
SCRIPT=train_gpt_improved_04_15.py
LOG_DIR=logs
mkdir -p "$LOG_DIR"

run_eval() {
    local run_id="$1"; shift
    echo ""
    echo "================================================================"
    echo "=== EVAL $run_id at $(date -u) ==="
    echo "================================================================"
    local cmd="$COMMON $* RUN_ID=$run_id torchrun --standalone --nproc_per_node=$NPROC $SCRIPT"
    echo "CMD: $cmd"
    eval "$cmd" 2>&1 | tee "${LOG_DIR}/${run_id}_stdout.txt"
    echo "=== EVAL done $run_id at $(date -u) ==="
}

# ── Fullparam TTT sweep (primary direction) ──

# Exp 97: fullparam baseline (SGD lr=0.005 mom=0.9, exp58-proven default).
run_eval exp97_ttt_fp_base TTT_MODE=fullparam

# Exp 98: lower LR (3e-3) for more conservative adaptation.
run_eval exp98_ttt_fp_lr3e3 TTT_MODE=fullparam TTT_FP_LR=3e-3

# Exp 99: higher LR (1e-2). Stress-test divergence guards.
run_eval exp99_ttt_fp_lr1e2 TTT_MODE=fullparam TTT_FP_LR=1e-2

# Exp 100: single-epoch adaptation (faster, less overfit).
run_eval exp100_ttt_fp_ep1 TTT_MODE=fullparam TTT_FP_EPOCHS=1

# Exp 101: longer adaptation (5 epochs per chunk).
run_eval exp101_ttt_fp_ep5 TTT_MODE=fullparam TTT_FP_EPOCHS=5

# Exp 102: smaller chunks (16384 tokens) → more adaptation steps, finer grain.
run_eval exp102_ttt_fp_ch16k TTT_MODE=fullparam TTT_FP_CHUNK_TOKENS=16384

# Exp 103: larger chunks (65536 tokens) → fewer, larger updates.
run_eval exp103_ttt_fp_ch64k TTT_MODE=fullparam TTT_FP_CHUNK_TOKENS=65536

# Exp 104: AdamW with very small LR (3e-5) to avoid Adam's initial-variance blow-up.
run_eval exp104_ttt_fp_adamw_lr3e5 TTT_MODE=fullparam TTT_FP_OPTIMIZER=adamw TTT_FP_LR=3e-5

# ── LoRA variants with larger context window ──

# Exp 105: LoRA TTT with eval_seq_len=4096 (match sliding) and chunk=64.
# Hypothesis: matching the sliding protocol closes the scoring gap.
run_eval exp105_ttt_lora_sl4096 TTT_MODE=lora TTT_EVAL_SEQ_LEN=4096 TTT_CHUNK_SIZE=64

# Exp 106: LoRA QV-only rank 32 with eval_seq_len=4096 chunk=64.
# Minimum-footprint adaptation to avoid divergence.
run_eval exp106_ttt_lora_qv_sl4096 TTT_MODE=lora TTT_EVAL_SEQ_LEN=4096 TTT_CHUNK_SIZE=64 \
    TTT_LORA_RANK=32 TTT_K_LORA=0 TTT_MLP_LORA=0 TTT_O_LORA=0

echo ""
echo "================================================================"
echo "=== ALL 10 TTT EXPERIMENTS COMPLETE at $(date -u) ==="
echo "================================================================"
