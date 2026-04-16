#!/bin/bash
# Batch 3: 10 experiments (exp66-75) on train_gpt_stripped.py
# Run AFTER current batch (exp59-65) completes
# Base: exp58 config (GPTQ128, CLIP=15, MLP=4.35, stripped, TTT at eval)
# Current best: exp58 (pre=1.07409, post=1.08890, TTT=1.07045, artifact=15.99MB)

set -e
cd /root/parameter-golf
source .venv/bin/activate

COMMON="DATA_DIR=./data2/ FUSED_ROPE=0 MAX_WALLCLOCK_SECONDS=3600 GATED_ATTENTION=1 COMPRESS_ANS=1 COMPRESS_BROTLI=1"
NPROC=4
S=train_gpt_stripped.py
LOG_DIR=logs

run_exp() {
    local run_id="$1"; shift
    echo ""
    echo "================================================================"
    echo "=== Starting $run_id at $(date -u) ==="
    echo "================================================================"
    eval "$COMMON $* RUN_ID=$run_id torchrun --standalone --nproc_per_node=$NPROC $S" \
      2>&1 | tee "${LOG_DIR}/${run_id}_stdout.txt"
    echo "=== Finished $run_id at $(date -u) ==="
}

# ── EXP66: Reproducibility — same as exp58 with TTT SEED=42 + 
run_exp exp66_repro_seed42 \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=128 \
    TTT_ENABLED=1 SEED=42

# ── EXP67: SOFTCAP=18 — push softcap lower than 20 ──
run_exp exp67_softcap18 \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=128 \
    LOGIT_SOFTCAP=18

# ── EXP68: WARMDOWN=0.88 — push warmdown a bit further ──
run_exp exp68_warmdown088 \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=128 \
    WARMDOWN_FRAC=0.88

# ── EXP69: EMBED_BITS=6 — int6 embeddings to save artifact space for bigger model ──
run_exp exp69_embedbits6 \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=128 \
    EMBED_BITS=6

# ── EXP70: GRAD_CLIP=0.5 — looser gradient clipping ──
run_exp exp70_gradclip05 \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=128 \
    GRAD_CLIP_NORM=0.5

# ── EXP71: 12L MLP=3.85 + TTT — depth over width ──
run_exp exp71_12L_ttt \
    NUM_LAYERS=12 MLP_MULT=3.85 MATRIX_CLIP_SIGMAS=15 GPTQ_CALIBRATION_BATCHES=128 \
    TTT_ENABLED=1

# ── EXP72: EMBED_BITS=6 + MLP=4.45 — reclaim embed budget for more MLP ──
run_exp exp72_emb6_mlp445 \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.45 GPTQ_CALIBRATION_BATCHES=128 \
    EMBED_BITS=6

# ── EXP73: GPTQ_CAL=256 + TTT — push calibration further + eval boost ──
run_exp exp73_gptq256_ttt \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=256 \
    TTT_ENABLED=1

# ── EXP74: MUON_WD=0.08 — slightly less weight decay ──
run_exp exp74_muonwd08 \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=128 \
    MUON_WD=0.08

# ── EXP75: QK_GAIN=6.0 + SOFTCAP=18 — combo: sharper attn + tighter softcap ──
run_exp exp75_qk6_sc18 \
    MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=128 \
    QK_GAIN_INIT=6.0 LOGIT_SOFTCAP=18

echo ""
echo "================================================================"
echo "=== ALL 10 EXPERIMENTS (exp66-75) COMPLETE at $(date -u) ==="
echo "================================================================"
