#!/bin/bash
# Phase-5: Three exploration directions on top of best-so-far.
#   (A) TTT only on long-context docs  (TTT_MIN_DOC_LEN sweep)
#   (B) SLOT-4 parameter tuning
#   (C) Adaptive-LR parameter tuning
# Best-known TTT config: TTT_LORA_RANK=48 (+ TTT_PHASES=2 marginal)
set -u
cd /root/parameter-golf
source .venv/bin/activate

LOGDIR=logs/pg12_sweep
mkdir -p "$LOGDIR"

if [ ! -f final_model.int6.ptz ]; then
    echo "[p5] ERROR: no checkpoint, aborting"
    exit 1
fi

BASE_ENV=(
  MATRIX_CLIP_SIGMAS=14
  MLP_MULT=4.35
  GPTQ_CALIBRATION_BATCHES=128
  VARLEN_ATTENTION=1
  TTT_ENABLED=1
  SLIDING_WINDOW_ENABLED=1
  SKIP_TRAINING=1
  SLIDING_WINDOW_ENABLED=0
  MAX_WALLCLOCK_SECONDS=3600
  # phase-3/4 winner
  TTT_LORA_RANK=48
)

run_p5() {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    echo "[p5] === $name ==="
    env -i HOME=$HOME PATH=$PATH \
        RUN_ID="$name" \
        "${BASE_ENV[@]}" \
        "$@" \
        torchrun --standalone --nproc_per_node=4 train_gpt_improved_04_16.py \
        > "$log" 2>&1
    echo "[p5] $name exit=$?"
    grep -E "quantized_(ttt_lora|sliding_window) val_loss" "$log" | tail -3
}

# Helper for SLOT runs (need TTT_ENABLED=0 since SLOT runs in sliding-window pass)
run_p5_slot() {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    echo "[p5] === $name ==="
    env -i HOME=$HOME PATH=$PATH \
        RUN_ID="$name" \
        MATRIX_CLIP_SIGMAS=14 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=128 \
        VARLEN_ATTENTION=1 SLIDING_WINDOW_ENABLED=1 SKIP_TRAINING=1 \
        TTT_ENABLED=0 MAX_WALLCLOCK_SECONDS=3600 \
        SLOT_ENABLED=1 \
        "$@" \
        torchrun --standalone --nproc_per_node=4 train_gpt_improved_04_16.py \
        > "$log" 2>&1
    echo "[p5] $name exit=$?"
    grep -E "quantized_sliding_window val_loss" "$log" | tail -3
}

# === (A) TTT only on long-context docs ===
# val docs in fineweb_val are mixed length; gate by min token count.
run_p5 "pg12_ttt_minlen512"            TTT_MIN_DOC_LEN=512
run_p5 "pg12_ttt_minlen1024"           TTT_MIN_DOC_LEN=1024
run_p5 "pg12_ttt_minlen2048"           TTT_MIN_DOC_LEN=2048
run_p5 "pg12_ttt_minlen4096"           TTT_MIN_DOC_LEN=4096

# === (B) SLOT-4 tuning (sliding-window pass, defaults: steps=4 lr=0.01 wd=0.01) ===
run_p5_slot "pg12_slot_default"
run_p5_slot "pg12_slot_lr3e3"          SLOT_LR=3e-3
run_p5_slot "pg12_slot_lr3e2"          SLOT_LR=3e-2
run_p5_slot "pg12_slot_steps2"         SLOT_STEPS=2
run_p5_slot "pg12_slot_steps8"         SLOT_STEPS=8
run_p5_slot "pg12_slot_wd001"          SLOT_WD=0.001
run_p5_slot "pg12_slot_steps8_lr3e3"   SLOT_STEPS=8 SLOT_LR=3e-3

# === (C) Adaptive-LR tuning (combined with rank=48) ===
# Defaults: ema=0.95 min=0.5 max=2.0 power=1.0
run_p5 "pg12_adapt_pow05"              TTT_ADAPTIVE_LR=1 TTT_ADAPT_POWER=0.5
run_p5 "pg12_adapt_pow15"              TTT_ADAPTIVE_LR=1 TTT_ADAPT_POWER=1.5
run_p5 "pg12_adapt_ema80"              TTT_ADAPTIVE_LR=1 TTT_ADAPT_EMA=0.80
run_p5 "pg12_adapt_ema99"              TTT_ADAPTIVE_LR=1 TTT_ADAPT_EMA=0.99
run_p5 "pg12_adapt_tight"              TTT_ADAPTIVE_LR=1 TTT_ADAPT_MIN_SCALE=0.7 TTT_ADAPT_MAX_SCALE=1.5
run_p5 "pg12_adapt_wide"               TTT_ADAPTIVE_LR=1 TTT_ADAPT_MIN_SCALE=0.3 TTT_ADAPT_MAX_SCALE=3.0

echo "[p5] all phase-5 variants complete"
echo "[p5] === FULL LEADERBOARD (ttt_lora) ==="
for f in "$LOGDIR"/pg12_*.log; do
    bpb=$(grep "quantized_ttt_lora val_loss" "$f" | tail -1 | grep -oP "val_bpb:\K[0-9.]+")
    [ -n "$bpb" ] && printf "%-44s %s\n" "$(basename "$f" .log)" "$bpb"
done | sort -k2 -n
echo "[p5] === LEADERBOARD (sliding_window / SLOT) ==="
for f in "$LOGDIR"/pg12_slot_*.log; do
    bpb=$(grep "quantized_sliding_window val_loss" "$f" | tail -1 | grep -oP "val_bpb:\K[0-9.]+")
    [ -n "$bpb" ] && printf "%-44s %s\n" "$(basename "$f" .log)" "$bpb"
done | sort -k2 -n
