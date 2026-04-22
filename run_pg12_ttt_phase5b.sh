#!/bin/bash
# Phase-5b: re-run SLOT sweep now that train_and_eval actually dispatches
# eval_val_slot when SLOT_ENABLED=1. Waits for phase-5 to complete first.
set -u
cd /root/parameter-golf
source .venv/bin/activate

LOGDIR=logs/pg12_sweep
mkdir -p "$LOGDIR"

echo "[p5b] waiting for phase-5 to finish..."
while pgrep -f "run_pg12_ttt_phase5.sh" > /dev/null \
   || pgrep -f "train_gpt_improved_04_16.py" > /dev/null; do
    sleep 30
done
echo "[p5b] phase-5 done. Starting SLOT re-run."

if [ ! -f final_model.int6.ptz ]; then
    echo "[p5b] ERROR: no checkpoint, aborting"
    exit 1
fi

run_slot() {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    echo "[p5b] === $name ==="
    env -i HOME=$HOME PATH=$PATH \
        RUN_ID="$name" \
        MATRIX_CLIP_SIGMAS=14 MLP_MULT=4.35 GPTQ_CALIBRATION_BATCHES=128 \
        VARLEN_ATTENTION=1 SLIDING_WINDOW_ENABLED=1 SKIP_TRAINING=1 \
        TTT_ENABLED=0 MAX_WALLCLOCK_SECONDS=3600 \
        SLOT_ENABLED=1 \
        "$@" \
        torchrun --standalone --nproc_per_node=4 train_gpt_improved_04_16.py \
        > "$log" 2>&1
    echo "[p5b] $name exit=$?"
    grep -E "quantized_slot val_loss" "$log" | tail -3
}

# Replace earlier (no-op) SLOT logs with proper ones
for f in pg12_slot_default pg12_slot_lr3e3 pg12_slot_lr3e2 pg12_slot_steps2 \
         pg12_slot_steps8 pg12_slot_wd001 pg12_slot_steps8_lr3e3; do
    rm -f "$LOGDIR/${f}.log"
done

run_slot "pg12_slot_default"
run_slot "pg12_slot_lr3e3"          SLOT_LR=3e-3
run_slot "pg12_slot_lr3e2"          SLOT_LR=3e-2
run_slot "pg12_slot_steps2"         SLOT_STEPS=2
run_slot "pg12_slot_steps8"         SLOT_STEPS=8
run_slot "pg12_slot_wd001"          SLOT_WD=0.001
run_slot "pg12_slot_steps8_lr3e3"   SLOT_STEPS=8 SLOT_LR=3e-3

echo "[p5b] === SLOT LEADERBOARD ==="
for f in "$LOGDIR"/pg12_slot_*.log; do
    bpb=$(grep "quantized_slot val_loss" "$f" | tail -1 | grep -oP "val_bpb:\K[0-9.]+")
    [ -n "$bpb" ] && printf "%-44s %s\n" "$(basename "$f" .log)" "$bpb"
done | sort -k2 -n
