#!/bin/bash
# Phase-7: SLOT-in-TTT composition. Combines per-chunk SLOT logit delta with
# per-doc TTT LoRA (best config: r48 + phased2). Waits for phase-6 to finish.
set -u
cd /root/parameter-golf
source .venv/bin/activate

LOGDIR=logs/pg12_sweep
mkdir -p "$LOGDIR"

echo "[p7] waiting for phase-6 to finish..."
while pgrep -f "run_pg12_ttt_phase6.sh" > /dev/null \
   || pgrep -f "train_gpt_improved_04_16.py" > /dev/null; do
    sleep 30
done
echo "[p7] phase-6 done. Starting SLOT-in-TTT sweep."

if [ ! -f final_model.int6.ptz ]; then
    echo "[p7] ERROR: no checkpoint, aborting"
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
  TTT_LORA_RANK=48
  TTT_PHASES=2
  SLOT_IN_TTT=1
)

run_p7() {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    echo "[p7] === $name ==="
    env -i HOME=$HOME PATH=$PATH \
        RUN_ID="$name" \
        "${BASE_ENV[@]}" \
        "$@" \
        torchrun --standalone --nproc_per_node=4 train_gpt_improved_04_16.py \
        > "$log" 2>&1
    echo "[p7] $name exit=$?"
    grep -E "quantized_ttt_lora val_loss" "$log" | tail -3
}

# Defaults: steps=4 lr=0.01 wd=0.001 (best from phase-5b SLOT)
run_p7 "pg12_slotttt_default"
run_p7 "pg12_slotttt_steps2"          SLOT_IN_TTT_STEPS=2
run_p7 "pg12_slotttt_steps8"          SLOT_IN_TTT_STEPS=8
run_p7 "pg12_slotttt_lr3e3"           SLOT_IN_TTT_LR=3e-3
run_p7 "pg12_slotttt_lr3e2"           SLOT_IN_TTT_LR=3e-2
run_p7 "pg12_slotttt_wd0"             SLOT_IN_TTT_WD=0
run_p7 "pg12_slotttt_steps8_lr3e3"    SLOT_IN_TTT_STEPS=8 SLOT_IN_TTT_LR=3e-3

echo "[p7] all phase-7 variants complete"
echo "[p7] === FULL TTT LEADERBOARD ==="
for f in "$LOGDIR"/pg12_*.log; do
    bpb=$(grep "quantized_ttt_lora val_loss" "$f" | tail -1 | grep -oP "val_bpb:\K[0-9.]+")
    [ -n "$bpb" ] && printf "%-44s %s\n" "$(basename "$f" .log)" "$bpb"
done | sort -k2 -n
