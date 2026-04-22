#!/bin/bash
# Phase-3 TTT sweep: explore the "lower / smaller / larger" direction since
# higher LR, larger rank, and smaller chunks all regressed in phase-2.
# Waits for phase-2 combo driver to finish.
set -u
cd /root/parameter-golf
source .venv/bin/activate

LOGDIR=logs/pg12_sweep
mkdir -p "$LOGDIR"

echo "[p3] waiting for phase-2 combo sweep to finish..."
while pgrep -f "run_pg12_ttt_combos.sh" > /dev/null \
   || pgrep -f "run_pg12_ttt_sweep.sh" > /dev/null \
   || pgrep -f "train_gpt_improved_04_16.py" > /dev/null; do
    sleep 30
done
echo "[p3] phase-2 done. Starting phase-3."
ls -la final_model.int6.ptz

if [ ! -f final_model.int6.ptz ]; then
    echo "[p3] ERROR: no checkpoint, aborting"
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
)

run_p3() {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    echo "[p3] === $name ==="
    env -i HOME=$HOME PATH=$PATH \
        RUN_ID="$name" \
        "${BASE_ENV[@]}" \
        "$@" \
        torchrun --standalone --nproc_per_node=4 train_gpt_improved_04_16.py \
        > "$log" 2>&1
    echo "[p3] $name exit=$?"
    grep -E "quantized_ttt_lora val_loss" "$log" | tail -3
}

# Lower LR (default 1e-4)
run_p3 "pg12_ttt_lr5e5"      TTT_LORA_LR=5e-5
run_p3 "pg12_ttt_lr7e5"      TTT_LORA_LR=7e-5

# Smaller rank (default 96)
run_p3 "pg12_ttt_rank48"     TTT_LORA_RANK=48
run_p3 "pg12_ttt_rank32"     TTT_LORA_RANK=32

# Larger chunks (default 64)
run_p3 "pg12_ttt_chunk96"    TTT_CHUNK_SIZE=96
run_p3 "pg12_ttt_chunk128"   TTT_CHUNK_SIZE=128

echo "[p3] all phase-3 variants complete"
echo "[p3] === LEADERBOARD ==="
for f in "$LOGDIR"/pg12_*.log; do
    bpb=$(grep "quantized_ttt_lora val_loss" "$f" | tail -1 | grep -oP "val_bpb:\K[0-9.]+")
    [ -n "$bpb" ] && printf "%-40s %s\n" "$(basename "$f" .log)" "$bpb"
done | sort -k2 -n
