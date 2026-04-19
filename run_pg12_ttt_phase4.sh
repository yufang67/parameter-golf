#!/bin/bash
# Phase-4: combine the phase-3 winners. Smaller rank, lower LR, larger chunk
# all helped individually; explore their combinations.
set -u
cd /root/parameter-golf
source .venv/bin/activate

LOGDIR=logs/pg12_sweep
mkdir -p "$LOGDIR"

echo "[p4] waiting for previous sweeps to finish..."
while pgrep -f "run_pg12_ttt_phase3.sh" > /dev/null \
   || pgrep -f "run_pg12_ttt_combos.sh" > /dev/null \
   || pgrep -f "run_pg12_ttt_sweep.sh" > /dev/null \
   || pgrep -f "train_gpt_improved_04_16.py" > /dev/null; do
    sleep 30
done
echo "[p4] previous sweeps done. Starting phase-4."
ls -la final_model.int6.ptz

if [ ! -f final_model.int6.ptz ]; then
    echo "[p4] ERROR: no checkpoint, aborting"
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
  RUN_TTT_ONLY=1
  MAX_WALLCLOCK_SECONDS=3600
)

run_p4() {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    echo "[p4] === $name ==="
    env -i HOME=$HOME PATH=$PATH \
        RUN_ID="$name" \
        "${BASE_ENV[@]}" \
        "$@" \
        torchrun --standalone --nproc_per_node=4 train_gpt_improved_04_16.py \
        > "$log" 2>&1
    echo "[p4] $name exit=$?"
    grep -E "quantized_ttt_lora val_loss" "$log" | tail -3
}

# Top-2 winners combined
run_p4 "pg12_ttt_r48_lr5e5"           TTT_LORA_RANK=48 TTT_LORA_LR=5e-5
run_p4 "pg12_ttt_r48_lr7e5"           TTT_LORA_RANK=48 TTT_LORA_LR=7e-5

# Top winner + larger chunks
run_p4 "pg12_ttt_r48_chunk96"         TTT_LORA_RANK=48 TTT_CHUNK_SIZE=96
run_p4 "pg12_ttt_r48_chunk128"        TTT_LORA_RANK=48 TTT_CHUNK_SIZE=128

# Triples
run_p4 "pg12_ttt_r48_lr5e5_chunk96"   TTT_LORA_RANK=48 TTT_LORA_LR=5e-5 TTT_CHUNK_SIZE=96
run_p4 "pg12_ttt_r48_lr7e5_chunk96"   TTT_LORA_RANK=48 TTT_LORA_LR=7e-5 TTT_CHUNK_SIZE=96
run_p4 "pg12_ttt_r48_lr5e5_chunk128"  TTT_LORA_RANK=48 TTT_LORA_LR=5e-5 TTT_CHUNK_SIZE=128

# Even smaller rank + lower LR
run_p4 "pg12_ttt_r32_lr5e5"           TTT_LORA_RANK=32 TTT_LORA_LR=5e-5

# Best-so-far + phased2 (best of phase-2)
run_p4 "pg12_ttt_r48_phased2"         TTT_LORA_RANK=48 TTT_PHASES=2
run_p4 "pg12_ttt_r48_lr5e5_phased2"   TTT_LORA_RANK=48 TTT_LORA_LR=5e-5 TTT_PHASES=2

echo "[p4] all phase-4 variants complete"
echo "[p4] === FULL LEADERBOARD ==="
for f in "$LOGDIR"/pg12_*.log; do
    bpb=$(grep "quantized_ttt_lora val_loss" "$f" | tail -1 | grep -oP "val_bpb:\K[0-9.]+")
    [ -n "$bpb" ] && printf "%-42s %s\n" "$(basename "$f" .log)" "$bpb"
done | sort -k2 -n
