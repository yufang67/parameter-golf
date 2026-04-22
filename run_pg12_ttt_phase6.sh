#!/bin/bash
# Phase-6: untested combos stacked on top of best-so-far (r48_phased2 = 1.07183).
set -u
cd /root/parameter-golf
source .venv/bin/activate

LOGDIR=logs/pg12_sweep
mkdir -p "$LOGDIR"

if [ ! -f final_model.int6.ptz ]; then
    echo "[p6] ERROR: no checkpoint, aborting"
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

run_p6() {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    echo "[p6] === $name ==="
    env -i HOME=$HOME PATH=$PATH \
        RUN_ID="$name" \
        "${BASE_ENV[@]}" \
        "$@" \
        torchrun --standalone --nproc_per_node=4 train_gpt_improved_04_16.py \
        > "$log" 2>&1
    echo "[p6] $name exit=$?"
    grep -E "quantized_ttt_lora val_loss" "$log" | tail -3
}

# Stacks on top of r48 + phased2
run_p6 "pg12_r48_phased2_minlen512"  TTT_LORA_RANK=48 TTT_PHASES=2 TTT_MIN_DOC_LEN=512
run_p6 "pg12_r48_phased2_minlen256"  TTT_LORA_RANK=48 TTT_PHASES=2 TTT_MIN_DOC_LEN=256
run_p6 "pg12_r48_phased2_adapt_tight" TTT_LORA_RANK=48 TTT_PHASES=2 \
                                      TTT_ADAPTIVE_LR=1 TTT_ADAPT_MIN_SCALE=0.7 TTT_ADAPT_MAX_SCALE=1.5
run_p6 "pg12_r48_phased2_ema99"      TTT_LORA_RANK=48 TTT_PHASES=2 \
                                      TTT_ADAPTIVE_LR=1 TTT_ADAPT_EMA=0.99
run_p6 "pg12_r48_phased2_chunk80"    TTT_LORA_RANK=48 TTT_PHASES=2 TTT_CHUNK_SIZE=80

# Phase variants
run_p6 "pg12_r48_phased3"            TTT_LORA_RANK=48 TTT_PHASES=3
run_p6 "pg12_r48_phased2_sgd_ep2"    TTT_LORA_RANK=48 TTT_PHASES=2 TTT_PHASE_SGD_EPOCHS=2
run_p6 "pg12_r48_phased2_sgd_lr1e3"  TTT_LORA_RANK=48 TTT_PHASES=2 TTT_PHASE_SGD_LR=1e-3

# Rank ablation alongside phased2
run_p6 "pg12_r32_phased2"            TTT_LORA_RANK=32 TTT_PHASES=2
run_p6 "pg12_r64_phased2"            TTT_LORA_RANK=64 TTT_PHASES=2

echo "[p6] all phase-6 variants complete"
echo "[p6] === FULL TTT LEADERBOARD ==="
for f in "$LOGDIR"/pg12_*.log; do
    bpb=$(grep "quantized_ttt_lora val_loss" "$f" | tail -1 | grep -oP "val_bpb:\K[0-9.]+")
    [ -n "$bpb" ] && printf "%-44s %s\n" "$(basename "$f" .log)" "$bpb"
done | sort -k2 -n
