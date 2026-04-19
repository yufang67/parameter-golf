#!/bin/bash
# Phase-2 TTT combo sweep. Waits for phase-1 sweep to finish, then tries
# combinations of the most promising knobs against the same checkpoint.
set -u
cd /root/parameter-golf
source .venv/bin/activate

LOGDIR=logs/pg12_sweep
mkdir -p "$LOGDIR"

echo "[combo] waiting for phase-1 sweep to finish..."
# Wait until the phase-1 driver script is gone AND no torchrun is alive
while pgrep -f "run_pg12_ttt_sweep.sh" > /dev/null \
   || pgrep -f "train_gpt_improved_04_16.py" > /dev/null; do
    sleep 30
done
echo "[combo] phase-1 done. Starting combo sweep."
ls -la final_model.int6.ptz

if [ ! -f final_model.int6.ptz ]; then
    echo "[combo] ERROR: no checkpoint, aborting"
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

run_combo() {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    echo "[combo] === $name ==="
    env -i HOME=$HOME PATH=$PATH \
        RUN_ID="$name" \
        "${BASE_ENV[@]}" \
        "$@" \
        torchrun --standalone --nproc_per_node=4 train_gpt_improved_04_16.py \
        > "$log" 2>&1
    echo "[combo] $name exit=$?"
    grep -E "val_bpb|quantized_ttt|elapsed" "$log" | tail -10
}

# Capacity + faster learning
run_combo "pg12_ttt_r192_lr3e4"           TTT_LORA_RANK=192 TTT_LORA_LR=3e-4

# Capacity + adaptive LR scaling
run_combo "pg12_ttt_r192_adaptive"        TTT_LORA_RANK=192 TTT_ADAPTIVE_LR=1

# Smaller chunks + more steps per chunk (compute-equivalent if step is cheap)
run_combo "pg12_ttt_chunk32_steps2"       TTT_CHUNK_SIZE=32 TTT_GRAD_STEPS=2

# Smaller chunks + adaptive
run_combo "pg12_ttt_chunk32_adaptive"     TTT_CHUNK_SIZE=32 TTT_ADAPTIVE_LR=1

# Phased global SGD + faster LoRA LR
run_combo "pg12_ttt_phased2_lr3e4"        TTT_PHASES=2 TTT_LORA_LR=3e-4

# Phased + adaptive
run_combo "pg12_ttt_phased2_adaptive"     TTT_PHASES=2 TTT_ADAPTIVE_LR=1

# Kitchen sink: capacity + faster + adaptive
run_combo "pg12_ttt_r192_lr2e4_adaptive"  TTT_LORA_RANK=192 TTT_LORA_LR=2e-4 TTT_ADAPTIVE_LR=1

# Aggressive: capacity + smaller chunks + 2 steps
run_combo "pg12_ttt_r192_chunk32_steps2"  TTT_LORA_RANK=192 TTT_CHUNK_SIZE=32 TTT_GRAD_STEPS=2

# Phased ×3 alone (more global passes)
run_combo "pg12_ttt_phased3"              TTT_PHASES=3

# Phased ×3 + capacity
run_combo "pg12_ttt_phased3_r192"         TTT_PHASES=3 TTT_LORA_RANK=192

echo "[combo] all combos complete"
ls -la "$LOGDIR"/
