#!/bin/bash
# TTT variant sweep against the pg12_varlen_clip14 checkpoint.
# Waits for the training job to finish, then runs each variant with SKIP_TRAINING=1.
set -u
cd /root/parameter-golf
source .venv/bin/activate

LOGDIR=logs/pg12_sweep
mkdir -p "$LOGDIR"

# Wait for training torchrun to finish (any process whose cmdline mentions train_gpt_improved_04_16)
echo "[sweep] waiting for training to finish..."
while pgrep -f "train_gpt_improved_04_16.py" > /dev/null; do
    sleep 30
done
echo "[sweep] training done; checkpoint:"
ls -la final_model.int6.ptz

if [ ! -f final_model.int6.ptz ]; then
    echo "[sweep] ERROR: no checkpoint produced, aborting"
    exit 1
fi

# Common base env (matches training)
BASE_ENV=(
  MATRIX_CLIP_SIGMAS=14
  MLP_MULT=4.35
  GPTQ_CALIBRATION_BATCHES=128
  VARLEN_ATTENTION=1
  TTT_ENABLED=1
  SLIDING_WINDOW_ENABLED=1
  SKIP_TRAINING=1
  MAX_WALLCLOCK_SECONDS=3600
)

run_variant() {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    echo "[sweep] === $name ==="
    echo "[sweep] log: $log"
    env -i HOME=$HOME PATH=$PATH \
        RUN_ID="$name" \
        "${BASE_ENV[@]}" \
        "$@" \
        torchrun --standalone --nproc_per_node=4 train_gpt_improved_04_16.py \
        > "$log" 2>&1
    echo "[sweep] $name exit=$?"
    grep -E "val_bpb|quantized|elapsed" "$log" | tail -20
}

# Verification run: dense + SW + TTT (existing ropeFix log was TTT-only)
run_variant "pg12_eval_verify"

# Variants: TTT-only to save time (~10min vs 30min each)
# Larger LoRA rank
run_variant "pg12_ttt_rank192"   SLIDING_WINDOW_ENABLED=0  TTT_LORA_RANK=192

# 3) Higher LoRA LR
run_variant "pg12_ttt_lr3e4"     SLIDING_WINDOW_ENABLED=0  TTT_LORA_LR=3e-4

# 4) Smaller chunk size (more frequent updates)
run_variant "pg12_ttt_chunk32"   SLIDING_WINDOW_ENABLED=0  TTT_CHUNK_SIZE=32

# 5) More grad steps per chunk
run_variant "pg12_ttt_steps2"    SLIDING_WINDOW_ENABLED=0  TTT_GRAD_STEPS=2

# 6) Adaptive per-chunk LR
run_variant "pg12_ttt_adaptive"  SLIDING_WINDOW_ENABLED=0  TTT_ADAPTIVE_LR=1

# 7) Phased global SGD on already-scored prefix
run_variant "pg12_ttt_phased2"   SLIDING_WINDOW_ENABLED=0  TTT_PHASES=2

echo "[sweep] all variants complete"
