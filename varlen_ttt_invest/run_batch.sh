#!/bin/bash
# Run 9 diagnostic experiments sequentially. tv01 is already done (tv00_smoke).
set -u
cd /root/parameter-golf
source .venv/bin/activate

LOG_DIR=varlen_ttt_invest/logs
mkdir -p "$LOG_DIR"

COMMON_ENV=(
  DATA_DIR=./data2/ FUSED_ROPE=0 GATED_ATTENTION=1 COMPRESS_ANS=1 COMPRESS_BROTLI=1
  MATRIX_CLIP_SIGMAS=15 MLP_MULT=4.35 VARLEN_ATTENTION=1
  SKIP_TRAINING=1 DEBUG_VARLEN_PROBE=1 TTT_ENABLED=1 TTT_MAX_DOCS=50
)

run() {
  local run_id="$1"; shift
  local extra_env=("$@")
  echo ""
  echo "================================================================"
  echo "=== Starting $run_id at $(date -u) ==="
  echo "================================================================"
  env "${COMMON_ENV[@]}" "${extra_env[@]}" RUN_ID="$run_id" \
    torchrun --standalone --nproc_per_node=4 train_gpt_improved_04_16.py \
    > "$LOG_DIR/${run_id}.txt" 2>&1
  local rc=$?
  echo "=== Finished $run_id rc=$rc at $(date -u) ==="
  # Print headline results
  grep -E "quantized val_bpb|quantized_sliding_window val_bpb|quantized_ttt_lora val_bpb|quantized_ttt_fullparam val_bpb|\[PROBE\]" "$LOG_DIR/${run_id}.txt" | tail -15
}

run tv02_force_dense   FORCE_DENSE_EVAL=1 RUN_TTT_ONLY=1
run tv03_ttt_no_compile TTT_COMPILE_DISABLE=1 RUN_TTT_ONLY=1
run tv04_ttt_pad_bos    EVAL_PAD_TOKEN=1 RUN_TTT_ONLY=1
run tv05_ttt_lora_lr0   TTT_LORA_LR=0 RUN_TTT_ONLY=1
run tv06_ttt_lora_lr1e5 TTT_LORA_LR=1e-5 RUN_TTT_ONLY=1
run tv07_ttt_lora_lr5e6 TTT_LORA_LR=5e-6 RUN_TTT_ONLY=1
run tv08_ttt_qv_only    TTT_K_LORA=0 TTT_MLP_LORA=0 TTT_O_LORA=0 TTT_LORA_RANK=32 TTT_LORA_LR=5e-5 TTT_BETA2=0.999 RUN_TTT_ONLY=1
run tv09_ttt_rank8      TTT_LORA_RANK=8 RUN_TTT_ONLY=1
run tv10_force_dense_lr0 FORCE_DENSE_EVAL=1 TTT_LORA_LR=0 RUN_TTT_ONLY=1

echo ""
echo "================================================================"
echo "=== ALL 9 EXPERIMENTS COMPLETE at $(date -u) ==="
echo "================================================================"
