#!/bin/bash
# program.md execution: full sweep including architectural changes that require
# retraining. Waits for phase-7 to finish first.
#
# Each variant: trains a fresh checkpoint with one knob changed (~1hr), then
# runs the best-known TTT eval config (r48 + phased3) on it (~25min).
# Total: ~50 runs × ~1.5hr = ~3 days.
#
# Naming: pgm_<sweep>_<value>
set -u
cd /root/parameter-golf
source .venv/bin/activate

LOGDIR=logs/program_md
mkdir -p "$LOGDIR"

echo "[pgm] waiting for phase-7 to finish..."
while pgrep -f "run_pg12_ttt_phase7.sh" > /dev/null \
   || pgrep -f "run_pg12_ttt_phase6.sh" > /dev/null \
   || pgrep -f "train_gpt_improved_04_16.py" > /dev/null; do
    sleep 30
done
echo "[pgm] queue clear. Starting program.md sweeps."

# Common training+eval base env (matches current pg12_varlen_clip14 baseline).
# All architectural variants override one knob and re-train.
COMMON_TRAIN_ENV=(
  MATRIX_CLIP_SIGMAS=14
  MLP_MULT=4.35
  GPTQ_CALIBRATION_BATCHES=128
  VARLEN_ATTENTION=1
  TTT_ENABLED=1
  MAX_WALLCLOCK_SECONDS=3600
  # New defaults from phases 3-6
  TTT_LORA_RANK=48
  TTT_PHASES=3
)

# train_then_eval: trains a fresh ckpt with the given overrides, then logs all
# three eval passes (dense + SW + TTT) in the SAME run. Saves ckpt under a
# variant-specific name so we don't clobber the existing one.
train_then_eval() {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    local ckpt="${name}.int6.ptz"
    if [ -f "$LOGDIR/${name}.done" ]; then
        echo "[pgm] SKIP $name (already done)"
        return
    fi
    echo "[pgm] === $name === (train+eval)"
    # The script always writes to final_model.int6.ptz; we move it after.
    env -i HOME=$HOME PATH=$PATH \
        RUN_ID="$name" \
        "${COMMON_TRAIN_ENV[@]}" \
        "$@" \
        torchrun --standalone --nproc_per_node=4 train_gpt_improved_04_16.py \
        > "$log" 2>&1
    local ec=$?
    echo "[pgm] $name exit=$ec"
    grep -E "quantized(_sliding_window|_ttt_lora|_slot)? val_loss" "$log" | tail -5
    # Snapshot the produced checkpoint so we can audit later.
    if [ -f final_model.int6.ptz ]; then
        cp final_model.int6.ptz "$LOGDIR/$ckpt"
    fi
    touch "$LOGDIR/${name}.done"
}

# ============================================================================
# (1) XSA_LAST_N — coverage of XSA layers, 0 cost change but architectural
# ============================================================================
for v in 0 4 7 9; do
    train_then_eval "pgm_xsa${v}"  XSA_LAST_N=$v
done
# 11 = current default, skip

# ============================================================================
# (2) LOOP_EMBEDDINGS — per-pass bias for looped mid-stack
# ============================================================================
train_then_eval "pgm_loopemb1"  LOOP_EMBEDDINGS=1
# Deeper recurrence with loop embeddings on
train_then_eval "pgm_loopemb1_loops3"  LOOP_EMBEDDINGS=1 NUM_LOOPS=3
train_then_eval "pgm_loopemb1_loops4"  LOOP_EMBEDDINGS=1 NUM_LOOPS=4

# ============================================================================
# (3) CLIP_MULT_* — per-group GPTQ clip multipliers (post-training, but the
#     full pipeline is rerun since quantization happens at end-of-train).
# ============================================================================
for grp in EARLY LOOP MID LATE; do
    for v in 0.75 1.25 1.5; do
        train_then_eval "pgm_clip${grp,,}${v//./}"  CLIP_MULT_${grp}=$v
    done
done

# ============================================================================
# (4) ENABLE_LOOPING_AT — curriculum switch fraction
# ============================================================================
for v in 0.0 0.15 0.5 0.7; do
    name="pgm_loopat${v//./_}"
    train_then_eval "$name"  ENABLE_LOOPING_AT=$v
done
# 0.35 = current default

# ============================================================================
# (5) PARALLEL_RESIDUAL_START — placement of parallel residuals
# ============================================================================
for v in 11 9 5 3 0; do
    train_then_eval "pgm_pr${v}"  PARALLEL_RESIDUAL_START=$v
done
# 7 = current default

# ============================================================================
# (6) WINDOW_ATTN_SIZE — sliding-window training-time attention
# ============================================================================
for v in 256 512 1024; do
    train_then_eval "pgm_wattn${v}"  WINDOW_ATTN_SIZE=$v
done
# Layer pattern sweep at best window size (deferred — requires manual choice
# after step above completes; skip for now)

# ============================================================================
# Final leaderboard
# ============================================================================
echo "[pgm] === FULL PROGRAM.MD LEADERBOARD ==="
echo "name                                       dense    sw       ttt"
for f in "$LOGDIR"/pgm_*.log; do
    name=$(basename "$f" .log)
    d=$(grep "^quantized val_loss"            "$f" | tail -1 | grep -oP "val_bpb:\K[0-9.]+")
    s=$(grep "quantized_sliding_window val_loss" "$f" | tail -1 | grep -oP "val_bpb:\K[0-9.]+")
    t=$(grep "quantized_ttt_lora val_loss"      "$f" | tail -1 | grep -oP "val_bpb:\K[0-9.]+")
    printf "%-42s %s %s %s\n" "$name" "${d:--}" "${s:--}" "${t:--}"
done | sort -k4
