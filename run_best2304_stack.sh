#!/bin/bash
set -u

cd /root/parameter-golf
source .venv/bin/activate

LOGDIR=logs/best2304_stack
mkdir -p "$LOGDIR"

SCRIPT_SRC="train_gpt_improved_04_23.py"
SCRIPT_PACKED="train_gpt.py"
TORCHRUN="/root/parameter-golf/.venv/bin/torchrun"
SIZE_BUDGET=16777216

pack_entrypoint() {
    python3 pack_submission_file.py "$SCRIPT_SRC" "$SCRIPT_PACKED"
}

wait_for_queue() {
    while pgrep -f "torchrun --standalone --nproc_per_node=4 train_gpt" > /dev/null; do
        sleep 30
    done
}

# Stacked baseline: best2304 winners (wattn=512, pr=9, clipearly=1.25)
COMMON_STACK_ENV=(
  VARLEN_ATTENTION=1
  MATRIX_CLIP_SIGMAS=14
  MLP_MULT=4.35
  HESSIAN_CLIP_LAMBDA=0.0
  WINDOW_ATTN_SIZE=512
  PARALLEL_RESIDUAL_START=9
  CLIP_MULT_EARLY=1.25
  GPTQ_CALIBRATION_BATCHES=128
  COMPRESS_ANS=1
  COMPRESS_BROTLI=0
  COMPRESS_LZMA=0
)

snapshot_log() {
    local log="$1"
    {
        echo "===== CODE SNAPSHOT: ${SCRIPT_SRC} ====="
        echo "git_head: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "sha256:   $(sha256sum "$SCRIPT_SRC" | awk '{print $1}')"
        echo "===== BEGIN ${SCRIPT_SRC} ====="
        cat "$SCRIPT_SRC"
        echo "===== END ${SCRIPT_SRC} ====="
        echo
    } > "$log"
}

run_torchrun() {
    local name="$1"; shift
    local log="$1"; shift
    env -i HOME="$HOME" PATH="$PATH" \
        RUN_ID="$name" \
        "${COMMON_STACK_ENV[@]}" \
        "$@" \
        "$TORCHRUN" --standalone --nproc_per_node=4 "$SCRIPT_PACKED" \
        >> "$log" 2>&1
}

# Run a variant: scout first, then full if scout fits the budget
# Args: name, then any extra ENV=value overrides
run_variant() {
    local name="$1"; shift
    local scout_log="$LOGDIR/${name}_scout.log"
    local full_log="$LOGDIR/${name}.log"
    local rc=0

    if [ -f "$LOGDIR/${name}.done" ] || [ -f "$LOGDIR/${name}.skip" ]; then
        echo "[stack] SKIP $name (already finalized)"
        return
    fi

    # ------- size scout -------
    if [ ! -f "$LOGDIR/${name}_scout.done" ]; then
        echo "[stack] === scout $name === $(date -Iseconds)"
        snapshot_log "$scout_log"
        run_torchrun "${name}_scout" "$scout_log" \
            MAX_WALLCLOCK_SECONDS=60 TTT_ENABLED=0 SIZE_ONLY=1 "$@" || rc=$?
        echo "[stack] scout $name exit=$rc"
        touch "$LOGDIR/${name}_scout.done"
    fi

    local size
    size=$(grep -oE 'Total submission size quantized\+[a-z]+: [0-9]+ bytes' "$scout_log" \
            | tail -1 | grep -oE '[0-9]+' | tail -1)
    if [ -z "$size" ]; then
        echo "[stack] $name: no size in scout log; skipping full"
        echo "no_size" > "$LOGDIR/${name}.skip"
        return
    fi
    echo "[stack] $name scout_size=$size  budget=$SIZE_BUDGET"

    if [ "$size" -ge "$SIZE_BUDGET" ]; then
        echo "[stack] $name OVER BUDGET ($size >= $SIZE_BUDGET); skipping full"
        echo "over_budget=$size" > "$LOGDIR/${name}.skip"
        return
    fi

    # ------- full TTT-only run -------
    echo "[stack] === full $name === $(date -Iseconds)"
    snapshot_log "$full_log"
    run_torchrun "$name" "$full_log" \
        MAX_WALLCLOCK_SECONDS=3000 TTT_ENABLED=1 SLIDING_WINDOW_ENABLED=0 "$@" || rc=$?
    echo "[stack] full $name exit=$rc"
    grep -E "Total submission size|quantized_sliding_window val_loss|quantized_ttt_lora val_loss" "$full_log" | tail -6
    touch "$LOGDIR/${name}.done"
}

print_board() {
    echo "[stack] === results ==="
    printf "%-30s %12s %12s %14s\n" name sw_bpb ttt_bpb total_bytes
    for f in "$LOGDIR"/*.log; do
        [ -f "$f" ] || continue
        local base
        base=$(basename "$f" .log)
        case "$base" in
            *_scout) continue ;;
        esac
        local sw ttt size
        sw=$(grep "quantized_sliding_window val_loss" "$f" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
        ttt=$(grep "quantized_ttt_lora val_loss" "$f" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
        size=$(grep -oE 'Total submission size quantized\+[a-z]+: [0-9]+ bytes' "$f" | tail -1 | grep -oE '[0-9]+' | tail -1)
        printf "%-30s %12s %12s %14s\n" "$base" "${sw:--}" "${ttt:--}" "${size:--}"
    done | sort -k3
}

wait_for_queue
pack_entrypoint

# ---------------- variants ----------------
# Phase A: stacked control
run_variant stack_control

# Phase B: clip late/loop tightening (single knob, ≥0.85)
run_variant stack_late085  CLIP_MULT_LATE=0.85
run_variant stack_late09   CLIP_MULT_LATE=0.9
run_variant stack_loop085  CLIP_MULT_LOOP=0.85
run_variant stack_loop09   CLIP_MULT_LOOP=0.9

# Phase C: matrix clip sigmas
run_variant stack_sigmas13 MATRIX_CLIP_SIGMAS=13
run_variant stack_sigmas15 MATRIX_CLIP_SIGMAS=15

# Phase D: hessian clip lambda
run_variant stack_hcl01    HESSIAN_CLIP_LAMBDA=0.1
run_variant stack_hcl02    HESSIAN_CLIP_LAMBDA=0.2
run_variant stack_hcl045   HESSIAN_CLIP_LAMBDA=0.45

# Phase E: clip early extreme
run_variant stack_early2   CLIP_MULT_EARLY=2.0

print_board
echo "[stack] all done $(date -Iseconds)"
