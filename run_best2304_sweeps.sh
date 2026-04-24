#!/bin/bash
set -u

cd /root/parameter-golf
source .venv/bin/activate

LOGDIR=logs/best2304_sweeps
mkdir -p "$LOGDIR"

SECTION="${SECTION:-clip_scout}"
SCRIPT_SRC="train_gpt_improved_04_23.py"
SCRIPT_PACKED="train_gpt.py"
TORCHRUN="/root/parameter-golf/.venv/bin/torchrun"

pack_entrypoint() {
    python3 pack_submission_file.py "$SCRIPT_SRC" "$SCRIPT_PACKED"
}

wait_for_queue() {
    while pgrep -f "torchrun --standalone --nproc_per_node=4 train_gpt" > /dev/null \
       || pgrep -f "torchrun --standalone --nproc_per_node=4 train_gpt_improved_04_23.py" > /dev/null; do
        sleep 30
    done
}

COMMON_BASE_ENV=(
  VARLEN_ATTENTION=1
  MATRIX_CLIP_SIGMAS=14
  MLP_MULT=4.35
  HESSIAN_CLIP_LAMBDA=0.0
  ENABLE_LOOPING_AT=0.5
  PARALLEL_RESIDUAL_START=7
  XSA_LAST_N=9
  GPTQ_CALIBRATION_BATCHES=128
  COMPRESS_ANS=1
  COMPRESS_BROTLI=0
  COMPRESS_LZMA=0
)

run_one() {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    local rc=0
    if [ -f "$LOGDIR/${name}.done" ]; then
        echo "[best2304] SKIP $name (already done)"
        return
    fi
    echo "[best2304] === $name === $(date -Iseconds)"
    {
        echo "===== CODE SNAPSHOT: ${SCRIPT_SRC} ====="
        echo "git_head: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "sha256:   $(sha256sum "$SCRIPT_SRC" | awk '{print $1}')"
        echo "===== BEGIN ${SCRIPT_SRC} ====="
        cat "$SCRIPT_SRC"
        echo "===== END ${SCRIPT_SRC} ====="
        echo
    } > "$log"
    env -i HOME="$HOME" PATH="$PATH" \
        RUN_ID="$name" \
        "${COMMON_BASE_ENV[@]}" \
        "$@" \
        "$TORCHRUN" --standalone --nproc_per_node=4 "$SCRIPT_PACKED" \
        >> "$log" 2>&1 || rc=$?
    echo "[best2304] $name exit=$rc"
    grep -E "Total submission size|quantized_sliding_window val_loss|quantized_ttt_lora val_loss|SIZE_ONLY=1" "$log" | tail -6
    touch "$LOGDIR/${name}.done"
}

print_size_board() {
    local prefix="$1"
    echo "[best2304] === ${SECTION} sizes ==="
    printf "%-32s %14s\n" name total_bytes
    for f in "$LOGDIR"/${prefix}*.log; do
        [ -f "$f" ] || continue
        local name
        local size
        name=$(basename "$f" .log)
        size=$(grep -oE 'Total submission size quantized\+[a-z]+: [0-9]+ bytes' "$f" | tail -1 | grep -oE '[0-9]+' | tail -1)
        printf "%-32s %14s\n" "$name" "${size:--}"
    done | sort -k2n
}

print_eval_board() {
    local prefix="$1"
    echo "[best2304] === ${SECTION} evals ==="
    printf "%-32s %10s %10s %14s\n" name sw_bpb ttt_bpb total_bytes
    for f in "$LOGDIR"/${prefix}*.log; do
        [ -f "$f" ] || continue
        local name
        local sw
        local ttt
        local size
        name=$(basename "$f" .log)
        sw=$(grep "quantized_sliding_window val_loss" "$f" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
        ttt=$(grep "quantized_ttt_lora val_loss" "$f" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
        size=$(grep -oE 'Total submission size quantized\+[a-z]+: [0-9]+ bytes' "$f" | tail -1 | grep -oE '[0-9]+' | tail -1)
        printf "%-32s %10s %10s %14s\n" "$name" "${sw:--}" "${ttt:--}" "${size:--}"
    done | sort -k3
}

run_clip_scout() {
    run_one "best2304_base_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60
    run_one "best2304_clip_late075_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 CLIP_MULT_LATE=0.75
    run_one "best2304_clip_loop075_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 CLIP_MULT_LOOP=0.75
    run_one "best2304_clip_lateloop075_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 CLIP_MULT_LATE=0.75 CLIP_MULT_LOOP=0.75
    run_one "best2304_clip_late05_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 CLIP_MULT_LATE=0.5
    run_one "best2304_clip_loop05_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 CLIP_MULT_LOOP=0.5
    run_one "best2304_clip_lateloop05_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 CLIP_MULT_LATE=0.5 CLIP_MULT_LOOP=0.5
    run_one "best2304_clip_lateloop075_h0_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 HESSIAN_CLIP_LAMBDA=0.0 CLIP_MULT_LATE=0.75 CLIP_MULT_LOOP=0.75
    run_one "best2304_clip_lateloop075_h015_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 HESSIAN_CLIP_LAMBDA=0.15 CLIP_MULT_LATE=0.75 CLIP_MULT_LOOP=0.75
    run_one "best2304_clip_lateloop075_h045_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 HESSIAN_CLIP_LAMBDA=0.45 CLIP_MULT_LATE=0.75 CLIP_MULT_LOOP=0.75
    print_size_board "best2304_clip_"
}

run_moe_scout() {
    run_one "best2304_moe79_e2_k1_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 MOE_ENABLED=1 MOE_LAYERS=7,9 MOE_NUM_EXPERTS=2 MOE_TOP_K=1
    run_one "best2304_moe79_e2_k2_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 MOE_ENABLED=1 MOE_LAYERS=7,9 MOE_NUM_EXPERTS=2 MOE_TOP_K=2
    run_one "best2304_moe79_e3_k1_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 MOE_ENABLED=1 MOE_LAYERS=7,9 MOE_NUM_EXPERTS=3 MOE_TOP_K=1
    run_one "best2304_moe79_e3_k2_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 MOE_ENABLED=1 MOE_LAYERS=7,9 MOE_NUM_EXPERTS=3 MOE_TOP_K=2
    run_one "best2304_moe9_e2_k1_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 MOE_ENABLED=1 MOE_LAYERS=9 MOE_NUM_EXPERTS=2 MOE_TOP_K=1
    run_one "best2304_moe579_e2_k1_size60" \
        TTT_ENABLED=0 SIZE_ONLY=1 MAX_WALLCLOCK_SECONDS=60 MOE_ENABLED=1 MOE_LAYERS=5,7,9 MOE_NUM_EXPERTS=2 MOE_TOP_K=1
    print_size_board "best2304_moe"
}

run_ntk_eval() {
    run_one "best2304_no_ntk_3000" \
        MAX_WALLCLOCK_SECONDS=3000 DISABLE_NTK_ROPE=1
    print_eval_board "best2304_no_ntk"
}

run_clip_full() {
    run_one "best2304_clip_late075_3000" \
        MAX_WALLCLOCK_SECONDS=3000 CLIP_MULT_LATE=0.75
    run_one "best2304_clip_loop075_3000" \
        MAX_WALLCLOCK_SECONDS=3000 CLIP_MULT_LOOP=0.75
    run_one "best2304_clip_lateloop075_3000" \
        MAX_WALLCLOCK_SECONDS=3000 CLIP_MULT_LATE=0.75 CLIP_MULT_LOOP=0.75
    print_eval_board "best2304_clip_"
}

run_section3() {
    for grp in EARLY LOOP MID LATE; do
        for v in 0.75 1.25 1.5; do
            tag="${v//./}"
            run_one "best2304_s3_clip${grp,,}${tag}" \
                MAX_WALLCLOCK_SECONDS=3000 TTT_ENABLED=1 CLIP_MULT_${grp}=$v
        done
    done
    print_eval_board "best2304_s3_clip"
}

run_section5() {
    for v in 11 9 5 3 0; do
        run_one "best2304_s5_pr${v}" \
            MAX_WALLCLOCK_SECONDS=3000 TTT_ENABLED=1 PARALLEL_RESIDUAL_START=$v
    done
    print_eval_board "best2304_s5_pr"
}

run_section6() {
    for v in 256 512 1024; do
        run_one "best2304_s6_wattn${v}" \
            MAX_WALLCLOCK_SECONDS=3000 TTT_ENABLED=1 WINDOW_ATTN_SIZE=$v
    done
    print_eval_board "best2304_s6_wattn"
}

wait_for_queue
pack_entrypoint

case "$SECTION" in
    section3)
        run_section3
        ;;
    section5)
        run_section5
        ;;
    section6)
        run_section6
        ;;
    program_s356)
        run_section3
        run_section5
        run_section6
        ;;
    clip_scout)
        run_clip_scout
        ;;
    moe_scout)
        run_moe_scout
        ;;
    ntk_eval)
        run_ntk_eval
        ;;
    clip_full)
        run_clip_full
        ;;
    all_scouts)
        run_clip_scout
        run_moe_scout
        ;;
    *)
        echo "Unknown SECTION=$SECTION"
        exit 1
        ;;
esac
