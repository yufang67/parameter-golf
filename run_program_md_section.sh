#!/bin/bash
# program.md Section 1 of 6: XSA_LAST_N sweep.
# Waits for phase-6/7 + any in-flight torchrun to finish, then runs the section.
set -u
cd /root/parameter-golf
source .venv/bin/activate

LOGDIR=logs/program_md
mkdir -p "$LOGDIR"

SECTION="${SECTION:-1}"
echo "[pgm-s${SECTION}] waiting for prior queue to drain..."
while pgrep -f "run_pg12_ttt_phase" > /dev/null \
   || pgrep -f "train_gpt_improved_04_16.py" > /dev/null; do
    sleep 30
done
echo "[pgm-s${SECTION}] queue clear. Starting section ${SECTION}."

COMMON_TRAIN_ENV=(
  MATRIX_CLIP_SIGMAS=14
  MLP_MULT=4.35
  GPTQ_CALIBRATION_BATCHES=128
  VARLEN_ATTENTION=1
  TTT_ENABLED=1
  MAX_WALLCLOCK_SECONDS=3600
  TTT_LORA_RANK=48
  TTT_PHASES=3
  HESSIAN_CLIP_LAMBDA=0.3
)

train_then_eval() {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    local ckpt="${name}.int6.ptz"
    if [ -f "$LOGDIR/${name}.done" ]; then
        echo "[pgm] SKIP $name (already done)"
        return
    fi
    echo "[pgm] === $name === (train+eval)"
    {
        echo "===== CODE SNAPSHOT: train_gpt_improved_04_16.py ====="
        echo "git_head: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "sha256:   $(sha256sum train_gpt_improved_04_16.py | awk '{print $1}')"
        echo "===== BEGIN train_gpt_improved_04_16.py ====="
        cat train_gpt_improved_04_16.py
        echo "===== END train_gpt_improved_04_16.py ====="
        echo
    } > "$log"
    env -i HOME=$HOME PATH=$PATH \
        RUN_ID="$name" \
        "${COMMON_TRAIN_ENV[@]}" \
        "$@" \
        torchrun --standalone --nproc_per_node=4 train_gpt_improved_04_16.py \
        >> "$log" 2>&1
    echo "[pgm] $name exit=$?"
    grep -E "quantized(_sliding_window|_ttt_lora)? val_loss" "$log" | tail -3
    if [ -f final_model.int6.ptz ]; then
        cp final_model.int6.ptz "$LOGDIR/$ckpt"
    fi
    touch "$LOGDIR/${name}.done"
}

print_section_leaderboard() {
    local prefix="$1"
    echo "[pgm-s${SECTION}] === SECTION ${SECTION} LEADERBOARD ==="
    echo "name                                       dense    sw       ttt"
    for f in "$LOGDIR"/${prefix}*.log; do
        [ -f "$f" ] || continue
        name=$(basename "$f" .log)
        d=$(grep "^quantized val_loss"               "$f" | tail -1 | grep -oP "val_bpb:\K[0-9.]+")
        s=$(grep "quantized_sliding_window val_loss" "$f" | tail -1 | grep -oP "val_bpb:\K[0-9.]+")
        t=$(grep "quantized_ttt_lora val_loss"       "$f" | tail -1 | grep -oP "val_bpb:\K[0-9.]+")
        printf "%-42s %s %s %s\n" "$name" "${d:--}" "${s:--}" "${t:--}"
    done | sort -k4
}

case "$SECTION" in
1)
    # XSA_LAST_N — current default 11; sweep 0,4,7,9
    for v in 0 4 7 9; do
        train_then_eval "pgm_xsa${v}"  XSA_LAST_N=$v
    done
    print_section_leaderboard "pgm_xsa"
    ;;
2)
    # LOOP_EMBEDDINGS — current default 0
    train_then_eval "pgm_loopemb1"           LOOP_EMBEDDINGS=1
    train_then_eval "pgm_loopemb1_loops3"    LOOP_EMBEDDINGS=1 NUM_LOOPS=3
    train_then_eval "pgm_loopemb1_loops4"    LOOP_EMBEDDINGS=1 NUM_LOOPS=4
    print_section_leaderboard "pgm_loopemb"
    ;;
3)
    # CLIP_MULT_* — per-group GPTQ multipliers
    for grp in EARLY LOOP MID LATE; do
        for v in 0.75 1.25 1.5; do
            tag="${v//./}"
            train_then_eval "pgm_clip${grp,,}${tag}"  CLIP_MULT_${grp}=$v
        done
    done
    print_section_leaderboard "pgm_clip"
    ;;
4)
    # ENABLE_LOOPING_AT — current default 0.35
    for v in 0.0 0.15 0.5 0.7; do
        name="pgm_loopat${v//./_}"
        train_then_eval "$name"  ENABLE_LOOPING_AT=$v
    done
    print_section_leaderboard "pgm_loopat"
    ;;
5)
    # PARALLEL_RESIDUAL_START — current default 7
    for v in 11 9 5 3 0; do
        train_then_eval "pgm_pr${v}"  PARALLEL_RESIDUAL_START=$v
    done
    print_section_leaderboard "pgm_pr"
    ;;
6)
    # WINDOW_ATTN_SIZE — current default 0
    for v in 256 512 1024; do
        train_then_eval "pgm_wattn${v}"  WINDOW_ATTN_SIZE=$v
    done
    print_section_leaderboard "pgm_wattn"
    ;;
7)
    # CLIP_MULT stacking + sub-0.75 follow-up to section 3.
    # Section 3 found every group prefers 0.75; LATE and LOOP are most sensitive.
    train_then_eval "pgm_clip_lateloop075"   CLIP_MULT_LATE=0.75 CLIP_MULT_LOOP=0.75
    train_then_eval "pgm_clip_late05"        CLIP_MULT_LATE=0.5
    train_then_eval "pgm_clip_late06"        CLIP_MULT_LATE=0.6
    train_then_eval "pgm_clip_loop05"        CLIP_MULT_LOOP=0.5
    train_then_eval "pgm_clip_loop06"        CLIP_MULT_LOOP=0.6
    train_then_eval "pgm_clip_lateloop05"    CLIP_MULT_LATE=0.5  CLIP_MULT_LOOP=0.5
    train_then_eval "pgm_clip_all075"        CLIP_MULT_EARLY=0.75 CLIP_MULT_LOOP=0.75 CLIP_MULT_MID=0.75 CLIP_MULT_LATE=0.75
    print_section_leaderboard "pgm_clip_"
    ;;
8)
    # Push sub-0.5 on the winners + full 4-way stacks.
    # Section 7 found LATE=0.5 + LOOP=0.5 -> 1.06372 TTT (record).
    # Defaults already bumped to LATE=0.5, LOOP=0.5; explicit env overrides kept for clarity.
    train_then_eval "pgm_clip_lateloop035"   CLIP_MULT_LATE=0.35 CLIP_MULT_LOOP=0.35
    train_then_eval "pgm_clip_late035"       CLIP_MULT_LATE=0.35 CLIP_MULT_LOOP=0.5
    train_then_eval "pgm_clip_loop035"       CLIP_MULT_LATE=0.5  CLIP_MULT_LOOP=0.35
    train_then_eval "pgm_clip_all05"         CLIP_MULT_EARLY=0.5 CLIP_MULT_LOOP=0.5 CLIP_MULT_MID=0.5 CLIP_MULT_LATE=0.5
    train_then_eval "pgm_clip_4way_mix"      CLIP_MULT_EARLY=0.75 CLIP_MULT_LOOP=0.5 CLIP_MULT_MID=0.75 CLIP_MULT_LATE=0.5
    print_section_leaderboard "pgm_clip_"
    ;;
*)
    echo "Unknown SECTION=$SECTION"; exit 1;;
esac

echo "[pgm-s${SECTION}] SECTION ${SECTION} COMPLETE"
