# varlen_ttt_invest — README

Investigation per `invest.md`: why does TTT-LoRA regress on varlen-trained checkpoints?

## What's in here

- `results.md` — full results table + root-cause analysis. **Start here.**
- `commands.txt` — launch commands for all 10 runs (tv01..tv10).
- `logs/` — raw tee'd stdout per run (tv00_smoke = tv01_baseline).
- `patches/train_gpt_improved_04_16.original.py` — pristine copy of the file before the investigation edits.
- `run_batch.sh` — sequential runner for tv02-tv10.

## One-line root cause

**Not a varlen bug.** `GPT.forward_ttt` / `GPT._block_with_lora` (the hand-rolled LoRA-aware forward) is ≈ +0.26 BPB worse than `GPT.forward_logits` / `Block.forward` *even at `TTT_LORA_LR=0`* (zero-LoRA probe, tv05 & tv10). Varlen only makes the regression visible because its sliding-window baseline is tighter; the same bug exists on the non-varlen path but is masked because non-varlen TTT uses the fullparam code path (`eval_val_ttt_fullparam`), which calls the real `Block.forward`, not `_block_with_lora`.

Full analysis: `results.md`.

## Scope of this batch (what this batch does NOT do)

Per the user's instruction, this batch runs 10 **eval-only diagnostics** reusing `final_model.int6.ptz` (no training). It therefore:

- **confirms the regression reproduces** on a reusable checkpoint (tv01_baseline = 1.208 ttt_lora vs 1.074 sw),
- **rules out** H0 sub-hypotheses (varlen-kernel, compile, pad-token, deserialize-drops-varlen),
- **confirms** H1 (forward-path mismatch) via the zero-LoRA probe,
- does **not** produce a fix patch — the next step is to run `TTT_SANITY_CHECK=1` to localise the diverging layer and then rewrite `_block_with_lora` to delegate to `Block.forward` with monkey-patched projection weights.

## Upstream PR survey (step 0)

Not performed — no network access in this environment. A follow-up investigation should file this as step 0 before any fix PR, as invest.md §"Step 0" instructs.

## Code changes to `train_gpt_improved_04_16.py`

Diff vs `patches/train_gpt_improved_04_16.original.py`:

1. Added env flags `SKIP_TRAINING`, `DEBUG_VARLEN_PROBE`, `EVAL_PAD_TOKEN`, `TTT_COMPILE_DYNAMIC`, `TTT_COMPILE_DISABLE`, `FORCE_DENSE_EVAL`, `RUN_TTT_ONLY`, `RUN_SW_ONLY` (Hyperparameters block).
2. `train_and_eval`: gate training/serialize on `not skip_training`; add H0 probe block after `deserialize()`; respect `FORCE_DENSE_EVAL` / `RUN_TTT_ONLY` / `RUN_SW_ONLY`.
3. `eval_val_ttt`: honor `EVAL_PAD_TOKEN` (was hard-coded `0`), honor `TTT_COMPILE_DISABLE` / `TTT_COMPILE_DYNAMIC`, emit `[PROBE:eval_val_ttt]` line.

None of these changes affect the training path or the non-TTT eval paths.
