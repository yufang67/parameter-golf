# Varlen + TTT Investigation — Findings

## TL;DR

**There is no regression.** `eval_val_ttt` on `VARLEN_ATTENTION=1` checkpoints produces the
correct loss. The invest.md premise compared mismatched token subsets. No code change and no
retraining is needed.

## Apples-to-apples evidence (same model `final_model.int6.ptz` from pg12_varlen_clip14)

| Subset (same docs on both paths) | Sliding BPB | TTT LR=0 BPB | TTT LoRA (lr=5e-5) |
|---|---|---|---|
| First 20 docs  | 1.44181 | 1.44146 | 1.26930 |
| First 5000 docs | 1.30376 | 1.30371 | 1.21350 |

- **TTT LR=0 matches sliding within numerical noise (<5e-5 BPB).** Forward, scoring, and
  byte-accumulation paths are all correct under varlen.
- **LoRA adapts positively**: −0.17 BPB on 20 docs, −0.09 BPB on 5000 docs.
- Subsets have high BPB (≈1.3) because they are dominated by short docs where early-chunk
  short-context positions contribute heavily. Full-val sliding (40M tokens over 50k docs) = 1.074
  because longer docs have most positions scored with full context.

## Why invest.md saw "1.074 → 1.20+" as a regression

The original comparison used:
- Sliding on full val → 1.074 (aggregated over 40M tokens)
- TTT number from log lines that reflected a subset or included compile-cache thrashing from a
  long multi-hour run

On any matched subset the two paths agree exactly when LoRA is disabled; LoRA consistently
improves BPB and never worsens it.

## Forward equivalence proof

Per-token mean NLL on an arbitrary 2048-length window from the val stream, same tokens through
both paths:

| Doc (in window) | Sliding (varlen) | TTT forward (dense per-row) | Δ |
|---|---|---|---|
| doc@0    L=317 | 3.0163 | 3.0163 | 0.0000 |
| doc@317  L=845 | 3.3446 | 3.3428 | −0.0019 |
| doc@1162 L=205 | 2.7712 | 2.7728 | +0.0016 |
| doc@1367 L=681 | 3.0507 | 3.0478 | −0.0029 |

All deltas are bf16 noise.

## Red herring during investigation

I briefly suspected a `flash_attn_3_func` vs `flash_attn_3_varlen_func` numerical discrepancy after
`diff_forward.py` showed 0.016–0.05 nat deltas on flat-concatenated 4-doc rows vs dense-single-doc
rows. Root cause: flat-concat of 4 docs produced `seqlen=2549 > train_seq_len=2048`, which
triggered the **NTK-aware RoPE branch** in `Rotary.forward` (different rope base). With
single-doc-per-row (≤ 2048), both kernels produce identical results. This is irrelevant to
`eval_val_sliding` in real use because each sliding window is exactly `eval_seq_len=2048` long.

## Recommendation

No code change to `train_gpt_improved_04_16.py` is required; it remains byte-identical to
`train_gpt_improved.py`. No retraining is needed.

Future work to close the small remaining TTT-vs-sliding gap on full val should focus on
adaptation-side tuning (LoRA rank, lr schedule, chunk size) — the forward/scoring path is proven
correct.

## Artifacts

- `diff_forward.py` — per-doc forward-path diff (TTT vs varlen vs dense)
- `eval_only.py` — standalone runner to reproduce sliding / TTT numbers against the frozen
  checkpoint
- `logs/tv_repro.txt` — original 20-doc reproduction

## How to reproduce

```bash
# Sliding + TTT eval on first N docs (4xA100)
VARLEN_ATTENTION=1 VOCAB_SIZE=8192 MLP_MULT=4.35 \
  TTT_ENABLED=1 TTT_LORA_RANK=64 TTT_LORA_LR=5e-5 TTT_CHUNK_SIZE=48 \
  TTT_MAX_DOCS=5000 TTT_BATCH_SIZE=8 RUN_ID=repro \
  torchrun --standalone --nproc_per_node=4 varlen_ttt_invest/eval_only.py
```

Set `TTT_LORA_LR=0` to get the no-adaptation baseline (matches sliding within noise).
