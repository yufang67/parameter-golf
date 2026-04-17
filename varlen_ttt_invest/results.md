# varlen_ttt_invest — results

All runs reuse `final_model.int6.ptz` from the prior varlen-trained checkpoint
(`exp106`-style config: num_layers=11, model_dim=512, mlp_mult=4.35, VARLEN_ATTENTION=1),
via a new `SKIP_TRAINING=1` flag added to `train_gpt_improved_04_16.py`.
`TTT_MAX_DOCS=50`, 4×A100. TTT defaults: rank=96, lora_lr=1e-4, chunk=64, k/mlp/o LoRA all enabled, weight_decay=0.2.

## Results table

| Run | Change vs baseline | quant | sw | ttt_lora | Δ vs sw |
|---|---|---:|---:|---:|---:|
| tv01_baseline | — (default TTT_LORA_LR=1e-4, rank=96, all LoRAs) | 1.08123 | **1.07407** | 1.20776 | **+0.1337** |
| tv02_force_dense | FORCE_DENSE_EVAL=1 (flip use_varlen off on eval model) | — | — | 1.20768 | +0.1336 |
| tv03_ttt_no_compile | TTT_COMPILE_DISABLE=1 | — | — | 1.20779 | +0.1337 |
| tv04_ttt_pad_bos | EVAL_PAD_TOKEN=1 (BOS_ID instead of 0) | — | — | 1.20782 | +0.1338 |
| tv05_ttt_lora_lr0 | **TTT_LORA_LR=0 (zero-LoRA probe)** | — | — | **1.33587** | **+0.2618** |
| tv06_ttt_lora_lr1e5 | TTT_LORA_LR=1e-5 | — | — | 1.28693 | +0.2129 |
| tv07_ttt_lora_lr5e6 | TTT_LORA_LR=5e-6 | — | — | 1.30973 | +0.2357 |
| tv08_ttt_qv_only | k_lora=0, mlp_lora=0, o_lora=0, rank=32, lr=5e-5, β2=0.999 | — | — | 1.28039 | +0.2063 |
| tv09_ttt_rank8 | TTT_LORA_RANK=8 | — | — | 1.25862 | +0.1846 |
| tv10_force_dense_lr0 | FORCE_DENSE_EVAL=1 + TTT_LORA_LR=0 | — | — | **1.33589** | **+0.2618** |

Reference (same checkpoint, `tv01_baseline`): quant=1.08123, sw=1.07407.
The "target" for a working TTT would be ≤ 1.071.

## Root-cause analysis

### Hypotheses H0 (shared eval-preamble) — **RULED OUT**

Every H0 sub-hypothesis is falsified:

1. **`deserialize()` drops `use_varlen`** — *false*. H0 probe on every run:
   ```
   [PROBE] eval_model.use_varlen = True
   [PROBE] per-block attn.use_varlen = [True, True, True, True, True, True, True, True, True, True, True]
   ```
   All 11 blocks correctly inherit `use_varlen=True` after `deserialize()`, because `GPT.__init__` re-runs and propagates the flag (lines 1047-1049).

2. **Varlen kernel corrupts eval** — *false*. `tv02_force_dense` (use_varlen=False on eval model → every block runs dense `flash_attn_3_func`) gives ttt_lora=1.20768, **identical** to the varlen-kernel baseline (1.20776). The varlen kernel is not the problem.

3. **Pad token mismatch (0 vs BOS=1)** — *false*. `tv04_ttt_pad_bos` (`EVAL_PAD_TOKEN=1`) gives 1.20782, indistinguishable from baseline. Causal attention means pads at row-end don't leak into scored positions.

4. **torch.compile caches wrong trace** — *false*. `tv03_ttt_no_compile` (compile fully disabled on `forward_ttt`) gives 1.20779, identical.

5. **`cu_seqlens` off-by-one / bucket mismatch** — not actually in play for TTT path: `forward_ttt` / `_block_with_lora` never call `flash_attn_3_varlen_func` and never build `cu_seqlens`. This is visible in `_block_with_lora` L1185: `y = flash_attn_3_func(q, k, v, causal=True, window_size=window)` — no varlen branch at all.

### Hypothesis H1 (per-path forward mismatch in `_block_with_lora`/`forward_ttt`) — **CONFIRMED**

The decisive diagnostic is `tv05_ttt_lora_lr0` (zero-LoRA probe):

> With `TTT_LORA_LR=0`, the LoRA optimizer never updates. `BatchedLinearLoRA.reset()` sets `B=0` at every doc boundary, so every LoRA adapter output is exactly `(x @ A.T) @ 0 = 0`. In this configuration `forward_ttt(x, y, zero_lora)` **should be bitwise-equivalent** (modulo bf16 reduction order) to the `forward_logits(x)` path used by `eval_val_sliding`.

> Measured: **zero-LoRA ttt_bpb = 1.33587** vs sliding-window bpb = **1.07407**. That is a **+0.262 BPB gap on a supposedly-identical forward pass.**

`tv10_force_dense_lr0` (dense + zero-LoRA) reports 1.33589 — same gap, confirming the bug is not varlen-specific and lives on the TTT forward path itself.

### LR sweep is consistent with H1

With the forward path already +0.26 worse at zero LoRA, higher `TTT_LORA_LR` lets LoRA compensate for some of the systematic error — but cannot fully recover:

```
lr=0     → 1.33587
lr=5e-6  → 1.30973
lr=1e-5  → 1.28693
lr=1e-4  → 1.20776   ← "optimal" corruption-compensator, still +0.134 worse than sw
```

This is exactly the "LoRA drifting in a direction that hurts rather than helps" pattern predicted by invest.md (line 114) — except the drift isn't hurting, the *forward-pass floor* is hurting and LoRA is half-heroically undoing it.

### Hypothesis H2 (TTT hyperparameters) — **irrelevant until H1 is fixed**

`tv08_ttt_qv_only` (match exp106's minimal-LoRA config: Q/V only, rank=32, lr=5e-5, β2=0.999) → 1.28039. `tv09_ttt_rank8` (rank=8) → 1.25862. Both worse than the baseline (rank=96, all LoRAs) because larger/richer LoRA space is better at repairing the broken base forward. H2 tuning is downstream of the real bug.

## Where the bug lives

`_block_with_lora` (L1167-1224) and `forward_ttt` (L1225-1270) are a manual re-implementation of `GPT.forward_logits` (L1130-1166) + `Block.forward` (L938-954) + `CausalSelfAttention.forward` (L787-824). At zero LoRA they must be mathematically equivalent but are not. The **+0.26 BPB floor at lr=0** localises the bug inside this manual re-implementation.

Candidates worth auditing line-by-line (unchecked after this batch):

- **Skip connections**: `forward_ttt` L1247-1257 pops `skips` at `skip_idx`; `forward_logits` L1149-1158 uses the same pattern. The `_add_loop_emb(x, i, lpc)` is called in both, but the `lpc` dict shared between encoder and decoder passes *might* produce different pass-indices between the two paths (both use `lpc` but `slot` is separate — verify that `lpc` is not mutated differently). 
- **`num_skip_weights`** vs `slot` counter: in the decoder loop, `skip_idx` counts from 0, `slot` counts continuing from encoder. A subtle off-by-one between `skip_idx`-indexed `self.skip_weights` and `slot`-indexed `lora.*_loras` is plausible but does not affect zero-LoRA output.
- **`lora.lm_head_lora(x)` addition at L1265** is after `self.logit_softcap * tanh(logits/softcap)` in `forward_logits` but **before** softcap in `forward_ttt`:
  ```python
  # forward_logits (L1165-1166):
  logits_proj = self.lm_head(x)
  return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
  # forward_ttt (L1263-1266):
  logits = self.lm_head(x)
  logits = logits + lora.lm_head_lora(x)         # zero at lr=0
  logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
  ```
  These are equivalent at zero LoRA. Not the bug.
- **`final_norm` / `head_proj` / `tie_embeddings`** — both paths use the same sequence. Not the bug.

The most productive next step is to **turn on the built-in sanity check** (`TTT_SANITY_CHECK=1`, L2559-2626), which computes `forward_logits` vs zero-LoRA `forward_ttt` on the same tokens and prints per-position `|delta|` statistics, plus a per-block delta. That will point a finger at a specific layer. This is out of scope for the 10-run budget for this investigation but is the next clear step.

## ⚠️ 2026-04-17 update — H1 conclusion below is FALSIFIED

Two follow-up batches done on 2026-04-17 (logs in
`/root/parameter-golf/logs/sanity_check_04_17_stdout.txt`,
`logs/no_lora_probe_*_04_17_stdout.txt`,
`logs/warmup_w*_04_17_stdout.txt`,
`logs/fair_*_warmup1024_04_17_stdout.txt`) overturn the H1-confirmed claim:

### Diagnostic 1 — `TTT_SANITY_CHECK=1 TTT_LAYER_DEBUG=1`

For a 317-token `prefix` probe and a 2048-token `tail_window` probe, the
per-layer comparison between `forward_logits` and zero-LoRA `forward_ttt`
gives **mean_abs = max_abs = 0.000000e+00 at every one of the 19 states**
(embed → 11 blocks → final_norm → logits). `_block_with_lora` is
mathematically equivalent to `Block.forward`. The "+0.26 BPB at zero LoRA
is a forward-path bug" hypothesis is **bitwise falsified**.

### Diagnostic 2 — `TTT_NO_LORA_PROBE` (ad-hoc flag, since reverted)

Patched `eval_val_ttt` to call `forward_logits(x)` (no LoRA, the trained
forward path used by sliding-window) inside the TTT chunk loop, reusing
TTT's windowing and `_accumulate_bpb`:

| `TTT_MAX_DOCS` | `TTT_CHUNK_SIZE` | val_bpb (no-LoRA probe) |
|---|---|---|
| 50   | 64   | 1.33718 |
| 200  | 64   | 1.29099 |
| 1000 | 64   | 1.29086 |
| 5000 | 64   | 1.31149 |
| 1000 | 256  | 1.29081 |
| 1000 | 1024 | 1.25751 |
| 1000 | 2048 | 1.24523 |

All numbers are far above the 1.07407 sliding-window baseline, even though
the model forward is the trained one. The gap therefore lives in the
**eval protocol itself** (windowing + which positions are scored + bpb
denominator construction), not in `forward_ttt` / `_block_with_lora`.

### Diagnostic 3 — `TTT_SCORE_WARMUP` (ad-hoc flag, since reverted)

Tried to align TTT's scored-position set with sliding-window's by skipping
the first `N` positions of every chunk's scored region (matching
`eval_val_sliding`'s `s = context_size = seq_len - stride` skip). With
`TTT_MAX_DOCS=1000`, `TTT_CHUNK_SIZE=64`, no-LoRA probe:

| `TTT_SCORE_WARMUP` | val_bpb |
|---|---|
|    0 | 1.29086 |
|   64 | 1.29006 |
|  256 | 1.33529 |
|  512 | 1.40500 |
| 1024 | 1.50072 |

**Counter-intuitive:** scoring positions with *more* left context produces
*higher* bpb, not lower. The most likely explanation is that this is a
**bytes-per-token denominator artifact**: late-document tokens in the first
~1000 FineWeb docs are dominated by short / low-byte pieces (URLs,
references, formulaic footers), so the bpb denominator
(`base_bytes_lut[y]`) shrinks faster than the cross-entropy numerator,
inflating bpb. Sliding-window's full-shard scoring averages this out.

### Diagnostic 4 — fair-comparison run

`TTT_LORA_LR=1e-4 TTT_SCORE_WARMUP=1024 TTT_MAX_DOCS=1000` →
val_bpb = **1.22775**. Real LoRA is still ~0.07 BPB above the
no-LoRA-probe-with-same-warmup floor of 1.50, i.e., LoRA *does* improve
on its own protocol baseline (−0.273 BPB), but the protocol-noise
dwarfs the LoRA gain at this sample size.

### Real root cause

`eval_val_ttt` and `eval_val_sliding` measure *different quantities*:

1. **Scored-position set differs.** Sliding-window only scores the trailing
   `stride` tokens of each window. TTT scores every position of every chunk
   window, including cold-start positions with very short left context.
2. **Token sample differs.** TTT (with `TTT_MAX_DOCS≪all`) only scores the
   first N documents; sliding-window scores all 10 M tokens. The first N
   FineWeb docs have skewed bytes-per-token statistics, especially in their
   tails, which moves the bpb denominator independently of the model.
3. **`cu_seqlens` handling differs.** `eval_val_sliding` (varlen) builds
   `cu_seqlens` from BOS positions across the packed batch and forces a
   boundary at offset 0 of every window. `forward_ttt` auto-synthesizes
   `cu_seqlens=[0, sl_in]` per row. Late-chunk windows that start mid-doc
   therefore look identical between the two protocols, but early-chunk
   windows differ.

LoRA TTT is **genuinely working**: at `TTT_LORA_LR=1e-4` it pulls the score
from the protocol-floor (~1.336 with zero LoRA, 50 docs) down to 1.208 — a
real −0.128 BPB improvement *on its own protocol*. The "regression vs sw"
this batch reported was apples-to-oranges.

### Action

- Don't rewrite `_block_with_lora` (it's not broken).
- The TTT-vs-sliding-window comparison cannot be fixed by skipping
  positions alone — it also requires aligning the scored-token set
  (use the same docs, same number of tokens) AND building cu_seqlens the
  way the trained model expects on validation data.
- The **next** investigation should compare TTT-with-LoRA to a
  **TTT-without-LoRA control run on the exact same scored-token set**.
  That is the only meaningful TTT-improvement number. The
  `TTT_NO_LORA_PROBE` flag (used here, then reverted) is the right
  mechanism — it should be re-introduced as a permanent diagnostic flag,
  paired with a "score-only-from-doc-end" mode, to make
  `Δ(LoRA) := bpb_TTT_lora − bpb_TTT_no_lora` a meaningful number.
- The headline metric for the leaderboard is whatever the official eval
  uses — `eval_val_sliding`. Improving TTT-vs-sw requires running TTT on
  the same scored-token set as sliding-window (large refactor of
  `eval_val_ttt`), not a `_block_with_lora` rewrite.
- Patches `TTT_NO_LORA_PROBE` and `TTT_SCORE_WARMUP` were reverted from
  `train_gpt_improved_04_16.py` after this batch; raw logs preserved
  under `/root/parameter-golf/logs/` (filenames listed at top of this
  section).

The conclusion below ("forward-path mismatch") is left in place for
historical context but should be considered overturned.

---

## ✅ 2026-04-18 update #2 — ROOT CAUSE FOUND: RoPE base mismatch (FIXED)

After falsifying the gated-attention hypothesis (above), tested the next
candidate: **RoPE positional encoding mismatch between training and TTT eval**.

`Rotary.forward(seq_len)` switches between vanilla cached cos/sin and
NTK-aware-rescaled cos/sin based on whether `seq_len > train_seq_len=2048`:

- Training (varlen): rows are ~98k tokens → NTK path.
- Sliding eval (varlen): rows are ~65k tokens → NTK path (matches training).
- TTT eval: rows are exactly 2048 tokens → vanilla cached path → **mismatch**.

### Bidirectional confirmation

| Run | RoPE base | val_bpb |
|---|---|---:|
| Sliding (default NTK) | scaled | **1.0742** |
| Sliding `ROPE_FORCE_BASE_SEQLEN=2048` | vanilla | 1.2887 |
| TTT lora (default vanilla) | vanilla | 1.1920 |
| TTT lora `ROPE_FORCE_BASE_SEQLEN=65536` | scaled | 1.0786 |
| **TTT lora auto-fix** | scaled (98304) | **1.0778** |

Forcing the wrong RoPE base into either protocol moves it onto the other's
curve. **Gap collapsed from +0.114 BPB to +0.004 BPB.**

### Why PR #1530 doesn't have this

PR #1530 has no NTK rescale — its Rotary always uses the same base, regardless
of seq_len. Their training and TTT see identical RoPE. We added NTK rescaling
(probably as a long-context win during training) but never propagated it to
TTT eval.

### Fix shipped

Added `ttt_rope_base_seqlen` (env `TTT_ROPE_BASE_SEQLEN`); auto-derived at TTT
eval entry from `train_batch_tokens / (world × grad_accum)`. Set via
`ROPE_FORCE_BASE_SEQLEN` env in `Rotary.forward` for TTT duration only; sliding
unaffected. Logs: `ttt_ropebase65k_d2000_04_17.txt`,
`sw_ropevanilla2_04_17.txt`, `ttt_ropefix_auto_d2000_04_17.txt`.

This overturns both the prior "scoring-protocol" conclusion and the original
"forward-path bug" conclusion. The protocols *do* differ, but the dominant
+0.11 BPB came from RoPE; remaining +0.004 is noise.

---

## ⚠️ 2026-04-18 update — gated-attention hypothesis FALSIFIED

Tested whether per-head sigmoid `gated_attention` (which we have, PR #1530 does not)
explains why our zero-LoRA TTT floor is 1.29 while sliding is 1.07.

Patch (`DISABLE_GATED_ATTN_EVAL=1`) bypasses `attn._disable_gate=True` on every
block in both `_block_with_lora` and `CausalSelfAttention.forward` at eval.

| Run | gates | val_bpb |
|---|---|---:|
| sliding | ON  | **1.0742** |
| sliding | OFF | 1.5796 |
| TTT lora (2000 docs) | ON  | 1.1920 |
| TTT lora (2000 docs) | OFF | 1.5602 |

Disabling gates regresses sliding by +0.51 and TTT by +0.37. The gates are
universally essential — removing them does **not** narrow the TTT-vs-sliding gap;
it widens it (sliding takes the bigger hit). Hypothesis falsified.

Notably, with gates OFF both protocols converge to ~1.56-1.58. With gates ON,
sliding (1.07) and TTT (1.19) diverge by 0.12. The gates *amplify* the protocol
difference rather than causing it.

Logs: `logs/ttt_nogate_d2000_04_17.txt`, `logs/sw_nogate_04_17.txt`.

### Remaining open hypotheses

1. **RoPE position distribution (most likely)** — In sliding eval, packed flat
   layout puts each doc at a varied row offset, so RoPE positions span 0..seq_len.
   In TTT, every doc lives in its own row at positions 0..doclen-1. Training also
   packs docs (mostly non-zero starts) so the position-0 distribution may be
   thinner. Earlier `TTT_ROPE_OFFSET` test (token-0 padding) was confounded by
   adding OOD attention context. A clean test requires a `position_offset`
   parameter on `Rotary.forward` that shifts cos/sin without padding tokens.
2. **Skip-gates / fused-RoPE / bigram-trigram path numerical drift between
   `_block_with_lora` and `Block.forward`** — already partially tested via
   bitwise sanity check (identical); but worth re-checking with full numerical
   `torch.allclose(atol=1e-7)` of *intermediate* tensors rather than just final
   logits.
3. **q_gain / logit_softcap** — both apply equally to both protocols, but worth
   ruling out by setting them at parity.

### Patches reverted
`DISABLE_GATED_ATTN_EVAL` and earlier `TTT_NO_LORA_PROBE` / `TTT_ROPE_OFFSET`
patches reverted from `train_gpt_improved_04_16.py`. Raw logs retained.

---

## Conclusion / one-line root cause

**Varlen + LoRA-TTT regression is not a varlen bug.** It is a forward-path mismatch inside `GPT.forward_ttt` / `GPT._block_with_lora` vs the trained forward (`GPT.forward_logits` / `Block.forward`), worth ≈ +0.26 BPB at zero LoRA. Varlen-trained checkpoints expose it more visibly (+0.134 with the default LoRA LR) because their sw baseline is tighter, but the same bug must exist on the non-varlen path too — it is simply masked there because non-varlen uses the separate `eval_val_ttt_fullparam` code path which re-uses the compiled `Block.forward` (no `_block_with_lora` at all). That matches invest.md's own observation (line 57-63) that "fullparam TTT works on non-varlen" — not because varlen breaks it, but because **fullparam TTT runs through `Block.forward`, while LoRA TTT runs through the hand-rolled `_block_with_lora` that has drifted from `Block.forward`.**

## Next steps (outside this 10-run batch)

1. Enable `TTT_SANITY_CHECK=1` to get per-doc, per-block delta and localise the diverging layer.
2. Diff `_block_with_lora` vs `Block.forward` + `CausalSelfAttention.forward` line-by-line for xsa / gating / skip_gates / resid_mix ordering, using the sanity-check output as a guide.
3. Alternative fix: rewrite `_block_with_lora` to call `block.forward(...)` with LoRA weights monkey-patched into the projection linears, instead of re-implementing the block.
4. Once zero-LoRA matches `forward_logits` exactly, re-run the LR sweep (H2) — `lora_lr=1e-4` may no longer be optimal.
