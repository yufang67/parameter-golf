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

## Conclusion / one-line root cause

**Varlen + LoRA-TTT regression is not a varlen bug.** It is a forward-path mismatch inside `GPT.forward_ttt` / `GPT._block_with_lora` vs the trained forward (`GPT.forward_logits` / `Block.forward`), worth ≈ +0.26 BPB at zero LoRA. Varlen-trained checkpoints expose it more visibly (+0.134 with the default LoRA LR) because their sw baseline is tighter, but the same bug must exist on the non-varlen path too — it is simply masked there because non-varlen uses the separate `eval_val_ttt_fullparam` code path which re-uses the compiled `Block.forward` (no `_block_with_lora` at all). That matches invest.md's own observation (line 57-63) that "fullparam TTT works on non-varlen" — not because varlen breaks it, but because **fullparam TTT runs through `Block.forward`, while LoRA TTT runs through the hand-rolled `_block_with_lora` that has drifted from `Block.forward`.**

## Next steps (outside this 10-run batch)

1. Enable `TTT_SANITY_CHECK=1` to get per-doc, per-block delta and localise the diverging layer.
2. Diff `_block_with_lora` vs `Block.forward` + `CausalSelfAttention.forward` line-by-line for xsa / gating / skip_gates / resid_mix ordering, using the sanity-check output as a guide.
3. Alternative fix: rewrite `_block_with_lora` to call `block.forward(...)` with LoRA weights monkey-patched into the projection linears, instead of re-implementing the block.
4. Once zero-LoRA matches `forward_logits` exactly, re-run the LR sweep (H2) — `lora_lr=1e-4` may no longer be optimal.
