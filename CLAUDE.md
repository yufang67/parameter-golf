# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Parameter Golf** is an OpenAI-sponsored competition to train the best language model that fits in a **16MB artifact** (code + compressed model) and trains in **under 10 minutes on 8xH100s**. Models are evaluated on FineWeb validation set compression using **bits-per-byte (BPB)**, which is tokenizer-agnostic. The challenge runs March 18 – April 30, 2026.

## Key Commands

### Data Download
```bash
# Download cached FineWeb with 1024-token SentencePiece vocabulary
python3 data/cached_challenge_fineweb.py --variant sp1024

# Smaller subset for local iteration
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

### Training (CUDA — the primary path)
```bash
# Single GPU
RUN_ID=my_run \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8xH100 (leaderboard config)
RUN_ID=my_run \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Override defaults via environment variables (all hyperparams are env-configurable):
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
ITERATIONS=20000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Training (MLX — Apple Silicon local development)
```bash
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm

RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

### Retokenization / Custom Tokenizers
```bash
python3 data/download_hf_docs_and_tokenize.py \
  --repo-id your-hf-username/your-dataset-repo \
  --remote-root your_export_root \
  --output-root /tmp/my_custom_tokenizer_export \
  --tokenizer-config ./data/tokenizer_specs.json
```

## Architecture

### Repository Structure

- `train_gpt.py` — Main CUDA training script (~1100 lines, self-contained). Contains model definition, training loop, quantization, evaluation, and serialization. This is what submissions modify.
- `train_gpt_mlx.py` — MLX port for local Apple Silicon development. Mirrors `train_gpt.py` architecture but uses `mlx` instead of PyTorch.
- `data/` — Dataset download helpers and tokenizer specs. `cached_challenge_fineweb.py` fetches pre-tokenized shards from HuggingFace.
- `records/` — Submission archive organized by track:
  - `track_10min_16mb/` — Leaderboard submissions (10min 8xH100 limit)
  - `track_non_record_16mb/` — Unlimited compute / non-record submissions
  - Each submission folder contains: `README.md`, `submission.json`, `train.log`, `train_gpt.py`, and optional `requirements.txt`

### Model Architecture (baseline in `train_gpt.py`)

The GPT model uses a **U-Net skip-connection** pattern:
- Layers are split into encoder half and decoder half
- Encoder layers store skip activations; decoder layers consume them in reverse order (like U-Net)
- `skip_weights` (learnable per-layer) scale the skip connections

Each **Block** contains:
- `resid_mix` — learnable residual mixing between current hidden state and initial embedding (`x0`)
- `CausalSelfAttention` with **GQA** (grouped-query attention), **RoPE**, QK-norm, learnable `q_gain`, and logit softcapping
- `MLP` using **ReLU²** activation (relu then square)
- Learnable `attn_scale` and `mlp_scale` per block

### Optimizer Setup

Three optimizer groups with separate learning rates:
1. **Token embedding** → Adam with `TIED_EMBED_LR` (or `EMBED_LR` if untied)
2. **2D matrix params in transformer blocks** → **Muon optimizer** (Newton-Schulz orthogonalization) with `MATRIX_LR`
3. **Scalar/vector params** (norms, scales, skip weights) → Adam with `SCALAR_LR`
4. **LM head** (if untied) → Adam with `HEAD_LR`

### Evaluation Pipeline

- **BPB (bits-per-byte)**: The primary metric. Converts token-level cross-entropy to byte-level compression using SentencePiece lookup tables, making scores comparable across different tokenizers.
- Validation always runs on the fixed first-50k-document FineWeb validation split.
- Sliding window evaluation at various strides is a common optimization in submissions.

### Serialization & Size Budget

1. Model trains in bf16/fp32
2. Post-training: **int8 quantization** with per-row scales for 2D tensors, per-tensor scales for vectors
3. Compressed with **zlib level 9**
4. Total artifact = `len(train_gpt.py)` + `len(compressed_model)` ≤ **16,000,000 bytes** (decimal, not MiB)
5. Round-trip validation: model is reloaded from the compressed artifact and re-evaluated to produce the final `val_bpb`

### Data Format

- Pre-tokenized binary shards: `fineweb_train_*.bin` and `fineweb_val_*.bin`
- Each shard is a flat array of uint16 token IDs loaded via `np.memmap`
- Default tokenizer: 1024-vocab SentencePiece BPE (`fineweb_1024_bpe.model`)

## Submission Rules

- New SOTA must beat existing by ≥0.005 nats at p < 0.01 (typically 3 runs)
- Submissions are PRs adding a folder under `records/track_10min_16mb/` or `records/track_non_record_16mb/`
- No network calls or training data access during evaluation
- Custom tokenizer submissions are scrutinized more carefully
- External packages are allowed but can't sneak in extra compute or code size
- Both `train_gpt.py` and `train_gpt_mlx.py` are capped at 1500 lines to stay readable
