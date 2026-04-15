#!/usr/bin/env python3
"""Run Test-Time Training (TTT) evaluation on a saved checkpoint.

Loads a checkpoint (raw .pt or quantized .ptz) and runs eval_val_ttt from
train_gpt_improved.py with configurable TTT hyperparameters.

Usage examples:

  # Quantized checkpoint (default paths)
  python run_ttt.py

  # Raw fp32 checkpoint
  python run_ttt.py --checkpoint final_model.pt --format raw

  # Quantized checkpoint with custom TTT params
  python run_ttt.py --ttt-lora-rank 96 --ttt-lora-lr 0.0001 --ttt-chunk-size 32

  # Multi-GPU
  torchrun --standalone --nproc_per_node=8 run_ttt.py

  # Also run non-TTT sliding window for comparison
  python run_ttt.py --also-sliding
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

# Import everything we need from train_gpt_improved
import train_gpt_improved as tg


def parse_args() -> argparse.Namespace:
    h_defaults = tg.Hyperparameters()
    p = argparse.ArgumentParser(description="Run TTT eval on a saved checkpoint")

    # Checkpoint
    p.add_argument("--checkpoint", type=str, default=h_defaults.quantized_model_path,
                   help="Path to checkpoint (.pt for raw, .ptz for quantized)")
    p.add_argument("--format", choices=("auto", "raw", "quantized"), default="auto",
                   help="Checkpoint format. 'auto' infers from extension.")

    # TTT hyperparameters
    p.add_argument("--ttt-lora-rank", type=int, default=h_defaults.ttt_lora_rank)
    p.add_argument("--ttt-lora-lr", type=float, default=h_defaults.ttt_lora_lr)
    p.add_argument("--ttt-chunk-size", type=int, default=h_defaults.ttt_chunk_size)
    p.add_argument("--ttt-batch-size", type=int, default=h_defaults.ttt_batch_size)
    p.add_argument("--ttt-grad-steps", type=int, default=h_defaults.ttt_grad_steps)

    # Eval settings
    p.add_argument("--eval-seq-len", type=int, default=h_defaults.eval_seq_len)
    p.add_argument("--eval-stride", type=int, default=h_defaults.eval_stride)
    p.add_argument("--batch-seqs", type=int, default=32,
                   help="Number of sequences per eval batch")

    # Optional extra evals
    p.add_argument("--also-sliding", action="store_true",
                   help="Also run non-TTT sliding window eval for comparison")
    p.add_argument("--also-standard", action="store_true",
                   help="Also run standard (non-sliding) eval for comparison")

    return p.parse_args()


def setup_env() -> tuple[torch.device, bool]:
    """Set up CUDA, distributed, and torch settings."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    from torch.backends.cuda import (
        enable_cudnn_sdp, enable_flash_sdp,
        enable_math_sdp, enable_mem_efficient_sdp,
    )
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    return device, distributed


def load_model(args: argparse.Namespace, h: tg.Hyperparameters,
               device: torch.device) -> tg.GPT:
    """Load model from checkpoint."""
    fmt = args.format
    if fmt == "auto":
        fmt = "quantized" if args.checkpoint.endswith(".ptz") else "raw"

    if fmt == "quantized":
        # Use the deserialize path which handles ANS/brotli/lzma
        h.quantized_model_path = args.checkpoint
        model = tg.deserialize(h, device)
    else:
        # Raw .pt checkpoint
        model = tg.GPT(h).to(device).bfloat16()
        tg.restore_fp32_params(model)
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)

    # Enable layer looping if configured
    if h.num_loops > 0:
        model.looping_active = True

    return model


def main():
    args = parse_args()

    device, distributed = setup_env()

    # Build hyperparameters with TTT overrides
    # Set env vars so Hyperparameters picks them up
    os.environ["TTT_ENABLED"] = "1"
    os.environ["TTT_LORA_RANK"] = str(args.ttt_lora_rank)
    os.environ["TTT_LORA_LR"] = str(args.ttt_lora_lr)
    os.environ["TTT_CHUNK_SIZE"] = str(args.ttt_chunk_size)
    os.environ["TTT_BATCH_SIZE"] = str(args.ttt_batch_size)
    os.environ["TTT_GRAD_STEPS"] = str(args.ttt_grad_steps)
    os.environ["EVAL_SEQ_LEN"] = str(args.eval_seq_len)
    os.environ["EVAL_STRIDE"] = str(args.eval_stride)

    h = tg.Hyperparameters()
    tg.set_logging_hparams(h)

    tg.log(f"Loading checkpoint: {args.checkpoint}")
    tg.log(f"TTT config: rank={h.ttt_lora_rank} lr={h.ttt_lora_lr} "
           f"chunk={h.ttt_chunk_size} batch={h.ttt_batch_size} grad_steps={h.ttt_grad_steps}")
    tg.log(f"Eval config: seq_len={h.eval_seq_len} stride={h.eval_stride}")

    # Load validation data
    val_data = tg.ValidationData(h, device)
    tg.log(f"val_tokens: {val_data.val_tokens.numel() - 1}")

    # Load model
    model = load_model(args, h, device)
    tg.log(f"model_params: {sum(p.numel() for p in model.parameters())}")

    # Optional: standard eval
    if args.also_standard:
        compiled = torch.compile(model, dynamic=False, fullgraph=True)
        tg.timed_eval("standard", tg.eval_val, h, device, val_data, compiled)
        del compiled
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    # Optional: sliding window eval (no TTT)
    if args.also_sliding:
        tg.timed_eval("sliding_window", tg.eval_val_sliding,
                       h, device, val_data, model, args.batch_seqs)
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    # TTT eval — reload fresh model since sliding may have compiled it
    if args.also_sliding:
        del model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        model = load_model(args, h, device)

    tg.timed_eval("ttt", tg.eval_val_ttt, h, device, val_data,
                   model, args.batch_seqs)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
