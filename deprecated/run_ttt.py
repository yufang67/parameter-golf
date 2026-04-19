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

    # TTT mode
    p.add_argument("--ttt-mode", type=str, default="lora",
                   choices=("lora", "fullparam"),
                   help="TTT mode: 'lora' (per-doc LoRA) or 'fullparam' (full-param SGD)")

    # LoRA TTT hyperparameters
    p.add_argument("--ttt-lora-rank", type=int, default=h_defaults.ttt_lora_rank,
                   help="LoRA rank")
    p.add_argument("--ttt-lora-lr", type=float, default=h_defaults.ttt_lora_lr)
    p.add_argument("--ttt-chunk-size", type=int, default=h_defaults.ttt_chunk_size,
                   help="Chunk size in tokens")
    p.add_argument("--ttt-batch-size", type=int, default=h_defaults.ttt_batch_size)
    p.add_argument("--ttt-grad-steps", type=int, default=h_defaults.ttt_grad_steps)

    # Phased global SGD TTT
    p.add_argument("--ttt-phases", type=int, default=h_defaults.ttt_phases,
                   help="Number of phases for global SGD (1=disabled)")
    p.add_argument("--ttt-phase-sgd-lr", type=float, default=h_defaults.ttt_phase_sgd_lr)
    p.add_argument("--ttt-phase-sgd-epochs", type=int, default=h_defaults.ttt_phase_sgd_epochs)
    p.add_argument("--ttt-phase-sgd-momentum", type=float, default=h_defaults.ttt_phase_sgd_momentum)
    p.add_argument("--ttt-phase-sgd-seq-len", type=int, default=h_defaults.ttt_phase_sgd_seq_len)
    p.add_argument("--ttt-phase-sgd-batch", type=int, default=h_defaults.ttt_phase_sgd_batch)
    p.add_argument("--ttt-phase-sgd-opt", type=str, default=h_defaults.ttt_phase_sgd_optimizer,
                   choices=("sgd", "adamw"))

    # Adaptive per-chunk LR
    p.add_argument("--ttt-adaptive-lr", type=int, default=0,
                   help="Scale LoRA LR per chunk by chunk_loss/EMA (1=enabled)")
    p.add_argument("--ttt-adapt-ema", type=float, default=h_defaults.ttt_adapt_ema)
    p.add_argument("--ttt-adapt-min-scale", type=float, default=h_defaults.ttt_adapt_min_scale)
    p.add_argument("--ttt-adapt-max-scale", type=float, default=h_defaults.ttt_adapt_max_scale)
    p.add_argument("--ttt-adapt-power", type=float, default=h_defaults.ttt_adapt_power)

    # SLOT-4
    p.add_argument("--slot-enabled", type=int, default=0,
                   help="Run SLOT-4 eval (per-window logit-delta) after TTT")
    p.add_argument("--slot-only", action="store_true",
                   help="Run only SLOT (skip TTT)")
    p.add_argument("--slot-steps", type=int, default=h_defaults.slot_steps)
    p.add_argument("--slot-lr", type=float, default=h_defaults.slot_lr)
    p.add_argument("--slot-wd", type=float, default=h_defaults.slot_wd)
    p.add_argument("--slot-train-mode", type=str, default=h_defaults.slot_train_mode,
                   choices=("context", "all"),
                   help="'context' = legal (train on already-scored context), 'all' = train on all window positions")
    p.add_argument("--slot-batch-seqs", type=int, default=4,
                   help="Batch windows per SLOT step (lower for memory)")

    # Full-param TTT hyperparameters
    p.add_argument("--ttt-fp-lr", type=float, default=h_defaults.ttt_fp_lr,
                   help="Full-param TTT learning rate")
    p.add_argument("--ttt-fp-epochs", type=int, default=h_defaults.ttt_fp_epochs,
                   help="Full-param TTT adaptation epochs per chunk")
    p.add_argument("--ttt-fp-chunk-tokens", type=int, default=h_defaults.ttt_fp_chunk_tokens,
                   help="Full-param TTT chunk size in tokens")

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
    os.environ["TTT_PHASES"] = str(args.ttt_phases)
    os.environ["TTT_PHASE_SGD_LR"] = str(args.ttt_phase_sgd_lr)
    os.environ["TTT_PHASE_SGD_EPOCHS"] = str(args.ttt_phase_sgd_epochs)
    os.environ["TTT_PHASE_SGD_MOMENTUM"] = str(args.ttt_phase_sgd_momentum)
    os.environ["TTT_PHASE_SGD_SEQ_LEN"] = str(args.ttt_phase_sgd_seq_len)
    os.environ["TTT_PHASE_SGD_BATCH"] = str(args.ttt_phase_sgd_batch)
    os.environ["TTT_PHASE_SGD_OPT"] = str(args.ttt_phase_sgd_opt)
    os.environ["TTT_FP_LR"] = str(args.ttt_fp_lr)
    os.environ["TTT_FP_EPOCHS"] = str(args.ttt_fp_epochs)
    os.environ["TTT_FP_CHUNK_TOKENS"] = str(args.ttt_fp_chunk_tokens)
    os.environ["EVAL_SEQ_LEN"] = str(args.eval_seq_len)
    os.environ["EVAL_STRIDE"] = str(args.eval_stride)

    h = tg.Hyperparameters()
    # Override instance attributes from args (class attrs are evaluated at module import
    # time so env var setting above is too late for them).
    h.ttt_lora_rank = args.ttt_lora_rank
    h.ttt_lora_lr = args.ttt_lora_lr
    h.ttt_chunk_size = args.ttt_chunk_size
    h.ttt_batch_size = args.ttt_batch_size
    h.ttt_grad_steps = args.ttt_grad_steps
    h.ttt_phases = args.ttt_phases
    h.ttt_phase_sgd_lr = args.ttt_phase_sgd_lr
    h.ttt_phase_sgd_epochs = args.ttt_phase_sgd_epochs
    h.ttt_phase_sgd_momentum = args.ttt_phase_sgd_momentum
    h.ttt_phase_sgd_seq_len = args.ttt_phase_sgd_seq_len
    h.ttt_phase_sgd_batch = args.ttt_phase_sgd_batch
    h.ttt_phase_sgd_optimizer = args.ttt_phase_sgd_opt
    h.ttt_adaptive_lr = bool(args.ttt_adaptive_lr)
    h.ttt_adapt_ema = args.ttt_adapt_ema
    h.ttt_adapt_min_scale = args.ttt_adapt_min_scale
    h.ttt_adapt_max_scale = args.ttt_adapt_max_scale
    h.ttt_adapt_power = args.ttt_adapt_power
    h.slot_enabled = bool(args.slot_enabled) or args.slot_only
    h.slot_steps = args.slot_steps
    h.slot_lr = args.slot_lr
    h.slot_wd = args.slot_wd
    h.slot_train_mode = args.slot_train_mode
    h.ttt_fp_lr = args.ttt_fp_lr
    h.ttt_fp_epochs = args.ttt_fp_epochs
    h.ttt_fp_chunk_tokens = args.ttt_fp_chunk_tokens
    h.eval_seq_len = args.eval_seq_len
    h.eval_stride = args.eval_stride
    tg.set_logging_hparams(h)

    ttt_mode = args.ttt_mode
    tg.log(f"Loading checkpoint: {args.checkpoint}")
    tg.log(f"TTT mode: {ttt_mode}")
    if ttt_mode == 'lora':
        tg.log(f"TTT config: rank={h.ttt_lora_rank} lr={h.ttt_lora_lr} "
               f"chunk={h.ttt_chunk_size} batch={h.ttt_batch_size} grad_steps={h.ttt_grad_steps}")
    else:
        tg.log(f"TTT config: lr={h.ttt_fp_lr} epochs={h.ttt_fp_epochs} "
               f"chunk_tokens={h.ttt_fp_chunk_tokens}")
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
                       h, device, val_data, model, batch_seqs=args.batch_seqs)
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    # TTT eval — reload fresh model since sliding may have compiled it
    if args.also_sliding:
        del model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        model = load_model(args, h, device)

    if args.slot_only:
        # SLOT only, no TTT
        tg.timed_eval("slot", tg.eval_val_slot,
                      h, device, val_data, model, None, args.slot_batch_seqs)
        if distributed:
            dist.destroy_process_group()
        return

    if ttt_mode == 'fullparam':
        # Full-param TTT: no LoRA, no compile warmup needed
        tg.timed_eval("ttt_fullparam", tg.eval_val_ttt_fullparam,
                       h, device, val_data, model, args.batch_seqs)
    else:
        for p in model.parameters():
            p.requires_grad_(False)

        # Lazy torch.compile with warmup
        def _fwd_ttt_inner(input_ids, target_ids, lora):
            return model.forward_ttt(input_ids, target_ids, lora=lora)
        _fwd_ttt_compiled_inner = None
        def _fwd_ttt(input_ids, target_ids, lora):
            nonlocal _fwd_ttt_compiled_inner
            if _fwd_ttt_compiled_inner is None:
                _fwd_ttt_compiled_inner = torch.compile(_fwd_ttt_inner, dynamic=True)
            return _fwd_ttt_compiled_inner(input_ids, target_ids, lora=lora)
        fwd_ttt_compiled = _fwd_ttt

        tg.log("ttt_lora:warming up compile")
        import time as _time
        t_warmup = _time.perf_counter()
        val_tokens_idx = val_data.val_tokens.to(torch.int32)
        wl = tg.BatchedTTTLoRA(
            h.ttt_batch_size, model, h.ttt_lora_rank,
            k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
        ).to(device)
        wo = torch.optim.AdamW(
            wl.parameters(), lr=h.ttt_lora_lr,
            betas=(h.ttt_beta1, h.ttt_beta2),
            eps=1e-10, weight_decay=h.ttt_weight_decay, fused=True,
        )
        for ctx_len in (h.ttt_chunk_size, h.ttt_eval_seq_len):
            col_w = torch.arange(ctx_len + 1)
            idx_w = col_w.clamp_(max=val_data.val_tokens.numel() - 1)
            row_w = val_tokens_idx[idx_w].to(device=device, dtype=torch.int64)
            xw = row_w[:ctx_len].unsqueeze(0).expand(h.ttt_batch_size, -1).contiguous()
            yw = row_w[1:ctx_len + 1].unsqueeze(0).expand(h.ttt_batch_size, -1).contiguous()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ptl = fwd_ttt_compiled(xw, yw, lora=wl)
            ptl[:, :min(h.ttt_chunk_size, ctx_len)].mean(dim=-1).sum().backward()
            wo.step()
            wo.zero_grad(set_to_none=True)
        del wl, wo, val_tokens_idx
        torch.cuda.empty_cache()
        tg.log(f"ttt_lora:compile warmup done ({_time.perf_counter() - t_warmup:.1f}s)")

        tg.timed_eval("ttt_lora", tg.eval_val_ttt, h, device, val_data,
                       model, forward_ttt_train=fwd_ttt_compiled)

    if h.slot_enabled and not args.slot_only:
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        tg.timed_eval("slot_post_ttt", tg.eval_val_slot,
                      h, device, val_data, model, None, args.slot_batch_seqs)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
