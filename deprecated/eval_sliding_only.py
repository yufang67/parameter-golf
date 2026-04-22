"""Eval-only harness for comparing VARLEN_PERDOC_SLIDING=0 vs =1.

Loads the existing quantized checkpoint (`final_model.int6.ptz`) and runs
`eval_val_sliding` twice to compare the old cross-document varlen path with
the new per-document path.

Env switches:
  MODE=old|new|stride|bs1_normal|all (default: all)
  EVAL_STRIDE_OVERRIDE=int  (used when MODE=stride)

Usage:
  RUN_ID=sliding_compare MLP_MULT=4.35 MATRIX_CLIP_SIGMAS=15 VARLEN_ATTENTION=1 \
  torchrun --standalone --nproc_per_node=4 eval_sliding_only.py
"""
from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_gpt_improved_04_16 as tg


def _run_stride(h, device, val_data, eval_model, stride: int):
    """Re-run new per-doc path with a smaller stride (affects long docs only)."""
    old_stride = h.__class__.eval_stride  # type: ignore[attr-defined]
    h.__class__.eval_stride = stride      # type: ignore[attr-defined]
    os.environ["VARLEN_PERDOC_SLIDING"] = "1"
    tg.log(f">>> NEW path with EVAL_STRIDE={stride}", console=True)
    try:
        tg.timed_eval(f"new_sliding_perdoc_s{stride}", tg.eval_val_sliding,
                      h, device, val_data, eval_model)
    finally:
        h.__class__.eval_stride = old_stride  # type: ignore[attr-defined]


def eval_bs1_normal(h, device, val_data, base_model):
    """Per-document evaluation with batch-size-1 and NORMAL (non-varlen) attention.
    Passes cu_seqlens=None so the attention module falls back to flash_attn_3_func.
    Long docs are split into intra-doc sliding windows (matches per-doc algo)."""
    base_model.eval()
    seq_len = h.eval_seq_len
    stride = h.eval_stride
    segments = tg._perdoc_varlen_segments(val_data.val_tokens, seq_len, stride)
    if getattr(h, 'slot_max_windows', 0) > 0 and len(segments) > h.slot_max_windows:
        segments = segments[: h.slot_max_windows]
    my_segments = segments[h.rank::h.world_size]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    is_master = h.rank == 0
    total = len(my_segments)
    log_every = max(1, total // 20)
    t0 = time.perf_counter()
    val_tokens = val_data.val_tokens
    with torch.inference_mode():
        for si, (ds, L, slo, shi) in enumerate(my_segments):
            if is_master and (si % log_every == 0 or si == total - 1):
                elapsed = time.perf_counter() - t0
                rl = float(loss_sum.item() / token_count.item()) if token_count.item() > 0 else 0.0
                rb = float((rl / math.log(2.0)) * token_count.item() / byte_count.item()) if byte_count.item() > 0 else 0.0
                tg.log(f"bs1_normal: seg {si+1}/{total} L={L} "
                       f"tokens:{int(token_count.item())} running_loss:{rl:.4f} "
                       f"running_bpb:{rb:.4f} elapsed:{elapsed:.1f}s")
            chunk = val_tokens[ds:ds + L + 1].to(dtype=torch.int64, device=device)
            x = chunk[:-1].unsqueeze(0)   # [1, L]
            y = chunk[1:]                 # [L]
            bucket_L = tg._bucket_max_seqlen(L)
            if bucket_L > L:
                pad = torch.zeros((1, bucket_L - L), dtype=torch.int64, device=device)
                x_padded = torch.cat([x, pad], dim=1)
            else:
                x_padded = x
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(x_padded, cu_seqlens=None, max_seqlen=bucket_L)
            logits = logits[0, :L]  # [L, V]
            nll = F.cross_entropy(logits.float(), y, reduction="none")
            if shi > slo:
                scored = nll[slo:shi].to(torch.float64)
                loss_sum += scored.sum()
                token_count += float(shi - slo)
                tgt = y[slo:shi]
                prev = x[0, slo:shi]
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (val_data.has_leading_space_lut[tgt] &
                       ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train()
    return tg._loss_bpb(loss_sum, token_count, byte_count)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("VARLEN_ATTENTION", "1")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import (
        enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    )
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False

    h = tg.Hyperparameters()
    tg.set_logging_hparams(h)
    mode = os.environ.get("MODE", "all")
    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        tg.log("=" * 100, console=False)
        tg.log(f"eval_sliding_only mode={mode}", console=True)
        tg.log(f"  model_path={h.model_path} quantized_model_path={h.quantized_model_path}", console=True)
        tg.log(f"  eval_seq_len={h.eval_seq_len} eval_stride={h.eval_stride}", console=True)

    val_data = tg.ValidationData(h, device)
    tg.log(f"val_tokens: {val_data.val_tokens.numel() - 1}", console=True)

    eval_model = tg.deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True

    if mode in ("all", "old"):
        os.environ["VARLEN_PERDOC_SLIDING"] = "0"
        tg.log(">>> OLD path: VARLEN_PERDOC_SLIDING=0", console=True)
        tg.timed_eval("old_sliding", tg.eval_val_sliding, h, device, val_data, eval_model)
    if mode in ("all", "new"):
        os.environ["VARLEN_PERDOC_SLIDING"] = "1"
        tg.log(">>> NEW path: VARLEN_PERDOC_SLIDING=1", console=True)
        tg.timed_eval("new_sliding_perdoc", tg.eval_val_sliding, h, device, val_data, eval_model)
    if mode in ("all", "stride"):
        override = os.environ.get("EVAL_STRIDE_OVERRIDE", "")
        strides = [int(s) for s in override.split(",") if s] if override else [16, 8]
        for s in strides:
            _run_stride(h, device, val_data, eval_model, s)
    if mode in ("all", "bs1_normal"):
        tg.log(">>> BS1 NORMAL attention path (cu_seqlens=None, one doc per forward)",
               console=True)
        tg.timed_eval("bs1_normal_attn", eval_bs1_normal,
                      h, device, val_data, eval_model)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

