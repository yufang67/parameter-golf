#!/usr/bin/env python3
"""
quant_only.py — load a saved fp32 state_dict and run only the
GPTQ-quantization + compression block from train_gpt_improved_04_16.py.

Use this to iterate on quantization parameters (LOOP_LAYER_BITS,
CLIP_MULT_*, MATRIX_CLIP_SIGMAS, COMPRESS_*, LOOP_LAYER_KEEP_*, etc.)
without re-training. Hessians are collected once and cached.

Usage (single GPU):
  python3 quant_only.py [--ckpt final_model.pt] [--hess-cache hess.pt]
                       [--recompute-hess] [--out-prefix tag_]

All quantization knobs come from environment variables exactly like
the main training script — e.g.:

  CLIP_MULT_LATE=0.5 CLIP_MULT_LOOP=0.5 LOOP_LAYER_BITS=5 \\
  COMPRESS_ANS=1 COMPRESS_BROTLI=1 COMPRESS_LZMA=1 \\
  python3 quant_only.py

The architecture env vars (NUM_LAYERS, MODEL_DIM, MLP_MULT, ...) MUST
match those used to produce the checkpoint, otherwise state_dict load
will fail.
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

# Reuse implementation from the main training script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_gpt_improved_04_16 as tg


def _setup_runtime() -> torch.device:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import (
        enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp,
    )
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    return device


def _build_calib_loader(h, base_model, device):
    if getattr(base_model, "use_varlen", False) and tg._HAS_VARLEN:
        return tg.DocumentPackingLoader(h, device)
    return tg.ShuffledSequenceLoader(h, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="final_model.pt",
                        help="path to saved state_dict (default: final_model.pt)")
    parser.add_argument("--hess-cache", default="final_model.hess.pt",
                        help="path to Hessians cache (default: final_model.hess.pt)")
    parser.add_argument("--recompute-hess", action="store_true",
                        help="ignore --hess-cache and recompute Hessians")
    parser.add_argument("--out-prefix", default="",
                        help="prefix for output quant blob filename(s)")
    parser.add_argument("--write-blob", action="store_true",
                        help="write the chosen compressed blob to disk")
    args = parser.parse_args()

    device = _setup_runtime()
    h = tg.Hyperparameters()
    if h.rope_force_base_seqlen <= 0:
        h.rope_force_base_seqlen = max(
            h.train_seq_len + 1,
            h.train_batch_tokens // max(1, h.world_size * h.grad_accum_steps),
        )
    os.environ["ROPE_FORCE_BASE_SEQLEN"] = str(h.rope_force_base_seqlen)
    tg.set_logging_hparams(h)

    if h.is_main_process:
        tg.log("=" * 100, console=False)
        tg.log("quant_only — Hyperparameters:", console=True)
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                tg.log(f"  {k}: {v}", console=False)
        tg.log("=" * 100, console=False)
        tg.log(f"Args: ckpt={args.ckpt} hess_cache={args.hess_cache} "
               f"recompute_hess={args.recompute_hess}", console=True)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")

    tg.log(f"Loading state_dict from {args.ckpt} "
           f"({os.path.getsize(args.ckpt)} bytes)")
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)

    # Build a model so we can run forward passes for Hessian collection,
    # and also so use_varlen / loop flags are wired up correctly.
    base_model = tg.GPT(h).to(device).bfloat16()
    missing, unexpected = base_model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        tg.log(f"state_dict mismatch — missing={len(missing)} unexpected={len(unexpected)}")
        for n in list(missing)[:8]:
            tg.log(f"  missing: {n}")
        for n in list(unexpected)[:8]:
            tg.log(f"  unexpected: {n}")
    if h.num_loops > 0:
        base_model.looping_active = True

    # Hessians: load cached or collect once.
    hessians = None
    if not args.recompute_hess and os.path.isfile(args.hess_cache):
        try:
            tg.log(f"Loading cached Hessians from {args.hess_cache} "
                   f"({os.path.getsize(args.hess_cache)} bytes)")
            cached = torch.load(args.hess_cache, map_location="cpu", weights_only=True)
            if isinstance(cached, dict) and all(isinstance(v, torch.Tensor) for v in cached.values()):
                hessians = cached
                tg.log(f"Loaded {len(hessians)} cached Hessians")
            else:
                tg.log("Cache format unrecognized — recomputing")
        except Exception as e:
            tg.log(f"Failed to load cache ({e!r}) — recomputing")

    if hessians is None:
        tg.log(f"Collecting Hessians ({h.gptq_calibration_batches} calibration batches)...")
        t0 = time.perf_counter()
        calib_loader = _build_calib_loader(h, base_model, device)
        hessians = tg.collect_hessians(
            base_model, calib_loader, h, device,
            n_calibration_batches=h.gptq_calibration_batches,
        )
        tg.log(f"Collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s")
        if h.is_main_process:
            tg.log(f"Caching Hessians to {args.hess_cache}")
            torch.save(hessians, args.hess_cache)

    # Need fp32 weights on CPU for GPTQ (matches serialize()).
    base_model.float()
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}

    tg.log("Running gptq_mixed_quantize...")
    t0 = time.perf_counter()
    quant_result, quant_meta = tg.gptq_mixed_quantize(sd_cpu, hessians, h)
    tg.log(f"GPTQ done in {time.perf_counter()-t0:.1f}s")

    candidates: dict[str, bytes] = {}
    t1 = time.perf_counter()
    candidates["ans"] = tg._ans_compress_quant(quant_result, quant_meta)
    tg.log(f"compress:ans {len(candidates['ans'])} bytes ({time.perf_counter()-t1:.1f}s)")

    # Approximate code size to give a "Total submission" number comparable
    # to the training-script log line. Use the actual packed entry script if
    # present, falling back to the source file.
    code_path = None
    for cand in ("train_gpt.py", "train_gpt_improved_04_16.py"):
        if os.path.isfile(cand):
            code_path = cand
            break
    code_bytes = os.path.getsize(code_path) if code_path else 0
    tg.log(f"Code size: {code_bytes} bytes (from {code_path})")

    for name, blob in sorted(candidates.items(), key=lambda kv: len(kv[1])):
        tg.log(f"  {name}: model={len(blob)} total={len(blob)+code_bytes}")
    chosen = min(candidates, key=lambda k: len(candidates[k]))
    quant_blob = candidates[chosen]
    bytes_total = len(quant_blob) + code_bytes
    tg.log(f"compress:using {chosen} ({len(quant_blob)} bytes)")
    tg.log(f"Serialized model quantized+{chosen}: {len(quant_blob)} bytes")
    tg.log(f"Total submission size quantized+{chosen}: {bytes_total} bytes")

    if args.write_blob and h.is_main_process:
        out_path = f"{args.out_prefix}{h.quantized_model_path}"
        with open(out_path, "wb") as f:
            f.write(quant_blob)
        tg.log(f"Wrote quant blob to {out_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
