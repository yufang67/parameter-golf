#!/usr/bin/env python3
"""Run standalone eval or TTT using train_gpt_stripped.py.

Examples:

  python run_ttt.py
  python run_ttt.py --mode all
  python run_ttt.py --mode sliding --checkpoint final_model.pt --format raw
  python run_ttt.py --data-dir ./data --vocab-size 8192 --mlp-mult 4.35 --fused-rope
  torchrun --standalone --nproc_per_node=4 run_ttt.py --mode ttt
"""

import argparse
import os

import torch
import torch.distributed as dist

import train_gpt_stripped as tg


def parse_args() -> argparse.Namespace:
    defaults = tg.Hyperparameters()
    parser = argparse.ArgumentParser(description="Run standalone eval/TTT with train_gpt_stripped.py")

    parser.add_argument("--checkpoint", type=str, default=defaults.quantized_model_path,
                        help="Checkpoint path (.pt for raw or .ptz for quantized)")
    parser.add_argument("--format", choices=("auto", "raw", "quantized"), default="auto",
                        help="Checkpoint format; auto infers from file extension")
    parser.add_argument("--mode", choices=("ttt", "standard", "sliding", "all"), default="ttt",
                        help="Which evaluation path to run")
    parser.add_argument("--logfile", type=str, default=None,
                        help="Optional log file path")

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--datasets-dir", type=str, default=None)
    parser.add_argument("--train-files", type=str, default=None)
    parser.add_argument("--val-files", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)

    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--mlp-mult", type=float, default=None)
    parser.add_argument("--rope-dims", type=int, default=None)
    parser.add_argument("--num-loops", type=int, default=None)
    parser.add_argument("--loop-start", type=int, default=None)
    parser.add_argument("--loop-end", type=int, default=None)
    parser.add_argument("--gptq-calibration-batches", type=int, default=None)
    parser.add_argument("--matrix-clip-sigmas", type=float, default=None)
    parser.add_argument("--gated-attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fused-rope", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--eval-seq-len", type=int, default=defaults.eval_seq_len)
    parser.add_argument("--eval-stride", type=int, default=defaults.eval_stride)
    parser.add_argument("--val-batch-tokens", type=int, default=defaults.val_batch_tokens)
    parser.add_argument("--batch-seqs", type=int, default=32,
                        help="Batch size for sliding-window and TTT eval loops")

    parser.add_argument("--ttt-lr", type=float, default=defaults.ttt_lr)
    parser.add_argument("--ttt-epochs", type=int, default=defaults.ttt_epochs)
    parser.add_argument("--ttt-momentum", type=float, default=defaults.ttt_momentum)
    parser.add_argument("--ttt-chunk-tokens", type=int, default=defaults.ttt_chunk_tokens)
    return parser.parse_args()


def setup_runtime() -> tuple[torch.device, bool]:
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

    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    return device, distributed


def apply_overrides(h: tg.Hyperparameters, args: argparse.Namespace) -> None:
    for name in (
        "vocab_size", "num_layers", "model_dim", "embedding_dim", "num_heads",
        "num_kv_heads", "mlp_mult", "rope_dims", "num_loops", "loop_start",
        "loop_end", "gptq_calibration_batches", "matrix_clip_sigmas",
        "gated_attention", "fused_rope",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(h, name, value)

    if args.data_dir is not None:
        h.data_dir = args.data_dir

    h.datasets_dir = args.datasets_dir or os.path.join(h.data_dir, "datasets", f"fineweb10B_sp{h.vocab_size}")
    h.train_files = args.train_files or os.path.join(h.datasets_dir, "fineweb_train_*.bin")
    h.val_files = args.val_files or os.path.join(h.datasets_dir, "fineweb_val_*.bin")
    h.tokenizer_path = args.tokenizer_path or os.path.join(h.data_dir, "tokenizers", f"fineweb_{h.vocab_size}_bpe.model")

    h.eval_seq_len = args.eval_seq_len
    h.eval_stride = args.eval_stride
    h.val_batch_tokens = args.val_batch_tokens

    h.ttt_enabled = True
    h.ttt_lr = args.ttt_lr
    h.ttt_epochs = args.ttt_epochs
    h.ttt_momentum = args.ttt_momentum
    h.ttt_chunk_tokens = args.ttt_chunk_tokens

    h.logfile = args.logfile


def load_model(args: argparse.Namespace, h: tg.Hyperparameters, device: torch.device) -> tg.GPT:
    fmt = args.format
    if fmt == "auto":
        fmt = "quantized" if args.checkpoint.endswith(".ptz") else "raw"

    if fmt == "quantized":
        h.quantized_model_path = args.checkpoint
        model = tg.deserialize(h, device)
    else:
        model = tg.GPT(h).to(device).bfloat16()
        tg.restore_fp32_params(model)
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)

    if h.num_loops > 0:
        model.looping_active = True
    return model


def reload_model(args: argparse.Namespace, h: tg.Hyperparameters, device: torch.device, model: tg.GPT) -> tg.GPT:
    del model
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return load_model(args, h, device)


def run_mode(mode: str, args: argparse.Namespace, h: tg.Hyperparameters, device: torch.device,
             val_data: tg.ValidationData, model: tg.GPT) -> tg.GPT:
    if mode == "standard":
        compiled = torch.compile(model, dynamic=False, fullgraph=True)
        tg.timed_eval("standard", tg.eval_val, h, device, val_data, compiled)
        del compiled
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        return model

    if mode == "sliding":
        tg.timed_eval("sliding_window", tg.eval_val_sliding, h, device, val_data, model, args.batch_seqs)
        return model

    tg.timed_eval("ttt", tg.eval_val_ttt, h, device, val_data, model, args.batch_seqs)
    return model


def main() -> None:
    args = parse_args()
    device, distributed = setup_runtime()

    h = tg.Hyperparameters()
    apply_overrides(h, args)
    tg.set_logging_hparams(h)

    if h.is_main_process and h.logfile is not None:
        os.makedirs(os.path.dirname(h.logfile) or ".", exist_ok=True)

    tg.log(f"Loading checkpoint: {args.checkpoint}")
    tg.log(f"Mode: {args.mode}")
    tg.log(
        f"TTT config: lr={h.ttt_lr} epochs={h.ttt_epochs} momentum={h.ttt_momentum} "
        f"chunk_tokens={h.ttt_chunk_tokens}"
    )
    tg.log(
        f"Model config: vocab={h.vocab_size} layers={h.num_layers} dim={h.model_dim} "
        f"heads={h.num_heads} kv_heads={h.num_kv_heads} mlp_mult={h.mlp_mult} fused_rope={h.fused_rope}"
    )
    tg.log(f"Eval config: seq_len={h.eval_seq_len} stride={h.eval_stride}")

    val_data = tg.ValidationData(h, device)
    tg.log(f"val_tokens: {val_data.val_tokens.numel() - 1}")

    model = load_model(args, h, device)
    tg.log(f"model_params: {sum(p.numel() for p in model.parameters())}")

    if args.mode == "all":
        for mode in ("standard", "sliding", "ttt"):
            model = run_mode(mode, args, h, device, val_data, model)
            if mode != "ttt":
                model = reload_model(args, h, device, model)
    else:
        run_mode(args.mode, args, h, device, val_data, model)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()