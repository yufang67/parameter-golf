from __future__ import annotations

import argparse
import gc
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm
import torch
import torch.distributed as dist
import zstandard

import train_gpt_new as tg


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    num_layers: int
    model_dim: int
    num_heads: int
    num_kv_heads: int
    mlp_mult: int
    tie_embeddings: bool
    attn_type: str
    kv_latent_dim: int


@dataclass(frozen=True)
class Scenario:
    name: str
    eval_seq_len: int
    rope_type: str
    sliding: bool
    stride: int | None = None


@dataclass(frozen=True)
class EvalResult:
    checkpoint_label: str
    checkpoint_path: str
    scenario_name: str
    rope_type: str
    eval_seq_len: int
    stride: int | None
    val_loss: float
    val_bpb: float
    elapsed_ms: float


def parse_args() -> argparse.Namespace:
    defaults = tg.Hyperparameters()
    parser = argparse.ArgumentParser(description="Evaluate a parameter-golf checkpoint using roundtrip-restored weights with standard and sliding-window validation.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a raw .pt checkpoint or a saved int8+zstd roundtrip artifact (.ptz).")
    parser.add_argument(
        "--checkpoint-format",
        choices=("auto", "raw", "int8_zstd"),
        default="auto",
        help="Checkpoint format. Auto infers .ptz as int8+zstd artifact and everything else as raw checkpoint.",
    )
    parser.add_argument(
        "--simulate-roundtrip-from-raw",
        action="store_true",
        help="For a raw checkpoint, quantize and dequantize it in memory before evaluation. Raw checkpoints are not evaluated directly.",
    )
    parser.add_argument("--tokenizer-path", type=Path, default=Path(defaults.tokenizer_path))
    parser.add_argument("--val-files", type=str, default=defaults.val_files)
    parser.add_argument("--train-seq-len", type=int, default=defaults.train_seq_len)
    parser.add_argument("--control-eval-seq-len", type=int, default=defaults.train_seq_len)
    parser.add_argument("--stress-eval-seq-len", type=int, default=defaults.eval_seq_len)
    parser.add_argument(
        "--model-yarn-max-len",
        type=int,
        default=0,
        help="YaRN target context length baked into the restored model. Defaults to stress eval seq len for yarn models.",
    )
    parser.add_argument("--eval-stride", type=int, default=defaults.eval_stride)
    parser.add_argument("--val-batch-size", type=int, default=defaults.val_batch_size)
    parser.add_argument("--rope-type", choices=("rope", "yarn"), default=defaults.rope_type)
    parser.add_argument(
        "--include-rope-compare",
        action="store_true",
        help="Add rope-only stress scenarios alongside the default rope type.",
    )
    parser.add_argument("--rope-base", type=float, default=defaults.rope_base)
    parser.add_argument("--qk-gain-init", type=float, default=defaults.qk_gain_init)
    parser.add_argument("--tied-embed-init-std", type=float, default=defaults.tied_embed_init_std)
    parser.add_argument("--logit-softcap", type=float, default=defaults.logit_softcap)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for sliding-window logits.")
    parser.add_argument(
        "--reference-log",
        type=Path,
        default=None,
        help="Optional training log to compare against recorded final roundtrip and sliding metrics.",
    )
    parser.add_argument(
        "--limit-val-tokens",
        type=int,
        default=0,
        help="Optional limit for validation tokens after concatenation. Use 0 for the full split.",
    )
    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for eval_checkpoint.py because train_gpt_new eval uses CUDA autocast.")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    return distributed, rank, world_size, local_rank, device


def infer_checkpoint_format(checkpoint: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    return "int8_zstd" if checkpoint.suffix == ".ptz" else "raw"


def unwrap_raw_checkpoint(obj: object) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            candidate = obj.get(key)
            if isinstance(candidate, dict) and all(isinstance(v, torch.Tensor) for v in candidate.values()):
                return candidate
    raise TypeError("Unsupported raw checkpoint structure; expected a state_dict-like mapping.")


def load_checkpoint_state_dict(args: argparse.Namespace) -> tuple[dict[str, torch.Tensor], str]:
    checkpoint_format = infer_checkpoint_format(args.checkpoint, args.checkpoint_format)
    if checkpoint_format == "int8_zstd":
        compressed = args.checkpoint.read_bytes()
        decompressed = zstandard.ZstdDecompressor().decompress(compressed)
        quant_obj = torch.load(io.BytesIO(decompressed), map_location="cpu")
        state_dict = tg.dequantize_state_dict_int8(quant_obj)
        return state_dict, "roundtrip_restored_from_artifact"

    raw_obj = torch.load(args.checkpoint, map_location="cpu")
    state_dict = unwrap_raw_checkpoint(raw_obj)
    if not args.simulate_roundtrip_from_raw:
        raise ValueError(
            "Raw checkpoints are not evaluated directly. Pass --simulate-roundtrip-from-raw to evaluate the roundtrip-restored state, "
            "or point --checkpoint at a .ptz artifact to restore the roundtrip state from disk."
        )
    quant_obj, _ = tg.quantize_state_dict_int8(state_dict)
    state_dict = tg.dequantize_state_dict_int8(quant_obj)
    return state_dict, "roundtrip_simulated_from_raw"


def format_checkpoint_label(args: argparse.Namespace, checkpoint_label: str) -> str:
    return f"{args.checkpoint.stem}:{checkpoint_label}"


def infer_model_config(state_dict: dict[str, torch.Tensor]) -> ModelConfig:
    tok_emb = state_dict.get("tok_emb.weight")
    if tok_emb is None or tok_emb.ndim != 2:
        raise KeyError("Checkpoint is missing tok_emb.weight; cannot infer model shape.")

    block_indices = []
    for key in state_dict:
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 2 and parts[1].isdigit():
                block_indices.append(int(parts[1]))
    if not block_indices:
        raise KeyError("Checkpoint has no blocks.* parameters; cannot infer num_layers.")

    vocab_size, model_dim = tok_emb.shape
    num_layers = max(block_indices) + 1
    tie_embeddings = "lm_head.weight" not in state_dict

    q_gain = state_dict.get("blocks.0.attn.q_gain")
    if q_gain is None:
        raise KeyError("Checkpoint is missing blocks.0.attn.q_gain; cannot infer num_heads.")
    num_heads = int(q_gain.numel())

    kv_latent_dim = tg.Hyperparameters.kv_latent_dim
    if "blocks.0.attn.c_q_krope.weight" in state_dict:
        attn_type = "mla"
        num_kv_heads = num_heads
        kv_down = state_dict.get("blocks.0.attn.kv_down.weight")
        if kv_down is not None:
            kv_latent_dim = int(kv_down.shape[0])
    else:
        attn_type = "gqa"
        qkv_weight = state_dict.get("blocks.0.attn.c_qkv.weight")
        if qkv_weight is None:
            raise KeyError("Checkpoint is missing blocks.0.attn.c_qkv.weight; cannot infer num_kv_heads.")
        head_dim = model_dim // num_heads
        num_kv_heads = int((qkv_weight.shape[0] - model_dim) // (2 * head_dim))

    mlp_fc = state_dict.get("blocks.0.mlp.fc.weight")
    if mlp_fc is None:
        raise KeyError("Checkpoint is missing blocks.0.mlp.fc.weight; cannot infer mlp_mult.")
    mlp_mult = int(mlp_fc.shape[0] // model_dim)

    return ModelConfig(
        vocab_size=int(vocab_size),
        num_layers=int(num_layers),
        model_dim=int(model_dim),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        mlp_mult=int(mlp_mult),
        tie_embeddings=bool(tie_embeddings),
        attn_type=attn_type,
        kv_latent_dim=int(kv_latent_dim),
    )


def build_hparams(args: argparse.Namespace, config: ModelConfig, scenario: Scenario) -> tg.Hyperparameters:
    hparams = tg.Hyperparameters()
    hparams.tokenizer_path = str(args.tokenizer_path)
    hparams.val_files = args.val_files
    hparams.vocab_size = config.vocab_size
    hparams.num_layers = config.num_layers
    hparams.model_dim = config.model_dim
    hparams.num_heads = config.num_heads
    hparams.num_kv_heads = config.num_kv_heads
    hparams.mlp_mult = config.mlp_mult
    hparams.tie_embeddings = config.tie_embeddings
    hparams.attn_type = config.attn_type
    hparams.kv_latent_dim = config.kv_latent_dim
    hparams.train_seq_len = args.train_seq_len
    hparams.eval_seq_len = scenario.eval_seq_len
    hparams.eval_stride = args.eval_stride
    hparams.val_batch_size = args.val_batch_size
    hparams.rope_type = scenario.rope_type
    hparams.yarn_max_len = resolve_model_yarn_max_len(args, scenario)
    hparams.rope_base = args.rope_base
    hparams.qk_gain_init = args.qk_gain_init
    hparams.tied_embed_init_std = args.tied_embed_init_std
    hparams.logit_softcap = args.logit_softcap
    return hparams


def resolve_model_yarn_max_len(args: argparse.Namespace, scenario: Scenario) -> int:
    if scenario.rope_type != "yarn":
        return max(args.train_seq_len, args.stress_eval_seq_len)
    if args.model_yarn_max_len > 0:
        return args.model_yarn_max_len
    return args.stress_eval_seq_len


def build_scenarios(args: argparse.Namespace) -> list[Scenario]:
    scenarios = [
        Scenario(name=f"standard_{args.control_eval_seq_len}_{args.rope_type}", eval_seq_len=args.control_eval_seq_len, rope_type=args.rope_type, sliding=False),
        Scenario(
            name=f"sliding_{args.control_eval_seq_len}_{args.rope_type}",
            eval_seq_len=args.control_eval_seq_len,
            rope_type=args.rope_type,
            sliding=True,
            stride=args.eval_stride,
        ),
        Scenario(name=f"standard_{args.stress_eval_seq_len}_{args.rope_type}", eval_seq_len=args.stress_eval_seq_len, rope_type=args.rope_type, sliding=False),
        Scenario(
            name=f"sliding_{args.stress_eval_seq_len}_{args.rope_type}",
            eval_seq_len=args.stress_eval_seq_len,
            rope_type=args.rope_type,
            sliding=True,
            stride=args.eval_stride,
        ),
    ]
    if args.include_rope_compare and args.rope_type != "rope":
        scenarios.extend(
            [
                Scenario(name=f"standard_{args.stress_eval_seq_len}_rope", eval_seq_len=args.stress_eval_seq_len, rope_type="rope", sliding=False),
                Scenario(
                    name=f"sliding_{args.stress_eval_seq_len}_rope",
                    eval_seq_len=args.stress_eval_seq_len,
                    rope_type="rope",
                    sliding=True,
                    stride=args.eval_stride,
                ),
            ]
        )
    return scenarios


def load_validation_resources(
    args: argparse.Namespace,
    config: ModelConfig,
    device: torch.device,
    max_eval_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer_path))
    val_tokens = tg.load_validation_tokens(args.val_files, max_eval_seq_len)
    if args.limit_val_tokens > 0:
        usable = min(args.limit_val_tokens, val_tokens.numel() - 1)
        usable = (usable // max_eval_seq_len) * max_eval_seq_len
        if usable <= 0:
            raise ValueError("--limit-val-tokens is too small for the requested eval sequence length.")
        val_tokens = val_tokens[: usable + 1]
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tg.build_sentencepiece_luts(
        sp,
        config.vocab_size,
        device,
    )
    return val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def instantiate_model(
    args: argparse.Namespace,
    config: ModelConfig,
    scenario: Scenario,
    device: torch.device,
    state_dict: dict[str, torch.Tensor],
) -> tg.GPT:
    model = tg.GPT(
        vocab_size=config.vocab_size,
        num_layers=config.num_layers,
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        mlp_mult=config.mlp_mult,
        tie_embeddings=config.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        attn_type=config.attn_type,
        kv_latent_dim=config.kv_latent_dim,
        rope_type=scenario.rope_type,
        yarn_max_len=resolve_model_yarn_max_len(args, scenario),
        train_seq_len=args.train_seq_len,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, tg.CastedLinear):
            module.float()
    tg.restore_low_dim_params_to_fp32(model)
    model.load_state_dict(state_dict, strict=True)
    return model


def run_scenario(
    args: argparse.Namespace,
    checkpoint_label: str,
    config: ModelConfig,
    scenario: Scenario,
    device: torch.device,
    rank: int,
    world_size: int,
    state_dict: dict[str, torch.Tensor],
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
) -> EvalResult:
    hparams = build_hparams(args, config, scenario)
    model = instantiate_model(args, config, scenario, device, state_dict)
    torch.cuda.synchronize(device)
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    if scenario.sliding:
        val_loss, val_bpb = tg.eval_val_sliding(
            hparams,
            model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=scenario.stride or args.eval_stride,
            eval_seq_len=scenario.eval_seq_len,
            compile_enabled=args.compile,
        )
    else:
        val_loss, val_bpb = tg.eval_val(
            hparams,
            model,
            rank,
            world_size,
            device,
            1,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            eval_seq_len=scenario.eval_seq_len,
        )
    t1.record()
    torch.cuda.synchronize(device)
    elapsed_ms = float(t0.elapsed_time(t1))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return EvalResult(
        checkpoint_label=checkpoint_label,
        checkpoint_path=str(args.checkpoint),
        scenario_name=scenario.name,
        rope_type=scenario.rope_type,
        eval_seq_len=scenario.eval_seq_len,
        stride=scenario.stride,
        val_loss=val_loss,
        val_bpb=val_bpb,
        elapsed_ms=elapsed_ms,
    )


def find_same_length_standard(results: list[EvalResult], result: EvalResult) -> EvalResult | None:
    for candidate in results:
        if (
            candidate.checkpoint_label == result.checkpoint_label
            and candidate.eval_seq_len == result.eval_seq_len
            and candidate.rope_type == result.rope_type
            and candidate.stride is None
        ):
            return candidate
    return None


def print_summary(results: list[EvalResult], rank: int) -> None:
    if rank != 0:
        return
    checkpoint_width = max(len("checkpoint"), max(len(result.checkpoint_label) for result in results)) + 2
    scenario_width = max(len("scenario"), max(len(result.scenario_name) for result in results)) + 2
    header = (
        "checkpoint".ljust(checkpoint_width)
        + "scenario".ljust(scenario_width)
        + "rope".ljust(8)
        + "seq".rjust(8)
        + "stride".rjust(8)
        + "val_loss".rjust(14)
        + "val_bpb".rjust(14)
        + "delta_same_len".rjust(18)
        + "eval_s".rjust(10)
    )
    print(header)
    print("-" * len(header))
    for result in results:
        baseline = find_same_length_standard(results, result)
        delta = result.val_bpb - baseline.val_bpb if baseline is not None and baseline is not result else 0.0
        stride_text = "-" if result.stride is None else str(result.stride)
        print(
            f"{result.checkpoint_label:<{checkpoint_width}}"
            f"{result.scenario_name:<{scenario_width}}"
            f"{result.rope_type:<8}"
            f"{result.eval_seq_len:>8}"
            f"{stride_text:>8}"
            f"{result.val_loss:>14.8f}"
            f"{result.val_bpb:>14.8f}"
            f"{delta:>18.8f}"
            f"{result.elapsed_ms / 1000.0:>10.2f}"
        )


def parse_reference_metrics(reference_log: Path) -> dict[str, tuple[float, float]]:
    metrics: dict[str, tuple[float, float]] = {}
    for line in reference_log.read_text(encoding="utf-8").splitlines():
        if line.startswith("final_int8_zstd_roundtrip_exact val_loss:"):
            parts = line.split()
            metrics["roundtrip_2048"] = (
                float(parts[1].split(":", 1)[1]),
                float(parts[2].split(":", 1)[1]),
            )
        elif line.startswith("final_sliding_window_exact val_loss:"):
            parts = line.split()
            metrics["sliding_2048"] = (
                float(parts[1].split(":", 1)[1]),
                float(parts[2].split(":", 1)[1]),
            )
    return metrics


def reference_key_for_result(result: EvalResult) -> str | None:
    if result.eval_seq_len != 2048 or result.rope_type != "yarn":
        return None
    if result.stride is None:
        return "roundtrip_2048"
    if result.stride == 64:
        return "sliding_2048"
    return None


def print_reference_comparison(results: list[EvalResult], reference_log: Path, rank: int) -> None:
    if rank != 0:
        return
    metrics = parse_reference_metrics(reference_log)
    if not metrics:
        print(f"reference_log: {reference_log} (no final roundtrip/sliding exact metrics found)")
        return
    print(f"reference_log: {reference_log}")
    print("reference_comparison:")
    for result in results:
        key = reference_key_for_result(result)
        if key is None or key not in metrics:
            continue
        ref_loss, ref_bpb = metrics[key]
        print(
            f"  {result.scenario_name}: "
            f"ref_loss={ref_loss:.8f} ref_bpb={ref_bpb:.8f} "
            f"delta_loss={result.val_loss - ref_loss:.8f} delta_bpb={result.val_bpb - ref_bpb:.8f}"
        )


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, _local_rank, device = setup_distributed()
    try:
        scenarios = build_scenarios(args)
        state_dict, checkpoint_label = load_checkpoint_state_dict(args)
        checkpoint_label = format_checkpoint_label(args, checkpoint_label)
        config = infer_model_config(state_dict)
        max_eval_seq_len = max(s.eval_seq_len for s in scenarios)
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = load_validation_resources(
            args,
            config,
            device,
            max_eval_seq_len,
        )

        if rank == 0:
            print(f"checkpoint: {args.checkpoint}")
            print(
                "model_config: "
                f"layers={config.num_layers} dim={config.model_dim} heads={config.num_heads} "
                f"kv_heads={config.num_kv_heads} mlp_mult={config.mlp_mult} attn_type={config.attn_type}"
            )
            print(f"model_yarn_max_len_default: {args.model_yarn_max_len if args.model_yarn_max_len > 0 else args.stress_eval_seq_len}")
            print(f"validation_tokens: {val_tokens.numel() - 1}")

        results: list[EvalResult] = []
        for scenario in scenarios:
            if distributed:
                dist.barrier()
            result = run_scenario(
                args,
                checkpoint_label,
                config,
                scenario,
                device,
                rank,
                world_size,
                state_dict,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            results.append(result)
            if rank == 0:
                stride_text = "standard" if result.stride is None else f"sliding stride={result.stride}"
                print(
                    f"completed {result.scenario_name}: {stride_text}, "
                    f"val_loss={result.val_loss:.8f}, val_bpb={result.val_bpb:.8f}, eval_s={result.elapsed_ms / 1000.0:.2f}"
                )

        print_summary(results, rank)
        if args.reference_log is not None:
            print_reference_comparison(results, args.reference_log, rank)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()