"""Eval-only: load final_model.int6.ptz and run sliding-window + LoRA TTT.

Env vars must match the training config of the checkpoint (pg12_varlen_clip14).
"""
import os, sys, time, torch
import torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_gpt_improved_04_16 as tg


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
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
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False

    h = tg.Hyperparameters()
    tg.set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                tg.log(f"  {k}: {v}", console=False)
    val_data = tg.ValidationData(h, device)
    tg.log(f"val_tokens: {val_data.val_tokens.numel() - 1}")

    eval_model = tg.deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    tg.log(f"use_varlen: {eval_model.use_varlen} ; per-block: "
           f"{all(b.attn.use_varlen for b in eval_model.blocks)}")

    # Sliding-window baseline
    if int(os.environ.get("RUN_SLIDING", "1")):
        tg.timed_eval("sw", tg.eval_val_sliding, h, device, val_data, eval_model)

    # LoRA TTT
    if h.ttt_enabled:
        for p in eval_model.parameters():
            p.requires_grad_(False)
        def _fwd_ttt_inner(input_ids, target_ids, lora):
            return eval_model.forward_ttt(input_ids, target_ids, lora=lora)
        _compiled = [None]
        def _fwd_ttt(input_ids, target_ids, lora):
            if _compiled[0] is None:
                _compiled[0] = torch.compile(_fwd_ttt_inner, dynamic=True)
            return _compiled[0](input_ids, target_ids, lora=lora)
        tg.log("ttt_lora:warming up compile")
        t0 = time.perf_counter()
        val_tokens_idx = val_data.val_tokens.to(torch.int32)
        wl = tg.BatchedTTTLoRA(h.ttt_batch_size, eval_model, h.ttt_lora_rank,
                               k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora).to(device)
        wo = torch.optim.AdamW(wl.parameters(), lr=h.ttt_lora_lr,
                               betas=(h.ttt_beta1, h.ttt_beta2), eps=1e-10,
                               weight_decay=h.ttt_weight_decay, fused=True)
        for ctx_len in (h.ttt_chunk_size, h.ttt_eval_seq_len):
            col_w = torch.arange(ctx_len + 1)
            idx_w = col_w.clamp_(max=val_data.val_tokens.numel() - 1)
            row_w = val_tokens_idx[idx_w].to(device=device, dtype=torch.int64)
            xw = row_w[:ctx_len].unsqueeze(0).expand(h.ttt_batch_size, -1).contiguous()
            yw = row_w[1:ctx_len + 1].unsqueeze(0).expand(h.ttt_batch_size, -1).contiguous()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ptl = _fwd_ttt(xw, yw, lora=wl)
            ptl[:, :min(h.ttt_chunk_size, ctx_len)].mean(dim=-1).sum().backward()
            wo.step(); wo.zero_grad(set_to_none=True)
        del wl, wo, val_tokens_idx
        torch.cuda.empty_cache()
        tg.log(f"ttt_lora:compile warmup done ({time.perf_counter() - t0:.1f}s)")
        tg.timed_eval("ttt_lora", tg.eval_val_ttt, h, device, val_data, eval_model,
                      forward_ttt_train=_fwd_ttt)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
