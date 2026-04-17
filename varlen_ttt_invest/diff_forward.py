"""Diff forward_ttt (zero LoRA) vs forward_logits (varlen) on same docs."""
import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_gpt_improved_04_16 as tg


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    h = tg.Hyperparameters()
    tg.set_logging_hparams(h)
    val_data = tg.ValidationData(h, device)
    eval_model = tg.deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    eval_model.eval()
    for p in eval_model.parameters():
        p.requires_grad_(False)

    all_tokens = val_data.val_tokens
    docs = tg._find_docs(all_tokens)[:4]
    tg.log(f"Testing {len(docs)} docs, lens={[d[1] for d in docs]}")

    # --- Path A: forward_ttt (dense, zero LoRA) per doc ---
    max_len = max(L for _, L in docs)
    bsz = len(docs)
    x_batch = torch.zeros(bsz, max_len - 1, dtype=torch.int64, device=device)
    y_batch = torch.zeros(bsz, max_len - 1, dtype=torch.int64, device=device)
    wlens = []
    for i, (s, L) in enumerate(docs):
        x_batch[i, :L-1] = all_tokens[s:s+L-1].to(device=device, dtype=torch.int64)
        y_batch[i, :L-1] = all_tokens[s+1:s+L].to(device=device, dtype=torch.int64)
        wlens.append(L - 1)
    lora = tg.BatchedTTTLoRA(bsz, eval_model, h.ttt_lora_rank,
                             k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ptl_ttt = eval_model.forward_ttt(x_batch, y_batch, lora=lora)
    loss_ttt_per_doc = []
    for i, wl in enumerate(wlens):
        loss_ttt_per_doc.append(ptl_ttt[i, :wl].float().mean().item())
    tg.log(f"forward_ttt per-doc mean loss: {loss_ttt_per_doc}")

    # --- Path B: forward_logits (varlen) on concatenated docs ---
    # Flat concat with cu_seqlens
    flat_x_parts = []
    flat_y_parts = []
    cu_starts = [0]
    offset = 0
    for s, L in docs:
        flat_x_parts.append(all_tokens[s:s+L-1])
        flat_y_parts.append(all_tokens[s+1:s+L])
        offset += L - 1
        cu_starts.append(offset)
    flat_x = torch.cat(flat_x_parts).to(device=device, dtype=torch.int64).unsqueeze(0)
    flat_y = torch.cat(flat_y_parts).to(device=device, dtype=torch.int64)
    cu_bucket = 64
    padded_len = ((len(cu_starts) + cu_bucket - 1) // cu_bucket) * cu_bucket
    cu_seqlens = torch.full((padded_len,), offset, dtype=torch.int32, device=device)
    cu_seqlens[:len(cu_starts)] = torch.tensor(cu_starts, dtype=torch.int32, device=device)
    max_seq = max(L - 1 for _, L in docs)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits_vl = eval_model.forward_logits(flat_x, cu_seqlens=cu_seqlens, max_seqlen=max_seq)
    flat_nll = F.cross_entropy(logits_vl.reshape(-1, logits_vl.size(-1)).float(),
                                flat_y, reduction="none")
    loss_vl_per_doc = []
    for i, (s, L) in enumerate(docs):
        lo = cu_starts[i]
        hi = cu_starts[i+1]
        loss_vl_per_doc.append(flat_nll[lo:hi].mean().item())
    tg.log(f"forward_logits (varlen) per-doc mean loss: {loss_vl_per_doc}")

    # --- Path C: forward_logits (dense, no cu_seqlens) per doc ---
    loss_dense_per_doc = []
    for i, (s, L) in enumerate(docs):
        x_row = all_tokens[s:s+L-1].to(device=device, dtype=torch.int64).unsqueeze(0)
        y_row = all_tokens[s+1:s+L].to(device=device, dtype=torch.int64)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_d = eval_model.forward_logits(x_row)
        nll = F.cross_entropy(logits_d.reshape(-1, logits_d.size(-1)).float(),
                              y_row, reduction="none")
        loss_dense_per_doc.append(nll.mean().item())
    tg.log(f"forward_logits (dense, single-doc row) per-doc mean loss: {loss_dense_per_doc}")

    # --- Diffs ---
    for i in range(len(docs)):
        tg.log(f"doc{i} L={docs[i][1]}: ttt={loss_ttt_per_doc[i]:.4f} "
               f"varlen={loss_vl_per_doc[i]:.4f} dense_single={loss_dense_per_doc[i]:.4f} "
               f"ttt-varlen={loss_ttt_per_doc[i]-loss_vl_per_doc[i]:+.4f} "
               f"ttt-dense_single={loss_ttt_per_doc[i]-loss_dense_per_doc[i]:+.4f}")


if __name__ == "__main__":
    main()
