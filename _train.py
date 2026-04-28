import collections
import copy
import glob
import io
import lzma
import struct
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor, nn
from flash_attn_interface import flash_attn_func as flash_attn_3_func
try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_varlen_func
    _HAS_VARLEN = True
except ImportError:
    _HAS_VARLEN = False
_HAS_TRITON = False
try:
    import triton, triton.language as tl
    _HAS_TRITON = True

    @triton.jit
    def _lrelu_sq_fwd(X, Y, N: tl.constexpr, B: tl.constexpr):
        o = tl.program_id(0) * B + tl.arange(0, B)
        m = o < N
        x = tl.load(X + o, mask=m)
        y = tl.where(x >= 0, x, x * 0.5)
        tl.store(Y + o, y * y, mask=m)

    @triton.jit
    def _lrelu_sq_bwd(X, DY, DX, N: tl.constexpr, B: tl.constexpr):
        o = tl.program_id(0) * B + tl.arange(0, B)
        m = o < N
        x, dy = (tl.load(X + o, mask=m), tl.load(DY + o, mask=m))
        act = tl.where(x >= 0, x, x * 0.5)
        tl.store(DX + o, dy * 2.0 * act * tl.where(x >= 0, 1.0, 0.5), mask=m)

    class _FusedLReluSq(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            y = torch.empty_like(x)
            N, BLK = (x.numel(), 1024)
            _lrelu_sq_fwd[(N + BLK - 1) // BLK,](x, y, N, BLK)
            ctx.save_for_backward(x)
            return y

        @staticmethod
        def backward(ctx, dy):
            x, = ctx.saved_tensors
            dx = torch.empty_like(x)
            N, BLK = (x.numel(), 1024)
            _lrelu_sq_bwd[(N + BLK - 1) // BLK,](x, dy.contiguous(), dx, N, BLK)
            return dx
    fused_leaky_relu_sq = _FusedLReluSq.apply

    @triton.jit
    def _norm_rope_gain_fwd(X, Y, COS, SIN, GAIN, stride_xn, stride_cn, N, H: tl.constexpr, D: tl.constexpr, RH: tl.constexpr, T, HAS_GAIN: tl.constexpr):
        row = tl.program_id(0)
        base = row * stride_xn
        h_idx = tl.arange(0, H)
        d_idx = tl.arange(0, D)
        r_idx = tl.arange(0, RH)
        hd = h_idx[:, None] * D + d_idx[None, :]
        hr = h_idx[:, None] * D + r_idx[None, :]
        hr2 = h_idx[:, None] * D + RH + r_idx[None, :]
        x = tl.load(X + base + hd).to(tl.float32)
        var = tl.sum(x * x, axis=1) / D
        rrms = 1.0 / tl.sqrt(var + 1e-06)
        xn = x * rrms[:, None]
        xn1 = tl.load(X + base + hr).to(tl.float32) * rrms[:, None]
        xn2 = tl.load(X + base + hr2).to(tl.float32) * rrms[:, None]
        t_idx = row % T
        c = tl.load(COS + t_idx * stride_cn + r_idx).to(tl.float32)
        s = tl.load(SIN + t_idx * stride_cn + r_idx).to(tl.float32)
        r1 = xn1 * c[None, :] + xn2 * s[None, :]
        r2 = xn1 * -s[None, :] + xn2 * c[None, :]
        if HAS_GAIN:
            g = tl.load(GAIN + h_idx).to(tl.float32)
            r1 = r1 * g[:, None]
            r2 = r2 * g[:, None]
        tl.store(Y + base + hr, r1.to(tl.bfloat16))
        tl.store(Y + base + hr2, r2.to(tl.bfloat16))
        if D > 2 * RH:
            xn_out = xn
            if HAS_GAIN:
                xn_out = xn_out * g[:, None]
            pmask = d_idx[None, :] >= 2 * RH
            tl.store(Y + base + hd, xn_out.to(tl.bfloat16), mask=pmask)

    @triton.jit
    def _norm_rope_gain_bwd(X, DY, DX, COS, SIN, GAIN, DGAIN, stride_xn, stride_cn, N, H: tl.constexpr, D: tl.constexpr, RH: tl.constexpr, T, HAS_GAIN: tl.constexpr):
        row = tl.program_id(0)
        base = row * stride_xn
        h_idx = tl.arange(0, H)
        d_idx = tl.arange(0, D)
        r_idx = tl.arange(0, RH)
        hd = h_idx[:, None] * D + d_idx[None, :]
        hr = h_idx[:, None] * D + r_idx[None, :]
        hr2 = h_idx[:, None] * D + RH + r_idx[None, :]
        x = tl.load(X + base + hd).to(tl.float32)
        var = tl.sum(x * x, axis=1) / D
        rrms = 1.0 / tl.sqrt(var + 1e-06)
        xn = x * rrms[:, None]
        xn1 = tl.load(X + base + hr).to(tl.float32) * rrms[:, None]
        xn2 = tl.load(X + base + hr2).to(tl.float32) * rrms[:, None]
        t_idx = row % T
        c = tl.load(COS + t_idx * stride_cn + r_idx).to(tl.float32)
        s = tl.load(SIN + t_idx * stride_cn + r_idx).to(tl.float32)
        dy_full = tl.load(DY + base + hd).to(tl.float32)
        dy1 = tl.load(DY + base + hr).to(tl.float32)
        dy2 = tl.load(DY + base + hr2).to(tl.float32)
        if HAS_GAIN:
            g = tl.load(GAIN + h_idx).to(tl.float32)
            r1 = xn1 * c[None, :] + xn2 * s[None, :]
            r2 = xn1 * -s[None, :] + xn2 * c[None, :]
            dg = tl.sum(dy1 * r1, axis=1) + tl.sum(dy2 * r2, axis=1)
            if D > 2 * RH:
                pmask = d_idx[None, :] >= 2 * RH
                dg += tl.sum(tl.where(pmask, dy_full * xn, 0.0), axis=1)
            tl.atomic_add(DGAIN + h_idx, dg)
            dy1 = dy1 * g[:, None]
            dy2 = dy2 * g[:, None]
        dxn1 = dy1 * c[None, :] + dy2 * -s[None, :]
        dxn2 = dy1 * s[None, :] + dy2 * c[None, :]
        dot = tl.sum(dxn1 * xn1, axis=1) + tl.sum(dxn2 * xn2, axis=1)
        dy_pass = None
        if D > 2 * RH:
            if HAS_GAIN:
                dy_pass = dy_full * g[:, None]
            else:
                dy_pass = dy_full
            pmask2 = d_idx[None, :] >= 2 * RH
            dot += tl.sum(tl.where(pmask2, dy_pass * xn, 0.0), axis=1)
        corr_scale = dot[:, None] / D
        dx1 = (dxn1 - xn1 * corr_scale) * rrms[:, None]
        dx2 = (dxn2 - xn2 * corr_scale) * rrms[:, None]
        tl.store(DX + base + hr, dx1.to(tl.bfloat16))
        tl.store(DX + base + hr2, dx2.to(tl.bfloat16))
        if D > 2 * RH:
            dx_pass = (dy_pass - xn * corr_scale) * rrms[:, None]
            tl.store(DX + base + hd, dx_pass.to(tl.bfloat16), mask=pmask2)

    class _FusedNormRopeGain(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, cos, sin, gain, rope_half, has_gain):
            B, T, H, D = x.shape
            N = B * T
            x_flat = x.reshape(N, H, D).contiguous()
            y = torch.empty_like(x_flat)
            cos_2d = cos.reshape(T, rope_half).contiguous()
            sin_2d = sin.reshape(T, rope_half).contiguous()
            gain_c = gain.contiguous().float() if gain is not None else torch.empty(1, device=x.device)
            _norm_rope_gain_fwd[N,](x_flat, y, cos_2d, sin_2d, gain_c, H * D, rope_half, N, H, D, rope_half, T, has_gain)
            ctx.save_for_backward(x_flat, cos_2d, sin_2d, gain_c if has_gain else torch.empty(0, device=x.device))
            ctx.has_gain = has_gain
            ctx.shape_info = (B, T, H, D, rope_half)
            return y.reshape(B, T, H, D)

        @staticmethod
        def backward(ctx, dy):
            x_flat, cos_2d, sin_2d, gain_saved = ctx.saved_tensors
            has_gain = ctx.has_gain
            B, T, H, D, rope_half = ctx.shape_info
            N = B * T
            dy_flat = dy.reshape(N, H, D).contiguous()
            dx = torch.empty_like(x_flat)
            dgain = torch.zeros(H, dtype=torch.float32, device=x_flat.device) if has_gain else None
            _norm_rope_gain_bwd[N,](x_flat, dy_flat, dx, cos_2d, sin_2d, gain_saved if has_gain else torch.empty(1, device=x_flat.device), dgain if dgain is not None else torch.empty(1, device=x_flat.device), H * D, rope_half, N, H, D, rope_half, T, has_gain)
            return (dx.reshape(B, T, H, D), None, None, dgain, None, None)

    def fused_norm_rope_gain(x, cos, sin, gain, rope_half):
        return _FusedNormRopeGain.apply(x, cos, sin, gain, rope_half, True)

    def fused_norm_rope(x, cos, sin, rope_half):
        return _FusedNormRopeGain.apply(x, cos, sin, None, rope_half, False)
except ImportError:
    pass
_DATA_DIR = os.environ.get('DATA_DIR', './data/')

class Hyperparameters:
    seed = int(os.environ.get('SEED', 1337))
    run_id = os.environ.get('RUN_ID', str(uuid.uuid4()))
    iterations = int(os.environ.get('ITERATIONS', 20000))
    warmdown_frac = float(os.environ.get('WARMDOWN_FRAC', 0.85))
    warmup_steps = int(os.environ.get('WARMUP_STEPS', 20))
    train_batch_tokens = int(os.environ.get('TRAIN_BATCH_TOKENS', 2048 * 48 * 8))
    train_seq_len = int(os.environ.get('TRAIN_SEQ_LEN', 2048))
    max_wallclock_seconds = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 600.0))
    eval_seq_len = int(os.environ.get('EVAL_SEQ_LEN', 2048))
    vocab_size = int(os.environ.get('VOCAB_SIZE', 8192))
    num_layers = int(os.environ.get('NUM_LAYERS', 11))
    xsa_last_n = int(os.environ.get('XSA_LAST_N', 9))
    model_dim = int(os.environ.get('MODEL_DIM', 512))
    embedding_dim = int(os.environ.get('EMBEDDING_DIM', 512))
    num_kv_heads = int(os.environ.get('NUM_KV_HEADS', 4))
    num_heads = int(os.environ.get('NUM_HEADS', 8))
    mlp_mult = float(os.environ.get('MLP_MULT', 4.35))
    skip_gates_enabled = bool(int(os.environ.get('SKIP_GATES_ENABLED', '1')))
    tie_embeddings = bool(int(os.environ.get('TIE_EMBEDDINGS', '1')))
    logit_softcap = float(os.environ.get('LOGIT_SOFTCAP', 20.0))
    rope_base = float(os.environ.get('ROPE_BASE', 10000.0))
    rope_dims = int(os.environ.get('ROPE_DIMS', 32))
    ln_scale = bool(int(os.environ.get('LN_SCALE', '1')))
    qk_gain_init = float(os.environ.get('QK_GAIN_INIT', 5.25))
    num_loops = int(os.environ.get('NUM_LOOPS', 2))
    loop_start = int(os.environ.get('LOOP_START', 3))
    loop_end = int(os.environ.get('LOOP_END', 5))
    enable_looping_at = float(os.environ.get('ENABLE_LOOPING_AT', 0.5))
    min_lr = float(os.environ.get('MIN_LR', 0.0))
    embed_lr = float(os.environ.get('EMBED_LR', 0.6))
    tied_embed_lr = float(os.environ.get('TIED_EMBED_LR', 0.03))
    tied_embed_init_std = float(os.environ.get('TIED_EMBED_INIT_STD', 0.005))
    matrix_lr = float(os.environ.get('MATRIX_LR', 0.022))
    scalar_lr = float(os.environ.get('SCALAR_LR', 0.022))
    muon_momentum = float(os.environ.get('MUON_MOMENTUM', 0.97))
    muon_backend_steps = int(os.environ.get('MUON_BACKEND_STEPS', 5))
    muon_momentum_warmup_start = float(os.environ.get('MUON_MOMENTUM_WARMUP_START', 0.92))
    muon_momentum_warmup_steps = int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS', 1500))
    muon_row_normalize = bool(int(os.environ.get('MUON_ROW_NORMALIZE', '1')))
    beta1 = float(os.environ.get('BETA1', 0.9))
    beta2 = float(os.environ.get('BETA2', 0.95))
    adam_eps = float(os.environ.get('ADAM_EPS', 1e-08))
    grad_clip_norm = float(os.environ.get('GRAD_CLIP_NORM', 0.3))
    adam_wd = float(os.environ.get('ADAM_WD', 0.02))
    muon_wd = float(os.environ.get('MUON_WD', 0.095))
    embed_wd = float(os.environ.get('EMBED_WD', 0.085))
    ema_decay = float(os.environ.get('EMA_DECAY', 0.9965))
    ttt_enabled = bool(int(os.environ.get('TTT_ENABLED', '1')))
    ttt_mode = os.environ.get('TTT_MODE', 'lora')
    ttt_lr = float(os.environ.get('TTT_LR', 0.005))
    ttt_epochs = int(os.environ.get('TTT_EPOCHS', 3))
    ttt_momentum = float(os.environ.get('TTT_MOMENTUM', 0.9))
    ttt_chunk_tokens = int(os.environ.get('TTT_CHUNK_TOKENS', 32768))
    eval_stride = int(os.environ.get('EVAL_STRIDE', 64))
    ttt_lora_rank = int(os.environ.get('TTT_LORA_RANK', 48))
    ttt_lora_lr = float(os.environ.get('TTT_LORA_LR', 0.0001))
    ttt_chunk_size = int(os.environ.get('TTT_CHUNK_SIZE', 64))
    ttt_eval_seq_len = int(os.environ.get('TTT_EVAL_SEQ_LEN', 2048))
    rope_force_base_seqlen = int(os.environ.get('ROPE_FORCE_BASE_SEQLEN', 0))
    ttt_rope_base_seqlen = int(os.environ.get('TTT_ROPE_BASE_SEQLEN', 0))
    ttt_batch_size = int(os.environ.get('TTT_BATCH_SIZE', 64))
    ttt_grad_steps = int(os.environ.get('TTT_GRAD_STEPS', 1))
    ttt_grad_clip = float(os.environ.get('TTT_GRAD_CLIP', 1.0))
    ttt_weight_decay = float(os.environ.get('TTT_WEIGHT_DECAY', 0.2))
    ttt_beta1 = float(os.environ.get('TTT_BETA1', 0.9))
    ttt_beta2 = float(os.environ.get('TTT_BETA2', 0.99))
    ttt_k_lora = bool(int(os.environ.get('TTT_K_LORA', '1')))
    ttt_mlp_lora = bool(int(os.environ.get('TTT_MLP_LORA', '1')))
    ttt_o_lora = bool(int(os.environ.get('TTT_O_LORA', '1')))
    gptq_calibration_batches = int(os.environ.get('GPTQ_CALIBRATION_BATCHES', 128))
    gptq_reserve_seconds = float(os.environ.get('GPTQ_RESERVE_SECONDS', 12.0))
    matrix_bits = int(os.environ.get('MATRIX_BITS', 6))
    embed_bits = int(os.environ.get('EMBED_BITS', 8))
    matrix_clip_sigmas = float(os.environ.get('MATRIX_CLIP_SIGMAS', 15))
    embed_clip_sigmas = float(os.environ.get('EMBED_CLIP_SIGMAS', 20.0))
    hessian_clip_lambda = float(os.environ.get('HESSIAN_CLIP_LAMBDA', 0.0))
    clip_mult_early = float(os.environ.get('CLIP_MULT_EARLY', 1.0))
    clip_mult_loop = float(os.environ.get('CLIP_MULT_LOOP', 1.0))
    clip_mult_mid = float(os.environ.get('CLIP_MULT_MID', 1.0))
    clip_mult_late = float(os.environ.get('CLIP_MULT_LATE', 1.0))
    loop_layer_bits = int(os.environ.get('LOOP_LAYER_BITS', 0))
    loop_layer_clip_sigmas = float(os.environ.get('LOOP_LAYER_CLIP_SIGMAS', 0))
    parallel_residual_start = int(os.environ.get('PARALLEL_RESIDUAL_START', 9))
    gated_attention = bool(int(os.environ.get('GATED_ATTENTION', '1')))
    fused_mlp = bool(int(os.environ.get('FUSED_MLP', '1')))
    fused_rope = bool(int(os.environ.get('FUSED_ROPE', '1')))
    cpu_prefetch_depth = int(os.environ.get('CPU_PREFETCH_DEPTH', 1))
    gpu_prefetch_depth = int(os.environ.get('GPU_PREFETCH_DEPTH', 1))
    loop_embeddings = bool(int(os.environ.get('LOOP_EMBEDDINGS', '1')))
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size
    datasets_dir = os.path.join(_DATA_DIR, 'datasets', f'fineweb10B_sp{vocab_size}')
    train_files = os.path.join(datasets_dir, 'fineweb_train_*.bin')
    val_files = os.path.join(datasets_dir, 'fineweb_val_*.bin')
    tokenizer_path = os.path.join(_DATA_DIR, 'tokenizers', f'fineweb_{vocab_size}_bpe.model')
    logfile = f'logs/{run_id}.txt'
    model_path = 'final_model.pt'
    quantized_model_path = 'final_model.int6.ptz'
_logger_hparams = None

def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams = h

def log(msg, console=True):
    if _logger_hparams is None:
        print(msg)
        return
    if _logger_hparams.is_main_process:
        if console:
            print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, 'a', encoding='utf-8') as f:
                print(msg, file=f)

class ValidationData:

    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(f'VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}')
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = build_sentencepiece_luts(self.sp, h.vocab_size, device)

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    assert sp.piece_to_id('▁') != sp.unk_id(), "Tokenizer must have '▁' (space) as its own token for correct BPB byte counting"
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith('▁'):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode('utf-8'))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device), torch.tensor(has_leading_space_np, dtype=torch.bool, device=device), torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f'No files found for pattern: {pattern}')
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = (tokens.numel() - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError(f'Validation split is too short for TRAIN_SEQ_LEN={seq_len}')
    return tokens[:usable + 1]

def load_data_shard(file):
    header_bytes = 256 * np.dtype('<i4').itemsize
    token_bytes = np.dtype('<u2').itemsize
    header = np.fromfile(file, dtype='<i4', count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f'Unexpected shard header for {file}')
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f'Shard size mismatch for {file}: expected {expected_size} bytes')
    tokens_np = np.fromfile(file, dtype='<u2', count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f'Short read for {file}')
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))
BOS_ID = 1

def _get_next_multiple_of_n(v, n):
    return (v + n - 1) // n * n

def _build_cu_seqlens(bos_pos, total_len, device, max_doc_len=0, bucket_size=64):
    if not bos_pos or bos_pos[0] != 0:
        bos_pos = [0] + bos_pos
    seg_starts = []
    starts_with_end = bos_pos + [total_len]
    for i in range(len(starts_with_end) - 1):
        start = starts_with_end[i]
        end = starts_with_end[i + 1]
        if max_doc_len > 0:
            pos = start
            while pos < end:
                seg_starts.append(pos)
                pos += max_doc_len
        else:
            seg_starts.append(start)
    boundaries = seg_starts + [total_len]
    padded_len = _get_next_multiple_of_n(len(boundaries), bucket_size)
    cu = torch.full((padded_len,), total_len, dtype=torch.int32, device=device)
    cu[:len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32, device=device)
    seg_ends = seg_starts[1:] + [total_len]
    max_seqlen = max((end - start for start, end in zip(seg_starts, seg_ends)))
    return (cu, max_seqlen)

class DocumentPackingLoader:
    _shard_pool = ThreadPoolExecutor(1)

    def __init__(self, h, device, cu_bucket_size=64):
        self.rank = h.rank
        self.world_size = h.world_size
        self.device = device
        self.cu_bucket_size = cu_bucket_size
        self.max_seq_len = h.train_seq_len
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f'No files found for pattern: {h.train_files}')
        self.files = all_files
        self.file_iter = iter(self.files)
        self._init_shard(load_data_shard(next(self.file_iter)))
        self._next_shard = self._submit_next_shard()
        self.cpu_prefetch_depth = max(1, int(getattr(h, 'cpu_prefetch_depth', 2)))
        self._batch_pool = ThreadPoolExecutor(1)
        self._next_batches = collections.deque()

    def _init_shard(self, tokens):
        self.tokens = tokens
        self.shard_size = tokens.numel()
        self.bos_idx = (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        if self.bos_idx.size == 0:
            self.bos_idx = np.array([0], dtype=np.int64)
        self.cursor = int(self.bos_idx[0])

    def _submit_next_shard(self):
        try:
            path = next(self.file_iter)
            return self._shard_pool.submit(load_data_shard, path)
        except StopIteration:
            return None

    def _advance_shard(self):
        if self._next_shard is None:
            self.file_iter = iter(self.files)
            self._next_shard = self._shard_pool.submit(load_data_shard, next(self.file_iter))
        self._init_shard(self._next_shard.result())
        self._next_shard = self._submit_next_shard()

    def _local_doc_starts(self, local_start, total_len):
        lo = np.searchsorted(self.bos_idx, local_start, side='left')
        hi = np.searchsorted(self.bos_idx, local_start + total_len, side='left')
        return (self.bos_idx[lo:hi] - local_start).tolist()

    def _prepare_batch(self, num_tokens_local, max_seq_len):
        per_rank_span = num_tokens_local + 1
        global_span = per_rank_span * self.world_size
        while self.cursor + global_span > self.shard_size:
            self._advance_shard()
        local_start = self.cursor + self.rank * per_rank_span
        buf = self.tokens[local_start:local_start + per_rank_span]
        inputs = buf[:-1].to(dtype=torch.int64).pin_memory()
        targets = buf[1:].to(dtype=torch.int64).pin_memory()
        starts = self._local_doc_starts(local_start, inputs.numel())
        cu_seqlens, max_seqlen = _build_cu_seqlens(starts, inputs.numel(), inputs.device, max_seq_len, self.cu_bucket_size)
        cu_seqlens = cu_seqlens.pin_memory()
        self.cursor += global_span
        return (inputs, targets, cu_seqlens, max_seqlen)

    def next_batch(self, global_tokens, grad_accum_steps):
        num_tokens_local = global_tokens // (self.world_size * grad_accum_steps)
        while len(self._next_batches) < self.cpu_prefetch_depth:
            self._next_batches.append(self._batch_pool.submit(self._prepare_batch, num_tokens_local, self.max_seq_len))
        inputs, targets, cu_seqlens, max_seqlen = self._next_batches.popleft().result()
        while len(self._next_batches) < self.cpu_prefetch_depth:
            self._next_batches.append(self._batch_pool.submit(self._prepare_batch, num_tokens_local, self.max_seq_len))
        return (inputs[None].to(self.device, non_blocking=True), targets[None].to(self.device, non_blocking=True), cu_seqlens.to(self.device, non_blocking=True), max_seqlen)

class RMSNorm(nn.Module):

    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):

    def forward(self, x):
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

class Rotary(nn.Module):

    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        t = torch.arange(train_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('_cos_cached', freqs.cos()[None, :, None, :].clone(), persistent=False)
        self.register_buffer('_sin_cached', freqs.sin()[None, :, None, :].clone(), persistent=False)

    def forward(self, seq_len, device, dtype, ntk_seq_len=0):
        force = int(os.environ.get('ROPE_FORCE_BASE_SEQLEN', '0'))
        if force > 0:
            ntk_seq_len = force
        n = max(seq_len, ntk_seq_len) if ntk_seq_len > 0 else seq_len
        if n <= self.train_seq_len:
            return (self._cos_cached[:, :seq_len].to(dtype=dtype), self._sin_cached[:, :seq_len].to(dtype=dtype))
        rd = self.rope_dims
        scale = n / self.train_seq_len
        new_base = self.base * scale ** (rd / (rd - 2))
        inv_freq = 1.0 / new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd)
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        return (freqs.cos()[None, :, None, :].to(dtype=dtype), freqs.sin()[None, :, None, :].to(dtype=dtype))

def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = (x[..., :rope_dims], x[..., rope_dims:])
        half = rope_dims // 2
        x1, x2 = (x_rope[..., :half], x_rope[..., half:])
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = (x[..., :half], x[..., half:])
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len, gated=False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('model_dim must be divisible by num_heads')
        if num_heads % num_kv_heads != 0:
            raise ValueError('num_heads must be divisible by num_kv_heads')
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError('head_dim must be even for RoPE')
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = False
        self.use_varlen = False
        self.use_fused_rope = False
        self.gated = gated
        if gated:
            self.gate_proj = CastedLinear(dim, num_heads, bias=True)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 2.94)

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x, cu_seqlens=None, max_seqlen=None):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        cos, sin = self.rotary(seqlen, x.device, q.dtype, ntk_seq_len=getattr(self, 'eval_ntk_seq_len', 0))
        rope_half = self.rope_dims // 2 if self.rope_dims > 0 else self.head_dim // 2
        if self.use_fused_rope:
            q = fused_norm_rope_gain(q, cos, sin, self.q_gain, rope_half)
            k = fused_norm_rope(k, cos, sin, rope_half)
        else:
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            q = apply_rotary_emb(q, cos, sin, self.rope_dims)
            k = apply_rotary_emb(k, cos, sin, self.rope_dims)
            q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if self.use_varlen and cu_seqlens is not None and _HAS_VARLEN:
            q_flat = q.reshape(-1, self.num_heads, self.head_dim)
            k_flat = k.reshape(-1, self.num_kv_heads, self.head_dim)
            v_flat = v.reshape(-1, self.num_kv_heads, self.head_dim)
            y = flash_attn_3_varlen_func(q_flat, k_flat, v_flat, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        else:
            y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        if self.gated:
            gate = torch.sigmoid(self.gate_proj(x))
            y = y * gate.unsqueeze(-1)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)

class MLP(nn.Module):

    def __init__(self, dim, mlp_mult, use_fused=False):
        super().__init__()
        hidden = int(mlp_mult * dim)
        hidden = (hidden + 63) // 64 * 64
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.use_fused = use_fused and _HAS_TRITON

    def forward(self, x):
        h = self.fc(x)
        if self.use_fused:
            return self.proj(fused_leaky_relu_sq(h))
        return self.proj(F.leaky_relu(h, negative_slope=0.5).square())

class BatchedLinearLoRA(nn.Module):

    def __init__(self, bsz, in_features, out_features, rank):
        super().__init__()
        self._bound = 1.0 / math.sqrt(in_features)
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features).uniform_(-self._bound, self._bound))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))

    def reset(self):
        with torch.no_grad():
            self.A.uniform_(-self._bound, self._bound)
            self.B.zero_()

    def forward(self, x):
        return x @ self.A.transpose(1, 2) @ self.B.transpose(1, 2)

class BatchedTTTLoRA(nn.Module):

    def __init__(self, bsz, model, rank, k_lora=True, mlp_lora=True, o_lora=True):
        super().__init__()
        self.bsz = bsz
        dim = model.blocks[0].attn.c_q.weight.shape[0]
        kv_dim = model.blocks[0].attn.c_k.weight.shape[0]
        vocab = model.tok_emb.num_embeddings
        embed_dim = model.tok_emb.embedding_dim
        if getattr(model, 'looping_active', False):
            num_slots = len(model.encoder_indices) + len(model.decoder_indices)
        else:
            num_slots = len(model.blocks)
        self.lm_head_lora = BatchedLinearLoRA(bsz, embed_dim, vocab, rank)
        self.q_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)])
        self.v_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(num_slots)])
        self.k_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(num_slots)]) if k_lora else None
        self.mlp_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]) if mlp_lora else None
        self.o_loras = nn.ModuleList([BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]) if o_lora else None

    def reset(self):
        with torch.no_grad():
            self.lm_head_lora.reset()
            for loras in [self.q_loras, self.v_loras, self.k_loras, self.mlp_loras, self.o_loras]:
                if loras is not None:
                    for lora in loras:
                        lora.reset()

class Block(nn.Module):

    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, train_seq_len, layer_idx=0, ln_scale=False, gated_attention=False, use_fused_mlp=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len, gated=gated_attention)
        self.mlp = MLP(dim, mlp_mult, use_fused=use_fused_mlp)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel = False

    def forward(self, x, x0, cu_seqlens=None, max_seqlen=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        normed_attn = self.attn_norm(x_in) * self.ln_scale_factor
        attn_out = self.attn(normed_attn, cu_seqlens, max_seqlen)
        if self.parallel:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
        else:
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out

class GPT(nn.Module):

    def __init__(self, h):
        super().__init__()
        if h.logit_softcap <= 0.0:
            raise ValueError(f'logit_softcap must be positive, got {h.logit_softcap}')
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.use_varlen = _HAS_VARLEN
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        if h.embedding_dim != h.model_dim:
            self.embed_proj = CastedLinear(h.embedding_dim, h.model_dim, bias=False)
            self.head_proj = CastedLinear(h.model_dim, h.embedding_dim, bias=False)
        else:
            self.embed_proj = None
            self.head_proj = None
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        self.blocks = nn.ModuleList([Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult, h.rope_base, h.qk_gain_init, h.train_seq_len, layer_idx=i, ln_scale=h.ln_scale, gated_attention=h.gated_attention, use_fused_mlp=h.fused_mlp) for i in range(h.num_layers)])
        if h.rope_dims > 0:
            head_dim = h.model_dim // h.num_heads
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary = Rotary(head_dim, base=h.rope_base, train_seq_len=h.train_seq_len, rope_dims=h.rope_dims)
        if h.fused_rope and _HAS_TRITON:
            for block in self.blocks:
                block.attn.use_fused_rope = True
        self.final_norm = RMSNorm()
        self.lm_head = None if h.tie_embeddings else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if h.xsa_last_n > 0:
            for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True
        if h.parallel_residual_start >= 0:
            for i in range(h.parallel_residual_start, h.num_layers):
                self.blocks[i].parallel = True
        if self.use_varlen:
            for block in self.blocks:
                block.attn.use_varlen = True
        self.looping_active = False
        if h.num_loops > 0:
            loop_seg = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, h.num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_enc]
            self.decoder_indices = all_indices[num_enc:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))
        self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)) if h.skip_gates_enabled else None
        self.loop_embeddings_enabled = h.loop_embeddings and h.num_loops > 0
        if self.loop_embeddings_enabled:
            total_passes = h.num_loops + 1
            self.loop_embs = nn.Parameter(torch.zeros(total_passes, h.model_dim, dtype=torch.float32))
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, '_zero_init', False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and (module.weight.shape[1] >= 64):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _add_loop_emb(self, x, layer_idx, counter):
        if not self.loop_embeddings_enabled:
            return x
        p = counter.get(layer_idx, 0)
        counter[layer_idx] = p + 1
        if hasattr(self, 'loop_embs') and p < self.loop_embs.size(0):
            emb = self.loop_embs[p].to(dtype=x.dtype)[None, None, :]
            if self.looping_active:
                x = x + emb
            else:
                x = x + emb * 0
        return x

    def forward_logits(self, input_ids, cu_seqlens=None, max_seqlen=None):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips = []
        enc_iter = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
        dec_iter = self.decoder_indices if self.looping_active else range(self.num_encoder_layers, self.num_encoder_layers + self.num_decoder_layers)
        lpc = {}
        for i in enc_iter:
            x = self._add_loop_emb(x, i, lpc)
            x = self.blocks[i](x, x0, cu_seqlens, max_seqlen)
            skips.append(x)
        for skip_idx, i in enumerate(dec_iter):
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip
            x = self._add_loop_emb(x, i, lpc)
            x = self.blocks[i](x, x0, cu_seqlens, max_seqlen)
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def _block_with_lora(self, block, x, x0, lora, slot):
        mix = block.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = block.attn_norm(x_in) * block.ln_scale_factor
        attn = block.attn
        bsz, seqlen, dim = n.shape
        q = (attn.c_q(n) + lora.q_loras[slot](n)).reshape(bsz, seqlen, attn.num_heads, attn.head_dim)
        k = attn.c_k(n)
        if lora.k_loras is not None:
            k = k + lora.k_loras[slot](n)
        k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
        v = (attn.c_v(n) + lora.v_loras[slot](n)).reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
        cos, sin = attn.rotary(seqlen, n.device, q.dtype)
        rope_half = attn.rope_dims // 2 if attn.rope_dims > 0 else attn.head_dim // 2
        if attn.use_fused_rope:
            q = fused_norm_rope_gain(q, cos, sin, attn.q_gain, rope_half)
            k = fused_norm_rope(k, cos, sin, rope_half)
        else:
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            q = apply_rotary_emb(q, cos, sin, attn.rope_dims)
            k = apply_rotary_emb(k, cos, sin, attn.rope_dims)
            q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if attn.use_xsa:
            y = attn._xsa_efficient(y, v)
        if attn.gated:
            gate = torch.sigmoid(attn.gate_proj(n))
            y = y * gate.unsqueeze(-1)
        y = y.reshape(bsz, seqlen, dim)
        attn_out = attn.proj(y)
        if lora.o_loras is not None:
            attn_out = attn_out + lora.o_loras[slot](n)
        if block.parallel:
            mlp_n = block.mlp_norm(x_in) * block.ln_scale_factor
            mlp_out = block.mlp(mlp_n)
            if lora.mlp_loras is not None:
                mlp_out = mlp_out + lora.mlp_loras[slot](mlp_n)
            x_out = x_in + block.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out + block.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
        else:
            x_out = x_in + block.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            mlp_n = block.mlp_norm(x_out) * block.ln_scale_factor
            mlp_out = block.mlp(mlp_n)
            if lora.mlp_loras is not None:
                mlp_out = mlp_out + lora.mlp_loras[slot](mlp_n)
            x_out = x_out + block.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * mlp_out
        return x_out

    def forward_ttt(self, input_ids, target_ids, lora):
        logits = self._forward_ttt_logits(input_ids, lora)
        bsz, sl, V = logits.shape
        return F.cross_entropy(logits.float().reshape(-1, V), target_ids.reshape(-1), reduction='none').reshape(bsz, sl)

    def _forward_ttt_logits(self, input_ids, lora):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips = []
        enc_iter = self.encoder_indices if self.looping_active else list(range(self.num_encoder_layers))
        dec_iter = self.decoder_indices if self.looping_active else list(range(self.num_encoder_layers, self.num_encoder_layers + self.num_decoder_layers))
        lpc = {}
        slot = 0
        for i in enc_iter:
            x = self._add_loop_emb(x, i, lpc)
            x = self._block_with_lora(self.blocks[i], x, x0, lora, slot)
            slot += 1
            skips.append(x)
        for skip_idx, i in enumerate(dec_iter):
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip
            x = self._add_loop_emb(x, i, lpc)
            x = self._block_with_lora(self.blocks[i], x, x0, lora, slot)
            slot += 1
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = logits + lora.lm_head_lora(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits

    def forward(self, input_ids, target_ids, cu_seqlens=None, max_seqlen=0):
        logits = self.forward_logits(input_ids, cu_seqlens, max_seqlen)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction='mean')

def classify_param(name):
    if 'tok_emb' in name or 'lm_head' in name:
        return 'embed'
    if '.mlp.' in name:
        return 'mlp'
    if '.attn.' in name or ('.proj.' in name and '.mlp.' not in name):
        return 'attn'
    return 'other'

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-07):
    a, b, c = (3.4445, -4.775, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):

    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=0.0, row_normalize=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay, row_normalize=row_normalize))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group['params']
            if not params:
                continue
            lr = group['lr']
            momentum = group['momentum']
            backend_steps = group['backend_steps']
            nesterov = group['nesterov']
            total_params = sum((int(p.numel()) for p in params))
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    if group.get('row_normalize', False):
                        row_norms = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-07)
                        g = g / row_norms.to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get('weight_decay', 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss
CONTROL_TENSOR_NAME_PATTERNS = tuple((pattern for pattern in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS', 'attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,gate_proj,loop_embs').split(',') if pattern))

class Optimizers:

    def __init__(self, h, base_model):
        block_named_params = list(base_model.blocks.named_parameters())
        matrix_params = [p for name, p in block_named_params if p.ndim == 2 and (not any((pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)))]
        scalar_params = [p for name, p in block_named_params if p.ndim < 2 or any((pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))]
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
            scalar_params.append(base_model.skip_gates)
        if hasattr(base_model, 'loop_embs'):
            scalar_params.append(base_model.loop_embs)
        if base_model.lm_head is not None:
            scalar_params.append(base_model.lm_head.weight)
        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tok_params = [{'params': [base_model.tok_emb.weight], 'lr': token_lr, 'base_lr': token_lr}]
        self.optimizer_tok = torch.optim.AdamW(tok_params, betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.embed_wd, fused=True)
        self.optimizer_muon = Muon(matrix_params, lr=h.matrix_lr, momentum=h.muon_momentum, backend_steps=h.muon_backend_steps, weight_decay=h.muon_wd, row_normalize=h.muon_row_normalize)
        for group in self.optimizer_muon.param_groups:
            group['base_lr'] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW([{'params': scalar_params, 'lr': h.scalar_lr, 'base_lr': h.scalar_lr}], betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.adam_wd, fused=True)
        self.optimizers = [self.optimizer_tok, self.optimizer_muon, self.optimizer_scalar]

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def step(self):
        for opt in self.optimizers:
            opt.step()
        self.zero_grad_all()

def restore_fp32_params(model):
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, param in model.named_parameters():
        if (param.ndim < 2 or any((pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))) and param.dtype != torch.float32:
            param.data = param.data.float()

def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}
    hooks = []

    def make_hook(name):

        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
            hessians[name].addmm_(x.T, x)
        return hook_fn
    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > 65536:
            cat = classify_param(name + '.weight')
            if cat in ('mlp', 'attn'):
                hooks.append(module.register_forward_hook(make_hook(name + '.weight')))
    if model.tie_embeddings:
        hook_module = model.head_proj if model.head_proj is not None else model.final_norm

        def make_output_hook(name):

            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(x.T, x)
            return hook_fn
        hooks.append(hook_module.register_forward_hook(make_output_hook('tok_emb.weight')))
    model.eval()
    model.bfloat16()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _, cu, ms = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x, cu, ms)
    model.float()
    for hook in hooks:
        hook.remove()
    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_calibration_batches
    return hessians

def gptq_quantize_weight(w, H, clip_sigmas=3.0, clip_range=63, block_size=128, hessian_clip_lambda=0.0):
    W_orig = w.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * H.diag().mean()
    H.diagonal().add_(damp)
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    row_std = W_orig.std(dim=1)
    if hessian_clip_lambda > 0.0:
        diagH = torch.diag(H[invperm][:, invperm]).clamp_min(1e-08)
        col_importance = diagH / diagH.mean()
        row_importance = (W_orig.abs() * col_importance.unsqueeze(0)).mean(dim=1)
        row_importance = row_importance / row_importance.mean()
        adj = 1.0 + hessian_clip_lambda * (row_importance - 1.0)
        s = (clip_sigmas * row_std * adj / clip_range).clamp_min(1e-10).to(torch.float16)
    else:
        s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return (Q[:, invperm], s)

def _is_loop_layer(name, h):
    m = re.search('blocks\\.(\\d+)\\.', name)
    if m:
        idx = int(m.group(1))
        return h.loop_start <= idx <= h.loop_end
    return False

def _get_group_clip_mult(name, h):
    m = re.search('blocks\\.(\\d+)\\.', name)
    if not m:
        return 1.0
    idx = int(m.group(1))
    if idx <= 2:
        return h.clip_mult_early
    if h.loop_start <= idx <= h.loop_end:
        return h.clip_mult_loop
    if idx in (3, 6, 7):
        return h.clip_mult_mid
    if idx >= 8:
        return h.clip_mult_late
    return 1.0

def gptq_mixed_quantize(state_dict, hessians, h):
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = 'passthrough (float16)'
            continue
        if 'tok_emb' in name:
            cs = h.embed_clip_sigmas
            bits = h.embed_bits
        elif _is_loop_layer(name, h) and h.loop_layer_bits > 0:
            cs = h.loop_layer_clip_sigmas if h.loop_layer_clip_sigmas > 0 else h.matrix_clip_sigmas
            bits = h.loop_layer_bits
            log(f'  loop_layer_quant: {name} -> int{bits} clip={cs:.2f}')
        else:
            cs = h.matrix_clip_sigmas
            bits = h.matrix_bits
        group_mult = _get_group_clip_mult(name, h)
        if group_mult != 1.0:
            cs = cs * group_mult
        clip_range = 2 ** (bits - 1) - 1
        H = hessians.get(name)
        if H is not None and H.shape[0] == t.shape[1]:
            q, s = gptq_quantize_weight(t, H, clip_sigmas=cs, clip_range=clip_range, hessian_clip_lambda=h.hessian_clip_lambda)
            meta[name] = f'gptq (int{bits})'
        else:
            row_std = t.float().std(dim=1)
            if H is not None and H.shape[0] == t.shape[0]:
                freq = H.diag().float()
                freq_weight = (freq / freq.mean().clamp_min(1e-08)).sqrt().clamp(0.5, 2.0)
                s = (cs * row_std / (clip_range * freq_weight)).clamp_min(1e-10).to(torch.float16)
                meta[name] = f'freq-weighted (int{bits})'
            else:
                s = (cs * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
                meta[name] = f'absmax (int{bits})'
            q = torch.clamp(torch.round(t.float() / s.float().unsqueeze(1)), -clip_range, clip_range).to(torch.int8)
        result[name + '.q'] = q
        result[name + '.scale'] = s
    categories = collections.defaultdict(set)
    for name, cat in meta.items():
        short = re.sub('\\.\\d+$', '', re.sub('blocks\\.\\d+', 'blocks', name))
        categories[cat].add(short)
    log('Quantized weights:')
    for cat in sorted(categories):
        log(f'  {cat}: {', '.join(sorted(categories[cat]))}')
    return (result, meta)

def dequantize_mixed(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if 'passthrough' in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = (result[name + '.q'], result[name + '.scale'])
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *[1] * (q.ndim - 1))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out
_RANS_PREC = 16
_RANS_TOTAL = 1 << _RANS_PREC
_RANS_LOWER = 1 << 23

def _rans_build_freqs(symbols, num_sym):
    counts = collections.Counter(symbols.tolist())
    total = len(symbols)
    freqs = [max(1, round(counts.get(s, 0) / total * _RANS_TOTAL)) for s in range(num_sym)]
    diff = _RANS_TOTAL - sum(freqs)
    if diff != 0:
        best = max(range(num_sym), key=lambda s: freqs[s])
        freqs[best] = max(1, freqs[best] + diff)
    return freqs

def _rans_cumfreqs(freqs):
    cum = [0] * (len(freqs) + 1)
    for i in range(len(freqs)):
        cum[i + 1] = cum[i] + freqs[i]
    return cum

def _rans_encode(symbols, freqs):
    cumfreqs = _rans_cumfreqs(freqs)
    state = _RANS_LOWER
    output = bytearray()
    for i in range(len(symbols) - 1, -1, -1):
        s = int(symbols[i])
        freq = freqs[s]
        start = cumfreqs[s]
        max_state = freq * (_RANS_LOWER >> _RANS_PREC) << 8
        while state >= max_state:
            output.append(state & 255)
            state >>= 8
        state = state // freq * _RANS_TOTAL + state % freq + start
    for _ in range(4):
        output.append(state & 255)
        state >>= 8
    output.reverse()
    return bytes(output)

def _rans_decode(data, freqs, count):
    cumfreqs = _rans_cumfreqs(freqs)
    num_sym = len(freqs)
    sym_table = [0] * _RANS_TOTAL
    for s in range(num_sym):
        for j in range(cumfreqs[s], cumfreqs[s + 1]):
            sym_table[j] = s
    pos = 0
    state = 0
    for _ in range(4):
        state = state << 8 | data[pos]
        pos += 1
    symbols = np.zeros(count, dtype=np.uint8)
    for i in range(count):
        slot = state % _RANS_TOTAL
        s = sym_table[slot]
        symbols[i] = s
        freq = freqs[s]
        start = cumfreqs[s]
        state = freq * (state // _RANS_TOTAL) + state % _RANS_TOTAL - start
        while state < _RANS_LOWER and pos < len(data):
            state = state << 8 | data[pos]
            pos += 1
    return symbols

def _ans_compress_quant(quant_result, quant_meta):
    result = bytearray(b'ANSW')
    q_keys = sorted((k for k in quant_result if k.endswith('.q')))
    other_keys = sorted((k for k in quant_result if not k.endswith('.q')))
    other_dict = {k: quant_result[k] for k in other_keys}
    other_buf = io.BytesIO()
    torch.save({'w_other': other_dict, 'm': quant_meta}, other_buf)
    other_blob = lzma.compress(other_buf.getvalue(), preset=6)
    result.extend(struct.pack('<I', len(other_blob)))
    result.extend(other_blob)
    result.extend(struct.pack('<I', len(q_keys)))
    for key in q_keys:
        q_tensor = quant_result[key]
        q_np = q_tensor.numpy().astype(np.int8)
        shape = list(q_np.shape)
        flat = q_np.flatten()
        flat_u = (flat.astype(np.int16) + 127).astype(np.uint8)
        num_sym = 256
        freqs = _rans_build_freqs(flat_u, num_sym)
        compressed = _rans_encode(flat_u, freqs)
        key_bytes = key.encode()
        result.extend(struct.pack('<H', len(key_bytes)))
        result.extend(key_bytes)
        result.extend(struct.pack('<B', len(shape)))
        for d in shape:
            result.extend(struct.pack('<I', d))
        for f in freqs:
            result.extend(struct.pack('<H', f))
        result.extend(struct.pack('<I', len(compressed)))
        result.extend(compressed)
    return bytes(result)

def _ans_decompress_quant(data):
    assert data[:4] == b'ANSW', f'Bad ANS magic: {data[:4]}'
    pos = 4
    other_len = struct.unpack('<I', data[pos:pos + 4])[0]
    pos += 4
    other_blob = data[pos:pos + other_len]
    pos += other_len
    other_state = torch.load(io.BytesIO(lzma.decompress(other_blob)), map_location='cpu')
    result = dict(other_state['w_other'])
    meta = other_state['m']
    num_q = struct.unpack('<I', data[pos:pos + 4])[0]
    pos += 4
    for _ in range(num_q):
        key_len = struct.unpack('<H', data[pos:pos + 2])[0]
        pos += 2
        key = data[pos:pos + key_len].decode()
        pos += key_len
        n_dims = struct.unpack('<B', data[pos:pos + 1])[0]
        pos += 1
        shape = []
        for _ in range(n_dims):
            shape.append(struct.unpack('<I', data[pos:pos + 4])[0])
            pos += 4
        num_sym = 256
        freqs = []
        for _ in range(num_sym):
            freqs.append(struct.unpack('<H', data[pos:pos + 2])[0])
            pos += 2
        data_len = struct.unpack('<I', data[pos:pos + 4])[0]
        pos += 4
        compressed = data[pos:pos + data_len]
        pos += data_len
        count = 1
        for d in shape:
            count *= d
        flat_u = _rans_decode(compressed, freqs, count)
        flat_s = flat_u.astype(np.int16) - 127
        q_np = flat_s.astype(np.int8).reshape(shape)
        result[key] = torch.from_numpy(q_np)
    return (result, meta)

def serialize(h, base_model, code):
    code_bytes = len(code.encode('utf-8'))
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f'Serialized model: {model_bytes} bytes')
        log(f'Code size: {code_bytes} bytes')
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    device = torch.device('cuda', h.local_rank)
    log('GPTQ:collecting Hessians from calibration data...')
    t0 = time.perf_counter()
    calib_loader = DocumentPackingLoader(h, device)
    hessians = collect_hessians(base_model, calib_loader, h, device, n_calibration_batches=h.gptq_calibration_batches)
    log(f'GPTQ:collected {len(hessians)} Hessians in {time.perf_counter() - t0:.1f}s')
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)
    t1 = time.perf_counter()
    quant_blob = _ans_compress_quant(quant_result, quant_meta)
    log(f'compress:ans {len(quant_blob)} bytes ({time.perf_counter() - t1:.1f}s)')
    log(f'  ans: model={len(quant_blob)} total={len(quant_blob) + code_bytes}')
    quant_file_bytes = len(quant_blob)
    bytes_total = quant_file_bytes + code_bytes
    if h.is_main_process:
        with open(h.quantized_model_path, 'wb') as f:
            f.write(quant_blob)
        log(f'Serialized model quantized+ans: {quant_file_bytes} bytes')
        log(f'Total submission size quantized+ans: {bytes_total} bytes')
    return (bytes_total, quant_file_bytes)

def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    sd_cpu = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
    with open(h.quantized_model_path, 'rb') as f:
        quant_blob_disk = f.read()
    quant_w, quant_m = _ans_decompress_quant(quant_blob_disk)
    quant_state = {'w': quant_w, 'm': quant_m}
    deq_state = dequantize_mixed(quant_state['w'], quant_state['m'], sd_cpu)
    eval_model.load_state_dict(deq_state, strict=True)
    return eval_model

def _find_docs(all_tokens):
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) if i + 1 < len(bos_positions) else all_tokens.numel()
        if i + 1 < len(bos_positions):
            end += 1
        if end - start >= 2:
            docs.append((start, end - start))
    return docs

def _compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len):
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_start = ci * chunk_size
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return (win_start, win_len, chunk_offset, chunk_len)

def _build_ttt_global_batches(doc_entries, h, ascending=False):
    batch_size = h.ttt_batch_size
    global_doc_entries = sorted(doc_entries, key=lambda x: x[1][1])
    global_batches = [global_doc_entries[i:i + batch_size] for i in range(0, len(global_doc_entries), batch_size)]
    indexed = list(enumerate(global_batches))
    if not ascending:
        indexed.sort(key=lambda ib: -max((dl for _, (_, dl) in ib[1])))
    return indexed

def _init_batch_counter(path):
    with open(path, 'wb') as f:
        f.write(0 .to_bytes(4, 'little'))

def _claim_next_batch(counter_path, queue_len):
    import fcntl
    try:
        with open(counter_path, 'r+b') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            idx = int.from_bytes(f.read(4), 'little')
            f.seek(0)
            f.write((idx + 1).to_bytes(4, 'little'))
            f.flush()
    except FileNotFoundError:
        return queue_len
    return idx

def _accumulate_bpb(ptl, x, y, chunk_offsets, chunk_lens, pos_idx, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, loss_sum, byte_sum, token_count):
    pos = pos_idx[:x.size(1)].unsqueeze(0)
    mask = (chunk_lens.unsqueeze(1) > 0) & (pos >= chunk_offsets.unsqueeze(1)) & (pos < (chunk_offsets + chunk_lens).unsqueeze(1))
    mask_f64 = mask.to(torch.float64)
    tok_bytes = base_bytes_lut[y].to(torch.float64)
    tok_bytes += (has_leading_space_lut[y] & ~is_boundary_token_lut[x]).to(torch.float64)
    loss_sum += (ptl.to(torch.float64) * mask_f64).sum()
    byte_sum += (tok_bytes * mask_f64).sum()
    token_count += chunk_lens.to(torch.float64).sum()

def eval_val_ttt(h, device, val_data, base_model, forward_ttt_train=None, batch_seqs=32):
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    all_tokens = val_data.val_tokens
    all_tokens_idx = all_tokens.to(torch.int32)
    docs = _find_docs(all_tokens)
    doc_entries = list(enumerate(docs))
    chunk_size, eval_seq_len = (h.ttt_chunk_size, h.ttt_eval_seq_len)
    ttt_bsz = h.ttt_batch_size
    log(f'ttt_lora:docs:{len(doc_entries)} rank:{h.ttt_lora_rank} lr:{h.ttt_lora_lr} chunk:{chunk_size}')
    if forward_ttt_train is None:
        forward_ttt_train = base_model.forward_ttt
    global_batches_sorted = _build_ttt_global_batches(doc_entries, h)
    queue_len = len(global_batches_sorted)
    counter_path = f'/tmp/ttt_counter_{h.run_id}'
    if h.rank == 0:
        _init_batch_counter(counter_path)
    if dist.is_available() and dist.is_initialized():
        path_list = [counter_path]
        dist.broadcast_object_list(path_list, src=0)
        counter_path = path_list[0]
        dist.barrier()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    reusable_lora = BatchedTTTLoRA(ttt_bsz, base_model, h.ttt_lora_rank, k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora).to(device)
    reusable_opt = torch.optim.AdamW(reusable_lora.parameters(), lr=h.ttt_lora_lr, betas=(h.ttt_beta1, h.ttt_beta2), eps=1e-10, weight_decay=h.ttt_weight_decay, fused=True)
    while True:
        queue_idx = _claim_next_batch(counter_path, queue_len)
        if queue_idx >= queue_len:
            break
        orig_batch_idx, batch_entries = global_batches_sorted[queue_idx]
        batch = [doc for _, doc in batch_entries]
        bsz = len(batch)
        if bsz == reusable_lora.bsz:
            reusable_lora.reset()
            for s in reusable_opt.state.values():
                for k, v in s.items():
                    if isinstance(v, torch.Tensor):
                        v.zero_()
                    elif k == 'step':
                        s[k] = 0
            cur_lora = reusable_lora
            cur_opt = reusable_opt
        else:
            cur_lora = BatchedTTTLoRA(bsz, base_model, h.ttt_lora_rank, k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora).to(device)
            cur_opt = torch.optim.AdamW(cur_lora.parameters(), lr=h.ttt_lora_lr, betas=(h.ttt_beta1, h.ttt_beta2), eps=1e-10, weight_decay=h.ttt_weight_decay, fused=True)
        pred_lens = [doc_len - 1 for _, doc_len in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)
        num_chunks_t = torch.tensor(num_chunks, dtype=torch.int64, device=device)
        for ci in range(max_nc):
            active = [ci < nc for nc in num_chunks]
            needs_train = any((ci < nc - 1 for nc in num_chunks))
            tok_starts = torch.zeros(bsz, dtype=torch.int64)
            tok_wls = torch.zeros(bsz, dtype=torch.int64)
            chunk_offsets_cpu = torch.zeros(bsz, dtype=torch.int64)
            chunk_lens_cpu = torch.zeros(bsz, dtype=torch.int64)
            for b in range(bsz):
                if not active[b]:
                    continue
                doc_start, doc_len = batch[b]
                win_start, win_len, chunk_offset, chunk_len = _compute_chunk_window(ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len)
                tok_starts[b] = doc_start + win_start
                tok_wls[b] = win_len
                chunk_offsets_cpu[b] = chunk_offset
                chunk_lens_cpu[b] = chunk_len
            _, context_size, chunk_offset, _ = _compute_chunk_window(ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len)
            col_idx = torch.arange(context_size + 1)
            idx = tok_starts.unsqueeze(1) + col_idx.unsqueeze(0)
            idx.clamp_(max=all_tokens.numel() - 1)
            gathered_gpu = all_tokens_idx[idx].to(device=device, dtype=torch.int64, non_blocking=True)
            valid = (col_idx[:context_size].unsqueeze(0) < tok_wls.unsqueeze(1)).to(device, non_blocking=True)
            chunk_offsets = chunk_offsets_cpu.to(device, non_blocking=True)
            chunk_lens = chunk_lens_cpu.to(device, non_blocking=True)
            x = torch.where(valid, gathered_gpu[:, :context_size], 0)
            y = torch.where(valid, gathered_gpu[:, 1:context_size + 1], 0)
            ctx_pos = torch.arange(context_size, device=device, dtype=torch.int64)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
            with torch.no_grad():
                _accumulate_bpb(per_tok_loss, x, y, chunk_offsets, chunk_lens, ctx_pos, val_data.base_bytes_lut, val_data.has_leading_space_lut, val_data.is_boundary_token_lut, loss_sum, byte_sum, token_count)
            if needs_train:
                activate_chunk_mask = (num_chunks_t - 1 > ci).float()
                train_mask = ((ctx_pos.unsqueeze(0) >= chunk_offsets.unsqueeze(1)) & (ctx_pos.unsqueeze(0) < (chunk_offsets + chunk_lens).unsqueeze(1))).to(per_tok_loss.dtype)
                for gi in range(h.ttt_grad_steps):
                    if gi > 0:
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
                    cur_opt.zero_grad(set_to_none=True)
                    denom = train_mask.sum(dim=-1).clamp_min(1)
                    per_doc = (per_tok_loss * train_mask).sum(dim=-1) / denom
                    (per_doc * activate_chunk_mask).sum().backward()
                    torch.nn.utils.clip_grad_norm_(cur_lora.parameters(), h.ttt_grad_clip)
                    cur_opt.step()
            else:
                del per_tok_loss
        if cur_lora is not reusable_lora:
            del cur_lora, cur_opt
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.train()
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_sum.item())
    return (val_loss, val_bpb)

def eval_val_ttt_full(h, device, val_data, base_model, batch_seqs=32):
    """Score-first full-parameter TTT (port of train_gpt_sub_04_15.eval_val_ttt):
    score all sliding windows in a chunk, then SGD-adapt on that chunk's data
    before moving to the next chunk."""
    rank, world_size = (h.rank, h.world_size)
    seq_len = h.eval_seq_len
    stride = h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    ttt_chunk = h.ttt_chunk_tokens
    context_size = seq_len - stride
    window_starts = [ws for ws in range(0, total_tokens, stride) if ws + context_size < total_tokens]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        s = 0 if ws == 0 else context_size
        ci = min((ws + s) // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)
    log(f'ttt:start chunks={num_chunks} ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs}')
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    ttt_params = list(base_model.parameters())
    for p in ttt_params:
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(ttt_params, lr=h.ttt_lr, momentum=h.ttt_momentum)
    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        my_s = len(windows) * rank // world_size
        my_e = len(windows) * (rank + 1) // world_size
        my_windows = windows[my_s:my_e]
        base_model.eval()
        with torch.no_grad():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    we = min(ws + seq_len, total_tokens)
                    wlen = we - ws
                    wlens.append(wlen)
                    chunk_tok = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = compiled_logits(x_batch)
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y_batch.reshape(-1), reduction='none').reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else context_size
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()
        is_last_chunk = ci == num_chunks - 1
        if not is_last_chunk and h.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = h.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                my_seq_s = chunk_seqs * rank // world_size
                my_seq_e = chunk_seqs * (rank + 1) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(h.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, batch_seqs):
                        be = min(bs + batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_data.val_tokens.numel():
                            continue
                        local = val_data.val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                        optimizer.step()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.eval()
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return (val_loss, val_bpb)

def timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    log(f'{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms')
    return (val_loss, val_bpb)

def train_model(h, device, val_data):
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    if h.distributed:
        model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False, gradient_as_bucket_view=True, bucket_cap_mb=100)
    else:
        model = compiled_model
    log(f'model_params:{sum((p.numel() for p in base_model.parameters()))}')
    optimizers = Optimizers(h, base_model)
    train_loader = DocumentPackingLoader(h, device)
    log(f'prefetch:cpu_depth={h.cpu_prefetch_depth} gpu_depth={h.gpu_prefetch_depth}')
    max_wallclock_ms = 1000.0 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1000.0
        log(f'gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms')

    def training_frac(step, elapsed_ms):
        if max_wallclock_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-09)

    def lr_mul(frac):
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0
    copy_stream = torch.cuda.Stream()
    gpu_prefetch_depth = max(1, int(h.gpu_prefetch_depth))
    gpu_prefetch_queue = collections.deque()

    def _prefetch_gpu_batch():
        with torch.cuda.stream(copy_stream):
            batch = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            event = torch.cuda.Event()
            event.record(copy_stream)
        return (batch, event)

    def _fill_gpu_prefetch_queue():
        while len(gpu_prefetch_queue) < gpu_prefetch_depth:
            gpu_prefetch_queue.append(_prefetch_gpu_batch())

    def _next_gpu_batch():
        _fill_gpu_prefetch_queue()
        batch, event = gpu_prefetch_queue.popleft()
        torch.cuda.current_stream().wait_event(event)
        _fill_gpu_prefetch_queue()
        return batch
    _fill_gpu_prefetch_queue()

    def step_fn(step, lr_scale):
        optimizers.zero_grad_all()
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            x, y, cu, ms = _next_gpu_batch()
            loss = model(x, y, cu, ms)
            (loss / h.grad_accum_steps).backward()
        frac = min(step / h.muon_momentum_warmup_steps, 1.0) if h.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group['momentum'] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group['lr'] = group['base_lr'] * lr_scale
        if 0 < h.grad_clip_norm < 1000000.0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm, foreach=True)
        optimizers.step()
    if h.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0)
            if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                log(f'warmup_step: {warmup_step + 1}/{h.warmup_steps}')
        if h.num_loops > 0:
            base_model.looping_active = True
            log(f'loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}')
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0)
                if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                    log(f'loop_warmup_step: {warmup_step + 1}/{h.warmup_steps}')
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed:
            model.require_backward_grad_sync = True
        copy_stream.synchronize()
        gpu_prefetch_queue.clear()
        train_loader = DocumentPackingLoader(h, device)
        _fill_gpu_prefetch_queue()
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = h.ema_decay
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == h.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(f'stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}')
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        scale = lr_mul(frac)
        if h.num_loops > 0 and (not base_model.looping_active) and (frac >= h.enable_looping_at):
            base_model.looping_active = True
            log(f'layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}')
        step_fn(step, scale)
        with torch.no_grad():
            ema_vals = list(ema_state.values())
            cur_vals = [t.detach().float() for t in base_model.state_dict().values()]
            torch._foreach_mul_(ema_vals, ema_decay)
            torch._foreach_add_(ema_vals, cur_vals, alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log(f'peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB')
    log('ema:applying EMA weights')
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    return base_model

def _run_ttt_eval(h, device, val_data, ttt_model):
    if h.ttt_mode == 'full':
        log(f'ttt_full:lr={h.ttt_lr} epochs={h.ttt_epochs} momentum={h.ttt_momentum} chunk={h.ttt_chunk_tokens} stride={h.eval_stride}')
        timed_eval('quantized_ttt_full', eval_val_ttt_full, h, device, val_data, ttt_model)
        return
    if h.ttt_rope_base_seqlen > 0:
        os.environ['ROPE_FORCE_BASE_SEQLEN'] = str(h.ttt_rope_base_seqlen)
        log(f'ttt_lora:rope_base_override ROPE_FORCE_BASE_SEQLEN={h.ttt_rope_base_seqlen}')
    else:
        log(f'ttt_lora:rope_base inherited ROPE_FORCE_BASE_SEQLEN={os.environ.get('ROPE_FORCE_BASE_SEQLEN', '0')}')
    for p in ttt_model.parameters():
        p.requires_grad_(False)
    _compiled = [None]

    def _fwd_ttt_inner(input_ids, target_ids, lora):
        return ttt_model.forward_ttt(input_ids, target_ids, lora=lora)

    def fwd_ttt_compiled(input_ids, target_ids, lora):
        if _compiled[0] is None:
            _compiled[0] = torch.compile(_fwd_ttt_inner, dynamic=True)
        return _compiled[0](input_ids, target_ids, lora=lora)
    log('ttt_lora:warming up compile')
    val_tokens_idx = val_data.val_tokens.to(torch.int32)
    t_warmup = time.perf_counter()
    wl = BatchedTTTLoRA(h.ttt_batch_size, ttt_model, h.ttt_lora_rank, k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora).to(device)
    wo = torch.optim.AdamW(wl.parameters(), lr=h.ttt_lora_lr, betas=(h.ttt_beta1, h.ttt_beta2), eps=1e-10, weight_decay=h.ttt_weight_decay, fused=True)
    for ctx_len in (h.ttt_chunk_size, h.ttt_eval_seq_len):
        col_w = torch.arange(ctx_len + 1)
        idx_w = col_w.clamp_(max=val_data.val_tokens.numel() - 1)
        row_w = val_tokens_idx[idx_w].to(device=device, dtype=torch.int64)
        xw = row_w[:ctx_len].unsqueeze(0).expand(h.ttt_batch_size, -1).contiguous()
        yw = row_w[1:ctx_len + 1].unsqueeze(0).expand(h.ttt_batch_size, -1).contiguous()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            ptl = fwd_ttt_compiled(xw, yw, lora=wl)
        ptl[:, :min(h.ttt_chunk_size, ctx_len)].mean(dim=-1).sum().backward()
        wo.step()
        wo.zero_grad(set_to_none=True)
    del wl, wo, val_tokens_idx
    torch.cuda.empty_cache()
    log(f'ttt_lora:compile warmup done ({time.perf_counter() - t_warmup:.1f}s)')
    timed_eval('quantized_ttt_lora', eval_val_ttt, h, device, val_data, ttt_model, forward_ttt_train=fwd_ttt_compiled)

def train_and_eval(h, device):
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    val_data = ValidationData(h, device)
    log(f'train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}')
    log(f'val_tokens: {val_data.val_tokens.numel() - 1}')
    base_model = train_model(h, device, val_data)
    torch._dynamo.reset()
    code_path = Path(os.environ.get('_ORIG_SCRIPT', __file__))
    serialize(h, base_model, code_path.read_text(encoding='utf-8'))
    if h.distributed:
        dist.barrier()
    if h.ttt_enabled:
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        _run_ttt_eval(h, device, val_data, ttt_model)
        del ttt_model

def main():
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if world_size <= 0:
        raise ValueError(f'WORLD_SIZE must be positive, got {world_size}')
    if 8 % world_size != 0:
        raise ValueError(f'WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend='nccl', device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    h = Hyperparameters()
    if h.rope_force_base_seqlen <= 0:
        h.rope_force_base_seqlen = max(h.train_seq_len + 1, h.train_batch_tokens // max(1, h.world_size * h.grad_accum_steps))
    os.environ['ROPE_FORCE_BASE_SEQLEN'] = str(h.rope_force_base_seqlen)
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs('logs', exist_ok=True)
        log(100 * '=', console=False)
        log('Hyperparameters:', console=True)
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith('_'):
                log(f'  {k}: {v}', console=True)
        log('=' * 100, console=False)
        log(f'Running Python {sys.version}', console=False)
        log(f'Running PyTorch {torch.__version__}', console=False)
        log(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
        log('=' * 100, console=False)
    train_and_eval(h, device)
    if distributed:
        dist.destroy_process_group()
if __name__ == '__main__':
    main()