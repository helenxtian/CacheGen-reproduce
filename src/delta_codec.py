"""
Token-wise delta ("change-based") encoding for CacheGen, reimplemented from the
paper (SIGCOMM'24, Sec. 5.2). The released codec only does layer-wise
quantization + arithmetic coding; this adds the missing delta step.

Paper design (Sec. 5.2):
  * Split the context into groups of 10 contiguous tokens.
  * The first token of each group is the anchor, kept at 8-bit precision.
  * Every other token is encoded as its delta from the group's anchor (not
    from the previous token), so groups decode independently.
  * Deltas get layer-wise quantization, then arithmetic coding with one
    probability distribution per (layer, channel).

Implementation choices (the paper does not pin these down):
  * Anchors, two modes:
      - "level" (default): quantized exactly like the original CacheGen codec
        at the current QUANT_LEVEL and arithmetic-coded in their own stream.
      - "int8": the paper's literal 8-bit anchors, stored as raw int8 (the
        CUDA coder only supports alphabets of <= 47 symbols, so it cannot
        entropy-code 8-bit symbols). On test data the raw anchors cost more
        than the whole original codec, so "level" is the default.
    Because deltas are taken against the decoded anchor (next point), a
    coarser anchor does not add error to the other tokens in its group.
  * Delta step size = the step the original CacheGen level uses for that token
    (token absmax / (bins // 2 - 1), with bins from the QUANT_LEVEL config). So
    rounding error matches the original codec at the same level; the deltas
    are just smaller numbers, which compress better.
  * Deltas are taken against the *dequantized* anchor (closed loop), so the
    anchor's own rounding error does not add to the delta's error.
  * Quantized deltas use a 47-symbol alphabet (+-23 steps), the most the CUDA
    encoder supports (MAX_LP = 48). Rare larger deltas are clamped; the
    fraction clamped is reported so you can check it is negligible.

The quantization math (`delta_quantize` / `delta_dequantize`) is plain torch
and runs on CPU; only `DeltaCacheGenCodec.to_bytes/from_bytes` need the CUDA
coder (torchac_cuda) built on the GPU node.
"""
import io
import math
import pickle

import torch

GROUP_SIZE = 10
ANCHOR_MAX = 127          # 8-bit symmetric quantization for anchors
DELTA_BINS = 47           # alphabet size for quantized deltas (encoder MAX_LP = 48)
DELTA_HALF = (DELTA_BINS - 1) // 2   # symbols are offset by this; 23
EPS = 1e-6


# ----------------------------------------------------------------------------
# Pure-torch quantization (CPU or GPU)
# ----------------------------------------------------------------------------

def group_indices(ntokens, group_size=GROUP_SIZE, device="cpu"):
    """ Anchor positions, non-anchor positions, and each non-anchor token's
    anchor index (into the anchor list). """
    idx = torch.arange(ntokens, device=device)
    is_anchor = idx % group_size == 0
    anchor_idx = idx[is_anchor]
    other_idx = idx[~is_anchor]
    other_group = other_idx // group_size
    return anchor_idx, other_idx, other_group


def quantize_anchor(x):
    """ 8-bit vectorwise quantization. x: [L, A, C] -> int8 [L, A, C], scale [L, A, 1] """
    scale = x.abs().amax(-1, keepdim=True).clamp_min(EPS)
    q = torch.round(x / scale * ANCHOR_MAX).clamp(-ANCHOR_MAX, ANCHOR_MAX).to(torch.int8)
    return q, scale.to(torch.float16)


def dequantize_anchor(q, scale):
    return q.float() / ANCHOR_MAX * scale.float()


def _anchor_codec(xa, bins, anchor_mode):
    """ Quantize anchors; returns (symbols, scale, dequantized anchors). """
    if anchor_mode == "int8":
        q, scale = quantize_anchor(xa)
    elif anchor_mode == "level":
        q, scale = original_quantize(xa, bins)
    else:
        raise ValueError(f"unknown anchor_mode {anchor_mode}")
    return q, scale, _anchor_dequant(q, scale, bins, anchor_mode)


def _anchor_dequant(q, scale, bins, anchor_mode):
    if anchor_mode == "int8":
        return dequantize_anchor(q, scale)
    return original_dequantize(q, scale, bins)


def delta_quantize(x, bins, group_size=GROUP_SIZE, anchor_mode="level"):
    """
    x:    [L, T, C] float tensor (keys or values, heads merged into channels)
    bins: [L] tensor, the original CacheGen bin count per layer (sets the step)

    Returns a dict with:
      anchor_q [L, A, C] int8 (symbols in [0, bins-2] for "level", +-127 for "int8"),
      anchor_scale [L, A, 1] fp16,
      delta_sym [L, N, C] int8 in [0, DELTA_BINS - 1], token_scale [L, N, 1] fp16,
      clamped_frac (python float)
    where A = number of anchors and N = T - A.
    """
    x = x.float()
    L, T, C = x.shape
    anchor_idx, other_idx, other_group = group_indices(T, group_size, x.device)

    anchor_q, anchor_scale, anchor_hat = _anchor_codec(x[:, anchor_idx], bins, anchor_mode)
    ref = anchor_hat[:, other_group]                                 # [L, N, C]
    xo = x[:, other_idx]                                             # [L, N, C]

    # Same step size as the original CacheGen codec at this level
    token_scale = xo.abs().amax(-1, keepdim=True).clamp_min(EPS).to(torch.float16)
    levels = (bins.float().to(x.device) // 2 - 1)[:, None, None]    # [L, 1, 1]
    step = token_scale.float() / levels

    d = torch.round((xo - ref) / step)
    clamped_frac = (d.abs() > DELTA_HALF).float().mean().item()
    delta_sym = (d.clamp(-DELTA_HALF, DELTA_HALF) + DELTA_HALF).to(torch.int8)

    return {
        "anchor_q": anchor_q,
        "anchor_scale": anchor_scale,
        "delta_sym": delta_sym,
        "token_scale": token_scale,
        "clamped_frac": clamped_frac,
    }


def delta_dequantize(anchor_q, anchor_scale, delta_sym, token_scale, bins, ntokens,
                     group_size=GROUP_SIZE, anchor_mode="level"):
    """ Inverse of delta_quantize. delta_sym may be int8 or float. Returns [L, T, C] float32. """
    L, _, C = anchor_q.shape
    device = anchor_q.device
    anchor_idx, other_idx, other_group = group_indices(ntokens, group_size, device)

    anchor_hat = _anchor_dequant(anchor_q, anchor_scale, bins, anchor_mode)
    levels = (bins.float().to(device) // 2 - 1)[:, None, None]
    step = token_scale.float() / levels
    delta_hat = (delta_sym.float() - DELTA_HALF) * step

    out = torch.empty((L, ntokens, C), dtype=torch.float32, device=device)
    out[:, anchor_idx] = anchor_hat
    out[:, other_idx] = anchor_hat[:, other_group] + delta_hat
    return out


def original_quantize(x, bins):
    """ The released codec's quantizer (torch_quant_vectorized):
    symbols = round(x / step) + levels, step = token absmax / levels.
    The scale is rounded to fp16 *before* use so the encoder and decoder
    reconstruct exactly the same values. """
    x = x.float()
    levels = (bins.float().to(x.device) // 2 - 1)[:, None, None]
    scale = x.abs().amax(-1, keepdim=True).clamp_min(EPS).to(torch.float16)
    sym = torch.round(x * levels / scale.float() + levels)
    sym = torch.minimum(sym.clamp_min(0), 2 * levels)
    return sym.to(torch.int8), scale


def original_dequantize(sym, scale, bins):
    levels = (bins.float().to(sym.device) // 2 - 1)[:, None, None]
    return (sym.float() - levels) / levels * scale.float()


def entropy_bytes(sym, nsymbols):
    """ Ideal arithmetic-coded size (bytes) of sym [L, N, C] with one empirical
    distribution per (layer, channel), i.e. what the CUDA coder approximates.
    Excludes the CDF tables and chunk-length metadata. """
    L, N, C = sym.shape
    total_bits = 0.0
    offsets = (torch.arange(C, device=sym.device) * nsymbols)[None, :]
    for l in range(L):
        flat = (sym[l].long() + offsets).flatten()
        counts = torch.bincount(flat, minlength=nsymbols * C).view(C, nsymbols).float()
        p = counts / N
        total_bits += -(counts * torch.log2(p.clamp_min(1e-12))).sum().item()
    return total_bits / 8


# ----------------------------------------------------------------------------
# Full codec on the GPU (uses the released CUDA arithmetic coder)
# ----------------------------------------------------------------------------

def _hf_blob_to_kv(blob):
    """ [L, 2, H, T, D] (huggingface) -> K, V each [L, T, H*D], plus (H, D) """
    L, _, H, T, D = blob.shape
    kv = blob.permute(0, 1, 3, 2, 4).reshape(L, 2, T, H * D)
    return kv[:, 0], kv[:, 1], H, D


def _kv_to_hf_blob(k, v, H, D):
    L, T, _ = k.shape
    blob = torch.stack([k, v], dim=1).reshape(L, 2, T, H, D)
    return blob.permute(0, 1, 3, 2, 4).to(torch.float16)


def _encode_stream(sym, nsymbols):
    """ Arithmetic-code int8 symbols [nlayers, ntokens, nchannels] in [0, nsymbols - 1]
    with one CDF per (layer, channel), using the released CUDA kernels.
    Returns (cdf, data_chunks) exactly like the original encoder. """
    import torchac_cuda
    import lmcache.storage_backend.serde.cachegen_basics as CGBasics
    from lmcache.storage_backend.serde.cachegen_encoder import encode_ntokens

    sym = sym.contiguous()
    nlayers, ntokens, nchannels = sym.shape
    cdf = torchac_cuda.calculate_cdf(sym, nsymbols)            # [nlayers, nchannels, nsymbols + 1]
    chunk = CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK
    output_buffer = torch.zeros((nlayers, nchannels, chunk), dtype=torch.uint8, device=sym.device)
    output_lengths = torch.zeros((nlayers, nchannels), dtype=torch.int32, device=sym.device)
    data_chunks = []
    for start in range(0, ntokens, chunk):
        end = min(start + chunk, ntokens)
        bytestream = encode_ntokens(cdf, sym[:, start:end, :], output_buffer, output_lengths)
        data_chunks.append(CGBasics.CacheGenGPUBytestream(
            bytestream=bytestream.cpu(),
            bytestream_lengths=output_lengths.clone().cpu(),
            ntokens=end - start,
        ))
    return cdf.cpu(), data_chunks


def _decode_stream(cdf, data_chunks):
    """ Inverse of _encode_stream. Returns two float tensors (first half and
    second half of the layers, i.e. K and V), each [nlayers / 2, ntokens, nchannels]. """
    from lmcache.storage_backend.serde.cachegen_decoder import decode_function_gpu
    cdf = cdf.cuda()
    nlayers, nchannels, _ = cdf.shape
    ntokens = sum(c.ntokens for c in data_chunks)
    chunks = [type(c)(bytestream=c.bytestream.cuda(),
                      bytestream_lengths=c.bytestream_lengths.cuda(),
                      ntokens=c.ntokens) for c in data_chunks]
    output = torch.zeros((ntokens, nlayers * nchannels), dtype=torch.uint8, device="cuda")
    return decode_function_gpu(cdf, chunks, nlayers // 2, ntokens, output)


def _nbytes(t):
    return t.numel() * t.element_size()


class DeltaCacheGenCodec:
    """ Drop-in alternative to CacheGenSerializer/CacheGenDeserializer:
    to_bytes(blob) -> bytes and from_bytes(bytes) -> blob, with blob in the
    huggingface layout [nlayers, 2, nheads, ntokens, head_dim] used by main.py. """

    def __init__(self, model_name, group_size=GROUP_SIZE, anchor_mode="level"):
        # Reads QUANT_LEVEL from the environment, like the original codec
        from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig
        cfg = CacheGenConfig.from_model_name(model_name)
        n = cfg.key_third_layers
        self.key_bins = torch.full((n,), float(cfg.key_third_bins))
        self.key_bins[:cfg.key_second_layers] = cfg.key_second_bins
        self.key_bins[:cfg.key_first_layers] = cfg.key_first_bins
        self.value_bins = torch.full((n,), float(cfg.value_second_bins))
        self.value_bins[:cfg.value_first_layers] = cfg.value_first_bins
        self.group_size = group_size
        self.anchor_mode = anchor_mode
        self.last_stats = {}

    def to_bytes(self, blob):
        k, v, H, D = _hf_blob_to_kv(blob.cuda())
        L, T, C = k.shape
        assert T > self.group_size, f"need more than {self.group_size} tokens, got {T}"
        qk = delta_quantize(k, self.key_bins, self.group_size, self.anchor_mode)
        qv = delta_quantize(v, self.value_bins, self.group_size, self.anchor_mode)

        delta_cdf, delta_chunks = _encode_stream(
            torch.cat([qk["delta_sym"], qv["delta_sym"]], dim=0), DELTA_BINS)
        payload = {
            "version": 2, "group_size": self.group_size, "anchor_mode": self.anchor_mode,
            "ntokens": T, "num_heads": H, "head_size": D,
            "delta_cdf": delta_cdf, "delta_chunks": delta_chunks,
            "anchor_scale_k": qk["anchor_scale"].cpu(), "anchor_scale_v": qv["anchor_scale"].cpu(),
            "token_scale_k": qk["token_scale"].cpu(), "token_scale_v": qv["token_scale"].cpu(),
        }
        if self.anchor_mode == "level":
            # Same alphabet size the original encoder uses: the largest bin count
            nsym = int(max(self.key_bins.max().item(), self.value_bins.max().item()))
            anchor_cdf, anchor_chunks = _encode_stream(
                torch.cat([qk["anchor_q"], qv["anchor_q"]], dim=0), nsym)
            payload.update(anchor_cdf=anchor_cdf, anchor_chunks=anchor_chunks)
            anchors_mb = (sum(_nbytes(c.bytestream) for c in anchor_chunks) / 1e6)
            meta = [anchor_cdf] + [c.bytestream_lengths for c in anchor_chunks]
        else:
            payload.update(anchor_q_k=qk["anchor_q"].cpu(), anchor_q_v=qv["anchor_q"].cpu())
            anchors_mb = (_nbytes(qk["anchor_q"]) + _nbytes(qv["anchor_q"])) / 1e6
            meta = []
        meta += [delta_cdf] + [c.bytestream_lengths for c in delta_chunks]

        # Size breakdown for the report (the reported total is len(bytes))
        self.last_stats = {
            "delta_stream_mb": sum(_nbytes(c.bytestream) for c in delta_chunks) / 1e6,
            "anchors_mb": anchors_mb,
            "coder_metadata_mb": sum(_nbytes(t) for t in meta) / 1e6,
            "scales_mb": sum(_nbytes(q[s]) for q in (qk, qv)
                             for s in ("anchor_scale", "token_scale")) / 1e6,
            "clamped_frac": (qk["clamped_frac"] + qv["clamped_frac"]) / 2,
        }
        with io.BytesIO() as f:
            pickle.dump(payload, f)
            return f.getvalue()

    def from_bytes(self, bs):
        p = pickle.loads(bs)
        T, H, D, G, mode = (p["ntokens"], p["num_heads"], p["head_size"],
                            p["group_size"], p["anchor_mode"])
        sym_k, sym_v = _decode_stream(p["delta_cdf"], p["delta_chunks"])    # [L, N, C] float
        if mode == "level":
            aq_k, aq_v = _decode_stream(p["anchor_cdf"], p["anchor_chunks"])
        else:
            aq_k, aq_v = p["anchor_q_k"].cuda(), p["anchor_q_v"].cuda()

        k = delta_dequantize(aq_k, p["anchor_scale_k"].cuda(), sym_k, p["token_scale_k"].cuda(),
                             self.key_bins, T, G, mode)
        v = delta_dequantize(aq_v, p["anchor_scale_v"].cuda(), sym_v, p["token_scale_v"].cuda(),
                             self.value_bins, T, G, mode)
        return _kv_to_hf_blob(k, v, H, D)


# ----------------------------------------------------------------------------
# CPU size estimate on real KV caches (no CUDA coder needed)
# ----------------------------------------------------------------------------

def estimate(raw_kv_path, key_bins, value_bins, group_size=GROUP_SIZE):
    """ Compare the ideal entropy-coded size and error of the original codec vs.
    the delta codec (both anchor modes) on one raw_kv_{doc}.pt saved by main.py.
    Sizes exclude CDF tables / chunk lengths (similar for all), so this isolates
    the effect of the delta step. Runs on CPU; takes a minute or two per doc. """
    blob = torch.load(raw_kv_path, map_location="cpu")
    k, v, _, _ = _hf_blob_to_kv(blob)
    T = k.shape[1]
    anchor_idx, _, _ = group_indices(T, group_size)
    res = {"ntokens": T}
    acc = {m: {"mb": 0.0, "err": 0.0, "clamped": 0.0} for m in ("original", "level", "int8")}
    for x, bins in [(k, key_bins), (v, value_bins)]:
        x = x.float()
        nsym = int(bins.max().item())
        s, scale = original_quantize(x, bins)
        acc["original"]["mb"] += (entropy_bytes(s, nsym) + _nbytes(scale)) / 1e6
        acc["original"]["err"] += (original_dequantize(s, scale, bins) - x).abs().mean().item() / 2
        for mode in ("level", "int8"):
            q = delta_quantize(x, bins, group_size, mode)
            anchors = (entropy_bytes(q["anchor_q"], nsym) if mode == "level"
                       else _nbytes(q["anchor_q"]))
            acc[mode]["mb"] += (entropy_bytes(q["delta_sym"], DELTA_BINS) + anchors
                                + _nbytes(q["anchor_scale"]) + _nbytes(q["token_scale"])) / 1e6
            xh = delta_dequantize(q["anchor_q"], q["anchor_scale"], q["delta_sym"],
                                  q["token_scale"], bins, T, group_size, mode)
            acc[mode]["err"] += (xh - x).abs().mean().item() / 2
            acc[mode]["clamped"] += q["clamped_frac"] / 2
    for m, a in acc.items():
        name = "original" if m == "original" else f"delta_{m}_anchors"
        res[f"{name}_mb"] = a["mb"]
        res[f"{name}_mean_abs_err"] = a["err"]
        if m != "original":
            res[f"{name}_clamped_frac"] = a["clamped"]
    return res


# Mistral-7B bin counts per layer, copied from cachegen_basics.py (32 layers)
MISTRAL_BINS = {
    "1": ([16] * 20 + [12] * 12, [16] * 2 + [12] * 30),
    "2": ([32] * 10 + [16] * 22, [32] * 2 + [16] * 30),
    "3": ([32] * 32, [32] * 32),
}


if __name__ == "__main__":
    # python -m src.delta_codec <raw_kv_{doc}.pt> [quant_level]
    import sys
    level = sys.argv[2] if len(sys.argv) > 2 else "2"
    kb, vb = (torch.tensor(b, dtype=torch.float32) for b in MISTRAL_BINS[level])
    for k_, v_ in estimate(sys.argv[1], kb, vb).items():
        print(f"{k_:>36}: {v_:.4f}")
