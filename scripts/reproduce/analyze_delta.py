"""
Why didn't delta encoding help? (our analysis; runs on CPU)

Our reimplementation of the paper's token-wise delta encoding (src/delta_codec.py)
came out 9-17% LARGER than the released codec on Mistral-7B. Hypothesis: Hugging
Face caches keys AFTER rotary position embedding (RoPE). RoPE rotates each
token's key by a position-dependent angle, so neighbouring keys differ even when
the underlying content is similar.

For each raw KV cache saved by main.py, this script measures three streams:
  values,
  keys as cached (post-RoPE),
  keys with RoPE undone (pre-RoPE).
For each one it reports the ideal entropy-coded size under the original codec
(layer-wise quantization) and under the delta codec, plus a correlation
diagnostic:
  var ratio = Var(x_t - x_anchor) / Var(x_t)
This is about 0 for smooth streams and about 2 for independent tokens. Delta
encoding can only help when it is clearly below 1.

Usage (repository root, after main.py has saved raw_kv_*.pt):
  python scripts/reproduce/analyze_delta.py mistral7b_longchat_data/raw_kv_0.pt [more .pt files] \
      --level 2 --out reproduce-results/delta_analysis.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.delta_codec import (DELTA_BINS, GROUP_SIZE, MISTRAL_BINS, _hf_blob_to_kv, _nbytes,  # noqa: E402
                             delta_quantize, entropy_bytes, group_indices, original_quantize)


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def rope_cos_sin(T, D, theta):
    """Same as transformers' Mistral rotary embedding (positions 0..T-1)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    freqs = torch.outer(torch.arange(T, dtype=torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]


def apply_rope(k, H, D, theta):
    L, T, _ = k.shape
    x = k.view(L, T, H, D).float()
    cos, sin = rope_cos_sin(T, D, theta)
    return (x * cos + rotate_half(x) * sin).view(L, T, H * D)


def undo_rope(k, H, D, theta):
    """Inverse rotation, R(-theta): x = k*cos - rotate_half(k)*sin."""
    L, T, _ = k.shape
    x = k.view(L, T, H, D).float()
    cos, sin = rope_cos_sin(T, D, theta)
    return (x * cos - rotate_half(x) * sin).view(L, T, H * D)


def measure(x, bins, group_size):
    """Ideal sizes (MB) of one stream x [L, T, C] under the original and the delta codec."""
    x = x.float()
    nsym = int(bins.max().item())
    s, scale = original_quantize(x, bins)
    orig = (entropy_bytes(s, nsym) + _nbytes(scale)) / 1e6
    q = delta_quantize(x, bins, group_size, "level")
    delta = (entropy_bytes(q["delta_sym"], DELTA_BINS) + entropy_bytes(q["anchor_q"], nsym)
             + _nbytes(q["anchor_scale"]) + _nbytes(q["token_scale"])) / 1e6
    anchor_idx, other_idx, other_group = group_indices(x.shape[1], group_size)
    d = x[:, other_idx] - x[:, anchor_idx][:, other_group]
    var_ratio = (d.var(dim=1) / x[:, other_idx].var(dim=1).clamp_min(1e-12)).mean().item()
    return dict(original_mb=orig, delta_mb=delta, delta_vs_original=delta / orig, var_ratio=var_ratio,
                clamped_frac=q["clamped_frac"])


def analyze(path, level, theta, group_size):
    blob = torch.load(path, map_location="cpu")
    k, v, H, D = _hf_blob_to_kv(blob)
    kb, vb = (torch.tensor(b, dtype=torch.float32) for b in MISTRAL_BINS[level])
    return {"file": os.path.basename(path), "ntokens": k.shape[1],
            "values": measure(v, vb, group_size),
            "keys_post_rope": measure(k, kb, group_size),
            "keys_pre_rope": measure(undo_rope(k, H, D, theta), kb, group_size)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw_kv", nargs="+", help="raw_kv_<doc>.pt files saved by main.py")
    p.add_argument("--level", default="2", choices=sorted(MISTRAL_BINS))
    p.add_argument("--rope_theta", type=float, default=None,
                   help="default: read from the model config (Mistral-7B-v0.2: 1e6)")
    p.add_argument("--model_id", default="mistral-community/Mistral-7B-v0.2")
    p.add_argument("--group_size", type=int, default=GROUP_SIZE)
    p.add_argument("--out", default=None, help="write all results to this JSON file")
    args = p.parse_args()
    theta = args.rope_theta
    if theta is None:
        from transformers import AutoConfig
        theta = float(AutoConfig.from_pretrained(args.model_id).rope_theta)
    print(f"rope_theta = {theta:g}, level {args.level}, group size {args.group_size}")

    results = []
    print(f"\n{'file':<20} {'stream':<16} {'original MB':>11} {'delta MB':>9} {'delta/orig':>10} {'var ratio':>9}")
    for path in args.raw_kv:
        r = analyze(path, args.level, theta, args.group_size)
        results.append(r)
        for stream in ("values", "keys_post_rope", "keys_pre_rope"):
            m = r[stream]
            print(f"{r['file']:<20} {stream:<16} {m['original_mb']:>11.1f} {m['delta_mb']:>9.1f} "
                  f"{m['delta_vs_original']:>10.2f} {m['var_ratio']:>9.2f}")
    if args.out:
        json.dump(dict(level=args.level, rope_theta=theta, group_size=args.group_size, results=results),
                  open(args.out, "w"), indent=2)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
