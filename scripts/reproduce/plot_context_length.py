"""
Context-length experiment (our extension): the same 30 NarrativeQA questions with
10K, 20K and 30K-token contexts (test_data/nqa_{10,20,30}k.jsonl, built by
build_long_nqa.py).

Left panel: F1 against context length. Shows whether compression hurts more on
longer contexts.
Right panel: KV cache size per 1K tokens. Shows whether each method's compression
holds as the context grows.

Usage: python scripts/reproduce/plot_context_length.py --results_dir mistral7b_results \
           [--out context_length.png] [--prefix nqa]
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--results_dir", default="mistral7b_results")
p.add_argument("--out", default="context_length.png")
p.add_argument("--prefix", default="nqa", help="dataset prefix; folders <prefix>_<N>k are used")
p.add_argument("--model_name", default="Mistral-7B")
args = p.parse_args()

# (method, setting) -> style. Fixed categorical order; marker + line style so
# identity is never carried by color alone.
SERIES = [
    (("cachegen", "level2"), dict(label="CacheGen L2", color="#2a78d6", marker="o", linestyle="-")),
    (("quant", "8bit"), dict(label="8-bit quantization", color="#eb6834", marker="s", linestyle="--")),
    (("vanilla", "fp16"), dict(label="No compression (fp16)", color="#1baf7a", marker="^", linestyle="-")),
    (("cachegen_delta", "level2"), dict(label="CacheGen + delta L2 (ours)", color="#eda100", marker="D", linestyle=":")),
    (("cachegen", "level1"), dict(label="CacheGen L1", color="#e87ba4", marker="v", linestyle="-.")),
]
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"

dirs = {}
for d in glob.glob(os.path.join(args.results_dir, f"{args.prefix}_*k")):
    m = re.fullmatch(rf"{re.escape(args.prefix)}_(\d+)k", os.path.basename(d))
    if m:
        dirs[int(m.group(1))] = d
if not dirs:
    raise SystemExit(f"No {args.prefix}_<N>k folders under {args.results_dir}/")
lengths = sorted(dirs)

table = {}   # (method, setting) -> {k: result}
for k in lengths:
    for path in glob.glob(os.path.join(dirs[k], "*.json")):
        r = json.load(open(path))
        if r.get("avg_size_mb") is None or r.get("avg_metric") is None:
            continue
        table.setdefault((r["method"], r["setting"]), {})[k] = r

fig, (ax_q, ax_s) = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
for key, st in SERIES:
    if key not in table:
        continue
    ks = sorted(table[key])
    kw = dict(color=st["color"], marker=st["marker"], linestyle=st["linestyle"], linewidth=2, markersize=8,
              markeredgecolor="white", markeredgewidth=1.5, label=st["label"], zorder=3)
    ax_q.plot(ks, [table[key][k]["avg_metric"] for k in ks], **kw)
    # size per 1K tokens; prompts are exactly k thousand tokens (main.py drops the last token's KV)
    ax_s.plot(ks, [table[key][k]["avg_size_mb"] / k for k in ks], **kw)

for ax, ylabel in [(ax_q, "F1 score"), (ax_s, "KV cache size per 1K tokens (MB)")]:
    ax.set_xticks(lengths, [f"{k}K" for k in lengths])
    ax.set_xlabel("Context length (tokens)", color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)
ax_s.set_ylim(bottom=0)
n_docs = {r["num_docs"] for rs in table.values() for r in rs.values()}
fig.suptitle(f"{args.model_name}, NarrativeQA, same {'/'.join(map(str, sorted(n_docs)))} questions at each length",
             color=TEXT_PRIMARY, fontsize=11)
handles, labels = ax_q.get_legend_handles_labels()
fig.legend(handles, labels, loc="outside right center", frameon=False, labelcolor=TEXT_PRIMARY)
fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {args.out}")

# Table view (every method/setting, not just the plotted ones)
print(f"\n{'method':<16} {'setting':<8} " + " ".join(f"{f'{k}K F1':>7} {f'{k}K MB':>8}" for k in lengths))
for (m, s), rs in sorted(table.items()):
    cells = " ".join(f"{rs[k]['avg_metric']:>7.3f} {rs[k]['avg_size_mb']:>8.1f}" if k in rs else f"{'-':>7} {'-':>8}"
                     for k in lengths)
    print(f"{m:<16} {s:<8} {cells}")
