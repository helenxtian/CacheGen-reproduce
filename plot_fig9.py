"""
Plot Figure 9 of CacheGen (SIGCOMM'24): KV cache size vs. quality.

Reads the JSON files written by run_cachegen.py / run_quantization_baseline.py /
run_vanilla.py (results_dir/<dataset>/<method>_<setting>.json) and draws one panel
per dataset that has results.

Usage: python plot_fig9.py --results_dir mistral7b_results [--out fig9.png]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--results_dir", type=str, default="mistral7b_results")
p.add_argument("--out", type=str, default="fig9.png")
p.add_argument("--model_name", type=str, default="Mistral-7B")
args = p.parse_args()

DATASET_TITLES = {"longchat": "LongChat", "nqa": "NarrativeQA", "tqa": "TriviaQA"}
METRIC_LABELS = {"accuracy": "Accuracy", "f1": "F1 score"}

# Fixed categorical order (first three slots validate under all-pairs CVD checks).
# Each series also gets its own marker + line style so identity is not color-alone.
STYLES = {
    "cachegen": dict(label="CacheGen", color="#2a78d6", marker="o", linestyle="-"),
    "quant":    dict(label="Uniform quantization", color="#eb6834", marker="s", linestyle="--"),
    "vanilla":  dict(label="No compression (fp16)", color="#1baf7a", marker="^", linestyle="none"),
}
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(dataset_dir):
    runs = {}
    for path in glob.glob(os.path.join(dataset_dir, "*.json")):
        with open(path) as f:
            r = json.load(f)
        if r.get("avg_size_mb") is None or r.get("avg_metric") is None:
            continue
        runs.setdefault(r["method"], []).append(r)
    for m in runs:
        runs[m].sort(key=lambda r: r["avg_size_mb"])
    return runs


datasets = [d for d in ["longchat", "nqa", "tqa"]
            if os.path.isdir(os.path.join(args.results_dir, d))]
if not datasets:
    raise SystemExit(f"No results found under {args.results_dir}/")

fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 3.8), squeeze=False)
for ax, ds in zip(axes[0], datasets):
    runs = load(os.path.join(args.results_dir, ds))
    metric_name = "accuracy"
    for method in ["cachegen", "quant", "vanilla"]:
        if method not in runs:
            continue
        st = STYLES[method]
        xs = [r["avg_size_mb"] for r in runs[method]]
        ys = [r["avg_metric"] for r in runs[method]]
        metric_name = runs[method][0]["metric_name"]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                linewidth=2, markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                label=st["label"], zorder=3)
        # Selective direct labels: name each point's setting (level / bit-width)
        for r in runs[method]:
            ax.annotate(r["setting"], (r["avg_size_mb"], r["avg_metric"]),
                        textcoords="offset points", xytext=(6, -12),
                        fontsize=8, color=TEXT_SECONDARY)
    ax.set_title(f"{args.model_name}, {DATASET_TITLES.get(ds, ds)}", color=TEXT_PRIMARY, fontsize=11)
    ax.set_xlabel("KV cache size (MB)", color=TEXT_SECONDARY)
    ax.set_ylabel(METRIC_LABELS.get(metric_name, metric_name), color=TEXT_SECONDARY)
    ax.set_xlim(left=0)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
           bbox_to_anchor=(0.5, 1.02), labelcolor=TEXT_PRIMARY)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {args.out}")

# Table view of the same data (for the report / accessibility)
print(f"\n{'dataset':<10} {'method':<10} {'setting':<8} {'size (MB)':>10} {'quality':>8}  n")
for ds in datasets:
    for method, rs in load(os.path.join(args.results_dir, ds)).items():
        for r in rs:
            print(f"{ds:<10} {method:<10} {r['setting']:<8} {r['avg_size_mb']:>10.1f} "
                  f"{r['avg_metric']:>8.3f}  {r['num_docs']}")
