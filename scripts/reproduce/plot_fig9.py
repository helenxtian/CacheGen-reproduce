"""
Plot Figure 9 of CacheGen (SIGCOMM'24): KV cache size vs. quality.

Reads the JSON files written by run_cachegen.py / run_quantization_baseline.py /
run_vanilla.py (results_dir/<dataset>/<method>_<setting>.json) and draws one panel
per dataset that has results.

Usage: python scripts/reproduce/plot_fig9.py --results_dir mistral7b_results [--out fig9.png]
         [--xbreak 310 520 | --xbreak auto] [--logx] [--xlim MIN MAX] [--xticks ...]
"""
import argparse
import glob
import json
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--results_dir", type=str, default="mistral7b_results")
p.add_argument("--out", type=str, default="fig9.png")
p.add_argument("--model_name", type=str, default="Mistral-7B")
p.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
               help="x-axis range in MB, e.g. --xlim 100 320. Points outside it are "
                    "listed in a note under the axis instead of being drawn.")
p.add_argument("--logx", action="store_true",
               help="log-scale x-axis: spreads out the 130-300 MB region while keeping "
                    "8-bit and fp16 on the same plot")
p.add_argument("--xticks", type=float, nargs="+", default=None,
               help="explicit x tick positions in MB, e.g. --xticks 150 200 250 300 600 1200")
p.add_argument("--xbreak", nargs="+", default=None, metavar="A B | auto",
               help="broken x-axis: a wide panel up to A MB and a narrow panel from B MB, "
                    "e.g. --xbreak 310 520 (outer bounds from --xlim, else from the data). "
                    "'--xbreak auto' picks the break per panel: after the largest compressed "
                    "point, before 8-bit quantization and fp16")
p.add_argument("--datasets", nargs="+", default=None,
               help="which dataset folders to plot, in order (default: all found)")
p.add_argument("--paper_only", action="store_true",
               help="only the settings in the paper's Fig. 9: CacheGen levels 1-3, 3/4/8-bit "
                    "quantization, the delta codec and fp16 (drops our extension levels 0/-1/-2 and 2-bit)")
args = p.parse_args()

DATASET_TITLES = {"longchat": "LongChat", "nqa": "NarrativeQA", "tqa": "TriviaQA"}


def dataset_title(ds):
    """ "nqa_20k" -> "NarrativeQA, 20K tokens" """
    base, _, var = ds.partition("_")
    title = DATASET_TITLES.get(base, base)
    return f"{title}, {var.upper()} tokens" if var.endswith("k") else (f"{title} ({var})" if var else title)
METRIC_LABELS = {"accuracy": "Accuracy", "f1": "F1 score"}

# Fixed categorical order (first three slots validate under all-pairs CVD checks).
# Each series also gets its own marker + line style so identity is not color-alone.
STYLES = {
    "cachegen": dict(label="CacheGen", color="#2a78d6", marker="o", linestyle="-"),
    "quant":    dict(label="Uniform quantization", color="#eb6834", marker="s", linestyle="--"),
    "vanilla":  dict(label="No compression (fp16)", color="#1baf7a", marker="^", linestyle="none"),
    # Our reimplementation of the paper's delta encoding (src/delta_codec.py)
    "cachegen_delta":      dict(label="CacheGen + delta (ours)", color="#eda100", marker="D", linestyle=":"),
    "cachegen_delta_int8": dict(label="CacheGen + delta, int8 anchors (ours)", color="#e87ba4",
                                marker="v", linestyle="-."),
}
# Direct-label offsets (points) per series, so labels of nearby series don't collide
LABEL_OFFSET = {"cachegen": ((6, -12), "left"), "cachegen_delta": ((6, 6), "left"),
                "cachegen_delta_int8": ((6, 6), "left")}
METHOD_ORDER = ["cachegen", "quant", "vanilla", "cachegen_delta", "cachegen_delta_int8"]
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# Settings we added beyond the paper (scripts/reproduce/extensions.sh); --paper_only drops them
EXTENSION_SETTINGS = {"cachegen": ("level0", "level-1", "level-2"), "quant": ("2bit",)}


def load(dataset_dir):
    runs = {}
    for path in glob.glob(os.path.join(dataset_dir, "*.json")):
        with open(path) as f:
            r = json.load(f)
        if r.get("avg_size_mb") is None or r.get("avg_metric") is None:
            continue
        method = r["method"]
        if args.paper_only and (r["setting"] in EXTENSION_SETTINGS.get(method, ())):
            continue
        if method == "cachegen_delta" and r["setting"].endswith("_int8anchor"):
            method = "cachegen_delta_int8"
            r = dict(r, setting=r["setting"].replace("_int8anchor", ""))
        runs.setdefault(method, []).append(r)
    for m in runs:
        runs[m].sort(key=lambda r: r["avg_size_mb"])
    return runs


# Every sub-folder with result JSONs is a dataset; the paper's three come first.
# --datasets restricts/reorders the panels.
_found = sorted(d for d in os.listdir(args.results_dir)
                if not d.startswith("smoke") and glob.glob(os.path.join(args.results_dir, d, "*.json")))
datasets = [d for d in args.datasets if d in _found] if args.datasets else ([d for d in ["longchat", "nqa", "tqa"] if d in _found]
                             + [d for d in _found if d not in ("longchat", "nqa", "tqa")])
if not datasets:
    raise SystemExit(f"No results found under {args.results_dir}/")

def style_axis(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.margins(y=0.12)   # headroom so labels at accuracy 1.0 don't touch the title


def set_ticks(ax, ticks):
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def draw_series(ax, runs):
    """Draw every series on ax. Labels outside the axis' x-range are clipped automatically."""
    for method in METHOD_ORDER:
        if method not in runs:
            continue
        st = STYLES[method]
        kw = dict(color=st["color"], marker=st["marker"], linestyle=st["linestyle"], linewidth=2,
                  markersize=8, markeredgecolor="white", markeredgewidth=1.5, label=st["label"], zorder=3)
        if not runs[method]:     # entirely off-scale: keep the legend entry
            ax.plot([], [], **kw)
            continue
        ax.plot([r["avg_size_mb"] for r in runs[method]], [r["avg_metric"] for r in runs[method]], **kw)
        # Selective direct labels: name each point's setting (level / bit-width)
        offset, ha = LABEL_OFFSET.get(method, ((6, -12), "left"))
        for r in runs[method]:
            ax.annotate(r["setting"].replace("level", "L"), (r["avg_size_mb"], r["avg_metric"]),
                        textcoords="offset points", xytext=offset, ha=ha,
                        fontsize=8, color=TEXT_SECONDARY, annotation_clip=True)


panel_w = 6.0 if args.xbreak else 4.6
LEGEND_W = 2.8   # inches reserved for the legend on the right
fig = plt.figure(figsize=(panel_w * len(datasets) + LEGEND_W, 3.8), layout="constrained")
subfigs = fig.subfigures(1, len(datasets), squeeze=False)[0]
first_ax = None
for sf, ds in zip(subfigs, datasets):
    runs = load(os.path.join(args.results_dir, ds))
    metric_name = next((rs[0]["metric_name"] for rs in runs.values() if rs), "accuracy")
    sizes = [r["avg_size_mb"] for rs in runs.values() for r in rs]
    title = f"{args.model_name}, {dataset_title(ds)}"
    ylabel = METRIC_LABELS.get(metric_name, metric_name)

    if args.xbreak:
        # Broken x-axis: a wide panel for the interesting range, a narrow one for the rest
        if args.xbreak == ["auto"]:
            # Right panel: 8-bit quantization and fp16; left panel: everything else
            far = [r["avg_size_mb"] for m, rs in runs.items() for r in rs
                   if m == "vanilla" or (m == "quant" and r["setting"] == "8bit")]
            near = [r["avg_size_mb"] for m, rs in runs.items() for r in rs
                    if r["avg_size_mb"] not in far]
            if not near or not far:
                raise SystemExit(f"--xbreak auto needs both 8-bit/fp16 and smaller points; {ds} has "
                                 f"{len(near)} small and {len(far)} large. Use --xbreak A B or no break.")
            a, b = max(near) * 1.08, min(far) * 0.92
        else:
            a, b = (float(v) for v in args.xbreak)
        lo, hi = args.xlim if args.xlim else (min(sizes) * 0.92, max(sizes) * 1.05)
        ax_l, ax_r = sf.subplots(1, 2, sharey=True, gridspec_kw=dict(width_ratios=[3, 1], wspace=0.04))
        for ax, (x0, x1) in [(ax_l, (lo, a)), (ax_r, (b, hi))]:
            draw_series(ax, runs)
            style_axis(ax)
            ax.set_xlim(x0, x1)
        if args.xticks:
            set_ticks(ax_l, [t for t in args.xticks if lo <= t <= a])
            set_ticks(ax_r, [t for t in args.xticks if b <= t <= hi])
        else:
            if a - lo <= 300:
                ax_l.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(25))
            # Few, round ticks so the narrow panel's labels don't collide
            right = [t for t in (500, 600, 800, 1000, 1200, 1600) if b <= t <= hi]
            set_ticks(ax_r, right[::max(1, (len(right) + 1) // 2)][:2] or [round(hi, -2)])
        ax_r.spines["left"].set_visible(False)
        ax_r.tick_params(left=False, labelleft=False)
        # Diagonal break marks on the x-axis
        mark = dict(marker=[(-1, -1.5), (1, 1.5)], markersize=9, linestyle="none",
                    color=TEXT_SECONDARY, mec=TEXT_SECONDARY, mew=1, clip_on=False)
        ax_l.plot([1], [0], transform=ax_l.transAxes, **mark)
        ax_r.plot([0], [0], transform=ax_r.transAxes, **mark)
        ax_l.set_ylabel(ylabel, color=TEXT_SECONDARY)
        first_ax = first_ax or ax_l
    else:
        ax = sf.subplots()
        off_scale = []
        if args.xlim:
            lo, hi = args.xlim
            for method, rs in runs.items():
                off_scale += [f"{STYLES[method]['label']} {r['setting']}: {r['avg_size_mb']:.0f} MB, "
                              f"{r['avg_metric']:.2f}" for r in rs if not lo <= r["avg_size_mb"] <= hi]
                runs[method] = [r for r in rs if lo <= r["avg_size_mb"] <= hi]
        draw_series(ax, runs)
        style_axis(ax)
        if args.logx:
            ax.set_xscale("log")
        if args.xlim:
            ax.set_xlim(*args.xlim)
        elif not args.logx:
            ax.set_xlim(left=0)
        ticks = args.xticks or ([150, 200, 250, 300, 600, 1200] if args.logx else None)
        if ticks:
            set_ticks(ax, ticks)
        if off_scale:
            sf.text(0.02, 0.0, "Off-scale: " + "; ".join(off_scale), fontsize=7.5,
                    color=TEXT_SECONDARY, va="bottom", wrap=True)
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
        first_ax = first_ax or ax
    sf.suptitle(title, color=TEXT_PRIMARY, fontsize=11)
    sf.supxlabel("KV cache size (MB)" + (" (log scale)" if args.logx and not args.xbreak else ""),
                 color=TEXT_SECONDARY, fontsize=10)

handles, labels = first_ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="outside right center", frameon=False, labelcolor=TEXT_PRIMARY)
fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {args.out}")

# Table view of the same data (for the report / accessibility)
print(f"\n{'dataset':<10} {'method':<20} {'setting':<8} {'size (MB)':>10} {'quality':>8}  n")
for ds in datasets:
    for method, rs in load(os.path.join(args.results_dir, ds)).items():
        for r in rs:
            print(f"{ds:<10} {method:<20} {r['setting']:<8} {r['avg_size_mb']:>10.1f} "
                  f"{r['avg_metric']:>8.3f}  {r['num_docs']}")
