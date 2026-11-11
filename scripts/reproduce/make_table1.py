"""
Print the reproducible rows of Table 1 of CacheGen (SIGCOMM'24):
"Performance of CacheGen and the baselines on Mistral-7B with LongChat dataset".

Only the "8-bit quantization" and "CacheGen" rows are reproducible with this repo;
H2O and LLMLingua (and CacheGen on top of them) are not implemented here.

Usage: python scripts/reproduce/make_table1.py --results_dir mistral7b_results [--cachegen_level level2]
"""
import argparse
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--results_dir", type=str, default="mistral7b_results")
p.add_argument("--dataset", type=str, default="longchat")
p.add_argument("--cachegen_level", type=str, default="level2",
               help="Which CacheGen level to report (level2 is the repo's default config)")
args = p.parse_args()

PAPER = {  # values reported in the paper (200 LongChat contexts, GPT judge)
    "8-bit quantization": (622, 1.00),
    "CacheGen": (176, 0.98),
}
ROWS = [("8-bit quantization", "quant_8bit.json"),
        ("CacheGen", f"cachegen_{args.cachegen_level}.json"),
        # Our reimplementation of the paper's delta encoding (src/delta_codec.py)
        ("CacheGen + delta (ours)", f"cachegen_delta_{args.cachegen_level}.json")]

d = os.path.join(args.results_dir, args.dataset)
lines = ["| Technique | Size (MB), ours | Accuracy, ours | Size (MB), paper | Accuracy, paper |",
         "|---|---|---|---|---|"]
judge, n = None, None
for name, fname in ROWS:
    path = os.path.join(d, fname)
    if os.path.exists(path):
        r = json.load(open(path))
        ours = f"{r['avg_size_mb']:.0f} | {r['avg_metric']:.2f}"
        judge, n = r.get("judge"), r.get("num_docs")
    else:
        ours = "missing | missing"
    if name not in PAPER:
        if os.path.exists(path):
            lines.append(f"| {name} | {ours} | n/a | n/a |")
        continue
    ps, pa = PAPER[name]
    lines.append(f"| {name} | {ours} | {ps} | {pa:.2f} |")

# Normalized accuracy like the paper (relative to the uncompressed / 8-bit result)
vanilla = os.path.join(d, "vanilla_fp16.json")
if os.path.exists(vanilla):
    v = json.load(open(vanilla))
    lines.append(f"| (no compression, fp16) | {v['avg_size_mb']:.0f} | {v['avg_metric']:.2f} | n/a | n/a |")

print("\n".join(lines))
print(f"\nOurs: {n} docs, LongChat judge = {judge}. H2O / LLMLingua rows not reproduced (not in repo).")

out = os.path.join(args.results_dir, f"table1_{args.dataset}.md")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved {out}")
