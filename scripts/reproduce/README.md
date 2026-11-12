# Reproduction scripts

These are the scripts we wrote to reproduce Figure 9 and Table 1 of CacheGen (SIGCOMM '24) and to run some extensions. The authors' scripts are in `scripts/` one level up. Run everything from the repository root, with the `cachegen` conda environment active.

## Reproduction

| Script | What it does |
|---|---|
| `fig9_table1.sh [dataset] [start] [end]` | Runs the full pipeline for one dataset: saves raw KV caches, then runs CacheGen, the delta codec, uniform quantization and fp16. Writes one JSON per method/setting to `mistral7b_results/<dataset>/`. Its environment variables are listed at the top of the file. |
| `plot_fig9.py` | Draws Figure 9, KV cache size vs. quality, with one panel per dataset. Options: `--xbreak A B` or `--xbreak auto` for a broken x-axis, `--logx`, `--xlim`, `--xticks`, `--datasets`, and `--paper_only` to drop our extension settings (CacheGen levels 0/-1/-2, 2-bit). |
| `build_notebook.py` | Generates the Trovi notebook `cachegen_fig9_table1.ipynb`. Edit this script, not the notebook: `python scripts/reproduce/build_notebook.py cachegen_fig9_table1.ipynb`. |
| `make_table1.py` | Builds Table 1 for LongChat from our results and puts it next to the paper's numbers. |

## Extensions

| Script | What it does |
|---|---|
| `extensions.sh start\|status\|collect` | Runs four experiments in parallel, one GPU each: (1) NarrativeQA and (2) TriviaQA, both with every CacheGen level; (3) lower-bitrate CacheGen levels on LongChat; (4) the context-length experiment. `collect` packs the figures and data into `~/extensions_results.tgz`. |
| `merge_extensions.sh <tarball or folder>` | Merges the output of `extensions.sh collect` into `reproduce-results/` and redraws `figures/fig9_all_datasets.png`. It never overwrites the V100 LongChat files. |
| `build_long_nqa.py` | Builds `test_data/nqa_{10,20,30}k.jsonl`: the same 30 LongBench NarrativeQA questions at 10K, 20K and 30K tokens, using the artifact's prompt template and its middle truncation. |
| `plot_context_length.py` | Plots F1 and KV size per 1K tokens against context length. |
| `analyze_delta.py` | CPU analysis of why delta encoding didn't help. It compares values, keys as cached (post-RoPE) and keys with RoPE undone. |
| `rejudge_longchat.py` | Re-scores LongChat answers from a run log with the artifact's GPT judge. Needs `OPENAI_API_KEY`; no GPU. |

## Our additions outside this folder

- **Lower-bitrate CacheGen levels:** `0`, `-1` and `-2` in `LMCache/lmcache/storage_backend/serde/cachegen_basics.py`. They are below the artifact's levels 1-3.
- **Delta-encoding codec:** `src/delta_codec.py`, our reimplementation of paper §5.2.
- **Long-context datasets:** `src/utils.py` scores dataset variants such as `nqa_20k` like their base dataset.
