# Results: CacheGen (SIGCOMM '24) Figure 9, Table 1, and extensions

All of our results are in this folder:

1. **Reproduction** (V100): Figure 9 and Table 1 for Mistral-7B on 50 LongChat documents, from `scripts/reproduce/fig9_table1.sh` or the notebook `cachegen_fig9_table1.ipynb`.
2. **Extensions** (P100), from `scripts/reproduce/extensions.sh`:
   - the other two Figure 9 datasets, NarrativeQA and TriviaQA;
   - CacheGen encoding levels below the artifact's lowest;
   - an analysis of why delta encoding didn't help.

The two parts ran on different GPUs; see [Hardware](#hardware-and-comparability).

# Part 1: Reproduction

## Setup

| | |
|---|---|
| Hardware | Chameleon CHI@UC, node type `gpu_v100`: 1x Tesla V100-SXM2-32GB (driver 560.35.05), image `CC-Ubuntu24.04-CUDA`. The paper used A40/A100 GPUs. |
| Software | The authors' `env.yaml`: torch 2.3.1, transformers 4.42.3, fschat 0.2.36, LMCache plus the `torchac_cuda` coder |
| Model | `mistral-community/Mistral-7B-v0.2`, weights loaded in 8-bit by FastChat; the KV cache is fp16 |
| Data | LongChat documents 0-49 (`test_data/longchat.jsonl`); average context 8,935 tokens |
| Quality metric | Accuracy using a substring match of the label in the model's answer (`LONGCHAT_JUDGE=string`). The paper used a GPT judge. |

## Table 1 (Mistral-7B, LongChat)

| Technique | Size (MB), ours | Accuracy, ours | Size (MB), paper | Accuracy, paper |
|---|---|---|---|---|
| 8-bit quantization | 587 | 1.00 | 622 | 1.00 |
| CacheGen (level 2) | 173 | 1.00 | 176 | 0.98 |
| CacheGen + delta encoding (our reimplementation) | 196 | 1.00 | n/a | n/a |
| No compression (fp16) | 1171 | 1.00 | n/a | n/a |

The paper's H2O and LLMLingua rows were not reproduced, because they aren't part of the authors' artifact.

## Figure 9 data (all points)

| Method | Setting | Size (MB) | Accuracy |
|---|---|---|---|
| CacheGen | level 1 | 136.9 | 1.00 |
| CacheGen | level 2 | 172.9 | 1.00 |
| CacheGen | level 3 | 234.1 | 1.00 |
| Uniform quantization | 3-bit | 220.7 | 0.82 |
| Uniform quantization | 4-bit | 293.9 | 1.00 |
| Uniform quantization | 8-bit | 586.7 | 1.00 |
| CacheGen + delta (ours) | level 1 | 160.5 | 1.00 |
| CacheGen + delta (ours) | level 2 | 195.6 | 1.00 |
| CacheGen + delta (ours) | level 3 | 255.0 | 1.00 |
| No compression | fp16 | 1171.1 | 1.00 |

**Main finding:**
- CacheGen at level 1 (137 MB) keeps full accuracy.
- Uniform quantization loses accuracy at a larger size: 3-bit is 221 MB and scores 0.82.
- CacheGen at level 2 is 3.4x smaller than 8-bit quantization; the paper reports 3.5x.

**Delta encoding** (paper §5.2) is missing from the authors' released codec, so we reimplemented it in `src/delta_codec.py`. It was *larger* than the released codec at every level (+17%, +13% and +9% at levels 1, 2 and 3), with no clamping (`clamped_frac` = 0). The average size breakdown at level 2:

| Component | MB |
|---|---|
| Delta stream | 158.6 |
| Anchors | 15.8 |
| Coder metadata | 20.0 |
| Scales | 1.1 |
| **Total** | **195.6** |

The delta stream alone, which covers 90% of the tokens, is already about as large as the released codec's whole output. So on this model, deltas between tokens don't compress better than the quantized values themselves. Part 2 explains why.

# Part 2: Extensions

All extension runs used Mistral-7B with the same code and settings as Part 1, on 50 documents per dataset.

## Lower CacheGen levels on LongChat

The artifact's lowest level, level 1, kept full accuracy. So we added levels 0, -1 and -2 with fewer quantization bins (`LMCache/lmcache/storage_backend/serde/cachegen_basics.py`) and 2-bit uniform quantization as its counterpart.

| Method | Setting | Size (MB) | Accuracy | Run on |
|---|---|---|---|---|
| CacheGen | level 1 (artifact's lowest) | 136.9 | 1.00 | V100 |
| CacheGen | level 0 (ours) | 96.7 | 1.00 | P100 |
| CacheGen | level -1 (ours) | 60.6 | 0.44 | P100 |
| CacheGen | level -2 (ours) | 26.2 | 0.00 | P100 |
| Uniform quantization | 2-bit (ours) | 147.5 | 0.00 | P100 |
| Uniform quantization | 3-bit | 220.7 | 0.82 | V100 |

- **Level 0 keeps full accuracy at 97 MB.** That is 1.8x smaller than level 2, the artifact's default and the paper's Table 1 setting, and 6x smaller than 8-bit quantization. The released levels leave compression unused on this task.
- **Quality falls off sharply below level 0:** 0.44 at level -1, 0 at level -2.
- **Uniform quantization fails much earlier.** 2-bit scores 0 at 148 MB, which is larger than CacheGen level 0 at full accuracy.

## Why delta encoding didn't help

`scripts/reproduce/analyze_delta.py` measures, on LongChat documents 0-4 at level 2 (`data/delta_analysis_longchat.json`):
- the entropy-coded size of each stream with and without delta encoding;
- a variance ratio, Var(token − its group's anchor) / Var(token). This is about 0 for smooth streams and 2 for unrelated tokens. Delta encoding can only save space when it is clearly below 1.

Our hypothesis was RoPE. Hugging Face caches keys *after* rotary position embedding, which rotates each token by a position-dependent angle. So we also measured keys with RoPE undone (θ = 10⁶ from the model config).

| Stream | Without delta (MB) | With delta (MB) | Delta / without | Variance ratio |
|---|---|---|---|---|
| Values | 90.7 | 103.7 | 1.14 | 1.69 |
| Keys as cached (post-RoPE) | 67.4 | 70.5 | 1.05 | 1.20 |
| Keys with RoPE undone | 52.8 | 64.9 | 1.23 | 1.59 |

The averages are over 5 documents, from 8,815 to 9,091 tokens; the five documents agree to within ±0.01.

- **RoPE is not the cause.** Undoing it makes delta encoding *worse* (1.23 vs. 1.05).
- **The variance ratio is above 1 for every stream.** A token's difference from its anchor varies more than the token itself, so neighbouring tokens in Mistral-7B's KV cache are close to independent channel by channel.
- The paper's §5.2 motivation, that nearby tokens have similar KV values, does not hold for this model at the granularity the codec uses. This explains why our delta codec came out larger on all three datasets (next section).

## NarrativeQA and TriviaQA (rest of Figure 9)

Both datasets ran on the P100. Each used the artifact's prompts, documents 0-49, and F1 against the reference answers as the quality metric. From the fp16 sizes, the contexts average about 9,800 tokens for NarrativeQA and 4,700 for TriviaQA. `figures/fig9_all_datasets.png` shows all three datasets; `fig9_all_datasets_table.txt` lists every point.

| Method | Setting | NarrativeQA MB | NarrativeQA F1 | TriviaQA MB | TriviaQA F1 |
|---|---|---|---|---|---|
| CacheGen | level -2 (ours) | 29.9 | 0.000 | 15.0 | 0.000 |
| CacheGen | level -1 (ours) | 69.4 | 0.124 | 34.4 | 0.425 |
| CacheGen | level 0 (ours) | 109.6 | 0.193 | 54.2 | 0.854 |
| CacheGen | level 1 | 154.3 | 0.221 | 76.2 | 0.908 |
| CacheGen | level 2 | 193.8 | 0.186 | 96.4 | 0.908 |
| CacheGen | level 3 | 261.4 | 0.230 | 129.1 | 0.908 |
| CacheGen + delta (ours) | level 1 | 181.4 | 0.200 | 92.1 | 0.888 |
| CacheGen + delta (ours) | level 2 | 220.1 | 0.188 | 111.8 | 0.888 |
| CacheGen + delta (ours) | level 3 | 286.0 | 0.223 | 143.6 | 0.908 |
| Uniform quantization | 2-bit (ours) | 162.1 | 0.004 | 78.3 | 0.012 |
| Uniform quantization | 3-bit | 242.5 | 0.161 | 117.1 | 0.789 |
| Uniform quantization | 4-bit | 322.9 | 0.193 | 156.0 | 0.908 |
| Uniform quantization | 8-bit | 644.5 | 0.212 | 311.4 | 0.908 |
| No compression | fp16 | 1286.6 | 0.212 | 621.5 | 0.908 |

**Size: CacheGen's compression carries over.** At level 2, CacheGen is 3.3x smaller than 8-bit quantization on NarrativeQA and 3.2x smaller on TriviaQA, against 3.4x on LongChat. The paper reports 3.5x on LongChat.

**Quality, TriviaQA: the same result as LongChat.**
- CacheGen level 1 (76 MB) matches fp16 exactly (0.908).
- The smallest uniform quantization that does so is 4-bit (156 MB), so CacheGen is 2.0x smaller at equal quality.
- At about the same size, 2-bit quantization (78 MB) scores 0.01.

**Quality, NarrativeQA: too noisy to rank the methods.**
- F1 is low for every method, including fp16 (0.21).
- Answers are free-form, so most per-question scores are partial matches.
- The differences between CacheGen levels 0-3, 4-bit and fp16 are all within 1.5 standard errors of the paired per-question differences (SE 0.015-0.025). For example, level 2 (0.186) scores below level 1 (0.221).
- 8-bit gets the same score as fp16 on every one of the 50 questions.
- The clear signal is the low end: quality drops at level -1 (0.12) and at 3-bit (0.16), and collapses at 2-bit.

**Delta encoding is larger on all three datasets.** Compared with the released codec at the same level:

| Level | LongChat | NarrativeQA | TriviaQA |
|---|---|---|---|
| 1 | +17% | +18% | +21% |
| 2 | +13% | +14% | +16% |
| 3 | +9% | +9% | +11% |

Nothing was clamped (`clamped_frac` = 0) on any dataset.

## Not run: context length

We planned to run the same 30 NarrativeQA questions at 10K, 20K and 30K tokens. The prompts are in `test_data/nqa_{10,20,30}k.jsonl` (`scripts/reproduce/build_long_nqa.py`).

On the 16 GB P100, a 20K-token prompt ran out of GPU memory during prefill. This happened even with our low-memory prefill (`--low_mem_prefill`), which avoids computing full-sequence logits. The 8-bit weights plus a 20K-token fp16 KV cache and activations don't fit. A 24 GB or larger GPU should be able to run `extensions.sh context`. This experiment goes beyond the paper, so the reproduction does not depend on it.

# Hardware and comparability

| Part | Hardware |
|---|---|
| Part 1, and the CacheGen level 1-3, 3/4/8-bit and fp16 LongChat points above | Chameleon CHI@UC, 1x Tesla V100-SXM2-32GB, driver 560.35.05 |
| Part 2 (LongChat levels 0/-1/-2 and 2-bit, NarrativeQA, TriviaQA, delta analysis) | Chameleon CHI@TACC, 1x Tesla P100-SXM2-16GB per job, driver 560.35.05 (`logs/hardware_extensions.txt`) |

Both used the same software environment, model, prompts and code.

**Sizes are hardware-independent.** KV cache sizes depend only on the quantization and the coder, so they can be compared across GPUs.

**Quality is generated with greedy decoding.** Different GPUs can change fp16 rounding enough to flip a borderline answer. So we compare quality only within one dataset and one GPU, with one exception: the LongChat table above, where the V100 points serve only as reference markers.

The P100 lacks bf16 and has less memory, but neither affects the settings we ran.

# Files

| Path | Contents |
|---|---|
| `table1_longchat.md` | Table 1 written by `scripts/reproduce/make_table1.py` |
| `fig9_table.txt` | Text table of every Figure 9 point, printed by `scripts/reproduce/plot_fig9.py` |
| `figures/fig9_break.png` | Figure 9 with a broken x-axis (130-310 MB, then 520+ MB). **Use this one in the report.** |
| `figures/fig9_log.png` | Figure 9 with a log-scale x-axis |
| `figures/fig9.png` | Figure 9 with a linear x-axis starting at 0 |
| `figures/fig9_as_run.png` | The figure as generated on the node during the run |
| `figures/smoke_fig9.png` | 3-document smoke test |
| `figures/fig9_all_datasets.png` | All three datasets with every setting, including our extension levels (Part 2) |
| `fig9_all_datasets_table.txt` | Text table of every point in `fig9_all_datasets.png` |
| `data/longchat/*.json` | One file per method and setting. Each holds the average size and accuracy, the per-document sizes and accuracies, the judge, and the document count. Delta runs also include the size breakdown. Levels 0/-1/-2 and 2-bit are from the P100 (Part 2); the rest are from the V100. |
| `data/nqa/*.json`, `data/tqa/*.json` | NarrativeQA and TriviaQA, same format, F1 score (P100) |
| `data/delta_analysis_longchat.json` | Per-document output of `analyze_delta.py` |
| `data/smoke_longchat/*.json` | Smoke-test results (documents 0-2) |
| `logs/full.log`, `logs/smoke.log` | Complete console output of both Part 1 runs |
| `logs/ext_*.log`, `logs/hardware_extensions.txt` | Console output of each extension job, and the GPU/driver/code version they ran with. `ext_nqa.log` ends with an out-of-memory error in our delta decoder, which we then fixed (commit d22f5ba, bit-identical output); `ext_nqa_resume.log` runs the remaining NarrativeQA steps. |

# Regenerating the figures and tables from this data

`--paper_only` leaves out our extension levels, so the Part 1 figures show exactly the paper's settings:

```bash
python scripts/reproduce/plot_fig9.py   --results_dir reproduce-results/data --datasets longchat --paper_only --out reproduce-results/figures/fig9_break.png --xbreak 310 520
python scripts/reproduce/plot_fig9.py   --results_dir reproduce-results/data --datasets longchat --paper_only --out reproduce-results/figures/fig9_log.png --logx
python scripts/reproduce/make_table1.py --results_dir reproduce-results/data
```

The extension results come in through the tarball from `extensions.sh collect`. This merges it and redraws `fig9_all_datasets.png`; V100 files are never overwritten:

```bash
bash scripts/reproduce/merge_extensions.sh ~/extensions_results.tgz
```
