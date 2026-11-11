# Reproduction results: CacheGen (SIGCOMM '24) Figure 9 and Table 1

These are the results of our reproduction run: Mistral-7B on 50 LongChat documents, produced with `scripts/reproduce/fig9_table1.sh` (or the notebook `cachegen_fig9_table1.ipynb`).

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

The delta stream alone, which covers 90% of the tokens, is already about as large as the released codec's whole output. So on this model, deltas between tokens don't compress better than the quantized values themselves.

## Files

| Path | Contents |
|---|---|
| `table1_longchat.md` | Table 1 written by `scripts/reproduce/make_table1.py` |
| `fig9_table.txt` | Text table of every Figure 9 point, printed by `scripts/reproduce/plot_fig9.py` |
| `figures/fig9_break.png` | Figure 9 with a broken x-axis (130-310 MB, then 520+ MB). **Use this one in the report.** |
| `figures/fig9_log.png` | Figure 9 with a log-scale x-axis |
| `figures/fig9.png` | Figure 9 with a linear x-axis starting at 0 |
| `figures/fig9_as_run.png` | The figure as generated on the node during the run |
| `figures/smoke_fig9.png` | 3-document smoke test |
| `data/longchat/*.json` | One file per method and setting. Each holds the average size and accuracy, the per-document sizes and accuracies, the judge, and the document count. Delta runs also include the size breakdown. |
| `data/smoke_longchat/*.json` | Smoke-test results (documents 0-2) |
| `logs/full.log`, `logs/smoke.log` | Complete console output of both runs |

## Regenerating the figures and table from this data

```bash
python scripts/reproduce/plot_fig9.py   --results_dir reproduce-results/data --out reproduce-results/figures/fig9_break.png --xbreak 310 520
python scripts/reproduce/plot_fig9.py   --results_dir reproduce-results/data --out reproduce-results/figures/fig9_log.png --logx
python scripts/reproduce/make_table1.py --results_dir reproduce-results/data
```
