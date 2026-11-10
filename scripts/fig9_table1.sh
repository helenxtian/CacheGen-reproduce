#!/bin/bash
# Reproduce CacheGen (SIGCOMM'24) Figure 9 (KV cache size vs. quality) and the
# CacheGen / 8-bit quantization rows of Table 1, on Mistral-7B with a single GPU.
#
# Usage:  bash scripts/fig9_table1.sh [dataset] [start] [end]
#   dataset: longchat (default) | nqa | tqa
#   start/end: doc range, default 0 50  (use e.g. 0 5 for a quick smoke test)
#
# Env vars:
#   SAVE_DIR        where raw/encoded KV caches go (default .)   ~1.3 GB per doc!
#   MAX_GPU_MEM     GiB per GPU (default 76). Note: FastChat ignores this when
#                   num_gpus == 1, so it has no effect in this single-GPU setup.
#   LONGCHAT_JUDGE  string (default, no API key) | gpt (needs OPENAI_API_KEY)
#   QUANT_BITS      quantization baselines to run (default "3 4 8")
#   CACHEGEN_LEVELS CacheGen encoding levels to run (default "1 2 3")
set -e

DATASET=${1:-longchat}
START=${2:-0}
END=${3:-50}
SAVE_DIR=${SAVE_DIR:-.}
MAX_GPU_MEM=${MAX_GPU_MEM:-76}
QUANT_BITS=${QUANT_BITS:-"3 4 8"}
CACHEGEN_LEVELS=${CACHEGEN_LEVELS:-"1 2 3"}
export LONGCHAT_JUDGE=${LONGCHAT_JUDGE:-string}

export MODEL=mistral-community/Mistral-7B-v0.2
export MODEL_ID=mistral7b
DATA_DIR=${SAVE_DIR}/${MODEL_ID}_${DATASET}_data
RESULTS_DIR=${MODEL_ID}_results

echo "=== [1/4] Saving raw KV caches (${DATASET}, docs ${START}-${END}) ==="
python main.py \
    --model_id $MODEL \
    --save_dir ${DATA_DIR}/ \
    --start $START --end $END \
    --num_gpus 1 --max_gpu_memory $MAX_GPU_MEM \
    --dataset_name $DATASET

for LEVEL in $CACHEGEN_LEVELS; do
    echo "=== [2/4] CacheGen, encoding level ${LEVEL} ==="
    python run_cachegen.py \
        --model_id $MODEL \
        --save_dir ${DATA_DIR} \
        --start $START --end $END \
        --num_gpus 1 --max_gpu_memory $MAX_GPU_MEM \
        --encoded_dir ${SAVE_DIR}/encoded_${DATASET} \
        --results_dir ${RESULTS_DIR}/ \
        --dataset_name $DATASET \
        --quant_level $LEVEL \
        --calculate_metric 1
done

for BITS in $QUANT_BITS; do
    BINS=$((2 ** BITS))
    echo "=== [3/4] Uniform quantization baseline, ${BITS}-bit (${BINS} bins) ==="
    python run_quantization_baseline.py \
        --model_id $MODEL \
        --dataset_name $DATASET \
        --save_dir ${DATA_DIR}/ \
        --start $START --end $END \
        --num_gpus 1 --max_gpu_memory $MAX_GPU_MEM \
        --bins $BINS \
        --results_dir ${RESULTS_DIR}/ \
        --calculate_metric 1
done

echo "=== [4/4] Vanilla (uncompressed fp16 KV cache / full prefill) ==="
python run_vanilla.py \
    --model_id $MODEL \
    --dataset_name $DATASET \
    --save_dir ${DATA_DIR}/ \
    --start $START --end $END \
    --num_gpus 1 --max_gpu_memory $MAX_GPU_MEM \
    --results_dir ${RESULTS_DIR}/ \
    --calculate_metric 1

echo "=== Done. Results in ${RESULTS_DIR}/${DATASET}/ ==="
echo "Plot Figure 9:  python plot_fig9.py --results_dir ${RESULTS_DIR}"
echo "Print Table 1:  python make_table1.py --results_dir ${RESULTS_DIR}"
