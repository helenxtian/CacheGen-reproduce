#!/bin/bash
# Reproduce CacheGen (SIGCOMM'24) Figure 9 (KV cache size vs. quality) and the
# CacheGen / 8-bit quantization rows of Table 1, on Mistral-7B with a single GPU.
#
# Usage:  bash scripts/reproduce/fig9_table1.sh [dataset] [start] [end]
#   dataset: longchat (default) | nqa | tqa | nqa_10k / nqa_20k / nqa_30k
#            (long-context variants built by scripts/reproduce/build_long_nqa.py)
#   start/end: doc range, default 0 50  (use e.g. 0 5 for a quick smoke test)
#
# Env vars:
#   SAVE_DIR        where raw/encoded KV caches go (default .)   ~1.3 GB per doc!
#   MAX_GPU_MEM     GiB per GPU (default 76). Note: FastChat ignores this when
#                   num_gpus == 1, so it has no effect in this single-GPU setup.
#   LONGCHAT_JUDGE  string (default, no API key) | gpt (needs OPENAI_API_KEY)
#   QUANT_BITS      quantization baselines to run (default "3 4 8")
#   CACHEGEN_LEVELS CacheGen encoding levels to run (default "1 2 3"; our extension
#                   adds lower-bitrate levels "0 -1 -2")
#   DELTA_LEVELS    levels for the delta codec (default: same as CACHEGEN_LEVELS)
#   RUN_VANILLA     0 = skip the uncompressed baseline (default 1)
#   RUN_ESTIMATE    0 = skip the CPU delta-encoding estimate (default 1)
#   RESULTS_DIR     where result JSONs go (default mistral7b_results)
#   CLEAN_RAW       1 = delete this dataset's raw KV caches (DATA_DIR) and encoded
#                   files at the end, to save disk (default 0)
#   DELTA_ANCHORS   anchor modes for our delta-encoding codec (default "level";
#                   "level int8" runs both; "" skips the delta codec)
set -e
# Run from the repository root no matter where this script is called from
cd "$(dirname "$0")/../.."

DATASET=${1:-longchat}
START=${2:-0}
END=${3:-50}
SAVE_DIR=${SAVE_DIR:-.}
MAX_GPU_MEM=${MAX_GPU_MEM:-76}
QUANT_BITS=${QUANT_BITS:-"3 4 8"}
CACHEGEN_LEVELS=${CACHEGEN_LEVELS:-"1 2 3"}
DELTA_ANCHORS=${DELTA_ANCHORS-"level"}
DELTA_LEVELS=${DELTA_LEVELS-$CACHEGEN_LEVELS}
RUN_VANILLA=${RUN_VANILLA:-1}
RUN_ESTIMATE=${RUN_ESTIMATE:-1}
export LONGCHAT_JUDGE=${LONGCHAT_JUDGE:-string}

export MODEL=mistral-community/Mistral-7B-v0.2
export MODEL_ID=mistral7b
DATA_DIR=${SAVE_DIR}/${MODEL_ID}_${DATASET}_data
RESULTS_DIR=${RESULTS_DIR:-${MODEL_ID}_results}

echo "=== [1/5] Saving raw KV caches (${DATASET}, docs ${START}-${END}) ==="
python main.py \
    --model_id $MODEL \
    --save_dir ${DATA_DIR}/ \
    --start $START --end $END \
    --num_gpus 1 --max_gpu_memory $MAX_GPU_MEM \
    --dataset_name $DATASET

if [ "$RUN_ESTIMATE" = "1" ]; then
    echo "=== CPU estimate of delta encoding on doc ${START} (level 2, no CUDA coder) ==="
    python -m src.delta_codec ${DATA_DIR}/raw_kv_${START}.pt 2 || echo "(estimate failed; continuing)"
fi

for LEVEL in $CACHEGEN_LEVELS; do
    echo "=== [2/5] CacheGen, encoding level ${LEVEL} ==="
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

for ANCHOR in $DELTA_ANCHORS; do
    for LEVEL in $DELTA_LEVELS; do
        echo "=== [3/5] CacheGen + delta encoding (ours), level ${LEVEL}, ${ANCHOR} anchors ==="
        python run_cachegen.py \
            --model_id $MODEL \
            --save_dir ${DATA_DIR} \
            --start $START --end $END \
            --num_gpus 1 --max_gpu_memory $MAX_GPU_MEM \
            --encoded_dir ${SAVE_DIR}/encoded_${DATASET} \
            --results_dir ${RESULTS_DIR}/ \
            --dataset_name $DATASET \
            --quant_level $LEVEL \
            --codec delta --anchor_mode $ANCHOR \
            --calculate_metric 1
    done
done

for BITS in $QUANT_BITS; do
    BINS=$((2 ** BITS))
    echo "=== [4/5] Uniform quantization baseline, ${BITS}-bit (${BINS} bins) ==="
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

if [ "$RUN_VANILLA" = "1" ]; then
    echo "=== [5/5] Vanilla (uncompressed fp16 KV cache / full prefill) ==="
    python run_vanilla.py \
        --model_id $MODEL \
        --dataset_name $DATASET \
        --save_dir ${DATA_DIR}/ \
        --start $START --end $END \
        --num_gpus 1 --max_gpu_memory $MAX_GPU_MEM \
        --results_dir ${RESULTS_DIR}/ \
        --calculate_metric 1
fi

if [ "${CLEAN_RAW:-0}" = "1" ]; then
    echo "=== CLEAN_RAW=1: deleting ${DATA_DIR} and ${SAVE_DIR}/encoded_${DATASET} ==="
    rm -rf "${DATA_DIR}" "${SAVE_DIR}/encoded_${DATASET}"
fi

echo "=== Done. Results in ${RESULTS_DIR}/${DATASET}/ ==="
echo "Plot Figure 9:  python scripts/reproduce/plot_fig9.py --results_dir ${RESULTS_DIR} --xbreak 310 520"
echo "Print Table 1:  python scripts/reproduce/make_table1.py --results_dir ${RESULTS_DIR}"
