#!/bin/bash
# Extension experiments beyond the paper's Fig. 9 / Table 1, one GPU each, in parallel:
#
#   GPU 0  nqa       NarrativeQA, 50 docs: CacheGen levels -2..3, delta codec, 2/3/4/8-bit, fp16
#   GPU 1  tqa       TriviaQA, 50 docs: same as nqa
#   GPU 2  lowbit    LongChat, lower-bitrate CacheGen levels 0/-1/-2 and 2-bit quantization
#                    (reuses the saved LongChat KV caches if present), then the CPU
#                    delta-encoding analysis (scripts/reproduce/analyze_delta.py)
#   GPU 3  context   NarrativeQA, the same 30 questions at 10K/20K/30K tokens
#                    (test_data/nqa_{10,20,30}k.jsonl)
#
# Usage (repository root, conda env "cachegen" active):
#   bash scripts/reproduce/extensions.sh start      # launch all four in the background
#   bash scripts/reproduce/extensions.sh status     # progress of each job
#   bash scripts/reproduce/extensions.sh collect    # figures + tarball ~/extensions_results.tgz
#   bash scripts/reproduce/extensions.sh <job>      # run one job in the foreground
#
# Env: LOG_DIR (default ~/logs), JOBS (default "nqa tqa lowbit context"), FORCE=1 to
# skip the disk-space check. Jobs only add results to mistral7b_results/<dataset>/;
# the main LongChat results there are not overwritten.
set -e
cd "$(dirname "$0")/../.."
LOG_DIR=${LOG_DIR:-$HOME/logs}
JOBS=${JOBS:-"nqa tqa lowbit context"}
D=scripts/reproduce/fig9_table1.sh
RES=mistral7b_results
gpu_of() { case "$1" in nqa) echo 0 ;; tqa) echo 1 ;; lowbit) echo 2 ;; context) echo 3 ;; esac; }

run_job() {
    case "$1" in
    nqa|tqa)
        CLEAN_RAW=1 CACHEGEN_LEVELS="-2 -1 0 1 2 3" DELTA_LEVELS="1 2 3" QUANT_BITS="2 3 4 8" \
            bash $D "$1" 0 50 ;;
    lowbit)
        CACHEGEN_LEVELS="0 -1 -2" DELTA_LEVELS="" QUANT_BITS="2" RUN_VANILLA=0 RUN_ESTIMATE=0 \
            bash $D longchat 0 50
        echo "=== Delta-encoding analysis (CPU), LongChat docs 0-4, level 2 ==="
        python scripts/reproduce/analyze_delta.py ${SAVE_DIR:-.}/mistral7b_longchat_data/raw_kv_{0,1,2,3,4}.pt \
            --level 2 --out $RES/delta_analysis_longchat.json ;;
    context)
        for L in 10 20 30; do
            [ -f test_data/nqa_${L}k.jsonl ] || python scripts/reproduce/build_long_nqa.py --lengths 10000 20000 30000 --num 30
            CLEAN_RAW=1 RUN_ESTIMATE=0 CACHEGEN_LEVELS="1 2 3" DELTA_LEVELS="2" QUANT_BITS="4 8" \
                bash $D nqa_${L}k 0 30
        done ;;
    *) echo "unknown job: $1"; exit 2 ;;
    esac
}

case "${1:-}" in
start)
    nvidia-smi -L
    avail=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
    echo "free disk here: ${avail} GB (peak need is roughly 320 GB with all four jobs)"
    if [ "${avail:-0}" -lt 320 ] && [ "${FORCE:-0}" != "1" ]; then
        echo "Not enough disk for all jobs at once. Options: rm -rf mistral7b_longchat_data (if you"
        echo "won't run 'lowbit'), run fewer jobs (JOBS=\"nqa tqa\"), or FORCE=1 to start anyway."
        exit 1
    fi
    mkdir -p "$LOG_DIR"
    for j in $JOBS; do
        if pgrep -f "extensions.sh $j\$" > /dev/null; then echo "$j already running"; continue; fi
        rm -f "$LOG_DIR/ext_$j.done"
        CUDA_VISIBLE_DEVICES=$(gpu_of $j) nohup setsid bash -c \
            "bash scripts/reproduce/extensions.sh $j > '$LOG_DIR/ext_$j.log' 2>&1; echo \$? > '$LOG_DIR/ext_$j.done'" \
            > /dev/null 2>&1 < /dev/null &
        echo "started $j on GPU $(gpu_of $j) -> $LOG_DIR/ext_$j.log"
    done ;;
status)
    for j in $JOBS; do
        state="running"; [ -f "$LOG_DIR/ext_$j.done" ] && state="finished, exit code $(cat "$LOG_DIR/ext_$j.done")"
        [ -f "$LOG_DIR/ext_$j.log" ] || state="not started"
        echo "== $j (GPU $(gpu_of $j)): $state"
        [ -f "$LOG_DIR/ext_$j.log" ] && { grep "^===" "$LOG_DIR/ext_$j.log" | tail -2; grep -iE "error|Traceback|out of memory" "$LOG_DIR/ext_$j.log" | tail -3; }
    done
    df -h . | tail -1 ;;
collect)
    OUT=$HOME/extensions_results; rm -rf "$OUT"; mkdir -p "$OUT/figures" "$OUT/logs"
    cp -r $RES "$OUT/data"
    python scripts/reproduce/plot_fig9.py --results_dir $RES --datasets longchat nqa tqa \
        --out "$OUT/figures/fig9_all_datasets.png" --xbreak auto | sed '1,2d' > "$OUT/fig9_all_datasets_table.txt" || true
    python scripts/reproduce/plot_context_length.py --results_dir $RES \
        --out "$OUT/figures/context_length.png" | sed '1,2d' > "$OUT/context_length_table.txt" || true
    python scripts/reproduce/make_table1.py --results_dir $RES > /dev/null && mv $RES/table1_longchat.md "$OUT/" || true
    cp "$LOG_DIR"/ext_*.log "$OUT/logs/" 2>/dev/null || true
    { nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
      git log -1 --format='code: %H %s'; echo "collected: $(date -u +%FT%TZ)"; } > "$OUT/hardware.txt"
    tar czf ~/extensions_results.tgz -C ~ extensions_results
    echo "Wrote ~/extensions_results.tgz"; ls -R "$OUT" | head -40 ;;
nqa|tqa|lowbit|context)
    run_job "$1" ;;
*)
    sed -n '2,24p' "$0"; exit 2 ;;
esac
