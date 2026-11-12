#!/bin/bash
# Merge the output of 'extensions.sh collect' (the tarball or its unpacked folder) into
# reproduce-results/, next to the main V100 reproduction, and redraw the combined figure.
#
#   bash scripts/reproduce/merge_extensions.sh ~/extensions_results.tgz
#   bash scripts/reproduce/merge_extensions.sh path/to/extensions_results
#
# NarrativeQA and TriviaQA replace any earlier copy. LongChat extension runs (levels
# 0/-1/-2, 2-bit) are added to data/longchat/ next to the V100 runs, and no V100 file is
# ever overwritten. PYTHON selects the interpreter (needs matplotlib).
set -eo pipefail
cd "$(dirname "$0")/../.."
SRC=${1:?usage: merge_extensions.sh <extensions_results.tgz | extensions_results dir>}
PY=${PYTHON:-python}
R=reproduce-results
if [ -f "$SRC" ]; then
    TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
    tar xzf "$SRC" -C "$TMP"; SRC=$TMP/extensions_results
fi
[ -d "$SRC/data" ] || { echo "no data/ folder in $SRC"; exit 1; }

for ds in nqa tqa; do
    if [ -d "$SRC/data/$ds" ]; then
        rm -rf "$R/data/$ds"; cp -r "$SRC/data/$ds" "$R/data/$ds"
        echo "$ds: $(ls "$R/data/$ds"/*.json | wc -l | tr -d ' ') result files"
    fi
done
for f in "$SRC"/data/longchat/*.json; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    if [ -e "$R/data/longchat/$b" ]; then echo "longchat: kept the V100 $b (not overwritten)"
    else cp "$f" "$R/data/longchat/$b"; echo "longchat: added $b"; fi
done
[ -f "$SRC/data/delta_analysis_longchat.json" ] && cp "$SRC/data/delta_analysis_longchat.json" "$R/data/"
mkdir -p "$R/logs"
for f in "$SRC"/logs/ext_*.log; do [ -e "$f" ] && cp "$f" "$R/logs/"; done
[ -f "$SRC/hardware.txt" ] && cp "$SRC/hardware.txt" "$R/logs/hardware_extensions.txt"

for ds in nqa tqa; do
    n=$(ls "$R/data/$ds"/*.json 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" -ge 14 ] || echo "WARNING: $ds has $n of 14 result files; was 'collect' run before the jobs finished?"
done

# Combined figure: the three datasets, every setting (paper + extensions)
$PY scripts/reproduce/plot_fig9.py --results_dir $R/data --datasets longchat nqa tqa --xbreak auto \
    --out $R/figures/fig9_all_datasets.png | sed '1,2d' > $R/fig9_all_datasets_table.txt
echo "Wrote $R/figures/fig9_all_datasets.png and $R/fig9_all_datasets_table.txt"
