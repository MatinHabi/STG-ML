#!/bin/bash
# Runs the whole STG experiment from start to finish and writes <results>/summary.md.
#
#   QUICK=1 bash run_all.sh     tiny version of every step (a few minutes) to check nothing crashes -> results_quick/
#   bash run_all.sh             the real thing -> results/
#   MAX_JOBS=4 bash run_all.sh  how many trainings run at the same time (default 3)
#
# Safe to stop and start again: finished trainings are skipped. Run it inside tmux (or with nohup) so it keeps
# going if the SSH connection drops.
#
# Steps
#   0  make data_split??_full.pt (all 90% training plants) if missing; data_split??.pt (81/9 train/valid) is reused
#   1  data check: ridge on all markers vs the paper's LMM
#   2  tuning: every STG setting on every split, scored on validation plants only
#   3  pick the best (and best sparse) setting per split
#   4  final runs: chosen settings x seeds, trained on all 90%, final model scored on test
#   5  marker quality: refit test, stability, redundancy
#   6  simulation with known true markers (high vs low correlation)
#   7  summary report
set -euo pipefail
cd "$(dirname "$0")"

QUICK=${QUICK:-0}
MAX_JOBS=${MAX_JOBS:-3}
SPARSE_MAX=${SPARSE_MAX:-400}   # "sparse" setting = best one ending with at most this many open gates (VBS-ML kept 354)
SPLITS=(01 02 03 04)

if [ "$QUICK" = "1" ]; then
    R=results_quick
    LAMS=(1e-4 1e-3); L1S=(1e-5); EPOCH_GRID=(4); EVAL_EVERY=2
    SEEDS=(1 2); SIM_TRUE=(20); SIM_REPS=1; EXTRA="--quick"
else
    R=results
    LAMS=(1e-5 3e-5 1e-4 3e-4 1e-3)   # STG penalty strength
    L1S=(1e-3 1e-5)                   # L1 on the network weights: 1e-3 is the paper's value, 1e-5 in case that's too strong here
    EPOCH_GRID=(1000 3000)
    EVAL_EVERY=25
    SEEDS=(1 2 3 4 5)
    SIM_TRUE=(20 100); SIM_REPS=3; EXTRA=""
fi

CORES=$(getconf _NPROCESSORS_ONLN)
THREADS=$(( CORES / MAX_JOBS )); [ "$THREADS" -ge 1 ] || THREADS=1
mkdir -p "$R/logs"
rm -f "$R/failed.txt"

banner() { echo; echo "=== $(date '+%Y-%m-%d %H:%M:%S')  $*"; }

run_jobs() {   # runs the commands given as arguments, MAX_JOBS at a time
    local cmd
    for cmd in "$@"; do
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 2; done
        ( OMP_NUM_THREADS=$THREADS bash -c "$cmd" || echo "$cmd" >> "$R/failed.txt" ) &
    done
    wait
    if [ -f "$R/failed.txt" ]; then
        echo "WARNING: $(wc -l < "$R/failed.txt") run(s) failed - commands in $R/failed.txt, output in $R/logs/"
    fi
}

python - <<'EOF'
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - training will run on the CPU and be very slow")
EOF

banner "0/7  data files"
missing=0; for s in "${SPLITS[@]}"; do [ -f "data_split$s.pt" ] || missing=1; done
if [ "$missing" = 1 ]; then python make_splits.py; fi
missing=0; for s in "${SPLITS[@]}"; do [ -f "data_split${s}_full.pt" ] || missing=1; done
if [ "$missing" = 1 ]; then python make_splits.py --val_frac 0 --suffix _full; fi

banner "1/7  data check: ridge on all markers vs the paper's LMM"
if [ ! -f "$R/data_check.json" ]; then python baselines.py --check --results "$R" $EXTRA; else echo "already done"; fi

banner "2/7  tuning: ${#SPLITS[@]} splits x ${#LAMS[@]} STG lambdas x ${#L1S[@]} L1 x ${#EPOCH_GRID[@]} epoch counts"
cmds=()
for s in "${SPLITS[@]}"; do for lam in "${LAMS[@]}"; do for l1 in "${L1S[@]}"; do for ep in "${EPOCH_GRID[@]}"; do
    cmds+=("python main.py --mode select --filename data_split$s.pt --stg_lambda $lam --l1_lambda $l1 --epochs $ep --eval_every $EVAL_EVERY --seed 1 --out_dir $R >> $R/logs/tune_split${s}_stg${lam}_l1${l1}_ep${ep}.txt 2>&1")
done; done; done; done
run_jobs "${cmds[@]}"

banner "3/7  choosing settings on the validation plants"
python select_config.py --results "$R" --sparse_max "$SPARSE_MAX"

banner "4/7  final runs: train on all 90%, score the final model on the test plants (${#SEEDS[@]} seeds each)"
cmds=()
while read -r s label lam l1 ep; do
    for seed in "${SEEDS[@]}"; do
        cmds+=("python main.py --mode fixed --filename data_split${s}_full.pt --stg_lambda $lam --l1_lambda $l1 --epochs $ep --eval_every $EVAL_EVERY --seed $seed --out_dir $R >> $R/logs/final_split${s}_${label}_seed${seed}.txt 2>&1")
    done
done < "$R/chosen_config.txt"
if [ ${#cmds[@]} -gt 0 ]; then run_jobs "${cmds[@]}"; fi

banner "5/7  marker quality: refit test, stability, redundancy"
python baselines.py --markers --results "$R" $EXTRA
python stability.py --results "$R" $EXTRA

banner "6/7  simulation with known true markers"
python make_sim.py --base data_split01.pt --out_dir sim_data --n_true "${SIM_TRUE[@]}" --reps "$SIM_REPS"
read -r lam l1 ep < "$R/sim_config.txt"
cmds=()
for cond in high low; do for m in "${SIM_TRUE[@]}"; do for rep in $(seq 1 "$SIM_REPS"); do
    f="sim_data/sim_${cond}_m${m}_rep${rep}.pt"
    cmds+=("python main.py --mode fixed --filename $f --stg_lambda $lam --l1_lambda $l1 --epochs $ep --eval_every $EVAL_EVERY --seed $rep --out_dir $R >> $R/logs/sim_${cond}_m${m}_rep${rep}.txt 2>&1")
done; done; done
run_jobs "${cmds[@]}"
python score_sim.py --results "$R"

banner "7/7  summary"
python summarise.py --results "$R" > /dev/null
echo "Done. Report: $R/summary.md"
if [ -f "$R/failed.txt" ]; then echo "Some runs failed - see $R/failed.txt"; fi
