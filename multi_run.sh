#!/bin/bash
lams=(1e-5 1e-4 1e-3)   # higher than before: all 17,114 markers need more pressure to get near 354
lrs=(1e-3 1e-3 5e-4 5e-4 1e-4)
mkdir -p logs
#python main.py --stg_lambda ${lams[$i]} --lr ${lrs[$i]} --seed 2 &
for s in 01 02 03 04; do
    for i in "${!lams[@]}"; do
        python main.py --filename "data_split${s}.pt" --stg_lambda ${lams[$i]} --seed 2 > "logs/split${s}_stg${lams[$i]}.log" 2>&1 &
    done
    wait
done
echo "all runs finished"
grep -h -A2 "^FINAL" logs/split*_stg*.log