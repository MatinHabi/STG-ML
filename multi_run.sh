#!/bin/bash
lams=(1e-7 1e-6 1e-5)
lrs=(1e-3 1e-3 5e-4 5e-4 1e-4)
#python main.py --stg_lambda ${lams[$i]} --lr ${lrs[$i]} --seed 2 &
for i in "${!lams[@]}"; do
    python main.py --stg_lambda ${lams[$i]} --seed 2 &
done
wait
echo "all runs finished"