#!/bin/bash
for lam in 1e-6; do
    python main.py --stg_lambda $lam --seed 2 &
done
wait
echo "all runs finished"