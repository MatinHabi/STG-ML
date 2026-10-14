#!/bin/bash
for lam in 1e-5 1e-4 1e-3; do
    python main.py --stg_lambda $lam --seed 2 &
done
wait
echo "all runs finished"