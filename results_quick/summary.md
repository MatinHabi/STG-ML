# STG results

Paper numbers are 2014, Yan et al. (2023) Plant Methods 19:96, Table 1 (RE %) and Table 2 (r). Higher r is better, lower RE is better.

## 1. Data check: ridge on all markers vs the paper's LMM

Same plants and splits as the paper. If ridge lands close to LMM, your data matches theirs.

| Split | Ridge r | Paper LMM r | Ridge RE % | Paper LMM RE % |
|---|---|---|---|---|
| 01 | 0.525 | 0.52 | 7.36 | 7.38 |


## 2. Settings chosen on the validation plants

best = highest validation r. sparse = highest validation r among settings ending with ≤ the sparse limit of open gates (VBS-ML kept 354). Test plants were not used.

| Split | Setting | STG λ | L1 λ | Epochs | Valid r | Valid RE % | Open gates |
|---|---|---|---|---|---|---|---|
| 01 | best | 0.001 | 1e-05 | 4 | 0.388 | 7.82 | 17114 |
| 02 | best | 0.001 | 1e-05 | 4 | 0.421 | 7.54 | 17114 |
| 03 | best | 0.0001 | 1e-05 | 4 | 0.402 | 7.70 | 17114 |
| 04 | best | 0.001 | 1e-05 | 4 | 0.425 | 7.46 | 17114 |

On every split the sparse setting is the same as the best one (or no setting was sparse enough), so only 'best' is reported below.


## 3. Prediction on the test plants vs the paper

STG = final model after training on all 90% of training plants, mean ± sd over seeds. Nothing about these runs was chosen using the test plants.

**STG (best setting)**

| Split | STG r | STG RE % | STG markers | VBS-ML r | VBS-ML RE % | LMM r | LMM RE % | Seeds with lower RE than VBS-ML |
|---|---|---|---|---|---|---|---|---|
| 01 | 0.410 ± 0.002 | 7.91 ± 0.06 | 17114 | 0.66 | 6.49 | 0.52 | 7.38 | 0/2 |
| 02 | 0.412 ± 0.001 | 7.56 ± 0.08 | 17114 | 0.50 | 7.08 | 0.49 | 7.21 | 0/2 |
| 03 | 0.414 ± 0.005 | 7.89 ± 0.04 | 17114 | 0.47 | 7.58 | 0.50 | 7.44 | 0/2 |
| 04 | 0.394 ± 0.002 | 7.59 ± 0.02 | 17114 | 0.52 | 7.04 | 0.47 | 7.31 | 0/2 |
| **Average** | **0.408** | **7.74** |  | 0.54 | 7.05 | 0.49 | 7.33 |  |

Optimistic upper bound for STG (best): the lowest test RE seen during training, i.e. picked using the test plants. Only for context - the paper does not say how its reported epoch was chosen.

| Split | STG r | STG RE % | VBS-ML r | VBS-ML RE % |
|---|---|---|---|---|
| 01 | 0.407 ± 0.007 | 7.91 ± 0.05 | 0.66 | 6.49 |
| 02 | 0.412 ± 0.001 | 7.56 ± 0.08 | 0.50 | 7.08 |
| 03 | 0.414 ± 0.005 | 7.89 ± 0.04 | 0.47 | 7.58 |
| 04 | 0.394 ± 0.002 | 7.59 ± 0.02 | 0.52 | 7.04 |

Paper averages over the 4 splits: VBS-ML (354) r 0.54 / RE 7.05%, LMM r 0.49 / RE 7.33%, BayesA r 0.51 / RE 7.27%, BayesB r 0.52 / RE 7.21%, Naive-ML r 0.44 / RE 7.77%.

## 4. How good are STG's markers? Refit test

Ridge regression refitted on the training plants using only K markers, scored on the test plants. Cells are r / RE %, averaged over splits (STG: averaged over seeds first). "Random, best 5%" = 95th percentile of r (5th of RE) over random draws: STG should beat it.

**STG (best setting)**, splits 01, 02, 03, 04 - ridge on all markers: 0.525 / 7.36

| K | STG top-K | Random (median) | Random, best 5% | Top-K by correlation with yield | LASSO |
|---|---|---|---|---|---|
| 25 | 0.332 / 7.94 | 0.267 / 8.11 | 0.293 / 8.03 | 0.334 / 7.95 | – |
| 100 | 0.393 / 7.75 | 0.351 / 7.90 | 0.367 / 7.84 | 0.374 / 7.83 | – |

STG's own selection has 17114 markers on average - too many to refit, not scored.


## 5. Stability and redundancy of STG's markers

Overlap = share of one run's top-K markers also picked by the other run. LD-aware also counts a marker correlated at |r| ≥ 0.8 as found. Within split = different seeds; across splits = same seed, different training plants.

**STG (best setting)**

| K | Within split | Within (LD-aware) | Across splits | Across (LD-aware) | Random pair | Random pair (LD-aware) |
|---|---|---|---|---|---|---|
| 25 | 0.010 | 0.040 | 0.063 | 0.093 | 0.000 | 0.000 |
| 100 | 0.022 | 0.111 | 0.093 | 0.147 | 0.008 | 0.046 |
| own selection (~17114) | 1.000 | – | 1.000 | – | 1.000 | – |

Redundancy (best): share of selected markers that have a near-copy (|r| ≥ 0.8) inside the same selected set. Lower = fewer wasted slots.

| K | STG | Random | Top-K by correlation | LASSO |
|---|---|---|---|---|
| 25 | 9% (mean |r| 0.10) | 0% (mean |r| 0.07) | 96% (mean |r| 0.47) | – |
| 100 | 12% (mean |r| 0.09) | 4% (mean |r| 0.07) | 86% (mean |r| 0.35) | – |

Most often kept markers: `results_quick/top_markers_best.csv`


## 6. Simulation: does STG find the true markers, and does correlation between markers hurt?

Simulated yields on the real markers (heritability 0.62). high = true markers have a strongly correlated neighbour; low = they don't. Cells are exact / LD-aware.

| Condition | True markers (m) | Reps | STG open gates | STG recall | STG precision | STG top-m recall | Top-m by correlation recall | STG test r |
|---|---|---|---|---|---|---|---|---|
| high | 20 | 1 | 17114 | 1.00 / 1.00 | 0.00 / 0.01 | 0.15 / 0.15 | 0.10 / 0.15 | 0.610 |
| low | 20 | 1 | 17114 | 1.00 / 1.00 | 0.00 / 0.00 | 0.25 / 0.25 | 0.15 / 0.20 | 0.606 |


## Notes

- Training follows the paper's stated recipe: 256→128→1 network, dropout after the first layer, MAE loss, L1 on the network weights, Adam (lr 1e-4, betas 0.9/0.99, weight decay 5e-4), cosine learning-rate schedule, batch 512.
- Assumed (the paper doesn't say): dropout rate 0.5; weight decay is not applied to the STG gates.
- The paper contradicts itself on the network (it also mentions four fully connected layers and three residual blocks).
- The paper reports one run per split and doesn't say how its reported epoch was chosen.
- Settings were chosen separately for each split, using only that split's validation plants.
