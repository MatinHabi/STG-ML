# STG results

Paper numbers are 2014, Yan et al. (2023) Plant Methods 19:96, Table 1 (RE %) and Table 2 (r). Higher r is better, lower RE is better.

## 1. Data check: ridge on all markers vs the paper's LMM

Same plants and splits as the paper. If ridge lands close to LMM, your data matches theirs.

| Split | Ridge r | Paper LMM r | Ridge RE % | Paper LMM RE % |
|---|---|---|---|---|
| 01 | 0.525 | 0.52 | 7.36 | 7.38 |
| 02 | 0.489 | 0.49 | 7.19 | 7.21 |
| 03 | 0.499 | 0.50 | 7.46 | 7.44 |
| 04 | 0.479 | 0.47 | 7.30 | 7.31 |


## 2. Settings chosen on the validation plants

best = highest validation r. sparse = highest validation r among settings ending with ≤ the sparse limit of open gates (VBS-ML kept 354). Test plants were not used.

| Split | Setting | STG λ | L1 λ | Epochs | Valid r | Valid RE % | Open gates |
|---|---|---|---|---|---|---|---|
| 01 | best | 1e-05 | 0.001 | 1000 | 0.454 | 7.60 | 17103 |
| 01 | sparse | 0.0003 | 0.001 | 1000 | 0.424 | 7.83 | 326 |
| 02 | best | 1e-05 | 0.001 | 1000 | 0.489 | 7.30 | 17101 |
| 02 | sparse | 0.0003 | 0.001 | 1000 | 0.454 | 7.53 | 321 |
| 03 | best | 1e-05 | 0.001 | 1000 | 0.462 | 7.56 | 17099 |
| 03 | sparse | 0.0003 | 0.001 | 1000 | 0.418 | 7.82 | 328 |
| 04 | best | 1e-05 | 0.001 | 1000 | 0.486 | 7.32 | 17104 |
| 04 | sparse | 0.0003 | 0.001 | 1000 | 0.441 | 7.64 | 375 |


## 3. Prediction on the test plants vs the paper

STG = final model after training on all 90% of training plants, mean ± sd over seeds. Nothing about these runs was chosen using the test plants.

**STG (best setting)**

| Split | STG r | STG RE % | STG markers | VBS-ML r | VBS-ML RE % | LMM r | LMM RE % | Seeds with lower RE than VBS-ML |
|---|---|---|---|---|---|---|---|---|
| 01 | 0.498 ± 0.001 | 7.49 ± 0.01 | 16586 | 0.66 | 6.49 | 0.52 | 7.38 | 0/5 |
| 02 | 0.494 ± 0.002 | 7.20 ± 0.02 | 16775 | 0.50 | 7.08 | 0.49 | 7.21 | 0/5 |
| 03 | 0.471 ± 0.004 | 7.66 ± 0.02 | 16571 | 0.47 | 7.58 | 0.50 | 7.44 | 0/5 |
| 04 | 0.460 ± 0.002 | 7.37 ± 0.01 | 16781 | 0.52 | 7.04 | 0.47 | 7.31 | 0/5 |
| **Average** | **0.481** | **7.43** |  | 0.54 | 7.05 | 0.49 | 7.33 |  |

Optimistic upper bound for STG (best): the lowest test RE seen during training, i.e. picked using the test plants. Only for context - the paper does not say how its reported epoch was chosen.

| Split | STG r | STG RE % | VBS-ML r | VBS-ML RE % |
|---|---|---|---|---|
| 01 | 0.498 ± 0.003 | 7.48 ± 0.01 | 0.66 | 6.49 |
| 02 | 0.494 ± 0.003 | 7.18 ± 0.01 | 0.50 | 7.08 |
| 03 | 0.472 ± 0.004 | 7.63 ± 0.02 | 0.47 | 7.58 |
| 04 | 0.463 ± 0.003 | 7.33 ± 0.01 | 0.52 | 7.04 |

**STG (sparse setting)**

| Split | STG r | STG RE % | STG markers | VBS-ML r | VBS-ML RE % | LMM r | LMM RE % | Seeds with lower RE than VBS-ML |
|---|---|---|---|---|---|---|---|---|
| 01 | 0.431 ± 0.002 | 7.93 ± 0.02 | 242 | 0.66 | 6.49 | 0.52 | 7.38 | 0/5 |
| 02 | 0.432 ± 0.003 | 7.54 ± 0.01 | 237 | 0.50 | 7.08 | 0.49 | 7.21 | 0/5 |
| 03 | 0.402 ± 0.003 | 8.03 ± 0.02 | 239 | 0.47 | 7.58 | 0.50 | 7.44 | 0/5 |
| 04 | 0.421 ± 0.002 | 7.61 ± 0.01 | 256 | 0.52 | 7.04 | 0.47 | 7.31 | 0/5 |
| **Average** | **0.421** | **7.78** |  | 0.54 | 7.05 | 0.49 | 7.33 |  |

Optimistic upper bound for STG (sparse): the lowest test RE seen during training, i.e. picked using the test plants. Only for context - the paper does not say how its reported epoch was chosen.

| Split | STG r | STG RE % | VBS-ML r | VBS-ML RE % |
|---|---|---|---|---|
| 01 | 0.458 ± 0.012 | 7.85 ± 0.02 | 0.66 | 6.49 |
| 02 | 0.458 ± 0.004 | 7.43 ± 0.03 | 0.50 | 7.08 |
| 03 | 0.436 ± 0.004 | 7.90 ± 0.02 | 0.47 | 7.58 |
| 04 | 0.435 ± 0.005 | 7.53 ± 0.02 | 0.52 | 7.04 |

Paper averages over the 4 splits: VBS-ML (354) r 0.54 / RE 7.05%, LMM r 0.49 / RE 7.33%, BayesA r 0.51 / RE 7.27%, BayesB r 0.52 / RE 7.21%, Naive-ML r 0.44 / RE 7.77%.

## 4. How good are STG's markers? Refit test

Ridge regression refitted on the training plants using only K markers, scored on the test plants. Cells are r / RE %, averaged over splits (STG: averaged over seeds first). "Random, best 5%" = 95th percentile of r (5th of RE) over random draws: STG should beat it.

**STG (best setting)**, splits 01, 02, 03, 04 - ridge on all markers: 0.498 / 7.33

| K | STG top-K | Random (median) | Random, best 5% | Top-K by correlation with yield | LASSO |
|---|---|---|---|---|---|
| 25 | 0.350 / 7.89 | 0.274 / 8.11 | 0.314 / 8.00 | 0.334 / 7.95 | 0.390 / 7.74 |
| 50 | 0.379 / 7.80 | 0.322 / 7.99 | 0.351 / 7.89 | 0.354 / 7.89 | 0.420 / 7.65 |
| 100 | 0.411 / 7.67 | 0.355 / 7.88 | 0.376 / 7.80 | 0.374 / 7.83 | 0.448 / 7.50 |
| 250 | 0.442 / 7.55 | 0.388 / 7.77 | 0.416 / 7.67 | 0.390 / 7.76 | 0.487 / 7.37 |
| 354 | 0.453 / 7.51 | 0.406 / 7.71 | 0.425 / 7.63 | 0.409 / 7.67 | 0.491 / 7.34 |
| 500 | 0.471 / 7.44 | 0.421 / 7.65 | 0.438 / 7.56 | 0.423 / 7.63 | 0.502 / 7.27 |
| 1000 | 0.485 / 7.37 | 0.444 / 7.57 | 0.464 / 7.47 | 0.434 / 7.56 | 0.479 / 7.51 |
| 2000 | 0.495 / 7.33 | 0.464 / 7.48 | 0.479 / 7.41 | 0.459 / 7.47 | 0.467 / 7.72 |

STG's own selection has 16678 markers on average - too many to refit, not scored.

**STG (sparse setting)**, splits 01, 02, 03, 04 - ridge on all markers: 0.498 / 7.33

| K | STG top-K | Random (median) | Random, best 5% | Top-K by correlation with yield | LASSO |
|---|---|---|---|---|---|
| 25 | 0.379 / 7.79 | 0.274 / 8.11 | 0.314 / 8.00 | 0.334 / 7.95 | 0.390 / 7.74 |
| 50 | 0.402 / 7.71 | 0.322 / 7.99 | 0.351 / 7.89 | 0.354 / 7.89 | 0.420 / 7.65 |
| 100 | 0.430 / 7.59 | 0.355 / 7.88 | 0.376 / 7.80 | 0.374 / 7.83 | 0.448 / 7.50 |
| 250 | 0.458 / 7.49 | 0.388 / 7.77 | 0.416 / 7.67 | 0.390 / 7.76 | 0.487 / 7.37 |
| 354 | 0.469 / 7.44 | 0.406 / 7.71 | 0.425 / 7.63 | 0.409 / 7.67 | 0.491 / 7.34 |
| 500 | 0.478 / 7.41 | 0.421 / 7.65 | 0.438 / 7.56 | 0.423 / 7.63 | 0.502 / 7.27 |
| 1000 | 0.482 / 7.39 | 0.444 / 7.57 | 0.464 / 7.47 | 0.434 / 7.56 | 0.479 / 7.51 |
| 2000 | 0.490 / 7.35 | 0.464 / 7.48 | 0.479 / 7.41 | 0.459 / 7.47 | 0.467 / 7.72 |

STG's own selection (every open gate, 244 markers on average): r 0.457 / RE 7.49%, vs random sets of the same size r 0.388 / RE 7.77%.


## 5. Stability and redundancy of STG's markers

Overlap = share of one run's top-K markers also picked by the other run. LD-aware also counts a marker correlated at |r| ≥ 0.8 as found. Within split = different seeds; across splits = same seed, different training plants.

**STG (best setting)**

| K | Within split | Within (LD-aware) | Across splits | Across (LD-aware) | Random pair | Random pair (LD-aware) |
|---|---|---|---|---|---|---|
| 50 | 0.788 | 0.864 | 0.642 | 0.738 | 0.004 | 0.020 |
| 100 | 0.814 | 0.876 | 0.663 | 0.759 | 0.014 | 0.047 |
| 354 | 0.838 | 0.913 | 0.681 | 0.799 | 0.018 | 0.125 |
| 500 | 0.845 | 0.924 | 0.694 | 0.814 | 0.026 | 0.176 |
| 1000 | 0.854 | 0.934 | 0.714 | 0.836 | 0.063 | 0.304 |
| own selection (~16678) | 0.985 | – | 0.983 | – | 0.975 | – |

Redundancy (best): share of selected markers that have a near-copy (|r| ≥ 0.8) inside the same selected set. Lower = fewer wasted slots.

| K | STG | Random | Top-K by correlation | LASSO |
|---|---|---|---|---|
| 50 | 55% (mean |r| 0.20) | 2% (mean |r| 0.07) | 87% (mean |r| 0.40) | 21% (mean |r| 0.16) |
| 100 | 55% (mean |r| 0.17) | 5% (mean |r| 0.08) | 86% (mean |r| 0.35) | 21% (mean |r| 0.10) |
| 354 | 60% (mean |r| 0.14) | 11% (mean |r| 0.07) | 84% (mean |r| 0.28) | 13% (mean |r| 0.07) |
| 500 | 62% (mean |r| 0.13) | 15% (mean |r| 0.08) | 86% (mean |r| 0.25) | 15% (mean |r| 0.07) |
| 1000 | 68% (mean |r| 0.12) | 26% (mean |r| 0.08) | 86% (mean |r| 0.22) | 15% (mean |r| 0.06) |

Most often kept markers: `results/top_markers_best.csv`

**STG (sparse setting)**

| K | Within split | Within (LD-aware) | Across splits | Across (LD-aware) | Random pair | Random pair (LD-aware) |
|---|---|---|---|---|---|---|
| 50 | 0.818 | 0.878 | 0.693 | 0.744 | 0.008 | 0.030 |
| 100 | 0.761 | 0.847 | 0.656 | 0.742 | 0.002 | 0.023 |
| 354 | 0.768 | 0.893 | 0.640 | 0.772 | 0.011 | 0.119 |
| 500 | 0.760 | 0.895 | 0.646 | 0.794 | 0.034 | 0.185 |
| 1000 | 0.747 | 0.884 | 0.652 | 0.798 | 0.057 | 0.291 |
| own selection (~244) | 0.766 | 0.873 | 0.632 | 0.742 | 0.014 | – |

Redundancy (sparse): share of selected markers that have a near-copy (|r| ≥ 0.8) inside the same selected set. Lower = fewer wasted slots.

| K | STG | Random | Top-K by correlation | LASSO |
|---|---|---|---|---|
| 50 | 42% (mean |r| 0.18) | 2% (mean |r| 0.07) | 87% (mean |r| 0.40) | 21% (mean |r| 0.16) |
| 100 | 41% (mean |r| 0.14) | 2% (mean |r| 0.07) | 86% (mean |r| 0.35) | 21% (mean |r| 0.10) |
| 354 | 51% (mean |r| 0.12) | 11% (mean |r| 0.08) | 84% (mean |r| 0.28) | 13% (mean |r| 0.07) |
| 500 | 58% (mean |r| 0.12) | 15% (mean |r| 0.08) | 86% (mean |r| 0.25) | 15% (mean |r| 0.07) |
| 1000 | 65% (mean |r| 0.11) | 25% (mean |r| 0.08) | 86% (mean |r| 0.22) | 15% (mean |r| 0.06) |

Most often kept markers: `results/top_markers_sparse.csv`


## 6. Simulation: does STG find the true markers, and does correlation between markers hurt?

Simulated yields on the real markers (heritability 0.62). high = true markers have a strongly correlated neighbour; low = they don't. Cells are exact / LD-aware.

| Condition | True markers (m) | Reps | STG open gates | STG recall | STG precision | STG top-m recall | Top-m by correlation recall | STG test r |
|---|---|---|---|---|---|---|---|---|
| high | 20 | 3 | 203 | 0.68 / 0.75 | 0.07 / 0.54 | 0.35 / 0.40 | 0.12 / 0.13 | 0.764 |
| high | 100 | 3 | 437 | 0.48 / 0.58 | 0.11 / 0.62 | 0.24 / 0.26 | 0.05 / 0.07 | 0.742 |
| low | 20 | 3 | 228 | 0.87 / 0.87 | 0.08 / 0.22 | 0.60 / 0.60 | 0.15 / 0.17 | 0.755 |
| low | 100 | 3 | 487 | 0.56 / 0.61 | 0.12 / 0.26 | 0.37 / 0.39 | 0.06 / 0.06 | 0.741 |


## Notes

- Training follows the paper's stated recipe: 256→128→1 network, dropout after the first layer, MAE loss, L1 on the network weights, Adam (lr 1e-4, betas 0.9/0.99, weight decay 5e-4), cosine learning-rate schedule, batch 512.
- Assumed (the paper doesn't say): dropout rate 0.5; weight decay is not applied to the STG gates.
- The paper contradicts itself on the network (it also mentions four fully connected layers and three residual blocks).
- The paper reports one run per split and doesn't say how its reported epoch was chosen.
- Settings were chosen separately for each split, using only that split's validation plants.
