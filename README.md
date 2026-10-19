# STG-ML

**Genomic prediction with Stochastic Gates**

STG-ML predicts a crop trait (adjusted grain yield) from a plant's genetic markers using a small neural network with a **Stochastic Gates (STG)** layer in front of it. The gates learn which markers to keep and which to switch off while the network trains, so the model does feature selection and prediction at the same time.

The project is positioned as an alternative to **VBS-ML** (*"Improved genomic prediction using machine learning with Variational Bayesian sparsity"*, Yan, Fruzangohar, Taylor et al.). VBS-ML is **not** implemented in this repository, so there is no direct comparison in the code yet.

---

## Contents

1. [What it is](#what-it-is)
2. [How it works](#how-it-works)
3. [Data and inputs](#data-and-inputs)
4. [How to run it](#how-to-run-it)
5. [Project structure](#project-structure)
6. [Results](#results)
7. [Known issues and open questions](#known-issues-and-open-questions)
8. [References](#references)

---

## What it is

**Genomic prediction** estimates how a plant will perform from its genome alone, before it is grown in a field. Each plant is described by thousands of markers, and most of them are expected to have little or no effect on yield. A useful model has to pick out the markers that matter and ignore the rest.

STG-ML handles this by putting one learnable gate on every input marker. A gate can close completely, which drops that marker from the model. A penalty on the expected number of open gates pushes the model to use as few markers as it can while still predicting yield well.

---

## How it works

The pipeline has three steps, and each step writes a file that the next one reads:

```
markers.csv + yield.csv ──data.py──▶ data.pt ──corrolation.py──▶ data_low_corr.pt ──main.py──▶ mu_runs/mu_<run>.pt
```

1. **Split and standardise** ([data.py](data.py)): shuffle the rows, split them into train, validation and test sets, and standardise the target.
2. **Correlation filter** ([corrolation.py](corrolation.py)): keep only the 10% of markers that are least correlated with the other markers.
3. **Train the gated network** ([main.py](main.py) + [model2.py](model2.py)): train the model and save the learnt gate parameters.

### Stochastic gates

Every input marker $d$ gets a gate $z_d \in [0, 1]$ that multiplies the marker before it enters the network:

$$
\hat{y} = f_\theta(x \odot z)
$$

Each gate has one learnable parameter $\mu_d$, which starts at $0.5$. During training, Gaussian noise is added and the result is clamped to $[0, 1]$:

| Mode | Gate value (`StochasticGates.forward` in [model2.py](model2.py)) |
|---|---|
| Training (`model.train()`) | $z_d = \text{clamp}(\mu_d + \sigma\epsilon_d,\ 0,\ 1)$, with $\epsilon_d \sim \mathcal{N}(0, 1)$ |
| Evaluation (`model.eval()`) | $z_d = \text{clamp}(\mu_d,\ 0,\ 1)$ (no noise) |

- Because of the clamping, a gate can be **exactly 0** (marker removed) or **exactly 1** (marker fully passed through).
- $\sigma$ is a fixed hyperparameter (`--sigma`, default `0.5`), not a learnt one.
- The noise sits in $\epsilon$, so gradients still reach $\mu_d$ (the reparameterisation trick). The noise also lets a nearly closed gate occasionally reopen, so the model can re-test markers it was about to drop.

### The sparsity penalty

The real goal is to count how many markers are in use (the $L_0$ norm), but a count cannot be differentiated. Because a gate is open whenever $\mu_d + \sigma\epsilon_d > 0$, the **probability** that it is open is $\Phi(\mu_d / \sigma)$, where $\Phi$ is the standard normal CDF. Summing that probability over all gates gives a smooth stand-in for "number of markers used":

$$
\mathbb{E}\big[\|z\|_0\big] = \sum_{d} \Phi\!\left(\frac{\mu_d}{\sigma}\right)
$$

In the code, this is `torch.special.ndtr(model.gate.mu / model.gate.sigma)`.

### Network

`STGModel` in [model2.py](model2.py):

```
input markers (D)
  → StochasticGates (D learnable μ's)
  → Linear(D → 256) + ReLU
  → Linear(256 → 128) + ReLU
  → Linear(128 → 1)                → predicted standardised yield
```

The gate parameters $\mu$ and the network weights are trained together with the Adam optimiser.

### Loss

`loss_fn` in [model2.py](model2.py) builds the training loss from three terms:

$$
\mathcal{L} =
\underbrace{\tfrac{1}{N}\textstyle\sum_i |y_i - \hat{y}_i|}_{\text{MAE on standardised yield}}
+ \underbrace{\lambda_{1} \textstyle\sum_{l} \|W_l\|_1}_{\text{L1 on linear-layer weights}}
+ \underbrace{\lambda_{\text{STG}} \textstyle\sum_{d} \Phi(\mu_d / \sigma)}_{\text{expected open gates}}
$$

| Term | Flag | Default |
|---|---|---|
| Mean absolute error, on the standardised target | always on | – |
| L1 penalty on the weights of the three linear layers (biases and $\mu$ are not penalised) | `--l1_lambda` | `0` (off) |
| STG penalty | `--stg_lambda` | `1e-5` |

`loss_fn` also returns a second value: the same loss with the MAE calculated in **original yield units**, found by undoing the standardisation with the training mean and standard deviation. This value is only printed. It is never used for backpropagation.

### Training loop

| Setting | Value | Where |
|---|---|---|
| Optimiser | Adam, learning rate `--lr` (default `1e-3`) | `model2.py` |
| Batch size | 32, shuffled | `model2.py` |
| Epochs | 1500 | `EPOCH` in `main.py` |
| Gate noise $\sigma$ | `--sigma` (default `0.5`) | `model2.py` |
| Initial gate $\mu_d$ | 0.5 | `model2.py` |
| Random seed for weights, gate noise and batch order | `--seed` (default `42`) | `torch.manual_seed` in `model2.py` |
| Device | CUDA if available, otherwise CPU | `model2.py` |

Each epoch runs `trainCycle` and then `validCycle`. A gate counts as "open" when $\mu_d > 0$.

- `trainCycle` prints the average batch loss and the average number of open gates.
- `validCycle` runs in evaluation mode with no gradients. It prints the average standardised loss (`loss_reg`), the yield-unit loss from the **last batch only** (`loss`), and the number of open gates.

After the last epoch, `main.py` saves the learnt gate parameters $\mu$ to `mu_runs/mu_seed<seed>_stg<λ>_l1<λ1>_lr<lr>_sig<σ>.pt`. The network weights are **not** saved, and the test set is **not** evaluated (see [Known issues](#known-issues-and-open-questions)).

---

## Data and inputs

The data files are **not** included in the repository (`*.csv` and `*.pt` are listed in `.gitignore`).

[data.py](data.py) expects two CSV files in the project root:

| File | Contents |
|---|---|
| `markers.csv` | One row per plant and one column per marker, with marker names as column headers (for example `AXAFFSNP-00001`). |
| `yield.csv` | Must contain an `Adj.Grain.Yield` column. All other columns are ignored. |

From the notebook outputs in this repository:

- The marker matrix has **10,375 plants × 17,114 markers**.
- Marker values are mostly `-1`, `0` and `1`, with some non-integer values (for example `0.628741`).

**Preprocessing**

- **Split**: rows are shuffled with a fixed seed (`SEED = 42`) and split **60% train, 20% validation, 20% test**. With 10,375 plants, the training set has 6,225 rows.
- **Target**: yield is **standardised** (z-scored) using only the training set's mean and standard deviation:

$$
\tilde{y} = \frac{y - \mu_{\text{train}}}{\sigma_{\text{train}}}
$$

- **Markers**: the markers are **not** standardised before training. `model2.py` calculates an overall mean and standard deviation for the markers, but only prints them.
- **Correlation filter** ([corrolation.py](corrolation.py)): using the training set only, the filter:
  1. drops markers with zero variance,
  2. builds the full marker-by-marker Pearson correlation matrix,
  3. scores each marker by its mean absolute correlation with every other marker,
  4. keeps markers at or below the 10th percentile of that score.

  In [Untitled.ipynb](Untitled.ipynb), this filter cut 17,114 markers down to **1,708**. The indices of the kept markers are saved under `"selected"` in `data_low_corr.pt`, so they can be traced back to the original marker names (`"marker_names"` in `data.pt`).

---

## How to run it

The logs and compiled files in the repository were produced with Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # see the note on requirements below

# 1. Put markers.csv and yield.csv in the project root, then:
python data.py           # writes data.pt
python corrolation.py    # writes data_low_corr.pt

# 2. Train one model (main.py does not create mu_runs/ itself)
mkdir -p mu_runs
python main.py --stg_lambda 1e-5 --lr 1e-3 --sigma 0.5 --l1_lambda 0 --seed 42
```

| Flag | Meaning | Default |
|---|---|---|
| `--stg_lambda` | Weight of the STG (expected open gates) penalty | `1e-5` |
| `--l1_lambda` | Weight of the L1 penalty on the linear-layer weights | `0` |
| `--lr` | Adam learning rate | `1e-3` |
| `--sigma` | Gate noise standard deviation | `0.5` |
| `--seed` | Torch seed (weights, gate noise, batch order). It does **not** change the data split. | `42` |

The number of epochs (`EPOCH = 1500`) is set in [main.py](main.py), and the layer sizes (256, 128) are set in [model2.py](model2.py). Neither can be changed from the command line.

**Hyperparameter sweep:** [multi_run.sh](multi_run.sh) runs five trainings **in parallel**, all with `--seed 2`:

| `--stg_lambda` | 1e-7 | 1e-6 | 1e-5 | 1e-4 | 1e-3 |
|---|---|---|---|---|---|
| `--lr` | 1e-3 | 1e-3 | 5e-4 | 5e-4 | 1e-4 |

```bash
mkdir -p mu_runs
bash multi_run.sh
```

The script does not redirect output, so the five runs print to the same terminal and their lines mix together.

> Run training through `main.py`, not `python model2.py`. `model2.py` only defines the model and loads the data when it is imported. Running it directly does not train anything.

---

## Project structure

| Path | Description |
|---|---|
| [data.py](data.py) | Reads the CSVs, makes the 60/20/20 split, standardises the target and saves everything to `data.pt`. |
| [corrolation.py](corrolation.py) | Keeps the least-correlated 10% of markers and saves the result to `data_low_corr.pt`. |
| [model2.py](model2.py) | **Main implementation**: `StochasticGates`, `STGModel`, command-line options, the loss, and the `trainCycle` / `validCycle` / `testCycle` functions. Loads `data_low_corr.pt` when imported. |
| [main.py](main.py) | Entry point: runs 1,500 epochs of training and validation, then saves the gate parameters to `mu_runs/`. |
| [multi_run.sh](multi_run.sh) | Runs five STG-λ / learning-rate settings in parallel. |
| [model.py](model.py) | Earlier version of the gates and model, with detailed comments on the gating logic and its own CSV loading and splitting. Not used by the current pipeline. |
| [model3.py](model3.py) | Unfinished rewrite of the model. Not used. Its training-mode gate ignores $\mu$ (see [Known issues](#known-issues-and-open-questions)). |
| [Untitled.ipynb](Untitled.ipynb) | Explores the correlation filter: a histogram of the correlation scores and the filtered shape. |
| [Untitled1.ipynb](Untitled1.ipynb) | Checks the shapes in `data.pt` and plots a histogram of the standardised target. |
| [testing.ipynb](testing.ipynb) | Early look at the raw marker matrix (`genetic_marker_matrix_M.csv`). |
| [Regression-example.ipynb](Regression-example.ipynb) | Demo of the original `stg` library on a synthetic sine/cosine regression task, kept as a reference for how STG should behave. |
| [logs/](logs/) | Output from three earlier sweep runs. All three crashed on a typo in `model2.py` that has since been fixed, so they contain no results. |
| [requirements.txt](requirements.txt) | Python dependencies. |

---

## Results

> Replace the placeholders below once the runs are finished.

**Setup:** [number of markers after filtering] markers, [n train] / [n valid] / [n test] plants, [epochs] epochs, σ = [σ], seed = [seed].

| STG λ | LR | Final train loss | Validation loss (standardised) | Validation MAE (yield units) | Test MAE (yield units) | Open gates |
|---|---|---|---|---|---|---|
| 1e-7 | 1e-3 | [ ] | [ ] | [ ] | [ ] | [ ] / [D] |
| 1e-6 | 1e-3 | [ ] | [ ] | [ ] | [ ] | [ ] / [D] |
| 1e-5 | 5e-4 | [ ] | [ ] | [ ] | [ ] | [ ] / [D] |
| 1e-4 | 5e-4 | [ ] | [ ] | [ ] | [ ] | [ ] / [D] |
| 1e-3 | 1e-4 | [ ] | [ ] | [ ] | [ ] | [ ] / [D] |

**Open gates over training:** [plot of open gates against epoch for each λ]

**Selected markers:** [top markers by learnt μ for the best run, mapped back to marker names]

**Comparison with VBS-ML:** [baseline figures and the source they come from]

---

## Known issues and open questions

These came up while writing this README. None of them have been changed in the code.

1. **Test evaluation is not wired up, and would crash.** `main()` only calls `train()`. Also, `testCycle` calls `loss.item()`, but `loss_fn` returns a list, so it would raise an `AttributeError`.
2. **No model weights are saved.** Only $\mu$ is saved, so the test set cannot be scored after training finishes.
3. **`mu_runs/` must exist before training.** Otherwise `torch.save` fails at the end of a 1,500-epoch run.
4. **The validation `loss` value is from one batch.** It is the yield-unit loss of the last (shuffled) batch, not an average. Both validation numbers also include the L1 and STG penalty terms, so neither is a pure MAE.
5. **The markers are not standardised.** `x_mean` and `x_std` are calculated in `model2.py` but never applied. It is not clear whether this is intentional.
6. **What do the non-integer marker values mean?** Values such as `0.628741` appear in the matrix, possibly from imputation. This needs to be confirmed against the data source.
7. **Which raw marker file is current?** `testing.ipynb` reads `genetic_marker_matrix_M.csv`, but `data.py` reads `markers.csv`. Check that these are the same data.
8. **Rows are matched by position.** `markers.csv` and `yield.csv` are lined up by row order, with no ID-based join, so both files must be in the same plant order.
9. **`model3.py` gate bug.** In training mode it computes `clamp(epsilon, 0, 1)` without adding $\mu$, so the gates could never learn. The file is not used.
10. **`MAEPERCENT` in `model2.py` is unused.** It also returns the mean signed difference × 100, not an absolute or percentage error.
11. **`requirements.txt` is Linux/CUDA-specific.** It pins many `nvidia-*` and CUDA packages, which may not install on macOS or on machines without a GPU. It also leaves out `scipy`, `scikit-learn`, `stg` and Jupyter, which `Regression-example.ipynb` and the other notebooks need.
12. **The correlation filter uses a lot of memory.** It builds the full marker-by-marker matrix, which is about 1.2 GB in float32 for 17,114 markers.

---

## References

- Yan, Q., Fruzangohar, M., Taylor, J., et al. *Improved genomic prediction using machine learning with Variational Bayesian sparsity.* (VBS-ML)
- Yamada, Y., Lindenbaum, O., Negahban, S., Kluger, Y. (2020). *Feature Selection using Stochastic Gates.* ICML. (STG)
- Kingma, D. P., Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR.
