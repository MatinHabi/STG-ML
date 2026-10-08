# STG-ML

**Genomic prediction with Stochastic Gates: an alternative to VBS-ML**

STG-ML predicts a crop trait (adjusted grain yield) from a plant's DNA markers using a neural network that **learns which markers matter** while it trains. It builds on **VBS-ML**, the approach in *"Improved genomic prediction using machine learning with Variational Bayesian sparsity"* (Yan, Fruzangohar, Taylor et al.). VBS-ML gets its sparsity from a Bayesian treatment of the network's weights. STG-ML swaps that for **Stochastic Gates (STG)**: one learnable "on/off switch" per input feature.

---

## Table of contents

1. [The problem](#the-problem)
2. [The data (briefly)](#the-data-briefly)
3. [Background: why sparsity?](#background-why-sparsity)
4. [The VBS-ML approach](#the-vbs-ml-approach)
5. [The STG-ML approach](#the-stg-ml-approach)
6. [Architecture](#architecture)
7. [Loss function](#loss-function)
8. [Training, validation and testing](#training-validation-and-testing)
9. [VBS-ML vs STG-ML side by side](#vbs-ml-vs-stg-ml-side-by-side)
10. [Repository layout](#repository-layout)
11. [Running it](#running-it)
12. [References](#references)

---

## The problem

**Genomic prediction** means estimating how a plant will perform (here, its grain yield) from its genome alone, before anyone grows it in a field. Plant breeders use it to choose which lines to cross and keep.

This is a regression problem with an unusual shape:

- **Features:** thousands of genetic markers per plant.
- **Samples:** far fewer plants than markers.

When there are many more features than samples ($p \gg n$), an ordinary neural network can simply memorise the training set. Most markers also have little or no effect on yield. A good model has to **ignore the noise and focus on the few markers that matter**, and doing that well is the whole point of both VBS-ML and STG-ML.

---

## The data (briefly)

- **Inputs ($X$):** a numerically encoded genetic marker matrix. Each row is one plant and each column is one marker.
- **Target ($y$):** the adjusted grain yield measured for each plant.

The data files are **not** included in this repository (`*.csv` is in `.gitignore`).

Preprocessing:

- Rows are shuffled with a fixed seed (`SEED = 42`) and split **60% train / 20% validation / 20% test**.
- The target is **standardised (z-scored)** using only the training set's mean and standard deviation, so no information leaks from the validation or test sets:

$$
\tilde{y} = \frac{y - \mu_{\text{train}}}{\sigma_{\text{train}}}
$$

---

## Background: why sparsity?

A model is **sparse** when most of its parameters, or most of its inputs, are effectively zero. In genomic prediction, sparsity:

- reduces overfitting when $p \gg n$,
- makes the model easier to interpret, because you can see which markers it relies on,
- matches the biology, where only some regions of the genome affect a given trait.

The classic way to get sparsity is **L1 (Lasso) regularisation**, which adds $\lambda \sum |w|$ to the loss. VBS-ML and STG-ML are two more principled ways to get it inside a neural network.

---

## The VBS-ML approach

VBS-ML (**V**ariational **B**ayesian **S**parsity) uses a **Bayesian neural network**. Instead of learning one number for each weight, it learns a **probability distribution** over each weight.

### Core idea

1. **Prior:** put a *sparsity-inducing prior* $p(w)$ on the network's weights. This prior encodes the belief that most weights should be (close to) zero.
2. **Approximate posterior:** computing the true posterior $p(w \mid \mathcal{D})$ is intractable, so approximate it with a simpler distribution $q_\phi(w)$, typically a Gaussian per weight with a learnable mean and variance.
3. **Variational inference:** fit $q_\phi$ by maximising the **Evidence Lower Bound (ELBO)**:

$$
\mathcal{L}_{\text{ELBO}}(\phi) = \underbrace{\mathbb{E}_{q_\phi(w)}\big[\log p(\mathcal{D} \mid w)\big]}_{\text{fit the data}} \;-\; \underbrace{\mathrm{KL}\big(q_\phi(w)\,\|\,p(w)\big)}_{\text{stay close to the sparse prior}}
$$

In practice the network minimises the negative ELBO: a data-fit loss plus a KL-divergence penalty.

4. **Pruning:** after training, a weight whose posterior is dominated by noise (its variance is large relative to its mean) is treated as irrelevant and set to zero.

### In plain terms

Each weight has a "confidence level". The KL term pushes weights towards zero unless the data gives strong evidence that they are needed. **Sparsity happens at the level of individual weights, across the whole network.**

---

## The STG-ML approach

STG-ML keeps a standard (non-Bayesian) neural network and adds a **gating layer in front of it**. Every input feature (marker) $d$ gets its own gate $z_d \in [0, 1]$ that scales the feature before it enters the network:

$$
\hat{y} = f_\theta(x \odot z)
$$

Here $\odot$ is element-wise multiplication and $f_\theta$ is the network. A gate value of $z_d = 0$ removes marker $d$ entirely.

### The problem with real on/off switches

What we would really like is to penalise the **number of markers used**, which is the $L_0$ norm:

$$
\|z\|_0 = \sum_{d=1}^{D} \mathbb{1}[z_d \neq 0]
$$

That count is discrete and not differentiable, so gradient descent cannot optimise it directly.

### The trick: noisy, clipped Gaussian gates

STG relaxes each binary gate into a continuous random variable. Each gate has one learnable parameter $\mu_d$:

$$
z_d = \text{clamp}\big(\mu_d + \epsilon_d,\; 0,\; 1\big), \qquad \epsilon_d \sim \mathcal{N}(0, \sigma^2)
$$

- $\mu_d$ (learned, initialised to $0.5$) says how open the gate is.
- $\sigma$ (a fixed hyperparameter, $0.5$) controls how much random noise is injected.
- Clamping to $[0, 1]$ means a gate can be **exactly 0** (fully closed) or **exactly 1** (fully open).

This is the **reparameterisation trick**: the randomness lives in $\epsilon$, so gradients can still flow back to $\mu_d$.

**Why add noise?** During training the noise sometimes reopens a gate that is nearly closed, which makes the model re-test markers it may have dismissed too early. It also stops the network from compensating for a half-open gate by simply scaling up the next layer's weights.

**Training vs inference:**

| Mode | Gate value |
|---|---|
| Training (`model.train()`) | $z_d = \text{clamp}(\mu_d + \sigma\epsilon_d, 0, 1)$ (stochastic) |
| Evaluation (`model.eval()`) | $z_d = \text{clamp}(\mu_d, 0, 1)$ (deterministic, no noise) |

### Making $L_0$ differentiable

Because $z_d$ is now random, we can penalise the **expected** number of open gates. A gate is open when $\mu_d + \epsilon_d > 0$, so:

$$
\mathbb{P}(z_d > 0) = \mathbb{P}(\epsilon_d > -\mu_d) = \Phi\!\left(\frac{\mu_d}{\sigma}\right)
$$

where $\Phi$ is the standard normal CDF. Summing over all features:

$$
\mathbb{E}\big[\|z\|_0\big] = \sum_{d=1}^{D} \Phi\!\left(\frac{\mu_d}{\sigma}\right)
$$

This expression is smooth and differentiable in $\mu$. Minimising it pushes $\mu_d$ negative, closing the gate, for every marker that does not earn its place by reducing the prediction error. In the code this is `torch.special.ndtr(model.gate.mu / model.gate.sigma)`.

---

## Architecture

The model (`STGModel` in [model2.py](model2.py)) is a gated multilayer perceptron:

```
 input markers (D)
        │
        ▼
 ┌──────────────────────┐
 │  Stochastic Gates    │   z = clamp(μ + σε, 0, 1),  output = x ⊙ z
 │  (D learnable μ's)   │
 └──────────────────────┘
        │
        ▼
 Linear(D → 256) + ReLU
        │
        ▼
 Linear(256 → 128) + ReLU
        │
        ▼
 Linear(128 → 1)            →  predicted (standardised) yield
```

Mathematically:

$$
\begin{aligned}
\tilde{x} &= x \odot z \\
h_1 &= \text{ReLU}(W_1 \tilde{x} + b_1) \\
h_2 &= \text{ReLU}(W_2 h_1 + b_2) \\
\hat{y} &= W_3 h_2 + b_3
\end{aligned}
$$

Learnable parameters: the gate means $\mu \in \mathbb{R}^D$ and the network weights $\theta = \{W_1, b_1, W_2, b_2, W_3, b_3\}$. All of them are trained **jointly**, end to end, with backpropagation.

---

## Loss function

The total loss has three parts:

$$
\mathcal{L}(\theta, \mu) =
\underbrace{\frac{1}{N}\sum_{i=1}^{N} \big|\,y_i - \hat{y}_i\,\big|}_{\text{MAE (data fit)}}
\;+\;
\underbrace{\lambda_{1} \sum_{l} \|W_l\|_1}_{\text{L1 on weights}}
\;+\;
\underbrace{\lambda_{\text{STG}} \sum_{d=1}^{D} \Phi\!\left(\frac{\mu_d}{\sigma}\right)}_{\text{expected \# open gates}}
$$

| Term | What it does | Setting in `model2.py` |
|---|---|---|
| **Mean Absolute Error** | Measures prediction error. It is less sensitive to outliers than MSE. | always on |
| **L1 weight penalty** | Classic Lasso shrinkage on all linear layer weights | `L1_LAMBDA = 0` (disabled; kept for experiments) |
| **STG regulariser** | Differentiable relaxation of the $L_0$ norm over input features | `STG_LAMBDA = 0.001` |

The balance is simple. The MAE term wants to keep useful markers open, and the STG term charges a price $\lambda_{\text{STG}}$ for every gate that stays open. A marker stays open only if it reduces the error by more than it costs.

---

## Training, validation and testing

| Setting | Value |
|---|---|
| Optimiser | Adam, learning rate $10^{-3}$ |
| Batch size | 32 (shuffled mini-batches) |
| Epochs | 1000 |
| Gate noise $\sigma$ | 0.5 |
| Gate init $\mu_d$ | 0.5 (every gate starts half-open) |
| Device | CUDA if available, otherwise CPU |

- `trainModel` reports the average batch loss and the number of **open gates** ($\mu_d > 0$) at each epoch, so you can watch the model drop markers as training goes on.
- `validModel` and `testModel` run in `eval` mode (deterministic gates, no gradients) and report the loss and the final number of selected markers.

---

## VBS-ML vs STG-ML side by side

| | **VBS-ML** | **STG-ML** |
|---|---|---|
| **Framework** | Bayesian: distributions over weights | Frequentist network plus probabilistic gates on inputs |
| **What is made sparse** | Individual **weights** throughout the network | Whole **input features** (markers) |
| **Learnable parameters for sparsity** | A mean *and* a variance for **every weight** | One $\mu_d$ **per input feature** |
| **Regulariser** | $\mathrm{KL}\big(q_\phi(w)\,\|\,p(w)\big)$ against a sparsity-inducing prior | $\lambda \sum_d \Phi(\mu_d / \sigma)$, the expected $L_0$ norm |
| **Objective** | Negative ELBO (data fit + KL) | MAE + (optional L1) + STG penalty |
| **Source of randomness** | Sampling weights from $q_\phi(w)$ | Sampling gate noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ |
| **Exact zeros?** | Only after a post-hoc pruning threshold | Yes: clamping gives exact $z_d = 0$ during training |
| **Feature selection** | Indirect (a marker is "dropped" only if all its outgoing weights are pruned) | Direct: read off which gates are open |
| **Interpretability** | Uncertainty estimates per weight | A clear list of selected markers |
| **Parameter overhead** | Roughly doubles the network's parameter count | Adds only $D$ parameters |

**The key difference:** VBS-ML asks *"which connections in the network are necessary?"*, while STG-ML asks *"which markers are necessary?"*. For genomic prediction the second question is often the one breeders care about, and STG answers it directly, with a much simpler model and objective.

---

## Repository layout

| File | Description |
|---|---|
| [model2.py](model2.py) | **Main implementation**: stochastic gates, the gated MLP, the full loss (MAE + L1 + STG), data splitting, and the train/validation/test loops. |
| [model.py](model.py) | Earlier version of the gates and model (heavily commented walkthrough of the gating logic), with data loading and splitting. |
| [Regression-example.ipynb](Regression-example.ipynb) | Reference demo of the original `stg` library on a synthetic regression task, used to check expected STG behaviour. |
| [testing.ipynb](testing.ipynb) | Scratch notebook for inspecting the marker data's shape. |
| [requirements.txt](requirements.txt) | Python dependencies (PyTorch, pandas, NumPy, matplotlib, …). |

---

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# place the marker and yield CSV files in the project root, then:
python model2.py
```

Hyperparameters (`STG_LAMBDA`, `L1_LAMBDA`, `ADAM_LR`, `EPOCH`, layer sizes, split ratios) are constants at the top of [model2.py](model2.py).

---

## References

- Yan, Q., Fruzangohar, M., Taylor, J., et al. *Improved genomic prediction using machine learning with Variational Bayesian sparsity.* (VBS-ML)
- Yamada, Y., Lindenbaum, O., Negahban, S., Kluger, Y. (2020). *Feature Selection using Stochastic Gates.* ICML. (STG)
- Louizos, C., Welling, M., Kingma, D. P. (2018). *Learning Sparse Neural Networks through $L_0$ Regularization.* ICLR.
- Kingma, D. P., Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR.
