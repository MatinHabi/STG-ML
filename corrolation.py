import torch
import pandas as pd
import matplotlib.pyplot as plt

d = torch.load('data.pt')

X_train, X_valid, X_test = d["X_train"], d["X_valid"], d["X_test"]
Y_train, Y_valid, Y_test = d["Y_train"], d["Y_valid"], d["Y_test"]

X_train = X_train.float()

x_mu = X_train.mean(dim=0)
x_std = X_train.std(dim=0)

n = X_train.shape[0]

# drop markers with zero variance (would give NaN)
idx = torch.where(x_std > 0)[0]
Z = (X_train[:, idx] - x_mu[idx]) / x_std[idx]

# correlation matrix: every pair of columns dotted together
corr = (Z.T @ Z) / (n - 1)

# per-marker score: mean |r| with all other markers
p = corr.shape[0]
corr.fill_diagonal_(0)
score = corr.abs().sum(dim=1) / (p - 1)

# keep the least-correlated 10%
thr = torch.quantile(score, 0.25)
selected = idx[score <= thr]

X_train_sub = X_train[:, selected]
X_valid_sub = X_valid[:, selected].float()
X_test_sub = X_test[:, selected].float()

y_mu, y_std = d["mean"] , d["std"]

torch.save({
    "X_train": X_train_sub, "X_valid": X_valid_sub, "X_test": X_test_sub,
    "Y_train": Y_train,     "Y_valid": Y_valid,     "Y_test": Y_test,
    "selected": selected,   # original column indices, for tracing markers back
    "mean": y_mu,
    "std": y_std,
}, "data_low_corr2.pt")