# Same downstream model (ridge) on different marker sets, same split -> isolates selection quality.
#   python baselines.py data_split01.pt mu_runs/mu_data_split01_seed2_stg0.0001_l10.0_lr0.001_sig0.5.pt
# Use split 01: 354_idx.txt is a single VBS selection, so on other splits it may have seen their test lines.
import sys
import numpy as np
import torch
from sklearn.linear_model import RidgeCV

d = torch.load(sys.argv[1])
mu = torch.load(sys.argv[2]) if len(sys.argv) > 2 else None
vbs = np.loadtxt("VBS-ML-Genomic-Prediction/2014_Data/354_idx.txt", dtype=int)   # VBS's 354 selected markers
K = len(vbs)

Xtr, ytr = d["X_train"].numpy(), d["Y_train"].numpy()
if not d.get("valid_is_test", False):   # train on all of VBS's train IDs, like their LMM did
    Xtr = np.vstack([Xtr, d["X_valid"].numpy()])
    ytr = np.concatenate([ytr, d["Y_valid"].numpy()])
Xte, yte = d["X_test"].numpy(), d["Y_test"].numpy()

def score(cols=slice(None)):
    fit = RidgeCV(alphas=np.logspace(-1, 6, 15)).fit(Xtr[:, cols], ytr)
    pred = fit.predict(Xte[:, cols])
    return f"r = {np.corrcoef(pred, yte)[0, 1]:.3f}  MSE = {np.mean((pred - yte) ** 2):.3f}"

print(f"ridge, all {Xtr.shape[1]} markers  {score()}   (paper GBLUP/LMM r: 0.52 0.49 0.50 0.47)")
print(f"ridge, VBS {K}            {score(vbs)}")
print(f"ridge, random {K}         {score(np.random.default_rng(0).choice(Xtr.shape[1], K, replace=False))}")
if mu is not None:
    stg = torch.argsort(mu, descending=True)[:K].numpy()
    print(f"ridge, STG top-{K}        {score(stg)}   overlap with VBS: {len(np.intersect1d(stg, vbs))}/{K}")
