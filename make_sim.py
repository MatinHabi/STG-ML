# Builds SIMULATED yields on the REAL marker matrix, so we know exactly which markers matter.
# Using the real markers keeps the real correlation structure (LD) between neighbouring markers.
#
#   y = sum over the true markers of (marker value x effect) + noise
#   noise is scaled so the true markers explain --h2 of the variation (0.62 = the paper's 2014 heritability, Fig. 2)
#
# Two conditions, to test how much correlation between markers hurts STG's selection:
#   high  the true markers are drawn from markers that have a strong correlate elsewhere (top 25% of max |r|)
#   low   the true markers are drawn from markers with no strong correlate (bottom 25% of max |r|)
#
# Each file only stores the simulated yields (+ which markers are true). main.py takes the markers from --base.
import argparse
from pathlib import Path

import numpy as np
import torch

import results_utils as ru

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="data_split01.pt") #real split file (train / valid / test plants and their markers)
parser.add_argument("--out_dir", default="sim_data")
parser.add_argument("--n_true", type=int, nargs="+", default=[20, 100]) #how many markers really affect yield
parser.add_argument("--reps", type=int, default=3) #different random true markers + effects + noise each time
parser.add_argument("--h2", type=float, default=0.62)
parser.add_argument("--min_sd", type=float, default=0.4) #skip rare markers (almost everyone has the same value) as true markers
args = parser.parse_args()
out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

d = ru.load(args.base)
X_train = d["X_train"].float()
n_tr, n_va = X_train.shape[0], d["X_valid"].shape[0]
p = X_train.shape[1]

# for every marker, its strongest |correlation| with any OTHER marker (training plants only), cached because it takes a while
cache = out_dir / f"maxcorr_{Path(args.base).stem}.pt"
if cache.exists():
    maxcorr = torch.load(cache)
else:
    Z = ru.standardise(X_train.to(device))
    maxcorr = torch.zeros(p)
    for start in range(0, p, 2048): #a block of columns at a time, so the full 17k x 17k matrix is never held at once
        cols = torch.arange(start, min(start + 2048, p), device=device)
        C = (Z[:, cols].T @ Z).abs() / (n_tr - 1)
        C[torch.arange(len(cols), device=device), cols] = 0.0 #ignore each marker's correlation with itself
        maxcorr[start:start + len(cols)] = C.max(1).values.cpu()
    del Z, C
    torch.save(maxcorr, cache)

sd = X_train.std(0)
ok = sd >= args.min_sd
q_low, q_high = torch.quantile(maxcorr[ok], torch.tensor([0.25, 0.75])).tolist()
pools = {"high": torch.where(ok & (maxcorr >= q_high))[0].numpy(),
         "low": torch.where(ok & (maxcorr <= q_low))[0].numpy()}
print(f"candidate true markers: high pool {len(pools['high'])} (max |r| >= {q_high:.2f}), "
      f"low pool {len(pools['low'])} (max |r| <= {q_low:.2f})")

X_all = torch.cat([X_train, d["X_valid"].float(), d["X_test"].float()]).numpy()
centre = X_all[:n_tr].mean(0)

for ci, cond in enumerate(["high", "low"]):
    for m in args.n_true:
        for rep in range(1, args.reps + 1):
            out = out_dir / f"sim_{cond}_m{m}_rep{rep}.pt"
            if out.exists():
                continue
            rng = np.random.default_rng(10000 * ci + 100 * m + rep) #fixed, so the same file is made every time
            true_idx = np.sort(rng.choice(pools[cond], m, replace=False))
            beta = rng.standard_normal(m)
            g = (X_all[:, true_idx] - centre[true_idx]) @ beta #the "genetic value" of every plant
            var_e = g[:n_tr].var(ddof=1) * (1 - args.h2) / args.h2
            y = g + rng.normal(0.0, np.sqrt(var_e), size=len(g))
            z = (y - y[:n_tr].mean()) / y[:n_tr].std(ddof=1) #standardised with the training plants, like make_splits.py

            torch.save({
                "base": args.base,
                "Y_train": torch.tensor(z[:n_tr], dtype=torch.float32),
                "Y_valid": torch.tensor(z[n_tr:n_tr + n_va], dtype=torch.float32),
                "Y_test": torch.tensor(z[n_tr + n_va:], dtype=torch.float32),
                "mean": d["mean"], "std": d["std"], #puts the simulated yield on the real kg/ha scale, so RE makes sense
                "true_idx": torch.tensor(true_idx),
                "condition": cond, "n_true": m, "rep": rep, "h2": args.h2,
            }, out)
            print(f"wrote {out}")
