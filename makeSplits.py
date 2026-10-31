# Builds data_split01.pt ... data_split04.pt using VBS-ML's exact 2014 train/test IDs.
# Splits come from: git clone https://github.com/mfruzan/VBS-ML-Genomic-Prediction
# Same output keys as data.py, so model2.py loads them unchanged (--filename data_split01.pt).
import numpy as np
import pandas as pd
import torch

SPLITS = "VBS-ML-Genomic-Prediction/2014_splits"   # folders 01-04 = paper splits 1-4
VAL_FRAC = 0.1   # share of VBS's train IDs held out to pick epoch/lambda; 0 -> valid = test (VBS's own protocol)
SEED = 42

X = pd.read_csv("markers.csv")
Y = pd.read_csv("yield.csv")["Adj.Grain.Yield"]
assert X.shape == (10375, 17114), f"{X.shape}: must match VBS's marker matrix or the split IDs / 354_idx won't line up"

X_all = torch.tensor(X.values, dtype=torch.float32)
Y_all = Y.values.astype(np.float32)

for s in ["01", "02", "03", "04"]:
    train_ids = np.loadtxt(f"{SPLITS}/{s}/trainID.txt", dtype=int)   # 0-based rows, 9338 lines
    test_ids = np.loadtxt(f"{SPLITS}/{s}/testID.txt", dtype=int)     # 0-based rows, 1037 lines
    if VAL_FRAC > 0:
        perm = np.random.default_rng(SEED).permutation(train_ids)
        n_val = int(VAL_FRAC * len(perm))
        valid_ids, train_ids = perm[:n_val], perm[n_val:]
    else:
        valid_ids = test_ids

    mean, std = Y_all[train_ids].mean(), Y_all[train_ids].std(ddof=1)
    z = lambda ids: torch.tensor((Y_all[ids] - mean) / std, dtype=torch.float32)
    rows = lambda ids: X_all[torch.as_tensor(ids)]

    torch.save({
        "X_train": rows(train_ids), "Y_train": z(train_ids),
        "X_valid": rows(valid_ids), "Y_valid": z(valid_ids),
        "X_test": rows(test_ids),   "Y_test": z(test_ids),
        "mean": torch.tensor(mean, dtype=torch.float32),
        "std": torch.tensor(std, dtype=torch.float32),
        "test_ids": torch.as_tensor(test_ids),
        "valid_is_test": VAL_FRAC == 0,
        "marker_names": list(X.columns),
    }, f"data_split{s}.pt")
    print(f"split {s}: train {len(train_ids)}, valid {len(valid_ids)}, test {len(test_ids)}")