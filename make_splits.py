#The purpose of this file is to train and test STG on the same 2014 data splits as VBS to detemine how the results differ
#
#   python make_splits.py                              -> data_split01.pt ... (10% of VBS's training plants set aside for validation, used for tuning)
#   python make_splits.py --val_frac 0 --suffix _full  -> data_split01_full.pt ... (all 90% used for training, exactly like the paper)

import argparse
import numpy as np
import pandas as pd
import torch

SPLITS = "VBS-ML-Genomic-Prediction/2014_splits"

parser = argparse.ArgumentParser()
parser.add_argument("--val_frac", type=float, default=0.1) #the amount of VBS training data set aside for validation. note that VBS paper says that they did not have a validation set just 90/10 training/test split
                                                           #0.1 uses 10% of the training data for validation so settings can be chosen without looking at the test plants
parser.add_argument("--suffix", type=str, default="")      #added to the file names, so the two versions don't overwrite each other
args = parser.parse_args()

VAL_FRAC = args.val_frac
SEED = 42

X = pd.read_csv("markers.csv")
Y = pd.read_csv("yield.csv")["Adj.Grain.Yield"]

assert X.shape == (10375, 17114), f"{X.shape}: must match VBS's marker matrix" #asset shows message if the condition is false, nothing if true

X_all = torch.tensor(X.values, dtype=torch.float32)
Y_all = Y.values.astype(np.float32)

def z(ids):
    return torch.tensor((Y_all[ids] - mean) / std, dtype=torch.float32) #gives the standardised yields for the training/validation/test plants

def rows(ids):
    return X_all[torch.as_tensor(ids)] #pulls the entire row for specific plant ids (their 17114 marker values)

for s in ["01", "02", "03", "04"]:
    train_ids = np.loadtxt(f"{SPLITS}/{s}/trainID.txt", dtype=int)
    test_ids = np.loadtxt(f"{SPLITS}/{s}/testID.txt", dtype=int)
    if VAL_FRAC > 0:
        perm = np.random.default_rng(SEED).permutation(train_ids) #shuffling the training_ids - fixed by SEED
        n_val = int(VAL_FRAC * len(perm)) #length of validation set
        valid_ids, train_ids = perm[:n_val], perm[n_val:]
    else:
        valid_ids = test_ids #no validation set: the "validation" plants are just the test plants, so nothing may be chosen with them

    mean , std= Y_all[train_ids].mean(), Y_all[train_ids].std(ddof=1) #ddof is delta degrees of freedom. sets what the standard devaiation dvides by. on pandas its = 1, on np its = 0 so here we're fixing that.
                                                                    #std = sqrt( sum((y - mean)²) / (n - ddof) )

    out = f"data_split{s}{args.suffix}.pt"
    torch.save({
        "X_train": rows(train_ids), "Y_train": z(train_ids),
        "X_valid": rows(valid_ids), "Y_valid": z(valid_ids),
        "X_test": rows(test_ids), "Y_test": z(test_ids),
        "mean": torch.tensor(mean, dtype=torch.float32),
        "std": torch.tensor(std, dtype=torch.float32),
        "test_ids": torch.as_tensor(test_ids), # turning nupmy array to tensor
        "valid_is_test": VAL_FRAC == 0, #True = there is no separate validation set (old files called this "is_valid_set")
        "marker_names": list(X.columns),
    }, out)
    print(f"{out}: training {len(train_ids)}, valid {len(valid_ids)}, test {len(test_ids)}")
