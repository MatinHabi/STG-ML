import pandas as pd
import torch
import numpy as np

TRAIN, VALID, TEST = 0.6, 0.2, 0.2
SEED = 42
X = pd.read_csv("markers.csv")
Y = pd.read_csv("yield.csv")
Y = Y["Adj.Grain.Yield"]

n = len(X)

shuffled_rows = np.random.default_rng(SEED).permutation(n)

train_last_row = int(TRAIN*n)
valid_last_row = train_last_row + int(VALID*n)

train_rows = shuffled_rows[:train_last_row]
valid_rows = shuffled_rows[train_last_row:valid_last_row]
test_rows = shuffled_rows[valid_last_row:]

X_train, X_valid, X_test = X.iloc[train_rows], X.iloc[valid_rows], X.iloc[test_rows]
Y_train, Y_valid, Y_test = Y.iloc[train_rows], Y.iloc[valid_rows], Y.iloc[test_rows]

mean = Y_train.mean()
std = Y_train.std()

Y_train, Y_valid, Y_test = (Y_train - mean)/std ,(Y_valid - mean)/std, (Y_test - mean)/std


data = {
    "X": torch.tensor(X.values, dtype = torch.float32),
    "Y": torch.tensor(Y.values, dtype = torch.float32),
    "X_train":torch.tensor(X_train.values, dtype = torch.float32),
    "Y_train":torch.tensor(Y_train.values, dtype = torch.float32),
    "X_valid":torch.tensor(X_valid.values, dtype = torch.float32),
    "Y_valid":torch.tensor(Y_valid.values, dtype = torch.float32),
    "X_test":torch.tensor(X_test.values, dtype = torch.float32),
    "Y_test":torch.tensor(Y_test.values, dtype = torch.float32),
    "marker_names": list(X.columns)
}

torch.save(data, "data.pt")
print(f"saved {data['X'].shape[0]} rows, {data['X'].shape[1]} markers")
