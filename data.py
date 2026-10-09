import pandas as pd
import torch

X = pd.read_csv("markers.csv")
Y = pd.read_csv("yield.csv")
Y = Y["Adj.Grain.Yield"]

data = {
    "X": torch.tensor(X.values, dtype = torch.float32),
    "Y": torch.tensor(Y.values, dtype = torch.float32),
    "marker_names": list(X.columns)
}

torch.save(data, "data.pt")
print(f"saved {data['X'].shape[0]} rows, {data['X'].shape[1]} markers")
