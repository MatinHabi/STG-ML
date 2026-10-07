import torch 
import torch.nn as nn
import pandas as pd
import numpy as np


class StochasticGates(nn.Module):
    def __init__(self, input_data, sigma_val = 0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.full((input_data,), 0.5))
        self.sigma = sigma_val

    def forward(self, input_data):
        if self.training:
            epsilon = torch.randn_like(input_data)
            gates = torch.clamp((self.mu + (epsilon * self.sigma)), 0, 1)
        else:
            gates = torch.clamp(self.mu,0,1)
        
        return input_data * gates


class STGModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, output_dim):
        super().__init__()
        self.gate = StochasticGates(input_dim)
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim2)
        self.output = nn.Linear(hidden_dim2, output_dim)

    def forward(self, input_data):
        gates = self.gate(input_data)
        h1 = nn.ReLU(self.hidden1(gates))
        h2 = nn.ReLU(self.hidden2(h1))
        out = self.output(h2)
        return out

X = pd.read_csv('markers.csv')
Y = pd.read_csv('yield.csv')
Y = Y["Adj.Grain.Yield"]
TRAIN, VALID, TEST = 0.6, 0.2, 0.2
L1_LAMBDA = 0
STG_LAMBDA = 0
SEED = 42

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

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
model = STGModel(X_train.shape[1], 256, 128, 1).to(device)

#loss = MAE + L1 + regularisation
def loss_fn(pred, label):
    mae = torch.mean(torch.abs(label - pred))
    
    l1 = L1_LAMBDA * (
        torch.sum(torch.abs(model.hidden1.weight))+
        torch.sum(torch.abs(model.hidden2.weight))+
        torch.sum(torch.abs(model.output.weight))
    )
    reg = STG_LAMBDA * torch.sum(torch.special.ndtr((model.gate.mu / model.gate.sigma)))

    return mae + l1 + reg