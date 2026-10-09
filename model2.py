import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class StochasticGates(nn.Module):
    def __init__(self, input_data, sigma_val = 0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.full((input_data,), 0.5)) #
        self.sigma = sigma_val

    def forward(self, input_data):
        if self.training:
            epsilon = torch.randn_like(self.mu) #
            gates = torch.clamp((self.mu + (epsilon * self.sigma)), 0, 1) 
        else:
            gates = torch.clamp(self.mu,0,1) #
        
        return input_data * gates


class STGModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, output_dim):
        super().__init__()
        self.gate = StochasticGates(input_dim)
        self.hidden1 = nn.Linear(input_dim, hidden_dim) #
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim2)
        self.output = nn.Linear(hidden_dim2, output_dim) #

    def forward(self, input_data):
        gates = self.gate(input_data)
        h1 = torch.relu(self.hidden1(gates))#
        h2 = torch.relu(self.hidden2(h1)) #
        out = self.output(h2)
        return out

d = torch.load("data.pt")
X,Y = d["X"] , d["Y"]
TRAIN, VALID, TEST = 0.6, 0.2, 0.2
L1_LAMBDA = 0
STG_LAMBDA = 0
ADAM_LR = 0.001
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

X_train, X_valid, X_test = torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(X_valid.values, dtype=torch.float32), torch.tensor(X_test.values, dtype=torch.float32)
Y_train, Y_valid, Y_test = torch.tensor(Y_train.values, dtype= torch.float32), torch.tensor(Y_valid.values, dtype= torch.float32), torch.tensor(Y_test.values, dtype= torch.float32)

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
model = STGModel(X_train.shape[1], 256, 128, 1).to(device)

#loss = MAE + L1 + regularisation
def loss_fn(pred, label):
    mae = torch.mean(torch.abs(label.unsqueeze(1) - pred)) #becuae label is a list like [1,2,3] and pred is like [[1],[2],[3]]
                                                           #with .unsqueeze(1) label = [[1],[2],[3]] SAME AS pred = [[1],[2],[3]]
    l1 = L1_LAMBDA * (
        torch.sum(torch.abs(model.hidden1.weight))+
        torch.sum(torch.abs(model.hidden2.weight))+
        torch.sum(torch.abs(model.output.weight))
    )
    reg = STG_LAMBDA * torch.sum(torch.special.ndtr((model.gate.mu / model.gate.sigma)))

    return mae + l1 + reg

optimiser = torch.optim.Adam(model.parameters(), lr = ADAM_LR)

EPOCH = 1000

def trainCycle(X_train, Y_train, device, model, epoch):
    model.train()
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    dataloader = DataLoader(TensorDataset(X_train,Y_train),32,shuffle=True)
    running_loss = 0.0
    open_gates = 0.0
    for x_t, y_t in dataloader:
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
        open_gates += (model.gate.mu > 0).sum().item()

    print(f"epoch: {epoch} || avg_batch_loss: {running_loss/len(dataloader)} || avg_open_gates: {open_gates/len(dataloader)}")


def validCycle(X_valid,Y_valid,device,model, epoch):
    model.eval()
    X_valid = X_valid.to(device)
    Y_valid = Y_valid.to(device)
    dataloader = DataLoader(TensorDataset(X_valid, Y_valid), 32, shuffle=True)
    running_loss = 0.0
    open_gates = 0.0
    with torch.no_grad():
        for x_t, y_t in dataloader:
            pred = model(x_t)
            loss = loss_fn(pred,y_t)
            running_loss += loss.item()
            open_gates = (model.gate.mu > 0).sum().item()

    print(f"epoch: {epoch} || loss: {running_loss/len(dataloader)} || open_gates: {open_gates}")



def testCycle(X_test, Y_test, model, device, epoch):
    model.eval()
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    dataloader = DataLoader(TensorDataset(X_test, Y_test), 32, shuffle = True)
    running_loss = 0.0
    open_gates = 0.0
    with torch.no_grad() :
        for x_t , y_t in dataloader:
            pred = model(x_t)
            loss = loss_fn(pred, y_t)
            running_loss += loss.item()
            open_gates = (model.gate.mu > 0).sum().item()

    print(f"epoch: {epoch} || average loss per batch: {running_loss/len(dataloader)} || open_gates: {open_gates}")


