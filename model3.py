import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse

class StochasticGates(nn.Module):
    def __init__(self,input_dim, sigma_val):
        super().__init__()
        self.mu = nn.Parameter(torch.full((input_dim,),0.5))
        self.sigma = sigma_val

    def forward(self,input_data):
        if self.training :
            epsilon = torch.randn_like(self.mu) #take gaussian noise of the same shape as mu
            gate = torch.clamp(self.mu + (epsilon * self.sigma), 0 , 1) # clamp the gates between 0 & 1
        else:
            gate = torch.clamp(self.mu, 0 , 1) #use the learnt embeddings (which means no gaussian noise)

        return input_data * gate


class STGModel(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim, sigma_val):
        super().__init__()
        self.gates = StochasticGates(input_dim, sigma_val)
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output = nn.Linear(hidden2_dim, output_dim)

    def forward(self, input_data):
        gates = self.gates(input_data)
        h1 = torch.relu(self.hidden1(gates))
        h2 = torch.relu(self.hidden2(h1))
        out = self.output(h2)
        return out 

parser = argparse.ArgumentParser() #parser object
parser.add_argument("--filename", type=str, default="data.pt")
parser.add_argument("--l1_lambda", type= float, default=0)
parser.add_argument("--stg_lambda", type= float, default = 0)
parser.add_argument("--lr", type=float, default= 0.001)
parser.add_argument("--sigma", type=float, default=0.5)
parser.add_argument("--seed", type=int, default = 42)
args = parser.parse_args() #reads the command line

FILE = args.filename
L1_LAMBDA = args.l1_lambda
STG_LAMBDA = args.stg_lambda
ADAM_LR = args.lr
SIGMA = args.sigma
SEED = args.seed

print(f'stg_lambda: {STG_LAMBDA}')

d = torch.load(FILE)

X_train, X_valid, X_test = d['X_train'], d['X_valid'], d['X_test']
Y_train, Y_valid, Y_test = d['Y_train'], d['Y_valid'], d['Y_test']

y_mu, y_std = d['mean'], d['std']

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
torch.manual_seed(SEED)
model = STGModel(X_train.shape[1], 256, 128, 1, SIGMA).to(device)

def loss_fn(pred,label):
    label_unreg = label * y_std + y_mu
    pred_unreg = pred * y_std + y_mu
    mae_unreg = torch.mean(torch.abs(label_unreg.unqueeze(1) - pred_unreg))
    mae = torch.mean(torch.abs(label.unsqueeze(1) - pred))

    l1 = L1_LAMBDA * (
        torch.sum(torch.abs(model.hidden1.weight))+
        torch.sum(torch.abs(model.hidden2.weight))+
        torch.sum(torch.abs(model.output.weight))
    )

    stg_pen = STG_LAMBDA * torch.sum(torch.special.ndtr((model.gate.mu/model.gate.sigma)))
    return [mae + l1 + stg_pen, mae_unreg + l1 + stg_pen] # [reg_loss , unreg_loss]

optimiser = torch.optim.Adam(model.parameters(), lr = ADAM_LR)

def MAEPERCENT(pred, label):
    return torch.mean(label - pred) * 100 

def training(epoch):
    X = X_train.to(device)
    Y = Y_train.to(device)
    dataloader = DataLoader(TensorDataset(X , Y), 32, shuffle = True).to(device) #Matin: remember this line
    for e in range(epoch):
        running_loss = 0.0
        open_gates = 0.0
        for x,y in dataloader:
            pred = model(x)
            loss = loss_fn(pred,y)
            optimiser.zero_grad()
            loss[1].backward()
            optimiser.step

            running_loss += loss[0].item()
            open_gates += (model.gate.mu > 0).sum().item()

        print(f'TRAINING -- STG_LAMBDA: {STG_LAMBDA} || lr: {ADAM_LR} , || epoch: {e} || avg_batch_loss: {running_loss/len(dataloader)} || avg_open_gates: {open_gates/len(dataloader)}')


def validation(epoch):
    model.eval()
    X = X_valid.to(device)
    Y = Y_valid.to(device)
    dataloader = DataLoader(TensorDataset(X,Y),32,shuffle=True).to(device)
    running_loss = 0.0
    open_gates = 0.0

    with torch.no_grad:
        for x,y in dataloader:
            pred = model(x)
            loss = loss_fn(pred,y)
            running_loss += loss[0].item()
            open_gates += (model.gates.mu > 0 ).sum().item()

        print(f'EVAL -- STG_LAMBDA: {STG_LAMBDA} || lr: {ADAM_LR} , || epoch: {epoch} || avg_batch_loss: {running_loss/len(dataloader)} || avg_open_gates: {open_gates/len(dataloader)}')

def testing(epoch):
    X = X_test.to(device)
    Y = Y_test.to(device)
    dataloader = DataLoader(TensorDataset(X,Y),32,shuffle=True).to(device)
    running_loss = 0.0
    open_gates = 0.0
    
    with torch.no_grad:
        for x,y in dataloader:
            pred = model(x)
            loss = loss_fn(pred,y)
            running_loss += loss[0].item()
            open_gates += (model.gates.mu > 0 ).sum().item()

        print(f'TEST -- STG_LAMBDA: {STG_LAMBDA} || lr: {ADAM_LR} , || epoch: {epoch} || avg_batch_loss: {running_loss/len(dataloader)} || avg_open_gates: {open_gates/len(dataloader)}')

