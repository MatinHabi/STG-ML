import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class StochasticGates(nn.Module):
    def __init__(self, input_data, sigma_val = 0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.full((input_data,), 0.5)) #
        self.sigma = sigma_val

    def forward(self, input_data):
        if self.training:
            epsilon = torch.randn_like(self.mu) # Take gaussian noise
            gates = torch.clamp((self.mu + (epsilon * self.sigma)), 0, 1) 
        else:
            gates = torch.clamp(self.mu,0,1) #clip between 0 & 1
        
        return input_data * gates


class STGModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, output_dim, sigma_val):
        super().__init__()
        self.gate = StochasticGates(input_dim, sigma_val)
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
L1_LAMBDA = 0
STG_LAMBDA = 1e-5
ADAM_LR = 0.001
SIGMA = 0.5
RUN_SEED = 2
SEED = 42

X_train, X_valid, X_test = d["X_train"], d["X_valid"], d["X_test"]
Y_train, Y_valid, Y_test = d["Y_train"], d["Y_valid"], d["Y_test"]

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
torch.manual_seed(RUN_SEED)
model = STGModel(X_train.shape[1], 256, 128, 1, SIGMA).to(device)

#loss = MAE + L1 + regularisation
def loss_fn(pred, label):
    mae = torch.mean(torch.abs(label.unsqueeze(1) - pred)) #becuae label is a list like [1,2,3] and pred is like [[1],[2],[3]]
                                                           #with .unsqueeze(1) label = [[1],[2],[3]] SAME AS pred = [[1],[2],[3]]
    l1 = L1_LAMBDA * (
        torch.sum(torch.abs(model.hidden1.weight))+
        torch.sum(torch.abs(model.hidden2.weight))+
        torch.sum(torch.abs(model.output.weight))
    )
    reg = STG_LAMBDA * torch.sum(torch.special.ndtr((model.gate.mu/model.gate.sigma)))
    #print(torch.special.ndtr((model.gate.mu / model.gate.sigma)))
    #print(torch.sum(torch.special.ndtr((model.gate.mu / model.gate.sigma))).item())
    return mae + l1 + reg

optimiser = torch.optim.Adam(model.parameters(), lr = ADAM_LR)

def MAEPERCENT(pred,actual):
    return torch.mean(actual-pred)*100

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



def testCycle(X_test, Y_test, model, device):
    model.eval()
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    dataloader = DataLoader(TensorDataset(X_test, Y_test), 32, shuffle = True)
    #print(len(dataloader))
    running_loss = 0.0
    open_gates = 0.0
    with torch.no_grad() :
        for x_t , y_t in dataloader:
            pred = model(x_t)
            loss = loss_fn(pred, y_t)
            running_loss += loss.item()
            open_gates = (model.gate.mu > 0).sum().item()

    print(f"average loss per batch: {running_loss/len(dataloader)} || open_gates: {open_gates}")


