import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path

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
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim, sigma_val, dropout_p):
        super().__init__()
        self.gates = StochasticGates(input_dim, sigma_val)
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.dropout = nn.Dropout(dropout_p) #paper (Fig. 1): one dropout layer after the first hidden layer
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output = nn.Linear(hidden2_dim, output_dim)

    def forward(self, input_data):
        gates = self.gates(input_data)
        h1 = torch.relu(self.dropout(self.hidden1(gates)))
        h2 = torch.relu(self.hidden2(h1))
        out = self.output(h2)
        return out

parser = argparse.ArgumentParser() #parser object
#defaults follow the VBS-ML paper (Yan et al. 2023, "Computations and Benchmarking") wherever the paper gives a value
parser.add_argument("--filename", type=str, default="data_split01.pt")
parser.add_argument("--l1_lambda", type= float, default=1e-3) #paper: lambda_theta = 1e-3 (L1 on the network weights)
parser.add_argument("--stg_lambda", type= float, default = 1e-4)
parser.add_argument("--lr", type=float, default= 1e-4) #paper: 1e-4
parser.add_argument("--sigma", type=float, default=0.5)
parser.add_argument("--seed", type=int, default = 42)
parser.add_argument("--batch_size", type=int, default=512) #paper: 512
parser.add_argument("--weight_decay", type=float, default=5e-4) #paper: 5e-4 (not applied to the gates - see the optimiser below)
parser.add_argument("--dropout", type=float, default=0.5) #the paper uses dropout but never says the rate; 0.5 is PyTorch's default
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--eval_every", type=int, default=25)
parser.add_argument("--mode", choices=["select", "fixed"], default="select") #select = tuning run (only looks at validation plants)
                                                                              #fixed  = final run (trains for --epochs, scores the final model on test once)
parser.add_argument("--out_dir", type=str, default="results")
parser.add_argument("--overwrite", action="store_true") #re-run even if this run's results file already exists
args = parser.parse_args() #reads the command line

FILE = args.filename
L1_LAMBDA = args.l1_lambda
STG_LAMBDA = args.stg_lambda
ADAM_LR = args.lr
SIGMA = args.sigma
RUN_SEED = args.seed
BATCH_SIZE = args.batch_size
WEIGHT_DECAY = args.weight_decay
DROPOUT = args.dropout
EPOCHS = args.epochs
EVAL_EVERY = args.eval_every
MODE = args.mode
CONFIG = vars(args)

#every run writes its numbers to <out_dir>/runs/<RUN_NAME>.json and its learnt mu to <out_dir>/mu/
#note: the name only holds the settings run_all.sh changes. If you change lr/dropout/batch size by hand, use a different --out_dir
RUN_NAME = f"{Path(FILE).stem}_{MODE}_stg{STG_LAMBDA:g}_l1{L1_LAMBDA:g}_ep{EPOCHS}_seed{RUN_SEED}"
RUN_JSON = Path(args.out_dir) / "runs" / f"{RUN_NAME}.json"
MU_PATH = Path(args.out_dir) / "mu" / f"mu_{RUN_NAME}.pt"

if RUN_JSON.exists() and not args.overwrite: #lets run_all.sh be stopped and restarted without repeating finished runs
    print(f"{RUN_NAME}: already finished ({RUN_JSON}), skipping")
    sys.exit(0)

d = torch.load(FILE, weights_only=False)
if "base" in d: #simulated yields (make_sim.py) only store Y - the marker matrix comes from the real split file
    base = torch.load(d["base"], weights_only=False)
    for k in ("X_train", "X_valid", "X_test"):
        d[k] = base[k]
    del base

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"

#everything goes onto the device ONCE here (the old code copied all of X_train to the GPU every epoch)
X_train, X_valid, X_test = (d[k].float().to(device) for k in ("X_train", "X_valid", "X_test"))
Y_train, Y_valid, Y_test = (d[k].float().to(device) for k in ("Y_train", "Y_valid", "Y_test"))

y_mu, y_std = d['mean'].float(), d['std'].float() #kept on the CPU, only used to turn predictions back into kg/ha
del d

print(f'{RUN_NAME} || device: {device} || train {tuple(X_train.shape)} valid {tuple(X_valid.shape)} test {tuple(X_test.shape)}', flush=True)

torch.manual_seed(RUN_SEED)
model = STGModel(X_train.shape[1], 256, 128, 1, SIGMA, DROPOUT).to(device)

def loss_fn(pred,label):
    mae = torch.mean(torch.abs(label.unsqueeze(1) - pred)) #MAE on the standardised yield (paper eq. 9 uses MAE too)

    l1 = L1_LAMBDA * (
        torch.sum(torch.abs(model.hidden1.weight))+
        torch.sum(torch.abs(model.hidden2.weight))+
        torch.sum(torch.abs(model.output.weight))
    )

    stg_pen = STG_LAMBDA * torch.sum(torch.special.ndtr((model.gates.mu/model.gates.sigma))) #expected number of open gates
    return mae + l1 + stg_pen

#weight decay only goes on the network weights. If it also went on mu it would pull every gate towards 0,
#so the number of selected markers would no longer be set by STG_LAMBDA alone
net_params = [p for name, p in model.named_parameters() if not name.startswith("gates.")]
optimiser = torch.optim.Adam(
    [{"params": net_params, "weight_decay": WEIGHT_DECAY},
     {"params": [model.gates.mu], "weight_decay": 0.0}],
    lr = ADAM_LR, betas=(0.9, 0.99)) #paper: beta1 = 0.9, beta2 = 0.99
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS) #paper: cosine annealing learning rate

def training():
    model.train()
    n = X_train.shape[0]
    order = torch.randperm(n, device=device) #new random order of the plants every epoch
    running_loss = 0.0
    batches = 0
    for start in range(0, n, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        pred = model(X_train[idx])
        loss = loss_fn(pred, Y_train[idx])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        batches += 1
    scheduler.step() #one step of the cosine schedule per epoch
    return running_loss / batches


def open_gates():
    return int((model.gates.mu > 0).sum()) #a gate is open (its marker is used) when mu > 0


def evaluate(X, Y , top_k=None):
    model.eval()
    with torch.no_grad():
        mu = model.gates.mu
        saved = mu.clone() #copies gates

        if top_k is not None:
            mu[torch.argsort(saved, descending=True)[top_k:]] = 0.0 #sets all except top_k markers to 0.0
        gates = int((mu > 0).sum()) #counts gates
        pred = model(X).squeeze(1).cpu() #predicts, then moves the predictions to the CPU
        mu.copy_(saved) #restores backup & overrights value of mu itself (the trailing '_' means in-place)

    y = Y.cpu() #labels on CPU
    pred_kg, y_kg = pred * y_std + y_mu , y * y_std + y_mu
    return {
        'r' : torch.corrcoef(torch.stack([pred, y]))[0,1].nan_to_num(0.0).item(), #creates corrolation matrix to show how well predictions track the truth
        're' : (torch.abs(y_kg - pred_kg) / torch.abs(y_kg)).mean().item() * 100, #paper's relative error (%), in kg/ha
        'mse' : torch.mean((pred - y) ** 2).item(),
        'gates' : gates,
    }
