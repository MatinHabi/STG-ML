import torch
import torch.nn as nn
import pandas as pd

class StochasticGates(nn.Module):
    def __init__(self,input_dim):
        super().__init__() 
        self.mu = nn.Parameter(torch.full((input_dim,), 0.5))

    def forward(self,input_featues):
        return self.mu * input_featues # A * B = matrix mult , A @ B = dot prod



class STGModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gates = StochasticGates(input_dim)
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2 , output_dim)


    def forward(self,x):
        gated = self.gates(x) #returns
        h1 = torch.relu(self.hidden1(gated))
        h2 = torch.relu(self.hidden2(h1))
        out = self.output(h2)
        return out
