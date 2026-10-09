import torch
import torch.nn as nn
import pandas as pd
import numpy as py
from torch.utils.data import DataLoader, TensorDataset

class StochasticGates(nn.Module):
    def __init__(self,input_dim, sigma_val = 0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.full((input_dim,),0.5))
        self.sigma = sigma_val

    def forward(self, input_data):
        if self.training:
            #epsilon = sigma * N(0,1)
            epsilon = torch.randn_like(input_data) * self.sigma
            gate = torch.clamp(epsilon,0,1)
            gate += input_data
        else:
            gate = torch.clamp(self.mu, 0,1)
        
        return gate

class STGModel(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super().__init__()
        self.gate = StochasticGates(input_dim)
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output = nn.Linear(hidden2_dim , output_dim)

    def forward(self, s_t):
        gates = self.gate(s_t)
        h1 = torch.relu(self.hidden1(gates))
        h2 = torch.relu(self.hidden2(h1))
        out = self.output(h2)
        
        return out


