import torch
import torch.nn as nn
import pandas as pd

class StochasticGates(nn.Module):
    def __init__(self,input_dim,sigma_val=0.5):
        super().__init__() 
        self.mu = nn.Parameter(torch.full((input_dim,), 0.5))
        self.sigma = sigma_val

    # A * B = matrix mult , A @ B = dot prod
    def forward(self,input_featues):
        noise = torch.randn_like(self.mu) #creates vector of the same size as mu filled with random values from a standard N(0,1) gaussian distribution
        noise = noise * self.sigma #multiplied by sigma to stretch or narrow the distribution (this is also cuz randn_like has no std_dev input value)
        gate = noise + self.mu # adds the gaussian noise to the initial gates (mu)
        gate = torch.clamp(gate,0,1) # clipping the gate values between 0 and 1
        return gate * input_featues # returns the gate * input features 



class STGModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gates = StochasticGates(input_dim)
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2 , output_dim)


    def forward(self,x):
        gated = self.gates(x) #returns input_data * gates
        h1 = torch.relu(self.hidden1(gated)) # activation vector of neurons in layer 1 from hidden1 * gated
        h2 = torch.relu(self.hidden2(h1)) #activation vector of neurons in layer 2 from hidden1 * hidden2
        out = self.output(h2) #output which is w1*h2_1+ ... wn*h2_n + b
        return out
