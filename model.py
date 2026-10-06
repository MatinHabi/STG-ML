import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class StochasticGates(nn.Module):
    def __init__(self,input_dim,sigma_val=0.5):
        super().__init__() 
        self.mu = nn.Parameter(torch.full((input_dim,), 0.5))
        self.sigma = sigma_val

    # A * B = matrix (element-wise) mult , A @ B = dot prod
    def forward(self,input_featues):
        if self.training :
            noise = torch.randn_like(self.mu) #creates vector of the same size as mu filled with random values from a standard N(0,1) gaussian distribution
            noise = noise * self.sigma #multiplied by sigma to stretch or narrow the distribution (this is also cuz randn_like has no std_dev input value)
            gate = noise + self.mu # adds the gaussian noise to the initial gates (mu)
            gate = torch.clamp(gate,0,1) # clipping the gate values between 0 and 1
            return gate * input_featues # returns the gate * input features 
        else:
            #during training we have these learnt features, mu, for each input feature. mu has a gaussian noise which is added to it in TRAINING only.
            #The gaussian noise is multiplied by the hyperparmeter sigma and it's job to make the model re-evaluate some of the input features it may have ruled out DURING TRAINING ONLY.
            #during testing these features are learnt so there's no point adding a noise anymore its just input features * clipped gates
            gates = torch.clamp(self.mu, 0,1) #clipping the gates between 0 and 1
            return input_featues * gates


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

X = pd.read_csv('markers.csv')
Y = pd.read_csv('yield.csv')
Y = Y["Adj.Grain.Yield"] #only keep the yeild column
TRAIN, VALID, TEST = 0.6, 0.2, 0.2
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

# Sanity check : if you wanted it without shuffiling
#X_train, X_valid, X_test = X.iloc[:train_last_row], X.iloc[train_last_row : valid_last_row], X.iloc[valid_last_row:]
#Y_train, Y_valid, Y_test = Y.iloc[:train_last_row], Y.iloc[train_last_row : valid_last_row], Y.iloc[valid_last_row:]

mean = Y_train.mean()
std = Y_train.std()

Y_train, Y_valid, Y_test = (Y_train - mean)/std ,(Y_valid - mean)/std, (Y_test - mean)/std