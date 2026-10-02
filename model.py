import torch
import torch.nn as nn
import pandas as pd

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        def forward(self, input_data):
            predictions = self.network(input_data)
            return predictions
        
data = pd.read_csv('all_Melbourne.csv')

model = MLP(data.shape[1],data.shape[1]//2,1)

print(model)
