import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    def __init__(self, num_in_pol, num_out_pol, hidden_dim=64):
        super(MLPNetwork, self).__init__()
        self.fc1 = nn.Linear(num_in_pol, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Add batch normalization layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # Add batch normalization layer
        self.fc3 = nn.Linear(hidden_dim, num_out_pol)
        self.nonlin = F.relu

    def forward(self, X):
        #print("Input shape:", X.shape)
        h1 = self.fc1(X.view(X.size(0), -1))
        #print("h1 shape:", h1.shape)
        h1 = self.nonlin(h1)
        #print("h1 after nonlin shape:", h1.shape)
        h1 = self.bn1(h1)
        #print("h1 after bn1 shape:", h1.shape)

        h2 = self.fc2(h1)
        #print("h2 shape:", h2.shape)
        h2 = self.nonlin(h2)
        #print("h2 after nonlin shape:", h2.shape)
        h2 = self.bn2(h2)
        #print("h2 after bn2 shape:", h2.shape)

        out = self.fc3(h2)
        #print("Output shape:", out.shape)
        return out
