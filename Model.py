import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, p_drop=0.2, out_dim=1):
        super().__init__()
        if len(hidden) == 2:
            h1, h3 = hidden
            h2 = max(h3, h1 // 2)
        elif len(hidden) == 3:
            h1, h2, h3 = hidden
       
        self.fc1 = nn.Linear(in_dim, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)

        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.head = nn.Linear(h3, out_dim)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.head]:
            nn.init.constant_(layer.weight, 0.1)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = self.drop(self.act(self.bn1(self.fc1(x))))
        x = self.drop(self.act(self.bn2(self.fc2(x))))
        x = self.drop(self.act(self.bn3(self.fc3(x))))
        return self.head(x).squeeze(-1)
