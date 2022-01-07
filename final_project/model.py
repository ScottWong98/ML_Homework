import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, target_dim: int):
        super().__init__()
        # 1 x 30 x 32
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 6 x 26 x 28
        self.pool = nn.MaxPool2d(2, 2)
        # 6 x 13 x 14
        self.fc = nn.Linear(6 * 13 * 14, target_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
