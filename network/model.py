import torch
import torch.nn as nn

INPUT_SIZE = 81
HIDDEN1 = 256
HIDDEN2 = 128
HIDDEN3 = 64


class HeuristicPredictor(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden1=HIDDEN1, hidden2=HIDDEN2, hidden3=HIDDEN3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(0.3),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(0.3),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden3),

            nn.Linear(hidden3, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
