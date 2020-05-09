"""
Code relating to attack model
"""
import torch


class AttackModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(9216, 5_000),
            torch.nn.ReLU(),
            torch.nn.Linear(5_000, 5_000),
            torch.nn.ReLU(),
            torch.nn.Linear(5_000, 1_000),
            torch.nn.ReLU(),
            torch.nn.Linear(1_000, 784),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)
