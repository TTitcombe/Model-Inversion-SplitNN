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


class ConvAttackModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=1,
            ),
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1,
                stride=1,
                output_padding=0,
            ),
            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=8,
                kernel_size=3,
                padding=0,
                stride=1,
                output_padding=0,
            ),
            torch.nn.ConvTranspose2d(
                in_channels=8,
                out_channels=1,
                kernel_size=3,
                padding=0,
                stride=1,
                output_padding=0,
            ),
        )

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.view(-1, 64, 12, 12)

        return self.layers(x)
