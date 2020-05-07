"""
Neural networks
"""
import torch


class SplitNN(torch.nn.Module):
    def __init__(self, noise_scale: float = 0.0) -> None:
        super().__init__()

        self.noise_scale = noise_scale

        self.part1 = torch.nn.Sequential(torch.nn.Linear(784, 500), torch.nn.ReLU(),)

        # TODO add noise

        self.part2 = torch.nn.Sequential(
            torch.nn.Linear(500, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.part2(self.part1(x))

    def encode(self, x):
        x = x.view(-1, 784)
        return self.part1(x)
