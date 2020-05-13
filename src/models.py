"""
Neural networks
"""
import torch


class SplitNN(torch.nn.Module):
    def __init__(self, noise_scale: float = 0.0) -> None:
        super().__init__()

        self._noise = torch.distributions.Laplace(0.0, noise_scale)

        self.part1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
        )

        self.part2 = torch.nn.Sequential(
            torch.nn.Linear(9216, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
            torch.nn.Softmax(dim=1),
        )

    @property
    def noise(self):
        return self._noise.scale.item()

    def forward(self, x):
        return self.part2(self.encode(x))

    def encode(self, x):
        out = self.part1(x)
        out += self._noise.sample(out.size())

        return out
