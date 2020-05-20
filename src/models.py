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
            torch.nn.Linear(9216, 500),
            torch.nn.ReLU(),
        )

        self.part2 = torch.nn.Sequential(
            torch.nn.Linear(500, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
            torch.nn.Softmax(dim=1),
        )

    @property
    def noise(self):
        return self._noise.scale.item()

    @noise.setter
    def noise(self, noise_scale):
        self._noise = torch.distributions.Laplace(0.0, noise_scale)

    def forward(self, x):
        intermediate = self.part1(x)
        out = self.part2(intermediate + self._noise.sample(intermediate.size()).to(intermediate.device))
        return out, intermediate

    def encode(self, x):
        out = self.part1(x)
        out += self._noise.sample(out.size()).to(out.device)

        return out

    def decode(self, x):
        return self.part2(x)
