"""
Code relating to attack model
"""
import torch


# ----- Models -----
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


# ----- Dataset -----
class AttackDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.intermediate_data = None  # Inputs
        self.actual_data = None  # Targets

    def push(self, intermediate, actual):
        assert intermediate.size(0) == actual.size(0)

        if self.intermediate_data is None:
            self.intermediate_data = intermediate
        else:
            self.intermediate_data = torch.cat([self.intermediate_data, intermediate])

        if self.actual_data is None:
            self.actual_data = actual
        else:
            self.actual_data = torch.cat([self.actual_data, actual])

    def __len__(self):
        if self.intermediate_data is None:
            return 0
        else:
            return self.intermediate_data.size(0)

    def __getitem__(self, idx):
        return self.intermediate_data[idx], self.actual_data[idx]
