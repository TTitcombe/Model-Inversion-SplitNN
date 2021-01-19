"""
Code relating to attack model
"""
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST

from .models import SplitNN


# ----- Models -----
class AttackModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(500, 1_000),
            torch.nn.ReLU(),
            torch.nn.Linear(1_000, 1_000),
            torch.nn.ReLU(),
            torch.nn.Linear(1_000, 784),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class ConvAttackModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.linear1 = torch.nn.Linear(500, 9216)

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
        x = self.linear1(x)

        if len(x.size()) == 2:
            x = x.view(-1, 64, 12, 12)

        return self.layers(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx: int):
        data, targets = batch

        outputs = self(data)
        targets = targets.view(outputs.size())

        loss = ((outputs - targets) ** 2).mean()

        output = {
            "loss": loss,
        }

        return output

    def validation_step(self, batch, batch_idx: int):
        data, targets = batch

        outputs = self(data)
        targets = targets.view(outputs.size())

        loss = ((outputs - targets) ** 2).mean()

        return {
            "val_loss": loss,
        }

    def validation_epoch_end(self, outs):
        total_loss = 0.0

        for out in outs:
            total_loss += out["val_loss"]

        avg_loss = total_loss.true_divide(len(outs))

        results = {"val_loss": avg_loss}

        return {"progress_bar": results, "log": results}

    def test_step(self, batch, batch_idx):
        data, targets = batch

        outputs = self(data)
        targets = targets.view(outputs.size())

        loss = ((outputs - targets) ** 2).mean()

        return {
            "test_loss": loss,
        }

    def test_epoch_end(self, outs):
        total_loss = 0.0

        for out in outs:
            total_loss += out["test_loss"]

        avg_loss = total_loss.true_divide(len(outs))

        results = {"test_loss": avg_loss}

        return {"avg_test_loss": avg_loss, "log": results}


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


# ----- Attack validation -----
class AttackValidationSplitNN(SplitNN):
    def prepare_data(self):
        data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # PyTorch examples; https://github.com/pytorch/examples/blob/master/mnist/main.py
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        data_dir = Path.cwd() / "data"

        # Train the validation classifier on the target classifiers test
        # dataset
        self.train_data = (
            MNIST(data_dir, download=True, train=False, transform=data_transform),
        )

        self.val_data = Subset(
            MNIST(data_dir, download=True, train=True, transform=data_transform),
            range(5000),
        )
