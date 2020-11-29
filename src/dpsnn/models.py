"""
Neural networks
"""
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST

from .nopeek_loss import NoPeekLoss


class SplitNN(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.hparams = hparams

        self._noise = torch.distributions.Laplace(0.0, self.hparams.noise_scale)

        self.part1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(9216, 500),
            torch.nn.Tanh(),  # tanh to bound outputs, otherwise cannot be D.P.
        )

        self.part2 = torch.nn.Sequential(
            torch.nn.Linear(500, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        intermediate = self.part1(x)
        out = self.part2(
            intermediate
            + self._noise.sample(intermediate.size()).to(intermediate.device)
        )
        return out, intermediate

    def encode(self, x):
        out = self.part1(x)
        out += self._noise.sample(out.size()).to(out.device)

        return out

    def decode(self, x):
        return self.part2(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx: int):
        data, targets = batch

        predictions, intermediates = self(data)

        loss = NoPeekLoss(self.hparams.nopeek_weight)(
            data, intermediates, predictions, targets
        )
        correct = predictions.max(1)[1].eq(targets.flatten())

        output = {
            "loss": loss,
            "progress_bar": {
                "accuracy": 100 * correct.sum().true_divide(correct.size(0)),
            },
        }

        return output

    def validation_step(self, batch, batch_idx: int):
        data, targets = batch

        predictions, intermediates = self(data)

        loss = NoPeekLoss(self.hparams.nopeek_weight)(
            data, intermediates, predictions, targets
        )
        correct = predictions.max(1)[1].eq(targets.flatten())

        return {"val_loss": loss, "val_correct": correct}

    def validation_epoch_end(self, outs):
        preds = []
        total_loss = 0.0

        for out in outs:
            preds.append(out["val_correct"])
            total_loss += out["val_loss"]

        avg_loss = total_loss.true_divide(len(outs))
        preds = torch.cat(preds)
        acc = 100 * preds.sum().true_divide(preds.size(0))

        results = {"val_loss": avg_loss, "val_accuracy": acc}

        return {"progress_bar": results, "log": results}

    def test_step(self, batch, batch_idx):
        data, targets = batch

        predictions, intermediates = self(data)

        loss = NoPeekLoss(self.hparams.nopeek_weight)(
            data, intermediates, predictions, targets
        )
        correct = predictions.max(1)[1].eq(targets.flatten())

        return {"test_loss": loss, "test_correct": correct}

    def test_epoch_end(self, outs):
        preds = []
        total_loss = 0.0

        for out in outs:
            preds.append(out["test_correct"])
            total_loss += out["test_loss"]

        avg_loss = total_loss.true_divide(len(outs))
        preds = torch.cat(preds)
        acc = 100 * preds.sum().true_divide(preds.size(0))

        results = {"test_loss": avg_loss, "test_acc": acc}

        return {"avg_test_loss": avg_loss, "log": results}

    def prepare_data(self):
        data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # PyTorch examples; https://github.com/pytorch/examples/blob/master/mnist/main.py
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        data_dir = Path.cwd() / "data"
        self.train_data = Subset(
            MNIST(data_dir, download=True, train=True, transform=data_transform),
            range(40_000),
        )

        self.val_data = Subset(
            MNIST(data_dir, download=True, train=False, transform=data_transform),
            range(5000),
        )

        # Test data
        self.test_data = Subset(
            MNIST(data_dir, download=True, train=False, transform=data_transform),
            range(5000, 10_000),
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size)


class ReLUSplitNN(SplitNN):
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
