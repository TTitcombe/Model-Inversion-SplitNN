"""Script for training an attacker on a trained model"""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from dpsnn import AttackDataset, AttackModel, ConvAttackModel, SplitNN


def _load_model(root: Path, model_name: str) -> SplitNN:
    """
    Load a SplitNN given a filename

    parameters
    ----------
    root : pathlib.Path
        The project root
    model_name : str
        The name of the model to load.
        Suffix ".ckpt" will be added if not present

    Returns
    -------
    dpsnn.SplitNN
        A trained SplitNN model

    Raises
    ------
    ValueError
        The model checkpoint file does not exist

    Notes
    -----
    It is assumed that the model to load
    is stored in root/models/classifiers/
    """
    model_path = (root / "models" / "classifiers" / model_name).with_suffix(".ckpt")

    if not model_path.exists():
        raise ValueError(f"{model_path} does not exist")

    model = SplitNN.load_from_checkpoint(str(model_path))
    model.eval()
    model.freeze()

    return model


def _load_attack_training_dataset(root):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # PyTorch examples; https://github.com/pytorch/examples/blob/master/mnist/main.py
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train = torch.utils.data.Subset(
        MNIST(root / "data", download=True, train=True, transform=transform),
        range(40_000, 45_000),  # first 40_000 are used to train target model
    )

    val = torch.utils.data.Subset(
        MNIST(root / "data", download=True, train=True, transform=transform),
        range(45_000, 50_000),
    )

    trainloader = torch.utils.data.DataLoader(train, batch_size=256)
    valloader = torch.utils.data.DataLoader(val, batch_size=256)

    attack_train = AttackDataset()
    attack_val = AttackDataset()

    # Train data
    for data, _ in trainloader:
        data = data

        # Get target model output
        with torch.no_grad():
            _, intermediate_data = target_model(data)

        attack_train.push(intermediate_data, data)

    # Validation data
    for data, _ in valloader:
        data = data

        # Get target model output
        with torch.no_grad():
            _, intermediate_data = target_model(data)

        attack_val.push(intermediate_data, data)

    attack_trainloader = torch.utils.data.DataLoader(attack_train, batch_size=128)
    attack_valloader = torch.utils.data.DataLoader(attack_val, batch_size=128)

    return attack_trainloader, attack_valloader


def main(root, args):
    attack_trainloader, attack_valloader = _load_attack_training_dataset(root)

    """checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=root
        / "models" / "attackers"
        / (
            f"{args.saveas}_model<{args.model}>_{epoch:02d}"
        ),
        monitor="val_accuracy",
        mode="max",
    )"""

    attack_trainer = pl.Trainer(
        default_root_dir=root / "models" / "attackers",
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        # checkpoint_callback=checkpoint_callback,
    )
    attack_trainer.fit(attack_model, attack_trainloader, attack_valloader)
    attack_trainer.test(attack_model, test_dataloaders=attack_valloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a SplitNN with differential privacy optionally applied to intermediate data"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Name of the model to attack. It is assumed that it is stored in models/classifiers",
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Batch size (default 128)"
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="Starting learning rate (default 1e-4)",
    )
    parser.add_argument(
        "--saveas",
        default="mnist_attacker",
        type=str,
        help="Name of model to save as (default is 'mnist_attacker')."
        "Note that '_<model>' will be appended to the end of the name",
    )
    parser.add_argument(
        "--overfit_pct",
        default=0.0,
        type=float,
        help="Proportion of training data to use (default 0.0 [all data])",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Number of epoch to train for (default = 10)",
    )
    parser.add_argument(
        "--gpus", default=None, help="Number of gpus to use (default None)"
    )

    args = parser.parse_args()

    # File paths
    project_root = Path(__file__).resolve().parents[1]

    # ----- Models -----
    target_model = _load_model(project_root, args.model)
    attack_model = ConvAttackModel(args)

    # ----- Train model -----
    main(project_root, args)
