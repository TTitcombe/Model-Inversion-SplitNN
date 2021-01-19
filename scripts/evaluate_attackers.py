"""
Evaluate attacks on MNIST classifiers in terms
of Distance Correlation
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from pytorch_lightning import metrics
from torchvision.datasets import MNIST

from dpsnn import AttackValidationSplitNN, DistanceCorrelationLoss, SplitNN
from dpsnn.utils import (get_root_model_name, load_attacker, load_classifier,
                         load_validator)


def _load_attack_validation_data(project_root):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # PyTorch examples; https://github.com/pytorch/examples/blob/master/mnist/main.py
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    val = torch.utils.data.Subset(
        MNIST(project_root / "data", download=True, train=True, transform=transform),
        range(45_000, 50_000),
    )

    return torch.utils.data.DataLoader(val, batch_size=256)


def _evaluate_attacker_accuracy(classifier, attacker, validator, validation_dataloader) -> float:
    valid_accuracy = metrics.Accuracy(compute_on_step=False)

    for x, y in validation_dataloader:
        with torch.no_grad():
            _, intermediate = classifier(x)
            reconstructed_x = attacker(intermediate)
            y_hat, _ = validator(reconstructed_x)

        valid_accuracy(y_hat, y)

    total_valid_accuracy = valid_accuracy.compute()
    return total_valid_accuracy.item() * 100


def _evaluate_distance_correlation(
    classifier, attacker, validation_dataloader: torch.utils.data.DataLoader
) -> Tuple[float, float]:
    distance_correlation = DistanceCorrelationLoss()

    dcorr_valid = []

    for x, _ in validation_dataloader:
        with torch.no_grad():
            _, intermediate = classifier(x)
            reconstructed_x = attacker(intermediate)

        dcorr_valid.append(distance_correlation(x, reconstructed_x))

    return (
        round(np.mean(dcorr_valid), 3),
        round(np.std(dcorr_valid) / np.sqrt(len(dcorr_valid)), 3),
    )


def _evaluate_attackers(
    project_root: Path, models_path: Path, results_path: Path, args
) -> None:
    results = pd.DataFrame(
        columns=[
            "Model",
            "Attacker",
            "MeanValDCorr",
            "SEValDCorr",
            "ValAccuracy"
        ]
    )

    validator = load_validator((models_path / "classifiers" / args.validation_model).with_suffix(".ckpt"))

    val_loader = _load_attack_validation_data(project_root)

    try:
        for classifier_path in (models_path / "classifiers").glob("*.ckpt"):
            model = load_classifier(classifier_path)

            classifier_root_name = get_root_model_name(classifier_path.stem)

            attacker_name = None
            for _attacker in os.listdir(models_path / "attackers"):
                if classifier_root_name in _attacker:
                    attacker_name = _attacker

            if not attacker_name:
                logging.info(
                    f"Attacker not found for classifier {classifier_path.stem}"
                )
                continue

            attacker = load_attacker(models_path / "attackers" / attacker_name)

            logging.info(f"Benchmarking {classifier_path.stem} and {attacker_name}")

            val_acc = _evaluate_attacker_accuracy(model, attacker, validator, val_loader)

            (
                val_dcorr_mean,
                val_dcorr_se,
            ) = _evaluate_distance_correlation(model, attacker, val_loader)

            model_results = {
                "Model": classifier_path.stem,
                "Attacker": attacker_name,
                "MeanValDCorr": val_dcorr_mean,
                "SEValDCorr": val_dcorr_se,
                "ValAccuracy": val_acc,
            }

            results = results.append(model_results, ignore_index=True)
    except KeyboardInterrupt:
        pass

    results.to_csv(results_path / "attack_performances.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating an attacker's reconstructions")
    parser.add_argument("--validation-model", required=True, type=str, help="Name of the classifier to evaluate attack accuracy")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%I:%M:%S"
    )

    project_root = Path(__file__).parents[1]
    models_path = project_root / "models"
    results_path = project_root / "results" / "quantitative_measures"

    _evaluate_attackers(project_root, models_path, results_path, args)
