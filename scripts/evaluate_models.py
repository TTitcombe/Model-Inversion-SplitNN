"""
Evaluate MNIST classifiers in terms of accuracy and
Distance Correlation (input and intermediate tensor)
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import metrics

from dpsnn import DistanceCorrelationLoss, SplitNN


def _evaluate_model_accuracy(model):
    train_accuracy = metrics.Accuracy(compute_on_step=False)
    valid_accuracy = metrics.Accuracy(compute_on_step=False)

    for x, y in model.train_dataloader():
        with torch.no_grad():
            _, y_hat = model(x)

        train_accuracy(y_hat, y)

    for x, y in model.val_dataloader():
        with torch.no_grad():
            _, y_hat = model(x)

        valid_accuracy(y_hat, y)

    total_train_accuracy = train_accuracy.compute()
    total_valid_accuracy = valid_accuracy.compute()

    return total_train_accuracy, total_valid_accuracy


def _evaluate_distance_correlation(model) -> Tuple[List, List]:
    distance_correlation = DistanceCorrelationLoss()

    dcorr_train = []

    for x, _ in model.train_dataloader():
        with torch.no_grad():
            intermediate, _ = model(x)

        dcorr_train.append(distance_correlation(x, intermediate))

    dcorr_valid = []

    for x, _ in model.val_dataloader():
        with torch.no_grad():
            intermediate, _ = model(x)

        dcorr_valid.append(distance_correlation(x, intermediate))

    return (
        np.mean(dcorr_train),
        np.std(dcorr_train) / np.sqrt(len(dcorr_train)),
        np.mean(dcorr_valid),
        np.std(dcorr_valid) / np.sqrt(len(dcorr_valid)),
    )


def _load_model(model_path: Path) -> SplitNN:
    model = SplitNN.load_from_checkpoint((str(model_path)))
    model.prepare_data()
    model.freeze()

    return model


def _evaluate_models(models_path: Path, results_path: Path) -> None:
    results = pd.DataFrame(
        columns=[
            "Model",
            "MeanTrainAcc",
            "SETrainAcc",
            "MeanValAcc",
            "SEValAcc",
            "MeanTrainDCorr",
            "SETrainDCorr",
            "MeanValDCorr",
            "SEValDCorr",
        ]
    )

    for model_path in models_path.glob("*.ckpt"):
        print(model_path.stem)
        model = _load_model(model_path)

        train_acc, val_acc = _evaluate_model_accuracy(model)
        (
            train_dcorr_mean,
            train_dcorr_se,
            val_dcorr_mean,
            val_dcorr_se,
        ) = _evaluate_distance_correlation(model)

        model_results = {
            "Model": model_path.stem,
            "MeanTrainAcc": train_acc,
            "SETrainAcc": None,
            "MeanValAcc": val_acc,
            "SEValAcc": None,
            "MeanTrainDCorr": train_dcorr_mean,
            "SETrainDcorr": train_dcorr_se,
            "MeanValDCorr": val_dcorr_mean,
            "SEValDCorr": val_dcorr_se,
        }

        results = results.append(model_results, ignore_index=True)

        break

    results.to_csv(results_path / "model_performances.csv", index=False)


if __name__ == "__main__":
    project_root = Path(__file__).parents[1]
    models_path = project_root / "models" / "classifiers"
    results_path = project_root / "results"

    _evaluate_models(models_path, results_path)
