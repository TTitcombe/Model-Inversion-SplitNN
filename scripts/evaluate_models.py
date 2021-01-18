"""
Evaluate MNIST classifiers in terms of accuracy and
Distance Correlation (input and intermediate tensor)
"""
import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import metrics

from dpsnn import DistanceCorrelationLoss, SplitNN
from dpsnn.utils import load_classifier


def _evaluate_model_accuracy(model):
    train_accuracy = metrics.Accuracy(compute_on_step=False)
    valid_accuracy = metrics.Accuracy(compute_on_step=False)

    for x, y in model.train_dataloader():
        with torch.no_grad():
            y_hat, _ = model(x)

        train_accuracy(y_hat, y)

    for x, y in model.val_dataloader():
        with torch.no_grad():
            y_hat, _ = model(x)

        valid_accuracy(y_hat, y)

    total_train_accuracy = train_accuracy.compute()
    total_valid_accuracy = valid_accuracy.compute()

    return total_train_accuracy.item() * 100, total_valid_accuracy.item() * 100


def _evaluate_distance_correlation(model) -> Tuple[List, List]:
    distance_correlation = DistanceCorrelationLoss()

    dcorr_train = []

    for x, _ in model.train_dataloader():
        with torch.no_grad():
            _, intermediate = model(x)

        dcorr_train.append(distance_correlation(x, intermediate))

    dcorr_valid = []

    for x, _ in model.val_dataloader():
        with torch.no_grad():
            _, intermediate = model(x)

        dcorr_valid.append(distance_correlation(x, intermediate))

    return (
        round(np.mean(dcorr_train), 3),
        round(np.std(dcorr_train) / np.sqrt(len(dcorr_train)), 3),
        round(np.mean(dcorr_valid), 3),
        round(np.std(dcorr_valid) / np.sqrt(len(dcorr_valid)), 3),
    )


def _evaluate_models(models_path: Path, results_path: Path, args) -> None:
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

    results_file_path = results_path / "model_performances.csv"
    if results_file_path.exists():
        existing_models = pd.read_csv(results_file_path)["Model"].tolist()
    else:
        existing_models = []

    try:
        for model_path in models_path.glob("*.ckpt"):
            if not args.evaluate_all and model_path.stem in existing_models:
                logging.info(f"Skipping {model_path.stem} - Already evaluated")
                continue

            logging.info(f"Benchmarking {model_path.stem}")

            model = load_classifier(model_path)

            train_acc, val_acc = _evaluate_model_accuracy(model)
            logging.info(
                f"{model_path.stem} - Train acc: {train_acc:.3f}; Val acc: {val_acc:.3f}"
            )

            (
                train_dcorr_mean,
                train_dcorr_se,
                val_dcorr_mean,
                val_dcorr_se,
            ) = _evaluate_distance_correlation(model)
            logging.info(
                f"{model_path.stem} - Train DCorr: {train_dcorr_mean} +/- {train_dcorr_se}; Val DCorr: {val_dcorr_mean} +/- {val_dcorr_se}"
            )

            model_results = {
                "Model": model_path.stem,
                "MeanTrainAcc": train_acc,
                "SETrainAcc": None,
                "MeanValAcc": val_acc,
                "SEValAcc": None,
                "MeanTrainDCorr": train_dcorr_mean,
                "SETrainDCorr": train_dcorr_se,
                "MeanValDCorr": val_dcorr_mean,
                "SEValDCorr": val_dcorr_se,
            }

            results = results.append(model_results, ignore_index=True)
    except KeyboardInterrupt:
        pass

    results.to_csv(results_file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate classifier characteristics")
    parser.add_argument('--all', dest="evaluate_all", action='store_true', help="Provide this flag to validate all models in 'classifiers' folder. Otherwise"
    " only validate models not already in 'model_performances.csv' results file.")
    parser.set_defaults(evaluate_all=False)

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%I:%M:%S"
    )

    project_root = Path(__file__).parents[1]
    models_path = project_root / "models" / "classifiers"
    results_path = project_root / "results" / "quantitative_measures"

    _evaluate_models(models_path, results_path, args)
