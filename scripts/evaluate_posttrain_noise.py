"""
Evaluate performance of classifiers
with noise added after training,
not during it
"""
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import metrics

from dpsnn import DistanceCorrelationLoss, SplitNN
from dpsnn.utils import get_root_model_name, load_classifier


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
    results_file_path = results_path / "posttrain_noise_model_performances.csv"

    if not results_file_path.exists():
        results = pd.DataFrame(
            columns=[
                "DateEvaluated",
                "Model",
                "Noise",
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
    else:
        results = pd.read_csv(results_file_path)

    classifier_path = (models_path / "classifiers" / args.model).with_suffix(".ckpt")
    """classifier_root_name = get_root_model_name(classifier_path.stem)

    attacker_name = None
    for _attacker in os.listdir(models_path / "attackers"):
        if classifier_root_name in _attacker:
            attacker_name = _attacker

    if not attacker_name:
        logging.info(
            f"Attacker not found for classifier {classifier_path.stem}"
        )
        continue"""

    logging.info(f"Benchmarking {classifier_path.stem} with {args.noise} noise scale")

    model = load_classifier(classifier_path, noise=args.noise)

    train_acc, val_acc = _evaluate_model_accuracy(model)
    logging.info(
        f"{classifier_path.stem} - Train acc: {train_acc:.3f}; Val acc: {val_acc:.3f}"
    )

    """(
        train_dcorr_mean,
        train_dcorr_se,
        val_dcorr_mean,
        val_dcorr_se,
    ) = _evaluate_distance_correlation(model)
    logging.info(
        f"{model_path.stem} - Train DCorr: {train_dcorr_mean} +/- {train_dcorr_se}; Val DCorr: {val_dcorr_mean} +/- {val_dcorr_se}"
    )"""

    model_results = {
        "DateEvaluated": str(datetime.now().date()),
        "Model": classifier_path.stem,
        "Noise": args.noise,
        "MeanTrainAcc": train_acc,
        "SETrainAcc": None,
        "MeanValAcc": val_acc,
        "SEValAcc": None,
        "MeanTrainDCorr": None,  # train_dcorr_mean,
        "SETrainDCorr": None,  # train_dcorr_se,
        "MeanValDCorr": None,  # val_dcorr_mean,
        "SEValDCorr": None,  # val_dcorr_se,
    }

    results = results.append(model_results, ignore_index=True)

    results.to_csv(results_file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate classifier characteristics")
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the classifier to evaluate"
    )
    parser.add_argument(
        "--noise",
        type=float,
        required=True,
        help="Scale of noise to add to intermediate data",
    )

    args = parser.parse_args()

    assert args.noise >= 0.0

    logging.basicConfig(
        format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%I:%M:%S"
    )

    project_root = Path(__file__).parents[1]
    models_path = project_root / "models"
    results_path = project_root / "results" / "quantitative_measures"

    _evaluate_models(models_path, results_path, args)
