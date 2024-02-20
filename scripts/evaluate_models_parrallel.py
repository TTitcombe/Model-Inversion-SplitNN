import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torchmetrics as metrics
import random
from concurrent.futures import ProcessPoolExecutor
from dpsnn import DistanceCorrelationLoss, SplitNN
from dpsnn.utils import load_classifier
from collections import defaultdict


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def evaluate_model(model_path, seed):
    set_seed(seed)
    model = load_classifier(model_path)
    accuracy = metrics.Accuracy(compute_on_step=False)
    distance_correlation = DistanceCorrelationLoss()
    dcorr = []

    # Assuming model.train_dataloader() and model.val_dataloader() are defined
    for dataloader in [model.train_dataloader(), model.val_dataloader()]:
        for x, y in dataloader:
            with torch.no_grad():
                y_hat, intermediate = model(x)
            accuracy(y_hat, y)
            dcorr.append(distance_correlation(x, intermediate).item())

    acc = accuracy.compute().item() * 100
    dcorr_mean = np.mean(dcorr)
    return acc, dcorr_mean

def parallel_evaluate_model(args):
    model_path, seed, i = args
    logging.info(f"Evaluating {model_path.name} with seed {seed} (iteration {i+1})")
    acc, dcorr_mean = evaluate_model(model_path, seed)
    return acc, dcorr_mean

def _evaluate_models(models_path: Path, results_path: Path, evaluate_all: bool) -> None:
    # Use a defaultdict to store lists of results for each model
    model_results = defaultdict(list)
    RANGE = 10

    with ProcessPoolExecutor() as executor:
        futures = []
        for model_path in models_path.glob("*.ckpt"):
            for i in range(RANGE):  # Run each model evaluation with different seeds
                seed = 42 + i
                # Include model_path in the future for identification
                futures.append((model_path.name, executor.submit(parallel_evaluate_model, (model_path, seed, i))))

        for model_name, future in futures:
            acc, dcorr_mean = future.result()
            # Append the results to the list for the specific model
            model_results[model_name].append((acc, dcorr_mean))
            logging.info(f"Completed evaluation with accuracy: {acc}, distance correlation: {dcorr_mean}")

    # Process the results to calculate averages
    averaged_results = []
    for model_name, results in model_results.items():
        avg_acc = sum(result[0] for result in results) / len(results)
        avg_dcorr = sum(result[1] for result in results) / len(results)
        averaged_results.append((model_name, avg_acc, avg_dcorr))

    # Convert the averaged results to a DataFrame
    df_results = pd.DataFrame(averaged_results, columns=['Model', 'AverageAccuracy', 'AverageDistanceCorrelation'])
    df_results.to_csv(results_path / "model_performances_averaged.csv", index=False)
    logging.info("Averaged results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate classifier characteristics")
    parser.add_argument("--all", dest="evaluate_all", action="store_true",
                        help="Validate all models in 'classifiers' folder.")
    parser.set_defaults(evaluate_all=False)
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

    project_root = Path(__file__).resolve().parents[1]
    models_path = project_root / "models" / "classifiers"
    results_path = project_root / "results" / "quantitative_measures"
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    _evaluate_models(models_path, results_path, args.evaluate_all)
