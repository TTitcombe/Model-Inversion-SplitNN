import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchmetrics import Accuracy
import random
from dpsnn import DistanceCorrelationLoss
from dpsnn.utils import load_classifier

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
    accuracy = Accuracy(compute_on_step=False)
    distance_correlation = DistanceCorrelationLoss()
    dcorr = []

    for dataloader in [model.val_dataloader()]:
        for x, y in dataloader:
            with torch.no_grad():
                y_hat, intermediate = model(x)
            accuracy(y_hat, y)
            dcorr.append(distance_correlation(x, intermediate).item())

    acc = accuracy.compute().item() * 100
    dcorr_mean = round(np.mean(dcorr), 3)
    dcorr_std = round(np.std(dcorr) / np.sqrt(len(dcorr)), 3)  # Corrected to not be a tuple
    return acc, dcorr_mean, dcorr_std

def _evaluate_models(models_path: Path, results_path: Path, evaluate_all: bool) -> None:
    model_results = []

    RANGE = 10  # Number of evaluations per model to calculate standard deviation

    for model_path in models_path.glob("*.ckpt"):
        accuracies = []
        dcorrs = []
        dcorr_stds = []

        for i in range(RANGE):
            seed = 42 + i
            logging.info(f"Evaluating {model_path.stem} with seed {seed} (iteration {i+1})")
            acc, dcorr_mean, dcorr_std = evaluate_model(model_path, seed)
            accuracies.append(acc)
            dcorrs.append(dcorr_mean)
            dcorr_stds.append(dcorr_std)

        avg_acc = round(np.mean(accuracies), 3)
        std_acc = round(np.std(accuracies), 3)
        avg_dcorr = round(np.mean(dcorrs), 3)
        avg_dcorr_std = round(np.mean(dcorr_stds), 3)  # Average of standard deviations

        model_results.append((model_path.stem, avg_acc, std_acc, avg_dcorr, avg_dcorr_std))

    df_results = pd.DataFrame(model_results, columns=['Model', 'AverageAccuracy', 'StdAccuracy', 'AverageDistanceCorrelation', 'AvgStdDistanceCorrelation'])
    df_sorted = df_results.sort_values(by='Model')
    csv_file_path = results_path / "model_performances_averaged.csv"
    df_sorted.to_csv(csv_file_path, index=False)
    logging.info(f"Averaged results with standard deviations saved to {csv_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate classifier characteristics")
    parser.add_argument("--all", dest="evaluate_all", action="store_true", help="Validate all models in 'classifiers' folder.")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

    project_root = Path(__file__).resolve().parents[1]
    models_path = project_root / "models" / "classifiers"
    results_path = project_root / "results" / "quantitative_measures"
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    _evaluate_models(models_path, results_path, args.evaluate_all)
