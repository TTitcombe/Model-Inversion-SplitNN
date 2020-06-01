"""Script for training a model with or without differential privacy"""
import argparse
from pathlib import Path

import pytorch_lightning as pl

from src import SplitNN


def main(args):
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         gpus=args.gpus)
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a SplitNN with differential privacy optionally applied to intermediate data"
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        required=True,
        help="Scale of laplacian noise from which to draw. If 0.0, no noise is added. Required.",
    )
    parser.add_argument(
        "--nopeek_weight",
        type=float,
        required=True,
        help="Weighting of nopeek loss term. Required."
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
        default="splitnn",
        type=str,
        help="Name of model to save as (default is 'splitnn')."
        "Note that '_{noisescale}noise' will be appended to the end of the name",
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
        help="Number of epoch to train for (default = 10)"
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Number of gpus to use (default None)"
    )

    args = parser.parse_args()

    # File paths
    project_root = Path(__file__).resolve().parent

    # ----- Model -----
    model = SplitNN(args)

    # ----- Train model -----
    main(args)
