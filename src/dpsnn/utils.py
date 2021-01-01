"""Utility code"""
from pathlib import Path
from typing import Optional

from .models import SplitNN


def load_classifier(model_path: Path, noise: Optional[float] = None) -> SplitNN:
    """
    Load a SplitNN given a filename

    parameters
    ----------
    model_path : pathlib.Path
        The path to the model to load.
        ".ckpt" suffix will be added if not present
    noise : float, optional
        If provided, set model noise to this value

    Returns
    -------
    dpsnn.SplitNN
        A trained SplitNN model
    """
    model_path = model_path.with_suffix(".ckpt")

    model = SplitNN.load_from_checkpoint((str(model_path)))
    model.prepare_data()
    model.eval()

    if noise:
        model.set_noise(noise)

    model.freeze()

    return model
