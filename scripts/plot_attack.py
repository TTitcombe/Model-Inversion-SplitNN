"""
Plot reconstruction made
by an attacker - 1 example from each class
in the data
"""
import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST, MNIST

from dpsnn import AttackDataset, ConvAttackModel, SplitNN, plot_images
from dpsnn.utils import load_attacker, load_classifier


def main(root, args):
    target_model = load_classifier(
        (root / "models" / "classifiers" / args.model).with_suffix(".ckpt")
    )
    attack_model = load_attacker(
        (root / "models" / "attackers" / args.attacker).with_suffix(".ckpt")
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # PyTorch examples; https://github.com/pytorch/examples/blob/master/mnist/main.py
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    if args.use_emnist:
        attack_val_dataset = torch.utils.data.Subset(
            EMNIST(project_root / "data", "letters", download=True, train=True, transform=transform),
            range(45_000, 50_000),
        )
    else:
        attack_val_dataset = torch.utils.data.Subset(
            MNIST(project_root / "data", download=True, train=True, transform=transform),
            range(45_000, 50_000),
        )

    ims = []

    if args.use_emnist:
        class_labels = list(range(1, 27))
        rows = 8
    else:
        class_labels = list(range(10))
        rows = 4

    class_idx = 0
    idx = 10

    while True:
        image, im_label = attack_val_dataset[idx]
        idx += 1

        if im_label != class_labels[class_idx]:
            continue

        ims.append(image)

        with torch.no_grad():
            intermediate = target_model.encode(image.unsqueeze(0))
            reconstructed = attack_model(intermediate)

        reconstructed = reconstructed.squeeze(0)
        ims.append(reconstructed)

        class_idx += 1
        if class_idx == len(class_labels):
            break

    if args.savepath:
        savepath = (root / "results" / "figures" / args.savepath).with_suffix(".png")
    else:
        savepath = None

    plot_images(ims, rows=rows, savepath=savepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attack visualisation script")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of classifier. Assumed to be in models/classifiers/ directory.",
    )
    parser.add_argument(
        "--attacker",
        type=str,
        required=True,
        help="Name of attack model. Assumed to be in models/attackers/ directory.",
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default=None,
        help="Name to save plot as. Will be placed in results/figures/ directory.",
    )
    parser.add_argument(
        "--emnist",
        dest="use_emnist",
        action="store_true"
    )

    parser.set_defaults(use_emnist=False)
    args = parser.parse_args()

    project_root = Path(__file__).parents[1]

    main(project_root, args)
