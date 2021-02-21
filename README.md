# Defending SplitNN with Noise

Applying noise to Split Neural networks.
Code for NOT YET PUBLISHED.

## Summary

### Motivation

Data input SplitNNs have been shown to be susceptible to,
amongst other attacks,
black box model inversion.
In this attack,
an adversary trains an "inversion" model to
turn intermediate data (data sent between model parts)
back into raw input data.
This attack is a particularly relevant for a computational server
colluding with a data holder.
Applying differential privacy directly to the model
(differentially private stochastic gradient descent - the Abadi method)
does not defend against this attack
as output from a trained model part is deterministic
and therefore a decoder model can be trained.

### Aims

This project aims to protect SplitNNs
from black box model inversion attack
by adding noise to the data being transferred between model parts.
The idea is that the stochasticity of intermediate data can stop a model
from learning to invert it back into raw data.
Additionally,
we combine the noise addition with NoPeekNN,
in which the model learns to create an intermediate distribution
as uncorrelated with the input data as possible.
While NoPeekNN does not provide any guarantees on data leakage,
unlike differential privacy,
we aim to demonstrate that it can provide some protection against
a model inversion attack.

## Get started

### Requirements

Developed in Python `3.8`,
but similar minor versions should work.

#### Environment

A [conda environment](./environment.yml),
`dpsnn`,
has been provided
with all packages required to run the experiments,
including the local source code
(Pytorch-cpu only - remove `cpuonly` to enable GPU computation).
- Run `conda env create -f environment.yml` to create the environment
using the latest packages OR
- Run `conda env create -f environment-lock.yml` to use fixed package versions
(for reproducibility).
- `conda activate dpsnn` to activate the environment

### Build from source

To install the local source code only:
1. Clone this repo
1. In a terminal, navigate to the repo
1. Run `pip install -e .`.

This installs the local package `dpsnn`.

### Train models

Scripts to train a classifier and attacker can be found in [`scripts`](./scripts):

- `python scripts/train_model.py --noise_scale <noise_level> --nopeek_weight <weight>` to train a differentially private model
using noise drawn from Laplacian distribution with scale `<noise_level>` and NoPeek loss weighted by `<weight>`.

- `python scripts/train_attacker.py --model <name>` to train an attacker on a trained model,
`<name>`

Classifiers are stored in [models/classifiers](./models/classifiers/)
and are named like `mnist_<noise>noise_<nopeek>nopeek_epoch=<X>.ckpt`,
where `<noise>` is the scale of laplacian noise added to the intermediate
tensor _during training_ as a decimal. `...05noise` means scale 0.5,
`...10noise` means scale 1.0.
`<nopeek>` is the weighting of NoPeek loss
in the loss function,
using the same decimal scheme as with noise.
`<X>` is the number of training epochs
during which the classifier was performing the best.

Attack models are stored in [models/attackers](./models/attackers/)
and are named like
`mnist_attacker_model<<classifier>>_set<noise>noise.ckpt`,
where `<classifier>` is the stem
(everything but the ckpt suffix)
of the classifier it's attacking.
`<noise>` refers to the scale of noise applied
to the intermediate tensor
_after training_.
`_set<noise>noise` is not included
if the noise scale of the classifier
does not change from what it was trained on.

### Run experiments

To replicate all experiments present in the paper,
run `./main.sh <arg>`,
where `<arg>` is:

- `noise` to train models with noise
- `nopeek` to train models with NoPeek
- `combo` to train models with both NoPeek and noise
- `plain` to train a model without defences
- `performance` to calculate the accuracy and Distance Correlation of each model in `models/classifiers/`
- `all` to run all experiments

## Notebooks

We have provided relevant analysis in the [`notebooks/`](notebooks) folder.
Be aware that previous exploratory notebooks were removed.
Look over previous commits for a full history of experimentation.

## Data

The [`data/`](./data/)
folder is intentionally
left empty
to preserve the project
structure.
This project uses the
MNIST
and EMNIST
datasets.
Each dataset
will be downloaded to
`data/`
when first used
by a script.

## Contributing

If you have a question about
the paper/
experiments/
results,
or have
noticed a bug in the code,
please open an issue
in this repository.

If you are providing code,
please follow these conventions:

- `black` to format code
- `isort` to format imports
- Add type hints
- Add docstrings to functions and classes
- Use `pytorch_lightning` to build PyTorch models

## License

Apache 2.0. See the full [license](LICENSE).
