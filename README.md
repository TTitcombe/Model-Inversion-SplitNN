# Differentially-Private-SplitNN

Applying differential privacy to Split Neural networks (DPSNN).

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

### Roadmap

- [x] Proof of concept: show that the attack can be protected against
    - Eyeball data being protected (i.e. no formal evaluation of privacy)
    - Brief analysis of privacy/accuracy trade-off
- [ ] Improve models
    - Current models have large accuracy/privacy trade-off. Tune NoPeek and DP-SplitNN to improve performance
- [ ] Explore limits
    - Impact on model size / architecture
    - Can it work with datasets more complex than MNIST
    - How many data points does the attacker need
- [x] Integrate NoPeekNN
    - NoPeekNN cannot defend against the attack on its own,
    but it has an effect similar to introducing more layers into the SplitNN
    (recreations from the attack get worse)
- [ ] Formalise
    - Evaluate privacy parameters
    - Bound intermediate data with tanh to get epsilon estimate?
    - Evaluate privacy by measuring raw data/recreated data similarity

## Get started

### Requirements

Python `3.6, 3.7, 3.8, 3.9`.

A [conda environment](./environment.yml),
`dpsnn`,
has been provided with all packages required to run the experiments,
including the local source code
(Pytorch-cpu only - remove `cpuonly` to enable GPU computation).
Run `conda env create -f environment.yml` to create the environment
using the latest packages,
or `conda env create -f environment-lock.yml` to use fixed package versions
(for reproducibility).
`conda activate dpsnn` to activate the environment.

Alternatively,
run `pip install -e .` to install the local source code only,
`dpsnn`.

### Train models

Training scripts are in [`scripts`](./scripts).

- `python scripts/train_model.py --noise_scale <noise_level> --nopeek_weight <weight>` to train a differentially private model
using noise drawn from Laplacian distribution with scale `<noise_level>` and NoPeek loss weighted by `<weight>`.


## Notebooks

We have provided relevant analysis in the [`notebooks/`](notebooks) folder.
Be aware that previous exploratory notebooks were removed.
Look over previous commits for a full history of experimentation.

## Contributing

- Use `black` to format code
and `isort` to format imports.
Where possible,
add type hints
- Use `pytorch_lightning` to build PyTorch models that can scale

## License

Apache 2.0. See the full [license](LICENSE).
