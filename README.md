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
Applying differential privacy to the model
(differentially private stochastic gradient descent - the Abadi method)
does not defend against this attack
as output from a trained model part is deterministic
and therefore a decoder model can be trained.

### Aims
This project aims to project SplitNNs
from black box model inversion attack
by applying differential privacy to the data being transferred between model parts.
The idea is that the stochasticity of intermediate data can stop a model
from learning to invert it back into raw data.

### Roadmap
- [ ] Proof of concept: show that the attack can be protected against
    - Eyeball data being protected (i.e. no formal evaluation of privacy)
    - Brief analysis of privacy/accuracy trade-off
- [ ] Explore limits
    - Impact on model size / architecture
    - Can it work with datasets more complex than MNIST
    - How many data points does the attacker need
- [ ] Integrate NoPeekNN
    - NoPeekNN cannot defend against the attack on its own,
    but it has an effect similar to introducing more layers into the SplitNN
    (recreations from the attack get worse)
- [ ] Formalise
    - Evaluate privacy parameters
    - Evaluate privacy by measuring raw data/recreated data similarity

## Get started
A conda environment, `dpsnn`, has been provided (Pytorch-cpu only).
Run `conda env create -f environment.yml` to create the environment
using the latest packages,
or `conda env create -f environment-lock.yml` to use fixed package versions
(for reproducibility).


## License
Apache 2.0. See the full [license](LICENSE).