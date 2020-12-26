#!/bin/bash

for noise in 0.0 0.1 0.2 0.5 1.0
do
    echo "Training model with noise $noise"
    python train/train_model.py --noise_scale $noise --nopeek_weight 0.0
done
