#!/bin/bash

if [ $1 = 'plain' ] || [ $1 = 'all' ]; then
    echo "Training model without noise or NoPeek"
    python train/train_model.py --noise_scale 0.0 --nopeek_weight 0.0
fi

if [ $1 = 'noise' ] || [ $1 = 'all' ]; then
    for noise in 0.1 0.2 0.5 1.0 2.0 5.0
    do
        echo "Training model with noise $noise"
        python scripts/train_model.py --noise_scale $noise --nopeek_weight 0.0
    done
fi

# NoPeek is very computationally expensive!
if [ $1 = 'nopeek' ] || [ $1 = 'all' ]; then
    for nopeek in 0.05 0.1 0.25 0.5 1.0
    do
        echo "Training model with nopeek weight $nopeek"
        python scripts/train_model.py --noise_scale 0.0 --nopeek_weight $nopeek --batch_size 32
    done
fi

if [ $1 = 'combo' ] || [ $1 = 'all' ]; then
    for noise in 0.1 0.2 0.5 1.0
    do
        for nopeek in 0.1 0.25 0.5
        do
            echo "Training model with noise $noise and nopeek weight $nopeek"
            python scripts/train_model.py --noise_scale $noise --nopeek_weight $nopeek --batch_size 32
        done
    done
fi

if [ $1 = 'performance' ] || [ $1 = 'all' ]; then
    echo "Calculating model performances"
    python scripts/evaluate_models.py
fi
