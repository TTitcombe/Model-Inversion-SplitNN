#!/bin/bash

if [ $1 = 'plain' ] || [ $1 = 'all' ]; then
    echo -e "\nTraining model without noise or NoPeek\n"
    python3 train/train_model.py --noise_scale 0.0 --nopeek_weight 0.0
fi

if [ $1 = 'noise' ] || [ $1 = 'all' ]; then
    for noise in 0.1 0.2 0.5 1.0 2.0 5.0
    do
        echo -e "\nTraining model with noise $noise\n"
        python3 scripts/train_model.py --noise_scale $noise --nopeek_weight 0.0
    done
fi

# NoPeek is very computationally expensive!
if [ $1 = 'nopeek' ] || [ $1 = 'all' ]; then
    for nopeek in 0.05 0.1 0.25 0.5 1.0
    do
        echo -e "\nTraining model with nopeek weight $nopeek\n"
        python3 scripts/train_model.py --noise_scale 0.0 --nopeek_weight $nopeek --batch_size 32
    done
fi

# NoPeek is very computationally expensive!
if [ $1 = 'combo' ] || [ $1 = 'all' ]; then
    for noise in 0.0 0.1 0.2 0.5
    do
        for nopeek in 0.1 0.2 0.5
        do
            echo -e "\nTraining model with noise $noise and nopeek weight $nopeek\n"
            python3 scripts/train_model.py --noise-scale $noise --nopeek-weight $nopeek --batch-size 32 --max-epochs 5
        done
    done
fi

if [ $1 = 'performance' ] || [ $1 = 'all' ]; then
    echo -e "\nEvaluating classifier performances\n"
    python3 scripts/evaluate_models.py

    echo -e "\nEvaluating attack performances\n"
    python3 scripts/evaluate_attackers.py
fi
