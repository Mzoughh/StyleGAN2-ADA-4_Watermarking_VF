#!/bin/bash

METRICS="uchida_extraction,fid50k_full"
NETWORK="/home/mzoughebi/personal_study/stylegan2-ada-pytorch_VF/training-runs-uchi_test/00000-CelebA_adapted_128-watermarking1-resumecustom/network-snapshot-000006.pkl"



echo "==== Running pruning attacks ===="
for PERCENT in 10 40 80 90 99
do
    echo "Running pruning with $PERCENT% sparsity..."
    python calc_metrics_after_network_attacks.py \
        --metrics="$METRICS" \
        --network="$NETWORK" \
        --attack_name="pruning" \
        --attacks_parameters="$PERCENT"
done

echo "==== Running quantization attacks ===="
for BITS in 4 6 8 2
do
    echo "Running quantization with $BITS bits..."
    python calc_metrics_after_network_attacks.py \
        --metrics="$METRICS" \
        --network="$NETWORK" \
        --attack_name="quantization" \
        --attacks_parameters="$BITS"
done

echo "==== Running no attack ===="
python calc_metrics_after_network_attacks.py \
    --metrics="$METRICS" \
    --network="$NETWORK" \
    --attack_name="none"


