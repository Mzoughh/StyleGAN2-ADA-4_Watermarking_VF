#!/bin/bash

# METRICS="T4G_extraction"
METRICS="T4G_extraction,fid50k_full,uchida_extraction"
NETWORK_W="/home/mzoughebi/personal_study/stylegan2-ada-pytorch_VF_test_trigger/training_run_uchida_B1/00001-CelebA_adapted_128-watermarking1-noaug-resumecustom/network-snapshot-000184.pkl"
NETWORK="/home/mzoughebi/personal_study/stylegan2-ada-pytorch_VF/training-runs/00001-CelebA_adapted_128-auto1-resumecustom/network-snapshot-010000.pkl"


echo "==== Running no attack ===="
python calc_metrics_after_network_attacks.py \
    --metrics="$METRICS" \
    --network="$NETWORK_W" \
    --attack_name="none"


# echo "==== Running pruning attacks ===="
# for PERCENT in 5 7 10 15 20 25 30 35 40 50 60 80 90
# do
#     echo "Running pruning with $PERCENT% sparsity..."
#     python calc_metrics_after_network_attacks.py \
#         --metrics="$METRICS" \
#         --network="$NETWORK_W" \
#         --attack_name="pruning" \
#         --attacks_parameters="$PERCENT"
# done

# echo "==== Running quantization attacks ===="
# for BITS in 2 3 4 5 6 7 8 9 10 11 12 
# do
#     echo "Running quantization with $BITS bits..."
#     python calc_metrics_after_network_attacks.py \
#         --metrics="$METRICS" \
#         --network="$NETWORK_W" \
#         --attack_name="quantization" \
#         --attacks_parameters="$BITS"
# done



