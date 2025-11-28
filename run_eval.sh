#!/bin/bash

# METRICS="T4G_extraction"
METRICS="fid50k_full" 
NETWORK_W="/home/mzoughebi/personal_study/StyleGAN2-ADA-4_Watermarking_VF/bash_launching_scripts/training_run_test_IPR/00011-CelebA_adapted_128-watermarking1-noaug-resumecustom/network-snapshot-001484.pkl"
NETWORK="/home/mzoughebi/personal_study/stylegan2-ada-pytorch_VF/training-runs/00001-CelebA_adapted_128-auto1-resumecustom/network-snapshot-010000.pkl"

export CUDA_HOME=/home/mzoughebi/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TORCH_CUDA_ARCH_LIST="7.5" # Set the CUDA architecture list
export CC=gcc
export CXX=g++

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



