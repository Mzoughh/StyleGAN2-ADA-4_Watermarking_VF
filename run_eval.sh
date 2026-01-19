#!/bin/bash

METRICS="T4G_extraction,fid50k_full"
# METRICS="uchida_extraction"
NETWORK_W="/lustre/fswork/projects/rech/lfb/uak91sd/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF/training_run_experiences/T4G_PROD/0001_B16//network-snapshot-001590.pkl"
# NETWORK_W="/home/mzoughebi/personal_study/PROD/training_run_PROD_UCHIDA/CUSTOM_L1_E10/network-snapshot-000010.pkl"

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
# for PERCENT in 20  # 40 50 60 5 15 35
# do
#     echo "Running pruning with $PERCENT% sparsity..."
#     python calc_metrics_after_network_attacks.py \
#         --metrics="$METRICS" \
#         --network="$NETWORK_W" \
#         --attack_name="pruning" \
#         --attacks_parameters="$PERCENT"
# done

# echo "==== Running quantization attacks ===="
# for BITS in  6 7 # 8 9 10 4 5
# do
#     echo "Running quantization with $BITS bits..."
#     python calc_metrics_after_network_attacks.py \
#         --metrics="$METRICS" \
#         --network="$NETWORK_W" \
#         --attack_name="quantization" \
#         --attacks_parameters="$BITS"
# done

# echo "==== Noise attacks ===="
# for POWER in  5 #15   1 3 57 10
# do
#   echo "Running quantization with $POWER power..."
#   python calc_metrics_after_network_attacks.py \
#     --metrics="$METRICS" \
#     --network="$NETWORK_W" \
#     --attack_name="noise" \
#     --attacks_parameters="$POWER"
# done

# echo " ===> Evaluation completed successfully on 2080Ti <=== "

