#!/bin/bash
#
# Evaluation script (metrics + attacks) on Jean Zay (V100)
#

#SBATCH -A lfb@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --job-name=sg2_eval_v100
#SBATCH --output=EVAL-%j.out

set -eo pipefail
module purge
module load pytorch-gpu/py3/1.8.0
set -u
export PYTHONUNBUFFERED=1

# -----------------------------
# Paths (ADAPT)
# -----------------------------
CODEDIR="$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF"

# Exemple : snapshot à évaluer (mets le bon .pkl)
NETWORK_W="$CODEDIR/training_run_experiences/IPR_PROD/00000-CelebA_adapted_128-watermarking1-noaug-resumecustom/network-snapshot-001426.pkl"

# Metrics
METRICS="IPR_extraction,fid50k_full"

# V100 => sm_70
export TORCH_CUDA_ARCH_LIST="7.0"
export CC=gcc
export CXX=g++

cd "$CODEDIR"

# echo "==== Running no attack ===="
# python calc_metrics_after_network_attacks.py \
#   --metrics="$METRICS" \
#   --network="$NETWORK_W" \
#   --attack_name="none"

# echo "==== Running pruning attacks ===="
# for PERCENT in 5 15 25 35 40 50 60
# do
#   echo "Running pruning with $PERCENT% sparsity..."
#   python calc_metrics_after_network_attacks.py \
#     --metrics="$METRICS" \
#     --network="$NETWORK_W" \
#     --attack_name="pruning" \
#     --attacks_parameters="$PERCENT"
# done

# echo "==== Running quantization attacks ===="
# for BITS in 4 5 6 7 8 9 10
# do
#   echo "Running quantization with $BITS bits..."
#   python calc_metrics_after_network_attacks.py \
#     --metrics="$METRICS" \
#     --network="$NETWORK_W" \
#     --attack_name="quantization" \
#     --attacks_parameters="$BITS"
# done

echo "==== Noise attacks ===="
for POWER in 100 50 10 1 
do
  echo "Running quantization with $POWER power..."
  python calc_metrics_after_network_attacks.py \
    --metrics="$METRICS" \
    --network="$NETWORK_W" \
    --attack_name="noise" \
    --attacks_parameters="$POWER"
done

echo " ===> Evaluation completed successfully on V100 <=== "
