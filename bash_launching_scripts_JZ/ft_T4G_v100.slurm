#!/bin/bash
#
# Training script for fine-tuning StyleGAN2-ADA with Uchida method on Jean Zay (V100)
#

#SBATCH -A lfb@v100                 # Project/account
#SBATCH --partition=gpu_p2          # V100 GPU partition
#SBATCH --qos=qos_gpu-t3            # GPU QoS
#SBATCH --gres=gpu:1                # 1 GPU
#SBATCH --cpus-per-task=10          # 10 CPU cores
#SBATCH --hint=nomultithread        # Physical cores only
#SBATCH --time=06:00:00             # Max walltime
#SBATCH --job-name=sg2_IPR_v100     # Job name
#SBATCH --output=T4G-%j.out        # Log file: sg2_IPR_v100-<jobid>.out



# -----------------------------
# 1. Load environment (PyTorch)
# -----------------------------
set -eo pipefail
module purge
module load pytorch-gpu/py3/1.8.0     
set -u 

export PYTHONUNBUFFERED=1           # Live logging

# -----------------------------
# 2. Paths and config (ADAPT THESE)
# -----------------------------

# Base project dir on WORK 
CODEDIR="$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF"

# FOR WEIGHTS AND DATASET
WDATA="$WORK/ENV_JEAN_ZAY"

# Snapshot interval
SNAP=1                              # Snapshot interval

# Output directory for training results
OUTDIR="$CODEDIR/training_run_experiences/Test_T4G"

# Path to the dataset (zip) on WORK
DATA="$WDATA/CelebA_adapted_128.zip"

# Path to the pre-trained model (on WORK)
RESUME="$WDATA/weights/trained_16000.pkl"
RESUMEW="$OUTDIR/WARMUP_PT_1_SSIM_L20_E800/network-snapshot-000764.pkl"

# Watermarking configuration
CFG="watermarking"

# Path configuration
PATH_CONF="$CODEDIR/configs_JZ/watermarking_dict_conf_T4G.json"

# Number of GPUs to use
GPUS=1

# Metric for the evaluation of IPR method
METRICS="T4G_extraction"

# CUDA arch for V100 (sm_70)
export TORCH_CUDA_ARCH_LIST="7.0"
export CC=gcc
export CXX=g++

# -----------------------------
# 3. Run training
# -----------------------------
cd "$CODEDIR"

python train.py \
    --snap="$SNAP" \
    --outdir="$OUTDIR" \
    --data="$DATA" \
    --resume="$RESUMEW" \
    --cfg="$CFG" \
    --gpus="$GPUS" \
    --metrics="$METRICS" \
    --aug=noaug \
    --water_config_path="$PATH_CONF"

echo " ===> Training completed successfully on V100 <=== "
