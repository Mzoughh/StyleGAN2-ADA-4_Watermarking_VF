#!/bin/bash

# =================================================================
# Training script for fine-tuning StyleGAN2-ADA with TONDI  method on Jean Zay (V100)
# =================================================================

#SBATCH -A lfb@v100                 # Project/account
#SBATCH --partition=gpu_p2          # V100 GPU partition
#SBATCH --qos=qos_gpu-t3            # GPU QoS
#SBATCH --gres=gpu:1                # 1 GPU
#SBATCH --cpus-per-task=10          # 10 CPU cores
#SBATCH --hint=nomultithread        # Physical cores only
#SBATCH --time=18:00:00             # Max walltime
#SBATCH --job-name=sg2_TONDI_V100     # Job name
#SBATCH --output=TONDI-%j.out         # Log file: sg2_TONDI_v100-<jobid>.out

# -----------------------------
# 1. Load environment (PyTorch)
# -----------------------------
set -eo pipefail
module purge
module load pytorch-gpu/py3/1.8.0     
set -u 
export PYTHONUNBUFFERED=1           

# -----------------------------
# 2. Paths and config 
# -----------------------------

CODEDIR="$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF"
SNAP=1                              
OUTDIR="$CODEDIR/training_run_experiences/TONDI_128_128_FFHQ"
DATA="$CODEDIR/datasets/ffhq128x128.zip"
RESUME="$CODEDIR/vanilla_weights/vanilla_ffhq_128x128_e1600.pkl"
CFG="watermarking"
PATH_CONF="$CODEDIR/configs/watermarking_dict_conf_TONDI.json"
GPUS=1
METRICS="TONDI_extraction,fid50k_full"

# --- SEPECIFIC ACCORDING THE ARCHITECTURE: CUDA arch for V100 (sm_70)
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
    --resume="$RESUME" \
    --cfg="$CFG" \
    --gpus="$GPUS" \
    --metrics="$METRICS" \
    --aug=noaug \
    --water_config_path="$PATH_CONF"

echo " ===> Training completed successfully on V100 <=== "

