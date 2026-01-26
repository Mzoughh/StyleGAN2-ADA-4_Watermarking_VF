#!/bin/bash

# =================================================================
# Training script for fine-tuning StyleGAN2-ADA with UCHI method
# =================================================================

# --- Default parameters  ---
SNAP=1 
OUTDIR="./UCHI_128_128_AFHQ" 
DATA="./datasets/afhqcat_128.zip" 
RESUME="./vanilla_weights/vanilla_afhq_128x128.pkl" 
CFG="watermarking" # Configuration for watermarking
GPUS=1 # Number of GPUs to use
METRICS="UCHI_extraction,fid50k_full" # Metric to evaluate of IPR method
PATH_CONF="./configs/watermarking_dict_conf_UCHI.json"  # Path configuration

cd .. 

# --- SEPECIFIC ACCORDING THE ARCHITECTURE ---
export CUDA_HOME=/home/mzoughebi/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TORCH_CUDA_ARCH_LIST="7.5" # Set the CUDA architecture list
export CC=gcc
export CXX=g++

# --- RUN the command ---
# no aug for desactivate ADA augmentation during FT
python ./train.py \
    --snap=$SNAP \
    --outdir="$OUTDIR" \
    --data="$DATA" \
    --resume="$RESUME" \
    --cfg="$CFG" \
    --gpus="$GPUS" \
    --metrics="$METRICS" \
    --aug=noaug \
    --water_config_path="$PATH_CONF"

echo " ===> Training completed successfully <=== "

