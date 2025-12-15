#!/bin/bash

# ==============================
# Training script for fine-tuning StyleGAN2-ADA with Uchida method
# ==============================

# --- Default parameters  ---
SNAP=1 # Snapshot interval
OUTDIR="./training_PROD_TONDI" # Output directory for training results
DATA="/home/mzoughebi/personal_study/CelebA_adapted_128.zip" # Path to the dataset
RESUME="/home/mzoughebi/personal_study/weights/trained_16000.pkl" # Path to the pre-trained model
# RESUME="/home/mzoughebi/personal_study/StyleGAN2-ADA-4_Watermarking_VF/bash_launching_scripts/training_run_test_IPR/00005-CelebA_adapted_128-watermarking1-noaug-resumecustom/network-snapshot-001000.pkl"
CFG="watermarking" # Configuration for watermarking
GPUS=1 # Number of GPUs to use
METRICS="TONDI_extraction,fid50k_full" # Metric for the evaluation of T4G method

# Path configuration
PATH_CONF="/home/mzoughebi/personal_study/StyleGAN2-ADA-4_Watermarking_VF/configs/watermarking_dict_conf_TONDI.json"

export CUDA_HOME=/home/mzoughebi/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TORCH_CUDA_ARCH_LIST="7.5" # Set the CUDA architecture list
export CC=gcc
export CXX=g++
# --- RUN the command ---
python /home/mzoughebi/personal_study/StyleGAN2-ADA-4_Watermarking_VF/train.py \
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


# --- Configurations uses in train.py for T4G ---
#'watermarking': dict(ref_gpus=-1, kimg=250,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1, gamma=-1,   ema=1,  ramp=0.05, map=2),       # Based on 'auto' but for watermarking finetunning.