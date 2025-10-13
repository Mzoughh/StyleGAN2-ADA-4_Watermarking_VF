#!/bin/bash

# ==============================
# Training script for fine-tuning StyleGAN2-ADA with Uchida method
# ==============================

# --- Default parameters  ---
SNAP=1 # Snapshot interval
OUTDIR="./training_run_trigger_test_A" # Output directory for training results
DATA="./dataset/CelebA_adapted_128.zip" # Path to the dataset
RESUME="/home/mzoughebi/personal_study/stylegan2-ada-pytorch_VF/training-runs/00001-CelebA_adapted_128-auto1-resumecustom/network-snapshot-010000.pkl" # Path to the pre-trained model
CFG="watermarking" # Configuration for watermarking
GPUS=1 # Number of GPUs to use
METRICS="uchida_extraction" # Metric for the evaluation of Uchida method

export TORCH_CUDA_ARCH_LIST="6.1" # Set the CUDA architecture list
# --- RUN the command ---
python train.py \
    --snap=$SNAP \
    --outdir="$OUTDIR" \
    --data="$DATA" \
    --resume="$RESUME" \
    --cfg="$CFG" \
    --gpus="$GPUS" \
    --metrics="$METRICS" \
    --aug=noaug

echo " ===> Training completed successfully <=== "


# --- Configurations uses in train.py for Uchida ---
# 'watermarking': dict(ref_gpus=-1, kimg=6,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1, gamma=-1,   ema=1,  ramp=0.05, map=2),       # Based on 'auto' but for watermarking finetunning.