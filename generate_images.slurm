#!/bin/bash
#
# Image generation script with StyleGAN2 on Jean Zay (V100)
#

#SBATCH -A lfb@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --job-name=sg2_generate_v100
#SBATCH --output=GENERATE-%j.out

set -eo pipefail
module purge
module load pytorch-gpu/py3/1.8.0
set -u
export PYTHONUNBUFFERED=1

# -----------------------------
# Paths (ADAPT)
# -----------------------------
CODEDIR="$WORK/ENV_JEAN_ZAY/StyleGAN2-ADA-4_Watermarking_VF"

# Configuration - Modifiez ces valeurs selon vos besoins
NETWORK="$CODEDIR/training_run_experiences/IPR_PROD/00000-CelebA_adapted_128-watermarking1-noaug-resumecustom/network-snapshot-001426.pkl"
OUTDIR="generated_image_IPR_W"
SEEDS="0-99"
NOISE_MODE="none"
TRUNC="1.0"
CLASS_IDX=""

# V100 => sm_70
export TORCH_CUDA_ARCH_LIST="7.0"
export CC=gcc
export CXX=g++

cd "$CODEDIR"

echo "=== Génération d'images StyleGAN2 sur Jean Zay ==="
echo "Modèle:         $NETWORK"
echo "Dossier sortie: $OUTDIR"
echo "Graines:        $SEEDS"
echo "Mode bruit:     $NOISE_MODE (désactive le bruit)"
echo "Troncature:     $TRUNC (style mixing désactivé)"
if [[ -n "$CLASS_IDX" ]]; then
    echo "Classe:         $CLASS_IDX"
fi
echo "================================="

# Création du dossier de sortie
mkdir -p "$OUTDIR"

# Construction et exécution de la commande
if [[ -n "$CLASS_IDX" ]]; then
    python generate.py --network="$NETWORK" --outdir="$OUTDIR" --seeds="$SEEDS" --noise-mode="$NOISE_MODE" --trunc="$TRUNC" --class="$CLASS_IDX"
else
    python generate.py --network="$NETWORK" --outdir="$OUTDIR" --seeds="$SEEDS" --noise-mode="$NOISE_MODE" --trunc="$TRUNC"
fi

echo " ===> Image generation completed successfully on V100 <=== "