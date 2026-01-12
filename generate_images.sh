#!/bin/bash

# =============================================================================
# Script simple pour générer des images avec StyleGAN2
# Modifiez les paramètres ci-dessous selon vos besoins
# =============================================================================

export CUDA_HOME=/home/mzoughebi/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TORCH_CUDA_ARCH_LIST="7.5" # Set the CUDA architecture list
export CC=gcc
export CXX=g++


# Configuration - Modifiez ces valeurs selon vos besoins
NETWORK="/home/mzoughebi/personal_study/PROD/training_PROD_TONDI/00002-CelebA_adapted_128-watermarking1-noaug-resumecustom/network-snapshot-000750.pkl"
OUTDIR="generated_image_TONDI"
SEEDS="0"
NOISE_MODE="none"
TRUNC="1.0"
CLASS_IDX="" 

# =============================================================================
# Ne pas modifier en dessous de cette ligne
# =============================================================================

echo "=== Génération d'images StyleGAN2 ==="
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

echo "Terminé !" 