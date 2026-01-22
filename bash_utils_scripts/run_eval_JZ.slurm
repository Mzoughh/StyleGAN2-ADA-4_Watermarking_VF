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
#SBATCH --time=18:00:00
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
NETWORK_NAMES=('UCHI' 'T4G' 'TONDI' 'IPR')
NETWORK_NUMBERS=('000000' '000001' '000003' '000006')


for i in "${!NETWORK_NAMES[@]}"; do
    NAME="${NETWORK_NAMES[$i]}"
    NUMBER="${NETWORK_NUMBERS[$i]}"
    METRICS="${NAME}_extraction,fid50k_full"
    NETWORK_W="./${NAME}_128_128_FFHQ/network-snapshot-${NUMBER}.pkl"

    echo "==== Running no attack ===="
    python calc_metrics_after_network_attacks.py \
        --metrics="$METRICS" \
        --network="$NETWORK_W" \
        --attack_name="none"

    echo "==== Running pruning attacks ===="
    for PERCENT in 5 15 25 32 40 45
    do
        echo "Running pruning with $PERCENT% sparsity..."
        python calc_metrics_after_network_attacks.py \
            --metrics="$METRICS" \
            --network="$NETWORK_W" \
            --attack_name="pruning" \
            --attacks_parameters="$PERCENT"
    done

    echo "==== Running quantization attacks ===="
    for BITS in  7 8 9 10 
    do
        echo "Running quantization with $BITS bits..."
        python calc_metrics_after_network_attacks.py \
            --metrics="$METRICS" \
            --network="$NETWORK_W" \
            --attack_name="quantization" \
            --attacks_parameters="$BITS"
    done

    echo "==== Noise attacks ===="
    for POWER in  1 3 5 7 
    do
      echo "Running quantization with $POWER power..."
      python calc_metrics_after_network_attacks.py \
        --metrics="$METRICS" \
        --network="$NETWORK_W" \
        --attack_name="noise" \
        --attacks_parameters="$POWER"
    done
done

echo " ===> Evaluation completed successfully on V100 <=== "