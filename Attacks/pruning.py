# pruning

import matplotlib.pyplot as plt
import torch
import torch.nn.utils.prune as prune
from Attacks.utils import *

# --------------- W -----------------#
def prune_model_l1_unstructured(model, percentage):
    """
    Prune manuellement un paramètre d'un modèle StyleGAN2-ADA selon le critère L1 (valeur absolue).
    
    :param model: modèle PyTorch (e.g., StyleGAN2-ADA)
    :param percentage: pourcentage de poids à mettre à zéro (entre 0 et 100)
    :return: modèle après pruning
    """
    if percentage <= 0.0 or percentage >= 100.0:
        print("Percentage must be between 0 and 100 (exclusive).")
        return model  
    
    
    percentage = percentage / 100.0
    target_name = "synthesis.b32.conv0.weight"

    for name, parameters in model.named_parameters():
        # if name == target_name:
        if "synthesis" in name:
            print(f"Pruning parameter: {name}")

            tensor = parameters.data
            total = tensor.numel()
            abs_tensor = tensor.abs()

            threshold = torch.quantile(abs_tensor.view(-1), percentage)

            mask = (abs_tensor >= threshold).float()
            before_nonzero = torch.count_nonzero(tensor).item()

            tensor.mul_(mask)

            after_nonzero = torch.count_nonzero(tensor).item()
            sparsity = 100 * (1 - after_nonzero / total)

            print(f"Before pruning: {before_nonzero}/{total} non-zero weights")
            print(f"After pruning:  {after_nonzero}/{total} non-zero weights")
            print(f"Sparsity: {sparsity:.2f}%")

    return model
#--------------------------------#