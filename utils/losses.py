import torch.nn as nn
from pytorch_msssim import SSIM, MS_SSIM
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM(data_range=1)

    def forward(self, x, y):
        return 1 - self.ssim(x, y)

class MSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, y):
        # x, y: tensors batchés, normalisés entre 0 et 1
        return F.mse_loss(x, y, reduction=self.reduction)
