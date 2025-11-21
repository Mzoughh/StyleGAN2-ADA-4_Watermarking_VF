import torch.nn as nn
from pytorch_msssim import SSIM, MS_SSIM


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM(data_range=1)

    def forward(self, x, y):
        return 1 - self.ssim(x, y)
