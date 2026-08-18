import torch.nn as nn

from .dot_ops import Gaussian


class Gaussianlayer(nn.Module):
    def __init__(self, sigma=None, kernel_size=15):
        super().__init__()
        if sigma is None:
            sigma = [4]
        self.gaussian = Gaussian(1, sigma, kernel_size=kernel_size, padding=kernel_size // 2, froze=True)

    def forward(self, dotmaps):
        return self.gaussian(dotmaps)
