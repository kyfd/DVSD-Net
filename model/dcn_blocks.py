import warnings

import torch
import torch.nn as nn

try:
    from torchvision.ops import DeformConv2d

    HAS_TORCHVISION_DCN = True
    DCN_BACKEND = "torchvision_deformconv2d"
except Exception:
    DeformConv2d = None
    HAS_TORCHVISION_DCN = False
    DCN_BACKEND = "fallback_conv"


class DeformableRefineBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.use_deform_conv = HAS_TORCHVISION_DCN

        if self.use_deform_conv:
            self.offset = nn.Conv2d(channels, 2 * kernel_size * kernel_size, 3, 1, 1)
            self.mask = nn.Conv2d(channels, kernel_size * kernel_size, 3, 1, 1)
            self.refine = DeformConv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )

            nn.init.constant_(self.offset.weight, 0.0)
            nn.init.constant_(self.offset.bias, 0.0)
            nn.init.constant_(self.mask.weight, 0.0)
            nn.init.constant_(self.mask.bias, 0.0)
        else:
            warnings.warn(
                "torchvision.ops.DeformConv2d is unavailable. Falling back to a regular convolution block.",
                RuntimeWarning,
            )
            self.refine = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)

        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_deform_conv:
            offset = self.offset(x)
            mask = torch.sigmoid(self.mask(x))
            y = self.refine(x, offset, mask)
        else:
            y = self.refine(x)

        y = self.act(self.norm(y))
        return x + y
