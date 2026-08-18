import math

import torch
import torch.nn as nn


class Gaussian(nn.Module):
    def __init__(self, in_channels, sigmalist, kernel_size=64, stride=1, padding=0, froze=True):
        super().__init__()
        out_channels = len(sigmalist) * in_channels
        mu = kernel_size // 2
        gauss_funcs = [
            lambda sigma, x=x: math.exp(-((x - mu) ** 2) / float(2 * sigma ** 2))
            for x in range(kernel_size)
        ]

        windows = []
        for sigma in sigmalist:
            gauss = torch.tensor([gauss_func(sigma) for gauss_func in gauss_funcs])
            gauss /= gauss.sum()
            window_1d = gauss.unsqueeze(1)
            window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
            windows.append(window_2d.expand(in_channels, 1, kernel_size, kernel_size).contiguous())

        kernels = torch.stack(windows).permute(1, 0, 2, 3, 4)
        weight = kernels.reshape(out_channels, in_channels, kernel_size, kernel_size)

        self.gkernel = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.gkernel.weight = nn.Parameter(weight)
        if froze:
            self.freeze_parameters()

    def forward(self, dotmaps):
        return self.gkernel(dotmaps)

    def freeze_parameters(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
