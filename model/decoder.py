import torch.nn as nn
import torch.nn.functional as F


class _DensityDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decode_head0 = self._block(256, 256)
        self.decode_head1 = self._block(256, 256)
        self.decode_head2 = self._block(256, 256)
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
        )

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(self.decode_head0(x), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.decode_head1(x), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.decode_head2(x), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.decode_head3(x), scale_factor=2, mode="bilinear", align_corners=False)
        return x


class GlobalDecoder(_DensityDecoder):
    pass


class ShareDecoder(_DensityDecoder):
    pass
