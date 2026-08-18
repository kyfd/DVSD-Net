import torch
import torch.nn as nn
import torch.nn.functional as F


class ForegroundSuppressionGate(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or max(channels // 2, 64)

        self.target_proj = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, stride=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.support_proj = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, stride=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=1),
        )

        nn.init.constant_(self.spatial_gate[-1].bias, 2.5)
        nn.init.constant_(self.channel_gate[-1].bias, 2.0)

    def forward(self, target_feature, support_feature):
        target_embed = self.target_proj(target_feature)
        support_embed = self.support_proj(support_feature)
        support_embed = F.interpolate(
            support_embed,
            size=target_embed.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        diff_embed = torch.abs(target_embed - support_embed)

        spatial_gate = torch.sigmoid(self.spatial_gate(torch.cat([target_embed, support_embed, diff_embed], dim=1)))
        channel_gate = torch.sigmoid(self.channel_gate(torch.cat([target_embed, support_embed], dim=1)))

        spatial_scale = 0.25 + 0.75 * spatial_gate
        channel_scale = 0.5 + 0.5 * channel_gate
        return target_feature * spatial_scale * channel_scale
