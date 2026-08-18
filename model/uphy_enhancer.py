import torch
import torch.nn as nn
import torch.nn.functional as F


def _sanitize_tensor(x, clamp_value=1e4):
    x = torch.nan_to_num(x, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
    if clamp_value is not None:
        x = torch.clamp(x, min=-clamp_value, max=clamp_value)
    return x


def _depthwise_block(channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
        nn.GELU(),
        nn.Conv2d(channels, channels, kernel_size=1, stride=1),
        nn.GELU(),
    )


class BackgroundEliminationModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lowpass = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=1, padding=4),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        background = _sanitize_tensor(self.lowpass(x), clamp_value=100.0)
        foreground = x - background
        output = x + self.gate(foreground) * foreground
        return _sanitize_tensor(output, clamp_value=100.0)


class FrequencyEnhancementBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Tanh(),
        )
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

    @staticmethod
    def _build_radial_mask(height, width, device, dtype):
        fy = torch.fft.fftfreq(height, d=1.0, device=device).abs().unsqueeze(1)
        fx = torch.fft.rfftfreq(width, d=1.0, device=device).abs().unsqueeze(0)
        radius = torch.sqrt(fy * fy + fx * fx)
        radius = radius / (radius.max() + 1e-6)
        return radius.unsqueeze(0).unsqueeze(0).to(dtype)

    def forward(self, x):
        feat = _sanitize_tensor(self.pre(x), clamp_value=100.0)
        batch_size, channels, height, width = feat.shape

        spectrum = torch.fft.rfft2(feat.float(), norm="ortho")
        amplitude = torch.abs(spectrum)
        radial_mask = self._build_radial_mask(height, width, feat.device, amplitude.dtype)

        low_energy = torch.log1p((amplitude * (1.0 - radial_mask)).mean(dim=(2, 3)))
        high_energy = torch.log1p((amplitude * radial_mask).mean(dim=(2, 3)))
        gate_input = _sanitize_tensor(torch.cat([low_energy, high_energy], dim=1), clamp_value=100.0)
        gate = self.gate(gate_input).view(batch_size, channels, 1, 1)

        high_gain = 1.0 + 0.30 * gate
        low_gain = 1.0 - 0.08 * gate
        gain = low_gain * (1.0 - radial_mask) + high_gain * radial_mask
        enhanced_spectrum = spectrum * gain.to(spectrum.dtype)
        restored = torch.fft.irfft2(enhanced_spectrum, s=(height, width), norm="ortho")
        restored = _sanitize_tensor(restored, clamp_value=100.0).to(x.dtype)
        output = x + self.post(restored)
        return _sanitize_tensor(output, clamp_value=100.0)


class ColorAdaptiveCompensation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels * 2, kernel_size=1, stride=1),
        )

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt((x - mean).pow(2).mean(dim=(2, 3), keepdim=True) + 1e-6)
        stats = _sanitize_tensor(torch.cat([mean, std], dim=1), clamp_value=100.0)
        params = self.mlp(stats)
        scale, bias = torch.chunk(params, 2, dim=1)
        scale = 1.0 + 0.25 * torch.tanh(scale)
        bias = 0.08 * torch.tanh(bias)
        return _sanitize_tensor(x * scale + bias, clamp_value=100.0)


class IlluminationEnhancementModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            _depthwise_block(channels),
            nn.Conv2d(channels, 3, kernel_size=1, stride=1),
        )

    def forward(self, rgb):
        luminance = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        low_freq = F.avg_pool2d(luminance, kernel_size=11, stride=1, padding=5)
        delta = torch.tanh(_sanitize_tensor(self.body(torch.cat([rgb, low_freq], dim=1)), clamp_value=20.0))
        return _sanitize_tensor(delta, clamp_value=1.0)


class DetailBoostModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(6, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            _depthwise_block(channels),
            nn.Conv2d(channels, 3, kernel_size=1, stride=1),
        )

    def forward(self, rgb):
        blur = F.avg_pool2d(rgb, kernel_size=5, stride=1, padding=2)
        highpass = rgb - blur
        local_contrast = torch.sqrt(
            torch.clamp(
                F.avg_pool2d(rgb * rgb, kernel_size=7, stride=1, padding=3)
                - F.avg_pool2d(rgb, kernel_size=7, stride=1, padding=3).pow(2),
                min=0.0,
            )
            + 1e-6
        )
        delta = torch.tanh(
            _sanitize_tensor(self.body(torch.cat([highpass, local_contrast], dim=1)), clamp_value=20.0)
        )
        return _sanitize_tensor(delta, clamp_value=1.0)


class RetinexContrastModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(7, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            _depthwise_block(channels),
            nn.Conv2d(channels, 3, kernel_size=1, stride=1),
        )

    @staticmethod
    def _gray_std_map(gray):
        mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        mean_sq = F.avg_pool2d(gray * gray, kernel_size=9, stride=1, padding=4)
        return torch.sqrt(torch.clamp(mean_sq - mean.pow(2), min=0.0) + 1e-6)

    def forward(self, rgb):
        eps = 1e-3
        illumination = F.avg_pool2d(rgb, kernel_size=15, stride=1, padding=7)
        retinex = torch.log(torch.clamp(rgb, min=eps)) - torch.log(torch.clamp(illumination, min=eps))
        retinex = torch.clamp(retinex, min=-3.0, max=3.0) / 3.0

        highpass = rgb - illumination
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        contrast = self._gray_std_map(gray)
        features = torch.cat([retinex, highpass, contrast], dim=1)
        delta = torch.tanh(_sanitize_tensor(self.body(features), clamp_value=20.0))
        return _sanitize_tensor(delta, clamp_value=1.0)


class FusionGateModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.raw_stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.enh_stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.degradation_head = nn.Sequential(
            nn.Conv2d(3, max(channels // 2, 8), kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(max(channels // 2, 8), 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )
        self.foreground_head = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

        nn.init.constant_(self.foreground_head[-2].bias, 1.5)


    @staticmethod
    def _local_std_map(gray):
        mean = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        mean_sq = F.avg_pool2d(gray * gray, kernel_size=7, stride=1, padding=3)
        return torch.sqrt(torch.clamp(mean_sq - mean.pow(2), min=0.0) + 1e-6)

    def forward(self, raw_rgb, enhanced_rgb):
        raw_feat = _sanitize_tensor(self.raw_stem(raw_rgb), clamp_value=100.0)
        enh_feat = _sanitize_tensor(self.enh_stem(enhanced_rgb), clamp_value=100.0)

        raw_gray = 0.299 * raw_rgb[:, 0:1] + 0.587 * raw_rgb[:, 1:2] + 0.114 * raw_rgb[:, 2:3]
        enh_gray = 0.299 * enhanced_rgb[:, 0:1] + 0.587 * enhanced_rgb[:, 1:2] + 0.114 * enhanced_rgb[:, 2:3]
        contrast_gain = torch.clamp(self._local_std_map(enh_gray) - self._local_std_map(raw_gray), min=0.0)
        degradation_input = torch.cat(
            [
                torch.mean(torch.abs(enhanced_rgb - raw_rgb), dim=1, keepdim=True),
                1.0 - raw_gray,
                contrast_gain,
            ],
            dim=1,
        )
        degradation_map = _sanitize_tensor(self.degradation_head(degradation_input), clamp_value=1.0)
        fusion_input = torch.cat([raw_feat, enh_feat, degradation_map], dim=1)
        gate = _sanitize_tensor(
            self.gate_head(fusion_input),
            clamp_value=1.0,
        )
        foreground_prior = _sanitize_tensor(
            self.foreground_head(fusion_input),
            clamp_value=1.0,
        )
        return gate, degradation_map, foreground_prior


class UPhyInspiredEnhancer(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.feb = FrequencyEnhancementBlock(hidden_channels)
        self.bem = BackgroundEliminationModule(hidden_channels)
        self.cacm = ColorAdaptiveCompensation(hidden_channels)
        self.illumination = IlluminationEnhancementModule(hidden_channels)
        self.detail = DetailBoostModule(hidden_channels)
        self.retinex = RetinexContrastModule(hidden_channels)
        self.fusion = FusionGateModule(hidden_channels)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, kernel_size=1, stride=1),
        )

        self.residual_strength = nn.Parameter(torch.tensor(-0.8))
        self.illumination_strength = nn.Parameter(torch.tensor(0.0))
        self.detail_strength = nn.Parameter(torch.tensor(-0.4))
        self.retinex_strength = nn.Parameter(torch.tensor(-0.2))

        nn.init.normal_(self.to_rgb[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.to_rgb[-1].bias)

    @staticmethod
    def _gradient_components(x):
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        return grad_x, grad_y

    @staticmethod
    def _total_variation(x):
        grad_x, grad_y = UPhyInspiredEnhancer._gradient_components(x)
        return grad_x.abs().mean() + grad_y.abs().mean()

    @staticmethod
    def _local_std_map(gray):
        mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        mean_sq = F.avg_pool2d(gray * gray, kernel_size=9, stride=1, padding=4)
        return torch.sqrt(torch.clamp(mean_sq - mean.pow(2), min=0.0) + 1e-6)

    def _compute_aux_losses(self, raw_rgb, fused_rgb, degradation_map, gate, foreground_prior, fg_masks=None):
        channel_mean = fused_rgb.mean(dim=(2, 3))
        color_constancy = (channel_mean - channel_mean.mean(dim=1, keepdim=True)).abs().mean()

        raw_gray = 0.299 * raw_rgb[:, 0:1] + 0.587 * raw_rgb[:, 1:2] + 0.114 * raw_rgb[:, 2:3]
        fused_gray = 0.299 * fused_rgb[:, 0:1] + 0.587 * fused_rgb[:, 1:2] + 0.114 * fused_rgb[:, 2:3]

        patch_mean = F.avg_pool2d(fused_gray, kernel_size=16, stride=16)
        exposure_target = 0.42
        exposure_loss = torch.abs(patch_mean - exposure_target).mean()

        raw_dx, raw_dy = self._gradient_components(raw_gray)
        fused_dx, fused_dy = self._gradient_components(fused_gray)
        raw_grad = torch.sqrt(raw_dx.pow(2) + 1e-6).mean() + torch.sqrt(raw_dy.pow(2) + 1e-6).mean()
        fused_grad = torch.sqrt(fused_dx.pow(2) + 1e-6).mean() + torch.sqrt(fused_dy.pow(2) + 1e-6).mean()
        edge_preserve_loss = F.relu(raw_grad - fused_grad)

        raw_contrast = self._local_std_map(raw_gray)
        fused_contrast = self._local_std_map(fused_gray)
        contrast_gain_loss = F.relu(raw_contrast - fused_contrast).mean()

        tv_loss = self._total_variation(degradation_map) + 0.5 * self._total_variation(gate)

        if fg_masks is not None:
            fg_masks = torch.clamp(fg_masks, 0.0, 1.0)
            if fg_masks.dim() == 3:
                fg_masks = fg_masks.unsqueeze(1)
            bg_masks = 1.0 - fg_masks

            fg_denom = fg_masks.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
            bg_denom = bg_masks.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)

            raw_fg_mean = (raw_gray * fg_masks).sum(dim=(2, 3), keepdim=True) / fg_denom
            raw_bg_mean = (raw_gray * bg_masks).sum(dim=(2, 3), keepdim=True) / bg_denom
            fused_fg_mean = (fused_gray * fg_masks).sum(dim=(2, 3), keepdim=True) / fg_denom
            fused_bg_mean = (fused_gray * bg_masks).sum(dim=(2, 3), keepdim=True) / bg_denom

            raw_separation = torch.abs(raw_fg_mean - raw_bg_mean)
            fused_separation = torch.abs(fused_fg_mean - fused_bg_mean)
            separation_gain_loss = F.relu(raw_separation + 0.03 - fused_separation).mean()

            fg_change = ((fused_rgb - raw_rgb).abs() * fg_masks).sum(dim=(1, 2, 3)) / fg_masks.sum(dim=(1, 2, 3)).clamp(min=1.0)
            bg_change = ((fused_rgb - raw_rgb).abs() * bg_masks).sum(dim=(1, 2, 3)) / bg_masks.sum(dim=(1, 2, 3)).clamp(min=1.0)
            focus_loss = F.relu(bg_change - 0.85 * fg_change).mean()

            background_stability = ((fused_rgb - raw_rgb).abs() * bg_masks).sum(dim=(1, 2, 3)) / bg_masks.sum(dim=(1, 2, 3)).clamp(min=1.0)
            background_stability_loss = background_stability.mean()
            fg_prior_target = F.interpolate(fg_masks, size=foreground_prior.shape[-2:], mode="nearest")
            prior_align_loss = F.mse_loss(foreground_prior, fg_prior_target)

            color_constancy = color_constancy + 0.5 * focus_loss
            contrast_gain_loss = contrast_gain_loss + separation_gain_loss + 0.25 * background_stability_loss + 0.5 * prior_align_loss
            tv_loss = tv_loss + 0.1 * focus_loss + 0.1 * self._total_variation(foreground_prior)

        return {
            "color": _sanitize_tensor(color_constancy + 0.5 * exposure_loss, clamp_value=10.0),
            "edge": _sanitize_tensor(edge_preserve_loss + contrast_gain_loss, clamp_value=10.0),
            "tv": _sanitize_tensor(tv_loss, clamp_value=10.0),
        }

    def forward(self, x, fg_masks=None, return_aux=False):
        x = _sanitize_tensor(x, clamp_value=100.0)
        rgb = torch.clamp(x * self.std + self.mean, 0.0, 1.0)

        feat = _sanitize_tensor(self.stem(rgb), clamp_value=100.0)
        feat = self.feb(feat)
        feat = self.bem(feat)
        feat = self.cacm(feat)

        learned_residual = torch.tanh(_sanitize_tensor(self.to_rgb(feat), clamp_value=20.0))
        illumination_delta = self.illumination(rgb)
        detail_delta = self.detail(rgb)
        retinex_delta = self.retinex(rgb)

        residual_scale = 0.08 + 0.12 * torch.sigmoid(self.residual_strength)
        illumination_scale = 0.10 + 0.18 * torch.sigmoid(self.illumination_strength)
        detail_scale = 0.05 + 0.10 * torch.sigmoid(self.detail_strength)
        retinex_scale = 0.06 + 0.10 * torch.sigmoid(self.retinex_strength)

        candidate_rgb = torch.clamp(
            rgb
            + residual_scale * learned_residual
            + illumination_scale * illumination_delta
            + detail_scale * detail_delta
            + retinex_scale * retinex_delta,
            0.0,
            1.0,
        )

        gate, degradation_map, foreground_prior = self.fusion(rgb, candidate_rgb)
        fused_rgb = torch.clamp(rgb + gate * (candidate_rgb - rgb), 0.0, 1.0)
        fused = _sanitize_tensor((fused_rgb - self.mean) / self.std, clamp_value=100.0)

        if not return_aux:
            return fused, foreground_prior

        aux_losses = self._compute_aux_losses(
            rgb,
            fused_rgb,
            degradation_map,
            gate,
            foreground_prior,
            fg_masks=fg_masks,
        )
        return fused, aux_losses, foreground_prior
