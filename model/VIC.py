import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from model.dcn_blocks import DCN_BACKEND, HAS_TORCHVISION_DCN, DeformableRefineBlock
from model.foreground_gate import ForegroundSuppressionGate
from model.uphy_enhancer import UPhyInspiredEnhancer
from model.VGG.VGG16_FPN import VGG16_FPN_Encoder
from model.ViT.models_crossvit import CrossAttentionBlock, FeatureFusionModule
from model.decoder import GlobalDecoder, ShareDecoder


def build_gaussian_layer():
    try:
        from misc.layer import Gaussianlayer
    except Exception:
        return None
    return Gaussianlayer()


class Video_Counter(nn.Module):
    def __init__(self, cfg, cfg_data):
        super().__init__()
        self.cfg = cfg
        self.cfg_data = cfg_data
        self.num_dcfa_stages = int(getattr(cfg, "DCFA_STAGES", 3))
        self.use_dcfa = bool(getattr(cfg, "USE_DCFA", True))
        self.use_dcn = bool(getattr(cfg, "USE_DCN", False))
        self.use_fsg = bool(getattr(cfg, "USE_FSG", False))
        self.use_uphy_enhance = bool(getattr(cfg, "USE_UPHY_ENHANCE", False))
        self.dcn_backend = DCN_BACKEND if (self.use_dcn and HAS_TORCHVISION_DCN) else ("fallback_conv" if self.use_dcn else "disabled")
        self.use_share_branch = bool(getattr(cfg, "USE_SHARE_BRANCH", True))

        if cfg.encoder == "VGG16_FPN":
            self.Extractor = VGG16_FPN_Encoder(pretrained=bool(getattr(cfg, "BACKBONE_PRETRAINED", False)))
        else:
            raise ValueError(f"unsupported backbone: {cfg.encoder}")

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.share_cross_attention = None
        self.share_cross_attention_norm = None
        if self.use_dcfa:
            self.share_cross_attention = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            CrossAttentionBlock(
                                cfg.cross_attn_embed_dim,
                                cfg.cross_attn_num_heads,
                                cfg.mlp_ratio,
                                qkv_bias=True,
                                qk_scale=None,
                                norm_layer=norm_layer,
                            )
                            for _ in range(cfg.cross_attn_depth)
                        ]
                    )
                    for _ in range(self.num_dcfa_stages)
                ]
            )
            self.share_cross_attention_norm = norm_layer(cfg.cross_attn_embed_dim)

        self.image_enhancer = (
            UPhyInspiredEnhancer(hidden_channels=getattr(cfg, "UPHY_ENHANCE_CHANNELS", 32))
            if self.use_uphy_enhance
            else None
        )
        self.feature_fuse = FeatureFusionModule(cfg.FEATURE_DIM) if self.use_dcfa else None
        self.global_refine = DeformableRefineBlock(cfg.FEATURE_DIM) if self.use_dcn else None
        self.share_refine = DeformableRefineBlock(cfg.FEATURE_DIM) if (self.use_dcn and self.use_share_branch) else None
        self.global_gate = ForegroundSuppressionGate(cfg.FEATURE_DIM) if self.use_fsg else None
        self.share_gate = ForegroundSuppressionGate(cfg.FEATURE_DIM) if (self.use_fsg and self.use_share_branch) else None
        self.global_decoder = GlobalDecoder()
        self.share_decoder = ShareDecoder() if self.use_share_branch else None
        self.criterion = nn.MSELoss()
        self.Gaussian = build_gaussian_layer()

    @staticmethod
    def _apply_foreground_prior(feature, foreground_prior):
        if foreground_prior is None:
            return feature

        foreground_prior = F.interpolate(
            foreground_prior,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        foreground_prior = foreground_prior.clamp(0.0, 1.0)
        return feature * (0.35 + 0.65 * foreground_prior)

    def _run_dcfa(self, features):
        if (not self.use_dcfa) or self.share_cross_attention is None or len(features) < 2:
            return features[-1]

        share_features = None
        stage_features = features[: len(self.share_cross_attention)]

        for stage_idx, blocks in enumerate(self.share_cross_attention[: len(stage_features)]):
            current_feature = stage_features[stage_idx]
            batch_size, channels, height, width = current_feature.shape
            img_pair_num = batch_size // 2

            if share_features is not None:
                current_feature = self.feature_fuse(share_features, current_feature)

            share_results = []
            for pair_idx in range(img_pair_num):
                frame0 = current_feature[pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()
                frame1 = current_feature[pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous()

                query0 = frame0
                query1 = frame1
                for block in blocks:
                    query0 = block(query0, frame1)
                    query1 = block(query1, frame0)

                query0 = self.share_cross_attention_norm(query0)
                query1 = self.share_cross_attention_norm(query1)
                share_results.extend([query0, query1])

            share_features = torch.cat(share_results, dim=0)
            share_features = share_features.permute(0, 2, 1).reshape(batch_size, channels, height, width).contiguous()

        return share_features if share_features is not None else features[-1]

    def _predict_density_maps(self, img, return_enhance_aux=False, enhance_fg_masks=None):
        assert img.size(0) % 2 == 0, "DCFA expects image pairs packed as [2B, C, H, W]"

        enhance_aux = None
        foreground_prior = None
        if self.image_enhancer is not None:
            if return_enhance_aux:
                img, enhance_aux, foreground_prior = self.image_enhancer(img, fg_masks=enhance_fg_masks, return_aux=True)
            else:
                img, foreground_prior = self.image_enhancer(img)

        features = self.Extractor(img)
        support_feature = features[min(1, len(features) - 1)]

        global_features = self.global_refine(features[-1]) if self.global_refine is not None else features[-1]
        global_features = self._apply_foreground_prior(global_features, foreground_prior)
        if self.global_gate is not None:
            global_features = self.global_gate(global_features, support_feature)
        pre_global_den = self.global_decoder(global_features)
        pre_share_den = torch.zeros_like(pre_global_den)

        if self.use_share_branch and self.share_decoder is not None:
            share_features = self._run_dcfa(features)
            if self.share_refine is not None:
                share_features = self.share_refine(share_features)
            share_features = self._apply_foreground_prior(share_features, foreground_prior)
            if self.share_gate is not None:
                share_features = self.share_gate(share_features, support_feature)
            pre_share_den = self.share_decoder(share_features)

        return pre_global_den, pre_share_den, enhance_aux

    @staticmethod
    def _scatter_points(dot_map, batch_idx, points):
        if points.numel() == 0:
            return

        _, _, height, width = dot_map.shape
        x = points[:, 0].long().clamp(min=0, max=width - 1)
        y = points[:, 1].long().clamp(min=0, max=height - 1)
        dot_map[batch_idx, 0, y, x] = 1

    def forward(self, img, target):
        if self.Gaussian is None:
            raise RuntimeError("Gaussianlayer is unavailable. Please fix the local SciPy/NumPy environment before training or loss-based evaluation.")

        enhance_fg_masks = None
        if self.image_enhancer is not None:
            fg_masks = [sample.get("fg_mask") for sample in target]
            if all(mask is not None for mask in fg_masks):
                enhance_fg_masks = torch.stack(fg_masks, dim=0)

        pre_global_den, pre_share_den, enhance_aux = self._predict_density_maps(
            img,
            return_enhance_aux=True,
            enhance_fg_masks=enhance_fg_masks,
        )

        gt_global_dot_map = torch.zeros_like(pre_global_den)
        gt_share_dot_map = torch.zeros_like(pre_global_den)

        for batch_idx, sample in enumerate(target):
            points = sample["points"]
            self._scatter_points(gt_global_dot_map, batch_idx, points)

            if points.shape[0] > 0:
                if sample["share_mask"].numel() > 0 and sample["share_mask"].any():
                    self._scatter_points(gt_share_dot_map, batch_idx, points[sample["share_mask"]])

        gt_global_den = self.Gaussian(gt_global_dot_map)
        gt_share_den = self.Gaussian(gt_share_dot_map)

        global_mse_loss = self.criterion(pre_global_den, gt_global_den * self.cfg_data.DEN_FACTOR)
        pred_global_count = pre_global_den.view(pre_global_den.size(0), -1).sum(dim=1) / self.cfg_data.DEN_FACTOR
        gt_global_count = pre_global_den.new_tensor([sample["points"].shape[0] for sample in target])
        global_count_loss = F.smooth_l1_loss(pred_global_count, gt_global_count)
        share_mse_loss = pre_global_den.new_zeros(())
        enhance_loss = pre_global_den.new_zeros(())

        if self.use_share_branch and getattr(self.cfg, "LAMBDA_SHARE", 0.0) > 0:
            share_mse_loss = self.criterion(pre_share_den, gt_share_den * self.cfg_data.DEN_FACTOR)

        if enhance_aux is not None:
            enhance_loss = (
                enhance_aux["color"] * getattr(self.cfg, "LAMBDA_ENH_COLOR", 0.0)
                + enhance_aux["edge"] * getattr(self.cfg, "LAMBDA_ENH_EDGE", 0.0)
                + enhance_aux["tv"] * getattr(self.cfg, "LAMBDA_ENH_TV", 0.0)
            )

        all_loss = {
            "global": global_mse_loss * getattr(self.cfg, "LAMBDA_GLOBAL", 1.0),
            "share": share_mse_loss * getattr(self.cfg, "LAMBDA_SHARE", 1.0),
            "count": global_count_loss * getattr(self.cfg, "LAMBDA_COUNT", 0.0),
            "enhance": enhance_loss,
        }

        pre_global_den = pre_global_den.detach() / self.cfg_data.DEN_FACTOR
        pre_share_den = pre_share_den.detach() / self.cfg_data.DEN_FACTOR

        return (
            pre_global_den,
            gt_global_den,
            pre_share_den,
            gt_share_den,
            all_loss,
        )

    def test_forward(self, img):
        pre_global_den, pre_share_den, _ = self._predict_density_maps(img)
        pre_global_den = pre_global_den.detach() / self.cfg_data.DEN_FACTOR
        pre_share_den = pre_share_den.detach() / self.cfg_data.DEN_FACTOR
        return pre_global_den, pre_share_den
