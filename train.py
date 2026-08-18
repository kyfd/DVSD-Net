import csv
import argparse
import os

import torch
from easydict import EasyDict as edict
from torch import optim
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as standard_transforms
from tqdm import tqdm

from config import apply_ablation_profile, cfg, refresh_config
from datasets.fish_dataset import DeepFishPairDataset
from misc import tools
from misc.tools import is_main_process
from misc.utils import *
from model.VIC import Video_Counter


def _extract_state_dict(checkpoint_obj):
    if not isinstance(checkpoint_obj, dict):
        return checkpoint_obj

    for key in ("state_dict", "net", "model"):
        if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
            return checkpoint_obj[key]
    return checkpoint_obj


def _safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train DVSD-Net for DeepFish counting.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--profile", type=str, default=None, choices=cfg.ABLATION_PROFILES)
    parser.add_argument("--encoder", type=str, default=None, choices=cfg.ENCODER_CHOICES)
    parser.add_argument("--backbone-pretrained", type=int, choices=[0, 1], default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--gpu-id", type=str, default=None)
    parser.add_argument("--train-root", type=str, default=None)
    parser.add_argument("--val-root", type=str, default=None)
    parser.add_argument("--pretrain-path", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--start-val", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--min-crop-scale", type=float, default=None)
    parser.add_argument("--cross-attn-depth", type=int, default=None)
    parser.add_argument("--dcfa-stages", type=int, default=None)
    parser.add_argument("--use-dcfa", type=int, choices=[0, 1], default=None)
    parser.add_argument("--use-dcn", type=int, choices=[0, 1], default=None)
    parser.add_argument("--use-fsg", type=int, choices=[0, 1], default=None)
    parser.add_argument("--use-enhance", type=int, choices=[0, 1], default=None)
    parser.add_argument("--use-enhance-mask-supervision", type=int, choices=[0, 1], default=None)
    parser.add_argument("--use-share-branch", type=int, choices=[0, 1], default=None)
    parser.add_argument("--lambda-global", type=float, default=None)
    parser.add_argument("--lambda-share", type=float, default=None)
    parser.add_argument("--lambda-count", type=float, default=None)
    parser.add_argument("--lambda-enh-color", type=float, default=None)
    parser.add_argument("--lambda-enh-edge", type=float, default=None)
    parser.add_argument("--lambda-enh-tv", type=float, default=None)
    parser.add_argument("--enhance-lr-mult", type=float, default=None)
    parser.add_argument("--enhance-warmup-epochs", type=int, default=None)
    return parser


def apply_cli_overrides(config, args):
    manual_profile_override = False

    if args.seed is not None:
        config.SEED = args.seed
    if args.profile is not None:
        apply_ablation_profile(config, args.profile)

    if args.encoder is not None:
        config.encoder = args.encoder
    if args.backbone_pretrained is not None:
        config.BACKBONE_PRETRAINED = bool(args.backbone_pretrained)
    if args.name is not None:
        config.NAME = args.name
    if args.tag is not None:
        config.NAME = f"{config.NAME}_{args.tag}"
    if args.gpu_id is not None:
        config.GPU_ID = args.gpu_id
    if args.train_root is not None:
        config.TRAIN_ROOT = args.train_root
    if args.val_root is not None:
        config.VAL_ROOT = args.val_root
    if args.pretrain_path is not None:
        config.PRE_TRAIN_COUNTER = args.pretrain_path
    if args.lr is not None:
        config.LR_Base = args.lr
    if args.epochs is not None:
        config.MAX_EPOCH = args.epochs
    if args.save_interval is not None:
        config.SAVE_INTERVAL = max(1, int(args.save_interval))
    if args.val_interval is not None:
        config.VAL_INTERVAL = max(1, int(args.val_interval))
    if args.start_val is not None:
        config.START_VAL = max(0, int(args.start_val))
    if args.batch_size is not None:
        config.TRAIN_BATCH_SIZE = args.batch_size
    if args.val_batch_size is not None:
        config.VAL_BATCH_SIZE = args.val_batch_size
    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers
    if args.input_size is not None:
        config.INPUT_SIZE = args.input_size
    if args.max_points is not None:
        config.MAX_POINTS = args.max_points
    if args.min_crop_scale is not None:
        config.MIN_CROP_SCALE = args.min_crop_scale
    if args.cross_attn_depth is not None:
        config.cross_attn_depth = args.cross_attn_depth
    if args.dcfa_stages is not None:
        config.DCFA_STAGES = args.dcfa_stages

    for attr, value in (
        ("USE_DCFA", args.use_dcfa),
        ("USE_DCN", args.use_dcn),
        ("USE_FSG", args.use_fsg),
        ("USE_UPHY_ENHANCE", args.use_enhance),
        ("USE_ENHANCE_MASK_SUPERVISION", args.use_enhance_mask_supervision),
        ("USE_SHARE_BRANCH", args.use_share_branch),
    ):
        if value is not None:
            setattr(config, attr, bool(value))
            manual_profile_override = True

    for attr, value in (
        ("LAMBDA_GLOBAL", args.lambda_global),
        ("LAMBDA_SHARE", args.lambda_share),
        ("LAMBDA_COUNT", args.lambda_count),
        ("LAMBDA_ENH_COLOR", args.lambda_enh_color),
        ("LAMBDA_ENH_EDGE", args.lambda_enh_edge),
        ("LAMBDA_ENH_TV", args.lambda_enh_tv),
        ("ENHANCE_LR_MULT", args.enhance_lr_mult),
        ("ENHANCE_WARMUP_EPOCHS", args.enhance_warmup_epochs),
    ):
        if value is not None:
            setattr(config, attr, value)
            manual_profile_override = True

    if manual_profile_override:
        base_profile = args.profile if args.profile is not None else config.ABLATION_PROFILE
        config.ABLATION_PROFILE = f"custom-{base_profile}"

    if config.LAMBDA_SHARE <= 0:
        config.USE_SHARE_BRANCH = False

    refresh_config(config)


def print_experiment_summary(config):
    summary_lines = [
        "=== experiment summary ===",
        f"exp_name: {config.EXP_NAME}",
        f"seed: {config.SEED}",
        f"profile: {config.ABLATION_PROFILE}",
        f"encoder: {config.encoder}",
        f"backbone_pretrained: {getattr(config, 'BACKBONE_PRETRAINED', False)}",
        f"use_dcfa: {config.USE_DCFA}",
        f"use_dcn: {config.USE_DCN}",
        f"use_fsg: {config.USE_FSG}",
        f"use_enhance: {config.USE_UPHY_ENHANCE}",
        f"use_enhance_mask_supervision: {config.USE_ENHANCE_MASK_SUPERVISION}",
        f"use_share_branch: {config.USE_SHARE_BRANCH}",
        f"shared_feature_source: {'cross_view_dcfa' if config.USE_DCFA else 'single_view_backbone'}",
        f"loss_weights: global={config.LAMBDA_GLOBAL}, share={config.LAMBDA_SHARE}, count={config.LAMBDA_COUNT}",
        f"enhance_loss_weights: color={config.LAMBDA_ENH_COLOR}, edge={config.LAMBDA_ENH_EDGE}, tv={config.LAMBDA_ENH_TV}",
        f"enhance_optimizer: lr_mult={config.ENHANCE_LR_MULT}, warmup_epochs={config.ENHANCE_WARMUP_EPOCHS}",
        f"train_root: {config.TRAIN_ROOT}",
        f"val_root: {config.VAL_ROOT}",
        f"batch_size: train={config.TRAIN_BATCH_SIZE}, val={config.VAL_BATCH_SIZE}",
    ]
    print("\n".join(summary_lines))


class Trainer:
    def __init__(self, cfg_data, pwd):
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd
        self.cfg_data = cfg_data

        self.model = self.model_without_ddp = Video_Counter(cfg, cfg_data)
        self.model.cuda()
        self.val_frame_intervals = cfg_data.VAL_FRAME_INTERVALS

        if getattr(cfg, "distributed", False):
            sync_model = SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                sync_model,
                device_ids=[cfg.gpu],
                find_unused_parameters=False,
            )
            self.model_without_ddp = self.model.module

        train_root = cfg.TRAIN_ROOT
        val_root = cfg.VAL_ROOT

        if not os.path.exists(train_root):
            raise FileNotFoundError(f"training root not found: {train_root}")

        print(f"=== loading training split from {train_root} ===")
        train_dataset = DeepFishPairDataset(
            train_root,
            train=True,
            target_size=cfg.INPUT_SIZE,
            max_points=cfg.MAX_POINTS,
            min_crop_scale=cfg.MIN_CROP_SCALE,
        )
        self.train_sampler = DistributedSampler(train_dataset, shuffle=True) if getattr(cfg, "distributed", False) else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN_BATCH_SIZE,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        if os.path.exists(val_root):
            print(f"=== loading validation split from {val_root} ===")
            val_dataset = DeepFishPairDataset(
                val_root,
                train=False,
                target_size=cfg.INPUT_SIZE,
                max_points=cfg.MAX_POINTS,
                min_crop_scale=cfg.MIN_CROP_SCALE,
            )
            self.val_sampler = DistributedSampler(val_dataset, shuffle=False) if getattr(cfg, "distributed", False) else None
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.VAL_BATCH_SIZE,
                shuffle=False,
                sampler=self.val_sampler,
                num_workers=cfg.NUM_WORKERS,
                pin_memory=True,
                drop_last=False,
            )
        else:
            print(f"warning: validation root not found, skip validation: {val_root}")
            self.val_sampler = None
            self.val_loader = None

        self.restore_transform = standard_transforms.Compose([standard_transforms.ToPILImage()])

        enhancer_params = []
        enhancer_param_ids = set()
        if getattr(cfg, "USE_UPHY_ENHANCE", False) and getattr(self.model_without_ddp, "image_enhancer", None) is not None:
            enhancer_params = [param for param in self.model_without_ddp.image_enhancer.parameters() if param.requires_grad]
            enhancer_param_ids = {id(param) for param in enhancer_params}

        backbone_params = [
            param
            for param in self.model_without_ddp.parameters()
            if param.requires_grad and id(param) not in enhancer_param_ids
        ]
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "weight_decay": cfg.WEIGHT_DECAY, "lr_scale": 1.0})
        if enhancer_params:
            param_groups.append(
                {
                    "params": enhancer_params,
                    "weight_decay": cfg.WEIGHT_DECAY,
                    "lr_scale": float(getattr(cfg, "ENHANCE_LR_MULT", 1.0)),
                }
            )
        self.optimizer = optim.Adam(param_groups, lr=cfg.LR_Base)

        self.i_tb = 0
        self.epoch = 1
        self.num_iters = cfg.MAX_EPOCH * len(self.train_loader)
        self.timer = {"iter time": Timer(), "train time": Timer(), "val time": Timer()}
        self.train_record = {"best_mae": 1e20}
        self.backbone_frozen = False
        self.metrics_fieldnames = [
            "epoch",
            "train_total_loss",
            "train_global_loss",
            "train_share_loss",
            "train_count_loss",
            "train_enhance_loss",
            "val_total_loss",
            "val_global_loss",
            "val_share_loss",
            "val_count_loss",
            "val_enhance_loss",
            "val_mae",
        ]
        self.metrics_csv_path = None

        if cfg.RESUME:
            print(f"resuming from {cfg.RESUME_PATH}")
            latest_state = _safe_torch_load(cfg.RESUME_PATH, map_location="cpu")
            self.model.load_state_dict(latest_state["net"], strict=True)
            self.optimizer.load_state_dict(latest_state["optimizer"])
            self.epoch = latest_state["epoch"]
            self.i_tb = latest_state["i_tb"]

        if cfg.PRE_TRAIN_COUNTER and os.path.exists(cfg.PRE_TRAIN_COUNTER):
            try:
                counting_pre_train = _safe_torch_load(cfg.PRE_TRAIN_COUNTER, map_location="cpu")
                counting_pre_train = _extract_state_dict(counting_pre_train)
                counting_pre_train = {
                    key.replace("module.", ""): value for key, value in counting_pre_train.items()
                }
                model_dict = self.model_without_ddp.state_dict()
                new_dict = {k: v for k, v in counting_pre_train.items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(new_dict)
                self.model_without_ddp.load_state_dict(model_dict, strict=False)
                print(f"loaded {len(new_dict)} pretrained tensors from {cfg.PRE_TRAIN_COUNTER}")
            except Exception as exc:
                print(f"failed to load pretrained counter: {exc}")
        else:
            print(f"pretrained counter not loaded: {cfg.PRE_TRAIN_COUNTER}")

        if is_main_process():
            self.writer, self.log_txt = logger(
                self.exp_path,
                self.exp_name,
                self.pwd,
                ["exp", "eval", "figure", "img", "vis", "output", "visual_results"],
                resume=cfg.RESUME,
            )
            self.metrics_csv_path = os.path.join(self.exp_path, self.exp_name, "metrics.csv")
            self._init_metrics_csv()

    def _init_metrics_csv(self):
        if self.metrics_csv_path is None:
            return
        if cfg.RESUME and os.path.exists(self.metrics_csv_path):
            return

        with open(self.metrics_csv_path, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=self.metrics_fieldnames)
            writer.writeheader()

    def _append_epoch_metrics(self, metrics_row):
        if self.metrics_csv_path is None:
            return

        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=self.metrics_fieldnames)
            writer.writerow(metrics_row)

    def _stack_images(self, batch):
        imgs_stacked = torch.stack([batch["img1"], batch["img2"]], dim=1)
        return imgs_stacked.view(-1, 3, cfg.INPUT_SIZE, cfg.INPUT_SIZE).cuda(non_blocking=True)

    def build_targets_list(self, batch, device):
        target = []
        batch_size_curr = batch["img1"].shape[0]

        for batch_idx in range(batch_size_curr):
            count1 = int(batch["count1"][batch_idx].item())
            count2 = int(batch["count2"][batch_idx].item())

            points1 = batch["points1"][batch_idx, :count1].float().to(device)
            points2 = batch["points2"][batch_idx, :count2].float().to(device)

            share_mask1 = batch["share_mask1"][batch_idx, :count1].to(device)
            share_mask2 = batch["share_mask2"][batch_idx, :count2].to(device)

            target_view1 = {
                "points": points1,
                "share_mask": share_mask1,
                "image_path": str(batch["name"][batch_idx]),
            }
            target_view2 = {
                "points": points2,
                "share_mask": share_mask2,
                "image_path": str(batch["name"][batch_idx]),
            }

            if cfg.USE_UPHY_ENHANCE and cfg.USE_ENHANCE_MASK_SUPERVISION:
                target_view1["fg_mask"] = batch["fg_mask1"][batch_idx].float().to(device)
                target_view2["fg_mask"] = batch["fg_mask2"][batch_idx].float().to(device)

            target.append(target_view1)
            target.append(target_view2)

        return target

    def _set_backbone_trainable(self, trainable):
        extractor = getattr(self.model_without_ddp, "Extractor", None)
        if extractor is None:
            return

        for param in extractor.parameters():
            param.requires_grad = trainable

        self.backbone_frozen = not trainable
        print(f"backbone is now {'frozen' if self.backbone_frozen else 'unfrozen'}")

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH + 1):
            self.epoch = epoch

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            if getattr(cfg, "USE_UPHY_ENHANCE", False):
                warmup_epochs = int(getattr(cfg, "ENHANCE_WARMUP_EPOCHS", 0))
                should_freeze_backbone = epoch <= warmup_epochs
                if should_freeze_backbone != self.backbone_frozen:
                    self._set_backbone_trainable(not should_freeze_backbone)

            self.timer["train time"].tic()
            train_stats = self.train()
            self.timer["train time"].toc(average=False)
            print(f"train time: {self.timer['train time'].diff:.2f}s")

            save_folder = os.path.join(self.exp_path, self.exp_name, "output")
            os.makedirs(save_folder, exist_ok=True)

            save_interval = max(1, int(getattr(cfg, "SAVE_INTERVAL", 1)))
            if epoch % save_interval == 0 or epoch == cfg.MAX_EPOCH:
                save_path = os.path.join(save_folder, f"epoch_{epoch}.pth")
                torch.save(self.model_without_ddp.state_dict(), save_path)
                print(f"saved checkpoint to {save_path}")
            else:
                print(f"skipped periodic checkpoint at epoch {epoch} (save_interval={save_interval})")

            val_stats = {
                "mae": "",
                "total": "",
                "global": "",
                "share": "",
                "count": "",
                "enhance": "",
            }
            if self.val_loader is not None and epoch >= cfg.START_VAL and epoch % cfg.VAL_INTERVAL == 0:
                val_stats = self.validate()
                if val_stats["mae"] < self.train_record["best_mae"]:
                    self.train_record["best_mae"] = val_stats["mae"]
                    best_path = os.path.join(save_folder, "best_model.pth")
                    torch.save(self.model_without_ddp.state_dict(), best_path)
                    print(f"new best model saved to {best_path} (MAE {val_stats['mae']:.2f})")

            if is_main_process():
                self.writer.add_scalar("epoch/train_total_loss", train_stats["total"], epoch)
                self.writer.add_scalar("epoch/train_global_loss", train_stats["global"], epoch)
                self.writer.add_scalar("epoch/train_share_loss", train_stats["share"], epoch)
                self.writer.add_scalar("epoch/train_count_loss", train_stats["count"], epoch)
                self.writer.add_scalar("epoch/train_enhance_loss", train_stats["enhance"], epoch)

                if val_stats["mae"] != "":
                    self.writer.add_scalar("epoch/val_mae", val_stats["mae"], epoch)
                    self.writer.add_scalar("epoch/val_total_loss", val_stats["total"], epoch)
                    self.writer.add_scalar("epoch/val_global_loss", val_stats["global"], epoch)
                    self.writer.add_scalar("epoch/val_share_loss", val_stats["share"], epoch)
                    self.writer.add_scalar("epoch/val_count_loss", val_stats["count"], epoch)
                    self.writer.add_scalar("epoch/val_enhance_loss", val_stats["enhance"], epoch)

                self._append_epoch_metrics(
                    {
                        "epoch": epoch,
                        "train_total_loss": train_stats["total"],
                        "train_global_loss": train_stats["global"],
                        "train_share_loss": train_stats["share"],
                        "train_count_loss": train_stats["count"],
                        "train_enhance_loss": train_stats["enhance"],
                        "val_total_loss": val_stats["total"],
                        "val_global_loss": val_stats["global"],
                        "val_share_loss": val_stats["share"],
                        "val_count_loss": val_stats["count"],
                        "val_enhance_loss": val_stats["enhance"],
                        "val_mae": val_stats["mae"],
                    }
                )

            print("=" * 20)

    def train(self):
        self.model.train()
        if self.backbone_frozen:
            self.model_without_ddp.Extractor.eval()
        batch_loss = {}
        total_loss_meter = AverageMeter()

        for _, batch in enumerate(self.train_loader):
            self.i_tb += 1
            lr = adjust_learning_rate(self.optimizer, cfg.LR_Base, self.num_iters, self.i_tb)

            img = self._stack_images(batch)
            target = self.build_targets_list(batch, img.device)

            pre_global_den, gt_global_den, pre_share_den, gt_share_den, loss_dict = self.model(
                img,
                target,
            )

            pre_global_den = torch.relu(pre_global_den)
            pre_global_cnt = pre_global_den.view(pre_global_den.shape[0], -1).sum(dim=1).mean()
            gt_counts = torch.tensor(
                [sample["points"].shape[0] for sample in target],
                device=img.device,
                dtype=pre_global_cnt.dtype,
            )
            gt_global_cnt = gt_counts.mean()

            all_loss = sum(loss_dict.values())
            if not torch.isfinite(all_loss):
                print(
                    f"warning: non-finite loss detected at epoch {self.epoch}, iter {self.i_tb}; "
                    "skipping this batch."
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.optimizer.zero_grad()
            all_loss.backward()
            if getattr(cfg, "USE_UPHY_ENHANCE", False):
                torch.nn.utils.clip_grad_norm_(self.model_without_ddp.parameters(), max_norm=5.0)
            self.optimizer.step()

            loss_dict_reduced = reduce_dict(loss_dict)
            reduced_total_loss = sum(value.item() for value in loss_dict_reduced.values())
            total_loss_meter.update(reduced_total_loss)
            for key, value in loss_dict_reduced.items():
                if key not in batch_loss:
                    batch_loss[key] = AverageMeter()
                batch_loss[key].update(value.item())

            if self.i_tb % cfg.PRINT_FREQ == 0 and is_main_process():
                self.writer.add_scalar("lr", lr, self.i_tb)
                self.writer.add_scalar("train_total", reduced_total_loss, self.i_tb)
                for key, value in loss_dict_reduced.items():
                    self.writer.add_scalar(key, value.item(), self.i_tb)

                loss_str = f"[total {total_loss_meter.avg:.4f}]" + "".join([f"[{key} {meter.avg:.4f}]" for key, meter in batch_loss.items()])
                print(
                    f"[ep {self.epoch}][it {self.i_tb}]"
                    f"{loss_str} [gt: {gt_global_cnt.item():.2f} pred: {pre_global_cnt.item():.2f}]"
                )

            if self.i_tb % 50 == 0 and is_main_process():
                try:
                    save_visual_results(
                        [
                            img.detach(),
                            gt_global_den.detach(),
                            pre_global_den.detach(),
                            gt_share_den.detach(),
                            pre_share_den.detach(),
                        ],
                        self.restore_transform,
                        os.path.join(self.exp_path, self.exp_name, "training_visual"),
                        self.i_tb,
                        0,
                    )
                except Exception as exc:
                    print(f"failed to save visualization: {exc}")

        return {
            "total": total_loss_meter.avg,
            "global": batch_loss["global"].avg if "global" in batch_loss else 0.0,
            "share": batch_loss["share"].avg if "share" in batch_loss else 0.0,
            "count": batch_loss["count"].avg if "count" in batch_loss else 0.0,
            "enhance": batch_loss["enhance"].avg if "enhance" in batch_loss else 0.0,
        }

    def validate(self):
        self.model.eval()
        print(">>> running validation...")

        mae_sum = 0.0
        sample_count = 0
        batch_count = 0
        val_loss_meter = AverageMeter()
        val_loss_components = {}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, disable=not is_main_process()):
                img = self._stack_images(batch)
                target = self.build_targets_list(batch, img.device)

                outputs = self.model(img, target)
                pre_global = torch.relu(outputs[0])
                loss_dict = reduce_dict(outputs[-1])
                if not torch.isfinite(pre_global).all():
                    print("warning: non-finite validation prediction detected, abort validation early.")
                    return {
                        "mae": float("inf"),
                        "total": float("inf"),
                        "global": float("inf"),
                        "share": float("inf"),
                        "count": float("inf"),
                        "enhance": float("inf"),
                    }
                pred_counts = pre_global.view(pre_global.shape[0], -1).sum(dim=1)

                gt_counts = torch.tensor(
                    [sample["points"].shape[0] for sample in target],
                    device=img.device,
                    dtype=pred_counts.dtype,
                )

                mae_sum += torch.abs(pred_counts - gt_counts).sum().item()
                sample_count += gt_counts.numel()
                batch_count += 1

                reduced_total_loss = sum(value.item() for value in loss_dict.values())
                val_loss_meter.update(reduced_total_loss)
                for key, value in loss_dict.items():
                    if key not in val_loss_components:
                        val_loss_components[key] = AverageMeter()
                    val_loss_components[key].update(value.item())

        if getattr(cfg, "distributed", False) and torch.distributed.is_initialized():
            metrics = torch.tensor([mae_sum, float(sample_count)], device="cuda")
            torch.distributed.all_reduce(metrics)
            mae_sum = metrics[0].item()
            sample_count = int(metrics[1].item())

        mae = mae_sum / max(sample_count, 1)
        val_total_loss = val_loss_meter.avg if batch_count > 0 else 0.0
        print(f"validation finished | MAE: {mae:.2f} | loss: {val_total_loss:.4f}")
        return {
            "mae": mae,
            "total": val_total_loss,
            "global": val_loss_components["global"].avg if "global" in val_loss_components else 0.0,
            "share": val_loss_components["share"].avg if "share" in val_loss_components else 0.0,
            "count": val_loss_components["count"].avg if "count" in val_loss_components else 0.0,
            "enhance": val_loss_components["enhance"].avg if "enhance" in val_loss_components else 0.0,
        }


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    apply_cli_overrides(cfg, args)
    print_experiment_summary(cfg)

    tools.init_distributed_mode(cfg)
    tools.set_randomseed(cfg.SEED + tools.get_rank())
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    cfg_data = edict()
    cfg_data.DEN_FACTOR = 100
    cfg_data.VAL_FRAME_INTERVALS = 1

    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Trainer(cfg_data, pwd)
    cc_trainer.forward()
