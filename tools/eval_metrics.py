import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import apply_ablation_profile, cfg, refresh_config
from datasets.fish_dataset import DeepFishPairDataset
from model.VIC import Video_Counter


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate DeepFish model with MAE, MSE, RMSE, and GAME.")
    parser.add_argument("--test-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--encoder", type=str, default=None, choices=cfg.ENCODER_CHOICES)
    parser.add_argument("--backbone-pretrained", type=str2bool, default=None)
    parser.add_argument("--profile", type=str, default=None, choices=cfg.ABLATION_PROFILES)
    parser.add_argument("--use-dcfa", type=str2bool, default=None)
    parser.add_argument("--use-dcn", type=str2bool, default=None)
    parser.add_argument("--use-fsg", type=str2bool, default=None)
    parser.add_argument("--use-enhance", type=str2bool, default=None)
    parser.add_argument("--use-share-branch", type=str2bool, default=None)
    parser.add_argument("--lambda-share", type=float, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--game-levels", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--tta", type=str, default="none", choices=["none", "flip"])
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def apply_overrides(args):
    if args.profile is not None:
        apply_ablation_profile(cfg, args.profile)

    if args.encoder is not None:
        cfg.encoder = args.encoder
    if args.backbone_pretrained is not None:
        cfg.BACKBONE_PRETRAINED = args.backbone_pretrained
    if args.input_size is not None:
        cfg.INPUT_SIZE = args.input_size
    if args.use_dcfa is not None:
        cfg.USE_DCFA = args.use_dcfa
    if args.use_dcn is not None:
        cfg.USE_DCN = args.use_dcn
    if args.use_fsg is not None:
        cfg.USE_FSG = args.use_fsg
    if args.use_enhance is not None:
        cfg.USE_UPHY_ENHANCE = args.use_enhance
    if args.use_share_branch is not None:
        cfg.USE_SHARE_BRANCH = args.use_share_branch
    if args.lambda_share is not None:
        cfg.LAMBDA_SHARE = args.lambda_share

    if not cfg.USE_DCFA:
        cfg.USE_SHARE_BRANCH = False
    if cfg.LAMBDA_SHARE <= 0:
        cfg.USE_SHARE_BRANCH = False

    refresh_config(cfg)


def make_gt_count_map(points, map_size):
    gt_map = np.zeros((map_size, map_size), dtype=np.float32)
    if points.numel() == 0:
        return gt_map

    x = points[:, 0].long().clamp(min=0, max=map_size - 1)
    y = points[:, 1].long().clamp(min=0, max=map_size - 1)
    gt_map[y.cpu().numpy(), x.cpu().numpy()] += 1.0
    return gt_map


def game_score(pred_map, gt_map, level):
    grid_size = 2 ** level
    pred_splits = np.array_split(pred_map, grid_size, axis=0)
    gt_splits = np.array_split(gt_map, grid_size, axis=0)

    total_error = 0.0
    for pred_row, gt_row in zip(pred_splits, gt_splits):
        pred_cells = np.array_split(pred_row, grid_size, axis=1)
        gt_cells = np.array_split(gt_row, grid_size, axis=1)
        for pred_cell, gt_cell in zip(pred_cells, gt_cells):
            total_error += abs(float(pred_cell.sum()) - float(gt_cell.sum()))
    return total_error


def predict_with_tta(model, img, tta_mode):
    pred_global, _ = model.test_forward(img)
    pred_global = torch.relu(pred_global)

    if tta_mode == "none":
        return pred_global

    if tta_mode == "flip":
        flipped_img = torch.flip(img, dims=[3])
        pred_global_flip, _ = model.test_forward(flipped_img)
        pred_global_flip = torch.relu(pred_global_flip)
        pred_global_flip = torch.flip(pred_global_flip, dims=[3])
        return 0.5 * (pred_global + pred_global_flip)

    raise ValueError(f"unsupported tta mode: {tta_mode}")


def evaluate(args):
    apply_overrides(args)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    cfg_data = edict()
    cfg_data.DEN_FACTOR = 100
    cfg_data.VAL_FRAME_INTERVALS = 1

    model = Video_Counter(cfg, cfg_data).to(device)
    state_dict = safe_torch_load(args.model_path, map_location=device)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dataset = DeepFishPairDataset(
        args.test_root,
        train=False,
        target_size=cfg.INPUT_SIZE,
        max_points=cfg.MAX_POINTS,
        min_crop_scale=cfg.MIN_CROP_SCALE,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    mae_sum = 0.0
    mse_sum = 0.0
    sample_count = 0
    game_sums = {level: 0.0 for level in args.game_levels}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            imgs_stacked = torch.stack([batch["img1"], batch["img2"]], dim=1)
            img = imgs_stacked.view(-1, 3, cfg.INPUT_SIZE, cfg.INPUT_SIZE).to(device, non_blocking=True)
            pred_maps = predict_with_tta(model, img, args.tta).cpu().numpy()

            batch_size = batch["img1"].shape[0]
            for batch_idx in range(batch_size):
                pred_map = pred_maps[batch_idx * 2, 0]
                count1 = int(batch["count1"][batch_idx].item())
                gt_points = batch["points1"][batch_idx, :count1].float()
                gt_count = float(gt_points.shape[0])
                pred_count = float(pred_map.sum())

                abs_err = abs(pred_count - gt_count)
                mae_sum += abs_err
                mse_sum += (pred_count - gt_count) ** 2
                sample_count += 1

                gt_map = make_gt_count_map(gt_points, pred_map.shape[0])
                for level in args.game_levels:
                    game_sums[level] += game_score(pred_map, gt_map, level)

    results = {
        "model_path": args.model_path,
        "test_root": args.test_root,
        "encoder": cfg.encoder,
        "backbone_pretrained": getattr(cfg, "BACKBONE_PRETRAINED", False),
        "profile": cfg.ABLATION_PROFILE,
        "use_dcn": cfg.USE_DCN,
        "use_fsg": cfg.USE_FSG,
        "use_enhance": cfg.USE_UPHY_ENHANCE,
        "tta": args.tta,
        "samples": sample_count,
        "mae": mae_sum / max(sample_count, 1),
        "mse": mse_sum / max(sample_count, 1),
        "rmse": math.sqrt(mse_sum / max(sample_count, 1)),
        "game": {f"GAME{level}": game_sums[level] / max(sample_count, 1) for level in args.game_levels},
    }

    if not getattr(args, "quiet", False):
        print("=== evaluation results ===")
        print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.save_json is not None:
        with open(args.save_json, "w", encoding="utf-8") as file_obj:
            json.dump(results, file_obj, indent=2, ensure_ascii=False)
        if not getattr(args, "quiet", False):
            print(f"saved metrics to {args.save_json}")

    return results


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
