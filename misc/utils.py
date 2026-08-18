import os
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist


def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** power)
    for param_group in optimizer.param_groups:
        lr_scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = lr * lr_scale
    return lr


def logger(exp_path, exp_name, work_dir, exception, resume=False):
    from tensorboardX import SummaryWriter

    exp_dir = os.path.join(exp_path, exp_name)
    if not resume:
        os.makedirs(exp_dir, exist_ok=True)
        for folder_name in exception:
            os.makedirs(os.path.join(exp_dir, folder_name), exist_ok=True)

    writer = SummaryWriter(exp_dir)
    log_path = os.path.join(exp_dir, "log.txt")
    with open(log_path, "a", encoding="utf-8") as file_obj:
        file_obj.write(f"work_dir: {work_dir}\n")
    return writer, log_path


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class Timer:
    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / max(self.calls, 1)
        return self.average_time if average else self.diff


def reduce_dict(input_dict, average=True):
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []
        for key in sorted(input_dict.keys()):
            names.append(key)
            values.append(input_dict[key])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        return {key: value for key, value in zip(names, values)}


def _density_to_colormap(density_map):
    density_map = density_map.squeeze()
    density_min = density_map.min()
    density_max = density_map.max()
    normalized = (density_map - density_min) / (density_max - density_min + 1e-5)
    normalized = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def save_visual_results(data, restore_transform, save_base, iter_num, rank):
    assert (len(data) - 1) % 2 == 0
    num_pairs = (len(data) - 1) // 2
    height = data[0].size(2)
    width = data[0].size(3)
    batch_size = data[0].size(0)
    margin = 5

    canvas_width = width * len(data) + margin * num_pairs + 3 * margin * num_pairs
    canvas_height = height * batch_size + margin * (batch_size - 1)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    start_h = 0
    for batch_idx in range(batch_size):
        start_w = 0
        image = cv2.cvtColor(np.array(restore_transform(data[0][batch_idx])), cv2.COLOR_RGB2BGR)
        canvas[start_h:start_h + height, start_w:start_w + width] = image
        start_w += width + 3 * margin

        for pair_idx in range(num_pairs):
            gt_map = data[1 + pair_idx * 2][batch_idx].detach().cpu().numpy()
            pred_map = data[1 + pair_idx * 2 + 1][batch_idx].detach().cpu().numpy()
            canvas[start_h:start_h + height, start_w:start_w + width] = _density_to_colormap(gt_map)
            start_w += width + margin
            canvas[start_h:start_h + height, start_w:start_w + width] = _density_to_colormap(pred_map)
            start_w += width + 3 * margin

        start_h += height + margin

    os.makedirs(save_base, exist_ok=True)
    cv2.imwrite(os.path.join(save_base, f"{rank}_{iter_num}_visual.jpg"), canvas)
