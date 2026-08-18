import glob
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def find_mask_file(mask_dir, image_stem):
    candidates = [
        f"{image_stem}.png",
        f"{image_stem}.jpg",
        f"{image_stem}_mask.png",
        f"{image_stem}_mask.jpg",
    ]
    for candidate in candidates:
        mask_path = os.path.join(mask_dir, candidate)
        if os.path.exists(mask_path):
            return mask_path
    return None


def get_points_from_mask(mask_path):
    if mask_path is None or not os.path.exists(mask_path):
        return np.zeros((0, 2), dtype=np.float32)

    mask = cv2.imread(mask_path, 0)
    if mask is None:
        return np.zeros((0, 2), dtype=np.float32)

    mask = (mask > 0).astype(np.uint8)
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return np.zeros((0, 2), dtype=np.float32)

    return centroids[1:].astype(np.float32)


def load_binary_mask(mask_path, target_size):
    if mask_path is None or not os.path.exists(mask_path):
        return np.zeros((target_size, target_size), dtype=np.uint8)

    mask = cv2.imread(mask_path, 0)
    if mask is None:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    mask = (mask > 0).astype(np.uint8)
    mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return mask


class DeepFishPairDataset(Dataset):
    def __init__(self, root_path, train=True, target_size=512, max_points=512, min_crop_scale=0.75):
        self.root_path = root_path
        self.img_dir = os.path.join(root_path, "images")
        self.mask_dir = os.path.join(root_path, "masks")
        self.train = train
        self.target_size = int(target_size)
        self.max_points = int(max_points)
        self.min_crop_scale = float(min_crop_scale)

        image_paths = glob.glob(os.path.join(self.img_dir, "*.jpg"))
        image_paths += glob.glob(os.path.join(self.img_dir, "*.png"))
        self.img_list = sorted(set(image_paths))

        if len(self.img_list) == 0:
            raise FileNotFoundError(f"no images found under {self.img_dir}")

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)

    def __len__(self):
        return len(self.img_list)

    def _load_base_sample(self, index):
        img_path = self.img_list[index]
        image_name = os.path.basename(img_path)
        image_stem = os.path.splitext(image_name)[0]
        mask_path = find_mask_file(self.mask_dir, image_stem)

        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"failed to read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        fg_mask = load_binary_mask(mask_path, self.target_size)

        points = get_points_from_mask(mask_path)
        if points.size > 0:
            scale = np.array(
                [self.target_size / max(width, 1), self.target_size / max(height, 1)],
                dtype=np.float32,
            )
            points = points * scale
            points[:, 0] = np.clip(points[:, 0], 0, self.target_size - 1)
            points[:, 1] = np.clip(points[:, 1], 0, self.target_size - 1)

        return image, points.astype(np.float32), fg_mask.astype(np.uint8), image_name

    def _sample_view_params(self):
        if not self.train:
            return {
                "x0": 0,
                "y0": 0,
                "crop_size": self.target_size,
                "flip": False,
            }

        scale = random.uniform(self.min_crop_scale, 1.0)
        crop_size = max(64, int(round(self.target_size * scale)))
        max_offset = self.target_size - crop_size
        x0 = random.randint(0, max_offset) if max_offset > 0 else 0
        y0 = random.randint(0, max_offset) if max_offset > 0 else 0

        return {
            "x0": x0,
            "y0": y0,
            "crop_size": crop_size,
            "flip": random.random() < 0.5,
        }

    def _apply_view(self, image, points, fg_mask, params):
        x0 = params["x0"]
        y0 = params["y0"]
        crop_size = params["crop_size"]
        flip = params["flip"]

        crop = image[y0:y0 + crop_size, x0:x0 + crop_size]
        crop_mask = fg_mask[y0:y0 + crop_size, x0:x0 + crop_size]
        if crop.shape[0] != self.target_size or crop.shape[1] != self.target_size:
            crop = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            crop_mask = cv2.resize(crop_mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        if flip:
            crop = np.ascontiguousarray(crop[:, ::-1])
            crop_mask = np.ascontiguousarray(crop_mask[:, ::-1])
        crop_mask = (crop_mask > 0).astype(np.uint8)

        if points.shape[0] == 0:
            return crop, np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.bool_), crop_mask

        inside_x = (points[:, 0] >= x0) & (points[:, 0] < x0 + crop_size)
        inside_y = (points[:, 1] >= y0) & (points[:, 1] < y0 + crop_size)
        visible = inside_x & inside_y

        if not np.any(visible):
            return crop, np.zeros((0, 2), dtype=np.float32), visible, crop_mask

        view_points = points[visible].copy()
        scale = self.target_size / float(crop_size)
        view_points[:, 0] = (view_points[:, 0] - x0) * scale
        view_points[:, 1] = (view_points[:, 1] - y0) * scale

        if flip:
            view_points[:, 0] = (self.target_size - 1) - view_points[:, 0]

        view_points[:, 0] = np.clip(view_points[:, 0], 0, self.target_size - 1)
        view_points[:, 1] = np.clip(view_points[:, 1], 0, self.target_size - 1)

        return crop, view_points.astype(np.float32), visible, crop_mask

    def _image_to_tensor(self, image):
        pil_image = Image.fromarray(image)
        if self.train:
            pil_image = self.color_jitter(pil_image)
        tensor = self.to_tensor(pil_image)
        return self.normalize(tensor)

    def _pad_points(self, points):
        count = min(points.shape[0], self.max_points)
        padded = np.zeros((self.max_points, 2), dtype=np.float32)
        if count > 0:
            padded[:count] = points[:count]
        return torch.from_numpy(padded), torch.tensor(count, dtype=torch.long)

    def _pad_mask(self, mask, count):
        padded = torch.zeros(self.max_points, dtype=torch.bool)
        if count > 0:
            padded[:count] = torch.from_numpy(mask[:count].astype(np.bool_))
        return padded

    @staticmethod
    def _mask_to_tensor(mask):
        return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

    def __getitem__(self, index):
        image, points, fg_mask, image_name = self._load_base_sample(index)

        params1 = self._sample_view_params()
        params2 = self._sample_view_params() if self.train else params1

        view1, points1, visible1, fg_mask1 = self._apply_view(image, points, fg_mask, params1)
        view2, points2, visible2, fg_mask2 = self._apply_view(image, points, fg_mask, params2)

        share_mask1 = visible2[visible1] if visible1.shape[0] > 0 else np.zeros((0,), dtype=np.bool_)
        share_mask2 = visible1[visible2] if visible2.shape[0] > 0 else np.zeros((0,), dtype=np.bool_)

        img1 = self._image_to_tensor(view1)
        img2 = self._image_to_tensor(view2)

        points1_tensor, count1 = self._pad_points(points1)
        points2_tensor, count2 = self._pad_points(points2)

        count1_value = int(count1.item())
        count2_value = int(count2.item())

        sample = {
            "img1": img1,
            "img2": img2,
            "points1": points1_tensor,
            "points2": points2_tensor,
            "count1": count1,
            "count2": count2,
            "share_mask1": self._pad_mask(share_mask1, count1_value),
            "share_mask2": self._pad_mask(share_mask2, count2_value),
            "fg_mask1": self._mask_to_tensor(fg_mask1),
            "fg_mask2": self._mask_to_tensor(fg_mask2),
            "name": image_name,
        }

        return sample


MiniFishDataset = DeepFishPairDataset
