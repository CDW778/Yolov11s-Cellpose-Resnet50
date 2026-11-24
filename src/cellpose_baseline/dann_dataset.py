#!/usr/bin/env python3
"""Dataset for the clean DANN-ResNet50 baseline (3-channel RGB crops).

Each Cellpose mask becomes one fixed-size 224x224 RGB window. Windows are
centered on the mask bounding box (default) or on the mask centroid. Pixels
are ImageNet-normalized.
"""
import numpy as np
import cv2
import torch
from pathlib import Path
from scipy import ndimage
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH = 224
_CENT_CACHE = {}


def _bounds(cx, cy, size, H, W):
    half = size // 2
    l = max(0, cx - half); t = max(0, cy - half)
    r = l + size; b = t + size
    if r > W:
        r = W; l = r - size
    if b > H:
        b = H; t = b - size
    l = max(0, l); t = max(0, t)
    return l, t, r, b


def crop_patch(img_rgb, bbox, size=PATCH):
    """Crop a size x size window centered on bbox=(x0,y0,x1,y1)."""
    H, W = img_rgb.shape[:2]
    left, top, right, bottom = bbox
    cx = int((left + right) / 2)
    cy = int((top + bottom) / 2)
    l, t, r, b = _bounds(cx, cy, size, H, W)
    patch = img_rgb[t:b, l:r]
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_CUBIC)
    return patch


def crop_at(arr, cx, cy, size=PATCH, interp=cv2.INTER_LINEAR):
    """Crop a size x size window centered on (cx, cy)."""
    H, W = arr.shape[:2]
    l, t, r, b = _bounds(int(round(cx)), int(round(cy)), size, H, W)
    patch = arr[t:b, l:r]
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size), interpolation=interp)
    return patch, (l, t, r, b)


def light_augment(p):
    if np.random.rand() < 0.5:
        p = p[:, ::-1]
    if np.random.rand() < 0.5:
        p = p[::-1, :]
    k = np.random.randint(4)
    if k:
        p = np.rot90(p, k)
    return np.ascontiguousarray(p)


class CandidateDataset(Dataset):
    def __init__(self, split, cand, index, img_root, augment=False,
                 crop_size=PATCH, crop_center='bbox', mask_dir=None):
        self.split = split
        self.base = cand['base']
        self.label = cand['label']
        self.x0 = cand['x0']; self.y0 = cand['y0']
        self.x1 = cand['x1']; self.y1 = cand['y1']
        self.y = cand['y']
        self.index = np.asarray(index)
        self.img_root = Path(img_root)
        self.augment = augment
        self.crop_size = crop_size
        self.crop_center = crop_center
        self.mask_dir = Path(mask_dir) if mask_dir else None
        if crop_center == 'centroid' and self.mask_dir is None:
            raise ValueError('crop_center=centroid 需要提供 mask_dir')
        self._cache = {}
        self._mask_cache = {}
        self._cent_cache = {}
        if crop_center == 'centroid':
            for b in np.unique(self.base[self.index]):
                self._centroids(str(b))
            self._mask_cache.clear()

    def _img(self, base):
        if base not in self._cache:
            bgr = cv2.imread(str(self.img_root / f"{base}.bmp"))
            self._cache[base] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._cache[base]

    def _mask(self, base):
        if base not in self._mask_cache:
            self._mask_cache[base] = np.load(self.mask_dir / f"{base}_mask.npy")
        return self._mask_cache[base]

    def _centroids(self, base):
        key = (str(self.mask_dir), base)
        if key not in _CENT_CACHE:
            mask = self._mask(base)
            labels = np.unique(mask)
            labels = labels[labels != 0]
            cents = ndimage.center_of_mass(np.ones(mask.shape, np.float32), mask, labels)
            _CENT_CACHE[key] = {int(l): (float(c[0]), float(c[1]))
                                for l, c in zip(labels, cents)}
        return _CENT_CACHE[key]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = int(self.index[i])
        base = str(self.base[j])
        img = self._img(base)
        size = self.crop_size

        if self.crop_center == 'centroid':
            cent = self._centroids(base).get(int(self.label[j]))
            if cent is not None:
                cy, cx = cent
            else:
                cx = (int(self.x0[j]) + int(self.x1[j])) / 2.0
                cy = (int(self.y0[j]) + int(self.y1[j])) / 2.0
            p, _ = crop_at(img, cx, cy, size, cv2.INTER_CUBIC)
        else:
            bbox = (int(self.x0[j]), int(self.y0[j]), int(self.x1[j]), int(self.y1[j]))
            p = crop_patch(img, bbox, size)

        if self.augment:
            p = light_augment(p)

        x = p.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float()
        return x, int(self.y[j])
