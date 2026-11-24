#!/usr/bin/env python3
"""Run the stage-1 YOLO detector on a WSI and optionally save 224 crops."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None


def axis_positions(size, tile, stride):
    if size <= tile:
        return [0]
    pos = list(range(0, size - tile + 1, stride))
    if pos[-1] != size - tile:
        pos.append(size - tile)
    return pos


def suppress_by_center(boxes, scores, center_dist, iou_threshold=0.3):
    """Greedy suppression for boxes with almost the same centre."""
    if len(boxes) == 0 or center_dist is None:
        return boxes, scores
    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]
    keep = []
    for i in range(len(boxes)):
        if not keep:
            keep.append(i)
            continue
        cur = boxes[i]
        k = np.array(keep)
        xx1 = np.maximum(cur[0], boxes[k, 0])
        yy1 = np.maximum(cur[1], boxes[k, 1])
        xx2 = np.minimum(cur[2], boxes[k, 2])
        yy2 = np.minimum(cur[3], boxes[k, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_cur = max(0, cur[2] - cur[0]) * max(0, cur[3] - cur[1])
        area_other = (
            np.maximum(0, boxes[k, 2] - boxes[k, 0])
            * np.maximum(0, boxes[k, 3] - boxes[k, 1])
        )
        union = area_cur + area_other - inter
        iou = np.where(union > 0, inter / union, 0.0)
        centers_cur = np.array([(cur[0] + cur[2]) * 0.5, (cur[1] + cur[3]) * 0.5])
        centers_other = np.stack(
            [
                (boxes[k, 0] + boxes[k, 2]) * 0.5,
                (boxes[k, 1] + boxes[k, 3]) * 0.5,
            ],
            axis=1,
        )
        dist = np.sqrt(((centers_other - centers_cur) ** 2).sum(axis=1))
        if not np.any((dist < center_dist) & (iou > iou_threshold)):
            keep.append(i)
    keep = np.array(keep, dtype=np.int64)
    return boxes[keep], scores[keep]


def detect_wsi(
    model,
    image_path,
    tile=512,
    overlap=0.20,
    conf=0.01,
    tile_iou=0.3,
    max_det=300,
    batch=16,
    device=0,
    global_iou=0.5,
    center_dist=None,
):
    """Return (boxes_xyxy_global, scores, (width, height))."""
    stride = max(1, int(round(tile * (1.0 - overlap))))
    im = Image.open(image_path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    width, height = im.size
    xs = axis_positions(width, tile, stride)
    ys = axis_positions(height, tile, stride)

    raw_boxes = []
    raw_scores = []
    coords = [(x, y) for y in ys for x in xs]

    for start in range(0, len(coords), batch):
        chunk = coords[start : start + batch]
        crops = [im.crop((x, y, x + tile, y + tile)) for x, y in chunk]
        results = model.predict(
            crops,
            imgsz=tile,
            conf=conf,
            iou=tile_iou,
            max_det=max_det,
            device=device,
            verbose=False,
        )
        for res, (x, y) in zip(results, chunk):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            b = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            s = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
            b[:, [0, 2]] += x
            b[:, [1, 3]] += y
            b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, width)
            b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, height)
            raw_boxes.append(b)
            raw_scores.append(s)
    im.close()

    if not raw_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            (width, height),
        )

    boxes = np.concatenate(raw_boxes, axis=0)
    scores = np.concatenate(raw_scores, axis=0)
    keep = nms(
        torch.as_tensor(boxes, dtype=torch.float32),
        torch.as_tensor(scores, dtype=torch.float32),
        global_iou,
    ).numpy()
    boxes, scores = boxes[keep], scores[keep]
    boxes, scores = suppress_by_center(boxes, scores, center_dist)
    return boxes, scores, (width, height)


def crop_224(im, box, size=224):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    left = int(round(cx - size / 2))
    top = int(round(cy - size / 2))
    right = left + size
    bottom = top + size
    W, H = im.size
    if left < 0:
        left, right = 0, size
    if top < 0:
        top, bottom = 0, size
    if right > W:
        right, left = W, W - size
    if bottom > H:
        bottom, top = H, H - size
    return im.crop((left, top, right, bottom)), (left, top, right, bottom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--tile-iou", type=float, default=0.3)
    ap.add_argument("--global-iou", type=float, default=0.5)
    ap.add_argument("--center-dist", type=float, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--save-patches", action="store_true")
    ap.add_argument("--patch-dir", default=None)
    args = ap.parse_args()

    model = YOLO(args.model)
    boxes, scores, (W, H) = detect_wsi(
        model,
        args.image,
        tile=args.tile,
        overlap=args.overlap,
        conf=args.conf,
        tile_iou=args.tile_iou,
        global_iou=args.global_iou,
        center_dist=args.center_dist,
        batch=args.batch,
        device=args.device,
    )

    out = {
        "image": os.path.basename(args.image),
        "width": W,
        "height": H,
        "conf": args.conf,
        "candidates": [],
    }
    for b, s in zip(boxes, scores):
        out["candidates"].append(
            {
                "x1": float(b[0]),
                "y1": float(b[1]),
                "x2": float(b[2]),
                "y2": float(b[3]),
                "cx": float((b[0] + b[2]) / 2),
                "cy": float((b[1] + b[3]) / 2),
                "score": float(s),
            }
        )

    if args.save_patches:
        patch_dir = args.patch_dir or (
            os.path.splitext(args.output or args.image)[0] + "_patches"
        )
        os.makedirs(patch_dir, exist_ok=True)
        im = Image.open(args.image)
        if im.mode != "RGB":
            im = im.convert("RGB")
        meta = []
        for i, (b, s) in enumerate(zip(boxes, scores)):
            patch, coords = crop_224(im, b, 224)
            name = "%s_cand%04d.tif" % (Path(args.image).stem, i)
            patch.save(os.path.join(patch_dir, name))
            meta.append(
                {
                    "patch_name": name,
                    "score": float(s),
                    "original_bbox": [float(v) for v in b],
                    "patch_coords": list(coords),
                }
            )
        im.close()
        with open(os.path.join(patch_dir, "candidate_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        out["patch_dir"] = patch_dir

    output = args.output or (os.path.splitext(args.image)[0] + "_candidates.json")
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote %s (%d candidates)" % (output, len(boxes)))


if __name__ == "__main__":
    main()
