#!/usr/bin/env python3
"""
Prepare 512x512 YOLO tiles from the local MIDOG++ WSIs.

Key decisions
-------------
* Use the exact split in Phase 2/images_split (train/val/test) by reading the
  slide ids from the .tiff filenames in each split directory.
* Tile with 20% overlap (paper setting), shifting the last tile so every tile
  is a full 512x512 crop (no black padding).
* Labels come from MIDOGpp.json.  IMPORTANT: the JSON bbox is [x1, y1, x2, y2]
  (not standard COCO [x, y, w, h]); all boxes in this dataset are 50x50.
* Single-class labels: category_id 1 (mitotic) and 2 (non-mitotic hard
  negative) are both written as class 0 = candidate.
* Keep every tile that contains at least one box (center inside the tile).
  For the training split, add a random sample of background tiles.
  For val/test, keep all tiles by default so recall can be measured.
"""

import argparse
import csv
import json
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


def load_coco(coco_path):
    with open(coco_path, "r") as f:
        data = json.load(f)
    images = {int(im["id"]): im for im in data["images"]}
    anns = {}
    for a in data["annotations"]:
        cid = int(a["category_id"])
        if cid not in (1, 2):
            continue
        # MIDOGpp.json uses [x1, y1, x2, y2].
        x1, y1, x2, y2 = [float(v) for v in a["bbox"]]
        anns.setdefault(int(a["image_id"]), []).append((x1, y1, x2, y2, cid))
    return images, anns


def axis_positions(size, tile, stride):
    if size <= tile:
        return [0]
    pos = list(range(0, size - tile + 1, stride))
    if pos[-1] != size - tile:
        pos.append(size - tile)
    return pos


def make_plan(img_info, boxes, tile, stride, min_visible):
    """Return a list of tile plans for one slide, with local YOLO labels.

    Each box is assigned to exactly one tile:
      1. prefer a tile that fully contains the box, choosing the tile whose
         centre is closest to the box centre;
      2. if no tile fully contains it (box touches the image border or is
         larger than the overlap), use the tile containing its centre and clip.
    This avoids duplicate partial labels at internal tile edges.
    """
    width = int(img_info["width"])
    height = int(img_info["height"])
    xs = axis_positions(width, tile, stride)
    ys = axis_positions(height, tile, stride)
    plans = {(x, y): [] for y in ys for x in xs}

    for x1, y1, x2, y2, _cid in boxes:
        bcx = (x1 + x2) * 0.5
        bcy = (y1 + y2) * 0.5

        full_x = [x for x in xs if x <= x1 and x2 <= x + tile]
        full_y = [y for y in ys if y <= y1 and y2 <= y + tile]

        if full_x and full_y:
            # Choose the tile whose centre is nearest to the box centre.
            x = min(full_x, key=lambda t: abs((t + tile * 0.5) - bcx))
            y = min(full_y, key=lambda t: abs((t + tile * 0.5) - bcy))
            ix1, iy1, ix2, iy2 = x1, y1, x2, y2
        else:
            # Border case: tile containing the box centre, clipped.
            cand_x = [x for x in xs if x <= bcx < x + tile]
            cand_y = [y for y in ys if y <= bcy < y + tile]
            if not cand_x or not cand_y:
                continue
            x = min(cand_x, key=lambda t: abs((t + tile * 0.5) - bcx))
            y = min(cand_y, key=lambda t: abs((t + tile * 0.5) - bcy))
            ix1 = max(x1, x)
            iy1 = max(y1, y)
            ix2 = min(x2, x + tile)
            iy2 = min(y2, y + tile)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            area = max(1e-6, (x2 - x1) * (y2 - y1))
            visible = ((ix2 - ix1) * (iy2 - iy1)) / area
            if visible < min_visible:
                continue

        bw = (ix2 - ix1) / tile
        bh = (iy2 - iy1) / tile
        lcx = ((ix1 + ix2) * 0.5 - x) / tile
        lcy = ((iy1 + iy2) * 0.5 - y) / tile
        plans[(x, y)].append((0, lcx, lcy, bw, bh))

    return [{"x": x, "y": y, "labels": labels} for (x, y), labels in plans.items()]


def save_one_slide(task):
    """Worker: crop and write the selected tiles of one slide."""
    (slide, image_path, tiles, out_images, out_labels, debug_dir, jpeg_quality) = task
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    im = Image.open(image_path)
    if im.mode != "RGB":
        im = im.convert("RGB")

    written = 0
    boxes_written = 0
    debug_saved = 0
    manifest = []

    for t in tiles:
        x, y = int(t["x"]), int(t["y"])
        crop = im.crop((x, y, x + 512, y + 512))
        name = "%03d_%05d_%05d" % (slide, x, y)
        img_path = os.path.join(out_images, name + ".jpg")
        lbl_path = os.path.join(out_labels, name + ".txt")
        crop.save(img_path, quality=jpeg_quality, subsampling=0)

        with open(lbl_path, "w") as f:
            for cls, cx, cy, bw, bh in t["labels"]:
                f.write("%d %.6f %.6f %.6f %.6f\n" % (cls, cx, cy, bw, bh))
        written += 1
        boxes_written += len(t["labels"])
        manifest.append((slide, x, y, len(t["labels"]), img_path))

        if debug_dir and t["labels"] and debug_saved < 3:
            vis = crop.copy()
            draw = ImageDraw.Draw(vis)
            for _cls, cx, cy, bw, bh in t["labels"]:
                px = cx * 512
                py = cy * 512
                pw = bw * 512
                ph = bh * 512
                draw.rectangle(
                    [px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2],
                    outline=(255, 0, 0),
                    width=2,
                )
            os.makedirs(debug_dir, exist_ok=True)
            vis.save(os.path.join(debug_dir, name + ".jpg"), quality=95)
            debug_saved += 1

    im.close()
    return {
        "slide": slide,
        "written": written,
        "boxes": boxes_written,
        "manifest": manifest,
    }


def read_split(split_dir):
    files = sorted(Path(split_dir).glob("*.tiff"))
    out = []
    for p in files:
        try:
            slide = int(p.stem)
        except ValueError:
            continue
        out.append((slide, str(p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images-root",
        default="/home/hyw/dataset/chendawei/MIDOG-Mitosis/images_split",
    )
    ap.add_argument(
        "--coco",
        default="/home/hyw/dataset/chendawei/MIDOG++/images/MIDOGpp.json",
    )
    ap.add_argument(
        "--output-root",
        default="/home/hyw/dataset/chendawei/MIDOG-Mitosis/yolo_512_ov20",
    )
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument(
        "--train-neg-ratio",
        type=float,
        default=0.15,
        help="background tiles = ratio * positive tiles in train. -1 keeps all.",
    )
    ap.add_argument(
        "--eval-neg-ratio",
        type=float,
        default=-1.0,
        help="-1 keeps all val/test tiles.",
    )
    ap.add_argument("--min-visible", type=float, default=0.25)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit-slides", type=int, default=0)
    ap.add_argument("--debug-slides", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    stride = max(1, int(round(args.tile * (1.0 - args.overlap))))
    print("tile=%d overlap=%.2f stride=%d" % (args.tile, args.overlap, stride))

    images, anns = load_coco(args.coco)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    summary_path = out_root / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.load(open(summary_path))
        except Exception:
            summary = {}

    for split in splits:
        split_dir = Path(args.images_root) / split
        slides = read_split(split_dir)
        if args.limit_slides and len(slides) > args.limit_slides:
            slides = slides[: args.limit_slides]
        print("[%s] slides=%d" % (split, len(slides)))

        # pass 1: build plans from COCO only (no image decoding)
        per_slide = {}
        positive = []
        negative = []
        missing = []
        for slide, path in slides:
            if slide not in images:
                missing.append(slide)
                continue
            plan = make_plan(
                images[slide], anns.get(slide, []), args.tile, stride, args.min_visible
            )
            pos = [t for t in plan if t["labels"]]
            neg = [t for t in plan if not t["labels"]]
            per_slide[slide] = (path, pos, neg)
            positive.extend((slide, t) for t in pos)
            negative.extend((slide, t) for t in neg)

        if missing:
            print("[%s] WARNING missing in COCO: %s" % (split, missing[:10]))

        neg_ratio = (
            args.train_neg_ratio if split == "train" else args.eval_neg_ratio
        )
        if neg_ratio is None or neg_ratio < 0:
            selected_neg = negative
        else:
            n_neg = min(len(negative), int(round(len(positive) * neg_ratio)))
            selected_neg = random.sample(negative, n_neg)

        print(
            "[%s] positive tiles=%d negative available=%d selected negative=%d total=%d"
            % (
                split,
                len(positive),
                len(negative),
                len(selected_neg),
                len(positive) + len(selected_neg),
            )
        )

        # group selected tiles by slide
        by_slide = {}
        for slide, t in positive + selected_neg:
            by_slide.setdefault(slide, []).append(t)

        out_images = str(out_root / "images" / split)
        out_labels = str(out_root / "labels" / split)
        debug_dir = None
        if args.debug_slides:
            debug_dir = str(out_root / "debug" / split)

        tasks = []
        for slide, (path, _pos, _neg) in per_slide.items():
            tiles = by_slide.get(slide, [])
            if not tiles:
                continue
            tasks.append(
                (
                    slide,
                    path,
                    tiles,
                    out_images,
                    out_labels,
                    debug_dir,
                    args.jpeg_quality,
                )
            )

        total_written = 0
        total_boxes = 0
        manifest_path = out_root / ("manifest_%s.csv" % split)
        with open(manifest_path, "w", newline="") as mf:
            writer = csv.writer(mf)
            writer.writerow(["slide", "x", "y", "num_boxes", "image_path"])
            t0 = time.time()
            # "spawn" avoids fork-after-threads deadlocks when several splits
            # are processed in one invocation.
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=args.workers) as pool:
                for i, res in enumerate(pool.imap_unordered(save_one_slide, tasks), 1):
                    total_written += res["written"]
                    total_boxes += res["boxes"]
                    for row in res["manifest"]:
                        writer.writerow(row)
                    if i % 20 == 0 or i == len(tasks):
                        elapsed = time.time() - t0
                        print(
                            "[%s] slides %d/%d tiles=%d boxes=%d elapsed=%.1fs"
                            % (split, i, len(tasks), total_written, total_boxes, elapsed)
                        )

        summary[split] = {
            "slides": len(slides),
            "positive_tiles": len(positive),
            "negative_available": len(negative),
            "negative_selected": len(selected_neg),
            "tiles_written": total_written,
            "boxes_written": total_boxes,
            "stride": stride,
            "tile": args.tile,
            "overlap": args.overlap,
        }
        print("[%s] done: %s" % (split, summary[split]))

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write("path: %s\n" % out_root)
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("nc: 1\n")
        f.write("names: ['candidate']\n")
    print("wrote", yaml_path)


if __name__ == "__main__":
    main()
