#!/usr/bin/env python3
"""Prepare 512x512 YOLO tiles from ICPR2012 HPF images.

Key decisions
-------------
* Reuse the split used by the Cellpose/ICPR2012 baseline project:
    train = A/H 00-02, val = A/H 03, test = A/H 04.
* GT comes from the contest CSV files: one line per mitotic figure, stored as
  polygon vertex pairs ``x1,y1,x2,y2,...``.  We convert each polygon to its
  bounding box.
* Single-class labels: class 0 = candidate.
* Tile with 20% overlap (same as the MIDOG stage-1 pipeline).  2084x2084 images
  give a 5x5 grid of full 512x512 tiles (last row/col shifted, never padded).
* Keep every tile for train and val (no background sampling): ICPR2012 is tiny
  (train has only ~175 tiles containing a mitosis), so the background tiles are
  valuable.  Test is never tiled; it is evaluated on the full image.
"""

import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.paths import IMAGES_ROOT, RAW_ROOT, YOLO_V1  # noqa: E402
from prepare_tiles import axis_positions, make_plan  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

SRC_ANNO = RAW_ROOT
IMG_ROOT = IMAGES_ROOT
DEFAULT_OUT = YOLO_V1

SPLIT_PREFIXES = {
    "train": {"A00", "A01", "A02", "H00", "H01", "H02"},
    "val": {"A03", "H03"},
    "test": {"A04", "H04"},
}


def index_csvs(src_anno):
    """Map slide base name (e.g. A00_01) -> csv path."""
    out = {}
    for d in Path(src_anno).glob("*_v2"):
        for p in d.glob("*.csv"):
            out[p.stem] = p
    return out


def load_gt_boxes(csv_path):
    """Read polygon lines -> list of (x1, y1, x2, y2)."""
    boxes = []
    for line in open(csv_path):
        line = line.strip()
        if not line:
            continue
        vals = np.array([int(t) for t in line.split(",") if t != ""], dtype=np.int64)
        xs, ys = vals[0::2], vals[1::2]
        if len(xs) == 0:
            continue
        boxes.append((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))
    return boxes


def save_one_slide(task):
    (base, image_path, tiles, out_images, out_labels, debug_dir, jpeg_quality) = task
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
        name = "%s_%05d_%05d" % (base, x, y)
        img_path = os.path.join(out_images, name + ".jpg")
        lbl_path = os.path.join(out_labels, name + ".txt")
        crop.save(img_path, quality=jpeg_quality, subsampling=0)

        with open(lbl_path, "w") as f:
            for cls, cx, cy, bw, bh in t["labels"]:
                f.write("%d %.6f %.6f %.6f %.6f\n" % (cls, cx, cy, bw, bh))
        written += 1
        boxes_written += len(t["labels"])
        manifest.append((base, x, y, len(t["labels"]), img_path))

        if debug_dir and t["labels"] and debug_saved < 3:
            vis = crop.copy()
            draw = ImageDraw.Draw(vis)
            for _cls, cx, cy, bw, bh in t["labels"]:
                px, py = cx * 512, cy * 512
                pw, ph = bw * 512, bh * 512
                draw.rectangle(
                    [px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2],
                    outline=(255, 0, 0),
                    width=2,
                )
            os.makedirs(debug_dir, exist_ok=True)
            vis.save(os.path.join(debug_dir, name + ".jpg"), quality=95)
            debug_saved += 1

    im.close()
    return {"slide": base, "written": written, "boxes": boxes_written, "manifest": manifest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-anno", default=str(SRC_ANNO))
    ap.add_argument("--images-root", default=str(IMG_ROOT))
    ap.add_argument("--output-root", default=str(DEFAULT_OUT))
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--min-visible", type=float, default=0.25)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--debug-slides", type=int, default=0)
    args = ap.parse_args()

    csvs = index_csvs(args.src_anno)
    print("indexed csv files: %d" % len(csvs))
    stride = max(1, int(round(args.tile * (1.0 - args.overlap))))
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        prefixes = SPLIT_PREFIXES[split]
        img_dir = Path(args.images_root) / split
        bmps = sorted(img_dir.glob("*.bmp"))
        bmps = [p for p in bmps if p.stem[:3] in prefixes]
        print("[%s] slides=%d" % (split, len(bmps)))

        tasks = []
        positive = 0
        n_boxes = 0
        missing = []
        for bmp in bmps:
            base = bmp.stem
            if base not in csvs:
                missing.append(base)
                continue
            with Image.open(bmp) as im:
                w, h = im.size
            boxes = load_gt_boxes(csvs[base])
            n_boxes += len(boxes)
            plan = make_plan(
                {"width": w, "height": h},
                [(x1, y1, x2, y2, 1) for x1, y1, x2, y2 in boxes],
                args.tile,
                stride,
                args.min_visible,
            )
            positive += sum(1 for t in plan if t["labels"])
            tasks.append(
                (
                    base,
                    str(bmp),
                    plan,
                    str(out_root / "images" / split),
                    str(out_root / "labels" / split),
                    str(out_root / "debug" / split) if args.debug_slides else None,
                    args.jpeg_quality,
                )
            )
        if missing:
            print("[%s] WARNING missing CSV: %s" % (split, missing[:10]))

        total_written = 0
        total_boxes = 0
        manifest_path = out_root / ("manifest_%s.csv" % split)
        with open(manifest_path, "w", newline="") as mf:
            writer = csv.writer(mf)
            writer.writerow(["slide", "x", "y", "num_boxes", "image_path"])
            t0 = time.time()
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=args.workers) as pool:
                for i, res in enumerate(pool.imap_unordered(save_one_slide, tasks), 1):
                    total_written += res["written"]
                    total_boxes += res["boxes"]
                    writer.writerows(res["manifest"])
                    if i % 10 == 0 or i == len(tasks):
                        print(
                            "[%s] slides %d/%d tiles=%d boxes=%d elapsed=%.1fs"
                            % (split, i, len(tasks), total_written, total_boxes,
                               time.time() - t0)
                        )

        summary[split] = {
            "slides": len(bmps),
            "gt_mitotic": n_boxes,
            "tiles_written": total_written,
            "tiles_with_gt": positive,
            "background_tiles": total_written - positive,
            "boxes_written": total_boxes,
            "stride": stride,
            "tile": args.tile,
            "overlap": args.overlap,
        }
        print("[%s] done: %s" % (split, summary[split]))

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write("path: %s\n" % out_root)
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['candidate']\n")
    print("wrote", yaml_path)


if __name__ == "__main__":
    main()
