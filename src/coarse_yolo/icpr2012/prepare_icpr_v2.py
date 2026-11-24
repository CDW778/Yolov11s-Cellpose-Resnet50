#!/usr/bin/env python3
"""Build the v2a ICPR2012 training set: GT-centred random crops + v1 background.

For every mitotic GT in the train split we sample K 512x512 crops.  The GT box
is kept at least ``margin`` pixels away from the crop border (when the slide
allows it) and the crop offset is uniformly random, so each GT contributes K
genuinely different views.  All GT whose centre falls inside a crop are written
as labels, so neighbouring mitoses also become supervision.

Background tiles are reused from the v1 tiled dataset (all 512-grid tiles that
contain no GT).  The val split is symlinked from v1 unchanged.
"""

import argparse
import json
import multiprocessing
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.paths import IMAGES_ROOT, RAW_ROOT, YOLO_V1, YOLO_V2A  # noqa: E402
from prepare_tiles import axis_positions  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

SRC_ANNO = RAW_ROOT
IMG_ROOT = IMAGES_ROOT
V1_ROOT = YOLO_V1
DEFAULT_OUT = YOLO_V2A

SPLIT_PREFIXES = {
    "train": {"A00", "A01", "A02", "H00", "H01", "H02"},
    "val": {"A03", "H03"},
    "test": {"A04", "H04"},
}


def index_csvs(src_anno):
    out = {}
    for d in Path(src_anno).glob("*_v2"):
        for p in d.glob("*.csv"):
            out[p.stem] = p
    return out


def load_gt_boxes(csv_path):
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


def sample_start(lo, hi, rng):
    lo = int(np.ceil(lo))
    hi = int(np.floor(hi))
    if hi < lo:
        return None
    return rng.randint(lo, hi)


def build_crops(base, image_path, boxes, k, tile, margin, rng, out_images, out_labels,
                jpeg_quality):
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)
    im = Image.open(image_path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    W, H = im.size
    manifest = []
    written = 0
    labels_written = 0

    for gi, (x1, y1, x2, y2) in enumerate(boxes):
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        lo_x = max(0, x2 + margin - tile)
        hi_x = min(W - tile, x1 - margin)
        lo_y = max(0, y2 + margin - tile)
        hi_y = min(H - tile, y1 - margin)

        for ki in range(k):
            L = sample_start(lo_x, hi_x, rng)
            T = sample_start(lo_y, hi_y, rng)
            if L is None:
                L = int(np.clip(round(cx - tile / 2.0), 0, max(0, W - tile)))
            if T is None:
                T = int(np.clip(round(cy - tile / 2.0), 0, max(0, H - tile)))

            # Collect every GT whose centre lies inside this crop.
            labels = []
            for bx1, by1, bx2, by2 in boxes:
                bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
                if not (L <= bcx < L + tile and T <= bcy < T + tile):
                    continue
                ix1, iy1 = max(bx1, L), max(by1, T)
                ix2, iy2 = min(bx2, L + tile), min(by2, T + tile)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                vis = ((ix2 - ix1) * (iy2 - iy1)) / max(1e-6, (bx2 - bx1) * (by2 - by1))
                if vis < 0.25:
                    continue
                labels.append(
                    (
                        0,
                        ((ix1 + ix2) * 0.5 - L) / tile,
                        ((iy1 + iy2) * 0.5 - T) / tile,
                        (ix2 - ix1) / tile,
                        (iy2 - iy1) / tile,
                    )
                )

            name = "%s_g%03d_k%d_%05d_%05d" % (base, gi, ki, L, T)
            crop = im.crop((L, T, L + tile, T + tile))
            img_path = os.path.join(out_images, name + ".jpg")
            lbl_path = os.path.join(out_labels, name + ".txt")
            crop.save(img_path, quality=jpeg_quality, subsampling=0)
            with open(lbl_path, "w") as f:
                for cls, lcx, lcy, bw, bh in labels:
                    f.write("%d %.6f %.6f %.6f %.6f\n" % (cls, lcx, lcy, bw, bh))
            written += 1
            labels_written += len(labels)
            manifest.append((base, name, L, T, len(labels), img_path))

    im.close()
    return {
        "slide": base,
        "written": written,
        "boxes": labels_written,
        "manifest": manifest,
    }


def build_crops_star(task):
    return build_crops(*task)


def hardlink_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-anno", default=str(SRC_ANNO))
    ap.add_argument("--images-root", default=str(IMG_ROOT))
    ap.add_argument("--v1-root", default=str(V1_ROOT))
    ap.add_argument("--output-root", default=str(DEFAULT_OUT))
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--crops-per-gt", type=int, default=4)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    csvs = index_csvs(args.src_anno)
    out_root = Path(args.output_root)
    v1_root = Path(args.v1_root)
    out_img = out_root / "images" / "train"
    out_lbl = out_root / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.images_root) / "train"
    bmps = [p for p in sorted(img_dir.glob("*.bmp")) if p.stem[:3] in SPLIT_PREFIXES["train"]]
    print("[train] slides=%d" % len(bmps))

    ctx = multiprocessing.get_context("spawn")
    rng = random.Random(args.seed)
    tasks = []
    n_gt = 0
    for bmp in bmps:
        base = bmp.stem
        if base not in csvs:
            print("WARNING missing CSV:", base)
            continue
        boxes = load_gt_boxes(csvs[base])
        n_gt += len(boxes)
        tasks.append(
            (
                base,
                str(bmp),
                boxes,
                args.crops_per_gt,
                args.tile,
                args.margin,
                random.Random(rng.randrange(1 << 30)),
                str(out_img),
                str(out_lbl),
                args.jpeg_quality,
            )
        )

    total = 0
    total_labels = 0
    manifest_rows = []
    with ctx.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(build_crops_star, tasks), 1):
            total += res["written"]
            total_labels += res["boxes"]
            manifest_rows.extend(res["manifest"])
            if i % 10 == 0 or i == len(tasks):
                print("[train] slides %d/%d crops=%d labels=%d" % (i, len(tasks), total, total_labels))

    # Background tiles from v1 (no GT).
    v1_manifest = v1_root / "manifest_train.csv"
    bg = 0
    import csv as _csv

    with open(v1_manifest, newline="") as f:
        for row in _csv.DictReader(f):
            if int(row["num_boxes"]) != 0:
                continue
            src_img = Path(row["image_path"])
            src_lbl = v1_root / "labels" / "train" / (src_img.stem + ".txt")
            hardlink_or_copy(str(src_img), str(out_img / src_img.name))
            hardlink_or_copy(str(src_lbl), str(out_lbl / src_lbl.name))
            bg += 1
    print("[train] background tiles reused: %d" % bg)

    # Val symlinks (unchanged v1 split).
    for sub in ("images", "labels"):
        link = out_root / sub / "val"
        target = v1_root / sub / "val"
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            elif link.is_dir():
                shutil.rmtree(link)
        link.symlink_to(target)
    print("[val] symlinked to", v1_root)

    with open(out_root / "manifest_train.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["slide", "name", "x", "y", "num_boxes", "image_path"])
        w.writerows(manifest_rows)

    summary = {
        "slides": len(tasks),
        "gt_mitotic": n_gt,
        "crops_per_gt": args.crops_per_gt,
        "gt_crops": total,
        "gt_crops_labels": total_labels,
        "background_tiles": bg,
        "train_images": total + bg,
        "tile": args.tile,
        "margin": args.margin,
        "seed": args.seed,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("summary:", summary)

    with open(out_root / "data.yaml", "w") as f:
        f.write("path: %s\n" % out_root)
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['candidate']\n")
    print("wrote", out_root / "data.yaml")


if __name__ == "__main__":
    main()
