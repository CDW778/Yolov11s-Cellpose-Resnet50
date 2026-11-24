#!/usr/bin/env python3
"""Evaluate stage-1 YOLO ROI/candidate recall on ICPR2012 slides.

GT: polygon CSV -> centroid (mean of polygon vertices), consistent with
``utils_2012.load_gt_points`` in the Cellpose/ICPR2012 baseline project.

Metrics (at several confidence thresholds):
  * recall_patch224: GT centroid falls inside the 224x224 patch that downstream
    crops around a candidate box centre.  This is the ROI recall that matters.
  * recall_center16: candidate centre within 16 px of the GT centroid.
  * candidates / slide and candidates per GT (efficiency).

Test must be reported with a threshold selected on val; this script only prints
the swept table, the caller records the frozen threshold.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from configs.paths import IMAGES_ROOT, RAW_ROOT  # noqa: E402
from infer_wsi import detect_wsi  # noqa: E402
from ultralytics import YOLO  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

SRC_ANNO = RAW_ROOT
IMG_ROOT = IMAGES_ROOT
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


def gt_points(csv_path):
    """Polygon lines -> list of (cx, cy) using vertex mean."""
    pts = []
    for line in open(csv_path):
        line = line.strip()
        if not line:
            continue
        vals = np.array([int(t) for t in line.split(",") if t != ""], dtype=np.float64)
        xs, ys = vals[0::2], vals[1::2]
        if len(xs):
            pts.append((float(xs.mean()), float(ys.mean())))
    return pts


def center_covered(gx, gy, boxes, patch=224):
    half = patch / 2.0
    for b in boxes:
        cx = (b[0] + b[2]) * 0.5
        cy = (b[1] + b[3]) * 0.5
        if cx - half <= gx <= cx + half and cy - half <= gy <= cy + half:
            return True
    return False


def center_near(gx, gy, boxes, radius=16.0):
    r2 = radius * radius
    for b in boxes:
        cx = (b[0] + b[2]) * 0.5
        cy = (b[1] + b[3]) * 0.5
        if (cx - gx) ** 2 + (cy - gy) ** 2 <= r2:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--src-anno", default=str(SRC_ANNO))
    ap.add_argument("--images-root", default=str(IMG_ROOT))
    ap.add_argument("--conf-list", default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--raw-conf", type=float, default=0.01)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--tile-iou", type=float, default=0.3)
    ap.add_argument("--global-iou", type=float, default=0.5)
    ap.add_argument("--center-dist", type=float, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    confs = [float(x) for x in args.conf_list.split(",")]
    csvs = index_csvs(args.src_anno)
    img_dir = Path(args.images_root) / args.split
    prefixes = SPLIT_PREFIXES[args.split]
    items = [
        p for p in sorted(img_dir.glob("*.bmp")) if p.stem[:3] in prefixes
    ]
    if args.limit:
        items = items[: args.limit]

    model = YOLO(args.model)
    rows = []
    for k, bmp in enumerate(items, 1):
        base = bmp.stem
        if base not in csvs:
            print("[%d/%d] %s: missing CSV, skip" % (k, len(items), base))
            continue
        gts = gt_points(csvs[base])
        boxes, scores, _ = detect_wsi(
            model,
            str(bmp),
            tile=args.tile,
            overlap=args.overlap,
            conf=args.raw_conf,
            tile_iou=args.tile_iou,
            global_iou=args.global_iou,
            center_dist=args.center_dist,
            batch=args.batch,
            device=args.device,
        )
        for conf in confs:
            keep = scores >= conf
            cand = boxes[keep]
            n = int(keep.sum())
            rec224 = (
                sum(center_covered(gx, gy, cand) for gx, gy in gts) / len(gts)
                if gts else 0.0
            )
            rec16 = (
                sum(center_near(gx, gy, cand) for gx, gy in gts) / len(gts)
                if gts else 0.0
            )
            rows.append(
                {
                    "slide": base,
                    "conf": conf,
                    "gt_mitotic": len(gts),
                    "candidates": n,
                    "recall_patch224": rec224,
                    "recall_center16": rec16,
                }
            )
        print(
            "[%d/%d] %s gt=%d raw_candidates=%d"
            % (k, len(items), base, len(gts), len(boxes)),
            flush=True,
        )

    df = pd.DataFrame(rows)
    out = args.out or ("eval_icpr_%s.csv" % args.split)
    df.to_csv(out, index=False)

    print("\n=== aggregate (%s) ===" % args.split)
    agg = (
        df.groupby("conf")
        .apply(
            lambda g: pd.Series(
                {
                    "slides": len(g),
                    "gt_mitotic": g["gt_mitotic"].sum(),
                    "candidates_total": g["candidates"].sum(),
                    "candidates_per_slide": g["candidates"].mean(),
                    "candidates_per_gt": (
                        g["candidates"].sum() / max(1, g["gt_mitotic"].sum())
                    ),
                    "recall_patch224": (
                        (g["recall_patch224"] * g["gt_mitotic"]).sum()
                        / max(1, g["gt_mitotic"].sum())
                    ),
                    "recall_center16": (
                        (g["recall_center16"] * g["gt_mitotic"]).sum()
                        / max(1, g["gt_mitotic"].sum())
                    ),
                }
            )
        )
        .reset_index()
    )
    print(agg.to_string(index=False))
    agg.to_csv(out.replace(".csv", "_aggregate.csv"), index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
