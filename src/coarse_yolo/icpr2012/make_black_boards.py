#!/usr/bin/env python3
"""Make a black-background variant of the classifier_C boards.

The existing boards are built as "cell-on-white": every non-cell pixel is
filled with 255.  This script replaces that connected white background with
0 while leaving cell pixels untouched, so the only change is the background
colour.  The target-mask channel is copied unchanged.

Background detection uses a flood fill seeded at the top-left corner (the
boards always keep a white margin around the cells).  A small tolerance
absorbs the anti-aliased white rim left by board resizing.  Fallback boards
(direct crop, no white margin) are handled the same way: the connected
near-white region touching the corner is blackened.

Outputs a sibling dataset directory with boards/, masks/ and rewritten
manifest_*.csv files; the source classifier_C tree is never modified.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.paths import CLASSIFIER_C, ICPR_ROOT  # noqa: E402

SPLITS = ("train", "val", "test")


def blacken_background(rgb, tol):
    """Replace the near-white region connected to (0, 0) with black.

    FIXED_RANGE compares every candidate pixel against the seed colour, so the
    fill cannot creep into the cells through the anti-aliased rim (the default
    floating range would, one small step at a time).
    """
    h, w = rgb.shape[:2]
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    seed = max(corners, key=lambda p: int(rgb[p[1], p[0]].min()))
    if int(rgb[seed[1], seed[0]].min()) < 200:
        # Fallback board whose corners are tissue, not white background:
        # leave it untouched rather than blackening the cell.
        return rgb.copy(), np.zeros((h, w), bool)

    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    work = rgb.copy()
    diff = (tol, tol, tol)
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE
    cv2.floodFill(work, ff_mask, seed, (0, 0, 0), diff, diff, flags)
    bg = ff_mask[1:-1, 1:-1] > 0
    out = rgb.copy()
    out[bg] = 0
    return out, bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=str(CLASSIFIER_C))
    ap.add_argument("--out-dir", default=str(ICPR_ROOT / "classifier_C_black"))
    ap.add_argument("--tol", type=int, default=25,
                    help="flood-fill tolerance vs the corner seed colour")
    args = ap.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    summary = {}

    for split in SPLITS:
        rows = []
        with open(src / ("manifest_%s.csv" % split), newline="") as f:
            rows = list(csv.DictReader(f))
        out_img = out / "boards" / split
        out_msk = out / "masks" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_msk.mkdir(parents=True, exist_ok=True)

        bg_fracs = []
        fallback_bg = []
        for i, r in enumerate(rows, 1):
            rgb = cv2.imread(r["rgb"])
            if rgb is None:
                raise RuntimeError("bad board: %s" % r["rgb"])
            black, bg = blacken_background(rgb, args.tol)
            name = Path(r["rgb"]).name
            cv2.imwrite(str(out_img / name), black)

            msk = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
            if msk is None:
                raise RuntimeError("bad mask: %s" % r["mask"])
            cv2.imwrite(str(out_msk / name), msk)

            frac = float(bg.mean())
            bg_fracs.append(frac)
            if int(r.get("fallback", 0) or 0):
                fallback_bg.append(frac)
            r["rgb"] = str(out_img / name)
            r["mask"] = str(out_msk / name)
            if i % 500 == 0 or i == len(rows):
                print("[%s] %d/%d" % (split, i, len(rows)), flush=True)

        with open(out / ("manifest_%s.csv" % split), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        summary[split] = {
            "boards": len(rows),
            "bg_frac_mean": float(np.mean(bg_fracs)),
            "bg_frac_min": float(np.min(bg_fracs)),
            "bg_frac_max": float(np.max(bg_fracs)),
            "fallback_boards": len(fallback_bg),
            "fallback_bg_frac_mean": float(np.mean(fallback_bg)) if fallback_bg else None,
        }
        print("[%s] %s" % (split, summary[split]), flush=True)

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
