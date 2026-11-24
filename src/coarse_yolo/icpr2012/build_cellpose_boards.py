#!/usr/bin/env python3
"""Build experiment-C data: YOLO-box cell-on-white boards + target-mask channel.

Positive rule (pixel annotation):
    >=98% of a GT polygon's points inside the box  OR  IoU(box, GT bbox) > 0.5
Negative: the box contains zero GT pixels.
Ambiguous: contains GT pixels but meets neither rule -> y=-1 (kept only so that
val/test end-to-end evaluation still sees every candidate; excluded from
classifier training).

Board (one board per YOLO box):
    * crop = box + margin (default 16 px, min 64x64)
    * Cellpose on the crop
    * RGB: all cells whose centroid lies inside the box (plus GT masks), on
      white, union-bbox scaled so the longest side is 0.9*224
    * mask channel: only the target cell(s) -- GT masks for positives, the
      max-IoU seed + touching fragments for negatives/ambiguous
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from cellpose import core, models
from scipy import ndimage
from skimage.morphology import disk

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.paths import (  # noqa: E402
    CELLPOSE_WEIGHT,
    CLASSIFIER_B,
    CLASSIFIER_C_GTFREE,
    IMAGES_ROOT,
    RAW_ROOT,
)

SRC = RAW_ROOT
IMG_ROOT = IMAGES_ROOT
SPLIT_PREFIXES = {
    "train": {"A00", "A01", "A02", "H00", "H01", "H02"},
    "val": {"A03", "H03"},
    "test": {"A04", "H04"},
}
TOL = 2.0


def index_csvs(src):
    out = {}
    for d in Path(src).glob("*_v2"):
        for p in d.glob("*.csv"):
            out[p.stem] = p
    return out


def gt_polys(csv_path):
    out = []
    for line in open(csv_path):
        line = line.strip()
        if not line:
            continue
        v = np.array([int(t) for t in line.split(",") if t != ""], dtype=np.int64)
        out.append((v[0::2], v[1::2]))
    return out


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def inter_area(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def label_box(box, polys):
    """Return (y, qual, best_cov, best_iou). y=-1 means ambiguous."""
    qual = []
    best_cov = 0.0
    best_iou = 0.0
    has_gt = False
    for gi, (xs, ys) in enumerate(polys):
        cov = (
            (xs >= box[0] - TOL)
            & (xs <= box[2] + TOL)
            & (ys >= box[1] - TOL)
            & (ys <= box[3] + TOL)
        ).mean()
        gtb = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        iv = iou_xyxy(box, gtb)
        if cov > 0 or iv > 0:
            has_gt = True
        if cov >= 0.98 or iv > 0.5:
            qual.append((gi, float(cov), float(iv)))
        best_cov = max(best_cov, float(cov))
        best_iou = max(best_iou, float(iv))
    if qual:
        return 1, qual, best_cov, best_iou
    if has_gt:
        return -1, [], best_cov, best_iou
    return 0, [], best_cov, best_iou


def crop_box(img, box, margin, min_crop):
    H, W = img.shape[:2]
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    half = max(max(box[2] - box[0], box[3] - box[1]) / 2.0 + margin, min_crop / 2.0)
    xa = int(round(cx - half))
    ya = int(round(cy - half))
    xb = int(round(cx + half))
    yb = int(round(cy + half))
    if xa < 0:
        xb -= xa
        xa = 0
    if ya < 0:
        yb -= ya
        ya = 0
    if xb > W:
        xa -= xb - W
        xb = W
    if yb > H:
        ya -= yb - H
        yb = H
    xa = max(0, xa)
    ya = max(0, ya)
    return img[ya:yb, xa:xb], (xa, ya, xb, yb)


def make_board(img, box, polys, qual, y, model, args):
    """Return dict with rgb/mask arrays and metadata, or None if no mask.

    Cellpose fragments are merged first: the union of all masks is dilated by
    ``comp_dilate`` px and connected components become "cells".  Tiny
    components are discarded.  This keeps the board at 1-2 cells instead of
    showing dozens of fragments.
    """
    crop, (xa, ya, xb, yb) = crop_box(img, box, args.margin, args.min_crop)
    H, W = crop.shape[:2]
    masks, _, _ = model.eval(
        crop,
        batch_size=1,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob,
        min_size=args.min_size,
        normalize={"tile_norm_blocksize": 0},
    )
    union = (masks > 0).astype(np.uint8)
    if union.sum() == 0:
        return None
    k = 2 * args.comp_dilate + 1
    ud = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n_comp, comp = cv2.connectedComponents(ud)

    comps = []
    for c in range(1, n_comp):
        cm = (comp == c) & (masks > 0)
        area = int(cm.sum())
        if area < args.min_cell_area:
            continue
        ys_l, xs_l = np.where(cm)
        comps.append(
            {
                "id": c,
                "mask": cm,
                "bbox": (
                    float(xs_l.min()),
                    float(ys_l.min()),
                    float(xs_l.max() + 1),
                    float(ys_l.max() + 1),
                ),
                "cent": (float(xs_l.mean() + xa), float(ys_l.mean() + ya)),
                "area": area,
            }
        )
    if not comps:
        return None
    by_id = {c["id"]: c for c in comps}

    target = set()
    if y == 1:
        pts = []
        for gi, _cov, _iv in qual:
            xs, ys = polys[gi]
            ins = (
                (xs >= box[0] - TOL)
                & (xs <= box[2] + TOL)
                & (ys >= box[1] - TOL)
                & (ys <= box[3] + TOL)
            )
            pts.extend(zip(xs[ins].tolist(), ys[ins].tolist()))
        for x, yy in pts:
            px, py = int(x - xa), int(yy - ya)
            if 0 <= py < H and 0 <= px < W:
                c = int(comp[py, px])
                if c in by_id:
                    target.add(c)
        if not target and qual:
            # IoU-only positives: max GT-bbox overlap component
            for gi, _cov, _iv in qual:
                xs, ys = polys[gi]
                gtb = (
                    float(xs.min() - xa),
                    float(ys.min() - ya),
                    float(xs.max() + 1 - xa),
                    float(ys.max() + 1 - ya),
                )
                best_c, best_i = None, 0.0
                for c in comps:
                    iv = inter_area(gtb, c["bbox"])
                    if iv > best_i:
                        best_i, best_c = iv, c["id"]
                if best_c is not None:
                    target.add(best_c)
    else:
        # seed component = max intersection with the box
        best_c, best_i = None, 0.0
        for c in comps:
            bb = (c["bbox"][0] + xa, c["bbox"][1] + ya, c["bbox"][2] + xa, c["bbox"][3] + ya)
            iv = inter_area(box, bb)
            if iv > best_i:
                best_i, best_c = iv, c["id"]
        if best_c is not None:
            target.add(best_c)

    cells = {
        c["id"]
        for c in comps
        if box[0] <= c["cent"][0] <= box[2] and box[1] <= c["cent"][1] <= box[3]
    }
    cells |= target
    if not cells:
        return None

    # RGB board: all cells (dilated by 5) on white, union bbox, 0.9 fill
    show = np.zeros((H, W), np.uint8)
    for cid in cells:
        show |= by_id[cid]["mask"].astype(np.uint8)
    show = cv2.dilate(show, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    ys_s, xs_s = np.where(show > 0)
    if len(xs_s) == 0:
        return None
    bx0, by0, bx1, by1 = xs_s.min(), ys_s.min(), xs_s.max() + 1, ys_s.max() + 1
    cell_img = crop[by0:by1, bx0:bx1]
    keep = show[by0:by1, bx0:bx1] > 0
    white = np.full_like(cell_img, 255)
    white[keep] = cell_img[keep]

    ch, cw = white.shape[:2]
    scale = (args.board * args.fill) / max(ch, cw)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    rgb = cv2.resize(white, (nw, nh), interpolation=interp)
    board = np.full((args.board, args.board, 3), 255, np.uint8)
    ox, oy = (args.board - nw) // 2, (args.board - nh) // 2
    board[oy : oy + nh, ox : ox + nw] = rgb

    # mask channel aligned with the RGB board
    tm = np.zeros((H, W), np.uint8)
    for cid in target:
        tm[by_id[cid]["mask"]] = 255
    # Keep the target mask in the Cellpose crop coordinates so callers can
    # project the exact segmentation back onto the original image.
    tm_crop = tm.copy()
    tm = tm[by0:by1, bx0:bx1]
    tm = cv2.resize(tm, (nw, nh), interpolation=cv2.INTER_NEAREST)
    mb = np.zeros((args.board, args.board), np.uint8)
    mb[oy : oy + nh, ox : ox + nw] = tm

    # QA: fraction of GT points inside the box covered by the target mask
    gt_cov = None
    if y == 1 and qual:
        tot = 0
        hit = 0
        for gi, _cv, _iv in qual:
            xs, ys = polys[gi]
            ins = (
                (xs >= box[0] - TOL)
                & (xs <= box[2] + TOL)
                & (ys >= box[1] - TOL)
                & (ys <= box[3] + TOL)
            )
            for x, yy in zip(xs[ins].tolist(), ys[ins].tolist()):
                tot += 1
                px, py = int(x - xa), int(yy - ya)
                if 0 <= py < H and 0 <= px < W and int(comp[py, px]) in target:
                    hit += 1
        gt_cov = hit / max(1, tot)

    return {
        "rgb": board,
        "mask": mb,
        "n_cells": len(cells),
        "n_target": len(target),
        "gt_cov": gt_cov,
        "fallback": 0,
        "target_mask_crop": tm_crop,
        "crop_box": (xa, ya, xb, yb),
    }


def fallback_board(img, box, args):
    crop, _ = crop_box(img, box, args.margin, args.min_crop)
    rgb = cv2.resize(crop, (args.board, args.board), interpolation=cv2.INTER_CUBIC)
    return {
        "rgb": rgb,
        "mask": np.zeros((args.board, args.board), np.uint8),
        "n_cells": 0,
        "n_target": 0,
        "gt_cov": None,
        "fallback": 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-dir", default=str(CLASSIFIER_B / "conf002"))
    ap.add_argument("--out-dir", default=str(CLASSIFIER_C_GTFREE))
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--target-rule", choices=["gt", "box"], default="box",
                    help="gt: use GT points to pick the target mask (training only); "
                         "box: GT-free max-intersection target (deployable/eval)")
    ap.add_argument("--cellpose-weight", default=str(CELLPOSE_WEIGHT))
    ap.add_argument("--margin", type=int, default=16)
    ap.add_argument("--min-crop", type=int, default=64)
    ap.add_argument("--board", type=int, default=224)
    ap.add_argument("--fill", type=float, default=0.9)
    ap.add_argument("--flow-threshold", type=float, default=1.0)
    ap.add_argument("--cellprob", type=float, default=-3.0)
    ap.add_argument("--min-size", type=int, default=15)
    ap.add_argument("--comp-dilate", type=int, default=2,
                    help="dilate px to merge Cellpose fragments into one cell")
    ap.add_argument("--min-cell-area", type=int, default=20,
                    help="drop connected components smaller than this (px)")
    ap.add_argument("--limit-slides", type=int, default=0)
    args = ap.parse_args()

    csvs = index_csvs(SRC)
    model = models.CellposeModel(gpu=core.use_gpu(), pretrained_model=args.cellpose_weight)
    out_root = Path(args.out_dir)
    summary = {}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        cand = dict(
            np.load(Path(args.candidates_dir) / ("candidates_%s.npz" % split), allow_pickle=True)
        )
        base = cand["base"].astype(str)
        x0, y0, x1, y1 = cand["x0"], cand["y0"], cand["x1"], cand["y1"]
        ysco = cand["yolo_score"]
        img_dir = IMG_ROOT / split
        items = [
            p for p in sorted(img_dir.glob("*.bmp")) if p.stem[:3] in SPLIT_PREFIXES[split]
        ]
        if args.limit_slides:
            items = items[: args.limit_slides]
        out_img = out_root / "boards" / split
        out_msk = out_root / "masks" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_msk.mkdir(parents=True, exist_ok=True)
        rows = []
        counts = {"pos": 0, "neg": 0, "amb": 0, "fallback": 0}
        covs = []
        print("[%s] slides=%d candidates=%d" % (split, len(items), len(base)), flush=True)

        for si, bmp in enumerate(items, 1):
            b = bmp.stem
            polys = gt_polys(csvs[b])
            img = cv2.cvtColor(cv2.imread(str(bmp)), cv2.COLOR_BGR2RGB)
            idx = np.where(base == b)[0]
            for j in idx:
                box = (float(x0[j]), float(y0[j]), float(x1[j]), float(y1[j]))
                y, qual, bcov, biou = label_box(box, polys)
                rule = "pos" if y == 1 else ("amb" if y == -1 else "neg")
                # GT is only used for the manifest label when target_rule=box.
                board_y = y if args.target_rule == "gt" else 0
                board_polys = polys if board_y == 1 else []
                board_qual = qual if board_y == 1 else []
                res = make_board(img, box, board_polys, board_qual, board_y, model, args)
                if res is None:
                    res = fallback_board(img, box, args)
                counts["pos" if y == 1 else ("amb" if y == -1 else "neg")] += 1
                counts["fallback"] += res["fallback"]
                if res["gt_cov"] is not None:
                    covs.append(res["gt_cov"])
                name = "%s_c%04d" % (b, int(j))
                rgb_path = out_img / (name + ".png")
                msk_path = out_msk / (name + ".png")
                cv2.imwrite(str(rgb_path), cv2.cvtColor(res["rgb"], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(msk_path), res["mask"])
                rows.append(
                    {
                        "split": split,
                        "base": b,
                        "name": name,
                        "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                        "cx": (box[0] + box[2]) / 2.0,
                        "cy": (box[1] + box[3]) / 2.0,
                        "yolo_score": float(ysco[j]),
                        "y": y,
                        "rule": rule,
                        "gt_cov": "" if res["gt_cov"] is None else res["gt_cov"],
                        "gt_iou": biou,
                        "fallback": res["fallback"],
                        "n_cells": res["n_cells"],
                        "n_target": res["n_target"],
                        "rgb": str(rgb_path),
                        "mask": str(msk_path),
                    }
                )
            if si % 5 == 0 or si == len(items):
                print("  %d/%d rows=%d" % (si, len(items), len(rows)), flush=True)

        with open(out_root / ("manifest_%s.csv" % split), "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "split", "base", "name", "x0", "y0", "x1", "y1", "cx", "cy",
                    "yolo_score", "y", "rule", "gt_cov", "gt_iou", "fallback",
                    "n_cells", "n_target", "rgb", "mask",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        covs = np.asarray(covs, dtype=np.float64) if covs else np.array([])
        summary[split] = {
            **counts,
            "candidates": len(rows),
            "pos_mask_cov_mean": float(covs.mean()) if len(covs) else None,
            "pos_mask_cov_p5": float(np.percentile(covs, 5)) if len(covs) else None,
            "pos_mask_cov_ge95": float((covs >= 0.95).mean()) if len(covs) else None,
        }
        print("[%s] %s" % (split, summary[split]), flush=True)

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out_root / "summary.json")


if __name__ == "__main__":
    main()
