#!/usr/bin/env python3
"""Select the hard-negative YOLO checkpoint by candidates at fixed GT coverage.

For every checkpoint we run val at a low raw conf once, cache the boxes, and
sweep the operational conf.  The score of a checkpoint is the smallest number
of candidates/slide that still achieves the target patch recall (default 99%
and 100%).  The best (checkpoint, conf) is then re-run exactly on val/test.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infer_wsi import detect_wsi  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from eval_icpr_candidates import (  # noqa: E402
    IMG_ROOT,
    SRC_ANNO,
    SPLIT_PREFIXES,
    index_csvs,
    gt_points,
)

Image.MAX_IMAGE_PIXELS = None


def res_px(base):
    return 5.0 / (0.2456 if str(base).startswith("A") else (0.2273 + 0.22753) / 2.0)


def dedupe(boxes, scores, dist):
    if len(boxes) == 0:
        return boxes, scores
    order = np.argsort(-scores)
    boxes, scores = boxes[order], scores[order]
    keep = []
    for i in range(len(boxes)):
        if not keep:
            keep.append(i)
            continue
        cx = (boxes[i, 0] + boxes[i, 2]) * 0.5
        cy = (boxes[i, 1] + boxes[i, 3]) * 0.5
        k = np.asarray(keep)
        kx = (boxes[k, 0] + boxes[k, 2]) * 0.5
        ky = (boxes[k, 1] + boxes[k, 3]) * 0.5
        if np.min((kx - cx) ** 2 + (ky - cy) ** 2) >= dist * dist:
            keep.append(i)
    keep = np.asarray(keep, dtype=np.int64)
    return boxes[keep], scores[keep]


def discover(run_dir):
    wdir = Path(run_dir) / "weights"
    items = []
    for p in wdir.glob("epoch*.pt"):
        m = re.match(r"epoch(\d+)\.pt$", p.name)
        if m:
            items.append((int(m.group(1)), str(p)))
    for name in ("last.pt", "best.pt"):
        p = wdir / name
        if p.exists():
            items.append((10**6 if name == "last.pt" else 10**7, str(p)))
    items.sort()
    return items


def match(dets, gts, thr):
    ud, ug = set(), set()
    if dets:
        D = np.array([[d[0], d[1]] for d in dets])
        for gi, (gx, gy) in enumerate(gts):
            dd = np.sqrt((D[:, 0] - gx) ** 2 + (D[:, 1] - gy) ** 2)
            for di in np.argsort(dd):
                if dd[di] > thr:
                    break
                if di in ud:
                    continue
                ud.add(int(di))
                ug.add(gi)
                break
    return len(ug), len(dets) - len(ud), len(gts) - len(ug)


def eval_split(per_slide, conf, dedupe_dist=24.0):
    n_cand = 0
    patch_cov = 0
    c5 = 0
    tot = 0
    for base, gts, boxes, scores in per_slide:
        bb, ss = dedupe(boxes[scores >= conf], scores[scores >= conf], dedupe_dist)
        n_cand += len(bb)
        C = np.stack([(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2], 1) if len(bb) else np.zeros((0, 2))
        tot += len(gts)
        for gx, gy in gts:
            if len(C) and ((np.abs(C[:, 0] - gx) <= 112) & (np.abs(C[:, 1] - gy) <= 112)).any():
                patch_cov += 1
            if len(C) and ((C[:, 0] - gx) ** 2 + (C[:, 1] - gy) ** 2).min() <= res_px(base) ** 2:
                c5 += 1
    return {
        "conf": float(conf),
        "candidates": n_cand,
        "candidates_per_slide": n_cand / max(1, len(per_slide)),
        "patch_recall": patch_cov / max(1, tot),
        "center5_recall": c5 / max(1, tot),
        "gt": tot,
    }


def load_split(model, split, csvs, args):
    items = [
        p for p in sorted((IMG_ROOT / split).glob("*.bmp"))
        if p.stem[:3] in SPLIT_PREFIXES[split]
    ]
    out = []
    for bmp in items:
        gts = gt_points(csvs[bmp.stem])
        boxes, scores, _ = detect_wsi(
            model, str(bmp), tile=512, overlap=0.20, conf=args.raw_conf,
            tile_iou=0.3, global_iou=0.5, max_det=1000, batch=16, device=args.device,
        )
        boxes, scores = dedupe(boxes, scores, 16.0)
        out.append((bmp.stem, gts, boxes, scores))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--raw-conf", type=float, default=5e-4)
    ap.add_argument("--targets", default="0.99,1.0")
    ap.add_argument("--conf-grid", default="0.0005,0.0007,0.001,0.0015,0.002,0.003,0.004,0.005,0.007,0.01,0.015,0.02,0.03,0.05,0.08,0.1,0.2")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    csvs = index_csvs(SRC_ANNO)
    targets = [float(x) for x in args.targets.split(",")]
    confs = [float(x) for x in args.conf_grid.split(",")]
    ckpts = discover(args.run_dir)
    rows = []
    for ep, path in ckpts:
        name = Path(path).stem
        print("[%s]" % name, flush=True)
        model = YOLO(path)
        per_slide = load_split(model, "val", csvs, args)
        curve = [eval_split(per_slide, c, 24.0) for c in confs]
        row = {"checkpoint": name, "epoch": ep}
        for t in targets:
            ok = [r for r in curve if r["patch_recall"] >= t - 1e-9]
            if not ok:
                continue
            best = min(ok, key=lambda r: r["candidates_per_slide"])
            k = "t%.3f" % t
            row[k + "_conf"] = best["conf"]
            row[k + "_cand"] = best["candidates_per_slide"]
            row[k + "_patch"] = best["patch_recall"]
            row[k + "_c5"] = best["center5_recall"]
        row["curve"] = curve
        rows.append(row)
        print("  " + " ".join(
            "%s: conf=%.4g cand=%.1f patch=%.3f" % (k, row[k + "_conf"], row[k + "_cand"], row[k + "_patch"])
            for k in ["t%.3f" % t for t in targets] if k + "_conf" in row
        ), flush=True)
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # flatten for CSV
    flat = []
    for r in rows:
        d = {k: v for k, v in r.items() if k != "curve"}
        flat.append(d)
    if flat:
        keys = sorted(set().union(*[set(d.keys()) for d in flat]))
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(flat)

    # select best checkpoint for the first target (default 0.99)
    t0 = targets[0]
    cand_rows = [r for r in rows if ("t%.3f_cand" % t0) in r]
    best = min(cand_rows, key=lambda r: (r["t%.3f_cand" % t0], -r["t%.3f_patch" % t0]))
    info = {
        "target": t0,
        "selected_checkpoint": str(Path(args.run_dir) / "weights" / (best["checkpoint"] + ".pt")),
        "checkpoint": best["checkpoint"],
        "conf": best["t%.3f_conf" % t0],
        "candidates_per_slide": best["t%.3f_cand" % t0],
        "patch_recall": best["t%.3f_patch" % t0],
        "center5_recall": best["t%.3f_c5" % t0],
    }
    with open(args.out_json, "w") as f:
        json.dump({"selection": info, "rows": rows}, f, indent=2)
    print("\nselected:", info)


if __name__ == "__main__":
    main()
