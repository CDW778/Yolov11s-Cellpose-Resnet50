#!/usr/bin/env python3
"""Select (YOLO checkpoint, conf) on val with criterion A.

Criterion A: maximise GT patch recall first, then minimise candidates/slide.

Exact solution per checkpoint
-----------------------------
Run every val slide at a very low raw conf, de-duplicate candidates by centre.
For each GT compute the highest YOLO score among candidates whose 224x224 ROI
patch contains the GT centroid (its "cover score").  The highest threshold that
still keeps every detector-covered GT is the minimum cover score over GTs.
That threshold is the checkpoint's optimal conf under criterion A; the number
of surviving candidates is the tie-break.

Ranking: patch recall desc, candidates/slide asc.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infer_wsi import detect_wsi  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from eval_icpr_candidates import (  # noqa: E402
    IMG_ROOT,
    SRC_ANNO,
    SPLIT_PREFIXES,
    gt_points,
    index_csvs,
)

PATCH = 224


def dedupe_boxes(boxes, scores, dist=16.0):
    if len(boxes) == 0:
        return boxes, scores
    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]
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


def discover_checkpoints(run_dir):
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


def radius_um(base):
    return 5.0 / (0.2456 if str(base).startswith("A") else (0.2273 + 0.22753) / 2.0)


def eval_checkpoint(model, items, csvs, args):
    per_slide = []
    for bmp in items:
        base = bmp.stem
        gts = np.asarray(gt_points(csvs[base]), dtype=np.float64)
        boxes, scores, _ = detect_wsi(
            model,
            str(bmp),
            tile=args.tile,
            overlap=args.overlap,
            conf=args.raw_conf,
            tile_iou=args.tile_iou,
            global_iou=args.global_iou,
            max_det=1000,
            batch=args.batch,
            device=args.device,
        )
        boxes, scores = dedupe_boxes(boxes, scores, args.dedupe_dist)
        if len(boxes):
            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
            C = np.stack([cx, cy], axis=1)
        else:
            C = np.zeros((0, 2), dtype=np.float64)
        per_slide.append({"base": base, "gts": gts, "C": C, "scores": scores})

    n_gt = sum(len(s["gts"]) for s in per_slide)
    covered = 0
    thr_parts = []
    n_cand = 0
    center5 = 0
    for s in per_slide:
        gts, C, sc = s["gts"], s["C"], s["scores"]
        if len(gts) == 0:
            continue
        if len(C) == 0:
            continue
        dx = np.abs(C[:, 0:1] - gts[None, :, 0])
        dy = np.abs(C[:, 1:2] - gts[None, :, 1])
        hit = (dx <= PATCH / 2) & (dy <= PATCH / 2)  # N x M
        covered_gt = hit.any(axis=0)
        covered += int(covered_gt.sum())
        if covered_gt.any():
            cover_max = np.where(hit, sc[:, None], -1.0).max(axis=0)
            thr_parts.append(float(cover_max[covered_gt].min()))
        # candidates surviving the checkpoint threshold are counted later;
        # store data for the final threshold
        s["hit"] = hit
        s["covered_gt"] = covered_gt

    if not thr_parts:
        return None
    thr = min(thr_parts)

    for s in per_slide:
        sc = s["scores"]
        keep = sc >= thr
        n_cand += int(keep.sum())
        if len(s["gts"]) and keep.any():
            gts = s["gts"]
            C = s["C"][keep]
            rad = radius_um(s["base"])
            d2 = (C[:, 0:1] - gts[None, :, 0]) ** 2 + (C[:, 1:2] - gts[None, :, 1]) ** 2
            center5 += int((d2 <= rad * rad).any(axis=0).sum())

    return {
        "gt": n_gt,
        "covered": covered,
        "patch_recall": covered / max(1, n_gt),
        "center5um_recall": center5 / max(1, n_gt),
        "conf": thr,
        "candidates": n_cand,
        "candidates_per_slide": n_cand / max(1, len(per_slide)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--raw-conf", type=float, default=5e-5)
    ap.add_argument("--dedupe-dist", type=float, default=16.0)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--tile-iou", type=float, default=0.3)
    ap.add_argument("--global-iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0, help="limit val slides (smoke)")
    ap.add_argument("--limit-ckpts", type=int, default=0)
    args = ap.parse_args()

    csvs = index_csvs(SRC_ANNO)
    img_dir = Path(IMG_ROOT) / "val"
    items = [p for p in sorted(img_dir.glob("*.bmp")) if p.stem[:3] in SPLIT_PREFIXES["val"]]
    if args.limit:
        items = items[: args.limit]
    ckpts = discover_checkpoints(args.run_dir)
    if args.limit_ckpts:
        ckpts = ckpts[: args.limit_ckpts]
    print("val slides=%d checkpoints=%d raw_conf=%.5g" % (len(items), len(ckpts), args.raw_conf), flush=True)

    rows = []
    for idx, (ep, path) in enumerate(ckpts, 1):
        name = Path(path).stem
        print("\n[%d/%d] %s" % (idx, len(ckpts), name), flush=True)
        model = YOLO(path)
        res = eval_checkpoint(model, items, csvs, args)
        if res is None:
            print("  no covered GT, skip", flush=True)
            continue
        rows.append({"checkpoint": name, "epoch": ep, **res})
        print(
            "  patchR=%.4f center5=%.4f conf=%.5g cand/slide=%.1f"
            % (res["patch_recall"], res["center5um_recall"], res["conf"], res["candidates_per_slide"]),
            flush=True,
        )
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    df = pd.DataFrame(rows).sort_values(
        ["patch_recall", "candidates_per_slide"], ascending=[False, True]
    )
    df.to_csv(args.out_csv, index=False)
    print("\n=== criterion A ranking (patch recall desc, candidates asc) ===")
    print(df.to_string(index=False))

    best = df.iloc[0]
    info = {
        "criterion": "A: patch recall desc, candidates/slide asc",
        "selected_checkpoint": str(Path(args.run_dir) / "weights" / (str(best["checkpoint"]) + ".pt")),
        "checkpoint": str(best["checkpoint"]),
        "epoch": int(best["epoch"]),
        "conf": float(best["conf"]),
        "patch_recall": float(best["patch_recall"]),
        "center5um_recall": float(best["center5um_recall"]),
        "candidates": int(best["candidates"]),
        "candidates_per_slide": float(best["candidates_per_slide"]),
    }
    with open(args.out_json, "w") as f:
        json.dump(info, f, indent=2)
    print("\nselected:", info)


if __name__ == "__main__":
    main()
