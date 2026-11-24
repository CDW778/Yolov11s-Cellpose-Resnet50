#!/usr/bin/env python3
"""Build YOLO-candidate classification data for ICPR2012 (experiment B).

For every split we run the stage-1 detector on the full slide, de-duplicate
candidates by centre distance, and label each candidate by whether its
224x224 downstream ROI patch contains at least one GT centroid.  Only the
candidate table is saved; patches are cropped from the original BMP at train
time (pixel-exact, no recompression).

Outputs (in --out-dir):
  candidates_{split}.npz  fields expected by dann_dataset.CandidateDataset
                          plus yolo_score / gt_idx / gt_dist / gt_ncomp
  manifest_{split}.csv
  summary.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infer_wsi import detect_wsi  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from eval_icpr_candidates import (  # noqa: E402
    IMG_ROOT,
    SRC_ANNO,
    SPLIT_PREFIXES,
    index_csvs,
)

PATCH = 224


def load_gt(csv_path):
    """Return (points [(cx,cy)], n_components array)."""
    pts = []
    comps = []
    for line in open(csv_path):
        line = line.strip()
        if not line:
            continue
        vals = np.array([int(t) for t in line.split(",") if t != ""], dtype=np.int64)
        xs, ys = vals[0::2], vals[1::2]
        pts.append((float(xs.mean()), float(ys.mean())))
        h = int(ys.max() - ys.min() + 1)
        w = int(xs.max() - xs.min() + 1)
        m = np.zeros((h, w), np.uint8)
        m[ys - ys.min(), xs - xs.min()] = 1
        _, n = ndimage.label(m, structure=np.ones((3, 3)))
        comps.append(int(n))
    return pts, np.asarray(comps)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--conf-train", type=float, default=0.001)
    ap.add_argument("--conf-eval", type=float, default=0.002)
    ap.add_argument("--dedupe-dist", type=float, default=16.0)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--tile-iou", type=float, default=0.3)
    ap.add_argument("--global-iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csvs = index_csvs(SRC_ANNO)
    model = YOLO(args.detector)
    summary = {}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        conf = args.conf_train if split == "train" else args.conf_eval
        img_dir = Path(IMG_ROOT) / split
        prefixes = SPLIT_PREFIXES[split]
        items = [p for p in sorted(img_dir.glob("*.bmp")) if p.stem[:3] in prefixes]
        if args.limit:
            items = items[: args.limit]
        print("[%s] slides=%d conf=%.4f" % (split, len(items), conf), flush=True)

        rows = []
        n_gt = 0
        gt_covered = 0
        for k, bmp in enumerate(items, 1):
            base = bmp.stem
            pts, comps = load_gt(csvs[base])
            n_gt += len(pts)
            boxes, scores, _ = detect_wsi(
                model,
                str(bmp),
                tile=args.tile,
                overlap=args.overlap,
                conf=conf,
                tile_iou=args.tile_iou,
                global_iou=args.global_iou,
                center_dist=None,
                batch=args.batch,
                device=args.device,
            )
            boxes, scores = dedupe_boxes(boxes, scores, args.dedupe_dist)

            covered = set()
            for box, score in zip(boxes, scores):
                cx = float((box[0] + box[2]) * 0.5)
                cy = float((box[1] + box[3]) * 0.5)
                hits = []
                for gi, (gx, gy) in enumerate(pts):
                    if cx - PATCH / 2 <= gx <= cx + PATCH / 2 and cy - PATCH / 2 <= gy <= cy + PATCH / 2:
                        hits.append(gi)
                        covered.add(gi)
                if hits:
                    dists = [((pts[gi][0] - cx) ** 2 + (pts[gi][1] - cy) ** 2) for gi in hits]
                    gi = hits[int(np.argmin(dists))]
                    rows.append(
                        {
                            "base": base,
                            "name": "%s_c%04d" % (base, len(rows)),
                            "x0": float(box[0]),
                            "y0": float(box[1]),
                            "x1": float(box[2]),
                            "y1": float(box[3]),
                            "cx": cx,
                            "cy": cy,
                            "yolo_score": float(score),
                            "y": 1,
                            "gt_idx": int(gi),
                            "gt_dist": float(np.sqrt((pts[gi][0] - cx) ** 2 + (pts[gi][1] - cy) ** 2)),
                            "gt_ncomp": int(comps[gi]),
                            "scanner": base[0],
                        }
                    )
                else:
                    d = [np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2) for gx, gy in pts]
                    rows.append(
                        {
                            "base": base,
                            "name": "%s_c%04d" % (base, len(rows)),
                            "x0": float(box[0]),
                            "y0": float(box[1]),
                            "x1": float(box[2]),
                            "y1": float(box[3]),
                            "cx": cx,
                            "cy": cy,
                            "yolo_score": float(score),
                            "y": 0,
                            "gt_idx": -1,
                            "gt_dist": float(min(d)) if d else -1.0,
                            "gt_ncomp": 0,
                            "scanner": base[0],
                        }
                    )
            gt_covered += len(covered)
            if k % 7 == 0 or k == len(items):
                print("  %d/%d candidates=%d" % (k, len(items), len(rows)), flush=True)

        n_pos = sum(r["y"] for r in rows)
        n_neg = len(rows) - n_pos
        np.savez_compressed(
            out_dir / ("candidates_%s.npz" % split),
            base=np.array([r["base"] for r in rows]),
            label=np.array([r["gt_idx"] for r in rows], dtype=np.int32),
            x0=np.array([r["x0"] for r in rows], dtype=np.float32),
            y0=np.array([r["y0"] for r in rows], dtype=np.float32),
            x1=np.array([r["x1"] for r in rows], dtype=np.float32),
            y1=np.array([r["y1"] for r in rows], dtype=np.float32),
            cx=np.array([r["cx"] for r in rows], dtype=np.float32),
            cy=np.array([r["cy"] for r in rows], dtype=np.float32),
            y=np.array([r["y"] for r in rows], dtype=np.int8),
            yolo_score=np.array([r["yolo_score"] for r in rows], dtype=np.float32),
            gt_dist=np.array([r["gt_dist"] for r in rows], dtype=np.float32),
            gt_ncomp=np.array([r["gt_ncomp"] for r in rows], dtype=np.int32),
            scanner=np.array([r["scanner"] for r in rows]),
        )

        with open(out_dir / ("manifest_%s.csv" % split), "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "base", "name", "x0", "y0", "x1", "y1", "cx", "cy",
                    "yolo_score", "y", "gt_idx", "gt_dist", "gt_ncomp", "scanner",
                ],
            )
            w.writeheader()
            w.writerows(rows)

        summary[split] = {
            "conf": conf,
            "slides": len(items),
            "gt_mitotic": n_gt,
            "gt_covered": gt_covered,
            "gt_coverage": gt_covered / max(1, n_gt),
            "candidates": len(rows),
            "pos": int(n_pos),
            "neg": int(n_neg),
            "pos_rate": n_pos / max(1, len(rows)),
            "pos_multi_component": int(sum(1 for r in rows if r["y"] == 1 and r["gt_ncomp"] >= 2)),
            "candidates_per_slide": len(rows) / max(1, len(items)),
        }
        print("[%s] %s" % (split, summary[split]), flush=True)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
