#!/usr/bin/env python3
"""Select the v2a checkpoint by val ROI recall and report test once.

For every saved checkpoint we run full-slide inference on the val split, cache
the candidate boxes, then sweep confidence.  The checkpoint score is the best
``recall_patch224`` achievable while keeping the candidate budget
(``--budget`` candidates per slide).  After selecting the best checkpoint and
its confidence, the test split is evaluated once at that frozen confidence.
"""

import argparse
import glob
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
    center_covered,
    center_near,
    gt_points,
    index_csvs,
)


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


def evaluate_slides(model, split, csvs, args):
    img_dir = Path(args.images_root) / split
    prefixes = SPLIT_PREFIXES[split]
    items = [p for p in sorted(img_dir.glob("*.bmp")) if p.stem[:3] in prefixes]
    per_slide = []
    for k, bmp in enumerate(items, 1):
        base = bmp.stem
        if base not in csvs:
            print("  missing CSV:", base)
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
        per_slide.append((base, gts, boxes, scores))
        if k % 7 == 0 or k == len(items):
            print("  %d/%d" % (k, len(items)), flush=True)
    return per_slide


def sweep(per_slide, confs):
    n_gt = sum(len(g) for _, g, _, _ in per_slide)
    rows = []
    for conf in confs:
        n_cand = 0
        tp224 = 0
        tp16 = 0
        for _base, gts, boxes, scores in per_slide:
            cand = boxes[scores >= conf]
            n_cand += len(cand)
            tp224 += sum(center_covered(gx, gy, cand) for gx, gy in gts)
            tp16 += sum(center_near(gx, gy, cand) for gx, gy in gts)
        rows.append(
            {
                "conf": float(conf),
                "gt_mitotic": n_gt,
                "candidates": n_cand,
                "candidates_per_slide": n_cand / max(1, len(per_slide)),
                "recall_patch224": tp224 / max(1, n_gt),
                "recall_center16": tp16 / max(1, n_gt),
            }
        )
    return pd.DataFrame(rows)


def pick_conf(df, budget):
    """Lowest conf that keeps candidates/slide <= budget (max recall under budget)."""
    ok = df[df["candidates_per_slide"] <= budget]
    if len(ok) == 0:
        return None
    return ok.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--src-anno", default=str(SRC_ANNO))
    ap.add_argument("--images-root", default=str(IMG_ROOT))
    ap.add_argument("--budget", type=float, default=30.0,
                    help="max candidates per slide for checkpoint scoring")
    ap.add_argument("--conf-list", default=None)
    ap.add_argument("--raw-conf", type=float, default=0.005)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--tile-iou", type=float, default=0.3)
    ap.add_argument("--global-iou", type=float, default=0.5)
    ap.add_argument("--center-dist", type=float, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out-prefix", default="v2a_select")
    args = ap.parse_args()

    if args.conf_list:
        confs = [float(x) for x in args.conf_list.split(",")]
    else:
        confs = np.round(np.arange(0.005, 0.5001, 0.005), 4)

    csvs = index_csvs(args.src_anno)
    run_dir = Path(args.run_dir)
    ckpts = discover_checkpoints(run_dir)
    print("checkpoints: %d" % len(ckpts))

    rows = []
    for ep, path in ckpts:
        name = Path(path).stem
        print("\n=== %s (epoch=%s) ===" % (name, "last" if ep >= 10**6 else ep), flush=True)
        model = YOLO(path)
        per_slide = evaluate_slides(model, "val", csvs, args)
        df = sweep(per_slide, confs)
        pick = pick_conf(df, args.budget)
        if pick is None:
            pick = df.iloc[0]
            feasible = False
        else:
            feasible = True
        rows.append(
            {
                "checkpoint": name,
                "epoch": ep,
                "feasible": feasible,
                "conf": float(pick["conf"]),
                "candidates_per_slide": float(pick["candidates_per_slide"]),
                "recall_patch224": float(pick["recall_patch224"]),
                "recall_center16": float(pick["recall_center16"]),
            }
        )
        print(
            "[%s] conf=%.3f cand/slide=%.1f rec224=%.4f rec16=%.4f"
            % (
                name,
                pick["conf"],
                pick["candidates_per_slide"],
                pick["recall_patch224"],
                pick["recall_center16"],
            ),
            flush=True,
        )
        del model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    scores = pd.DataFrame(rows).sort_values(
        ["recall_patch224", "candidates_per_slide"], ascending=[False, True]
    )
    scores.to_csv("%s_checkpoint_scores.csv" % args.out_prefix, index=False)
    print("\n=== checkpoint ranking ===")
    print(scores.to_string(index=False))

    feasible = scores[scores["feasible"].astype(bool)]
    pool = feasible if len(feasible) else scores
    best = pool.iloc[0]
    frozen_conf = float(best["conf"])
    best_path = str(run_dir / "weights" / (str(best["checkpoint"]) + ".pt"))
    print("\nselected: %s conf=%.3f" % (best_path, frozen_conf))

    model = YOLO(best_path)
    summary = {
        "selected_checkpoint": best_path,
        "frozen_conf": frozen_conf,
        "budget": args.budget,
        "val": {},
        "test": {},
    }
    for split in ("val", "test"):
        per_slide = evaluate_slides(model, split, csvs, args)
        df = sweep(per_slide, [frozen_conf])
        row = df.iloc[0]
        summary[split] = {
            "gt_mitotic": int(row["gt_mitotic"]),
            "candidates": int(row["candidates"]),
            "candidates_per_slide": float(row["candidates_per_slide"]),
            "recall_patch224": float(row["recall_patch224"]),
            "recall_center16": float(row["recall_center16"]),
        }
        # per-slide csv at the frozen conf
        recs = []
        for base, gts, boxes, sc in per_slide:
            cand = boxes[sc >= frozen_conf]
            recs.append(
                {
                    "slide": base,
                    "conf": frozen_conf,
                    "gt_mitotic": len(gts),
                    "candidates": len(cand),
                    "recall_patch224": (
                        sum(center_covered(gx, gy, cand) for gx, gy in gts) / len(gts)
                        if gts else 0.0
                    ),
                    "recall_center16": (
                        sum(center_near(gx, gy, cand) for gx, gy in gts) / len(gts)
                        if gts else 0.0
                    ),
                }
            )
        pd.DataFrame(recs).to_csv(
            "%s_eval_icpr_%s.csv" % (args.out_prefix, split), index=False
        )
        print("[%s] %s" % (split, summary[split]), flush=True)

    with open("%s_summary.json" % args.out_prefix, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote %s_summary.json" % args.out_prefix)


if __name__ == "__main__":
    main()
