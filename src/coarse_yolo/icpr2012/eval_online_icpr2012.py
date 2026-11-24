#!/usr/bin/env python3
"""Online GT-free evaluation of the C classifier + fusion on ICPR2012.

For every YOLO candidate the 4-channel board is generated on the fly with
``make_board(..., y=0, polys=[])`` so no GT enters the model input.  GT is used
only to assign candidate labels (for AUC/AP) and for 5 um matching.

Protocol:
  val  : fit temperature T, select C threshold and fusion params
  test : apply frozen params once

Outputs per-candidate CSV and a metrics JSON under --out-prefix.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from cellpose import core, models
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src/coarse_yolo"))
sys.path.insert(0, str(PROJECT / "src/coarse_yolo/icpr2012"))
sys.path.insert(0, str(PROJECT / "src/cellpose_baseline"))

from configs.paths import (  # noqa: E402
    CELLPOSE_WEIGHT,
    DATA,
    IMAGES_ROOT,
    RAW_ROOT,
    RUNS,
    WEIGHTS,
)
from build_cellpose_boards import (  # noqa: E402
    SPLIT_PREFIXES,
    fallback_board,
    gt_polys,
    index_csvs,
    label_box,
    make_board,
)
from build_cls_candidates import dedupe_boxes  # noqa: E402
from make_black_boards import blacken_background  # noqa: E402
from dann_dataset import crop_patch  # noqa: E402
from stain_norm import load_ref, normalize_board  # noqa: E402
from calibrate_eval import fit_temperature, gt_points, match, res_px, softmax_np  # noqa: E402
from dann_model import DannResNet50  # noqa: E402
from infer_wsi import detect_wsi  # noqa: E402
from ultralytics import YOLO  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def board_tensor(res, in_chans=4):
    rgb = res["rgb"].astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    if in_chans == 3:
        x = rgb
    else:
        m = res["mask"].astype(np.float32) / 255.0
        x = np.concatenate([rgb, m[..., None]], axis=2)
    return torch.from_numpy(x.transpose(2, 0, 1)).float()


def raw_tensor(rgb):
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).float()


def run_logits(model, tensors, device, batch):
    logits = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch):
            xb = torch.stack(tensors[i : i + batch]).to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=str(device).startswith("cuda")):
                lg = model.predict_only(xb)
            logits.append(lg.float().cpu().numpy())
    if not logits:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(logits, 0)


def site_rows(yolo, cp, model, csvs, split, device, args, board_args):
    stain_ref = (
        load_ref(args.stain_ref)
        if (args.board_mode == "board" and args.stain_norm != "none")
        else None
    )
    rows = []
    items = [
        p
        for p in sorted((IMAGES_ROOT / split).glob("*.bmp"))
        if p.stem[:3] in SPLIT_PREFIXES[split]
    ]
    if getattr(args, "limit_slides", 0):
        items = items[: args.limit_slides]
    print("[%s] slides=%d" % (split, len(items)), flush=True)
    t0 = time.time()
    for si, bmp in enumerate(items, 1):
        base = bmp.stem
        polys = gt_polys(csvs[base])
        img = cv2.cvtColor(cv2.imread(str(bmp)), cv2.COLOR_BGR2RGB)
        boxes, scores, _ = detect_wsi(
            yolo,
            str(bmp),
            tile=args.tile,
            overlap=args.overlap,
            conf=args.conf,
            tile_iou=args.tile_iou,
            global_iou=args.global_iou,
            center_dist=None,
            batch=args.yolo_batch,
            device=args.device,
        )
        # Match build_cls_candidates.py: pure 16 px centre dedupe.
        boxes, scores = dedupe_boxes(boxes, scores, args.center_dist)
        tensors, meta = [], []
        for box, score in zip(boxes, scores):
            # GT only for the label; the board itself is GT-free.
            y, _qual, _cov, _iou = label_box(box, polys)
            fallback = 0
            if args.board_mode == "raw":
                rgb = crop_patch(
                    img,
                    (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    size=args.board,
                )
                tensors.append(raw_tensor(rgb))
                n_cells = 0
            else:
                res = make_board(img, box, [], [], 0, cp, board_args)
                if res is None:
                    res = fallback_board(img, box, board_args)
                    fallback = 1
                if stain_ref is not None:
                    res["rgb"] = normalize_board(res["rgb"], args.stain_norm, stain_ref)
                if args.board_bg == "black":
                    res["rgb"], _bg = blacken_background(res["rgb"], args.black_tol)
                tensors.append(board_tensor(res, args.in_chans))
                n_cells = int(res["n_cells"])
            meta.append(
                {
                    "split": split,
                    "base": base,
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "cx": float((box[0] + box[2]) * 0.5),
                    "cy": float((box[1] + box[3]) * 0.5),
                    "yolo_score": float(score),
                    "y": int(y),
                    "fallback": int(fallback),
                    "n_cells": n_cells,
                }
            )
        logits = run_logits(model, tensors, device, args.cls_batch)
        probs = softmax_np(logits, 1.0)[:, 1] if len(logits) else np.zeros((0,))
        for m, lg, p in zip(meta, logits, probs):
            m["logit0"] = float(lg[0])
            m["logit1"] = float(lg[1])
            m["p"] = float(p)
            rows.append(m)
        if si % 2 == 0 or si == len(items):
            print(
                "  %d/%d candidates=%d elapsed=%.0fs"
                % (si, len(items), len(rows), time.time() - t0),
                flush=True,
            )
    return rows


def cls_metrics(rows, p):
    idx = [i for i, r in enumerate(rows) if int(r["y"]) >= 0]
    y = np.array([int(rows[i]["y"]) for i in idx], dtype=np.int64)
    pp = np.asarray([p[i] for i in idx], dtype=np.float32)
    auc = float(roc_auc_score(y, pp)) if len(np.unique(y)) > 1 else None
    ap = float(average_precision_score(y, pp)) if len(np.unique(y)) > 1 else None
    best = {"f1": -1.0}
    for t in np.arange(0.01, 0.995, 0.01):
        pred = pp >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        if F > best["f1"]:
            best = {
                "f1": float(F), "thr": float(t), "precision": float(P),
                "recall": float(R), "tp": tp, "fp": fp, "fn": fn,
            }
    return {"auc": auc, "ap": ap, "best_f1": best}


def e2e_scores(rows, scores, csvs, thr):
    by = defaultdict(list)
    for r, s in zip(rows, scores):
        if float(s) >= thr:
            by[r["base"]].append((float(r["cx"]), float(r["cy"]), float(s)))
    TP = FP = FN = 0
    for base, dets in by.items():
        tp, fp, fn = match(dets, gt_points(csvs[base]), res_px(base))
        TP += tp
        FP += fp
        FN += fn
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    return {
        "thr": float(thr), "tp": int(TP), "fp": int(FP), "fn": int(FN),
        "precision": float(P), "recall": float(R),
        "f1": float(2 * P * R / (P + R)) if P + R else 0.0,
        "detections": int((np.asarray(scores) >= thr).sum()),
    }


def dedup_rows(rows, scores, thr, dist):
    """Greedy center dedup after score threshold; returns (rows, scores)."""
    keep = []
    for i in np.argsort(-np.asarray(scores)):
        s = float(scores[i])
        if s < thr:
            break
        r = rows[i]
        ok = True
        for j in keep:
            q = rows[j]
            if (r["cx"] - q["cx"]) ** 2 + (r["cy"] - q["cy"]) ** 2 < dist * dist:
                ok = False
                break
        if ok:
            keep.append(int(i))
    return [rows[i] for i in keep], np.asarray([scores[i] for i in keep], dtype=np.float32)


def best_e2e(rows, scores, csvs, dedup_px=None):
    best = {"f1": -1.0}
    for t in np.arange(0.01, 0.995, 0.01):
        if dedup_px:
            rr, ss = dedup_rows(rows, scores, float(t), dedup_px)
            m = e2e_scores(rr, ss, csvs, 0.0)
            m["thr"] = float(t)
        else:
            m = e2e_scores(rows, scores, csvs, float(t))
        if m["f1"] > best["f1"] or (
            m["f1"] == best["f1"] and m["detections"] < best.get("detections", 10**9)
        ):
            best = m
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default=str(WEIGHTS / "yolo/epoch25.pt"))
    ap.add_argument("--cls",
                    default=str(RUNS / "classifier_C_gtfree/run_C/best_cls.pth"))
    ap.add_argument("--out-prefix", default=str(RUNS / "online_C/online_C"))
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--conf", type=float, default=0.002)
    ap.add_argument("--tile-iou", type=float, default=0.3)
    ap.add_argument("--global-iou", type=float, default=0.5)
    ap.add_argument("--center-dist", type=float, default=16.0)
    ap.add_argument("--yolo-batch", type=int, default=16)
    ap.add_argument("--cls-batch", type=int, default=128)
    ap.add_argument("--device", default="0")
    ap.add_argument("--margin", type=int, default=16)
    ap.add_argument("--min-crop", type=int, default=64)
    ap.add_argument("--board", type=int, default=224)
    ap.add_argument("--fill", type=float, default=0.9)
    ap.add_argument("--flow-threshold", type=float, default=1.0)
    ap.add_argument("--cellprob", type=float, default=-3.0)
    ap.add_argument("--min-size", type=int, default=15)
    ap.add_argument("--comp-dilate", type=int, default=2)
    ap.add_argument("--min-cell-area", type=int, default=20)
    ap.add_argument("--yolo-only-score", type=float, default=0.40)
    ap.add_argument("--yolo-only-dedup", type=float, default=24.0)
    ap.add_argument("--stain-norm", default="none", choices=["none", "chroma", "lmean", "lab"])
    ap.add_argument("--stain-ref", default=str(DATA / "stain_ref_train.json"))
    ap.add_argument("--board-mode", default="board", choices=["board", "raw"])
    ap.add_argument("--board-bg", default="white", choices=["white", "black"])
    ap.add_argument("--in-chans", type=int, default=4, choices=[3, 4])
    ap.add_argument("--black-tol", type=int, default=25)
    ap.add_argument("--limit-slides", type=int, default=0)
    args = ap.parse_args()
    if args.board_mode == "raw":
        if args.in_chans != 3:
            ap.error("--board-mode raw requires --in-chans 3")
        if args.board_bg != "white":
            ap.error("--board-bg only applies to --board-mode board")
    elif args.in_chans not in (3, 4):
        ap.error("--board-mode board requires --in-chans 3 or 4")

    device = "cuda:" + args.device if torch.cuda.is_available() else "cpu"
    csvs = index_csvs(RAW_ROOT)
    yolo = YOLO(args.yolo)
    cp = None
    if args.board_mode == "board":
        cp = models.CellposeModel(gpu=core.use_gpu(), pretrained_model=str(CELLPOSE_WEIGHT))
    model = DannResNet50(num_classes=2, in_chans=args.in_chans)
    model.load_state_dict(torch.load(args.cls, map_location="cpu"))
    model.to(device)
    model.eval()

    board_args = SimpleNamespace(
        margin=args.margin,
        min_crop=args.min_crop,
        board=args.board,
        fill=args.fill,
        flow_threshold=args.flow_threshold,
        cellprob=args.cellprob,
        min_size=args.min_size,
        comp_dilate=args.comp_dilate,
        min_cell_area=args.min_cell_area,
    )
    args.board_args = board_args

    data = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        data[split] = site_rows(yolo, cp, model, csvs, split, device, args, board_args)

    va = data["val"]
    va_p = np.asarray([r["p"] for r in va], dtype=np.float32)
    va_y = np.array([int(r["y"]) for r in va], dtype=np.int64)
    idx = np.where(va_y >= 0)[0]
    logits_va = np.stack(
        [[r["logit0"] for r in va], [r["logit1"] for r in va]], axis=1
    ).astype(np.float32)
    T, nll = fit_temperature(logits_va[idx], va_y[idx])
    print("temperature=%.4f val_nll=%.4f" % (T, nll), flush=True)

    p_cal = {}
    for split, rows in data.items():
        lg = np.stack(
            [[r["logit0"] for r in rows], [r["logit1"] for r in rows]], axis=1
        ).astype(np.float32)
        p_cal[split] = softmax_np(lg, T)[:, 1]

    # C calibrated: threshold selected on val, frozen to test.
    c_val_best = best_e2e(va, p_cal["val"], csvs)
    c_res = {}
    for split, rows in data.items():
        p = p_cal[split]
        c_res[split] = {
            "cls": cls_metrics(rows, p),
            "e2e_best_on_split": best_e2e(rows, p, csvs),
        }
    if "test" in data:
        c_res["test"]["e2e_at_val_thr"] = e2e_scores(
            data["test"], p_cal["test"], csvs, c_val_best["thr"]
        )

    # Fusion f = p_cal^a * yolo^b; params selected on val only.
    fusion_best = {"f1": -1.0}
    for a in (0.0, 0.5, 1.0, 1.5, 2.0):
        for b in (0.0, 0.5, 1.0, 1.5, 2.0):
            if a == 0.0 and b == 0.0:
                continue
            fs = np.power(p_cal["val"], a) * np.power(
                np.asarray([r["yolo_score"] for r in va], dtype=np.float32), b
            )
            m = best_e2e(va, fs, csvs)
            if m["f1"] > fusion_best["f1"] or (
                m["f1"] == fusion_best["f1"]
                and m["detections"] < fusion_best.get("detections", 10**9)
            ):
                fusion_best = {**m, "a": float(a), "b": float(b)}
    a, b = fusion_best["a"], fusion_best["b"]
    f_val = np.power(p_cal["val"], a) * np.power(
        np.asarray([r["yolo_score"] for r in va], dtype=np.float32), b
    )
    fusion_res = {
        "params": {"a": a, "b": b, "thr": fusion_best["thr"]},
        "val": fusion_best,
    }
    if "test" in data:
        f_test = np.power(p_cal["test"], a) * np.power(
            np.asarray([r["yolo_score"] for r in data["test"]], dtype=np.float32), b
        )
        fusion_res["test"] = e2e_scores(
            data["test"], f_test, csvs, fusion_best["thr"]
        )

    # YOLO-only reference: score threshold + center dedup, val selected.
    y_sc = np.asarray([r["yolo_score"] for r in va], dtype=np.float32)
    yolo_best = best_e2e(va, y_sc, csvs, dedup_px=args.yolo_only_dedup)
    yolo_test = None
    if "test" in data:
        y_sc_test = np.asarray(
            [r["yolo_score"] for r in data["test"]], dtype=np.float32
        )
        rr, ss = dedup_rows(
            data["test"], y_sc_test, yolo_best["thr"], args.yolo_only_dedup
        )
        yolo_test = e2e_scores(rr, ss, csvs, 0.0)
        yolo_test["thr"] = yolo_best["thr"]

    metrics = {
        "inputs": {
            "yolo": args.yolo,
            "cls": args.cls,
            "conf": args.conf,
            "center_dist": args.center_dist,
            "board_rule": "GT-free make_board(y=0, polys=[])"
            if args.board_mode == "board" else "raw 224 RGB crop (bbox centre)",
            "board_mode": args.board_mode,
            "board_bg": args.board_bg,
            "in_chans": args.in_chans,
            "stain_norm": args.stain_norm,
            "stain_ref": args.stain_ref if args.stain_norm != "none" else None,
        },
        "temperature": {"T": T, "val_nll": nll},
        "c_calibrated": {
            "params": {"thr": c_val_best["thr"]},
            "val": c_res["val"],
            **({"test": c_res["test"]} if "test" in c_res else {}),
        },
        "fusion": fusion_res,
        "yolo_only": {
            "params": {"thr": yolo_best["thr"], "dedup_px": args.yolo_only_dedup},
            "val": yolo_best,
            "test": yolo_test,
        },
        "counts": {
            split: {
                "candidates": len(rows),
                "y1": int(sum(int(r["y"]) == 1 for r in rows)),
                "y0": int(sum(int(r["y"]) == 0 for r in rows)),
                "yneg": int(sum(int(r["y"]) < 0 for r in rows)),
                "fallback": int(sum(int(r["fallback"]) for r in rows)),
            }
            for split, rows in data.items()
        },
    }

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for split, rows in data.items():
        p = p_cal[split]
        f = (
            np.power(p, a)
            * np.power(np.asarray([r["yolo_score"] for r in rows], dtype=np.float32), b)
        )
        with open(str(out_prefix) + "_" + split + ".csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "base", "x1", "y1", "x2", "y2", "cx", "cy", "yolo_score",
                "y", "fallback", "n_cells", "p", "p_cal", "f",
            ])
            for r, pp, ff in zip(rows, p, f):
                w.writerow([
                    r["base"], r["x1"], r["y1"], r["x2"], r["y2"], r["cx"], r["cy"],
                    r["yolo_score"], r["y"], r["fallback"], r["n_cells"],
                    r["p"], float(pp), float(ff),
                ])
    with open(str(out_prefix) + "_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n=== counts ===")
    print(json.dumps(metrics["counts"], indent=2))
    print("\n=== C calibrated ===")
    print(json.dumps(metrics["c_calibrated"], indent=2))
    print("\n=== fusion ===")
    print(json.dumps(metrics["fusion"], indent=2))
    print("\n=== yolo only ===")
    print(json.dumps(metrics["yolo_only"], indent=2))
    print("wrote", out_prefix)


if __name__ == "__main__":
    main()
