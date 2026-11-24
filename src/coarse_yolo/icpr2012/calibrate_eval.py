#!/usr/bin/env python3
"""Temperature-calibrated evaluation of the C classifier.

1. Fit a temperature T on val logits (NLL).
2. Report classification metrics before/after calibration.
3. Select the e2e threshold on val (5 um matching), freeze it to test.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.paths import CELLPOSE_BASELINE, RAW_ROOT  # noqa: E402
sys.path.insert(0, str(CELLPOSE_BASELINE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dann_model import DannResNet50  # noqa: E402
from train_cls_boards import BoardDataset, load_manifest  # noqa: E402

SRC = RAW_ROOT


def index_csvs():
    out = {}
    for d in SRC.glob("*_v2"):
        for p in d.glob("*.csv"):
            out[p.stem] = p
    return out


def gt_points(csv_path):
    pts = []
    for line in open(csv_path):
        line = line.strip()
        if not line:
            continue
        v = np.array([int(t) for t in line.split(",") if t != ""], dtype=np.float64)
        pts.append((float(v[0::2].mean()), float(v[1::2].mean())))
    return pts


def res_px(base):
    return 5.0 / (0.2456 if str(base).startswith("A") else (0.2273 + 0.22753) / 2.0)


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


def e2e(rows, probs, csvs, thr):
    by = defaultdict(list)
    for r, p in zip(rows, probs):
        if p >= thr:
            by[r["base"]].append((float(r["cx"]), float(r["cy"]), float(p)))
    TP = FP = FN = 0
    for base, dets in by.items():
        tp, fp, fn = match(dets, gt_points(csvs[base]), res_px(base))
        TP += tp
        FP += fp
        FN += fn
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    return {
        "thr": float(thr), "tp": TP, "fp": FP, "fn": FN,
        "precision": P, "recall": R, "f1": 2 * P * R / (P + R) if P + R else 0.0,
        "detections": int((probs >= thr).sum()),
    }


def cls_metrics(rows, probs):
    idx = [i for i, r in enumerate(rows) if int(r["y"]) >= 0]
    y = np.array([int(rows[i]["y"]) for i in idx], dtype=np.int64)
    p = np.array([probs[i] for i in idx], dtype=np.float32)
    auc = float(roc_auc_score(y, p))
    ap = float(average_precision_score(y, p))
    best = {"f1": -1.0}
    for t in np.arange(0.01, 0.995, 0.01):
        pred = p >= t
        tp = int(((pred) & (y == 1)).sum())
        fp = int(((pred) & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        if F > best["f1"]:
            best = {"f1": F, "thr": float(t), "precision": P, "recall": R, "tp": tp, "fp": fp, "fn": fn}
    return {"auc": auc, "ap": ap, "best_f1": best}


def predict_logits(rows, weight, device, batch, workers):
    model = DannResNet50(num_classes=2, in_chans=4)
    model.load_state_dict(torch.load(weight, map_location="cpu"))
    model.to(device)
    ds = BoardDataset(rows, augment=False)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    model.eval()
    logits = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=str(device).startswith("cuda")):
                lg = model.predict_only(xb)
            logits.append(lg.float().cpu().numpy())
    return np.concatenate(logits, 0)


def softmax_np(logits, T=1.0):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits, y):
    def nll(logT):
        T = np.exp(logT)
        p = softmax_np(logits, T)[:, 1]
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(-1.5, 1.5), method="bounded")
    return float(np.exp(res.x)), float(res.fun)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    device = "cuda:" + args.device if torch.cuda.is_available() else "cpu"
    csvs = index_csvs()
    data = {}
    for sp in ("val", "test"):
        rows = load_manifest(Path(args.data_dir) / ("manifest_%s.csv" % sp))
        logits = predict_logits(rows, args.weight, device, args.batch, args.workers)
        data[sp] = (rows, logits)

    va_rows, va_logits = data["val"]
    y_va = np.array([int(r["y"]) for r in va_rows if int(r["y"]) >= 0], dtype=np.int64)
    idx_va = [i for i, r in enumerate(va_rows) if int(r["y"]) >= 0]
    T, nll = fit_temperature(va_logits[idx_va], y_va)
    print("temperature=%.4f val_nll=%.4f" % (T, nll), flush=True)

    result = {"temperature": T, "val_nll": nll, "raw": {}, "calibrated": {}}
    for mode in ("raw", "calibrated"):
        out = {}
        for sp in ("val", "test"):
            rows, logits = data[sp]
            probs = softmax_np(logits, 1.0 if mode == "raw" else T)[:, 1]
            cm = cls_metrics(rows, probs)
            best = {"f1": -1.0}
            for t in np.arange(0.01, 0.995, 0.01):
                m = e2e(rows, probs, csvs, float(t))
                if m["f1"] > best["f1"]:
                    best = m
            out[sp] = {"cls": cm, "e2e_best": best, "probs": probs}
        # freeze val e2e threshold to test
        thr = out["val"]["e2e_best"]["thr"]
        out["test"]["e2e_at_val_thr"] = e2e(data["test"][0], out["test"]["probs"], csvs, thr)
        out["val"]["e2e_at_val_thr"] = out["val"]["e2e_best"]
        result[mode] = {
            "temperature": 1.0 if mode == "raw" else T,
            "val": {k: v for k, v in out["val"].items() if k != "probs"},
            "test": {k: v for k, v in out["test"].items() if k != "probs"},
        }
        print("\n=== %s ===" % mode)
        print("val cls AUC=%.4f AP=%.4f | e2e best F1=%.4f @%.2f" % (
            out["val"]["cls"]["auc"], out["val"]["cls"]["ap"],
            out["val"]["e2e_best"]["f1"], out["val"]["e2e_best"]["thr"]))
        print("test cls AUC=%.4f AP=%.4f | e2e @val-thr F1=%.4f P=%.4f R=%.4f TP=%d FP=%d FN=%d" % (
            out["test"]["cls"]["auc"], out["test"]["cls"]["ap"],
            out["test"]["e2e_at_val_thr"]["f1"], out["test"]["e2e_at_val_thr"]["precision"],
            out["test"]["e2e_at_val_thr"]["recall"], out["test"]["e2e_at_val_thr"]["tp"],
            out["test"]["e2e_at_val_thr"]["fp"], out["test"]["e2e_at_val_thr"]["fn"]))

    prefix = args.out_prefix
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    with open(prefix + ".json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote", prefix + ".json")


if __name__ == "__main__":
    main()
