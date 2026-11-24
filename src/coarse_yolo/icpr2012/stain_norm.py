#!/usr/bin/env python3
"""Stain normalisation for 224x224 white-background cell boards.

Three deterministic modes:

  chroma : Reinhard mean/std matching on LAB a/b only; L (brightness) is kept
           so the dark mitotic nucleus signal is not normalised away.
  lmean  : chroma matching plus an L mean shift; L contrast (std) is kept.
  lab    : Reinhard mean/std matching on L, a and b (full LAB).

Only cell pixels (any RGB channel < ``cell_thresh``) are modified; the
background is left untouched.  The reference statistics must be computed from
the training split only (see compute_stain_reference.py).
"""

import json

import cv2
import numpy as np

DEFAULT_CELL_THRESH = 245
MIN_CELL_PIXELS = 50
EPS = 1e-3


def load_ref(path):
    with open(path, "r") as f:
        return json.load(f)


def cell_mask(rgb, thresh=DEFAULT_CELL_THRESH):
    """Pixels that are not the synthetic white background."""
    return np.any(rgb < thresh, axis=2)


def rgb_to_lab(rgb):
    """uint8 RGB -> float32 OpenCV LAB (L 0-255, a/b 0-255)."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def lab_to_rgb(lab):
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def lab_stats(rgb, thresh=DEFAULT_CELL_THRESH):
    """Mean/std/count of LAB values over cell pixels, or None if too few."""
    m = cell_mask(rgb, thresh)
    n = int(m.sum())
    if n < 1:
        return None
    v = rgb_to_lab(rgb)[m]
    return v.mean(0), v.std(0), n


def accumulate(rgb, acc, thresh=DEFAULT_CELL_THRESH):
    """Accumulate sum / sumsq / count for reference statistics."""
    m = cell_mask(rgb, thresh)
    n = int(m.sum())
    if n == 0:
        return
    v = rgb_to_lab(rgb)[m]
    acc["sum"] += v.sum(0)
    acc["sumsq"] += (v ** 2).sum(0)
    acc["n"] += n


def finalize(acc):
    n = max(int(acc["n"]), 1)
    mean = acc["sum"] / n
    var = np.maximum(acc["sumsq"] / n - mean ** 2, 1e-6)
    return {"mean": mean.tolist(), "std": np.sqrt(var).tolist(), "n_pixels": int(acc["n"])}


def new_acc():
    return {"sum": np.zeros(3, np.float64), "sumsq": np.zeros(3, np.float64), "n": 0}


def normalize_board(rgb, mode, ref, thresh=DEFAULT_CELL_THRESH, min_pixels=MIN_CELL_PIXELS):
    """Return a normalised uint8 RGB board.

    mode: "none" | "chroma" | "lab".
    ref:  dict loaded from compute_stain_reference.py output (uses ref["global"]).
    """
    if mode in (None, "", "none"):
        return rgb
    if ref is None:
        raise ValueError("stain normalisation requested but no reference given")
    if mode not in ("chroma", "lmean", "lab"):
        raise ValueError("unknown stain-norm mode: %s" % mode)

    m = cell_mask(rgb, thresh)
    if int(m.sum()) < min_pixels:
        return rgb

    lab = rgb_to_lab(rgb)
    v = lab[m]
    src_mean = v.mean(0)
    src_std = np.maximum(v.std(0), EPS)

    g = ref["global"]
    ref_mean = np.asarray(g["mean"], dtype=np.float32)
    ref_std = np.maximum(np.asarray(g["std"], dtype=np.float32), EPS)

    if mode in ("chroma", "lmean"):
        channels = (1, 2)
    else:
        channels = (0, 1, 2)
    for c in channels:
        lab[m, c] = (lab[m, c] - src_mean[c]) / src_std[c] * ref_std[c] + ref_mean[c]
    if mode == "lmean":
        lab[m, 0] = lab[m, 0] + (ref_mean[0] - src_mean[0])

    out = lab_to_rgb(lab)
    # Keep the original background (white for normal boards, raw crop for fallback).
    out[~m] = rgb[~m]
    return out
