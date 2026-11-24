#!/usr/bin/env python3
"""Train experiment-C classifier: RGB cell-on-white + target-mask (4 channels)."""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.paths import CELLPOSE_BASELINE, DATA, IMAGENET_V1  # noqa: E402
sys.path.insert(0, str(CELLPOSE_BASELINE))
from dann_model import DannResNet50, load_pretrained_dann  # noqa: E402
from stain_norm import load_ref, normalize_board  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _cell_mask(rgb):
    """Pixels that are not the synthetic white background."""
    return np.any(rgb < 245, axis=2)


def _jitter_cells(rgb, rng):
    """Stain/colour jitter applied to cells only; keep the white background white."""
    cell = _cell_mask(rgb)
    if not cell.any():
        return rgb
    out = rgb.astype(np.float32)
    out *= 1.0 + rng.uniform(-0.15, 0.15)                  # brightness
    mean = float(out[cell].mean())
    out = (out - mean) * (1.0 + rng.uniform(-0.15, 0.15)) + mean  # contrast
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= 1.0 + rng.uniform(-0.15, 0.15)           # saturation
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-0.03, 0.03) * 180.0) % 180.0
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    out[~cell] = 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _random_affine(rgb, mask, rng, scale_range=(0.85, 1.10), max_shift=6.0, max_angle=8.0):
    h, w = rgb.shape[:2]
    scale = rng.uniform(*scale_range)
    angle = rng.uniform(-max_angle, max_angle)
    tx = rng.uniform(-max_shift, max_shift)
    ty = rng.uniform(-max_shift, max_shift)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    rgb2 = cv2.warpAffine(
        rgb, M, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    m2 = cv2.warpAffine(
        mask, M, (w, h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return rgb2, m2


def _blur_noise(rgb, rng):
    if rng.rand() < 0.2:
        rgb = cv2.GaussianBlur(rgb, (3, 3), rng.uniform(0.1, 1.0))
    if rng.rand() < 0.2:
        cell = _cell_mask(rgb)
        if cell.any():
            out = rgb.astype(np.float32)
            noise = rng.normal(0.0, rng.uniform(1.0, 5.0), out.shape).astype(np.float32)
            out[cell] += noise[cell]
            rgb = np.clip(out, 0, 255).astype(np.uint8)
    return rgb


def _cutout_background(rgb, mask, rng):
    """Small white cutouts outside the target cell (avoid changing the label)."""
    if rng.rand() >= 0.2:
        return rgb
    h, w = rgb.shape[:2]
    target = mask > 0
    for _ in range(int(rng.randint(1, 3))):
        cw = int(rng.randint(12, 48))
        ch = int(rng.randint(12, 48))
        x = int(rng.randint(0, max(1, w - cw)))
        y = int(rng.randint(0, max(1, h - ch)))
        if target[y:y + ch, x:x + cw].any():
            continue
        rgb = rgb.copy()
        rgb[y:y + ch, x:x + cw] = 255
    return rgb


class BoardDataset(Dataset):
    def __init__(self, rows, augment=False, seed=0, aug_level="basic",
                 stain_norm="none", stain_ref=None, in_chans=4, cache=None):
        self.rows = rows
        self.augment = augment
        self.aug_level = aug_level
        self.in_chans = in_chans
        self.cache = cache
        self.rng = np.random.RandomState(seed)
        self.stain_norm = stain_norm or "none"
        self.stain_ref = None
        if self.stain_norm != "none":
            if stain_ref is None:
                raise ValueError("stain_norm=%s needs --stain-ref" % self.stain_norm)
            self.stain_ref = stain_ref if isinstance(stain_ref, dict) else load_ref(stain_ref)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        key = (r["rgb"], self.in_chans)
        if self.cache is not None and key in self.cache:
            rgb, m = self.cache[key]
        else:
            rgb = cv2.cvtColor(cv2.imread(r["rgb"]), cv2.COLOR_BGR2RGB)
            if rgb is None:
                raise RuntimeError("bad image: %s" % r["rgb"])
            m = None
            if self.in_chans == 4:
                m = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
                if m is None:
                    raise RuntimeError("bad mask: %s" % r["mask"])
            if self.cache is not None:
                self.cache[key] = (rgb, m)
        if self.stain_norm != "none":
            rgb = normalize_board(rgb, self.stain_norm, self.stain_ref)
        if self.augment:
            if self.rng.rand() < 0.5:
                rgb = rgb[:, ::-1]
                if m is not None:
                    m = m[:, ::-1]
            if self.rng.rand() < 0.5:
                rgb = rgb[::-1]
                if m is not None:
                    m = m[::-1]
            k = self.rng.randint(4)
            if k:
                rgb = np.rot90(rgb, k)
                if m is not None:
                    m = np.rot90(m, k)
            if self.aug_level == "strong":
                rgb = np.ascontiguousarray(rgb)
                if m is not None:
                    m = np.ascontiguousarray(m)
                    rgb, m = _random_affine(rgb, m, self.rng)
                rgb = _jitter_cells(rgb, self.rng)
                rgb = _blur_noise(rgb, self.rng)
                if m is not None:
                    rgb = _cutout_background(rgb, m, self.rng)
        rgb = np.ascontiguousarray(rgb).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        if self.in_chans == 3:
            x = rgb
        else:
            m = np.ascontiguousarray(m).astype(np.float32) / 255.0
            x = np.concatenate([rgb, m[..., None]], axis=2)
        return torch.from_numpy(x.transpose(2, 0, 1)).float(), int(r["y"])


def _bn_eval(module):
    """Keep frozen backbone BatchNorm in eval mode (no running-stat updates)."""
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


def _param_groups(model, args):
    """Optionally give the backbone a smaller LR than the classifier head."""
    head = [p for n, p in model.named_parameters()
            if p.requires_grad and n.startswith("mitosis_classifier.")]
    backbone = [p for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith("mitosis_classifier.")]
    if args.backbone_lr is None:
        return [{"params": backbone + head, "lr": args.lr}]
    print("layer-wise lr: backbone=%.2e head=%.2e" % (args.backbone_lr, args.lr),
          flush=True)
    groups = [{"params": head, "lr": args.lr},
              {"params": backbone, "lr": args.backbone_lr}]
    return [g for g in groups if g["params"]]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.view(-1, 1)).squeeze(1)
        pt = logpt.exp()
        return (-((1.0 - pt) ** self.gamma) * logpt).mean()


def load_manifest(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def evaluate(model, rows, device, batch, workers, stain_norm="none", stain_ref=None,
             in_chans=4, cache=None):
    ds = BoardDataset(rows, augment=False, stain_norm=stain_norm, stain_ref=stain_ref,
                      in_chans=in_chans, cache=cache)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=str(device).startswith("cuda")):
                lg = model.predict_only(xb)
            probs.extend(torch.softmax(lg.float(), 1)[:, 1].cpu().numpy().tolist())
    probs = np.asarray(probs, dtype=np.float32)
    y = np.array([int(r["y"]) for r in rows], dtype=np.int64)
    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)
    best = {"f1": -1.0}
    for thr in np.arange(0.01, 0.995, 0.01):
        pred = probs >= thr
        tp = int(((pred) & (y == 1)).sum())
        fp = int(((pred) & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best["f1"]:
            best = {"f1": f1, "thr": float(thr), "p": p, "r": r, "tp": tp, "fp": fp, "fn": fn}
    return {"auc": float(auc), "ap": float(ap), **best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone-lr", type=float, default=None,
                    help="separate LR for the backbone; None uses --lr for all")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--neg-per-pos", type=int, default=4)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--hard-score", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--aug-level", default="basic", choices=["basic", "strong"])
    ap.add_argument("--in-chans", type=int, default=4, choices=[3, 4])
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="freeze ResNet backbone, train only the classifier head")
    ap.add_argument("--stain-norm", default="none", choices=["none", "chroma", "lab", "lmean"])
    ap.add_argument("--stain-ref", default=str(DATA / "stain_ref_train.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = "cuda:" + args.device if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr_all = load_manifest(data_dir / "manifest_train.csv")
    va_all = load_manifest(data_dir / "manifest_val.csv")
    tr = [r for r in tr_all if int(r["y"]) >= 0]
    va = [r for r in va_all if int(r["y"]) >= 0]
    pos = [r for r in tr if int(r["y"]) == 1]
    neg = [r for r in tr if int(r["y"]) == 0]
    hard = [r for r in neg if int(r.get("hard", 0) or 0) == 1]
    if not hard:
        hard = [r for r in neg if float(r["yolo_score"]) >= args.hard_score]
    print(
        "train usable=%d pos=%d neg=%d hard=%d | val usable=%d pos=%d"
        % (len(tr), len(pos), len(neg), len(hard), len(va), sum(int(r["y"]) == 1 for r in va)),
        flush=True,
    )

    model = DannResNet50(num_classes=2, in_chans=args.in_chans)
    missing, unexpected = load_pretrained_dann(model, IMAGENET_V1)
    print("init missing=%d unexpected=%d" % (len(missing), len(unexpected)), flush=True)
    model.to(device)

    crit = FocalLoss(args.gamma)
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("mitosis_classifier."):
                p.requires_grad = False
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print("freeze-backbone: trainable=%d / %d" % (n_train, n_total), flush=True)
    opt = torch.optim.AdamW(
        _param_groups(model, args),
        lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda ep: (ep + 1) / max(1, args.warmup)
        if ep < args.warmup
        else 0.5 * (1.0 + np.cos(np.pi * (ep - args.warmup) / max(1, args.epochs - args.warmup))),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith("cuda"))

    log_path = out / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "loss", "lr", "train_auc", "train_ap", "train_best_f1",
            "val_auc", "val_ap", "val_best_f1", "val_thr", "sec",
        ])

    best_ap = -1.0
    best_f1 = -1.0
    bad = 0
    for ep in range(args.epochs):
        t0 = time.time()
        n_neg = len(pos) * args.neg_per_pos
        n_hard = min(len(hard), int(round(n_neg * args.hard_frac)))
        n_rand = n_neg - n_hard
        epoch_rows = list(pos)
        if n_hard:
            epoch_rows += [hard[i] for i in rng.choice(len(hard), size=n_hard, replace=len(hard) < n_hard)]
        if n_rand:
            epoch_rows += [neg[i] for i in rng.choice(len(neg), size=n_rand, replace=len(neg) < n_rand)]
        rng.shuffle(epoch_rows)

        ds = BoardDataset(epoch_rows, augment=True, seed=args.seed + ep,
                          aug_level=args.aug_level, stain_norm=args.stain_norm,
                          stain_ref=args.stain_ref, in_chans=args.in_chans)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True, drop_last=False)
        model.train()
        if args.freeze_backbone:
            for m in (model.stem, model.layer2, model.layer3, model.layer4):
                m.apply(_bn_eval)
        tot = 0.0
        nb = 0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=str(device).startswith("cuda")):
                logits = model.predict_only(xb)
                loss = crit(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        lr = opt.param_groups[0]["lr"]
        msg = "[ep %d/%d] loss=%.4f lr=%.2e n=%d time=%.1fs" % (
            ep + 1, args.epochs, tot / max(1, nb), lr, len(epoch_rows), time.time() - t0)
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m = evaluate(model, va, device, 128, args.workers,
                         stain_norm=args.stain_norm, stain_ref=args.stain_ref,
                         in_chans=args.in_chans)
            tm = evaluate(model, tr, device, 128, args.workers,
                          stain_norm=args.stain_norm, stain_ref=args.stain_ref,
                          in_chans=args.in_chans)
            msg += " | train AUC=%.4f AP=%.4f | val AUC=%.4f AP=%.4f bestF1=%.4f@%.2f" % (
                tm["auc"], tm["ap"], m["auc"], m["ap"], m["f1"], m["thr"])
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    ep + 1, tot / max(1, nb), lr,
                    tm["auc"], tm["ap"], tm["f1"],
                    m["auc"], m["ap"], m["f1"], m["thr"], time.time() - t0,
                ])
            if m["ap"] > best_ap:
                best_ap = m["ap"]
                bad = 0
                torch.save(model.state_dict(), out / "best_cls.pth")
                (out / "best_epoch.txt").write_text("%d\n" % (ep + 1))
                msg += " *saved-ap*"
            else:
                bad += 1
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                torch.save(model.state_dict(), out / "best_f1_cls.pth")
                (out / "best_f1_epoch.txt").write_text("%d\n" % (ep + 1))
                msg += " *saved-f1*"
            if bad >= args.patience:
                print(msg + " | early-stop", flush=True)
                break
        print(msg, flush=True)

    torch.save(model.state_dict(), out / "last_cls.pth")
    print("done. best val AP=%.4f -> %s" % (best_ap, out / "best_cls.pth"), flush=True)


if __name__ == "__main__":
    main()
