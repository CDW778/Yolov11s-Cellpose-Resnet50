#!/usr/bin/env python3
"""Train the stage-1 coarse candidate detector on the tiled MIDOG++ data."""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.paths import YOLO_V2A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=str(YOLO_V2A / "data.yaml"),
    )
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="yolo11s_pilot")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--scale", type=float, default=0.0)
    ap.add_argument("--mixup", type=float, default=0.3)
    ap.add_argument("--degrees", type=float, default=180.0)
    ap.add_argument("--save-period", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--val", dest="do_val", action="store_true", default=True)
    ap.add_argument("--no-val", dest="do_val", action="store_false")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit("missing data yaml: %s" % data_path)

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save_period=args.save_period,
        optimizer="AdamW",
        lr0=args.lr0,
        cos_lr=True,
        amp=True,
        cache=False,
        rect=False,
        single_cls=True,
        cls=1.2,
        degrees=args.degrees,
        scale=args.scale,
        mixup=args.mixup,
        close_mosaic=args.close_mosaic,
        seed=args.seed,
        deterministic=True,
        fraction=args.fraction,
        val=args.do_val,
        plots=True,
        exist_ok=True,
        resume=args.resume,
        verbose=True,
    )


if __name__ == "__main__":
    main()
