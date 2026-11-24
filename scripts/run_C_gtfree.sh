#!/usr/bin/env bash
# GT-free C pipeline: build train/val boards with box target rule, retrain C,
# then evaluate val/test online with GT-free boards.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY=/home/hyw/miniforge3/envs/midog/bin/python
CPY=/home/hyw/miniforge3/envs/cellpose/bin/python

DATA=/home/hyw/dataset/chendawei/ICPR2012/classifier_C_gtfree
CANDS=/home/hyw/dataset/chendawei/ICPR2012/classifier_B/conf002
OUT_RUN="$ROOT/runs/classifier_C_gtfree/run_C"
OUT_EVAL="$ROOT/runs/online_C_gtfree/online_C_dedup16"

mkdir -p "$DATA" "$OUT_RUN" "$(dirname "$OUT_EVAL")"

echo "[1/3] build GT-free train/val boards"
"$CPY" -u src/coarse_yolo/icpr2012/build_cellpose_boards.py \
  --candidates-dir "$CANDS" \
  --out-dir "$DATA" \
  --splits train,val \
  --target-rule box

echo "[2/3] train C on GT-free boards"
"$PY" -u src/coarse_yolo/icpr2012/train_cls_boards.py \
  --data-dir "$DATA" \
  --out-dir "$OUT_RUN" \
  --epochs 100 \
  --batch 64 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --warmup 3 \
  --neg-per-pos 4 \
  --hard-frac 0.5 \
  --hard-score 0.01 \
  --gamma 2 \
  --eval-every 2 \
  --patience 20 \
  --workers 4 \
  --device 0

echo "[3/3] online GT-free val/test evaluation"
"$PY" -u src/coarse_yolo/icpr2012/eval_online_icpr2012.py \
  --cls "$OUT_RUN/best_cls.pth" \
  --out-prefix "$OUT_EVAL" \
  --splits val,test

echo "done"
