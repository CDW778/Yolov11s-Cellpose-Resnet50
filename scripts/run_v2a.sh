#!/bin/bash
set -euo pipefail
cd /home/hyw/chendawei/Yolov11s-cellpose-Resnet50/src/coarse_yolo
PY=/home/hyw/miniforge3/envs/midog/bin/python
export PYTHONUNBUFFERED=1

echo "[$(date '+%F %T')] v2a train start"
$PY -u train_yolo.py \
  --data /home/hyw/dataset/chendawei/ICPR2012/yolo_512_gtcrop4/data.yaml \
  --model ../../weights/yolo11s.pt \
  --epochs 80 \
  --batch 16 \
  --imgsz 512 \
  --workers 8 \
  --patience 0 \
  --save-period 5 \
  --lr0 0.0005 \
  --scale 0.4 \
  --mixup 0.1 \
  --close-mosaic 15 \
  --project ../../runs/yolo \
  --name icpr2012_v2a_gtcrop4

echo "[$(date '+%F %T')] v2a checkpoint selection + test report start"
$PY -u icpr2012/select_and_report.py \
  --run-dir ../../runs/yolo/icpr2012_v2a_gtcrop4 \
  --budget 30 \
  --out-prefix ../../results/baseline/yolo/v2a

echo "[$(date '+%F %T')] v2a all done"
