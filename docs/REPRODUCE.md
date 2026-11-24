# Reproduce the ICPR2012 baseline

Two levels are supported.

## Environment

```bash
export ICPR=/home/hyw/dataset/chendawei/ICPR2012
export PY=/home/hyw/miniforge3/envs/midog/bin/python
export CPY=/home/hyw/miniforge3/envs/cellpose/bin/python
```

Versions: see `requirements.txt` (torch 2.4.1, ultralytics 8.3.253,
cellpose 4.1.1, numpy 1.24.4). Hardware used: RTX 2080 Ti 11 GB.

## Level A — exact replay (recommended)

Requires:

- `$ICPR/classifier_C_gtfree` (boards + masks + manifests)
- `$ICPR/classifier_B/conf002` (candidate npz files)
- `weights/yolo/epoch25.pt`
- `runs/classifier_C_gtfree/run_C/best_cls.pth`
- `weights/cellpose/cpsam_v2`

```bash
$PY src/coarse_yolo/icpr2012/eval_online_icpr2012.py \
  --cls runs/classifier_C_gtfree/run_C/best_cls.pth \
  --out-prefix runs/online_C_gtfree/online_C_dedup16 \
  --splits val,test
```

Expected:

| metric | value |
|---|---|
| candidate val AP | 0.7558 |
| candidate test AP | 0.8532 |
| fusion val F1 | 0.7544 |
| fusion test F1 | **0.8306** |
| T / a / b / threshold | 0.6577 / 0.5 / 0.5 / 0.35 |

The script fits `T` on val and searches `a,b,threshold` on val; test is applied
once. Test has 84 GT, so treat differences below ~0.04 as noise.

## Level B — rebuild from raw data

Level B also needs the Ultralytics YOLO11s pretrained checkpoint
(`weights/yolo11s.pt`) and the Cellpose model (`weights/cellpose/cpsam_v2`).

### 1. v1 tiles -> v2a tiles

```bash
$PY src/coarse_yolo/icpr2012/prepare_icpr_tiles.py \
  --src-anno $ICPR/raw \
  --images-root $ICPR/images \
  --output-root $ICPR/yolo_512_ov20 \
  --splits train,val --tile 512 --overlap 0.20

$PY src/coarse_yolo/icpr2012/prepare_icpr_v2.py \
  --v1-root $ICPR/yolo_512_ov20 \
  --output-root $ICPR/yolo_512_gtcrop4
```

`prepare_icpr_v2.py` reuses v1 background tiles and symlinks the v1 val split.

### 2. YOLO11s

`scripts/run_v2a.sh` contains the exact 80-epoch recipe; the short version:

```bash
$PY src/coarse_yolo/train_yolo.py \
  --data $ICPR/yolo_512_gtcrop4/data.yaml \
  --model weights/yolo11s.pt \
  --epochs 80 --batch 16 --imgsz 512 --workers 8 --patience 0 \
  --save-period 5 --lr0 0.0005 --scale 0.4 --mixup 0.1 \
  --close-mosaic 15 --project runs/yolo --name icpr2012_v2a_gtcrop4
```

The baseline uses `runs/yolo/icpr2012_v2a_gtcrop4` epoch 25
(`weights/yolo/epoch25.pt`), selected with
`src/coarse_yolo/icpr2012/select_and_report.py`.

### 3. Candidates

```bash
$PY src/coarse_yolo/icpr2012/build_cls_candidates.py \
  --detector weights/yolo/epoch25.pt \
  --out-dir $ICPR/classifier_B/conf002 \
  --splits train,val,test \
  --conf-train 0.002 --conf-eval 0.002 \
  --dedupe-dist 16 --tile 512 --overlap 0.20
```

Expected counts: 4233 / 1200 / 660.

### 4. GT-free boards

```bash
$CPY src/coarse_yolo/icpr2012/build_cellpose_boards.py \
  --candidates-dir $ICPR/classifier_B/conf002 \
  --out-dir $ICPR/classifier_C_gtfree \
  --splits train,val \
  --target-rule box
```

Expected: 217 pos / 3926 neg / 90 amb (train), 155 / 1028 / 17 (val).

### 5. Classifier

```bash
$PY src/coarse_yolo/icpr2012/train_cls_boards.py \
  --data-dir $ICPR/classifier_C_gtfree \
  --out-dir runs/classifier_C_gtfree/run_C \
  --epochs 100 --patience 20 --eval-every 2 \
  --batch 64 --lr 2e-4 --weight-decay 1e-4 --warmup 3 \
  --neg-per-pos 4 --hard-frac 0.5 --hard-score 0.01 \
  --gamma 2 --aug-level basic --in-chans 4 --seed 0
```

Expected: best val AP **0.7559 at epoch 8**; early stop at **epoch 48**.
The training is deterministic for seed 0 on the same software stack; the
11-16 run reproduced the 11-17 CSV point-for-point.

### 6. Online evaluation

Same command as Level A.

## Checksums

```
0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a  resnet50-0676ba61.pth
313898e3b46218459ff3b8c5fbdd43942c6779429b1654df1cd8cbab9b86ff63  yolo/epoch25.pt
568ebaaa1e5492d02e77b9de76f944d8b0eaaa5d29b85a426abf77e0091de636  run_C/best_cls.pth
813981379cf45c2e324f3b83558d3b2fdaba5bc3d41ff1a45d85792a22097be5  manifest_train.csv
097ea08cb6628cb72b7714fbda2e781f18ff0da6b92a7de83435139fb64ed739  manifest_val.csv
2ea4872cf03707d8771d4a075095c7cfe25745ba14ecf4a6f7b705e83f3c68c0  summary.json
```
