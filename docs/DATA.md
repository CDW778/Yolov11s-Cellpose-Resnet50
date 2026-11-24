# Data — ICPR2012 baseline

All datasets live outside this repository:

```
/home/hyw/dataset/chendawei/ICPR2012
```

## Directory layout

```
ICPR2012/
├── raw/{A,H}0?_v2/                 # GT polygon CSV + original BMP
├── images/{train,val,test}/        # 42 / 14 / 14 frames
├── cellpose_cache/                 # whole-image Cellpose masks (not used by C)
├── yolo_512_ov20/                  # v1 512-grid tiles (input to v2a builder)
├── yolo_512_gtcrop4/               # v2a YOLO training tiles
├── classifier_B/conf002/           # candidates_{train,val,test}.npz
└── classifier_C_gtfree/            # boards/ masks/ manifest_*.csv summary.json
```

Neither the raw data nor the boards are committed to git. For exact replay,
`classifier_C_gtfree` (271 MB) and `classifier_B/conf002` (196 MB) should be
provided as release assets or rebuilt with the scripts below.

## Slide split

`SPLIT_PREFIXES` in `build_cellpose_boards.py`:

- train: A00, A01, A02, H00, H01, H02
- val: A03, H03
- test: A04, H04

Splits are by slide; no frame appears in more than one split.

## Candidate counts

| split | candidates | pos | neg | ambiguous |
|---|---|---|---|---|
| train | 4233 | 217 | 3926 | 90 |
| val | 1200 | 155 | 1028 | 17 |
| test | 660 | 83 | 566 | 11 |

Candidates: YOLO `epoch25.pt`, `conf=0.002`, tile 512, overlap 0.20,
centre dedupe 16 px.

## Candidate labels

- `y=1`: >=98% of a GT polygon's points inside the box, or IoU(box, GT bbox) > 0.5
- `y=0`: no GT pixels inside the box
- `y=-1`: GT pixels present but neither rule is met; excluded from training,
  kept for end-to-end evaluation

GT is used only for labels/evaluation. The board and target mask are built
with `make_board(..., y=0, polys=[])` (`--target-rule box`), so no GT enters
the classifier input.

## Manifests and checksums

Small copies of the manifests are in `results/baseline/manifests/` for
provenance. The original manifests are in `classifier_C_gtfree/`.

```
813981379cf45c2e324f3b83558d3b2fdaba5bc3d41ff1a45d85792a22097be5  manifest_train.csv
097ea08cb6628cb72b7714fbda2e781f18ff0da6b92a7de83435139fb64ed739  manifest_val.csv
2ea4872cf03707d8771d4a075095c7cfe25745ba14ecf4a6f7b705e83f3c68c0  summary.json
```

The manifests contain absolute paths. If the dataset is mounted elsewhere,
rewrite the `rgb`/`mask` prefixes to the local `ICPR_ROOT` before training.
