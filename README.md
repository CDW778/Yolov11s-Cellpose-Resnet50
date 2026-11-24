# ICPR2012 Baseline — YOLO11s + Cellpose + 4-channel ResNet50

Two-stage mitotic-figure detection baseline on ICPR2012.

```
YOLO11s (512 tiles, conf 0.002)
  -> candidate boxes, centre dedupe 16 px
  -> Cellpose board: 224x224 cell-on-white + target mask (GT-free)
  -> 4-channel ResNet50 (ImageNet init, focal loss)
  -> temperature calibration
  -> score fusion f = p_cal^a * yolo_score^b
  -> 5 um one-to-one matching
```

## Results

Unified online evaluation (`eval_online_icpr2012.py`); temperature, fusion
`a/b` and threshold are selected on **val** and frozen for **test**.

| method | val F1 | test F1 |
|---|---|---|
| YOLO-only | 0.7262 | 0.8046 |
| C classifier (candidate level) | AP 0.7558 | AP 0.8532 |
| Score fusion | 0.7544 | **0.8306** (TP 76 / FP 23 / FN 8) |

Fusion parameters: `T=0.6577`, `a=0.5`, `b=0.5`, `threshold=0.35`.

**Caveats**

- val: 2 slides, 155 positives; test: 84 GT. Differences below ~0.04 F1 are
  within noise.
- Classifier best checkpoint: **epoch 8** (val AP 0.7559); early stop at
  **epoch 48**.
- The test split was evaluated once with the frozen baseline protocol. Do not
  tune on test.

## Repository layout

```
configs/paths.py                 # central dataset/weight paths
src/coarse_yolo/                 # YOLO training, WSI inference, tile prep
src/coarse_yolo/icpr2012/        # candidates, boards, classifier, evaluation
src/cellpose_baseline/           # 4-channel ResNet50 + dataset helper
scripts/run_v2a.sh               # YOLO v2a training + checkpoint selection
scripts/run_C_gtfree.sh          # boards -> classifier -> online evaluation
results/baseline/                # small metrics, manifests, checksums
docs/DATA.md                     # dataset layout and label rules
docs/REPRODUCE.md                # exact commands and expected numbers
docs/RESULTS.md                  # baseline results and protocol
```

## Dataset

All datasets live outside this repository:

```
/home/hyw/dataset/chendawei/ICPR2012
```

`docs/DATA.md` lists the required directories and counts. Dataset files and
model checkpoints are deliberately excluded from git.

## Environment

Two conda environments are used:

- `midog`: YOLO, classifier training, online evaluation
- `cellpose`: board construction

See `environment.txt` and `requirements.txt`.

## Reproduce

`docs/REPRODUCE.md` has two levels:

- **Level A**: exact replay with the prebuilt `classifier_C_gtfree` boards,
  `classifier_B/conf002` candidates and the two checkpoints.
- **Level B**: rebuild from raw ICPR2012 (tiles -> YOLO -> candidates ->
  boards -> classifier -> online evaluation).

## Known limitations

- The classifier sees a GT-free white-board rendering, not tissue context.
- The target-mask channel is not ImageNet-normalised and its first-conv
  weights start at zero; it contributes little (mask-zero AP 0.731 vs 0.758).
- The classifier cannot recover detector misses (val 5 um coverage 95.1%).
- The fusion grid is coarse (`a,b in {0,0.5,1,1.5,2}`); temperature values
  differ across checkpoints, so cross-checkpoint comparisons should account
  for the effective weight `a/T`.
