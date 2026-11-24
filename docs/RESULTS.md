# Results — ICPR2012 baseline

## Protocol

- Detector: `weights/yolo/epoch25.pt`
- Candidate generation: `conf=0.002`, tile 512, overlap 0.20, centre dedupe 16 px
- Classifier: 4-channel ResNet50 on GT-free 224x224 boards
- Fusion: `f = p_cal^a * yolo_score^b`
- `T`, `a`, `b`, threshold: selected on **val** only, frozen for **test**
- Matching: 5 um one-to-one greedy (A ~20.36 px, H ~21.99 px)
- Detection point: YOLO box centre

Split sizes: train 4233 candidates (217 pos / 3926 neg / 90 ambiguous),
val 1200 (155 / 1028 / 17), test 660 (83 / 566 / 11).

## Baseline results

| method | val F1 | test F1 |
|---|---|---|
| YOLO-only | 0.7262 | 0.8046 |
| C classifier (candidate level) | AP 0.7558 | AP 0.8532 |
| Score fusion | 0.7544 | **0.8306** (TP 76 / FP 23 / FN 8) |

Fusion parameters: `T=0.6577`, `a=0.5`, `b=0.5`, `threshold=0.35`.

## Classifier training

- Data: `classifier_C_gtfree` (GT-free boards; `target_rule=box`)
- 217 positives / 3926 negatives; ambiguous `y=-1` excluded
- Focal loss `gamma=2`; AdamW `lr=2e-4`, `weight_decay=1e-4`
- Batch 64; 1:4 positive:negative sampling; 50% hard negatives
- Basic augmentation: flips + 90-degree rotations
- Warmup 3 epochs, cosine to 0 over 100 epochs
- Eval every 2 epochs, patience 20 evaluations (40 epochs)
- Best: **epoch 8**, val AP **0.7559**; early stop: **epoch 48**

## Caveats

- val has only 2 slides / 155 positives; test has 84 GT. Differences below
  ~0.04 F1 are within noise.
- The old `0.8256` fusion number in earlier notes used a different protocol
  (`T=0.6149`, threshold 0.57); it is not the baseline number here.
- Test was evaluated once with the frozen protocol; do not tune on test.
