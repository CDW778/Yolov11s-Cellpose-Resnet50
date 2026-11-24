"""Central path configuration for the YOLO11s + Cellpose + ResNet50 project.

All datasets live under DATASET_ROOT and must never be stored inside this repo.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = Path("/home/hyw/dataset/chendawei")
ICPR_ROOT = DATASET_ROOT / "ICPR2012"

RAW_ROOT = ICPR_ROOT / "raw"
IMAGES_ROOT = ICPR_ROOT / "images"
CELLPOSE_CACHE = ICPR_ROOT / "cellpose_cache"
YOLO_V1 = ICPR_ROOT / "yolo_512_ov20"
YOLO_V2A = ICPR_ROOT / "yolo_512_gtcrop4"
YOLO_HARDNEG = ICPR_ROOT / "yolo_512_hardneg"
CLASSIFIER_C = ICPR_ROOT / "classifier_C"
CLASSIFIER_C_GTFREE = ICPR_ROOT / "classifier_C_gtfree"
CLASSIFIER_C_MIDOG = ICPR_ROOT / "classifier_C_midog"
CLASSIFIER_B = ICPR_ROOT / "classifier_B"

MIDOGPP_ROOT = DATASET_ROOT / "MIDOG++"
MIDOGPP_IMAGES = MIDOGPP_ROOT / "images"
MIDOGPP_META = DATASET_ROOT / "MIDOGpp-main" / "databases" / "MIDOG++.json"

WEIGHTS = PROJECT_ROOT / "weights"
CELLPOSE_BASELINE = PROJECT_ROOT / "src" / "cellpose_baseline"
CELLPOSE_WEIGHT = WEIGHTS / "cellpose" / "cpsam_v2"
IMAGENET_V1 = WEIGHTS / "resnet50-0676ba61.pth"
YOLO11S = WEIGHTS / "yolo11s.pt"

RUNS = PROJECT_ROOT / "runs"
DATA = PROJECT_ROOT / "data"
DOCS = PROJECT_ROOT / "docs"


def image_dir(split):
    """Directory with original ICPR2012 images for a split (train/val/test)."""
    return IMAGES_ROOT / split
