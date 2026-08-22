"""Inference on UBFC using the standalone PURE-trained cross checkpoint."""

from settings import PURE_CROSS_CHECKPOINT, UBFC
from trainer import evaluate_experiment


if __name__ == "__main__":
    print("Cross-dataset inference: PURE -> UBFC")
    evaluate_experiment(UBFC, PURE_CROSS_CHECKPOINT)
