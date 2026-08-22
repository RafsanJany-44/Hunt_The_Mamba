"""Inference on PURE using the standalone UBFC-trained cross checkpoint."""

from settings import PURE, UBFC_CROSS_CHECKPOINT
from trainer import evaluate_experiment


if __name__ == "__main__":
    print("Cross-dataset inference: UBFC -> PURE")
    evaluate_experiment(PURE, UBFC_CROSS_CHECKPOINT)
