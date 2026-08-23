"""Editable settings for independent PURE and UBFC-rPPG preprocessing.

This module intentionally replaces the official YAML/configuration stack.  Edit
the paths and run-mode below, then execute ``preprocess_all.py``.
"""

from dataclasses import dataclass
from pathlib import Path


# -----------------------------------------------------------------------------
# Raw dataset paths: edit these when the datasets move to another computer.
# -----------------------------------------------------------------------------

PURE_RAW_ROOT = Path("/media/data/rPPG/rPPG_Data/PURE")
UBFC_RAW_ROOT = Path("/media/data/rPPG/rPPG_Data/UBFC_rPPG")


# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------

DATA_ROOT = Path("/media/data/rPPG/rPPG_Data/Mamba_Hunt")

# Existing verified cache generated through the official repository.  It is
# read only by parity_check.py and is never changed by this package.
REFERENCE_CACHE_ROOT = DATA_ROOT / "RhythmMamba_Preprocessed"

# Independent outputs deliberately use new directories.  This prevents an
# accidental overwrite of the verified cache while experiments are running.
SMOKE_CACHE_ROOT = DATA_ROOT / "RhythmMamba_Preprocessed_Smoke_Independent"
FULL_CACHE_ROOT = DATA_ROOT / "RhythmMamba_Preprocessed_Independent"


# -----------------------------------------------------------------------------
# Run control
# -----------------------------------------------------------------------------

# Start with "smoke".  After validate_raw_data.py and parity_check.py pass,
# change this to "full" and rerun preprocess_all.py.
RUN_MODE = "full"  # allowed values: "smoke", "full"
DATASETS_TO_PROCESS = ("PURE", "UBFC")
SMOKE_RECORDINGS_PER_DATASET = 1
MAX_WORKERS = 4
OVERWRITE_EXISTING = False


# -----------------------------------------------------------------------------
# Official RhythmMamba preprocessing values used in our verified experiments
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformSettings:
    data_type: str = "Standardized"
    label_type: str = "Standardized"
    chunk_length: int = 160
    crop_face: bool = True
    use_large_face_box: bool = True
    large_box_coefficient: float = 1.5
    dynamic_detection: bool = False
    detection_frequency: int = 30
    use_median_face_box: bool = False
    width: int = 128
    height: int = 128


TRANSFORM = TransformSettings()


@dataclass(frozen=True)
class DatasetSettings:
    name: str
    raw_root: Path
    splits: tuple[tuple[float, float], ...]


DATASET_SETTINGS = {
    "PURE": DatasetSettings(
        name="PURE",
        raw_root=PURE_RAW_ROOT,
        splits=((0.0, 0.6), (0.6, 1.0), (0.0, 1.0)),
    ),
    "UBFC": DatasetSettings(
        name="UBFC",
        raw_root=UBFC_RAW_ROOT,
        splits=((0.0, 0.72), (0.72, 1.0), (0.0, 1.0)),
    ),
}


def selected_cache_root() -> Path:
    """Return the safe output root selected by RUN_MODE."""
    mode = RUN_MODE.lower().strip()
    if mode == "smoke":
        return SMOKE_CACHE_ROOT
    if mode == "full":
        return FULL_CACHE_ROOT
    raise ValueError(f"RUN_MODE must be 'smoke' or 'full', not {RUN_MODE!r}")


def cache_name(dataset_name: str) -> str:
    """Reproduce the official cache-directory spelling exactly."""
    t = TRANSFORM
    return (
        f"{dataset_name}_SizeW{t.width}_SizeH{t.height}"
        f"_ClipLength{t.chunk_length}_DataType{t.data_type}"
        f"_DataAugNone_LabelType{t.label_type}"
        f"_Crop_face{t.crop_face}_Large_box{t.use_large_face_box}"
        f"_Large_size{t.large_box_coefficient}"
        f"_Dyamic_Det{t.dynamic_detection}"
        f"_det_len{t.detection_frequency}"
        f"_Median_face_box{t.use_median_face_box}"
    )

