#!/usr/bin/env python3

from pathlib import Path
import os
import runpy
import sys

import scipy.__config__ as scipy_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = PROJECT_ROOT / "official" / "RhythmMamba"
OFFICIAL_MAIN = OFFICIAL_ROOT / "main.py"

if not OFFICIAL_MAIN.exists():
    raise FileNotFoundError(
        f"Official main.py not found: {OFFICIAL_MAIN}"
    )

# Compatibility for the unused MMPD loader import.
# This does not load or process the MMPD dataset.
if not hasattr(scipy_config, "get_info"):
    scipy_config.get_info = lambda *args, **kwargs: {}

if str(OFFICIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_ROOT))

# Required for the official relative resource paths.
os.chdir(OFFICIAL_ROOT)

# Forward all command-line arguments to official main.py.
sys.argv[0] = str(OFFICIAL_MAIN)

runpy.run_path(
    str(OFFICIAL_MAIN),
    run_name="__main__",
)
