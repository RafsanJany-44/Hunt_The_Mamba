# Independent RhythmMamba Data Preprocessing

This folder converts native PURE and UBFC-rPPG data into the `.npy` clips and
CSV manifests consumed by the simplified `Mamba_Hunt/dataset.py`. It does not
import the official RhythmMamba repository and does not require YAML files.

The implementation preserves the preprocessing settings used by the verified
baseline:

- first-frame Haar face detection;
- 1.5× enlarged face box;
- resize to 128 × 128;
- global per-recording video standardization;
- global per-recording label standardization;
- non-overlapping 160-frame chunks;
- PURE waveform resampling to the video length;
- native UBFC waveform handling without resampling;
- official-compatible cache names, clip names and split manifests.

## Safety

Independent outputs use new directories:

```text
/media/data/rPPG/rPPG_Data/Mamba_Hunt/
├── RhythmMamba_Preprocessed/                    # existing verified reference
├── RhythmMamba_Preprocessed_Smoke_Independent/  # new smoke output
└── RhythmMamba_Preprocessed_Independent/        # new full output
```

The existing `RhythmMamba_Preprocessed` directory is never written by this
package. The original PURE and UBFC-rPPG datasets are also read only.

## Step 1: edit paths

Open `Data_Preprocessing/settings.py` and verify:

```python
PURE_RAW_ROOT = Path("/media/data/rPPG/rPPG_Data/PURE")
UBFC_RAW_ROOT = Path("/media/data/rPPG/rPPG_Data/UBFC_rPPG")
DATA_ROOT = Path("/media/data/rPPG/rPPG_Data/Mamba_Hunt")
```

`PURE_RAW_ROOT` may point to `PURE` or directly to `PURE/ALL/ALL`. UBFC may
use the native `vid_1/vid_1.avi` layout or the compatible
`subject1/vid.avi` layout.

## Step 2: validate the native datasets

From the `Catch_The_Mamba` repository root, run:

```bash
python Mamba_Hunt/Data_Preprocessing/validate_raw_data.py
```

Expected counts for the current local data are 59 PURE recordings and 42 UBFC
recordings.

## Step 3: create independent smoke caches

Keep this value in `Data_Preprocessing/settings.py`:

```python
RUN_MODE = "smoke"
```

Then run:

```bash
python Mamba_Hunt/Data_Preprocessing/preprocess_all.py
```

Only the first recording of each dataset is processed.

## Step 4: verify exact preprocessing parity

```bash
python Mamba_Hunt/Data_Preprocessing/parity_check.py
```

Do not proceed to full preprocessing unless both PURE and UBFC report
`Parity result: PASSED`. Exact parity is expected when the same OpenCV version
and Haar-cascade file are used.

## Step 5: preprocess the complete datasets

Change one value:

```python
RUN_MODE = "full"
```

Run again:

```bash
python Mamba_Hunt/Data_Preprocessing/preprocess_all.py
```

You may process one dataset separately with:

```bash
python Mamba_Hunt/Data_Preprocessing/preprocess_pure.py
python Mamba_Hunt/Data_Preprocessing/preprocess_ubfc.py
```

Validate the resulting full cache before connecting it to training:

```bash
python Mamba_Hunt/Data_Preprocessing/validate_cache.py
```

## Step 6: connect Mamba_Hunt to the independent cache

Only after full preprocessing and validation, change the main
`Mamba_Hunt/settings.py`:

```python
PREPROCESSED_ROOT = Path(
    "/media/data/rPPG/rPPG_Data/Mamba_Hunt/RhythmMamba_Preprocessed_Independent"
)
```

Run the existing inference tests again before beginning new architecture work.

## Why RhythmMamba_DataView is no longer required

The previous compatibility view renamed the native layouts to the names
expected by the official loaders. The independent adapters read the native
layouts directly, so no symlink view is required. The existing DataView can
remain in place for official-repository experiments, but Mamba_Hunt no longer
depends on it.

## Registering a future dataset

For COHFACE, TokyoTech, or another dataset:

1. Create `datasets/cohface.py` (replace the name as appropriate).
2. Add an adapter class implementing:
   - `discover(raw_root)`;
   - `split(recordings, begin, end)`;
   - `read_frames(recording)`;
   - `read_label(recording)`;
   - `align_label(label, frame_count)`;
   - `probe(recording)`.
3. Use `Recording` from `common.py` to describe every raw recording.
4. Register the adapter in `dataset_registry.py`.
5. Add its raw path and split fractions to `DATASET_SETTINGS` in `settings.py`.
6. Run raw validation, one-record smoke preprocessing and parity/shape checks.
7. Only then run complete preprocessing.

Dataset-specific code should only read frames, read labels, align their timing,
and define subject-disjoint splits. Face cropping, standardization, chunking,
cache writing and manifest generation stay in `common.py`.

## Dependencies

The preprocessing package requires Python, NumPy, OpenCV and tqdm. The current
`mamba_hunting` environment already contains these dependencies. On another
machine they can be installed from `Data_Preprocessing/requirements.txt`. It
does not require PyTorch, Mamba, the official RhythmMamba code, pandas, YAML,
or SciPy.

## Attribution

The behavior in `common.py` and the PURE/UBFC adapters was independently
extracted and adapted from the official RhythmMamba preprocessing loaders for
compatibility. RhythmMamba is copyright 2024 Zizheng Guo and distributed under
the MIT License. See `THIRD_PARTY_LICENSES/RhythmMamba_LICENSE.txt`.
