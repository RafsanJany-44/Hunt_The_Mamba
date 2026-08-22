# Mamba Hunt: simplified RhythmMamba baseline

This flat project contains only the code needed to train and evaluate
RhythmMamba with the already-preprocessed PURE and UBFC-rPPG caches.
It has no YAML configuration and no command-line argument parser.

## Important baseline rule

Do not modify the official submodule. First establish parity between this
simplified baseline and the pinned official implementation. Architecture and
augmentation corrections belong to the later development phase.

## File roles

- `settings.py`: all editable paths and hyperparameters.
- `dataset.py`: cached `.npy` clip loader and official train/test file lists.
- `model.py`: public RhythmMamba model at commit `1533ad2`.
- `augmentation.py`: public training augmentation behavior.
- `loss.py`: negative Pearson and frequency-domain objective.
- `metrics.py`: FFT MAE, RMSE, MAPE, Pearson, and SNR.
- `trainer.py`: training, checkpoint saving, and evaluation.
- `parity_check.py`: fixed-weight official-versus-simplified comparison.
- `train_*.py` and `evaluate_*.py`: simple entry points.

## Run order

From the root of `Catch_The_Mamba`:

```bash
conda activate mamba_hunting
python Mamba_Hunt/parity_check.py
mkdir -p results/simplified/logs
python Mamba_Hunt/train_pure.py 2>&1 | tee results/simplified/logs/PURE_training.log
python Mamba_Hunt/train_ubfc.py 2>&1 | tee results/simplified/logs/UBFC_training.log
```

Run the full simplified training only after `parity_check.py` passes and the
official PURE and UBFC reference runs have completed.
