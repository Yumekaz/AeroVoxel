# Surrogate artifacts

This folder holds small, commit-friendly evaluation outputs for the educational
neural surrogate (metrics JSON, optional tiny model weights).

- **`last_eval.json`** — latest train/eval metrics (MAE / RMSE / R² vs baselines).
- **`surrogate_joblib.joblib`** — optional demo weights if the file is small
  enough to commit; otherwise train locally (weights live under
  `backend/data/ml_surrogate/`, gitignored).

Generate and train from `backend/`:

```bash
python scripts/generate_ml_dataset.py --n 100 --seed 42
python scripts/train_surrogate.py
python scripts/eval_surrogate.py
```

Labels come from this project's 2D LBM only — not certified CFD.
