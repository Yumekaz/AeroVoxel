# Surrogate artifacts

This folder holds small, commit-friendly evaluation outputs for the educational
neural surrogate (metrics JSON, optional tiny model weights, optional scatter plot).

- **`last_eval.json`** — latest train/eval metrics (MAE / RMSE / R² for the full ablation suite).
- **`surrogate_joblib.joblib`** — optional demo weights if the file is small
  enough to commit; otherwise train locally (weights live under
  `backend/data/ml_surrogate/`, gitignored).
- **`pred_vs_lbm.png`** — held-out primary-model predictions vs LBM labels (optional; needs matplotlib).

The **primary model** in the bundle is the lowest held-out MAE among non-mean models
(`mean`, `linear_geom`, `mlp_geom`, `ridge_mask`, `mlp_mask`, `hgb_mask`, `rf_mask`).

Generate and train from `backend/`:

```bash
python scripts/generate_ml_dataset.py --n 800 --seed 42
python scripts/train_surrogate.py --seed 42
python scripts/eval_surrogate.py
```

Labels come from this project's 2D LBM only — not certified CFD.
