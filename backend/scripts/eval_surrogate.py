"""
Evaluate a trained educational surrogate vs baselines (or retrain if missing).

Usage (from backend/):
    python scripts/eval_surrogate.py
    python scripts/eval_surrogate.py --model data/ml_surrogate/models/surrogate_joblib.joblib
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.ml.dataset import default_dataset_dir, load_dataset  # noqa: E402
from app.ml.features import batch_features  # noqa: E402
from app.ml.infer import load_predictor, resolve_model_path  # noqa: E402
from app.ml.metrics import format_metrics_table, latency_ms_per_sample, regression_metrics  # noqa: E402
from app.ml.train import train_surrogate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate educational LBM surrogate")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, help="Path to surrogate_joblib.joblib")
    parser.add_argument("--retrain", action="store_true", help="Force retrain before eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = args.dataset or default_dataset_dir(BACKEND_ROOT)
    model_path = resolve_model_path(args.model, BACKEND_ROOT)

    if args.retrain or model_path is None:
        if not os.path.isdir(dataset_dir):
            print(f"Dataset not found: {dataset_dir}")
            return 1
        print("[eval] Training surrogate…")
        payload = train_surrogate(dataset_dir=dataset_dir, seed=args.seed)
        print(format_metrics_table(payload["results"]))
        print(f"MLP beats mean MAE: {payload['mlp_beats_mean_mae']}")
        return 0 if payload["mlp_beats_mean_mae"] else 2

    predictor = load_predictor(model_path)
    bundle = predictor.bundle
    masks, labels, meta = load_dataset(dataset_dir)
    y = np.asarray(labels[bundle.get("target", "cd_force_proxy")], dtype=np.float64)

    test_idx = bundle.get("test_idx")
    if test_idx is None:
        # Fallback: re-split with same seed
        rng = np.random.default_rng(int(bundle.get("seed", args.seed)))
        idx = np.arange(len(y))
        rng.shuffle(idx)
        n_test = max(1, int(round(len(y) * float(bundle.get("test_size", 0.2)))))
        test_idx = idx[:n_test].tolist()

    test_idx = np.asarray(test_idx, dtype=int)
    y_te = y[test_idx]
    masks_te = masks[test_idx]

    results = []
    for name in ("mean", "linear_geom", "mlp_mask"):
        if name not in bundle["models"]:
            continue
        pred = predictor.predict_batch(masks_te, model_name=name)
        m = regression_metrics(y_te, pred)
        mode = bundle.get("feature_mode", {}).get(name, "mask+geom")
        include_mask = mode != "geom"
        X = batch_features(
            masks_te,
            include_mask=include_mask,
            ds_h=predictor.ds_h,
            ds_w=predictor.ds_w,
        )
        model = bundle["models"][name]
        lat = latency_ms_per_sample(lambda Z, mdl=model: mdl.predict(Z), X)
        results.append(
            {
                "model": name,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "latency_ms": lat,
            }
        )

    print("=" * 60)
    print("Surrogate evaluation (held-out LBM labels)")
    print("=" * 60)
    print(f"model_path={model_path}")
    print(f"n_test={len(test_idx)}  target={bundle.get('target')}")
    print(format_metrics_table(results))
    mean_mae = next(r["mae"] for r in results if r["model"] == "mean")
    mlp_mae = next(r["mae"] for r in results if r["model"] == "mlp_mask")
    beats = mlp_mae < mean_mae
    print(f"MLP beats mean MAE: {beats}  ({mlp_mae:.4f} < {mean_mae:.4f})")

    art = os.path.join(BACKEND_ROOT, "app", "ml", "artifacts", "last_eval.json")
    if os.path.isfile(art):
        with open(art, encoding="utf-8") as f:
            saved = json.load(f)
        print(f"Committed metrics snapshot: {art}")
        print(f"  trained_utc={saved.get('trained_utc')}")

    lbm_mean = float(meta.get("wall_time_mean_s", 0.0)) if meta else 0.0
    if lbm_mean > 0:
        print(f"LBM mean wall: {lbm_mean:.3f}s vs surrogate ~{results[-1]['latency_ms']:.3f} ms")

    return 0 if beats else 2


if __name__ == "__main__":
    raise SystemExit(main())
