"""Train educational aero surrogate models on LBM-labeled masks."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np

from app.ml.dataset import load_dataset
from app.ml.features import MASK_DS_H, MASK_DS_W, batch_features
from app.ml.metrics import latency_ms_per_sample, regression_metrics
from app.ml.models import MeanPredictor, build_linear_geom, build_mlp


def train_surrogate(
    dataset_dir: str,
    model_dir: str | None = None,
    target: str = "cd_force_proxy",
    test_size: float = 0.2,
    seed: int = 42,
    ds_h: int = MASK_DS_H,
    ds_w: int = MASK_DS_W,
    artifacts_dir: str | None = None,
) -> dict[str, Any]:
    """Train mean / linear_geom / mlp_mask models; save best bundle + metrics."""
    masks, labels, meta = load_dataset(dataset_dir)
    if target not in labels:
        raise KeyError(f"Target '{target}' not in labels: {list(labels.keys())}")

    y = np.asarray(labels[target], dtype=np.float64)
    n = y.shape[0]
    if n < 10:
        raise ValueError(f"Need at least 10 samples, got {n}")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(n * test_size)))
    n_test = min(n_test, n - 5)  # keep enough train
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_mask = batch_features(masks, include_mask=True, ds_h=ds_h, ds_w=ds_w)
    X_geom = batch_features(masks, include_mask=False, ds_h=ds_h, ds_w=ds_w)

    X_mask_tr, X_mask_te = X_mask[train_idx], X_mask[test_idx]
    X_geom_tr, X_geom_te = X_geom[train_idx], X_geom[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    mean_model = MeanPredictor().fit(X_geom_tr, y_tr)
    linear_model = build_linear_geom().fit(X_geom_tr, y_tr)
    mlp_model = build_mlp(random_state=seed).fit(X_mask_tr, y_tr)

    results: list[dict[str, Any]] = []
    for name, model, Xte, Xtr in (
        ("mean", mean_model, X_geom_te, X_geom_tr),
        ("linear_geom", linear_model, X_geom_te, X_geom_tr),
        ("mlp_mask", mlp_model, X_mask_te, X_mask_tr),
    ):
        pred = np.asarray(model.predict(Xte), dtype=np.float64).ravel()
        m = regression_metrics(y_te, pred)
        lat = latency_ms_per_sample(lambda X, mdl=model: mdl.predict(X), Xte)
        results.append(
            {
                "model": name,
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "latency_ms": lat,
                "n_test": m["n"],
                "n_train": int(len(y_tr)),
            }
        )

    lbm_mean_s = float(meta.get("wall_time_mean_s", 0.0)) if meta else 0.0
    if lbm_mean_s <= 0 and "wall_time_s" in labels:
        lbm_mean_s = float(np.mean(labels["wall_time_s"]))

    bundle = {
        "version": "1.0.0",
        "target": target,
        "ds_h": ds_h,
        "ds_w": ds_w,
        "seed": seed,
        "test_size": test_size,
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "models": {
            "mean": mean_model,
            "linear_geom": linear_model,
            "mlp_mask": mlp_model,
        },
        "primary_model": "mlp_mask",
        "feature_mode": {
            "mean": "geom",
            "linear_geom": "geom",
            "mlp_mask": "mask+geom",
        },
        "meta_dataset": meta,
        "trained_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "educational_disclaimer": (
            "Predicts educational LBM force proxies from this project only; "
            "not certified CFD and not a substitute for a live solve when accuracy matters."
        ),
    }

    if model_dir is None:
        model_dir = os.path.join(dataset_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "surrogate_joblib.joblib")
    joblib.dump(bundle, model_path)

    eval_payload = {
        "target": target,
        "n_samples": n,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "seed": seed,
        "test_size": test_size,
        "dataset_dir": os.path.abspath(dataset_dir),
        "model_path": os.path.abspath(model_path),
        "lbm_mean_wall_time_s": lbm_mean_s,
        "lbm_mean_latency_ms": lbm_mean_s * 1000.0,
        "results": results,
        "primary_model": "mlp_mask",
        "mlp_beats_mean_mae": bool(results[2]["mae"] < results[0]["mae"]),
        "trained_utc": bundle["trained_utc"],
        "educational_disclaimer": bundle["educational_disclaimer"],
        "note": (
            "Metrics compare models against held-out LBM labels from this project. "
            "Not certified CFD validation."
        ),
    }

    metrics_path = os.path.join(model_dir, "last_eval.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    if artifacts_dir is None:
        artifacts_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "artifacts")
        )
    os.makedirs(artifacts_dir, exist_ok=True)
    art_metrics = os.path.join(artifacts_dir, "last_eval.json")
    with open(art_metrics, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    # Optionally copy small joblib into artifacts if under 2 MB
    try:
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        if size_mb < 2.0:
            art_model = os.path.join(artifacts_dir, "surrogate_joblib.joblib")
            joblib.dump(bundle, art_model)
            eval_payload["artifacts_model_path"] = os.path.abspath(art_model)
            eval_payload["model_size_mb"] = size_mb
            with open(art_metrics, "w", encoding="utf-8") as f:
                json.dump(eval_payload, f, indent=2)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(eval_payload, f, indent=2)
        else:
            eval_payload["model_size_mb"] = size_mb
            eval_payload["artifacts_model_path"] = None
            eval_payload["note_model"] = (
                f"Model is {size_mb:.2f} MB; kept under data/ only (gitignored)."
            )
            with open(art_metrics, "w", encoding="utf-8") as f:
                json.dump(eval_payload, f, indent=2)
    except OSError:
        pass

    return eval_payload
