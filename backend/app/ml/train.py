"""Train educational aero surrogate models on LBM-labeled masks."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

import joblib
import numpy as np

from app.ml.dataset import load_dataset
from app.ml.features import MASK_DS_H, MASK_DS_W, batch_features
from app.ml.metrics import latency_ms_per_sample, regression_metrics
from app.ml.models import (
    FEATURE_MODE,
    MeanPredictor,
    build_hgb,
    build_linear_geom,
    build_mlp,
    build_rf,
    build_ridge,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False
    plt = None  # type: ignore


def _fit_safe(builder: Callable[[], Any], X: np.ndarray, y: np.ndarray, name: str) -> Any:
    """Fit a model; for MLP try adam fallback if lbfgs fails."""
    model = builder()
    try:
        model.fit(X, y)
        return model
    except Exception as exc:  # pragma: no cover
        print(f"[train] {name} fit failed ({exc}); retrying with fallback if available")
        if name in ("mlp_mask", "mlp_geom"):
            alt = build_mlp(solver="adam", max_iter=1500, random_state=42)
            alt.fit(X, y)
            return alt
        raise


def _save_pred_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    model_name: str,
    target: str,
) -> bool:
    if not HAS_MPL:
        return False
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=120)
    ax.scatter(y_true, y_pred, s=18, alpha=0.65, edgecolors="none", c="#2563eb")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0, label="y = x")
    ax.set_xlabel(f"LBM {target}")
    ax.set_ylabel(f"Predicted ({model_name})")
    ax.set_title(f"Held-out: {model_name} vs LBM")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


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
    """Train ablation suite; save best non-mean model as primary + metrics."""
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

    print(
        f"[train] n={n} train={len(train_idx)} test={len(test_idx)} "
        f"X_geom={X_geom.shape[1]}d X_mask={X_mask.shape[1]}d"
    )

    # Build + fit all models
    fitted: dict[str, Any] = {}
    fitted["mean"] = MeanPredictor().fit(X_geom_tr, y_tr)
    fitted["linear_geom"] = _fit_safe(build_linear_geom, X_geom_tr, y_tr, "linear_geom")
    fitted["mlp_geom"] = _fit_safe(
        lambda: build_mlp(hidden_layer_sizes=(32, 16), alpha=5e-2, random_state=seed),
        X_geom_tr,
        y_tr,
        "mlp_geom",
    )
    fitted["ridge_mask"] = _fit_safe(
        lambda: build_ridge(alpha=3.0), X_mask_tr, y_tr, "ridge_mask"
    )
    fitted["mlp_mask"] = _fit_safe(
        lambda: build_mlp(
            hidden_layer_sizes=(64, 32),
            alpha=1e-2,
            max_iter=8000,
            random_state=seed,
            solver="lbfgs",
        ),
        X_mask_tr,
        y_tr,
        "mlp_mask",
    )
    fitted["hgb_mask"] = _fit_safe(
        lambda: build_hgb(
            max_depth=5,
            learning_rate=0.08,
            max_iter=300,
            min_samples_leaf=max(5, len(y_tr) // 80),
            l2_regularization=0.1,
            random_state=seed,
        ),
        X_mask_tr,
        y_tr,
        "hgb_mask",
    )
    fitted["rf_mask"] = _fit_safe(
        lambda: build_rf(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=seed,
        ),
        X_mask_tr,
        y_tr,
        "rf_mask",
    )

    # Evaluate
    results: list[dict[str, Any]] = []
    preds_te: dict[str, np.ndarray] = {}
    report_order = (
        "mean",
        "linear_geom",
        "mlp_geom",
        "ridge_mask",
        "mlp_mask",
        "hgb_mask",
        "rf_mask",
    )
    for name in report_order:
        model = fitted[name]
        mode = FEATURE_MODE[name]
        Xte = X_geom_te if mode == "geom" else X_mask_te
        pred = np.asarray(model.predict(Xte), dtype=np.float64).ravel()
        preds_te[name] = pred
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
                "feature_mode": mode,
            }
        )
        print(
            f"  {name:14s}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
            f"R2={m['r2']:.4f}  ({mode})"
        )

    # Primary = lowest test MAE among non-mean models
    non_mean = [r for r in results if r["model"] != "mean"]
    best = min(non_mean, key=lambda r: r["mae"])
    primary = str(best["model"])
    linear_row = next(r for r in results if r["model"] == "linear_geom")
    mean_row = next(r for r in results if r["model"] == "mean")
    best_beats_linear = bool(best["mae"] < linear_row["mae"])
    best_beats_mean = bool(best["mae"] < mean_row["mae"])

    lbm_mean_s = float(meta.get("wall_time_mean_s", 0.0)) if meta else 0.0
    if lbm_mean_s <= 0 and "wall_time_s" in labels:
        lbm_mean_s = float(np.mean(labels["wall_time_s"]))

    bundle = {
        "version": "1.1.0",
        "target": target,
        "ds_h": ds_h,
        "ds_w": ds_w,
        "seed": seed,
        "test_size": test_size,
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "models": fitted,
        "primary_model": primary,
        "feature_mode": dict(FEATURE_MODE),
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

    def _rel_or_abs(path: str) -> str:
        """Prefer repo-relative POSIX-style paths in committed metrics JSON."""
        abs_p = os.path.abspath(path)
        # Walk up from artifacts/ml to find a sensible repo root (…/AeroVoxel)
        here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        try:
            rel = os.path.relpath(abs_p, here)
            if not rel.startswith(".."):
                return rel.replace("\\", "/")
        except ValueError:
            pass
        return abs_p.replace("\\", "/")

    eval_payload: dict[str, Any] = {
        "target": target,
        "n_samples": n,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "seed": seed,
        "test_size": test_size,
        "dataset_dir": _rel_or_abs(dataset_dir),
        "model_path": _rel_or_abs(model_path),
        "lbm_mean_wall_time_s": lbm_mean_s,
        "lbm_mean_latency_ms": lbm_mean_s * 1000.0,
        "results": results,
        "primary_model": primary,
        "best_mae": float(best["mae"]),
        "linear_geom_mae": float(linear_row["mae"]),
        "best_beats_linear_mae": best_beats_linear,
        "best_beats_mean_mae": best_beats_mean,
        "mae_margin_vs_linear": float(linear_row["mae"] - best["mae"]),
        # Keep legacy key for older scripts
        "mlp_beats_mean_mae": bool(
            next(r["mae"] for r in results if r["model"] == "mlp_mask") < mean_row["mae"]
        ),
        "trained_utc": bundle["trained_utc"],
        "educational_disclaimer": bundle["educational_disclaimer"],
        "note": (
            "Metrics compare models against held-out LBM labels from this project. "
            "Primary model is the lowest held-out MAE among non-mean models. "
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

    # Optional scatter: primary predictions vs LBM on held-out set
    scatter_path = os.path.join(artifacts_dir, "pred_vs_lbm.png")
    if _save_pred_scatter(
        y_te, preds_te[primary], scatter_path, primary, target
    ):
        eval_payload["pred_vs_lbm_path"] = _rel_or_abs(scatter_path)
        print(f"[train] Wrote scatter {scatter_path}")

    # Optionally copy small joblib into artifacts if under 5 MB
    try:
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        eval_payload["model_size_mb"] = size_mb
        if size_mb < 5.0:
            art_model = os.path.join(artifacts_dir, "surrogate_joblib.joblib")
            joblib.dump(bundle, art_model)
            eval_payload["artifacts_model_path"] = _rel_or_abs(art_model)
        else:
            eval_payload["artifacts_model_path"] = None
            eval_payload["note_model"] = (
                f"Model is {size_mb:.2f} MB; kept under data/ only (gitignored)."
            )
        with open(art_metrics, "w", encoding="utf-8") as f:
            json.dump(eval_payload, f, indent=2)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(eval_payload, f, indent=2)
    except OSError:
        pass

    print(
        f"[train] primary={primary}  MAE={best['mae']:.4f}  "
        f"linear_geom MAE={linear_row['mae']:.4f}  "
        f"beats_linear={best_beats_linear}"
    )
    return eval_payload
