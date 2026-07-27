"""Load trained surrogate checkpoint and run inference on 2D masks."""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np

from app.ml.features import MASK_DS_H, MASK_DS_W, mask_feature_vector

EDUCATIONAL_DISCLAIMER = (
    "Educational surrogate approximating this project's LBM force proxies only; "
    "not certified CFD."
)


def default_model_candidates(backend_root: str | None = None) -> list[str]:
    if backend_root is None:
        backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return [
        os.path.join(backend_root, "data", "ml_surrogate", "models", "surrogate_joblib.joblib"),
        os.path.join(backend_root, "app", "ml", "artifacts", "surrogate_joblib.joblib"),
    ]


def resolve_model_path(explicit: str | None = None, backend_root: str | None = None) -> str | None:
    if explicit and os.path.isfile(explicit):
        return explicit
    for p in default_model_candidates(backend_root):
        if os.path.isfile(p):
            return p
    return None


class SurrogatePredictor:
    """Thin wrapper around a joblib training bundle."""

    def __init__(self, bundle: dict[str, Any], model_path: str):
        self.bundle = bundle
        self.model_path = model_path
        self.primary = bundle.get("primary_model", "mlp_mask")
        self.ds_h = int(bundle.get("ds_h", MASK_DS_H))
        self.ds_w = int(bundle.get("ds_w", MASK_DS_W))
        self.target = bundle.get("target", "cd_force_proxy")

    def _features_for(self, mask: np.ndarray, model_name: str) -> np.ndarray:
        mode = self.bundle.get("feature_mode", {}).get(model_name, "mask+geom")
        include_mask = mode != "geom"
        vec = mask_feature_vector(
            mask, include_mask=include_mask, ds_h=self.ds_h, ds_w=self.ds_w
        )
        return vec.reshape(1, -1)

    def predict_mask(
        self,
        mask: np.ndarray,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        name = model_name or self.primary
        models = self.bundle.get("models", {})
        if name not in models:
            raise KeyError(f"Model '{name}' not in bundle; available={list(models.keys())}")
        m = np.asarray(mask)
        if m.ndim != 2:
            raise ValueError(f"Mask must be 2D (ny, nx); got shape {m.shape}")
        X = self._features_for(m, name)
        pred = float(np.asarray(models[name].predict(X)).ravel()[0])
        return {
            "cd_force_proxy_pred": pred,
            "target": self.target,
            "model_name": name,
            "educational_disclaimer": self.bundle.get(
                "educational_disclaimer", EDUCATIONAL_DISCLAIMER
            ),
            "grid": {"ny": int(m.shape[0]), "nx": int(m.shape[1])},
        }

    def predict_batch(
        self,
        masks: np.ndarray,
        model_name: str | None = None,
    ) -> np.ndarray:
        name = model_name or self.primary
        models = self.bundle.get("models", {})
        if name not in models:
            raise KeyError(f"Model '{name}' not in bundle")
        rows = []
        for i in range(masks.shape[0]):
            rows.append(self._features_for(masks[i], name)[0])
        X = np.stack(rows, axis=0)
        return np.asarray(models[name].predict(X), dtype=np.float64).ravel()


def load_predictor(model_path: str | None = None) -> SurrogatePredictor:
    path = resolve_model_path(model_path)
    if path is None:
        raise FileNotFoundError(
            "No trained surrogate found. Run: python scripts/train_surrogate.py"
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "models" not in bundle:
        raise ValueError(f"Invalid surrogate bundle at {path}")
    return SurrogatePredictor(bundle, path)
