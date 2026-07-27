"""Surrogate model definitions: baselines, MLP, tree ensembles on mask+geom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False
    LinearRegression = None  # type: ignore
    Ridge = None  # type: ignore
    MLPRegressor = None  # type: ignore
    HistGradientBoostingRegressor = None  # type: ignore
    RandomForestRegressor = None  # type: ignore
    Pipeline = None  # type: ignore
    StandardScaler = None  # type: ignore


class Predictor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class MeanPredictor:
    """Predict train-set mean for every sample."""

    mean_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanPredictor":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n, self.mean_, dtype=np.float64)


def build_linear_geom() -> Any:
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for LinearRegression baseline")
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_ridge(
    alpha: float = 1.0,
) -> Any:
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for Ridge")
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def build_mlp(
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    max_iter: int = 8000,
    random_state: int = 42,
    alpha: float = 1e-2,
    solver: str = "lbfgs",
) -> Any:
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for MLPRegressor")
    # Larger hidden layers + moderate L2; fall back to adam if lbfgs stalls on big X.
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver=solver,
                    alpha=alpha,
                    max_iter=max_iter,
                    random_state=random_state,
                    early_stopping=(solver == "adam"),
                    learning_rate_init=1e-3,
                    verbose=False,
                ),
            ),
        ]
    )


def build_hgb(
    max_depth: int = 5,
    learning_rate: float = 0.08,
    max_iter: int = 300,
    min_samples_leaf: int = 8,
    l2_regularization: float = 0.1,
    random_state: int = 42,
) -> Any:
    """HistGradientBoosting on dense tabular mask+geom features."""
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for HistGradientBoostingRegressor")
    return HistGradientBoostingRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )


def build_rf(
    n_estimators: int = 200,
    max_depth: int = 12,
    min_samples_leaf: int = 3,
    random_state: int = 42,
) -> Any:
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for RandomForestRegressor")
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )


# All models trained in the ablation suite (order is reporting order)
MODEL_NAMES = (
    "mean",
    "linear_geom",
    "mlp_geom",
    "ridge_mask",
    "mlp_mask",
    "hgb_mask",
    "rf_mask",
)

# Feature mode per model: "geom" or "mask+geom"
FEATURE_MODE = {
    "mean": "geom",
    "linear_geom": "geom",
    "mlp_geom": "geom",
    "ridge_mask": "mask+geom",
    "mlp_mask": "mask+geom",
    "hgb_mask": "mask+geom",
    "rf_mask": "mask+geom",
}
