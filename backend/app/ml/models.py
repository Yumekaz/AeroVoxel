"""Surrogate model definitions: mean baseline, linear geom, MLP on mask+geom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False
    LinearRegression = None  # type: ignore
    MLPRegressor = None  # type: ignore
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


def build_mlp(
    hidden_layer_sizes: tuple[int, ...] = (32, 16),
    max_iter: int = 5000,
    random_state: int = 42,
) -> Any:
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for MLPRegressor")
    # Compact MLP on geom + solid profiles + coarse mask (see features.py).
    # LBFGS + moderate L2 works well on 100–300 educational LBM labels.
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="lbfgs",
                    alpha=5e-2,
                    max_iter=max_iter,
                    random_state=random_state,
                    verbose=False,
                ),
            ),
        ]
    )


MODEL_NAMES = ("mean", "linear_geom", "mlp_mask")
