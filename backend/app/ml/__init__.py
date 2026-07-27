"""Educational ML surrogate: predict AeroVoxel LBM force proxies from 2D masks.

Approximates this project's D2Q9 LBM labels only — not certified CFD.
"""

from app.ml.infer import SurrogatePredictor, load_predictor
from app.ml.metrics import regression_metrics

__all__ = [
    "SurrogatePredictor",
    "load_predictor",
    "regression_metrics",
]
