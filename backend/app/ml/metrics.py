"""Regression metrics and latency helpers for the educational surrogate."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-15:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "n": int(np.asarray(y_true).size),
    }


def latency_ms_per_sample(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    warmup: int = 2,
    repeats: int = 5,
) -> float:
    """Median wall time in ms for a full batch, normalized per sample."""
    X = np.asarray(X)
    n = max(X.shape[0], 1)
    for _ in range(max(0, warmup)):
        predict_fn(X)
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        predict_fn(X)
        times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    return 1000.0 * med / n


def format_metrics_table(rows: list[dict[str, Any]]) -> str:
    """ASCII table for console / logs."""
    headers = ["model", "mae", "rmse", "r2", "latency_ms"]
    lines = []
    col_w = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            col_w[h] = max(col_w[h], len(f"{r.get(h, '')}"))
    lines.append("  ".join(h.upper().ljust(col_w[h]) for h in headers))
    lines.append("  ".join("-" * col_w[h] for h in headers))
    for r in rows:
        cells = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                cells.append(f"{v:.4f}".ljust(col_w[h]) if h != "latency_ms" else f"{v:.3f}".ljust(col_w[h]))
            else:
                cells.append(str(v).ljust(col_w[h]))
        lines.append("  ".join(cells))
    return "\n".join(lines)
