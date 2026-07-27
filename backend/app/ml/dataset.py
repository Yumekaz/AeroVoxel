"""Generate and load LBM-labeled mask datasets for the educational surrogate."""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.ml.features import char_length_from_mask
from app.ml.shapes import random_mask
from app.solvers.lbm_2d import LbmSolver2D, educational_force_metrics

DATASET_VERSION = "1.0.0"

DEFAULT_NX = 64
DEFAULT_NY = 32
DEFAULT_TAU = 0.6
DEFAULT_U_INLET = 0.08
DEFAULT_STEPS = 350
DEFAULT_FORCE_AVG = 30


@dataclass
class DatasetMeta:
    version: str
    nx: int
    ny: int
    tau: float
    u_inlet: float
    steps: int
    force_avg_steps: int
    n_samples: int
    seed: int
    created_utc: str
    label_columns: list[str]
    wall_time_total_s: float
    wall_time_mean_s: float
    note: str = (
        "Labels are educational LBM force proxies from this project only; "
        "not certified CFD."
    )


def default_dataset_dir(backend_root: str | None = None) -> str:
    if backend_root is None:
        backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(backend_root, "data", "ml_surrogate")


def _wake_score(velocity: np.ndarray, mask: np.ndarray, nx: int, ny: int) -> float:
    ux = velocity[0]
    fluid = ~mask.astype(bool)
    wake_pixels = int(np.sum((ux < 0.02) & fluid))
    score = float(wake_pixels / max(nx * (ny // 2), 1))
    return max(0.05, min(0.99, score))


def label_one_mask(
    mask: np.ndarray,
    nx: int,
    ny: int,
    tau: float = DEFAULT_TAU,
    u_inlet: float = DEFAULT_U_INLET,
    steps: int = DEFAULT_STEPS,
    force_avg_steps: int = DEFAULT_FORCE_AVG,
) -> dict[str, float]:
    """Run LBM on one mask and return educational labels + timing."""
    solver = LbmSolver2D(nx=nx, ny=ny, tau=tau, u_inlet=u_inlet, wind_angle_deg=0.0)
    t0 = time.perf_counter()
    u, pressure = solver.solve(mask, steps=steps, force_avg_steps=force_avg_steps)
    wall = time.perf_counter() - t0

    char_l = char_length_from_mask(mask)
    force_m = educational_force_metrics(
        pressure=pressure,
        mask=mask,
        u_ref=u_inlet,
        char_length=char_l,
        velocity=u,
        tau=tau,
        momentum_force_lu=solver.last_force_lu,
        momentum_method=solver.last_force_method,
    )
    cl_proxy = force_to_cl_proxy(force_m.get("force_y_lu", 0.0), u_inlet, char_l)
    return {
        "cd_force_proxy": float(force_m["cd_force_proxy"]),
        "cl_force_proxy": float(cl_proxy),
        "wake_score": _wake_score(u, mask, nx, ny),
        "char_length_lu": float(char_l),
        "force_x_lu": float(force_m["force_x_lu"]),
        "force_y_lu": float(force_m["force_y_lu"]),
        "wall_time_s": float(wall),
    }


def force_to_cl_proxy(force_y: float, u_ref: float, char_length: float, rho: float = 1.0) -> float:
    """Educational lift coefficient proxy: Cl = 2 Fy / (ρ U² L)."""
    denom = rho * (max(abs(u_ref), 1e-12) ** 2) * max(char_length, 1e-12)
    return float(2.0 * force_y / denom)


def generate_dataset(
    out_dir: str,
    n: int = 100,
    seed: int = 42,
    nx: int = DEFAULT_NX,
    ny: int = DEFAULT_NY,
    tau: float = DEFAULT_TAU,
    u_inlet: float = DEFAULT_U_INLET,
    steps: int = DEFAULT_STEPS,
    force_avg_steps: int = DEFAULT_FORCE_AVG,
    force: bool = False,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Generate masks + LBM labels. Skip if complete unless force=True."""
    os.makedirs(out_dir, exist_ok=True)
    masks_path = os.path.join(out_dir, "masks.npy")
    labels_path = os.path.join(out_dir, "labels.csv")
    meta_path = os.path.join(out_dir, "meta.json")

    if (
        not force
        and os.path.isfile(masks_path)
        and os.path.isfile(labels_path)
        and os.path.isfile(meta_path)
    ):
        with open(meta_path, encoding="utf-8") as f:
            existing = json.load(f)
        if int(existing.get("n_samples", 0)) >= n and existing.get("nx") == nx and existing.get("ny") == ny:
            print(f"[dataset] Existing dataset with n={existing['n_samples']} at {out_dir}; skip (use --force).")
            return existing

    rng = np.random.default_rng(seed)
    masks = np.zeros((n, ny, nx), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()

    label_cols = [
        "sample_id",
        "shape",
        "cd_force_proxy",
        "cl_force_proxy",
        "wake_score",
        "char_length_lu",
        "force_x_lu",
        "force_y_lu",
        "wall_time_s",
        "solid_fraction",
    ]

    print(f"[dataset] Generating n={n} on {nx}x{ny}, steps={steps}, seed={seed}")
    i = 0
    attempts = 0
    max_attempts = n * 8
    while i < n and attempts < max_attempts:
        attempts += 1
        mask, shape_name = random_mask(nx, ny, rng)
        labels = label_one_mask(
            mask,
            nx=nx,
            ny=ny,
            tau=tau,
            u_inlet=u_inlet,
            steps=steps,
            force_avg_steps=force_avg_steps,
        )
        cd = labels["cd_force_proxy"]
        # Drop rare coarse-grid force blow-ups so the surrogate sees finite labels
        if not np.isfinite(cd) or cd < 0.05 or cd > 12.0:
            continue
        if not np.isfinite(labels["cl_force_proxy"]):
            continue
        masks[i] = mask.astype(np.float32)
        rows.append(
            {
                "sample_id": i,
                "shape": shape_name,
                "cd_force_proxy": labels["cd_force_proxy"],
                "cl_force_proxy": labels["cl_force_proxy"],
                "wake_score": labels["wake_score"],
                "char_length_lu": labels["char_length_lu"],
                "force_x_lu": labels["force_x_lu"],
                "force_y_lu": labels["force_y_lu"],
                "wall_time_s": labels["wall_time_s"],
                "solid_fraction": float(np.mean(mask)),
            }
        )
        i += 1
        if i % progress_every == 0 or i == 1 or i == n:
            elapsed = time.perf_counter() - t_all
            eta = elapsed / i * (n - i)
            print(
                f"  [{i}/{n}] shape={shape_name:12s} "
                f"Cd={labels['cd_force_proxy']:.3f} "
                f"t={labels['wall_time_s']:.2f}s  "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )

    if i < n:
        raise RuntimeError(
            f"Only accepted {i}/{n} samples after {attempts} attempts "
            "(check shape generators / Cd filter)."
        )

    total_wall = time.perf_counter() - t_all
    mean_wall = float(np.mean([r["wall_time_s"] for r in rows]))

    np.save(masks_path, masks)
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=label_cols)
        writer.writeheader()
        writer.writerows(rows)

    meta = DatasetMeta(
        version=DATASET_VERSION,
        nx=nx,
        ny=ny,
        tau=tau,
        u_inlet=u_inlet,
        steps=steps,
        force_avg_steps=force_avg_steps,
        n_samples=n,
        seed=seed,
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        label_columns=label_cols,
        wall_time_total_s=float(total_wall),
        wall_time_mean_s=mean_wall,
    )
    meta_dict = asdict(meta)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    print(f"[dataset] Wrote {masks_path}")
    print(f"[dataset] Wrote {labels_path}")
    print(f"[dataset] Total wall {total_wall:.1f}s (mean {mean_wall:.2f}s/sample)")
    return meta_dict


def load_dataset(dataset_dir: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Load masks (N,H,W), label arrays, and meta.json."""
    masks_path = os.path.join(dataset_dir, "masks.npy")
    labels_path = os.path.join(dataset_dir, "labels.csv")
    meta_path = os.path.join(dataset_dir, "meta.json")
    if not os.path.isfile(masks_path):
        raise FileNotFoundError(f"Missing masks: {masks_path}")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Missing labels: {labels_path}")

    masks = np.load(masks_path)
    labels: dict[str, list] = {}
    with open(labels_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                labels.setdefault(k, []).append(v)

    arrays: dict[str, np.ndarray] = {}
    for k, vals in labels.items():
        if k in ("shape",):
            arrays[k] = np.array(vals)
        elif k in ("sample_id",):
            arrays[k] = np.array([int(x) for x in vals], dtype=np.int32)
        else:
            arrays[k] = np.array([float(x) for x in vals], dtype=np.float64)

    meta: dict[str, Any] = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    if masks.shape[0] != len(arrays.get("cd_force_proxy", [])):
        raise ValueError(
            f"Mask count {masks.shape[0]} != label count {len(arrays.get('cd_force_proxy', []))}"
        )
    return masks, arrays, meta
