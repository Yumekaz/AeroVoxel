"""
Generate synthetic masks + LBM educational labels for the ML surrogate.

Usage (from backend/):
    python scripts/generate_ml_dataset.py --n 100 --seed 42
    python scripts/generate_ml_dataset.py --n 300 --force
    python scripts/generate_ml_dataset.py --nx 128 --ny 64 --n 80 --steps 400
"""

from __future__ import annotations

import argparse
import os
import sys

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.ml.dataset import (  # noqa: E402
    DEFAULT_FORCE_AVG,
    DEFAULT_NX,
    DEFAULT_NY,
    DEFAULT_STEPS,
    DEFAULT_TAU,
    DEFAULT_U_INLET,
    default_dataset_dir,
    generate_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate LBM-labeled mask dataset for educational surrogate"
    )
    parser.add_argument("--n", type=int, default=100, help="Number of samples (default 100; production-ish 300–500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nx", type=int, default=DEFAULT_NX)
    parser.add_argument("--ny", type=int, default=DEFAULT_NY)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--u-inlet", type=float, default=DEFAULT_U_INLET, dest="u_inlet")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--force-avg-steps", type=int, default=DEFAULT_FORCE_AVG)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: backend/data/ml_surrogate)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if dataset already exists",
    )
    args = parser.parse_args()

    out_dir = args.out or default_dataset_dir(BACKEND_ROOT)
    meta = generate_dataset(
        out_dir=out_dir,
        n=args.n,
        seed=args.seed,
        nx=args.nx,
        ny=args.ny,
        tau=args.tau,
        u_inlet=args.u_inlet,
        steps=args.steps,
        force_avg_steps=args.force_avg_steps,
        force=args.force,
    )
    print("[done] meta n_samples=", meta.get("n_samples"), "wall_s=", meta.get("wall_time_total_s"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
