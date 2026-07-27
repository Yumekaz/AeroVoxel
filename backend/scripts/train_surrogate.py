"""
Train educational aero surrogate models on an LBM-labeled mask dataset.

Usage (from backend/):
    python scripts/train_surrogate.py
    python scripts/train_surrogate.py --dataset data/ml_surrogate --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.ml.dataset import default_dataset_dir  # noqa: E402
from app.ml.metrics import format_metrics_table  # noqa: E402
from app.ml.train import train_surrogate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train educational LBM surrogate")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset directory (default: backend/data/ml_surrogate)",
    )
    parser.add_argument("--target", type=str, default="cd_force_proxy")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Where to write joblib bundle (default: <dataset>/models)",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset or default_dataset_dir(BACKEND_ROOT)
    if not os.path.isdir(dataset_dir):
        print(f"Dataset not found: {dataset_dir}")
        print("Run: python scripts/generate_ml_dataset.py --n 100")
        return 1

    print(f"[train] dataset={dataset_dir}")
    eval_payload = train_surrogate(
        dataset_dir=dataset_dir,
        model_dir=args.model_dir,
        target=args.target,
        test_size=args.test_size,
        seed=args.seed,
    )

    print()
    print("=" * 60)
    print("Held-out metrics (educational LBM labels)")
    print("=" * 60)
    print(format_metrics_table(eval_payload["results"]))
    print()
    print(
        f"LBM mean wall time: {eval_payload['lbm_mean_wall_time_s']:.3f}s "
        f"({eval_payload['lbm_mean_latency_ms']:.1f} ms/sample)"
    )
    print(f"MLP beats mean MAE: {eval_payload['mlp_beats_mean_mae']}")
    print(f"Model: {eval_payload['model_path']}")
    print(f"Metrics JSON: artifacts + {os.path.dirname(eval_payload['model_path'])}/last_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
