"""Evaluate real reconstruction inputs from a typed CSV manifest."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.recon.evaluation import evaluate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="CSV with case_id,input_type,image_path")
    parser.add_argument("--output-dir", default="evaluation_outputs/recon")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--engine", choices=["auto", "sf3d", "depth_anything", "opencv"], default="auto")
    args = parser.parse_args()
    summary = evaluate_manifest(args.manifest, args.output_dir, device=args.device, engine=args.engine)
    print(f"Wrote {os.path.abspath(args.output_dir)}\\results.json")
    print(f"Engine: {args.engine}; succeeded={summary['n_succeeded']}; failed={summary['n_failed']}")
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
