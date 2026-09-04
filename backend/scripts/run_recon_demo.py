"""Run Stable Fast 3D and export an AeroVoxel-compatible center-slice mask."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.recon.sf3d_runner import run_sf3d
from app.recon.integration import register_mask_for_simulation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="One object photo (JPG/PNG) with a clean background")
    parser.add_argument("--output-dir", default="data/recon_outputs/demo")
    parser.add_argument("--model", default="stabilityai/stable-fast-3d")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--texture-resolution", type=int, default=512)
    parser.add_argument(
        "--register-for-api",
        action="store_true",
        help="Copy mask into app/assets/uploads and print a job_id for /api/simulate/simple",
    )
    args = parser.parse_args()
    result = run_sf3d(
        args.image,
        args.output_dir,
        model_id=args.model,
        device=args.device,
        texture_resolution=args.texture_resolution,
    )
    if args.register_for_api:
        uploads_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "assets", "uploads"))
        result["simulation_job"] = register_mask_for_simulation(result["mask"]["mask_path"], uploads_dir)
    print(json.dumps(result, indent=2))
    print(f"[recon] Feed the exported mask into: POST /api/simulate/simple (job_id upload contract)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
