"""Run Stable Fast 3D and export an AeroVoxel-compatible center-slice mask."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.recon.engine import select_reconstruction_engine
from app.recon.integration import register_mask_for_simulation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="One object photo (JPG/PNG) with a clean background")
    parser.add_argument("--output-dir", default="data/recon_outputs/demo")
    parser.add_argument("--engine", choices=["auto", "sf3d", "depth_anything", "opencv"], default="auto")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--texture-resolution", type=int, default=512)
    parser.add_argument(
        "--register-for-api",
        action="store_true",
        help="Copy mask into app/assets/uploads and print a job_id for /api/simulate/simple",
    )
    args = parser.parse_args()
    engine = select_reconstruction_engine(args.engine)
    result = engine.reconstruct(args.image, args.output_dir, device=args.device)
    result["selection"] = {
        "requested": args.engine,
        "selected": engine.name,
        "reason": getattr(engine, "selection_reason", "explicit engine selection"),
        "fallback": args.engine == "auto" and engine.name != "stable-fast-3d",
    }
    if args.register_for_api:
        uploads_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "assets", "uploads"))
        result["simulation_job"] = register_mask_for_simulation(result["mask"]["mask_path"], uploads_dir)
    print(json.dumps(result, indent=2))
    print(f"[recon] Feed the exported mask into: POST /api/simulate/simple (job_id upload contract)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
