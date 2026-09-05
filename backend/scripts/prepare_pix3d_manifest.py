"""Create a typed five-case manifest from a local Pix3D download."""

from __future__ import annotations

import argparse
import os
import sys

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.recon.public_dataset import prepare_pix3d_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pix3d-root", required=True, help="Directory containing pix3d.json and image files")
    parser.add_argument("--output", required=True, help="CSV path to create")
    parser.add_argument("--n", type=int, default=5, help="Number of usable cases (default: 5)")
    args = parser.parse_args()
    rows = prepare_pix3d_manifest(args.pix3d_root, args.output, args.n)
    print(f"Wrote {len(rows)} PUBLIC_DATASET cases to {args.output}")


if __name__ == "__main__":
    main()
