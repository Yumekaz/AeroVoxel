"""Prepare a typed manifest for one genuine phone-captured image."""

from __future__ import annotations

import argparse
import os
import sys

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.recon.phone_capture import prepare_phone_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the supplied phone JPG/PNG/WEBP")
    parser.add_argument("--output", required=True, help="CSV manifest path to create")
    parser.add_argument("--case-id", default="PHONE_01")
    args = parser.parse_args()
    row = prepare_phone_manifest(args.image, args.output, args.case_id)
    print(f"Wrote {row['input_type']} case {row['case_id']} to {args.output}")


if __name__ == "__main__":
    main()
