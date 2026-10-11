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
    parser.add_argument("--image", required=True, nargs="+", help="One or more supplied phone JPG/PNG/WEBP files")
    parser.add_argument("--output", required=True, help="CSV manifest path to create")
    parser.add_argument("--case-id", nargs="+", help="Optional case ids, one per image")
    args = parser.parse_args()
    case_ids = args.case_id if args.case_id else "PHONE_01"
    rows = prepare_phone_manifest(args.image, args.output, case_ids)
    print(f"Wrote {len(rows)} REAL_PHONE_CAPTURE case(s) to {args.output}")


if __name__ == "__main__":
    main()
