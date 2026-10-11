"""Preparation helpers for one consented local phone-capture evaluation."""

from __future__ import annotations

import csv
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def prepare_phone_manifest(image_path: str, output_path: str, case_id: str = "PHONE_01") -> dict[str, str]:
    """Validate one supplied image and write an explicitly typed PHONE_CAPTURE manifest."""
    source = Path(image_path).expanduser().resolve()
    if not case_id.strip():
        raise ValueError("case_id must be non-empty")
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension {source.suffix!r}; use JPG, PNG, or WEBP")
    try:
        from PIL import Image

        with Image.open(source) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"Phone capture is not a readable image: {source}") from exc

    row = {"case_id": case_id.strip(), "input_type": "PHONE_CAPTURE", "image_path": str(source)}
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "input_type", "image_path"])
        writer.writeheader()
        writer.writerow(row)
    return row
