"""Preparation helpers for one consented local phone-capture evaluation."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Sequence


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _as_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def prepare_phone_manifest(
    image_path: str | Sequence[str],
    output_path: str,
    case_id: str | Sequence[str] = "PHONE_01",
) -> list[dict[str, str]]:
    """Validate one or more supplied photos and write a typed phone manifest."""
    image_paths = _as_list(image_path)
    case_ids = _as_list(case_id) if not isinstance(case_id, str) or case_id != "PHONE_01" else []
    if not case_ids:
        case_ids = [f"PHONE_{index:02d}" for index in range(1, len(image_paths) + 1)]
    if len(case_ids) != len(image_paths):
        raise ValueError("Provide exactly one case id per image")

    from PIL import Image

    rows: list[dict[str, str]] = []
    seen_hashes: dict[str, str] = {}
    for raw_path, raw_case_id in zip(image_paths, case_ids):
        source = Path(raw_path).expanduser().resolve()
        case_id_value = raw_case_id.strip()
        if not case_id_value:
            raise ValueError("case_id must be non-empty")
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension {source.suffix!r}; use JPG, PNG, or WEBP")
        try:
            with Image.open(source) as image:
                image.verify()
            with Image.open(source) as image:
                width, height = image.size
        except Exception as exc:
            raise ValueError(f"Phone capture is not a readable image: {source}") from exc
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest in seen_hashes:
            raise ValueError(f"Duplicate phone capture: {source} has the same SHA-256 as {seen_hashes[digest]}")
        seen_hashes[digest] = str(source)
        rows.append(
            {
                "case_id": case_id_value,
                "input_type": "REAL_PHONE_CAPTURE",
                "image_path": str(source),
                "sha256": digest,
                "width": str(width),
                "height": str(height),
                "provenance": "user-supplied physical phone photograph",
            }
        )

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "input_type", "image_path", "sha256", "width", "height", "provenance"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows
