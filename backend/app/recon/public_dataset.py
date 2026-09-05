"""Preparation helpers for reproducible public real-world reconstruction evals."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


def _metadata_rows(metadata: Any) -> list[dict[str, Any]]:
    if isinstance(metadata, list):
        rows = metadata
    elif isinstance(metadata, dict) and isinstance(metadata.get("data"), list):
        rows = metadata["data"]
    else:
        raise ValueError("Pix3D metadata must be a JSON list of records")
    return [row for row in rows if isinstance(row, dict)]


def _is_flagged(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, (int, float)) and value == 1:
        return True
    return isinstance(value, str) and value.lower() in {"1", "true", "yes"}


def _safe_dataset_path(dataset_root: Path, relative_path: str) -> Path:
    if not relative_path or os.path.isabs(relative_path):
        raise ValueError(f"Pix3D path must be a relative path: {relative_path!r}")
    root = dataset_root.resolve()
    candidate = (root / relative_path.replace("/", os.sep)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Pix3D path escapes dataset root: {relative_path!r}") from exc
    return candidate


def prepare_pix3d_manifest(dataset_root: str, output_path: str, n_cases: int = 5) -> list[dict[str, str]]:
    """Select usable Pix3D image records and write a typed evaluation manifest.

    Pix3D is a public real-world image/shape dataset, not a phone-capture set.
    The generated manifest therefore uses ``PUBLIC_DATASET`` explicitly.  The
    function never downloads data and never invents missing image paths.
    """
    if n_cases < 1:
        raise ValueError("n_cases must be positive")
    root = Path(dataset_root)
    metadata_path = root / "pix3d.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Pix3D metadata not found: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        rows = _metadata_rows(json.load(handle))

    selected: list[dict[str, str]] = []
    skipped_missing = 0
    skipped_flagged = 0
    for row in rows:
        if _is_flagged(row.get("truncated")) or _is_flagged(row.get("occluded")):
            skipped_flagged += 1
            continue
        image_value = row.get("img") or row.get("image")
        if not isinstance(image_value, str):
            skipped_missing += 1
            continue
        try:
            image_path = _safe_dataset_path(root, image_value)
        except ValueError:
            skipped_missing += 1
            continue
        if not image_path.is_file():
            skipped_missing += 1
            continue
        selected.append(
            {
                "case_id": f"PIX3D_{len(selected) + 1:02d}",
                "input_type": "PUBLIC_DATASET",
                "image_path": str(image_path),
            }
        )
        if len(selected) == n_cases:
            break

    if len(selected) < n_cases:
        raise ValueError(
            f"Pix3D contains only {len(selected)} usable images; requested {n_cases} "
            f"(missing={skipped_missing}, flagged={skipped_flagged})"
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "input_type", "image_path"])
        writer.writeheader()
        writer.writerows(selected)
    return selected
