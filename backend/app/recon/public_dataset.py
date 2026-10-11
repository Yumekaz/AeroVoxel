"""Preparation helpers for reproducible public real-world reconstruction evals."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    candidates: list[tuple[int, dict[str, Any], Path]] = []
    skipped_missing = 0
    skipped_flagged = 0
    skipped_corrupt = 0
    for index, row in enumerate(rows):
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
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, ValueError):
            skipped_corrupt += 1
            continue
        candidates.append((index, row, image_path))

    # Prefer one sample per category for a small evaluation, then fill any
    # remaining slots in the dataset's original stable order.
    chosen: list[tuple[int, dict[str, Any], Path]] = []
    seen_categories: set[str] = set()
    for candidate in candidates:
        category = str(candidate[1].get("category") or "unknown")
        if category not in seen_categories:
            chosen.append(candidate)
            seen_categories.add(category)
            if len(chosen) == n_cases:
                break
    if len(chosen) < n_cases:
        chosen_indexes = {item[0] for item in chosen}
        chosen.extend(item for item in candidates if item[0] not in chosen_indexes)
        chosen = chosen[:n_cases]

    selected: list[dict[str, str]] = []
    for source_index, row, image_path in chosen:
        original_index = row.get("_source_index", source_index)
        try:
            original_index = int(original_index)
        except (TypeError, ValueError):
            original_index = source_index
        with Image.open(image_path) as image:
            width, height = image.size
        def local_reference(field: str) -> str:
            value = row.get(field)
            if not isinstance(value, str) or not value:
                return ""
            try:
                candidate = _safe_dataset_path(root, value)
            except ValueError:
                return ""
            return str(candidate) if candidate.is_file() else ""

        selected.append(
            {
                "case_id": f"PIX3D_{len(selected) + 1:02d}",
                "input_type": "PUBLIC_DATASET",
                "dataset": "Pix3D",
                "dataset_version": "official_release_version_unspecified",
                "dataset_source_url": "https://pix3d.csail.mit.edu/",
                "dataset_license": "CC BY 4.0",
                "dataset_root": str(root.resolve()),
                "dataset_split": "official_release_unsplit",
                "sample_index": str(original_index),
                "image_path": str(image_path.resolve()),
                "category": str(row.get("category") or "unknown"),
                "image_source": str(row.get("img_source") or "unknown"),
                "image_width": str(width),
                "image_height": str(height),
                "image_sha256": _sha256(image_path),
                "ground_truth_mask_path": local_reference("mask"),
                "ground_truth_model_path": local_reference("model"),
                "ground_truth_voxel_path": local_reference("voxel"),
                "truncated": str(bool(row.get("truncated", False))).lower(),
                "occluded": str(bool(row.get("occluded", False))).lower(),
                "evaluation_status": "READY",
                "failure_reason": "",
            }
        )

    if len(selected) < n_cases:
        raise ValueError(
            f"Pix3D contains only {len(selected)} usable images; requested {n_cases} "
            f"(missing={skipped_missing}, corrupt={skipped_corrupt}, flagged={skipped_flagged})"
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(selected[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return selected
