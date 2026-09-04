"""Explicit non-3D final fallback using the existing classical CV pipeline."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import cv2
import numpy as np

from app.recon.depth_mesh import extract_foreground_mask


def run_opencv_fallback(image_path: str, output_dir: str, **_: Any) -> dict[str, Any]:
    """Produce a real 2D domain when no ML reconstruction engine is runnable."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    os.makedirs(output_dir, exist_ok=True)
    started = time.perf_counter()
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")
    mask = extract_foreground_mask(image)
    resized = cv2.resize(mask.astype(np.uint8), (128, 64), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not np.any(resized):
        raise ValueError("OpenCV fallback produced an empty simulation mask")
    mask_path = os.path.join(output_dir, "mask.npy")
    np.save(mask_path, resized)
    result = {
        "engine": "opencv_2d_fallback",
        "device": "cpu",
        "image_path": os.path.abspath(image_path),
        "mask": {"mask_path": os.path.abspath(mask_path), "shape": [64, 128], "solid_cells": int(np.sum(resized))},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "mesh_path": None,
        "output_valid": True,
        "honesty": "Classical 2D silhouette only; this is not ML or 3D reconstruction.",
    }
    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
