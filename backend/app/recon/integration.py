"""Bridge exported reconstruction masks into the existing simulation job layout."""

from __future__ import annotations

import os
import uuid
from typing import Any

import numpy as np


def register_mask_for_simulation(mask_path: str, uploads_dir: str, nx: int = 128, ny: int = 64) -> dict[str, Any]:
    """Copy a validated recon mask into the API's existing upload-job contract.

    This deliberately does not run the solver or fabricate an API response. The
    returned UUID can be sent to the existing ``/api/simulate/simple`` endpoint.
    """
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(mask_path)
    mask = np.load(mask_path, allow_pickle=False)
    return register_mask_array(mask, uploads_dir, nx=nx, ny=ny)


def register_mask_array(mask: np.ndarray, uploads_dir: str, nx: int = 128, ny: int = 64) -> dict[str, Any]:
    """Validate and persist a normalized binary solver mask, with a fresh UUID."""
    if mask.shape != (ny, nx):
        raise ValueError(f"Recon mask shape {mask.shape} must be ({ny}, {nx})")
    if mask.dtype.kind not in "buif" or not np.all(np.isfinite(mask)) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Recon mask must contain finite binary 0/1 cells")
    mask = mask.astype(bool, copy=False)
    if not np.any(mask):
        raise ValueError("Recon mask has no solid cells")
    if np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
        raise ValueError("Recon mask must leave fluid cells at all domain boundaries")
    os.makedirs(uploads_dir, exist_ok=True)
    job_id = str(uuid.uuid4())
    destination = os.path.join(uploads_dir, f"mask_{job_id}.npy")
    np.save(destination, mask, allow_pickle=False)
    return {
        "job_id": job_id,
        "mask_path": os.path.abspath(destination),
        "shape": [ny, nx],
        "solid_cells": int(np.sum(mask)),
        "next_step": "POST /api/simulate/simple with this job_id",
    }
