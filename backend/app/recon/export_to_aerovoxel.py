"""Convert an SF3D mesh into the mask contract used by the AeroVoxel solver."""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np


def _load_mesh(mesh_path: str) -> Any:
    try:
        import trimesh  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Mesh export requires optional dependency 'trimesh'. "
            "Install backend/requirements-recon.txt."
        ) from exc

    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Mesh scene contains no geometry: {mesh_path}")
        mesh = loaded.to_geometry()
    else:
        mesh = loaded
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Mesh contains no vertices/faces: {mesh_path}")
    return mesh


def fit_mask_to_domain(mask: np.ndarray, nx: int = 128, ny: int = 64) -> np.ndarray:
    """Keep XY proportions inside the same central 60x32 envelope as uploads."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask) or nx < 16 or ny < 16:
        raise ValueError("A non-empty 2D mask and a domain >=16x16 are required")
    ys, xs = np.where(mask)
    cropped = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
    max_w, max_h = max(1, round(nx * 60 / 128)), max(1, round(ny / 2))
    scale = min(max_w / cropped.shape[1], max_h / cropped.shape[0])
    width, height = max(1, round(cropped.shape[1]*scale)), max(1, round(cropped.shape[0]*scale))
    obstacle = cv2.resize(cropped.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    domain = np.zeros((ny, nx), dtype=bool)
    left, top = (nx-width)//2, (ny-height)//2
    domain[top:top+height, left:left+width] = obstacle
    return domain


def mesh_to_center_slice_mask(
    mesh_path: str,
    output_path: str,
    nx: int = 128,
    ny: int = 64,
    z_fraction: float = 0.5,
    fill: bool = True,
    voxel_resolution_factor: float = 1.0,
) -> dict[str, Any]:
    """Voxelize a mesh and export its middle Z slice as a solver-ready mask.

    The exported array is ``(ny, nx)`` bool, matching ``/api/simulate/simple``.
    The mesh is normalized into a central envelope while preserving aspect ratio;
    it is intentionally a domain mask, not a metrically scaled reconstruction.
    """
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(mesh_path)
    if nx < 16 or ny < 16:
        raise ValueError("nx and ny must both be at least 16")
    if not 0.0 <= z_fraction <= 1.0:
        raise ValueError("z_fraction must be in [0, 1]")
    if voxel_resolution_factor <= 0:
        raise ValueError("voxel_resolution_factor must be positive")

    mesh = _load_mesh(mesh_path)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extent = bounds[1] - bounds[0]
    if np.any(extent <= 0) or not np.all(np.isfinite(extent)):
        raise ValueError("Mesh has degenerate or non-finite bounds")

    # A pitch based on the largest XY extent gives a predictable memory bound.
    target_pixels = max(16, int(round(max(nx, ny) * voxel_resolution_factor)))
    pitch = float(max(extent[0], extent[1], extent[2]) / target_pixels)
    voxels = mesh.voxelized(pitch)
    if fill:
        voxels = voxels.fill()
    dense = np.asarray(voxels.matrix, dtype=bool)
    if dense.ndim != 3 or not np.any(dense):
        raise ValueError("Mesh voxelization produced an empty occupancy grid")

    z_index = min(dense.shape[2] - 1, int(round(z_fraction * (dense.shape[2] - 1))))
    # Trimesh occupancy axes are X,Y,Z; solver arrays are rows Y, columns X.
    # Positive mesh Y is up; image/solver row Y grows down.
    slice_mask = dense[:, :, z_index].T[::-1, :]
    resized = fit_mask_to_domain(slice_mask, nx=nx, ny=ny)
    if not np.any(resized):
        raise ValueError("Center slice is empty; try a different z_fraction or mesh orientation")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, resized.astype(bool))
    return {
        "mask_path": os.path.abspath(output_path),
        "shape": [ny, nx],
        "solid_cells": int(np.sum(resized)),
        "solid_fraction": float(np.mean(resized)),
        "z_fraction": z_fraction,
        "voxel_resolution_factor": voxel_resolution_factor,
        "voxel_grid_shape": list(dense.shape),
        "note": "Best-effort center slice; scale is normalized and not metrology-grade.",
    }
