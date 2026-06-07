"""
FluidX3D integration wrapper for AeroVoxel Phase 6.

Strategy:
1. Try to import the `fluidx3d` Python package (GPU-accelerated).
2. If unavailable, fall back to the pure-Python D3Q19 solver.

Both paths produce identical output format:
  - velocity: ndarray (3, nz, ny, nx) float32
  - pressure: ndarray (nz, ny, nx) float32
  - mask: ndarray (nz, ny, nx) bool
"""

import os
import numpy as np

# Detect FluidX3D availability at import time
FLUIDX3D_AVAILABLE = False
try:
    import fluidx3d
    FLUIDX3D_AVAILABLE = True
    print("[FluidX3D] GPU-accelerated fluidx3d package detected.")
except ImportError:
    print("[FluidX3D] fluidx3d pip package not found. Using pure-Python D3Q19 fallback.")


def get_solver_info() -> dict:
    """Return information about available solver backends."""
    return {
        "fluidx3d_available": FLUIDX3D_AVAILABLE,
        "fallback_solver": "Pure-Python D3Q19 LBM (NumPy)",
        "active_solver": "FluidX3D (OpenCL GPU)" if FLUIDX3D_AVAILABLE else "Pure-Python D3Q19 LBM (NumPy)",
    }


def create_sphere_mask(nx: int, ny: int, nz: int, radius_fraction: float = 0.15) -> np.ndarray:
    """
    Create a 3D boolean mask with a sphere obstacle centered in the domain.

    The sphere is positioned at (0.3 * nx, 0.5 * ny, 0.5 * nz) — slightly
    upstream of center so we can see the wake region behind it.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.
    radius_fraction : float
        Sphere radius as a fraction of the smallest domain dimension.

    Returns
    -------
    mask : ndarray of bool, shape (nz, ny, nx)
    """
    radius = radius_fraction * min(nx, ny, nz)
    cx = int(0.3 * nx)
    cy = int(0.5 * ny)
    cz = int(0.5 * nz)

    z, y, x = np.ogrid[0:nz, 0:ny, 0:nx]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    mask = dist_sq <= radius ** 2

    print(f"[FluidX3D] Sphere mask created: center=({cx},{cy},{cz}), radius={radius:.1f}, "
          f"solid cells={int(np.sum(mask))}/{nx*ny*nz}")
    return mask


def run_3d_simulation(
    obstacle_mask: np.ndarray,
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    u_inlet: float = 0.05,
    wind_angle_deg: float = 0.0,
    steps: int = 200,
    tau: float = 0.8,
) -> tuple:
    """
    Run a 3D LBM simulation using the best available solver.

    Returns
    -------
    velocity : ndarray (3, nz, ny, nx) float32
    pressure : ndarray (nz, ny, nx) float32
    solver_label : str — which solver was actually used
    """
    solver_label = "Unknown"

    # --- Approach A: Try FluidX3D GPU solver ---
    if FLUIDX3D_AVAILABLE:
        try:
            solver_label = "FluidX3D OpenCL GPU (D3Q19)"
            print(f"[FluidX3D] Attempting GPU-accelerated simulation at {nx}×{ny}×{nz}...")
            # The fluidx3d pip package interface is command-line oriented.
            # For programmatic integration, we would need to write custom
            # setup.cpp and compile. For now, fall through to pure-Python.
            raise NotImplementedError("FluidX3D Python API does not support programmatic array export yet")
        except Exception as e:
            print(f"[FluidX3D] GPU solver failed ({e}). Falling back to pure-Python.")

    # --- Approach B: Pure-Python D3Q19 ---
    from app.solvers.lbm_3d import LbmSolver3D

    solver_label = "Pure-Python D3Q19 LBM (NumPy, CPU)"
    solver = LbmSolver3D(
        nx=nx, ny=ny, nz=nz,
        tau=tau,
        u_inlet=u_inlet,
        wind_angle_deg=wind_angle_deg,
    )
    velocity, pressure = solver.solve(obstacle_mask, steps=steps)

    return velocity, pressure, solver_label


def extract_center_slice_2d(
    velocity_3d: np.ndarray,
    pressure_3d: np.ndarray,
    mask_3d: np.ndarray,
    slice_axis: str = "z",
) -> tuple:
    """
    Extract a 2D center slice from 3D arrays for compatibility
    with AeroVoxel's existing 2D viewer.

    Parameters
    ----------
    velocity_3d : ndarray (3, nz, ny, nx)
    pressure_3d : ndarray (nz, ny, nx)
    mask_3d : ndarray (nz, ny, nx)
    slice_axis : str — 'z' takes the center z-plane (default)

    Returns
    -------
    velocity_2d : ndarray (2, ny, nx) — [ux, uy] at the center slice
    pressure_2d : ndarray (ny, nx)
    mask_2d : ndarray (ny, nx)
    """
    if slice_axis == "z":
        z_mid = velocity_3d.shape[1] // 2
        vel_2d = np.stack([
            velocity_3d[0, z_mid, :, :],  # ux
            velocity_3d[1, z_mid, :, :],  # uy
        ], axis=0).astype(np.float32)
        press_2d = pressure_3d[z_mid, :, :].astype(np.float32)
        mask_2d = mask_3d[z_mid, :, :]
    else:
        raise ValueError(f"Unsupported slice axis: {slice_axis}")

    return vel_2d, press_2d, mask_2d
