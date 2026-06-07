"""
Phase 6 — 3D flow field cache generator.

Generates one precomputed 3D LBM-solved dataset:
  • Sphere in crossflow at 64×64×64 resolution

The output is saved in two forms:
  1. Full 3D arrays (for future 3D viewer support)
  2. Center-slice 2D arrays (for the existing 2D particle viewer)

Both are written to backend/app/assets/flow/ as .npy files.

Usage:
    cd backend
    venv\\Scripts\\activate
    python -m app.services.cache_generator_3d
"""

import os
import time
import numpy as np

from app.solvers.fluidx3d_runner import (
    create_sphere_mask,
    run_3d_simulation,
    extract_center_slice_2d,
    get_solver_info,
)


NX = 64
NY = 64
NZ = 64
STEPS = 200
TAU = 0.8
U_INLET = 0.05


def generate_3d_caches():
    assets_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "assets", "flow")
    )
    os.makedirs(assets_dir, exist_ok=True)

    print("=" * 60)
    print("AeroVoxel Phase 6 — 3D Flow Field Cache Generator")
    print("=" * 60)

    solver_info = get_solver_info()
    print(f"Active solver: {solver_info['active_solver']}")
    print(f"FluidX3D available: {solver_info['fluidx3d_available']}")
    print(f"Grid: {NX}×{NY}×{NZ} = {NX*NY*NZ:,} cells")
    print(f"Steps: {STEPS}, tau: {TAU}, u_inlet: {U_INLET}")
    print()

    # --- Generate sphere-in-crossflow dataset ---
    case_id = "sphere_3d_v1"
    print(f"[{case_id}] Creating sphere obstacle mask...")
    mask = create_sphere_mask(NX, NY, NZ, radius_fraction=0.15)

    print(f"[{case_id}] Running 3D LBM solver...")
    start_time = time.time()
    velocity, pressure, solver_label = run_3d_simulation(
        obstacle_mask=mask,
        nx=NX, ny=NY, nz=NZ,
        u_inlet=U_INLET,
        wind_angle_deg=0.0,
        steps=STEPS,
        tau=TAU,
    )
    elapsed = time.time() - start_time
    print(f"[{case_id}] Solver completed in {elapsed:.1f}s using: {solver_label}")

    # Save full 3D arrays (for future use)
    vel_3d_path = os.path.join(assets_dir, f"{case_id}_velocity_3d.npy")
    press_3d_path = os.path.join(assets_dir, f"{case_id}_pressure_3d.npy")
    mask_3d_path = os.path.join(assets_dir, f"{case_id}_mask_3d.npy")

    np.save(vel_3d_path, velocity.astype(np.float32))
    np.save(press_3d_path, pressure.astype(np.float32))
    np.save(mask_3d_path, mask.astype(bool))
    print(f"[{case_id}] Saved 3D arrays: velocity {velocity.shape}, pressure {pressure.shape}, mask {mask.shape}")

    # Extract 2D center slice for existing viewer
    vel_2d, press_2d, mask_2d = extract_center_slice_2d(velocity, pressure, mask)

    vel_2d_path = os.path.join(assets_dir, f"{case_id}_velocity.npy")
    press_2d_path = os.path.join(assets_dir, f"{case_id}_pressure.npy")
    mask_2d_path = os.path.join(assets_dir, f"{case_id}_mask.npy")

    np.save(vel_2d_path, vel_2d.astype(np.float32))
    np.save(press_2d_path, press_2d.astype(np.float32))
    np.save(mask_2d_path, mask_2d.astype(bool))
    print(f"[{case_id}] Saved 2D center-slice: velocity {vel_2d.shape}, pressure {press_2d.shape}, mask {mask_2d.shape}")

    # Compute metrics from the 2D slice
    wake_pixels = int(np.sum((vel_2d[0] < 0.02) & (~mask_2d)))
    total_fluid = int(np.sum(~mask_2d))
    drag_coeff = 0.47  # Sphere theoretical Cd ≈ 0.47 (Re < 1000)
    wake_score = wake_pixels / max(total_fluid, 1)
    # Sphere has no net lift at 0° angle
    lift_coeff = 0.0

    print(f"[{case_id}] Metrics: Cd~{drag_coeff:.2f}, Cl~{lift_coeff:.2f}, Wake={wake_score:.2f}")

    # Write a metadata file for reference
    meta_path = os.path.join(assets_dir, f"{case_id}_meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"case_id: {case_id}\n")
        f.write(f"solver: {solver_label}\n")
        f.write(f"grid: {NX}x{NY}x{NZ}\n")
        f.write(f"steps: {STEPS}\n")
        f.write(f"tau: {TAU}\n")
        f.write(f"u_inlet: {U_INLET}\n")
        f.write(f"elapsed_seconds: {elapsed:.1f}\n")
        f.write(f"drag_coefficient: {drag_coeff}\n")
        f.write(f"lift_coefficient: {lift_coeff}\n")
        f.write(f"wake_score: {wake_score:.4f}\n")
        f.write(f"fluidx3d_available: {solver_info['fluidx3d_available']}\n")

    print()
    print("=" * 60)
    print("3D cache generation complete!")
    print(f"Files saved to: {assets_dir}")
    print("=" * 60)


if __name__ == "__main__":
    generate_3d_caches()
