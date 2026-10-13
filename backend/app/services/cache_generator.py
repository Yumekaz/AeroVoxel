import os
import json
from pathlib import Path
import numpy as np

from app.solvers.lbm_2d import LbmSolver2D

NX = 128
NY = 64
STEPS = 800
TAU = 0.6
U_INLET = 0.08


def _display_projection(case_id: str) -> np.ndarray:
    source = Path(__file__).resolve().parents[3] / "shared" / "demoGeometry.json"
    with source.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data["nx"] != NX or data["ny"] != NY:
        raise ValueError("Display projection grid does not match the cache grid")
    values = np.asarray(data["cases"][case_id]["mask"])
    if values.size != NX * NY or not np.all((values == 0) | (values == 1)):
        raise ValueError("Invalid display projection occupancy")
    return values.reshape(NY, NX).astype(bool)


def make_car_mask():
    """Exact side projection of the displayed concept-car geometry."""
    return _display_projection("sports_car_v1")


def make_drone_mask():
    """Exact side projection of the displayed quadcopter geometry."""
    return _display_projection("drone_v1")


def make_airfoil_mask():
    mask = np.zeros((NY, NX), dtype=bool)
    cx, cy = 60, 32
    chord = 36.0
    alpha = np.radians(8.0)

    for x in range(NX):
        for y in range(NY):
            dx = x - 42
            dy = y - cy
            rx = dx * np.cos(alpha) + dy * np.sin(alpha)
            ry = -dx * np.sin(alpha) + dy * np.cos(alpha)

            if 0 <= rx <= chord:
                x_frac = rx / chord
                half_thickness = chord * 5.0 * 0.12 * (
                    0.2969 * np.sqrt(x_frac)
                    - 0.1260 * x_frac
                    - 0.3516 * (x_frac ** 2)
                    + 0.2843 * (x_frac ** 3)
                    - 0.1015 * (x_frac ** 4)
                )
                if abs(ry) <= half_thickness:
                    mask[y, x] = True
    return mask


def make_cylinder_mask():
    """Circular cylinder for educational bluff-body validation (V1).

    Placed upstream of domain midplane so a recirculating wake can develop.
    Diameter ~16 lattice units on the shared 128×64 demo grid.
    """
    mask = np.zeros((NY, NX), dtype=bool)
    cx, cy = 40, 32
    radius = 8
    r2 = radius * radius
    for x in range(NX):
        for y in range(NY):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                mask[y, x] = True
    return mask


def generate_and_save_caches():
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "flow"))
    os.makedirs(assets_dir, exist_ok=True)

    print(f"Generating aerodynamic caches in: {assets_dir}")

    presets = [
        ("sports_car_v1", make_car_mask()),
        ("drone_v1", make_drone_mask()),
        ("airfoil_v1", make_airfoil_mask()),
        ("cylinder_v1", make_cylinder_mask()),
    ]

    solver = LbmSolver2D(nx=NX, ny=NY, tau=TAU, u_inlet=U_INLET)

    for name, mask in presets:
        print(f"Running LBM solver for preset: {name}...")
        u, pressure = solver.solve(mask, steps=STEPS)

        u_path = os.path.join(assets_dir, f"{name}_velocity.npy")
        p_path = os.path.join(assets_dir, f"{name}_pressure.npy")
        m_path = os.path.join(assets_dir, f"{name}_mask.npy")

        np.save(u_path, u.astype(np.float32))
        np.save(p_path, pressure.astype(np.float32))
        np.save(m_path, mask.astype(bool))

        print(f"Saved: {name} arrays (Velocity shape: {u.shape}, Pressure shape: {pressure.shape})")


if __name__ == "__main__":
    generate_and_save_caches()