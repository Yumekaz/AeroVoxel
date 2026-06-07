import os
import numpy as np

from app.solvers.lbm_2d import LbmSolver2D

NX = 128
NY = 64
STEPS = 800
TAU = 0.6
U_INLET = 0.08


def make_car_mask():
    mask = np.zeros((NY, NX), dtype=bool)
    for x in range(NX):
        for y in range(NY):
            if (40 <= x <= 80 and 48 <= y <= 60):
                if x < 50:
                    slope_height = int(48 + (50 - x) * 1.2)
                    if y >= slope_height:
                        mask[y, x] = True
                elif x > 72:
                    slope_height = int(48 + (x - 72) * 1.5)
                    if y >= slope_height:
                        mask[y, x] = True
                else:
                    mask[y, x] = True
            elif (50 <= x <= 72 and 38 <= y <= 48):
                if x < 56:
                    slope_height = int(38 + (56 - x) * 1.6)
                    if y >= slope_height:
                        mask[y, x] = True
                elif x > 66:
                    slope_height = int(38 + (x - 66) * 1.6)
                    if y >= slope_height:
                        mask[y, x] = True
                else:
                    mask[y, x] = True
    return mask


def make_drone_mask():
    mask = np.zeros((NY, NX), dtype=bool)
    cx, cy = 64, 32
    for x in range(NX):
        for y in range(NY):
            if (x - cx) ** 2 + (y - cy) ** 2 <= 9 ** 2:
                mask[y, x] = True
            if 36 <= x <= 92 and 30 <= y <= 34:
                mask[y, x] = True
            if (x - 36) ** 2 + (y - cy) ** 2 <= 4 ** 2:
                mask[y, x] = True
            if (x - 92) ** 2 + (y - cy) ** 2 <= 4 ** 2:
                mask[y, x] = True
    return mask


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


def generate_and_save_caches():
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "flow"))
    os.makedirs(assets_dir, exist_ok=True)

    print(f"Generating aerodynamic caches in: {assets_dir}")

    presets = [
        ("sports_car_v1", make_car_mask()),
        ("drone_v1", make_drone_mask()),
        ("airfoil_v1", make_airfoil_mask()),
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