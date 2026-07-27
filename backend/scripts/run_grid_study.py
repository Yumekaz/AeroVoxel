"""
AeroVoxel cylinder 2D grid study (V&V).

Runs the same non-dimensional cylinder setup at multiple resolutions
(default 64×32 and 128×64), reports wall time, Re estimate, wake proxies,
heuristic Cd, and force-based cd_force_proxy (momentum exchange + surface).

Outputs JSON/CSV under repo-root evaluation_outputs/ (gitignored).

Usage (from backend/):
    python scripts/run_grid_study.py
    python scripts/run_grid_study.py --resolutions 64x32,128x64
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.solvers.lbm_2d import (  # noqa: E402
    LbmSolver2D,
    educational_force_metrics,
)

REPO_ROOT = os.path.abspath(os.path.join(BACKEND_ROOT, ".."))
OUT_DIR = os.path.join(REPO_ROOT, "evaluation_outputs")

# Matched non-dimensional setup across grids (same Re):
#   u_inlet, tau fixed; cylinder diameter D fixed in lattice units.
# Finer grids resolve the same D with more cells and a larger domain.
TAU = 0.6
U_INLET = 0.08
DIAMETER_LU = 12.0  # fixed so Re is matched across resolutions
RADIUS = DIAMETER_LU / 2.0


def nu_from_tau(tau: float) -> float:
    return (tau - 0.5) / 3.0


def estimate_re(u: float, L: float, tau: float) -> float:
    return float(u * L / max(nu_from_tau(tau), 1e-12))


def make_cylinder_mask(nx: int, ny: int, diameter: float = DIAMETER_LU) -> np.ndarray:
    """Cylinder upstream of midplane; diameter fixed for matched Re."""
    mask = np.zeros((ny, nx), dtype=bool)
    cx = int(round(nx * 0.30))
    cy = ny // 2
    r = diameter / 2.0
    r2 = r * r
    yy, xx = np.ogrid[:ny, :nx]
    mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r2] = True
    return mask


def field_wake_metrics(
    velocity: np.ndarray,
    pressure: np.ndarray,
    mask: np.ndarray,
    nx: int,
    ny: int,
) -> dict[str, float]:
    ux = velocity[0]
    uy = velocity[1]
    fluid = ~mask.astype(bool)
    wake_pixels = int(np.sum((ux < 0.02) & fluid))
    heuristic_cd = float(0.15 + (wake_pixels / max(nx * ny, 1)) * 3.5)
    heuristic_cd = max(0.05, min(1.8, heuristic_cd))
    wake_score = float(wake_pixels / max(nx * (ny // 2), 1))
    wake_score = max(0.05, min(0.99, wake_score))

    # Qualitative: reverse flow leeward of solid bbox
    ys, xs = np.where(mask.astype(bool))
    mid_y = ny // 2
    if xs.size:
        x_max = int(xs.max())
        leeward = fluid & (np.arange(nx)[None, :] > x_max) & (
            np.abs(np.arange(ny)[:, None] - mid_y) < max(4, ny // 8)
        )
        reverse = bool(np.any(ux[leeward] < -0.005)) if np.any(leeward) else False
        # Windward vs leeward pressure
        x_min = int(xs.min())
        y_min, y_max = int(ys.min()), int(ys.max())
        band = slice(max(0, y_min - 2), min(ny, y_max + 3))
        left = slice(max(0, x_min - 6), max(0, x_min))
        right = slice(min(nx, x_max + 1), min(nx, x_max + 10))
        p_left = float(np.mean(pressure[band, left])) if left.stop > left.start else float("nan")
        p_right = float(np.mean(pressure[band, right])) if right.stop > right.start else float("nan")
        stagnation_front = bool(
            np.isfinite(p_left) and np.isfinite(p_right) and p_left > p_right
        )
    else:
        reverse = False
        stagnation_front = False

    return {
        "heuristic_cd": heuristic_cd,
        "wake_score": wake_score,
        "wake_pixels": wake_pixels,
        "leeward_reverse_flow_proxy": reverse,
        "stagnation_higher_pressure_windward": stagnation_front,
        "wake_present_qualitative": bool(reverse or wake_score > 0.08),
        "u_mag_mean_fluid": float(np.mean(np.sqrt(ux ** 2 + uy ** 2)[fluid])) if np.any(fluid) else 0.0,
    }


def steps_for_grid(nx: int, ny: int) -> int:
    """Scale steps lightly with domain size; keep laptop-friendly."""
    # Reference: ~600 steps at 128×64
    base = 600 * (nx * ny) / (128 * 64)
    return int(max(350, min(900, round(base))))


def parse_resolution(spec: str) -> tuple[int, int]:
    parts = spec.lower().replace("*", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"Bad resolution '{spec}', expected e.g. 64x32")
    return int(parts[0]), int(parts[1])


def run_one(nx: int, ny: int) -> dict[str, Any]:
    mask = make_cylinder_mask(nx, ny, DIAMETER_LU)
    steps = steps_for_grid(nx, ny)
    re_est = estimate_re(U_INLET, DIAMETER_LU, TAU)
    solid_frac = float(np.mean(mask))

    solver = LbmSolver2D(nx=nx, ny=ny, tau=TAU, u_inlet=U_INLET, wind_angle_deg=0.0)
    t0 = time.perf_counter()
    u, pressure = solver.solve(mask, steps=steps, force_avg_steps=min(50, max(20, steps // 10)))
    wall = time.perf_counter() - t0

    wake = field_wake_metrics(u, pressure, mask, nx, ny)
    force_m = educational_force_metrics(
        pressure=pressure,
        mask=mask,
        u_ref=U_INLET,
        char_length=DIAMETER_LU,
        velocity=u,
        tau=TAU,
        momentum_force_lu=solver.last_force_lu,
        momentum_method=solver.last_force_method,
    )

    return {
        "grid": f"{nx}x{ny}",
        "nx": nx,
        "ny": ny,
        "steps": steps,
        "tau": TAU,
        "u_inlet": U_INLET,
        "diameter_lu": DIAMETER_LU,
        "cells_per_diameter": DIAMETER_LU,
        "Re_est": re_est,
        "nu_lu": nu_from_tau(TAU),
        "solid_fraction": solid_frac,
        "wall_time_s": wall,
        "wake_score": wake["wake_score"],
        "heuristic_cd": wake["heuristic_cd"],
        "cd_force_proxy": force_m["cd_force_proxy"],
        "cd_surface_proxy": force_m["cd_surface_proxy"],
        "cd_momentum_exchange_proxy": force_m.get("cd_momentum_exchange_proxy"),
        "cd_force_proxy_method": force_m["cd_force_proxy_method"],
        "force_x_lu": force_m["force_x_lu"],
        "force_y_lu": force_m["force_y_lu"],
        "leeward_reverse_flow_proxy": wake["leeward_reverse_flow_proxy"],
        "stagnation_higher_pressure_windward": wake["stagnation_higher_pressure_windward"],
        "wake_present_qualitative": wake["wake_present_qualitative"],
        "u_mag_mean_fluid": wake["u_mag_mean_fluid"],
        "label": "educational — not certified CFD",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AeroVoxel cylinder 2D grid study")
    parser.add_argument(
        "--resolutions",
        type=str,
        default="64x32,128x64",
        help="Comma-separated nx x ny list (default: 64x32,128x64)",
    )
    args = parser.parse_args()

    specs = [s.strip() for s in args.resolutions.split(",") if s.strip()]
    resolutions = [parse_resolution(s) for s in specs]

    print("=" * 60)
    print("AeroVoxel cylinder 2D grid study")
    print(f"Matched setup: D={DIAMETER_LU} LU, u={U_INLET}, tau={TAU}")
    print(f"Re_est ≈ {estimate_re(U_INLET, DIAMETER_LU, TAU):.2f} (all grids)")
    print(f"Resolutions: {', '.join(f'{n}x{m}' for n, m in resolutions)}")
    print("=" * 60)

    rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    for nx, ny in resolutions:
        print(f"\n--- Running {nx}x{ny} ---")
        row = run_one(nx, ny)
        rows.append(row)
        print(
            f"  wall={row['wall_time_s']:.2f}s  Re≈{row['Re_est']:.1f}  "
            f"wake_score={row['wake_score']:.3f}  "
            f"Cd_heur={row['heuristic_cd']:.3f}  "
            f"Cd_force={row['cd_force_proxy']:.3f}  "
            f"wake_present={row['wake_present_qualitative']}"
        )

    total = time.perf_counter() - t_all
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "study": "m5_grid_study_cylinder_2d",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_note": "Target laptop: Ryzen 7 5700U, 16GB, iGPU only; 2D CPU LBM",
        "method": {
            "geometry": f"circular cylinder diameter={DIAMETER_LU} LU (fixed across grids for matched Re)",
            "Re": "Re = u_inlet * D / ν, ν = (τ−0.5)/3",
            "cd_force_proxy": (
                "Primary: discrete surface pressure + rough viscous traction on solid faces; "
                "Cd=2 Fx/(ρ U² D). Secondary: fluid-side bounce-back momentum exchange average. "
                "Educational only — not certified CFD."
            ),
            "heuristic_cd": "Wake-area formula (same family as simulate.py)",
            "literature_hint": "2D cylinder Cd ~1.5–1.7 near Re≈40; coarse LBM will not match quantitatively",
            "script": "backend/scripts/run_grid_study.py",
        },
        "rows": rows,
        "total_wall_time_s": total,
    }

    json_path = os.path.join(OUT_DIR, f"m5_grid_study_{stamp}.json")
    latest = os.path.join(OUT_DIR, "m5_grid_study_latest.json")
    csv_path = os.path.join(OUT_DIR, "m5_grid_study_summary.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    fieldnames = [
        "grid",
        "nx",
        "ny",
        "steps",
        "Re_est",
        "wall_time_s",
        "wake_score",
        "heuristic_cd",
        "cd_force_proxy",
        "cd_surface_proxy",
        "cd_momentum_exchange_proxy",
        "cd_force_proxy_method",
        "wake_present_qualitative",
        "leeward_reverse_flow_proxy",
        "stagnation_higher_pressure_windward",
        "label",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print("\n--- Grid study table ---")
    print(
        f"{'grid':<10} {'Re':>7} {'wall_s':>8} {'wake':>7} {'Cd_heur':>8} "
        f"{'Cd_force':>9} {'wake?':>6}"
    )
    for row in rows:
        print(
            f"{row['grid']:<10} {row['Re_est']:7.1f} {row['wall_time_s']:8.2f} "
            f"{row['wake_score']:7.3f} {row['heuristic_cd']:8.3f} "
            f"{row['cd_force_proxy']:9.3f} {str(row['wake_present_qualitative']):>6}"
        )
    print(f"\nWrote {json_path}")
    print(f"Wrote {latest}")
    print(f"Wrote {csv_path}")
    print(f"Total wall time: {total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
