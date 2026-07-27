"""
AeroVoxel evaluation runner (Milestone M3).

Loads locked validation cases (V1 cylinder 2D, V2 sphere 3D if present,
V3 airfoil 2D), reports grid / Reynolds estimates, catalog metrics, and
field-derived wake / pressure proxies. Optionally re-runs live 2D LBM for
wall-time measurement on the laptop.

Outputs JSON + CSV under repo-root evaluation_outputs/ (gitignored).

Usage (from backend/):
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --live-2d
    python scripts/run_evaluation.py --try-sphere-cache

Honest labels: Cd/Cl in this project are educational heuristics or catalog
constants — not force-integrated CFD coefficients.
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

# Allow `python scripts/run_evaluation.py` from backend/
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.routes.demo_cases import DEMO_CASES, FLOW_ASSETS_DIR, _case_assets_ready  # noqa: E402
from app.services.cache_generator import (  # noqa: E402
    NX as NX2,
    NY as NY2,
    STEPS as STEPS2,
    TAU as TAU2,
    U_INLET as U_INLET2,
    make_airfoil_mask,
    make_cylinder_mask,
)
from app.solvers.lbm_2d import LbmSolver2D  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(BACKEND_ROOT, ".."))
OUT_DIR = os.path.join(REPO_ROOT, "evaluation_outputs")

# Catalog case_id per validation ID
CASE_MAP = {
    "V1": "cylinder_v1",
    "V2": "sphere_3d_v1",
    "V3": "airfoil_v1",
}

# Literature order-of-magnitude references (not high-precision targets)
LITERATURE = {
    "V1": {
        "metric": "Cd (2D circular cylinder)",
        "reference": "O(1); ~1.5–1.7 near Re≈40 (classic experiments / textbooks)",
        "notes": "Coarse 128×64 LBM; catalog Cd is educational, not force-integrated.",
    },
    "V2": {
        "metric": "Cd (sphere) or qualitative wake/stagnation",
        "reference": "Subcritical sphere Cd ≈ 0.4–0.5 (Re ~10^3–10^5); at Re≪100 Cd is higher",
        "notes": "Catalog Cd 0.47 is a textbook subcritical placeholder; grid is coarse 64³.",
    },
    "V3": {
        "metric": "Cl proxy / suction-side Δp story",
        "reference": "NACA 0012 at small AoA: Cl ~ 2π α (rad) order; not XFoil-grade here",
        "notes": "Heuristic Cl from pressure asymmetry only; educational label required.",
    },
}


def nu_from_tau(tau: float) -> float:
    """BGK kinematic viscosity in lattice units: ν = (τ − 1/2) / 3."""
    return (tau - 0.5) / 3.0


def estimate_re(u: float, L: float, tau: float) -> float:
    return float(u * L / max(nu_from_tau(tau), 1e-12))


def load_case_arrays(case_id: str) -> dict[str, Any] | None:
    if not _case_assets_ready(case_id):
        return None
    vel = np.load(os.path.join(FLOW_ASSETS_DIR, f"{case_id}_velocity.npy"))
    pressure = np.load(os.path.join(FLOW_ASSETS_DIR, f"{case_id}_pressure.npy"))
    mask = np.load(os.path.join(FLOW_ASSETS_DIR, f"{case_id}_mask.npy"))
    return {"velocity": vel, "pressure": pressure, "mask": mask}


def catalog_entry(case_id: str) -> dict[str, Any] | None:
    return next((c for c in DEMO_CASES if c["case_id"] == case_id), None)


def field_metrics_2d(
    velocity: np.ndarray,
    pressure: np.ndarray,
    mask: np.ndarray,
    nx: int,
    ny: int,
) -> dict[str, float]:
    """Same educational heuristics as routes/simulate.py (+ a few structure proxies)."""
    # Velocity layout: (2, ny, nx) for 2D solver caches
    if velocity.ndim == 3 and velocity.shape[0] == 2:
        ux = velocity[0]
        uy = velocity[1]
    elif velocity.ndim == 2:
        ux = velocity
        uy = np.zeros_like(velocity)
    else:
        # Unexpected layout — best-effort first component
        ux = np.asarray(velocity[0] if velocity.shape[0] <= 3 else velocity)
        uy = np.zeros_like(ux)

    fluid = ~mask.astype(bool)
    wake_pixels = int(np.sum((ux < 0.02) & fluid))
    drag_coeff = float(0.15 + (wake_pixels / max(nx * ny, 1)) * 3.5)
    drag_coeff = max(0.05, min(1.8, drag_coeff))

    mid_y = ny // 2
    top_press = float(np.sum(pressure[:mid_y]))
    bottom_press = float(np.sum(pressure[mid_y:]))
    lift_coeff = float((bottom_press - top_press) * 1.5)
    lift_coeff = max(-0.8, min(1.5, lift_coeff))

    wake_score = float(wake_pixels / max(nx * (ny // 2), 1))
    wake_score = max(0.05, min(0.99, wake_score))

    # Structure proxies (not Cd/Cl)
    p_fluid = pressure[fluid]
    u_mag = np.sqrt(ux ** 2 + uy ** 2)
    u_fluid = u_mag[fluid]

    # Wake length proxy: farthest x column with mean fluid |u| < 0.4 * freestream-ish
    freestream = float(np.percentile(u_fluid, 75)) if u_fluid.size else 0.08
    low_u = (u_mag < 0.4 * max(freestream, 1e-6)) & fluid
    cols = np.where(np.any(low_u, axis=0))[0]
    wake_extent_x = int(cols.max() - cols.min() + 1) if cols.size else 0

    # Stagnation: max pressure on fluid (windward high-P expected)
    p_max = float(np.max(p_fluid)) if p_fluid.size else 0.0
    p_min = float(np.min(p_fluid)) if p_fluid.size else 0.0
    delta_p = p_max - p_min

    solid_frac = float(np.mean(mask.astype(bool)))

    return {
        "heuristic_cd": drag_coeff,
        "heuristic_cl": lift_coeff,
        "wake_score": wake_score,
        "wake_pixels": wake_pixels,
        "wake_extent_x_cells": wake_extent_x,
        "pressure_max": p_max,
        "pressure_min": p_min,
        "delta_p": delta_p,
        "u_mag_mean_fluid": float(np.mean(u_fluid)) if u_fluid.size else 0.0,
        "u_mag_max_fluid": float(np.max(u_fluid)) if u_fluid.size else 0.0,
        "solid_fraction": solid_frac,
    }


def qualitative_checks(
    velocity: np.ndarray,
    pressure: np.ndarray,
    mask: np.ndarray,
    case_kind: str,
) -> dict[str, Any]:
    """Simple automatic qualitative feature checks for education cases."""
    if velocity.ndim == 3 and velocity.shape[0] >= 2:
        ux, uy = velocity[0], velocity[1]
    else:
        ux = np.asarray(velocity[0] if getattr(velocity, "ndim", 0) == 3 else velocity)
        uy = np.zeros_like(ux)

    fluid = ~mask.astype(bool)
    ny, nx = mask.shape
    mid_y = ny // 2

    # Object bounding box
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0:
        return {"ok": False, "reason": "empty mask"}

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    cx = (x_min + x_max) // 2

    # Windward (left of object) vs leeward (right) mean pressure in a band
    band = slice(max(0, y_min - 2), min(ny, y_max + 3))
    left = slice(max(0, x_min - 8), max(0, x_min))
    right = slice(min(nx, x_max + 1), min(nx, x_max + 12))

    p_left = float(np.mean(pressure[band, left])) if left.stop > left.start else float("nan")
    p_right = float(np.mean(pressure[band, right])) if right.stop > right.start else float("nan")
    stagnation_front = bool(np.isfinite(p_left) and np.isfinite(p_right) and p_left > p_right)

    # Wake: low |u| behind object
    behind = slice(min(nx, x_max + 1), min(nx, x_max + 20))
    u_mag = np.sqrt(ux ** 2 + uy ** 2)
    u_behind = u_mag[band, behind]
    u_ahead = u_mag[band, left] if left.stop > left.start else u_mag[band, : max(1, x_min)]
    wake_slow = bool(u_behind.size and u_ahead.size and np.mean(u_behind) < np.mean(u_ahead))

    out: dict[str, Any] = {
        "stagnation_higher_pressure_windward": stagnation_front,
        "wake_slower_than_upstream": wake_slow,
        "object_bbox": [x_min, y_min, x_max, y_max],
    }

    if case_kind == "airfoil":
        # Suction side: above airfoil, expect higher speed / lower pressure than below at AoA>0
        top_band = slice(max(0, y_min - 6), y_min)
        bot_band = slice(y_max + 1, min(ny, y_max + 7))
        chord = slice(x_min, x_max + 1)
        u_top = float(np.mean(u_mag[top_band, chord])) if top_band.stop > top_band.start else float("nan")
        u_bot = float(np.mean(u_mag[bot_band, chord])) if bot_band.stop > bot_band.start else float("nan")
        p_top = float(np.mean(pressure[top_band, chord])) if top_band.stop > top_band.start else float("nan")
        p_bot = float(np.mean(pressure[bot_band, chord])) if bot_band.stop > bot_band.start else float("nan")
        out["mean_speed_above"] = u_top
        out["mean_speed_below"] = u_bot
        out["mean_pressure_above"] = p_top
        out["mean_pressure_below"] = p_bot
        out["suction_side_faster"] = bool(np.isfinite(u_top) and np.isfinite(u_bot) and u_top > u_bot)
        out["suction_side_lower_pressure"] = bool(
            np.isfinite(p_top) and np.isfinite(p_bot) and p_top < p_bot
        )

    if case_kind in ("cylinder", "sphere"):
        # Recirculation proxy: any negative ux in leeward fluid near midplane
        leeward = fluid & (np.arange(nx)[None, :] > x_max) & (np.abs(np.arange(ny)[:, None] - mid_y) < 8)
        reverse = bool(np.any(ux[leeward] < -0.005)) if np.any(leeward) else False
        out["leeward_reverse_flow_proxy"] = reverse

    return out


def run_live_2d(case_kind: str, steps: int = STEPS2) -> dict[str, Any]:
    if case_kind == "cylinder":
        mask = make_cylinder_mask()
        L = 16.0  # diameter lattice units
    elif case_kind == "airfoil":
        mask = make_airfoil_mask()
        L = 36.0  # chord lattice units
    else:
        raise ValueError(case_kind)

    solver = LbmSolver2D(nx=NX2, ny=NY2, tau=TAU2, u_inlet=U_INLET2, wind_angle_deg=0.0)
    t0 = time.perf_counter()
    u, pressure = solver.solve(mask, steps=steps)
    wall = time.perf_counter() - t0
    metrics = field_metrics_2d(u, pressure, mask, NX2, NY2)
    quals = qualitative_checks(u, pressure, mask, case_kind)
    return {
        "wall_time_s": wall,
        "steps": steps,
        "grid": {"nx": NX2, "ny": NY2, "nz": 1},
        "tau": TAU2,
        "u_inlet": U_INLET2,
        "Re_est": estimate_re(U_INLET2, L, TAU2),
        "char_length_lu": L,
        "metrics": metrics,
        "qualitative": quals,
    }


def try_generate_sphere_cache(timeout_hint_s: float = 900.0) -> dict[str, Any]:
    """Run offline 64³ sphere cache once. Returns status dict."""
    t0 = time.perf_counter()
    try:
        from app.services.cache_generator_3d import generate_3d_caches

        generate_3d_caches()
        elapsed = time.perf_counter() - t0
        return {
            "status": "generated",
            "wall_time_s": elapsed,
            "within_budget": elapsed <= timeout_hint_s,
            "assets_ready": _case_assets_ready("sphere_3d_v1"),
        }
    except Exception as exc:  # noqa: BLE001 — evaluation must continue
        return {
            "status": "failed",
            "wall_time_s": time.perf_counter() - t0,
            "error": str(exc),
            "assets_ready": False,
        }


def evaluate_v1(live: bool) -> dict[str, Any]:
    case_id = CASE_MAP["V1"]
    cat = catalog_entry(case_id)
    arrays = load_case_arrays(case_id)
    L = 16.0
    result: dict[str, Any] = {
        "validation_id": "V1",
        "case_id": case_id,
        "name": cat["name"] if cat else "cylinder",
        "grid": cat["grid"] if cat else {"nx": 128, "ny": 64, "nz": 1},
        "solver_path": "cached 2D LBM field (cylinder_v1) + optional live D2Q9",
        "tau_cache_gen": TAU2,
        "u_inlet_cache_gen": U_INLET2,
        "Re_est": estimate_re(U_INLET2, L, TAU2),
        "char_length_lu": L,
        "char_length_meaning": "cylinder diameter ≈ 16 LU",
        "catalog_metrics": {
            "drag_coefficient_estimate": cat["drag_coefficient_estimate"] if cat else None,
            "lift_coefficient_estimate": cat["lift_coefficient_estimate"] if cat else None,
            "wake_score": cat["wake_score"] if cat else None,
            "source": "hardcoded educational estimates in demo_cases.py",
        },
        "literature": LITERATURE["V1"],
        "assets_ready": arrays is not None,
    }
    if arrays:
        result["field_metrics_from_cache"] = field_metrics_2d(
            arrays["velocity"], arrays["pressure"], arrays["mask"], 128, 64
        )
        result["qualitative_from_cache"] = qualitative_checks(
            arrays["velocity"], arrays["pressure"], arrays["mask"], "cylinder"
        )
        result["array_shapes"] = {
            "velocity": list(arrays["velocity"].shape),
            "pressure": list(arrays["pressure"].shape),
            "mask": list(arrays["mask"].shape),
        }
    if live:
        result["live_2d"] = run_live_2d("cylinder")
    return result


def evaluate_v3(live: bool) -> dict[str, Any]:
    case_id = CASE_MAP["V3"]
    cat = catalog_entry(case_id)
    arrays = load_case_arrays(case_id)
    L = 36.0
    result: dict[str, Any] = {
        "validation_id": "V3",
        "case_id": case_id,
        "name": cat["name"] if cat else "airfoil",
        "grid": cat["grid"] if cat else {"nx": 128, "ny": 64, "nz": 1},
        "solver_path": "cached 2D LBM field (airfoil_v1) + optional live D2Q9",
        "tau_cache_gen": TAU2,
        "u_inlet_cache_gen": U_INLET2,
        "Re_est": estimate_re(U_INLET2, L, TAU2),
        "char_length_lu": L,
        "char_length_meaning": "airfoil chord ≈ 36 LU, geometric AoA ≈ 8°",
        "catalog_metrics": {
            "drag_coefficient_estimate": cat["drag_coefficient_estimate"] if cat else None,
            "lift_coefficient_estimate": cat["lift_coefficient_estimate"] if cat else None,
            "wake_score": cat["wake_score"] if cat else None,
            "source": "hardcoded educational estimates in demo_cases.py",
        },
        "literature": LITERATURE["V3"],
        "assets_ready": arrays is not None,
    }
    if arrays:
        result["field_metrics_from_cache"] = field_metrics_2d(
            arrays["velocity"], arrays["pressure"], arrays["mask"], 128, 64
        )
        result["qualitative_from_cache"] = qualitative_checks(
            arrays["velocity"], arrays["pressure"], arrays["mask"], "airfoil"
        )
        result["array_shapes"] = {
            "velocity": list(arrays["velocity"].shape),
            "pressure": list(arrays["pressure"].shape),
            "mask": list(arrays["mask"].shape),
        }
    if live:
        result["live_2d"] = run_live_2d("airfoil")
    return result


def evaluate_v2(try_cache: bool) -> dict[str, Any]:
    case_id = CASE_MAP["V2"]
    cat = catalog_entry(case_id)
    meta_path = os.path.join(FLOW_ASSETS_DIR, f"{case_id}_meta.txt")
    meta: dict[str, str] = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()

    # Sphere diameter from cache_generator_3d: radius_fraction 0.15 of min dim
    nx = 64
    radius = 0.15 * nx
    D = 2.0 * radius
    tau = float(meta.get("tau", 0.8))
    u_inlet = float(meta.get("u_inlet", 0.05))

    result: dict[str, Any] = {
        "validation_id": "V2",
        "case_id": case_id,
        "name": cat["name"] if cat else "sphere 3D",
        "grid": cat["grid"] if cat else {"nx": 64, "ny": 64, "nz": 64},
        "solver_path": "offline D3Q19 LBM precompute → center-slice cache",
        "meta_file": meta if meta else None,
        "tau": tau,
        "u_inlet": u_inlet,
        "Re_est": estimate_re(u_inlet, D, tau),
        "char_length_lu": D,
        "char_length_meaning": "sphere diameter ≈ 19.2 LU (radius_fraction=0.15 on 64³)",
        "catalog_metrics": {
            "drag_coefficient_estimate": cat["drag_coefficient_estimate"] if cat else 0.47,
            "lift_coefficient_estimate": cat["lift_coefficient_estimate"] if cat else 0.0,
            "wake_score": cat["wake_score"] if cat else None,
            "source": "hardcoded textbook subcritical Cd≈0.47 in demo_cases / cache_generator_3d",
        },
        "literature": LITERATURE["V2"],
        "assets_ready": _case_assets_ready(case_id),
    }

    if not result["assets_ready"] and try_cache:
        print("[V2] Sphere center-slice arrays missing — attempting offline 64³ generation...")
        gen = try_generate_sphere_cache()
        result["cache_generation"] = gen
        result["assets_ready"] = gen.get("assets_ready", False)

    if result["assets_ready"]:
        arrays = load_case_arrays(case_id)
        if arrays:
            # Center-slice is 2D
            ny, nx_s = arrays["mask"].shape
            result["field_metrics_from_cache"] = field_metrics_2d(
                arrays["velocity"], arrays["pressure"], arrays["mask"], nx_s, ny
            )
            result["qualitative_from_cache"] = qualitative_checks(
                arrays["velocity"], arrays["pressure"], arrays["mask"], "sphere"
            )
            result["array_shapes"] = {
                "velocity": list(arrays["velocity"].shape),
                "pressure": list(arrays["pressure"].shape),
                "mask": list(arrays["mask"].shape),
            }
    else:
        result["status"] = (
            "cache not generated; qualitative N/A — only meta.txt present or generation failed"
        )
        if meta:
            result["prior_meta_elapsed_s"] = meta.get("elapsed_seconds")
            result["prior_meta_wake_score"] = meta.get("wake_score")

    return result


def agreement_band(case_id: str, catalog_cd: float | None, re_est: float) -> str:
    if case_id == "V1":
        # Literature O(1) at moderate Re; catalog ~1.1
        return "Order-of-magnitude O(1) agreement with classic 2D cylinder Cd; not quantitative validation"
    if case_id == "V2":
        if re_est < 100:
            return (
                f"Re_est≈{re_est:.0f} is low; textbook subcritical Cd~0.47 is a poor match regime — "
                "use qualitative stagnation/wake only; Cd label educational/rough"
            )
        return "Band ~0.4–0.5 only if Re subcritical; coarse-grid caveat applies"
    if case_id == "V3":
        return "Qualitative lift direction / suction-side story only; Cl not XFoil-comparable"
    return "n/a"


def table_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for vid in ("V1", "V2", "V3"):
        r = results[vid]
        cat = r.get("catalog_metrics") or {}
        field = r.get("field_metrics_from_cache") or {}
        live = r.get("live_2d") or {}
        live_m = live.get("metrics") or {}

        if vid == "V1":
            metric_name = "Cd (catalog) / wake_score (field)"
            av_val = (
                f"Cd_cat={cat.get('drag_coefficient_estimate')}; "
                f"wake_score_field={field.get('wake_score')}; "
                f"Cd_heur_field={field.get('heuristic_cd')}"
            )
            if live_m:
                av_val += f"; live Cd_heur={live_m.get('heuristic_cd')} ({live.get('wall_time_s'):.2f}s)"
            ref = r["literature"]["reference"]
            label = "educational"
        elif vid == "V2":
            if not r.get("assets_ready"):
                metric_name = "Cd est. or qualitative score"
                av_val = "N/A — center-slice .npy cache not available"
                ref = r["literature"]["reference"]
                label = "fail / N/A"
            else:
                metric_name = "Cd (catalog) / wake + stagnation (field)"
                av_val = (
                    f"Cd_cat={cat.get('drag_coefficient_estimate')}; "
                    f"wake_score_field={field.get('wake_score')}; "
                    f"Δp={field.get('delta_p')}"
                )
                ref = r["literature"]["reference"]
                label = "educational / rough"
        else:
            metric_name = "Cl (catalog) / Cl_heur (field) / suction story"
            av_val = (
                f"Cl_cat={cat.get('lift_coefficient_estimate')}; "
                f"Cl_heur_field={field.get('heuristic_cl')}; "
                f"wake_score={field.get('wake_score')}"
            )
            if live_m:
                av_val += f"; live Cl_heur={live_m.get('heuristic_cl')} ({live.get('wall_time_s'):.2f}s)"
            ref = r["literature"]["reference"]
            label = "educational"

        grid = r.get("grid") or {}
        grid_s = f"{grid.get('nx')}×{grid.get('ny')}" + (
            f"×{grid.get('nz')}" if grid.get("nz", 1) and grid.get("nz") != 1 else ""
        )
        rows.append(
            {
                "case": vid,
                "grid": grid_s,
                "Re_est": r.get("Re_est"),
                "metric": metric_name,
                "aerovoxel_value": av_val,
                "reference": ref,
                "agreement": agreement_band(vid, cat.get("drag_coefficient_estimate"), float(r.get("Re_est") or 0)),
                "label": label,
                "assets_ready": r.get("assets_ready"),
                "live_wall_s": live.get("wall_time_s"),
                "qualitative": r.get("qualitative_from_cache") or (live.get("qualitative") if live else None),
            }
        )
    return rows


def write_outputs(payload: dict[str, Any]) -> tuple[str, str]:
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(OUT_DIR, f"m3_evaluation_{stamp}.json")
    csv_path = os.path.join(OUT_DIR, "m3_evaluation_summary.csv")
    latest_json = os.path.join(OUT_DIR, "m3_evaluation_latest.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    rows = payload["table_a_rows"]
    fieldnames = [
        "case",
        "grid",
        "Re_est",
        "metric",
        "aerovoxel_value",
        "reference",
        "agreement",
        "label",
        "assets_ready",
        "live_wall_s",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Wrote {json_path}")
    print(f"Wrote {latest_json}")
    print(f"Wrote {csv_path}")
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="AeroVoxel M3 evaluation runner")
    parser.add_argument(
        "--live-2d",
        action="store_true",
        default=True,
        help="Re-run live 2D LBM for V1/V3 wall times (default: on)",
    )
    parser.add_argument(
        "--no-live-2d",
        action="store_true",
        help="Skip live 2D re-solves; use caches only",
    )
    parser.add_argument(
        "--try-sphere-cache",
        action="store_true",
        default=True,
        help="If sphere .npy missing, run offline 64³ generator once (default: on)",
    )
    parser.add_argument(
        "--no-sphere-cache",
        action="store_true",
        help="Do not attempt sphere 3D cache generation",
    )
    args = parser.parse_args()
    live = args.live_2d and not args.no_live_2d
    try_sphere = args.try_sphere_cache and not args.no_sphere_cache

    print("=" * 60)
    print("AeroVoxel M3 evaluation")
    print(f"FLOW_ASSETS_DIR: {FLOW_ASSETS_DIR}")
    print(f"live_2d={live}, try_sphere_cache={try_sphere}")
    print("=" * 60)

    t_all = time.perf_counter()
    results = {
        "V1": evaluate_v1(live=live),
        "V3": evaluate_v3(live=live),
        "V2": evaluate_v2(try_cache=try_sphere),
    }
    wall_all = time.perf_counter() - t_all

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_note": "Target laptop: Ryzen 7 5700U, 16GB, iGPU only; live 2D CPU; 3D offline ≤64³",
        "method": {
            "catalog_metrics": "demo_cases.py hardcoded educational estimates",
            "field_heuristics": "Same formulas as app/routes/simulate.py (wake-area Cd, pressure-asymmetry Cl)",
            "Re": "Re = u_inlet * L / ν, ν = (τ−0.5)/3 lattice BGK",
            "cache_gen_2d": "app/services/cache_generator.py (800 steps, τ=0.6, u=0.08, 128×64)",
            "cache_gen_3d": "app/services/cache_generator_3d.py (200 steps, τ=0.8, u=0.05, 64³)",
            "script": "backend/scripts/run_evaluation.py",
        },
        "results": results,
        "table_a_rows": table_rows(results),
        "total_wall_time_s": wall_all,
    }

    write_outputs(payload)

    print("\n--- Table A summary ---")
    for row in payload["table_a_rows"]:
        print(
            f"{row['case']}: grid={row['grid']} Re≈{row['Re_est']:.1f} "
            f"ready={row['assets_ready']} label={row['label']}"
        )
        print(f"  value: {row['aerovoxel_value']}")
        print(f"  agreement: {row['agreement']}")
    print(f"\nTotal wall time: {wall_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
