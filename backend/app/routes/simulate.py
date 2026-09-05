import os
import time
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from app.safe_ids import require_safe_job_id
from app.solvers.lbm_2d import LbmSolver2D, educational_force_metrics

router = APIRouter(prefix="/api")

# Directory where uploaded files and custom run results are stored
UPLOADS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "uploads")
)

# Matches LbmSolver2D defaults used by this endpoint
_SIM_NX = 128
_SIM_NY = 64

class SimulateRequest(BaseModel):
    job_id: str
    wind_speed: float
    wind_angle_deg: float

class SimulateResponse(BaseModel):
    job_id: str
    status: str
    grid: dict
    velocity_url: str
    pressure_url: str
    mask_url: str
    metrics: dict

@router.post("/simulate/simple", response_model=SimulateResponse)
async def run_simulation(req: SimulateRequest):
    """Runs the 2D CPU-based LBM solver for the uploaded object silhouette mask."""
    job_id = require_safe_job_id(req.job_id)
    # Find the mask file
    mask_path = os.path.join(UPLOADS_DIR, f"mask_{job_id}.npy")
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Upload silhouette mask not found for this Job ID")
        
    try:
        # Load boundary mask
        mask = np.load(mask_path)
        if mask.ndim != 2 or mask.shape != (_SIM_NY, _SIM_NX):
            raise HTTPException(
                status_code=400,
                detail=f"Mask shape {getattr(mask, 'shape', None)} must be ({_SIM_NY}, {_SIM_NX})",
            )
        if not np.any(mask.astype(bool)):
            raise HTTPException(status_code=400, detail="Mask has no solid cells")
        
        # Scale physical wind speed to lattice velocity (Reference: 15 m/s = 0.08 lattice units)
        u_inlet = float(req.wind_speed * 0.08 / 15.0)
        u_inlet = max(0.02, min(0.15, u_inlet)) # keep LBM stable
        
        simulation_started = time.perf_counter()
        # Instantiate and run LBM solver
        solver = LbmSolver2D(
            nx=_SIM_NX,
            ny=_SIM_NY,
            tau=0.6,
            u_inlet=u_inlet,
            wind_angle_deg=req.wind_angle_deg,
        )
        # Run solver (600 iterations is plenty for visual convergence at 128x64)
        u, pressure = solver.solve(mask, steps=600, force_avg_steps=40)

        # Save computed arrays
        vel_path = os.path.join(UPLOADS_DIR, f"velocity_{job_id}.npy")
        press_path = os.path.join(UPLOADS_DIR, f"pressure_{job_id}.npy")

        np.save(vel_path, u.astype(np.float32))
        np.save(press_path, pressure.astype(np.float32))

        # Heuristic Cd from wake area (legacy educational estimate)
        fluid = mask == False
        wake_pixels = np.sum((u[0] < 0.02) & fluid)
        drag_coeff = float(0.15 + (wake_pixels / (_SIM_NX * _SIM_NY)) * 3.5)
        drag_coeff = max(0.05, min(1.8, drag_coeff))

        # Lift estimate: proportional to top-to-bottom pressure asymmetry
        top_press = np.sum(pressure[: _SIM_NY // 2])
        bottom_press = np.sum(pressure[_SIM_NY // 2 :])
        lift_coeff = float((bottom_press - top_press) * 1.5)
        lift_coeff = max(-0.8, min(1.5, lift_coeff))

        wake_score = float(wake_pixels / (_SIM_NX * (_SIM_NY // 2)))
        wake_score = max(0.05, min(0.99, wake_score))

        # Characteristic length ≈ solid height (frontal length) in lattice units
        ys, xs = np.where(mask.astype(bool))
        char_L = float(ys.max() - ys.min() + 1) if ys.size else 16.0
        force_m = educational_force_metrics(
            pressure=pressure,
            mask=mask,
            u_ref=u_inlet,
            char_length=char_L,
            velocity=u,
            tau=0.6,
            momentum_force_lu=solver.last_force_lu,
            momentum_method=solver.last_force_method,
        )

        return SimulateResponse(
            job_id=job_id,
            status="completed",
            grid={"nx": _SIM_NX, "ny": _SIM_NY, "nz": 1},
            velocity_url=f"/api/simulate/result/{job_id}/velocity",
            pressure_url=f"/api/simulate/result/{job_id}/pressure",
            mask_url=f"/api/simulate/result/{job_id}/mask",
            metrics={
                "drag_coefficient_estimate": drag_coeff,
                "lift_coefficient_estimate": lift_coeff,
                "wake_score": wake_score,
                "cd_force_proxy": force_m["cd_force_proxy"],
                "cd_surface_proxy": force_m["cd_surface_proxy"],
                "cd_force_proxy_method": force_m["cd_force_proxy_method"],
                "lbm_seconds": round(time.perf_counter() - simulation_started, 3),
                "confidence_label": "educational estimate (not certified CFD)",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.get("/simulate/result/{job_id}/velocity")
async def get_sim_velocity(job_id: str):
    job_id = require_safe_job_id(job_id)
    file_path = os.path.join(UPLOADS_DIR, f"velocity_{job_id}.npy")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Simulated velocity array not found")
    return FileResponse(file_path, media_type="application/octet-stream", filename=f"velocity_{job_id}.npy")

@router.get("/simulate/result/{job_id}/pressure")
async def get_sim_pressure(job_id: str):
    job_id = require_safe_job_id(job_id)
    file_path = os.path.join(UPLOADS_DIR, f"pressure_{job_id}.npy")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Simulated pressure array not found")
    return FileResponse(file_path, media_type="application/octet-stream", filename=f"pressure_{job_id}.npy")

@router.get("/simulate/result/{job_id}/mask")
async def get_sim_mask(job_id: str):
    job_id = require_safe_job_id(job_id)
    file_path = os.path.join(UPLOADS_DIR, f"mask_{job_id}.npy")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Silhouette mask array not found")
    return FileResponse(file_path, media_type="application/octet-stream", filename=f"mask_{job_id}.npy")
