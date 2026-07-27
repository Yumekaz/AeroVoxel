import os
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from app.solvers.lbm_2d import LbmSolver2D, educational_force_metrics

router = APIRouter(prefix="/api")

# Directory where uploaded files and custom run results are stored
UPLOADS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "uploads")
)

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
    # Find the mask file
    mask_path = os.path.join(UPLOADS_DIR, f"mask_{req.job_id}.npy")
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Upload silhouette mask not found for this Job ID")
        
    try:
        # Load boundary mask
        mask = np.load(mask_path)
        
        # Scale physical wind speed to lattice velocity (Reference: 15 m/s = 0.08 lattice units)
        u_inlet = float(req.wind_speed * 0.08 / 15.0)
        u_inlet = max(0.02, min(0.15, u_inlet)) # keep LBM stable
        
        # Instantiate and run LBM solver
        solver = LbmSolver2D(
            nx=128,
            ny=64,
            tau=0.6,
            u_inlet=u_inlet,
            wind_angle_deg=req.wind_angle_deg,
        )
        # Run solver (600 iterations is plenty for visual convergence at 128x64)
        u, pressure = solver.solve(mask, steps=600, force_avg_steps=40)

        # Save computed arrays
        vel_path = os.path.join(UPLOADS_DIR, f"velocity_{req.job_id}.npy")
        press_path = os.path.join(UPLOADS_DIR, f"pressure_{req.job_id}.npy")

        np.save(vel_path, u.astype(np.float32))
        np.save(press_path, pressure.astype(np.float32))

        # Heuristic Cd from wake area (legacy educational estimate)
        fluid = mask == False
        wake_pixels = np.sum((u[0] < 0.02) & fluid)
        drag_coeff = float(0.15 + (wake_pixels / (128 * 64)) * 3.5)
        drag_coeff = max(0.05, min(1.8, drag_coeff))

        # Lift estimate: proportional to top-to-bottom pressure asymmetry
        top_press = np.sum(pressure[:32])
        bottom_press = np.sum(pressure[32:])
        lift_coeff = float((bottom_press - top_press) * 1.5)
        lift_coeff = max(-0.8, min(1.5, lift_coeff))

        wake_score = float(wake_pixels / (128 * 32))
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
            job_id=req.job_id,
            status="completed",
            grid={"nx": 128, "ny": 64, "nz": 1},
            velocity_url=f"/api/simulate/result/{req.job_id}/velocity",
            pressure_url=f"/api/simulate/result/{req.job_id}/pressure",
            mask_url=f"/api/simulate/result/{req.job_id}/mask",
            metrics={
                "drag_coefficient_estimate": drag_coeff,
                "lift_coefficient_estimate": lift_coeff,
                "wake_score": wake_score,
                "cd_force_proxy": force_m["cd_force_proxy"],
                "cd_surface_proxy": force_m["cd_surface_proxy"],
                "cd_force_proxy_method": force_m["cd_force_proxy_method"],
                "confidence_label": "educational estimate (not certified CFD)",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.get("/simulate/result/{job_id}/velocity")
async def get_sim_velocity(job_id: str):
    file_path = os.path.join(UPLOADS_DIR, f"velocity_{job_id}.npy")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Simulated velocity array not found")
    return FileResponse(file_path, media_type="application/octet-stream", filename=f"velocity_{job_id}.npy")

@router.get("/simulate/result/{job_id}/pressure")
async def get_sim_pressure(job_id: str):
    file_path = os.path.join(UPLOADS_DIR, f"pressure_{job_id}.npy")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Simulated pressure array not found")
    return FileResponse(file_path, media_type="application/octet-stream", filename=f"pressure_{job_id}.npy")

@router.get("/simulate/result/{job_id}/mask")
async def get_sim_mask(job_id: str):
    file_path = os.path.join(UPLOADS_DIR, f"mask_{job_id}.npy")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Silhouette mask array not found")
    return FileResponse(file_path, media_type="application/octet-stream", filename=f"mask_{job_id}.npy")
