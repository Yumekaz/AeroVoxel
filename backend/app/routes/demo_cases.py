import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api")

# Define the absolute path to the precomputed flow assets
FLOW_ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "flow")
)

DEMO_CASES = [
    {
        "case_id": "sports_car_v1",
        "name": "Sports Car (Template)",
        "mode": "cached_2d",
        "mode_label": "Cached 2D demonstration field",
        "description": "Low-drag vehicle profile with ground boundary flow structures.",
        "explanation": (
            "Air hits the front bumper creating a high-pressure stagnation zone. It accelerates over "
            "the hood and windshield (low pressure), then separates at the rear, creating a "
            "recirculating wake that contributes to drag."
        ),
        "drag_coefficient_estimate": 0.28,
        "lift_coefficient_estimate": -0.05,
        "wake_score": 0.65,
        "grid": {"nx": 128, "ny": 64, "nz": 1},
    },
    {
        "case_id": "drone_v1",
        "name": "Quadcopter (Template)",
        "mode": "cached_2d",
        "mode_label": "Cached 2D demonstration field",
        "description": "Complex vertical thrust profile with high drag wake.",
        "explanation": (
            "Flow encounters bluff-body engine mounts and rotor disks. The arms create flow separation "
            "and localized vortex shedding, resulting in a large wake turbulence index and substantial "
            "drag relative to frontal area."
        ),
        "drag_coefficient_estimate": 1.15,
        "lift_coefficient_estimate": 1.42,
        "wake_score": 0.88,
        "grid": {"nx": 128, "ny": 64, "nz": 1},
    },
    {
        "case_id": "airfoil_v1",
        "name": "NACA 0012 Airfoil",
        "mode": "cached_2d",
        "mode_label": "Cached 2D demonstration field",
        "description": "Symmetric streamlined wing section at angle of attack.",
        "explanation": (
            "Flow remains attached to the smooth surface for most of the chord. Curvature differences "
            "create asymmetric pressure (higher velocity on top = lower pressure = upward lift) with a "
            "minimal wake profile."
        ),
        "drag_coefficient_estimate": 0.06,
        "lift_coefficient_estimate": 0.45,
        "wake_score": 0.12,
        "grid": {"nx": 128, "ny": 64, "nz": 1},
    },
]

@router.get("/demo-cases")
async def get_demo_cases():
    """Return list of available precomputed cases."""
    return DEMO_CASES

@router.get("/flow-field/{case_id}")
async def get_flow_field_metadata(case_id: str):
    """Return metadata for a specific case, including URLs to retrieve matrices."""
    # Find matching case
    case = next((c for c in DEMO_CASES if c["case_id"] == case_id), None)
    if not case:
        raise HTTPException(status_code=404, detail="Case profile not found")
        
    return {
        "case_id": case_id,
        "name": case["name"],
        "mode": case["mode"],
        "mode_label": case["mode_label"],
        "grid": case["grid"],
        "metrics": {
            "drag_coefficient_estimate": case["drag_coefficient_estimate"],
            "lift_coefficient_estimate": case["lift_coefficient_estimate"],
            "wake_score": case["wake_score"],
            "confidence_label": "educational estimate",
        },
        "velocity_url": f"/api/flow-field/{case_id}/velocity",
        "pressure_url": f"/api/flow-field/{case_id}/pressure",
        "mask_url": f"/api/flow-field/{case_id}/mask",
    }

@router.get("/flow-field/{case_id}/velocity")
async def get_velocity_binary(case_id: str):
    """Return raw binary .npy file for velocity."""
    file_path = os.path.join(FLOW_ASSETS_DIR, f"{case_id}_velocity.npy")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Velocity array not precalculated")
    return FileResponse(file_path, media_type="application/octet-stream", filename=f"{case_id}_velocity.npy")

@router.get("/flow-field/{case_id}/pressure")
async def get_pressure_binary(case_id: str):
    """Return raw binary .npy file for pressure."""
    file_path = os.path.join(FLOW_ASSETS_DIR, f"{case_id}_pressure.npy")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Pressure array not precalculated")
    return FileResponse(file_path, media_type="application/octet-stream", filename=f"{case_id}_pressure.npy")

@router.get("/flow-field/{case_id}/mask")
async def get_mask_binary(case_id: str):
    """Return raw binary .npy file for obstacle mask."""
    file_path = os.path.join(FLOW_ASSETS_DIR, f"{case_id}_mask.npy")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Obstacle mask array not precalculated")
    return FileResponse(file_path, media_type="application/octet-stream", filename=f"{case_id}_mask.npy")
