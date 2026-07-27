import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api")

# Define the absolute path to the precomputed flow assets
FLOW_ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "flow")
)

# Full catalog. Cases without on-disk velocity/pressure/mask arrays are hidden
# from /demo-cases so the UI never offers a broken load path (e.g. sphere cache
# missing until `python -m app.services.cache_generator_3d` is run).
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
    {
        "case_id": "cylinder_v1",
        "name": "Circular Cylinder (Validation)",
        "mode": "cached_2d",
        "mode_label": "Educational 2D LBM field",
        "description": "Classic 2D bluff body for educational wake and stagnation visualization.",
        "explanation": (
            "Oncoming flow stagnates on the windward face (high pressure), accelerates around the "
            "shoulders, then separates and forms a recirculating wake leeward of the cylinder. "
            "This is a classic educational bluff-body case — drag values are qualitative estimates, "
            "not certified Cd measurements."
        ),
        "drag_coefficient_estimate": 1.1,
        "lift_coefficient_estimate": 0.0,
        "wake_score": 0.82,
        "grid": {"nx": 128, "ny": 64, "nz": 1},
    },
    {
        "case_id": "sphere_3d_v1",
        "name": "Sphere 3D (Cached LBM)",
        "mode": "real_3d_lbm",
        "mode_label": "Cached 3D LBM center-slice (D3Q19, offline precompute)",
        "description": "Offline D3Q19-generated flow around a sphere — center-slice view.",
        "explanation": (
            "This flow field was precomputed by a D3Q19 Lattice Boltzmann solver (offline cache, not "
            "live 3D). The sphere creates a symmetric stagnation zone at the front face and a "
            "wake region behind it. Drag coefficient for a sphere at moderate Reynolds number is "
            "approximately 0.47 (educational estimate). This 2D view is a center-slice through the "
            "3D domain. Generate missing assets with: python -m app.services.cache_generator_3d"
        ),
        "drag_coefficient_estimate": 0.47,
        "lift_coefficient_estimate": 0.0,
        "wake_score": 0.35,
        "grid": {"nx": 64, "ny": 64, "nz": 64},
    },
]


def _case_assets_ready(case_id: str) -> bool:
    """True only when velocity, pressure, and mask .npy files all exist on disk."""
    required = (
        f"{case_id}_velocity.npy",
        f"{case_id}_pressure.npy",
        f"{case_id}_mask.npy",
    )
    return all(os.path.exists(os.path.join(FLOW_ASSETS_DIR, name)) for name in required)


def _available_demo_cases():
    """Catalog entries that have loadable flow arrays (happy-path safe)."""
    return [c for c in DEMO_CASES if _case_assets_ready(c["case_id"])]


@router.get("/demo-cases")
async def get_demo_cases():
    """Return demo cases that have precomputed flow assets available."""
    return _available_demo_cases()

@router.get("/flow-field/{case_id}")
async def get_flow_field_metadata(case_id: str):
    """Return metadata for a specific case, including URLs to retrieve matrices."""
    case = next((c for c in DEMO_CASES if c["case_id"] == case_id), None)
    if not case:
        raise HTTPException(status_code=404, detail="Case profile not found")
    if not _case_assets_ready(case_id):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Flow assets for '{case_id}' are not available. "
                "For the 3D sphere cache run: python -m app.services.cache_generator_3d"
            ),
        )

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
