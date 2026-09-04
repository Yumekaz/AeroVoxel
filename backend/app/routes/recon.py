"""Read-only capability endpoint for the optional neural reconstruction path."""

from fastapi import APIRouter

from app.recon.capability import get_recon_capability
from app.recon.diagnostics import collect_recon_diagnostics

router = APIRouter(prefix="/api/recon", tags=["reconstruction"])


@router.get("/capability")
async def recon_capability() -> dict:
    """Report installed/runtime capability without downloading or loading weights."""
    return get_recon_capability()


@router.get("/diagnostics")
async def recon_diagnostics() -> dict:
    """Run non-destructive runtime/host checks; never downloads weights or changes state."""
    return collect_recon_diagnostics()
