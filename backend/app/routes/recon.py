"""Read-only capability endpoint for the optional neural reconstruction path."""

from fastapi import APIRouter

from app.recon.capability import get_recon_capability

router = APIRouter(prefix="/api/recon", tags=["reconstruction"])


@router.get("/capability")
async def recon_capability() -> dict:
    """Report installed/runtime capability without downloading or loading weights."""
    return get_recon_capability()
