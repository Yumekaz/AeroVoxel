"""Read-only capability endpoint for the optional neural reconstruction path."""

from fastapi import APIRouter, File, UploadFile, HTTPException

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


@router.post("/import")
async def import_reconstruction_mask(file: UploadFile = File(...)) -> dict:
    """Import a bounded solver-ready .npy mask, never arbitrary mesh/model files."""
    import io
    import os
    import cv2
    import numpy as np
    from app.recon.integration import register_mask_array
    from app.routes.simulate import UPLOADS_DIR

    payload = await file.read(128 * 1024 + 1)
    if len(payload) > 128 * 1024:
        raise HTTPException(status_code=413, detail="Mask file exceeds 128 KiB")
    try:
        stream = io.BytesIO(payload)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Only NumPy v1/v2 arrays are supported")
        if shape != (64, 128) or dtype.hasobject or dtype.kind not in "buif" or dtype.itemsize > 8:
            raise ValueError("Expected a numeric or boolean (64, 128) mask")
        stream.seek(0)
        mask = np.load(stream, allow_pickle=False)
        job = register_mask_array(mask, UPLOADS_DIR)
    except (ValueError, TypeError, EOFError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    preview = cv2.resize(mask.astype(np.uint8) * 255, (512, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(UPLOADS_DIR, f"preview_{job['job_id']}.jpg"), preview)
    return {
        "job_id": job["job_id"], "shape": job["shape"], "solid_cells": job["solid_cells"],
        "preview_url": f"/api/upload/preview/{job['job_id']}",
        "mask_url": f"/api/simulate/result/{job['job_id']}/mask",
        "status": "ready", "note": "Imported reconstruction mask; inspect before running CPU LBM",
    }
