"""API: educational ML surrogate inference (mask → LBM force-proxy prediction)."""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.infer import EDUCATIONAL_DISCLAIMER, load_predictor, resolve_model_path

router = APIRouter(prefix="/api/surrogate", tags=["surrogate"])


class SurrogatePredictRequest(BaseModel):
    """Binary obstacle mask as nested lists (ny rows of nx values) or flat + dims."""

    mask: Optional[List[List[Union[int, float, bool]]]] = Field(
        default=None,
        description="2D mask, True/1 = solid",
    )
    flat_mask: Optional[List[Union[int, float, bool]]] = Field(
        default=None,
        description="Flattened mask in row-major order (requires nx, ny)",
    )
    nx: Optional[int] = None
    ny: Optional[int] = None
    model_name: Optional[str] = Field(
        default=None,
        description=(
            "mean | linear_geom | mlp_geom | ridge_mask | mlp_mask | hgb_mask | rf_mask "
            "(default: bundle primary_model, lowest held-out MAE)"
        ),
    )


class SurrogatePredictResponse(BaseModel):
    cd_force_proxy_pred: float
    model_name: str
    target: str
    educational_disclaimer: str
    grid: dict


def _mask_from_request(req: SurrogatePredictRequest) -> np.ndarray:
    if req.mask is not None:
        arr = np.asarray(req.mask, dtype=np.float32)
        if arr.ndim != 2:
            raise HTTPException(status_code=400, detail="mask must be a 2D nested list")
        return arr

    if req.flat_mask is not None:
        if req.nx is None or req.ny is None:
            raise HTTPException(
                status_code=400,
                detail="flat_mask requires nx and ny",
            )
        flat = np.asarray(req.flat_mask, dtype=np.float32).ravel()
        expected = int(req.nx) * int(req.ny)
        if flat.size != expected:
            raise HTTPException(
                status_code=400,
                detail=f"flat_mask length {flat.size} != nx*ny={expected}",
            )
        return flat.reshape(int(req.ny), int(req.nx))

    raise HTTPException(
        status_code=400,
        detail="Provide mask (2D list) or flat_mask with nx and ny",
    )


@router.get("/status")
async def surrogate_status() -> dict[str, Any]:
    path = resolve_model_path()
    return {
        "model_available": path is not None,
        "model_path": path,
        "educational_disclaimer": EDUCATIONAL_DISCLAIMER,
        "hint": None
        if path
        else "Run: python scripts/generate_ml_dataset.py && python scripts/train_surrogate.py",
    }


@router.post("/predict", response_model=SurrogatePredictResponse)
async def predict_surrogate(req: SurrogatePredictRequest) -> SurrogatePredictResponse:
    """Predict educational cd_force_proxy from a 2D solid mask (no live LBM)."""
    path = resolve_model_path()
    if path is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Surrogate model not trained. From backend/: "
                "python scripts/generate_ml_dataset.py --n 100 && "
                "python scripts/train_surrogate.py"
            ),
        )

    try:
        predictor = load_predictor(path)
        mask = _mask_from_request(req)
        # Binarize soft values
        mask_bin = mask > 0.5
        if not np.any(mask_bin):
            raise HTTPException(status_code=400, detail="Mask has no solid cells")
        out = predictor.predict_mask(mask_bin, model_name=req.model_name)
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Surrogate inference failed: {e}") from e

    return SurrogatePredictResponse(
        cd_force_proxy_pred=float(out["cd_force_proxy_pred"]),
        model_name=str(out["model_name"]),
        target=str(out["target"]),
        educational_disclaimer=str(out["educational_disclaimer"]),
        grid=out["grid"],
    )
