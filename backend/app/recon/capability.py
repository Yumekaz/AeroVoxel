"""Runtime capability checks for the optional neural reconstruction path."""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any


def _module_available(name: str) -> bool:
    if name == "sf3d":
        configured = os.environ.get("AEROVOXEL_SF3D_REPO")
        candidates = [
            configured,
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "stable-fast-3d")),
        ]
        for candidate in candidates:
            if candidate and os.path.isdir(os.path.join(candidate, "sf3d")) and candidate not in sys.path:
                sys.path.insert(0, candidate)
    return importlib.util.find_spec(name) is not None


def _sf3d_importable() -> tuple[bool, str | None]:
    if not _module_available("sf3d"):
        return False, "official SF3D source package not found"
    try:
        import importlib

        importlib.import_module("sf3d.system")
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def get_recon_capability() -> dict[str, Any]:
    """Return truthful local capability information without loading model weights."""
    torch_available = _module_available("torch")
    cuda_available = False
    cuda_device_count = 0
    torch_version = None
    cuda_version = None
    if torch_available:
        try:
            import torch  # type: ignore

            torch_version = torch.__version__
            cuda_available = bool(torch.cuda.is_available())
            cuda_device_count = int(torch.cuda.device_count())
            cuda_version = torch.version.cuda
        except Exception as exc:  # capability reporting must not break the API
            return {
                "torch_available": True,
                "torch_error": str(exc),
                "cuda_available": False,
                "cuda_device_count": 0,
                "sf3d_available": _module_available("sf3d"),
                "trimesh_available": _module_available("trimesh"),
                "rembg_available": _module_available("rembg"),
                "ready_for_inference": False,
            }

    sf3d_available = _module_available("sf3d")
    sf3d_importable, sf3d_import_error = _sf3d_importable()
    trimesh_available = _module_available("trimesh")
    rembg_available = _module_available("rembg")
    return {
        "torch_available": torch_available,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_version": cuda_version,
        "sf3d_available": sf3d_available,
        "sf3d_importable": sf3d_importable,
        "sf3d_import_error": sf3d_import_error,
        "trimesh_available": trimesh_available,
        "rembg_available": rembg_available,
        "ready_for_inference": bool(torch_available and sf3d_importable and trimesh_available),
        "recommended_device": "cuda" if cuda_available else "cpu",
    }
