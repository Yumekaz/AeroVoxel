"""Small engine-selection boundary for real reconstruction backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.recon.capability import get_recon_capability
from app.recon.sf3d_runner import run_sf3d
from app.recon.depth_anything_runner import run_depth_anything
from app.recon.opencv_fallback import run_opencv_fallback


class ReconstructionEngine(Protocol):
    """Interface shared by future image-to-3D backends."""

    name: str

    def reconstruct(self, image_path: str, output_dir: str, device: str | None = None) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class StableFast3DEngine:
    """Production adapter for the official Stable Fast 3D source checkout."""

    name: str = "stable-fast-3d"
    selection_reason: str = "SF3D dependencies and mesh export capability are available"

    def reconstruct(self, image_path: str, output_dir: str, device: str | None = None) -> dict[str, Any]:
        return run_sf3d(image_path, output_dir, device=device)


@dataclass(frozen=True)
class DepthAnythingEngine:
    """CPU-capable real ML depth engine used when SF3D is unavailable."""

    name: str = "depth_anything_v2_small"
    selection_reason: str = "SF3D is unavailable or not runnable; Depth Anything V2 Small is installed"

    def reconstruct(self, image_path: str, output_dir: str, device: str | None = None) -> dict[str, Any]:
        return run_depth_anything(image_path, output_dir, device=device)


@dataclass(frozen=True)
class OpenCV2DFallbackEngine:
    """Last-resort classical 2D domain engine; never presented as 3D/ML."""

    name: str = "opencv_2d_fallback"
    selection_reason: str = "No executable ML reconstruction engine is available"

    def reconstruct(self, image_path: str, output_dir: str, device: str | None = None) -> dict[str, Any]:
        return run_opencv_fallback(image_path, output_dir, device=device)


def select_reconstruction_engine(preferred: str = "auto") -> ReconstructionEngine:
    """Select a real engine in deterministic priority order with explicit fallback."""
    if preferred not in {"auto", "sf3d", "depth_anything", "opencv"}:
        raise ValueError("preferred must be auto, sf3d, depth_anything, or opencv")
    capability = get_recon_capability()
    if preferred in {"auto", "sf3d"} and capability.get("ready_for_inference"):
        return StableFast3DEngine()
    if preferred == "sf3d":
        raise RuntimeError("SF3D was explicitly requested but is not runnable: " + str(capability.get("sf3d_import_error")))
    if preferred in {"auto", "depth_anything"} and capability.get("depth_anything_available"):
        sf3d_error = capability.get("sf3d_import_error") or "SF3D runtime unavailable"
        return DepthAnythingEngine(selection_reason=f"SF3D unavailable: {sf3d_error}; Depth Anything V2 Small is installed")
    if preferred == "depth_anything":
        raise RuntimeError("Depth Anything was explicitly requested but transformers/torch is unavailable")
    if preferred == "opencv":
        return OpenCV2DFallbackEngine()
    if capability.get("depth_anything_available"):
        return DepthAnythingEngine()
    return OpenCV2DFallbackEngine()
