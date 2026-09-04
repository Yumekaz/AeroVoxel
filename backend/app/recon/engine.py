"""Small engine-selection boundary for real reconstruction backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.recon.capability import get_recon_capability
from app.recon.sf3d_runner import run_sf3d


class ReconstructionEngine(Protocol):
    """Interface shared by future image-to-3D backends."""

    name: str

    def reconstruct(self, image_path: str, output_dir: str, device: str | None = None) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class StableFast3DEngine:
    """Production adapter for the official Stable Fast 3D source checkout."""

    name: str = "stable-fast-3d"

    def reconstruct(self, image_path: str, output_dir: str, device: str | None = None) -> dict[str, Any]:
        return run_sf3d(image_path, output_dir, device=device)


def select_reconstruction_engine(preferred: str = "auto") -> ReconstructionEngine:
    """Select only a genuinely runnable engine; never silently downgrade to fake 3D."""
    if preferred not in {"auto", "sf3d"}:
        raise ValueError("preferred must be auto or sf3d")
    capability = get_recon_capability()
    if capability.get("ready_for_inference"):
        return StableFast3DEngine()
    blockers = [
        f"{key}={capability.get(key)}"
        for key in ("cuda_available", "sf3d_importable", "trimesh_available", "sf3d_import_error")
        if capability.get(key) in (False, None) or key == "sf3d_import_error"
    ]
    raise RuntimeError(
        "No reconstruction engine is runnable in this environment. "
        "The existing OpenCV upload path remains the explicit 2D silhouette fallback; "
        "it is not mislabeled as 3D reconstruction. Blockers: " + ", ".join(blockers)
    )
