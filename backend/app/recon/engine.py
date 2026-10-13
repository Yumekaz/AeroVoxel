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


@dataclass
class AutoReconstructionEngine:
    candidates: list[ReconstructionEngine]
    name: str = "auto_chain"
    selection_reason: str = "Device-aware engine chain; failed attempts are preserved"

    def reconstruct(self, image_path: str, output_dir: str, device: str | None = None) -> dict[str, Any]:
        import json
        import os
        attempts = []
        os.makedirs(output_dir, exist_ok=True)
        for engine in self.candidates:
            # Preserve failed partial outputs rather than overwriting them on fallback.
            attempt_dir = os.path.join(output_dir, "attempt_" + engine.name)
            try:
                result = engine.reconstruct(image_path, attempt_dir, device=device)
            except Exception as exc:
                attempts.append({"engine": engine.name, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "output_dir": attempt_dir})
                with open(os.path.join(output_dir, "attempts.json"), "w", encoding="utf-8") as handle:
                    json.dump(attempts, handle, indent=2)
                continue
            attempts.append({"engine": engine.name, "status": "SUCCEEDED", "output_dir": attempt_dir})
            result["engine"] = result.get("engine", engine.name)
            result["attempts"] = attempts
            result["fallback"] = engine.name != "stable-fast-3d"
            with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            return result
        raise RuntimeError("All reconstruction engines failed; see attempts.json: " + str(attempts))


def select_reconstruction_engine(preferred: str = "auto", device: str | None = None) -> ReconstructionEngine:
    """Auto avoids heavy CPU SF3D and conservatively gates its GPU memory."""
    if preferred not in {"auto", "sf3d", "depth_anything", "opencv"}:
        raise ValueError("preferred must be auto, sf3d, depth_anything, or opencv")
    if device is not None and device not in {"cpu", "cuda"} and not device.startswith("cuda:"):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    if preferred == "opencv":
        if device and device.startswith("cuda"):
            raise RuntimeError("OpenCV 2D fallback only supports CPU")
        return OpenCV2DFallbackEngine()
    capability = get_recon_capability()
    requested = device or capability.get("recommended_device", "cpu")
    cuda = requested.startswith("cuda")
    if cuda and not capability.get("cuda_available"):
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU execution")
    try:
        index = int(requested.split(":")[1]) if ":" in requested else 0
    except ValueError as exc:
        raise ValueError("Invalid CUDA device index") from exc
    memory = capability.get("cuda_memory_mb", [])
    if cuda and (index < 0 or index >= capability.get("cuda_device_count", len(memory))):
        raise RuntimeError("Requested CUDA device does not exist")
    sufficient_vram = cuda and index < len(memory) and memory[index] >= capability.get("sf3d_min_vram_mb", 6144)
    runtime = bool(capability.get("ready_for_inference"))
    sf3d_runnable = runtime and sufficient_vram
    if preferred == "sf3d":
        if runtime and (requested == "cpu" or sufficient_vram):
            return StableFast3DEngine()
        raise RuntimeError("SF3D was explicitly requested but runtime or VRAM is insufficient: " + str(capability.get("sf3d_import_error") or "AeroVoxel conservatively selects SF3D only with at least 6144 MiB VRAM"))
    if preferred == "depth_anything":
        if capability.get("depth_anything_available"):
            return DepthAnythingEngine()
        raise RuntimeError("Depth Anything was explicitly requested but its optional runtime is unavailable")
    candidates: list[ReconstructionEngine] = []
    if sf3d_runnable:
        candidates.append(StableFast3DEngine())
    if capability.get("depth_anything_available"):
        candidates.append(DepthAnythingEngine(selection_reason="SF3D unavailable for requested device/VRAM; Depth Anything runtime is installed"))
    # Respect an explicit CUDA request; classical CPU fallback is not a CUDA run.
    if not cuda:
        candidates.append(OpenCV2DFallbackEngine())
    if not candidates:
        raise RuntimeError("No reconstruction engine can execute on the requested device")
    if len(candidates) == 1:
        return candidates[0]
    return AutoReconstructionEngine(candidates=candidates)
