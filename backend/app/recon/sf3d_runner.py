"""Stable Fast 3D inference adapter, kept optional for the CPU application."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import nullcontext
from typing import Any

from app.recon.export_to_aerovoxel import mesh_to_center_slice_mask


MODEL_ID = "stabilityai/stable-fast-3d"


def _add_official_checkout_to_path() -> None:
    """Make an official source checkout importable without bundling it in AeroVoxel."""
    configured = os.environ.get("AEROVOXEL_SF3D_REPO")
    candidates = [
        configured,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "stable-fast-3d")),
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(os.path.join(candidate, "sf3d")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return


def run_sf3d(
    image_path: str,
    output_dir: str,
    model_id: str = MODEL_ID,
    device: str | None = None,
    foreground_ratio: float = 0.85,
    texture_resolution: int = 512,
    target_vertex_count: int = -1,
) -> dict[str, Any]:
    """Run the real SF3D model and export ``mesh.glb`` plus AeroVoxel mask.

    Dependencies are intentionally imported here so normal CPU app startup does
    not require PyTorch, SF3D, rembg, or a multi-GB model download.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    if not 0.0 < foreground_ratio <= 1.0:
        raise ValueError("foreground_ratio must be in (0, 1]")
    if texture_resolution < 256:
        raise ValueError("texture_resolution must be at least 256")
    _add_official_checkout_to_path()
    try:
        import torch
        from PIL import Image
        from sf3d.system import SF3D
        from sf3d.utils import resize_foreground
    except ImportError as exc:
        raise RuntimeError(
            "SF3D runtime dependencies are incomplete. Follow backend/app/recon/README.md; "
            f"the first missing import was: {exc}"
        ) from exc

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if selected_device not in {"cpu", "cuda"} and not selected_device.startswith("cuda:"):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")

    os.makedirs(output_dir, exist_ok=True)
    started = time.perf_counter()
    image = Image.open(image_path).convert("RGBA")
    # SF3D's official runner uses rembg here. An existing alpha channel is
    # respected; opaque images remain valid inputs but get a clear warning.
    if image.getchannel("A").getextrema() == (255, 255):
        print("[recon] Input has no alpha channel; install rembg for background removal.")
    image = resize_foreground(image, foreground_ratio)

    model = SF3D.from_pretrained(
        model_id,
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.to(selected_device)
    model.eval()
    if selected_device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast = nullcontext()
    with torch.inference_mode(), autocast:
        mesh, _ = model.run_image(
            [image],
            bake_resolution=texture_resolution,
            remesh="none",
            vertex_count=target_vertex_count,
        )

    mesh_path = os.path.join(output_dir, "mesh.glb")
    mesh.export(mesh_path, include_normals=True)
    mask_info = mesh_to_center_slice_mask(mesh_path, os.path.join(output_dir, "mask.npy"))
    result: dict[str, Any] = {
        "model": model_id,
        "device": selected_device,
        "cuda_available": bool(torch.cuda.is_available()),
        "image_path": os.path.abspath(image_path),
        "mesh_path": os.path.abspath(mesh_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_vram_mb": (
            round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            if selected_device.startswith("cuda")
            else None
        ),
        "mask": mask_info,
        "honesty": "Best-effort prototype geometry; downstream flow is educational LBM, not certified CFD.",
    }
    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
