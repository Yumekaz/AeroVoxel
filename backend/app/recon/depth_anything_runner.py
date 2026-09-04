"""CPU-capable Depth Anything V2 monocular depth → mesh reconstruction engine."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import cv2
import numpy as np

from app.recon.depth_mesh import depth_map_to_mesh, extract_foreground_mask, validate_mesh, validate_mesh_file
from app.recon.export_to_aerovoxel import mesh_to_center_slice_mask


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def run_depth_anything(
    image_path: str,
    output_dir: str,
    model_id: str = MODEL_ID,
    device: str | None = None,
    max_image_size: int = 768,
    mesh_size: int = 192,
) -> dict[str, Any]:
    """Run real Depth Anything V2 inference and export mesh/mask artifacts."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    if max_image_size < 128 or mesh_size < 32:
        raise ValueError("max_image_size must be >=128 and mesh_size must be >=32")
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise RuntimeError(f"Depth Anything runtime dependency missing: {exc}") from exc

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if selected_device not in {"cpu", "cuda"} and not selected_device.startswith("cuda:"):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    os.makedirs(output_dir, exist_ok=True)
    started = time.perf_counter()

    image = Image.open(image_path).convert("RGBA")
    scale = min(1.0, max_image_size / max(image.size))
    if scale < 1.0:
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    image_np = np.asarray(image)
    mask = extract_foreground_mask(image_np)

    model_load_started = time.perf_counter()
    cache_key = (model_id, selected_device)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is None:
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id).to(selected_device)
            model.eval()
            _MODEL_CACHE[cache_key] = (processor, model)
            cache_hit = False
        else:
            processor, model = cached
            cache_hit = True
    model_load_seconds = time.perf_counter() - model_load_started
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(selected_device) for key, value in inputs.items()}
    inference_started = time.perf_counter()
    with torch.inference_mode():
        predicted = model(**inputs).predicted_depth
    depth = torch.nn.functional.interpolate(
        predicted.unsqueeze(1), size=(image.height, image.width), mode="bicubic", align_corners=False
    ).squeeze().detach().cpu().numpy()
    inference_seconds = time.perf_counter() - inference_started

    mesh = depth_map_to_mesh(depth, mask, max_size=mesh_size)
    mesh_path = os.path.join(output_dir, "mesh.glb")
    mesh.export(mesh_path, include_normals=True)
    mesh_stats = validate_mesh(mesh)
    serialized_mesh_stats = validate_mesh_file(mesh_path)
    if not serialized_mesh_stats["valid"]:
        raise ValueError(f"Serialized reconstruction mesh failed validation: {serialized_mesh_stats}")
    mask_info = mesh_to_center_slice_mask(
        mesh_path,
        os.path.join(output_dir, "mask.npy"),
        nx=128,
        ny=64,
        voxel_resolution_factor=1.0,
    )
    np.save(os.path.join(output_dir, "depth.npy"), depth.astype(np.float32))
    depth_preview = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "depth_preview.png"), depth_preview)
    result: dict[str, Any] = {
        "engine": "depth_anything_v2_small",
        "model": model_id,
        "device": selected_device,
        "image_path": os.path.abspath(image_path),
        "mesh_path": os.path.abspath(mesh_path),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model_load_seconds": round(model_load_seconds, 3),
        "model_cache_hit": cache_hit,
        "inference_seconds": round(inference_seconds, 3),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if selected_device.startswith("cuda") else None,
        "mesh": mesh_stats,
        "serialized_mesh": serialized_mesh_stats,
        "mask": mask_info,
        "preprocessing": "Depth Anything V2 Small + classical foreground mask for object isolation",
        "honesty": "Relative-depth single-view mesh; hidden geometry and metric scale are ambiguous. Flow is educational LBM, not certified CFD.",
    }
    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
