"""Depth-conditioned single-image mesh reconstruction utilities."""

from __future__ import annotations

from typing import Any
from collections import Counter

import cv2
import numpy as np


def extract_foreground_mask(image: np.ndarray) -> np.ndarray:
    """Extract a conservative object mask for depth-to-mesh conversion.

    Alpha is preferred. For opaque photos, this is a classical border-distance
    segmentation step; the 3D shape itself still comes from the depth model.
    """
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 image, got {image.shape}")
    if image.shape[2] == 4 and np.any(image[:, :, 3] < 255):
        mask = image[:, :, 3] > 16
    else:
        bgr = image[:, :, :3]
        border = np.concatenate(
            [bgr[0, :, :], bgr[-1, :, :], bgr[:, 0, :], bgr[:, -1, :]], axis=0
        ).astype(np.float32)
        reference = np.median(border, axis=0)
        distance = np.linalg.norm(bgr.astype(np.float32) - reference, axis=2)
        distance = cv2.GaussianBlur(distance, (5, 5), 0)
        distance_u8 = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask_u8 = cv2.threshold(distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask_u8 > 0

    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Foreground segmentation found no object")
    largest = max(contours, key=cv2.contourArea)
    cleaned = np.zeros_like(mask_u8)
    cv2.drawContours(cleaned, [largest], -1, 255, thickness=-1)
    if int(np.sum(cleaned > 0)) < 100:
        raise ValueError("Foreground segmentation produced an unusably small object")
    return cleaned > 0


def depth_map_to_mesh(depth: np.ndarray, mask: np.ndarray, max_size: int = 192) -> Any:
    """Turn a relative depth map and mask into a watertight-ish thick surface mesh."""
    try:
        import trimesh  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Depth mesh export requires trimesh") from exc
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError(f"depth/mask shape mismatch: {depth.shape} vs {mask.shape}")
    if not np.any(mask):
        raise ValueError("Depth mesh mask is empty")

    h, w = depth.shape
    scale = min(1.0, max_size / max(h, w))
    out_w = max(16, int(round(w * scale)))
    out_h = max(16, int(round(h * scale)))
    depth_small = cv2.resize(depth.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    finite = np.isfinite(depth_small) & mask_small
    if not np.any(finite):
        raise ValueError("Depth map contains no finite foreground values")
    lo, hi = np.percentile(depth_small[finite], [2, 98])
    normalized = np.clip((depth_small - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    # Camera-facing relative depth is converted to a compact positive Z range.
    z_front = 0.08 + normalized * 0.42
    z_back = np.full_like(z_front, 0.03)

    ys, xs = np.indices((out_h, out_w))
    x_world = (xs / max(out_w - 1, 1) - 0.5) * 2.0
    y_world = (0.5 - ys / max(out_h - 1, 1)) * 2.0
    front = np.column_stack([x_world.ravel(), y_world.ravel(), z_front.ravel()])
    back = np.column_stack([x_world.ravel(), y_world.ravel(), z_back.ravel()])
    valid = mask_small.ravel()
    vertices = np.concatenate([front[valid], back[valid]], axis=0)
    index = -np.ones((out_h, out_w), dtype=np.int64)
    index[mask_small] = np.arange(int(np.sum(mask_small)))
    back_index = index + int(np.sum(mask_small))
    faces: list[list[int]] = []
    for y in range(out_h - 1):
        for x in range(out_w - 1):
            if not np.all(mask_small[y : y + 2, x : x + 2]):
                continue
            a, b = int(index[y, x]), int(index[y, x + 1])
            c, d = int(index[y + 1, x]), int(index[y + 1, x + 1])
            ba, bb = int(back_index[y, x]), int(back_index[y, x + 1])
            bc, bd = int(back_index[y + 1, x]), int(back_index[y + 1, x + 1])
            faces.extend([[a, c, b], [b, c, d], [bb, ba, bc], [bb, bc, bd]])
    if not faces:
        raise ValueError("Depth mesh contains no complete foreground cells")
    front_faces = np.asarray(faces, dtype=np.int64)
    edge_counts: Counter[tuple[int, int]] = Counter()
    for triangle in front_faces:
        if not np.all(triangle < int(np.sum(mask_small))):
            continue
        for first, second in zip(triangle, np.roll(triangle, -1)):
            edge = tuple(sorted((int(first), int(second))))
            edge_counts[edge] += 1
    # Close the relief along every exposed front edge. The back surface has the
    # same XY topology, so each boundary edge becomes a two-triangle wall.
    for (first, second), count in edge_counts.items():
        if count != 1:
            continue
        back_first = first + int(np.sum(mask_small))
        back_second = second + int(np.sum(mask_small))
        faces.extend(
            [
                [first, second, back_second],
                [first, back_second, back_first],
            ]
        )
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def validate_mesh(mesh: Any) -> dict[str, Any]:
    """Return objective mesh validity/statistics without declaring visual quality."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    finite_vertices = bool(np.all(np.isfinite(vertices)))
    degenerate = int(np.sum(np.asarray(mesh.area_faces) <= 1e-12)) if len(faces) else 0
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "finite_vertices": finite_vertices,
        "degenerate_faces": degenerate,
        "watertight": bool(mesh.is_watertight),
        "valid": bool(len(vertices) > 0 and len(faces) > 0 and finite_vertices and degenerate == 0),
        "bounds": np.asarray(mesh.bounds).round(6).tolist() if finite_vertices else None,
    }


def validate_mesh_file(mesh_path: str) -> dict[str, Any]:
    """Load an exported mesh artifact and validate the serialized geometry."""
    try:
        import trimesh  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Mesh validation requires trimesh") from exc
    if not mesh_path:
        raise ValueError("mesh_path is empty")
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    stats = validate_mesh(loaded)
    stats["path"] = mesh_path
    return stats
