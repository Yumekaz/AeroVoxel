"""Mask → model feature vectors for the educational aero surrogate."""

from __future__ import annotations

import numpy as np

# Compact mask encoding for mask+geom models (sample-efficient on small LBM sets)
MASK_DS_H = 8
MASK_DS_W = 16
PROFILE_X_BINS = 16
PROFILE_Y_BINS = 8
# Coarser multi-scale profiles (shape-sensitive, low-dimensional)
PROFILE_X_COARSE = 8
PROFILE_Y_COARSE = 4

GEOM_FEATURE_NAMES = (
    "solid_fraction",
    "centroid_x_norm",
    "centroid_y_norm",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_aspect",
    "moment_xx",
    "moment_yy",
    "moment_xy",
    "char_length_norm",
    "perimeter_norm",
    "solidity",
    "skew_x",
    "skew_y",
)


def downsample_mask(mask: np.ndarray, out_h: int = MASK_DS_H, out_w: int = MASK_DS_W) -> np.ndarray:
    """Block-average binary mask to (out_h, out_w) float32 in [0, 1]."""
    m = mask.astype(np.float32)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {m.shape}")
    ny, nx = m.shape
    ys = (np.linspace(0, ny, out_h + 1)).astype(int)
    xs = (np.linspace(0, nx, out_w + 1)).astype(int)
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            block = m[ys[i] : max(ys[i] + 1, ys[i + 1]), xs[j] : max(xs[j] + 1, xs[j + 1])]
            out[i, j] = float(np.mean(block)) if block.size else 0.0
    return out


def solid_profiles(
    mask: np.ndarray,
    n_x: int = PROFILE_X_BINS,
    n_y: int = PROFILE_Y_BINS,
) -> np.ndarray:
    """Column / row solid-fraction profiles (shape-sensitive, low-dimensional)."""
    m = mask.astype(np.float32)
    ny, nx = m.shape
    col = m.mean(axis=0)  # (nx,)
    row = m.mean(axis=1)  # (ny,)
    xs = (np.linspace(0, nx, n_x + 1)).astype(int)
    ys = (np.linspace(0, ny, n_y + 1)).astype(int)
    px = np.array(
        [float(np.mean(col[xs[j] : max(xs[j] + 1, xs[j + 1])])) for j in range(n_x)],
        dtype=np.float32,
    )
    py = np.array(
        [float(np.mean(row[ys[i] : max(ys[i] + 1, ys[i + 1])])) for i in range(n_y)],
        dtype=np.float32,
    )
    return np.concatenate([px, py])


def multi_scale_profiles(mask: np.ndarray) -> np.ndarray:
    """Fine + coarse solid-fraction profiles along streamwise and spanwise axes."""
    fine = solid_profiles(mask, n_x=PROFILE_X_BINS, n_y=PROFILE_Y_BINS)
    coarse = solid_profiles(mask, n_x=PROFILE_X_COARSE, n_y=PROFILE_Y_COARSE)
    return np.concatenate([fine, coarse]).astype(np.float32)


def perimeter_estimate(solid: np.ndarray) -> float:
    """Approximate solid–fluid interface length in lattice units (4-neighbour)."""
    s = solid.astype(bool)
    if not np.any(s):
        return 0.0
    # Count solid cells with at least one fluid 4-neighbour (pad fluid outside)
    padded = np.pad(s, 1, mode="constant", constant_values=False)
    core = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    edge = core & (~up | ~down | ~left | ~right)
    return float(np.sum(edge))


def geometric_features(mask: np.ndarray) -> np.ndarray:
    """Compact geometric descriptors of a binary solid mask (ny, nx)."""
    solid = mask.astype(bool)
    ny, nx = solid.shape
    n = float(nx * ny)
    n_solid = int(np.sum(solid))
    solid_frac = n_solid / n if n > 0 else 0.0

    if n_solid == 0:
        return np.zeros(len(GEOM_FEATURE_NAMES), dtype=np.float32)

    ys, xs = np.where(solid)
    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    bw = float(x_max - x_min + 1)
    bh = float(y_max - y_min + 1)
    aspect = bw / max(bh, 1.0)

    # Second moments about centroid, normalized by domain
    dy = (ys.astype(np.float64) - cy) / max(ny, 1)
    dx = (xs.astype(np.float64) - cx) / max(nx, 1)
    mxx = float(np.mean(dx * dx))
    myy = float(np.mean(dy * dy))
    mxy = float(np.mean(dx * dy))
    char_l = bh / max(ny, 1)  # frontal height / domain height

    perim = perimeter_estimate(solid)
    # Normalize perimeter by domain diagonal so scale is O(1)
    perim_norm = perim / max(float(np.hypot(nx, ny)), 1.0)
    solidity = float(n_solid) / max(bw * bh, 1.0)

    # Third-moment skewness proxies (shape asymmetry)
    std_x = float(np.sqrt(max(mxx, 1e-12)))
    std_y = float(np.sqrt(max(myy, 1e-12)))
    skew_x = float(np.mean(dx ** 3) / (std_x ** 3 + 1e-12))
    skew_y = float(np.mean(dy ** 3) / (std_y ** 3 + 1e-12))

    return np.array(
        [
            solid_frac,
            cx / max(nx, 1),
            cy / max(ny, 1),
            bw / max(nx, 1),
            bh / max(ny, 1),
            aspect,
            mxx,
            myy,
            mxy,
            char_l,
            perim_norm,
            solidity,
            skew_x,
            skew_y,
        ],
        dtype=np.float32,
    )


def mask_feature_vector(
    mask: np.ndarray,
    include_mask: bool = True,
    ds_h: int = MASK_DS_H,
    ds_w: int = MASK_DS_W,
) -> np.ndarray:
    """Feature vector: geometric (+ multi-scale profiles + coarse mask if include_mask).

    Linear / geom-only baselines use geometric features only (include_mask=False).
    """
    geom = geometric_features(mask)
    if not include_mask:
        return geom
    profiles = multi_scale_profiles(mask)
    # Coarse occupancy grid (keeps spatial structure without 512-D blow-up)
    ds = downsample_mask(mask, out_h=ds_h, out_w=ds_w).ravel()
    return np.concatenate([geom, profiles, ds]).astype(np.float32)


def batch_features(
    masks: np.ndarray,
    include_mask: bool = True,
    ds_h: int = MASK_DS_H,
    ds_w: int = MASK_DS_W,
) -> np.ndarray:
    """Stack feature vectors for masks shaped (N, H, W)."""
    if masks.ndim != 3:
        raise ValueError(f"Expected (N,H,W) masks, got {masks.shape}")
    rows = [
        mask_feature_vector(masks[i], include_mask=include_mask, ds_h=ds_h, ds_w=ds_w)
        for i in range(masks.shape[0])
    ]
    return np.stack(rows, axis=0)


def char_length_from_mask(mask: np.ndarray) -> float:
    """Characteristic length in lattice units for educational Cd/Cl proxies.

    Uses frontal (streamwise-normal) solid height, floored so extremely thin
    silhouettes do not explode Cd = 2 Fx / (U² L) on coarse grids.
    """
    solid = mask.astype(bool)
    ys, xs = np.where(solid)
    if ys.size == 0:
        return 4.0
    height = float(ys.max() - ys.min() + 1)
    width = float(xs.max() - xs.min() + 1)
    # Prefer frontal height; blend a little width so knife-edge ellipses stay finite
    area_scale = float(np.sqrt(max(ys.size, 1)))
    raw = max(height, 0.35 * width, 0.75 * area_scale)
    return float(max(raw, 4.0))
