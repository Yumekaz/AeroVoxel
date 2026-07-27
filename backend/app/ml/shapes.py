"""Synthetic binary obstacle masks for ML dataset generation.

Masks are (ny, nx) bool arrays with True = solid. Shapes leave inlet
clearance and stay clear of domain edges so the LBM inlet/outlet remain fluid.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ShapeFn = Callable[[int, int, np.random.Generator], np.ndarray]

# Keep solids away from x=0 inlet and domain walls
INLET_CLEAR_FRAC = 0.18
OUTLET_CLEAR_FRAC = 0.12
WALL_PAD = 2


def _domain_bounds(nx: int, ny: int) -> tuple[int, int, int, int]:
    """Return x0, x1, y0, y1 (exclusive x1/y1) for allowable solid placement."""
    x0 = max(WALL_PAD + 1, int(nx * INLET_CLEAR_FRAC))
    x1 = min(nx - WALL_PAD, int(nx * (1.0 - OUTLET_CLEAR_FRAC)))
    y0 = WALL_PAD
    y1 = ny - WALL_PAD
    if x1 - x0 < 6 or y1 - y0 < 4:
        raise ValueError(f"Grid {nx}x{ny} too small for solid placement")
    return x0, x1, y0, y1


def _validate_mask(mask: np.ndarray, nx: int, ny: int) -> bool:
    """Reject empty, full-domain, or inlet-blocking solids."""
    solid = mask.astype(bool)
    n_solid = int(np.sum(solid))
    if n_solid < 4:
        return False
    frac = n_solid / float(nx * ny)
    if frac > 0.45 or frac < 0.005:
        return False
    # Inlet column must stay mostly fluid
    if float(np.mean(solid[:, 0:2])) > 0.05:
        return False
    # Must not seal vertical cross-section (block all flow)
    for x in range(nx):
        if bool(np.all(solid[:, x])):
            return False
    return True


def make_circle(nx: int, ny: int, rng: np.random.Generator) -> np.ndarray:
    x0, x1, y0, y1 = _domain_bounds(nx, ny)
    mask = np.zeros((ny, nx), dtype=bool)
    r = float(rng.uniform(0.12, 0.28) * min(nx, ny))
    r = max(2.5, min(r, (y1 - y0) * 0.4, (x1 - x0) * 0.35))
    cx_lo, cx_hi = x0 + r, x1 - r
    cy_lo, cy_hi = y0 + r, y1 - r
    if cx_hi <= cx_lo or cy_hi <= cy_lo:
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    else:
        cx = float(rng.uniform(cx_lo, cx_hi))
        cy = float(rng.uniform(cy_lo, cy_hi))
    yy, xx = np.ogrid[:ny, :nx]
    mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = True
    return mask


def make_ellipse(nx: int, ny: int, rng: np.random.Generator) -> np.ndarray:
    x0, x1, y0, y1 = _domain_bounds(nx, ny)
    mask = np.zeros((ny, nx), dtype=bool)
    rx = float(rng.uniform(0.10, 0.30) * nx)
    ry = float(rng.uniform(0.12, 0.35) * ny)
    rx = max(3.0, min(rx, (x1 - x0) * 0.4))
    ry = max(2.5, min(ry, (y1 - y0) * 0.4))
    # Avoid knife-edge silhouettes that inflate educational Cd proxies
    if ry < 0.35 * rx:
        ry = 0.35 * rx
    pad = max(rx, ry) + 1.0
    cx_lo, cx_hi = x0 + pad * 0.5, x1 - pad * 0.5
    cy_lo, cy_hi = y0 + pad * 0.5, y1 - pad * 0.5
    if cx_hi <= cx_lo or cy_hi <= cy_lo:
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
    else:
        cx = float(rng.uniform(cx_lo, cx_hi))
        cy = float(rng.uniform(cy_lo, cy_hi))
    angle = float(rng.uniform(0.0, np.pi))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    yy, xx = np.mgrid[:ny, :nx]
    dx, dy = xx - cx, yy - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a
    mask[(xr / max(rx, 1e-6)) ** 2 + (yr / max(ry, 1e-6)) ** 2 <= 1.0] = True
    return mask


def make_rectangle(nx: int, ny: int, rng: np.random.Generator) -> np.ndarray:
    x0, x1, y0, y1 = _domain_bounds(nx, ny)
    mask = np.zeros((ny, nx), dtype=bool)
    w = int(rng.integers(max(4, nx // 12), max(5, nx // 4)))
    h = int(rng.integers(max(3, ny // 8), max(4, ny // 2)))
    w = min(w, x1 - x0 - 2)
    h = min(h, y1 - y0 - 2)
    sx = int(rng.integers(x0, max(x0 + 1, x1 - w)))
    sy = int(rng.integers(y0, max(y0 + 1, y1 - h)))
    mask[sy : sy + h, sx : sx + w] = True
    # Optional 45° diamond-ish crop via rotation of a square subset is skipped;
    # pure axis-aligned rect is fine for diversity.
    return mask


def make_airfoil_ish(nx: int, ny: int, rng: np.random.Generator) -> np.ndarray:
    """Simple cambered teardrop / NACA-like silhouette (educational, not a real airfoil)."""
    x0, x1, y0, y1 = _domain_bounds(nx, ny)
    mask = np.zeros((ny, nx), dtype=bool)
    chord = float(rng.uniform(0.22, 0.42) * nx)
    chord = max(8.0, min(chord, max(8.0, x1 - x0 - 2)))
    max_thick = max(3.0, (y1 - y0) * 0.45)
    thick = float(rng.uniform(0.15, 0.40) * ny)
    thick = max(3.0, min(thick, max_thick))
    aoa = float(rng.uniform(-18.0, 18.0)) * np.pi / 180.0
    # Margin for rotation + thickness so center stays in-domain
    margin_y = thick * 0.85 + 1.0
    cy_lo = y0 + margin_y
    cy_hi = y1 - margin_y
    if cy_hi <= cy_lo:
        cy = 0.5 * (y0 + y1)
    else:
        cy = float(rng.uniform(cy_lo, cy_hi))
    cx_lo = x0 + 0.05 * chord
    cx_hi = x1 - 0.85 * chord
    if cx_hi <= cx_lo:
        cx = float(x0 + 0.1 * (x1 - x0))
    else:
        cx = float(rng.uniform(cx_lo, cx_hi))

    # Parametric teardrop in body frame, then rotate by AoA
    n_pts = 64
    t = np.linspace(0.0, 1.0, n_pts)
    # Half-thickness: blunt leading edge, thin trailing edge
    half_t = thick * 0.5 * (np.sqrt(np.clip(t, 1e-6, 1.0)) * (1.0 - 0.85 * t))
    camber = 0.15 * thick * np.sin(np.pi * t) * float(rng.uniform(0.3, 1.0))
    xb = t * chord
    y_upper = camber + half_t
    y_lower = camber - half_t

    cos_a, sin_a = np.cos(aoa), np.sin(aoa)
    yy, xx = np.mgrid[:ny, :nx]
    # Transform grid into body frame relative to leading edge
    dx = xx - cx
    dy = yy - cy
    xb_g = dx * cos_a + dy * sin_a
    yb_g = -dx * sin_a + dy * cos_a

    # Interpolate upper/lower thickness along chord
    inside_x = (xb_g >= 0.0) & (xb_g <= chord)
    # Linear interp of half bounds
    y_u = np.interp(xb_g, xb, y_upper)
    y_l = np.interp(xb_g, xb, y_lower)
    mask[inside_x & (yb_g <= y_u) & (yb_g >= y_l)] = True
    return mask


def make_blob(nx: int, ny: int, rng: np.random.Generator) -> np.ndarray:
    """Random organic blob via summed radial bumps on a base circle."""
    x0, x1, y0, y1 = _domain_bounds(nx, ny)
    mask = np.zeros((ny, nx), dtype=bool)
    r0 = float(rng.uniform(0.10, 0.22) * min(nx, ny))
    r0 = max(3.0, min(r0, (y1 - y0) * 0.35, (x1 - x0) * 0.3))
    pad = r0 * 1.3
    cx_lo, cx_hi = x0 + pad, x1 - pad
    cy_lo, cy_hi = y0 + pad, y1 - pad
    if cx_hi <= cx_lo or cy_hi <= cy_lo:
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    else:
        cx = float(rng.uniform(cx_lo, cx_hi))
        cy = float(rng.uniform(cy_lo, cy_hi))
    n_harm = int(rng.integers(3, 7))
    amps = rng.uniform(0.05, 0.35, size=n_harm)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_harm)
    yy, xx = np.mgrid[:ny, :nx]
    dx, dy = xx - cx, yy - cy
    ang = np.arctan2(dy, dx)
    rad = np.sqrt(dx * dx + dy * dy)
    r_theta = r0 * (
        1.0 + sum(float(a) * np.cos(k * ang + float(ph)) for k, (a, ph) in enumerate(zip(amps, phases), start=1))
    )
    r_theta = np.clip(r_theta, r0 * 0.4, r0 * 1.6)
    mask[rad <= r_theta] = True
    return mask


def make_triangle(nx: int, ny: int, rng: np.random.Generator) -> np.ndarray:
    """Filled triangle (wedge / arrowhead) for bluff-body diversity."""
    x0, x1, y0, y1 = _domain_bounds(nx, ny)
    mask = np.zeros((ny, nx), dtype=bool)
    w = float(rng.uniform(0.12, 0.28) * nx)
    h = float(rng.uniform(0.20, 0.55) * ny)
    w = max(5.0, min(w, max(5.0, x1 - x0 - 2)))
    h = max(4.0, min(h, max(4.0, y1 - y0 - 2)))
    # Point left (into flow) or right randomly
    tip_left = bool(rng.random() < 0.5)
    cx_lo, cx_hi = x0 + w * 0.5, x1 - w * 0.5
    cy_lo, cy_hi = y0 + h * 0.5, y1 - h * 0.5
    if cx_hi <= cx_lo or cy_hi <= cy_lo:
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    else:
        cx = float(rng.uniform(cx_lo, cx_hi))
        cy = float(rng.uniform(cy_lo, cy_hi))
    if tip_left:
        p0 = (cx - w * 0.5, cy)
        p1 = (cx + w * 0.5, cy - h * 0.5)
        p2 = (cx + w * 0.5, cy + h * 0.5)
    else:
        p0 = (cx + w * 0.5, cy)
        p1 = (cx - w * 0.5, cy - h * 0.5)
        p2 = (cx - w * 0.5, cy + h * 0.5)

    yy, xx = np.mgrid[:ny, :nx].astype(np.float64)

    def _sign(px, py, qx, qy, rx, ry):
        return (px - rx) * (qy - ry) - (qx - rx) * (py - ry)

    b0 = _sign(xx, yy, p0[0], p0[1], p1[0], p1[1]) < 0.0
    b1 = _sign(xx, yy, p1[0], p1[1], p2[0], p2[1]) < 0.0
    b2 = _sign(xx, yy, p2[0], p2[1], p0[0], p0[1]) < 0.0
    mask[(b0 == b1) & (b1 == b2)] = True
    return mask


SHAPE_REGISTRY: dict[str, ShapeFn] = {
    "circle": make_circle,
    "ellipse": make_ellipse,
    "rectangle": make_rectangle,
    "airfoil_ish": make_airfoil_ish,
    "blob": make_blob,
    "triangle": make_triangle,
}


def random_mask(
    nx: int,
    ny: int,
    rng: np.random.Generator,
    shape_name: str | None = None,
    max_tries: int = 40,
) -> tuple[np.ndarray, str]:
    """Draw a valid random solid mask and return (mask, shape_name)."""
    names = list(SHAPE_REGISTRY.keys())
    for _ in range(max_tries):
        name = shape_name if shape_name in SHAPE_REGISTRY else str(rng.choice(names))
        mask = SHAPE_REGISTRY[name](nx, ny, rng)
        if _validate_mask(mask, nx, ny):
            return mask.astype(bool), name
    # Fallback: small centered circle
    mask = make_circle(nx, ny, np.random.default_rng(0))
    if not _validate_mask(mask, nx, ny):
        mask = np.zeros((ny, nx), dtype=bool)
        cy, cx = ny // 2, max(nx // 4, 8)
        r = max(3, min(ny // 5, 6))
        yy, xx = np.ogrid[:ny, :nx]
        mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = True
    return mask.astype(bool), "circle_fallback"
