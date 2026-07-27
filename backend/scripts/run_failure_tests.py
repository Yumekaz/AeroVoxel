"""
AeroVoxel failure / robustness matrix (Milestone M4).

Synthesizes F1–F5 smartphone-like inputs with OpenCV/numpy, posts them through
the same FastAPI upload route used by the demo (TestClient), and records whether
the pipeline completes without an unhandled crash while reporting honest
scale/mask behavior.

Outputs JSON under repo-root evaluation_outputs/ (gitignored).

Usage (from backend/):
    python scripts/run_failure_tests.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import cv2
import numpy as np

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(BACKEND_ROOT, ".."))
OUT_DIR = os.path.join(REPO_ROOT, "evaluation_outputs")
FIXTURE_DIR = os.path.join(OUT_DIR, "failure_fixtures")


# ---------------------------------------------------------------------------
# Fixture generators (lightweight synthetic images — not real photos)
# ---------------------------------------------------------------------------

def _encode_jpeg(bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def make_f1_no_scale_card() -> tuple[bytes, str, dict[str, Any]]:
    """Dark circular object on bright background; no credit-card rectangle."""
    img = np.full((480, 640, 3), 240, dtype=np.uint8)
    cv2.circle(img, (320, 240), 90, (30, 30, 30), thickness=-1)
    meta = {
        "description": "Circular object, no calibration card",
        "has_scale_card": False,
        "object": "circle",
    }
    return _encode_jpeg(img), "f1_no_scale.jpg", meta


def make_f2_poor_lighting() -> tuple[bytes, str, dict[str, Any]]:
    """Low-contrast dark-gray object on slightly lighter gray (poor lighting)."""
    img = np.full((480, 640, 3), 55, dtype=np.uint8)
    # Object only ~15 gray levels above background → weak Canny edges
    cv2.ellipse(img, (320, 250), (110, 50), 0, 0, 360, (70, 70, 70), thickness=-1)
    # Mild noise to mimic phone ISO grain
    noise = np.random.default_rng(42).integers(-8, 9, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    meta = {
        "description": "Low-contrast car-like blob on dark field",
        "has_scale_card": False,
        "contrast": "very_low",
    }
    return _encode_jpeg(img, quality=70), "f2_poor_lighting.jpg", meta


def make_f3_motion_blur() -> tuple[bytes, str, dict[str, Any]]:
    """Sharp object then strong horizontal motion blur (blurry phone shot)."""
    img = np.full((480, 640, 3), 230, dtype=np.uint8)
    cv2.rectangle(img, (220, 180), (420, 300), (25, 25, 40), thickness=-1)
    cv2.circle(img, (250, 300), 28, (20, 20, 20), thickness=-1)
    cv2.circle(img, (390, 300), 28, (20, 20, 20), thickness=-1)
    k = np.zeros((1, 31), dtype=np.float32)
    k[0, :] = 1.0 / 31.0
    blurred = cv2.filter2D(img, -1, k)
    # Extra soft blur
    blurred = cv2.GaussianBlur(blurred, (9, 9), 0)
    meta = {
        "description": "Car-like silhouette with horizontal motion blur",
        "has_scale_card": False,
        "blur": "motion_horizontal_31px",
    }
    return _encode_jpeg(blurred, quality=60), "f3_motion_blur.jpg", meta


def make_f4_extreme_angle_partial() -> tuple[bytes, str, dict[str, Any]]:
    """Partial object cut by frame edge (extreme angle / incomplete silhouette)."""
    img = np.full((480, 640, 3), 245, dtype=np.uint8)
    # Only the right half of an elongated body is visible (clipped at left)
    pts = np.array(
        [
            [0, 200],
            [180, 160],
            [200, 220],
            [190, 320],
            [0, 340],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [pts], (35, 40, 50))
    meta = {
        "description": "Partial silhouette clipped at image border",
        "has_scale_card": False,
        "partial": True,
        "extreme_angle": True,
    }
    return _encode_jpeg(img), "f4_partial_angle.jpg", meta


def make_f5_object_too_small() -> tuple[bytes, str, dict[str, Any]]:
    """Tiny object occupying a few percent of the frame."""
    img = np.full((720, 960, 3), 250, dtype=np.uint8)
    # ~12 px radius on 960-wide frame
    cv2.circle(img, (480, 360), 12, (20, 20, 20), thickness=-1)
    meta = {
        "description": "Very small object (~1% of frame width)",
        "has_scale_card": False,
        "object_radius_px": 12,
        "frame_w": 960,
    }
    return _encode_jpeg(img), "f5_too_small.jpg", meta


FIXTURE_BUILDERS = {
    "F1": {
        "builder": make_f1_no_scale_card,
        "condition": "No scale / calibration card",
        "expected": (
            "Scale unknown / default scale status; completes without crash; "
            "maps to some closest preset"
        ),
    },
    "F2": {
        "builder": make_f2_poor_lighting,
        "condition": "Poor lighting / low contrast",
        "expected": "Weak mask or fallback silhouette warning; no crash",
    },
    "F3": {
        "builder": make_f3_motion_blur,
        "condition": "Motion blur / blurry image",
        "expected": "Noisy or weak mask; no crash",
    },
    "F4": {
        "builder": make_f4_extreme_angle_partial,
        "condition": "Extreme angle / partial object",
        "expected": "Incomplete silhouette risk; still returns a result; no crash",
    },
    "F5": {
        "builder": make_f5_object_too_small,
        "condition": "Object too small in frame",
        "expected": "Unstable or fallback mask/scale; no crash",
    },
}


def _judge_pass(
    test_id: str,
    http_status: int,
    body: dict[str, Any] | None,
    error: str | None,
) -> tuple[bool, str]:
    """Pass = no unhandled 500 + honest educational behavior for the condition."""
    if http_status >= 500:
        return False, f"Server error {http_status}: {error or body}"
    if http_status >= 400 and body is None and error:
        # Controlled 400 is acceptable only if pipeline rejected unreadable input;
        # synthetic fixtures should be readable, so treat as fail for F1–F5 images.
        return False, f"Client error {http_status}: {error}"

    if body is None:
        return False, "Empty response body"

    status = str(body.get("status", ""))
    scale_status = str(body.get("scale_status", ""))
    detected = str(body.get("detected_object", ""))
    closest = str(body.get("closest_preset", ""))

    if http_status != 200 or status != "completed":
        return False, f"Unexpected status http={http_status} body_status={status}"

    notes: list[str] = []

    if test_id == "F1":
        # Must not claim a calibration card when none was present
        if "Calibration Card Detected" in scale_status:
            return False, f"False-positive scale card: {scale_status}"
        if "Default Scale" not in scale_status and "No marker" not in scale_status.lower():
            notes.append(f"scale_status={scale_status!r} (expected default/no marker wording)")
        notes.append(f"scale_status={scale_status}; closest={closest}")
        return True, "; ".join(notes) if notes else "Default scale, completed"

    if test_id == "F2":
        notes.append(f"detected={detected}; closest={closest}; scale={scale_status}")
        if "fallback" in detected.lower() or "weak" in detected.lower():
            notes.append("fallback/weak silhouette flagged")
        return True, "; ".join(notes)

    if test_id == "F3":
        notes.append(f"detected={detected}; closest={closest}")
        return True, "; ".join(notes)

    if test_id == "F4":
        notes.append(f"detected={detected}; closest={closest} (incomplete geometry risk accepted)")
        return True, "; ".join(notes)

    if test_id == "F5":
        notes.append(f"detected={detected}; closest={closest}; scale={scale_status}")
        if "fallback" in detected.lower() or "weak" in detected.lower():
            notes.append("weak/fallback silhouette as expected for tiny object")
        return True, "; ".join(notes)

    return True, "completed"


def run_matrix() -> dict[str, Any]:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    client = TestClient(app)
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for test_id, spec in FIXTURE_BUILDERS.items():
        builder = spec["builder"]
        case_t0 = time.perf_counter()
        payload_bytes, filename, meta = builder()
        fixture_path = os.path.join(FIXTURE_DIR, filename)
        with open(fixture_path, "wb") as f:
            f.write(payload_bytes)

        http_status = 0
        body: dict[str, Any] | None = None
        error: str | None = None
        try:
            files = {
                "file": (filename, BytesIO(payload_bytes), "image/jpeg"),
            }
            resp = client.post("/api/upload", files=files)
            http_status = resp.status_code
            try:
                body = resp.json()
            except Exception:
                body = None
                error = resp.text[:500]
            if http_status >= 400 and isinstance(body, dict):
                error = str(body.get("detail", body))
        except Exception as exc:
            http_status = 500
            error = f"Unhandled exception in TestClient: {type(exc).__name__}: {exc}"

        passed, judge_notes = _judge_pass(test_id, http_status, body, error)
        case_ms = (time.perf_counter() - case_t0) * 1000.0

        row = {
            "test_id": test_id,
            "condition": spec["condition"],
            "expected": spec["expected"],
            "http_status": http_status,
            "pass": passed,
            "fixture": fixture_path,
            "fixture_meta": meta,
            "response": body,
            "error": error,
            "observed": _observed_summary(http_status, body, error),
            "judge_notes": judge_notes,
            "wall_ms": round(case_ms, 1),
        }
        results.append(row)
        flag = "PASS" if passed else "FAIL"
        print(f"[{flag}] {test_id} http={http_status} {case_ms:.0f}ms — {judge_notes}")

    total_s = time.perf_counter() - t0
    summary = {
        "milestone": "M4",
        "title": "Failure / robustness matrix (smartphone input)",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware_note": "Ryzen 7 5700U class laptop; OpenCV CPU only; no GPU/AI recon",
        "total_wall_s": round(total_s, 3),
        "all_pass": all(r["pass"] for r in results),
        "results": results,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(OUT_DIR, f"m4_failure_{stamp}.json")
    latest_path = os.path.join(OUT_DIR, "m4_failure_latest.json")
    csv_path = os.path.join(OUT_DIR, "m4_failure_summary.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_id",
                "condition",
                "pass",
                "http_status",
                "observed",
                "judge_notes",
                "wall_ms",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "test_id": r["test_id"],
                    "condition": r["condition"],
                    "pass": r["pass"],
                    "http_status": r["http_status"],
                    "observed": r["observed"],
                    "judge_notes": r["judge_notes"],
                    "wall_ms": r["wall_ms"],
                }
            )

    print(f"\nWrote {json_path}")
    print(f"Wrote {latest_path}")
    print(f"Wrote {csv_path}")
    print(f"Total wall time: {total_s:.2f}s | all_pass={summary['all_pass']}")
    return summary


def _observed_summary(
    http_status: int,
    body: dict[str, Any] | None,
    error: str | None,
) -> str:
    if body and http_status == 200:
        return (
            f"HTTP {http_status}; status={body.get('status')}; "
            f"scale_status={body.get('scale_status')}; "
            f"detected={body.get('detected_object')}; "
            f"closest={body.get('closest_preset')}; "
            f"scale_estimate={body.get('scale_estimate')}"
        )
    if error:
        return f"HTTP {http_status}; error={error}"
    return f"HTTP {http_status}; body={body}"


if __name__ == "__main__":
    run_matrix()
