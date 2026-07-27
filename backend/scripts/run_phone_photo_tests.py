"""
Generate phone-like synthetic captures and exercise the upload path.

Creates 8 JPG images that approximate smartphone photos of objects (JPEG
compression, noise, uneven lighting, slight blur; scale card sometimes absent).
Images are written under evaluation_outputs/phone_like_photos/ (gitignored) and
each is uploaded via FastAPI TestClient. Results JSON lands in evaluation_outputs/.

This is not a claim of real-phone certification — fixtures are synthetic.

Usage (from backend/):
    python scripts/run_phone_photo_tests.py
    python scripts/run_phone_photo_tests.py --n 8 --seed 7
"""

from __future__ import annotations

import argparse
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
PHOTO_DIR = os.path.join(OUT_DIR, "phone_like_photos")


def _encode_jpeg(bgr: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _add_noise(img: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _vignette(img: np.ndarray, strength: float = 0.35) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
    factor = 1.0 - strength * np.clip(r, 0, 1.2) ** 1.5
    out = img.astype(np.float32) * factor[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_scale_card(img: np.ndarray, rng: np.random.Generator) -> None:
    """Approximate credit-card aspect (~85.6×53.98 mm) as a light rectangle."""
    h, w = img.shape[:2]
    card_w = int(w * rng.uniform(0.12, 0.18))
    card_h = int(card_w * 0.63)
    x0 = int(rng.integers(max(1, w // 20), max(2, w // 3)))
    y0 = int(rng.integers(int(h * 0.55), max(int(h * 0.55) + 1, h - card_h - 10)))
    color = (
        int(rng.integers(200, 245)),
        int(rng.integers(200, 245)),
        int(rng.integers(200, 245)),
    )
    cv2.rectangle(img, (x0, y0), (x0 + card_w, y0 + card_h), color, thickness=-1)
    # Thin border like plastic card edge
    cv2.rectangle(
        img,
        (x0, y0),
        (x0 + card_w, y0 + card_h),
        (int(color[0] * 0.7),) * 3,
        thickness=2,
    )


def _draw_object(img: np.ndarray, kind: str, rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    cx = int(w * rng.uniform(0.42, 0.58))
    cy = int(h * rng.uniform(0.38, 0.52))
    base = (
        int(rng.integers(20, 55)),
        int(rng.integers(20, 55)),
        int(rng.integers(25, 60)),
    )

    if kind == "circle":
        r = int(min(h, w) * rng.uniform(0.12, 0.18))
        cv2.circle(img, (cx, cy), r, base, thickness=-1)
    elif kind == "car":
        bw = int(w * rng.uniform(0.28, 0.38))
        bh = int(h * rng.uniform(0.12, 0.18))
        x0, y0 = cx - bw // 2, cy - bh // 2
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), base, thickness=-1)
        roof = np.array(
            [
                [x0 + int(bw * 0.2), y0],
                [x0 + int(bw * 0.35), y0 - int(bh * 0.7)],
                [x0 + int(bw * 0.7), y0 - int(bh * 0.7)],
                [x0 + int(bw * 0.85), y0],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(img, [roof], base)
        wr = max(8, bh // 3)
        cv2.circle(img, (x0 + int(bw * 0.22), y0 + bh), wr, (15, 15, 15), -1)
        cv2.circle(img, (x0 + int(bw * 0.78), y0 + bh), wr, (15, 15, 15), -1)
    elif kind == "airfoil":
        a = int(w * rng.uniform(0.22, 0.3))
        b = int(h * rng.uniform(0.04, 0.07))
        cv2.ellipse(img, (cx, cy), (a, b), int(rng.integers(-8, 9)), 0, 360, base, -1)
    elif kind == "drone":
        arm = int(min(h, w) * 0.14)
        thick = max(6, arm // 4)
        cv2.line(img, (cx - arm, cy - arm), (cx + arm, cy + arm), base, thick)
        cv2.line(img, (cx - arm, cy + arm), (cx + arm, cy - arm), base, thick)
        for dx, dy in ((-arm, -arm), (arm, -arm), (-arm, arm), (arm, arm)):
            cv2.circle(img, (cx + dx, cy + dy), thick + 4, (base[0] // 2,) * 3, -1)
        cv2.circle(img, (cx, cy), thick + 6, base, -1)
    elif kind == "box":
        bw = int(w * rng.uniform(0.18, 0.28))
        bh = int(h * rng.uniform(0.18, 0.28))
        cv2.rectangle(
            img,
            (cx - bw // 2, cy - bh // 2),
            (cx + bw // 2, cy + bh // 2),
            base,
            thickness=-1,
        )
    else:  # wedge / partial-ish blob
        pts = np.array(
            [
                [cx - int(w * 0.12), cy + int(h * 0.1)],
                [cx + int(w * 0.18), cy - int(h * 0.05)],
                [cx + int(w * 0.1), cy + int(h * 0.14)],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(img, [pts], base)


def generate_phone_like_photo(
    index: int,
    rng: np.random.Generator,
) -> tuple[bytes, str, dict[str, Any]]:
    """One synthetic smartphone-style capture."""
    # Common phone-ish resolutions (downscaled for laptop CV speed)
    sizes = [(960, 720), (1280, 720), (1080, 810), (800, 600)]
    w, h = sizes[index % len(sizes)]

    # Uneven base lighting (gradients)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    warm = 180 + 40 * xx + 25 * (1 - yy)
    cool = 160 + 30 * (1 - xx) + 20 * yy
    bg = np.stack(
        [
            np.clip(cool + rng.uniform(-10, 10), 40, 250),
            np.clip((warm + cool) / 2 + rng.uniform(-8, 8), 40, 250),
            np.clip(warm + rng.uniform(-10, 10), 40, 250),
        ],
        axis=-1,
    ).astype(np.uint8)

    kinds = ["circle", "car", "airfoil", "drone", "box", "wedge", "car", "circle"]
    kind = kinds[index % len(kinds)]
    has_card = index % 3 != 0  # ~2/3 have a scale-like card; some missing
    jpeg_q = int(rng.choice([48, 55, 62, 70, 78]))
    blur_k = int(rng.choice([0, 0, 3, 5, 7]))  # sometimes sharp

    img = bg.copy()
    _draw_object(img, kind, rng)
    if has_card:
        _draw_scale_card(img, rng)

    # Soft focus / handshake blur
    if blur_k >= 3:
        img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    # Occasional mild motion streak
    if index % 4 == 2:
        k = np.zeros((1, 11), dtype=np.float32)
        k[0, :] = 1.0 / 11.0
        img = cv2.filter2D(img, -1, k)

    img = _add_noise(img, rng, sigma=float(rng.uniform(4.0, 14.0)))
    img = _vignette(img, strength=float(rng.uniform(0.2, 0.45)))

    # Slight color cast (phone auto white-balance miss)
    cast = np.array(
        [rng.uniform(0.92, 1.08), rng.uniform(0.94, 1.06), rng.uniform(0.9, 1.1)],
        dtype=np.float32,
    )
    img = np.clip(img.astype(np.float32) * cast, 0, 255).astype(np.uint8)

    name = f"phone_like_{index + 1:02d}_{kind}.jpg"
    meta = {
        "index": index + 1,
        "object_kind": kind,
        "has_scale_card": has_card,
        "jpeg_quality": jpeg_q,
        "blur_ksize": blur_k,
        "width": w,
        "height": h,
        "note": "Synthetic phone-like capture (not a real smartphone photo)",
    }
    return _encode_jpeg(img, quality=jpeg_q), name, meta


def run_phone_like_tests(n: int = 8, seed: int = 7) -> dict[str, Any]:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PHOTO_DIR, exist_ok=True)

    readme = os.path.join(PHOTO_DIR, "README.txt")
    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as f:
            f.write(
                "Phone-like synthetic captures generated by scripts/run_phone_photo_tests.py.\n"
                "Gitignored via evaluation_outputs/. Not real phone photos.\n"
            )

    rng = np.random.default_rng(seed)
    client = TestClient(app)
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for i in range(n):
        case_t0 = time.perf_counter()
        payload, filename, meta = generate_phone_like_photo(i, rng)
        path = os.path.join(PHOTO_DIR, filename)
        with open(path, "wb") as fh:
            fh.write(payload)

        http_status = 0
        body: dict[str, Any] | None = None
        error: str | None = None
        try:
            resp = client.post(
                "/api/upload",
                files={"file": (filename, BytesIO(payload), "image/jpeg")},
            )
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
            error = f"{type(exc).__name__}: {exc}"

        ok = (
            http_status == 200
            and isinstance(body, dict)
            and body.get("status") == "completed"
        )
        # Honesty: if no card drawn, do not accept false-positive calibration wording
        honesty_ok = True
        honesty_note = ""
        if ok and not meta["has_scale_card"]:
            scale_status = str(body.get("scale_status", ""))
            if "Calibration Card Detected" in scale_status:
                honesty_ok = False
                honesty_note = f"false-positive scale card: {scale_status}"
                ok = False

        wall_ms = (time.perf_counter() - case_t0) * 1000.0
        row = {
            "test_id": f"PHONE_{i + 1:02d}",
            "file": filename,
            "fixture": path,
            "fixture_meta": meta,
            "http_status": http_status,
            "pass": ok and honesty_ok,
            "response": body,
            "error": error,
            "honesty_note": honesty_note or None,
            "observed": (
                f"HTTP {http_status}; status={body.get('status') if body else None}; "
                f"scale_status={body.get('scale_status') if body else None}; "
                f"detected={body.get('detected_object') if body else None}; "
                f"closest={body.get('closest_preset') if body else None}"
            ),
            "wall_ms": round(wall_ms, 1),
        }
        results.append(row)
        flag = "PASS" if row["pass"] else "FAIL"
        print(
            f"[{flag}] {row['test_id']} {filename} http={http_status} "
            f"{wall_ms:.0f}ms card={meta['has_scale_card']}"
        )

    summary = {
        "milestone": "phone_like",
        "title": "Phone-like synthetic photo robustness (upload path)",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware_note": "Laptop-safe OpenCV fixtures + TestClient; not real phone corpus",
        "disclaimer": (
            "Images are synthetic phone-like captures (noise, compression, lighting). "
            "Pass = completed upload without crash and basic scale honesty when no card."
        ),
        "seed": seed,
        "n_images": n,
        "folder": PHOTO_DIR,
        "total_wall_s": round(time.perf_counter() - t0, 3),
        "all_pass": all(r["pass"] for r in results),
        "n_pass": sum(1 for r in results if r["pass"]),
        "n_fail": sum(1 for r in results if not r["pass"]),
        "results": results,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(OUT_DIR, f"phone_like_{stamp}.json")
    latest_path = os.path.join(OUT_DIR, "phone_like_latest.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    note_path = os.path.join(OUT_DIR, "phone_like_summary.txt")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(
            f"Phone-like tests {summary['generated_at_utc']}: "
            f"{summary['n_pass']}/{summary['n_images']} pass "
            f"(all_pass={summary['all_pass']})\n"
            f"Fixtures: {PHOTO_DIR}\n"
            f"JSON: {latest_path}\n"
            f"{summary['disclaimer']}\n"
        )

    print(f"\nWrote {json_path}")
    print(f"Wrote {latest_path}")
    print(f"Wrote {note_path}")
    print(
        f"Total wall time: {summary['total_wall_s']:.2f}s | "
        f"pass {summary['n_pass']}/{summary['n_images']} all_pass={summary['all_pass']}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate phone-like photos and run upload robustness tests"
    )
    parser.add_argument("--n", type=int, default=8, help="Number of synthetic photos (default 8)")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    args = parser.parse_args()
    summary = run_phone_like_tests(n=max(1, args.n), seed=args.seed)
    return 0 if summary.get("all_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
