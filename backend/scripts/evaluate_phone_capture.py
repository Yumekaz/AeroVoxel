"""Run typed real-phone cases through reconstruction, GLB, mask, and LBM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

from PIL import Image
from fastapi.testclient import TestClient

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.main import app
from app.recon.evaluation import load_manifest
from app.recon.engine import select_reconstruction_engine
from app.recon.integration import register_mask_for_simulation


def _image_metadata(path: str) -> dict[str, object]:
    with Image.open(path) as image:
        width, height = image.size
    digest_builder = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    return {"sha256": digest, "width": width, "height": height}


def evaluate_phone_cases(manifest_path: str, output_dir: str, engine_name: str, device: str | None) -> dict[str, object]:
    rows = load_manifest(manifest_path)
    if any(row["input_type"] != "REAL_PHONE_CAPTURE" for row in rows):
        raise ValueError("Phone evaluation requires input_type=REAL_PHONE_CAPTURE")
    os.makedirs(output_dir, exist_ok=True)
    engine = select_reconstruction_engine(engine_name, device=device)
    client = TestClient(app)
    results: list[dict[str, object]] = []
    seen_hashes: dict[str, str] = {}
    started_all = time.perf_counter()
    for row in rows:
        started = time.perf_counter()
        result: dict[str, object] = {**row, "status": "FAILED", "error": None}
        try:
            image_metadata = _image_metadata(row["image_path"])
            if row.get("sha256") and image_metadata["sha256"] != row["sha256"]:
                raise ValueError("Phone image checksum differs from its prepared manifest")
            result.update(image_metadata)
            if result["sha256"] in seen_hashes:
                raise ValueError(f"Duplicate phone capture; same SHA-256 as {seen_hashes[result['sha256']]}")
            seen_hashes[str(result["sha256"])] = row["case_id"]
            case_dir = os.path.join(output_dir, row["case_id"])
            reconstruction = engine.reconstruct(row["image_path"], case_dir, device=device)
            job = register_mask_for_simulation(reconstruction["mask"]["mask_path"], os.path.join(BACKEND_ROOT, "app", "assets", "uploads"))
            response = client.post(
                "/api/simulate/simple",
                json={"job_id": job["job_id"], "wind_speed": 15, "wind_angle_deg": 0},
            )
            simulation = response.json()
            if response.status_code != 200 or simulation.get("status") != "completed":
                raise RuntimeError(f"LBM returned HTTP {response.status_code}")
            result.update(
                {
                    "status": "SUCCEEDED",
                    "engine": reconstruction.get("engine"),
                    "device": reconstruction.get("device"),
                    "mesh": reconstruction.get("serialized_mesh"),
                    "mask": reconstruction.get("mask"),
                    "timings": reconstruction.get("timings"),
                    "simulation": simulation,
                }
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_seconds"] = round(time.perf_counter() - started, 3)
        results.append(result)
        print(f"[{result['status']}] {row['case_id']} {result['wall_time_seconds']:.3f}s")

    summary = {
        "manifest": os.path.abspath(manifest_path),
        "n_cases": len(results),
        "n_succeeded": sum(item["status"] == "SUCCEEDED" for item in results),
        "n_failed": sum(item["status"] == "FAILED" for item in results),
        "total_wall_time_seconds": round(time.perf_counter() - started_all, 3),
        "dataset_type": "REAL_PHONE_CAPTURE",
        "results": results,
    }
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(output_dir, "results.csv"), "w", newline="", encoding="utf-8") as handle:
        fields = ["case_id", "input_type", "image_path", "sha256", "width", "height", "status", "engine", "device", "wall_time_seconds", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: item.get(field) for field in fields} for item in results)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--engine", choices=["auto", "sf3d", "depth_anything", "opencv"], default="auto")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    summary = evaluate_phone_cases(args.manifest, args.output_dir, args.engine, args.device)
    print(f"succeeded={summary['n_succeeded']} failed={summary['n_failed']}")
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
