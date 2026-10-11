"""Run public real-world images through reconstruction, voxelization, and LBM."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.recon.evaluation import load_manifest
from app.recon.engine import select_reconstruction_engine
from app.recon.integration import register_mask_for_simulation
from app.routes import simulate as simulate_route


def _image_metadata(path: str) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"actual_image_sha256": digest.hexdigest(), "actual_width": width, "actual_height": height}


def _array_summary(path: str) -> dict[str, Any]:
    array = np.load(path, allow_pickle=False)
    return {
        "shape": list(array.shape),
        "all_finite": bool(np.all(np.isfinite(array))),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def evaluate_public_dataset(
    manifest_path: str,
    output_dir: str,
    engine_name: str = "auto",
    device: str | None = None,
) -> dict[str, Any]:
    """Evaluate each typed public sample; failures are isolated and persisted."""
    rows = load_manifest(manifest_path)
    if any(row["input_type"] != "PUBLIC_DATASET" for row in rows):
        raise ValueError("Public dataset evaluation requires input_type=PUBLIC_DATASET")

    os.makedirs(output_dir, exist_ok=True)
    engine = select_reconstruction_engine(engine_name)
    client = TestClient(app)
    results: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    dataset_root = rows[0].get("dataset_root", "") if rows else ""
    provenance_path = os.path.join(dataset_root, "provenance.json") if dataset_root else ""

    for row in rows:
        started_case = time.perf_counter()
        case_dir = os.path.abspath(os.path.join(output_dir, row["case_id"]))
        result: dict[str, Any] = {
            **row,
            "status": "FAILED",
            "error": None,
            "output_dir": case_dir,
            "reconstruction": None,
            "simulation": None,
        }
        try:
            if not os.path.isfile(row["image_path"]):
                raise FileNotFoundError(row["image_path"])
            image_info = _image_metadata(row["image_path"])
            expected_hash = row.get("image_sha256")
            if expected_hash and image_info["actual_image_sha256"] != expected_hash:
                raise ValueError("Input image checksum differs from the prepared Pix3D manifest")
            result.update(image_info)

            reconstruction = engine.reconstruct(row["image_path"], case_dir, device=device)
            mesh_path = reconstruction.get("mesh_path")
            mesh_stats = reconstruction.get("serialized_mesh") or {}
            if not mesh_path or not os.path.isfile(mesh_path) or not mesh_stats.get("valid"):
                raise ValueError("Reconstruction did not produce a valid serialized geometry artifact")
            mask_path = reconstruction["mask"]["mask_path"]
            job = register_mask_for_simulation(mask_path, simulate_route.UPLOADS_DIR)
            simulation_started = time.perf_counter()
            response = client.post(
                "/api/simulate/simple",
                json={"job_id": job["job_id"], "wind_speed": 15, "wind_angle_deg": 0},
            )
            simulation_seconds = time.perf_counter() - simulation_started
            simulation = response.json()
            if response.status_code != 200 or simulation.get("status") != "completed":
                raise RuntimeError(f"LBM returned HTTP {response.status_code}: {simulation}")

            simulation_dir = os.path.join(case_dir, "simulation")
            os.makedirs(simulation_dir, exist_ok=True)
            arrays: dict[str, Any] = {}
            for kind in ("mask", "velocity", "pressure"):
                source = os.path.join(simulate_route.UPLOADS_DIR, f"{kind}_{job['job_id']}.npy")
                target = os.path.join(simulation_dir, f"{kind}.npy")
                if not os.path.isfile(source):
                    raise FileNotFoundError(f"LBM output missing: {source}")
                shutil.copyfile(source, target)
                array_stats = _array_summary(target)
                if not array_stats["all_finite"]:
                    raise ValueError(f"LBM {kind} output contains non-finite values")
                arrays[kind] = {"path": os.path.abspath(target), **array_stats}

            reconstruction_mask = np.load(mask_path, allow_pickle=False).astype(bool)
            result.update(
                {
                    "status": "SUCCEEDED",
                    "engine": reconstruction.get("engine"),
                    "device": reconstruction.get("device"),
                    "reconstruction": reconstruction,
                    "mesh_path": mesh_path,
                    "geometry_valid": bool(mesh_stats.get("valid")),
                    "vertex_count": mesh_stats.get("vertices"),
                    "face_count": mesh_stats.get("faces"),
                    "mesh_watertight": mesh_stats.get("watertight"),
                    "voxel_mask_path": os.path.abspath(mask_path),
                    "mask_dimensions": list(reconstruction_mask.shape),
                    "mask_nonzero_ratio": float(np.mean(reconstruction_mask)),
                    "lbm_seconds": simulation.get("metrics", {}).get("lbm_seconds"),
                    "simulation_wall_seconds": round(simulation_seconds, 3),
                    "simulation_metrics": simulation.get("metrics", {}),
                    "simulation_artifacts": arrays,
                    "simulation": simulation,
                    "quality_metrics_note": (
                        "Pix3D provides GT voxel/model references, but this view-conditioned relief mesh is not "
                        "in the aligned canonical frame required for valid 3D IoU/CD/EMD. No such score is claimed."
                    ),
                }
            )
        except Exception as exc:  # isolate each sample and persist its failure
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_seconds"] = round(time.perf_counter() - started_case, 3)
        results.append(result)
        print(f"[{result['status']}] {row['case_id']} {result['wall_time_seconds']:.3f}s")

    successful_results = [item for item in results if item["status"] == "SUCCEEDED"]
    inference_times = [
        item["reconstruction"]["timings"]["inference_seconds"]
        for item in successful_results
        if item.get("reconstruction", {}).get("timings", {}).get("inference_seconds") is not None
    ]
    lbm_times = [item["lbm_seconds"] for item in successful_results if item.get("lbm_seconds") is not None]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_type": "PUBLIC_DATASET",
        "dataset": "Pix3D",
        "dataset_source": "https://pix3d.csail.mit.edu/ (official archive, selective HTTP range extraction)",
        "dataset_version": "official_release_version_unspecified",
        "dataset_license": "CC BY 4.0",
        "dataset_provenance_path": provenance_path if os.path.isfile(provenance_path) else None,
        "dataset_split": "official_release_unsplit",
        "manifest": os.path.abspath(manifest_path),
        "output_dir": os.path.abspath(output_dir),
        "engine_requested": engine_name,
        "engine_selected": engine.name,
        "engine_selection_reason": engine.selection_reason,
        "requested_device": device,
        "n_cases": len(results),
        "n_succeeded": sum(item["status"] == "SUCCEEDED" for item in results),
        "n_failed": sum(item["status"] == "FAILED" for item in results),
        "success_rate": round(len(successful_results) / len(results), 4) if results else 0.0,
        "n_valid_geometry": sum(item.get("geometry_valid") is True for item in results),
        "n_voxel_masks": sum(bool(item.get("voxel_mask_path")) for item in results),
        "n_lbm_completed": sum((item.get("simulation") or {}).get("status") == "completed" for item in results),
        "total_wall_time_seconds": round(time.perf_counter() - started_all, 3),
        "mean_case_wall_time_seconds": round(
            sum(item["wall_time_seconds"] for item in results) / len(results), 3
        ) if results else None,
        "mean_inference_seconds": round(sum(inference_times) / len(inference_times), 3) if inference_times else None,
        "mean_lbm_seconds": round(sum(lbm_times) / len(lbm_times), 3) if lbm_times else None,
        "ground_truth_metrics": (
            "Not computed: generated view-conditioned relief meshes are not canonically aligned to Pix3D GT voxels/models."
        ),
        "results": results,
    }
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    csv_fields = [
        "case_id", "input_type", "dataset", "category", "image_source", "image_path",
        "image_sha256", "actual_image_sha256", "image_width", "image_height", "status",
        "engine", "device", "mesh_path", "geometry_valid", "vertex_count", "face_count",
        "ground_truth_voxel_path", "voxel_mask_path", "mask_nonzero_ratio", "lbm_seconds",
        "simulation_wall_seconds", "wall_time_seconds", "error",
    ]
    with open(os.path.join(output_dir, "results.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({field: item.get(field) for field in csv_fields} for item in results)
    return summary
