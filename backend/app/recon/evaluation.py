"""Reproducible manifest-driven reconstruction evaluation harness."""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable

from app.recon.engine import select_reconstruction_engine

VALID_INPUT_TYPES = {"PHONE_CAPTURE", "PUBLIC_DATASET", "SYNTHETIC_EVALUATION"}


def load_manifest(path: str) -> list[dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"case_id", "input_type", "image_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain columns: {', '.join(sorted(required))}")
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for row in rows:
        case_id = (row.get("case_id") or "").strip()
        input_type = (row.get("input_type") or "").strip().upper()
        image_path = (row.get("image_path") or "").strip()
        if not case_id or case_id in seen:
            raise ValueError(f"Manifest case_id must be non-empty and unique: {case_id!r}")
        if input_type not in VALID_INPUT_TYPES:
            raise ValueError(f"Unsupported input_type {input_type!r}; use {sorted(VALID_INPUT_TYPES)}")
        if not image_path:
            raise ValueError(f"Manifest image_path is empty for {case_id}")
        seen.add(case_id)
        normalized.append({"case_id": case_id, "input_type": input_type, "image_path": image_path})
    return normalized


def evaluate_manifest(
    manifest_path: str,
    output_dir: str,
    runner: Callable[..., dict[str, Any]] | None = None,
    device: str | None = None,
    engine: str = "auto",
) -> dict[str, Any]:
    """Run every manifest case and record successes/failures, never fabricating output."""
    rows = load_manifest(manifest_path)
    os.makedirs(output_dir, exist_ok=True)
    selected_engine = None if runner is not None else select_reconstruction_engine(engine)
    results: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for row in rows:
        started = time.perf_counter()
        case_output = os.path.join(output_dir, row["case_id"])
        result: dict[str, Any] = {
            **row,
            "status": "FAILED",
            "output_dir": os.path.abspath(case_output),
            "error": None,
        }
        try:
            model_result = (
                runner(row["image_path"], case_output, device=device)
                if runner is not None
                else selected_engine.reconstruct(row["image_path"], case_output, device=device)
            )
            result.update(
                {
                    "status": "SUCCEEDED",
                    "engine": model_result.get("engine") if isinstance(model_result, dict) else None,
                    "device": model_result.get("device") if isinstance(model_result, dict) else device,
                    "model_result": model_result,
                }
            )
        except Exception as exc:  # each case is isolated so one bad input cannot hide later failures
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_seconds"] = round(time.perf_counter() - started, 3)
        results.append(result)
        print(f"[{result['status']}] {row['case_id']} {result['wall_time_seconds']:.3f}s")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest": os.path.abspath(manifest_path),
        "n_cases": len(results),
        "n_succeeded": sum(r["status"] == "SUCCEEDED" for r in results),
        "n_failed": sum(r["status"] == "FAILED" for r in results),
        "total_wall_time_seconds": round(time.perf_counter() - started_all, 3),
        "results": results,
        "honesty": "No success, quality, latency, or VRAM claim is made for failed or unavailable model runs.",
    }
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(output_dir, "results.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "input_type", "image_path", "status", "engine", "device", "wall_time_seconds", "error"],
        )
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in results)
    return summary
