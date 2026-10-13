from __future__ import annotations

import json
import csv
import os

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from app.recon.capability import get_recon_capability
from app.recon.integration import register_mask_for_simulation
from app.recon.export_to_aerovoxel import mesh_to_center_slice_mask
from app.recon.evaluation import evaluate_manifest, load_manifest
from app.recon.engine import DepthAnythingEngine, OpenCV2DFallbackEngine, select_reconstruction_engine
import app.recon.engine as engine_module
import app.routes.simulate as simulate_route
from app.main import app
from app.recon.depth_mesh import depth_map_to_mesh, validate_mesh_file
from app.recon.sf3d_runner import run_sf3d
from app.recon.public_dataset import prepare_pix3d_manifest
import app.recon.public_evaluation as public_evaluation_module
from app.recon.depth_anything_runner import _resolve_model_source
from app.recon.phone_capture import prepare_phone_manifest


def test_capability_report_is_json_serializable() -> None:
    report = get_recon_capability()
    assert isinstance(report["cuda_available"], bool)
    json.dumps(report)


def test_register_mask_uses_existing_simulation_contract(tmp_path) -> None:
    source = tmp_path / "mask.npy"
    mask = np.zeros((64, 128), dtype=bool)
    mask[24:40, 52:76] = True
    np.save(source, mask)

    result = register_mask_for_simulation(str(source), str(tmp_path / "uploads"))
    assert result["shape"] == [64, 128]
    assert result["job_id"]
    assert np.array_equal(np.load(result["mask_path"]), mask)


def test_register_mask_rejects_wrong_shape(tmp_path) -> None:
    source = tmp_path / "mask.npy"
    np.save(source, np.ones((32, 64), dtype=bool))
    with pytest.raises(ValueError, match="must be"):
        register_mask_for_simulation(str(source), str(tmp_path / "uploads"))


def test_registered_reconstruction_mask_runs_real_lbm_endpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(simulate_route, "UPLOADS_DIR", str(tmp_path))
    source = tmp_path / "mask.npy"
    mask = np.zeros((64, 128), dtype=bool)
    mask[24:40, 52:76] = True
    np.save(source, mask)
    job = register_mask_for_simulation(str(source), str(tmp_path))

    response = TestClient(app).post(
        "/api/simulate/simple",
        json={"job_id": job["job_id"], "wind_speed": 15, "wind_angle_deg": 0},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["metrics"]["lbm_seconds"] >= 0
    assert np.isfinite(payload["metrics"]["cd_force_proxy"])


def test_mesh_export_produces_solver_shaped_center_slice(tmp_path) -> None:
    mesh_path = tmp_path / "box.ply"
    mask_path = tmp_path / "export" / "mask.npy"
    trimesh.creation.box(extents=(2.0, 1.0, 1.0)).export(mesh_path)

    result = mesh_to_center_slice_mask(str(mesh_path), str(mask_path))
    exported = np.load(mask_path, allow_pickle=False)
    assert exported.shape == (64, 128)
    assert exported.dtype == np.bool_
    assert np.any(exported)
    assert result["solid_cells"] == int(np.sum(exported))


def test_sf3d_missing_dependency_is_actionable(tmp_path) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"not-an-image")
    with pytest.raises((RuntimeError, OSError, ValueError)) as exc_info:
        run_sf3d(str(image), str(tmp_path / "out"))
    # On this CPU checkout sf3d is absent; if a developer installs it, malformed
    # input must still fail rather than returning a fake result.
    assert str(exc_info.value)


def test_evaluation_harness_records_typed_failures(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "case_id,input_type,image_path\nR1,SYNTHETIC_EVALUATION,missing.jpg\n",
        encoding="utf-8",
    )
    assert load_manifest(str(manifest))[0]["input_type"] == "SYNTHETIC_EVALUATION"
    summary = evaluate_manifest(str(manifest), str(tmp_path / "results"))
    assert summary["n_cases"] == 1
    assert summary["n_failed"] == 1
    assert summary["results"][0]["status"] == "FAILED"


def test_engine_selection_prefers_real_cpu_ml_before_opencv(monkeypatch) -> None:
    monkeypatch.setattr(engine_module, "get_recon_capability", lambda: {"ready_for_inference": False, "depth_anything_available": True, "recommended_device": "cpu"})
    engine = select_reconstruction_engine("depth_anything", device="cpu")
    assert isinstance(engine, DepthAnythingEngine)
    assert not isinstance(engine, OpenCV2DFallbackEngine)


def test_depth_mesh_is_real_serializable_geometry(tmp_path) -> None:
    depth = np.linspace(0.1, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    mask = np.zeros((64, 64), dtype=bool)
    mask[12:52, 16:48] = True
    mesh = depth_map_to_mesh(depth, mask, max_size=64)
    path = tmp_path / "depth_mesh.glb"
    mesh.export(path)
    stats = validate_mesh_file(str(path))
    assert stats["valid"] is True
    assert stats["vertices"] > 0
    assert stats["faces"] > 0


def test_engine_selection_matrix_has_explicit_final_fallback(monkeypatch) -> None:
    unavailable = {
        "ready_for_inference": False,
        "depth_anything_available": False,
        "sf3d_importable": False,
        "sf3d_import_error": "missing test dependency",
    }
    monkeypatch.setattr(engine_module, "get_recon_capability", lambda: unavailable)
    assert isinstance(select_reconstruction_engine("auto"), OpenCV2DFallbackEngine)
    with pytest.raises(RuntimeError, match="SF3D was explicitly requested"):
        select_reconstruction_engine("sf3d")
    with pytest.raises(RuntimeError, match="Depth Anything"):
        select_reconstruction_engine("depth_anything")
    with pytest.raises(ValueError, match="preferred"):
        select_reconstruction_engine("not-an-engine")


def test_pix3d_manifest_is_typed_and_requires_real_local_files(tmp_path) -> None:
    dataset_root = tmp_path / "pix3d"
    dataset_root.mkdir()
    metadata = []
    for index in range(5):
        relative = f"img/object_{index}.jpg"
        image_path = dataset_root / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32 + index, 24), color=(120, 140, 160)).save(image_path)
        metadata.append(
            {
                "img": relative,
                "category": f"category_{index}",
                "img_source": "internet",
                "mask": f"mask/object_{index}.png",
                "model": f"model/object_{index}/model.obj",
                "voxel": f"model/object_{index}/voxel.mat",
                "truncated": False,
                "occluded": False,
            }
        )
    (dataset_root / "pix3d.json").write_text(json.dumps(metadata), encoding="utf-8")

    output = tmp_path / "pix3d_manifest.csv"
    rows = prepare_pix3d_manifest(str(dataset_root), str(output), n_cases=5)

    assert len(rows) == 5
    assert {row["input_type"] for row in rows} == {"PUBLIC_DATASET"}
    assert {row["dataset"] for row in rows} == {"Pix3D"}
    assert {row["category"] for row in rows} == {f"category_{index}" for index in range(5)}
    assert all(len(row["image_sha256"]) == 64 for row in rows)
    assert all(row["image_width"] for row in rows)
    with output.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 5
    assert all(row["input_type"] == "PUBLIC_DATASET" for row in written)
    assert all(row["dataset_split"] == "official_release_unsplit" for row in written)


def test_pix3d_manifest_rejects_paths_outside_dataset_root(tmp_path) -> None:
    dataset_root = tmp_path / "pix3d"
    dataset_root.mkdir()
    (dataset_root / "pix3d.json").write_text(
        json.dumps([{"img": "../outside.jpg", "truncated": False, "occluded": False}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only 0 usable images"):
        prepare_pix3d_manifest(str(dataset_root), str(tmp_path / "manifest.csv"), n_cases=1)


def test_pix3d_manifest_skips_corrupt_image_and_records_optional_gt(tmp_path) -> None:
    root = tmp_path / "pix3d"
    (root / "img").mkdir(parents=True)
    (root / "model" / "chair").mkdir(parents=True)
    Image.new("RGB", (40, 30), color="white").save(root / "img" / "valid.png")
    (root / "img" / "corrupt.png").write_bytes(b"not an image")
    (root / "model" / "chair" / "voxel.mat").write_bytes(b"fixture")
    (root / "pix3d.json").write_text(
        json.dumps(
            [
                {"img": "img/corrupt.png", "category": "chair", "truncated": False, "occluded": False},
                {
                    "img": "img/valid.png",
                    "category": "chair",
                    "voxel": "model/chair/voxel.mat",
                    "truncated": False,
                    "occluded": False,
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = prepare_pix3d_manifest(str(root), str(tmp_path / "manifest.csv"), n_cases=1)

    assert rows[0]["image_path"].endswith("valid.png")
    assert rows[0]["ground_truth_voxel_path"].endswith("voxel.mat")
    assert rows[0]["image_width"] == "40"
    assert len(rows[0]["image_sha256"]) == 64


def test_public_dataset_evaluator_runs_lbm_and_persists_case_outputs(tmp_path, monkeypatch) -> None:
    from app.recon.depth_mesh import validate_mesh_file
    from app.recon.export_to_aerovoxel import mesh_to_center_slice_mask

    root = tmp_path / "pix3d"
    (root / "img").mkdir(parents=True)
    image_path = root / "img" / "input.png"
    Image.new("RGB", (80, 60), color=(200, 200, 200)).save(image_path)
    (root / "pix3d.json").write_text(
        json.dumps([{"img": "img/input.png", "category": "chair", "truncated": False, "occluded": False}]),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.csv"
    prepare_pix3d_manifest(str(root), str(manifest), n_cases=1)

    class TestEngine:
        name = "test-engine"
        selection_reason = "test-only geometry adapter"

        def reconstruct(self, input_path, output_dir, device=None):
            os.makedirs(output_dir, exist_ok=True)
            mesh_path = os.path.join(output_dir, "mesh.glb")
            trimesh.creation.box().export(mesh_path)
            stats = validate_mesh_file(mesh_path)
            mask_path = os.path.join(output_dir, "mask.npy")
            mask_info = mesh_to_center_slice_mask(mesh_path, mask_path)
            return {
                "engine": self.name,
                "device": "cpu",
                "mesh_path": mesh_path,
                "serialized_mesh": stats,
                "mask": mask_info,
            }

    monkeypatch.setattr(public_evaluation_module, "select_reconstruction_engine", lambda _name, **_kwargs: TestEngine())
    monkeypatch.setattr(simulate_route, "UPLOADS_DIR", str(tmp_path / "uploads"))
    out_dir = tmp_path / "public-eval"

    summary = public_evaluation_module.evaluate_public_dataset(
        str(manifest), str(out_dir), engine_name="auto", device="cpu"
    )

    assert summary["n_succeeded"] == 1, summary["results"][0].get("error")
    assert summary["n_lbm_completed"] == 1
    assert summary["success_rate"] == 1.0
    case = summary["results"][0]
    assert case["input_type"] == "PUBLIC_DATASET"
    assert case["geometry_valid"] is True
    assert case["simulation_artifacts"]["velocity"]["all_finite"] is True
    assert os.path.isfile(case["simulation_artifacts"]["pressure"]["path"])
    assert (out_dir / "results.json").is_file()
    assert (out_dir / "results.csv").is_file()


def test_public_dataset_evaluator_records_checksum_mismatch_without_inference(tmp_path, monkeypatch) -> None:
    root = tmp_path / "pix3d"
    (root / "img").mkdir(parents=True)
    image_path = root / "img" / "input.png"
    Image.new("RGB", (80, 60), color=(200, 200, 200)).save(image_path)
    (root / "pix3d.json").write_text(
        json.dumps([{"img": "img/input.png", "category": "chair", "truncated": False, "occluded": False}]),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.csv"
    prepare_pix3d_manifest(str(root), str(manifest), n_cases=1)
    Image.new("RGB", (81, 60), color=(20, 30, 40)).save(image_path)

    class MustNotRunEngine:
        name = "must-not-run"
        selection_reason = "checksum mismatch should stop before inference"

        def reconstruct(self, *_args, **_kwargs):
            raise AssertionError("inference must not run on a checksum-mismatched sample")

    monkeypatch.setattr(public_evaluation_module, "select_reconstruction_engine", lambda _name, **_kwargs: MustNotRunEngine())
    summary = public_evaluation_module.evaluate_public_dataset(
        str(manifest), str(tmp_path / "mismatch-output"), engine_name="auto", device="cpu"
    )

    assert summary["n_failed"] == 1
    assert summary["n_succeeded"] == 0
    assert "checksum differs" in summary["results"][0]["error"]
    persisted = json.loads((tmp_path / "mismatch-output" / "results.json").read_text(encoding="utf-8"))
    assert persisted["results"][0]["status"] == "FAILED"


def test_depth_runner_honors_explicit_local_model_directory(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "depth-model"
    model_dir.mkdir()
    monkeypatch.setenv("AEROVOXEL_RECON_MODEL_DIR", str(model_dir))
    source, local_only = _resolve_model_source("unavailable/model")
    assert source == str(model_dir)
    assert local_only is True


def test_phone_manifest_validates_and_preserves_phone_provenance(tmp_path) -> None:
    from PIL import Image

    image_path = tmp_path / "phone_capture.jpg"
    Image.new("RGB", (32, 32), color=(120, 140, 160)).save(image_path)
    manifest_path = tmp_path / "phone_manifest.csv"

    row = prepare_phone_manifest(str(image_path), str(manifest_path))

    assert row[0]["input_type"] == "REAL_PHONE_CAPTURE"
    assert row[0]["sha256"]
    assert row[0]["width"] == "32"
    assert load_manifest(str(manifest_path)) == row


def test_phone_manifest_rejects_duplicate_photos(tmp_path) -> None:
    from PIL import Image

    first = tmp_path / "phone_capture_01.jpg"
    second = tmp_path / "phone_capture_02.jpg"
    Image.new("RGB", (32, 32), color=(120, 140, 160)).save(first)
    second.write_bytes(first.read_bytes())

    with pytest.raises(ValueError, match="Duplicate phone capture"):
        prepare_phone_manifest([str(first), str(second)], str(tmp_path / "manifest.csv"))
