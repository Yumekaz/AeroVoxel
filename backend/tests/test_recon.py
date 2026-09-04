from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

from app.recon.capability import get_recon_capability
from app.recon.integration import register_mask_for_simulation
from app.recon.export_to_aerovoxel import mesh_to_center_slice_mask
from app.recon.sf3d_runner import run_sf3d


def test_capability_report_is_json_serializable() -> None:
    report = get_recon_capability()
    assert report["cuda_available"] is False
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
