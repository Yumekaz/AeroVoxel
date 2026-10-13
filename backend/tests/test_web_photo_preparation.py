import hashlib
import json

from PIL import Image
import pytest

from scripts import prepare_web_photo_manifest as preparation


def source_fixture(tmp_path, monkeypatch, checksum_valid=True):
    originals = tmp_path / "input"
    originals.mkdir()
    photo = originals / "object.jpg"
    Image.new("RGB", (24, 32), "red").save(photo)
    record = {"case_id": "PUBLIC_01", "input_type": "PUBLIC_DATASET", "file_name": photo.name,
              "sha256": hashlib.sha256(photo.read_bytes()).hexdigest() if checksum_valid else "0" * 64,
              "width": 24, "height": 32}
    metadata = tmp_path / "backend/app/recon/web_photo_sources.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps([record]))
    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    return originals


def test_local_originals_verify_without_opening_a_network_context(tmp_path, monkeypatch):
    source = source_fixture(tmp_path, monkeypatch)
    def no_network(**_kwargs):
        raise AssertionError("Local reuse must not need TLS or network access")
    monkeypatch.setattr(preparation.ssl, "create_default_context", no_network)
    output = tmp_path / "prepared"
    manifest = preparation.prepare(output, source)
    assert "PUBLIC_DATASET" in manifest.read_text()
    assert (output / "originals/object.jpg").read_bytes() == (source / "object.jpg").read_bytes()
    with pytest.raises(FileExistsError):
        preparation.prepare(output, source)


def test_changed_original_is_rejected_and_failure_is_retained(tmp_path, monkeypatch):
    source = source_fixture(tmp_path, monkeypatch, checksum_valid=False)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="failed preparation"):
        preparation.prepare(output, source)
    attempts = json.loads((output / "preparation.json").read_text())
    assert attempts[0]["status"] == "FAILED"
    assert "checksum changed" in attempts[0]["error"]
    assert (output / "originals/object.jpg").exists()
    assert not (output / "manifest.csv").exists()
