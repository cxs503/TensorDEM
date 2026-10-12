import hashlib
import json
from pathlib import Path

import pytest

from tensordem.run_manifest import build_manifest, validate_shared_uri_prefix


def test_manifest_records_real_artifact_checksums_and_unvalidated_status(tmp_path: Path):
    history = tmp_path / "history.csv"
    trajectory = tmp_path / "trajectory.pt"
    history.write_text("time,reaction_y\\n0,0\\n", encoding="utf-8")
    trajectory.write_bytes(b"trajectory-placeholder")

    manifest = build_manifest(tmp_path, "dem-ci-001", None)

    assert manifest["run_id"] == "dem-ci-001"
    assert manifest["status"] == "succeeded"
    assert manifest["validation_status"] == "not_run"
    assert len(manifest["outputs"]) == 2
    by_name = {item["metadata"]["filename"]: item for item in manifest["outputs"]}
    assert by_name["history.csv"]["checksum_sha256"] == hashlib.sha256(history.read_bytes()).hexdigest()
    assert by_name["trajectory.pt"]["checksum_sha256"] == hashlib.sha256(trajectory.read_bytes()).hexdigest()
    assert by_name["history.csv"]["uri"] == history.resolve().as_uri()


def test_manifest_uses_shared_uri_prefix_when_supplied(tmp_path: Path):
    (tmp_path / "history.csv").write_text("time\\n0\\n", encoding="utf-8")
    (tmp_path / "trajectory.pt").write_bytes(b"trajectory-placeholder")

    manifest = build_manifest(tmp_path, "dem-ci-002", "s3://example-bucket/dem-ci-002")
    assert {item["uri"] for item in manifest["outputs"]} == {
        "s3://example-bucket/dem-ci-002/history.csv",
        "s3://example-bucket/dem-ci-002/trajectory.pt",
    }


def test_manifest_accepts_three_dimensional_trajectory_name(tmp_path: Path):
    history = tmp_path / "history.csv"
    trajectory = tmp_path / "trajectory_3d.pt"
    history.write_text("time,reaction_z\n0,0\n", encoding="utf-8")
    trajectory.write_bytes(b"3d-trajectory-placeholder")

    manifest = build_manifest(tmp_path, "dem-3d-ci-001", None)

    by_name = {item["metadata"]["filename"]: item for item in manifest["outputs"]}
    assert set(by_name) == {"history.csv", "trajectory_3d.pt"}
    assert by_name["trajectory_3d.pt"]["checksum_sha256"] == hashlib.sha256(
        trajectory.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("prefix", ["", "/tmp/artifacts/run-1", "file:///tmp/run-1", "s3:///missing-bucket"])
def test_shared_uri_prefix_rejects_local_or_incomplete_values(prefix):
    with pytest.raises(ValueError, match="absolute shared-storage URI|not file://"):
        validate_shared_uri_prefix(prefix)


def test_shared_uri_prefix_normalizes_trailing_slash():
    assert validate_shared_uri_prefix("s3://bucket/run-1/") == "s3://bucket/run-1"
    assert validate_shared_uri_prefix("https://storage.example/runs/1") == "https://storage.example/runs/1"


def test_local_artifact_store_copies_and_verifies_solver_outputs(tmp_path: Path, monkeypatch):
    import sys
    import types
    from urllib.parse import unquote, urlsplit
    from tensordem.run_manifest import store_manifest_artifacts

    output = tmp_path / "solver output"
    output.mkdir()
    (output / "history.csv").write_text("time,reaction_y\\n0,0\\n", encoding="utf-8")
    (output / "trajectory.pt").write_bytes(b"trajectory bytes")

    class FakeArtifact:
        def __init__(self, path, media_type):
            self.uri = path.resolve().as_uri()
            self.media_type = media_type
            self.size_bytes = path.stat().st_size
            self.checksum_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            self.metadata = {"storage": "local", "verified": True}

        def to_dict(self):
            return {
                "uri": self.uri, "media_type": self.media_type,
                "checksum_sha256": self.checksum_sha256, "size_bytes": self.size_bytes,
                "schema_name": None, "schema_version": 1, "metadata": self.metadata,
            }

    class FakeStore:
        def __init__(self, root):
            self.root = Path(root).resolve()
            self.root.mkdir(parents=True, exist_ok=True)

        def put_file(self, source, key, *, media_type):
            import shutil
            destination = self.root / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            return FakeArtifact(destination, media_type)

        def verify(self, artifact):
            path = Path(unquote(urlsplit(artifact.uri).path))
            return path.is_file() and path.stat().st_size == artifact.size_bytes and (
                hashlib.sha256(path.read_bytes()).hexdigest() == artifact.checksum_sha256
            )

    fake_module = types.ModuleType("tensorphysics")
    fake_module.LocalArtifactStore = FakeStore
    monkeypatch.setitem(sys.modules, "tensorphysics", fake_module)

    manifest = build_manifest(output, "dem-local-001", None)
    stored = store_manifest_artifacts(manifest, output, tmp_path / "artifact store", "dem-local-001")

    assert stored["metadata"]["artifacts_verified"] is True
    assert stored["metadata"]["artifact_storage"] == "tensorphysics.LocalArtifactStore"
    assert all(item["uri"].startswith("file://") for item in stored["outputs"])
    assert {item["metadata"]["filename"] for item in stored["outputs"]} == {"history.csv", "trajectory.pt"}
    for record in stored["outputs"]:
        stored_path = Path(unquote(urlsplit(record["uri"]).path))
        source_path = output / record["metadata"]["filename"]
        assert stored_path.read_bytes() == source_path.read_bytes()
        assert record["size_bytes"] == source_path.stat().st_size


def test_local_artifact_store_requires_tensorphysics(tmp_path: Path, monkeypatch):
    import builtins
    import sys
    from tensordem.run_manifest import store_manifest_artifacts

    monkeypatch.delitem(sys.modules, "tensorphysics", raising=False)
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "history.csv").write_text("time\\n0\\n", encoding="utf-8")
    (output / "trajectory.pt").write_bytes(b"trajectory")
    manifest = build_manifest(output, "dem-local-missing-dependency", None)
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "tensorphysics":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match="requires TensorPhysics"):
        store_manifest_artifacts(manifest, output, tmp_path / "store", "dem-local-missing-dependency")
