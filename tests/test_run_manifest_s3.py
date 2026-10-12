import hashlib
from pathlib import Path
import sys
import types

import pytest

from tensordem.run_manifest import build_manifest, store_s3_manifest_artifacts


class FakeArtifact:
    def __init__(self, uri, source, media_type):
        self.uri = uri
        self.media_type = media_type
        self.size_bytes = source.stat().st_size
        self.checksum_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        self.metadata = {"storage": "s3", "verified": True}

    def to_dict(self):
        return {
            "uri": self.uri, "media_type": self.media_type,
            "checksum_sha256": self.checksum_sha256, "size_bytes": self.size_bytes,
            "schema_name": None, "schema_version": 1, "metadata": self.metadata,
        }


class FakeS3Store:
    def __init__(self, bucket, *, prefix="", endpoint_url=None, region_name=None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.objects = {}

    def put_file(self, source, key, *, media_type):
        source = Path(source)
        object_key = f"{self.prefix}/{key}" if self.prefix else key
        uri = f"s3://{self.bucket}/{object_key}"
        artifact = FakeArtifact(uri, source, media_type)
        self.objects[uri] = (source.read_bytes(), artifact.checksum_sha256, artifact.size_bytes)
        return artifact

    def verify(self, artifact):
        stored = self.objects.get(artifact.uri)
        return stored is not None and stored[1:] == (artifact.checksum_sha256, artifact.size_bytes)


def test_s3_store_uploads_actual_dem_outputs_and_verifies_them(tmp_path, monkeypatch):
    output = tmp_path / "solver-results"
    output.mkdir()
    (output / "history.csv").write_text("time,force\n0,0\n1,12\n", encoding="utf-8")
    (output / "trajectory_3d.pt").write_bytes(b"real-3d-output-bytes")
    store = FakeS3Store("test-bucket", prefix="ci")
    monkeypatch.setitem(sys.modules, "tensorphysics", types.SimpleNamespace(S3ArtifactStore=lambda *a, **kw: store))

    manifest = build_manifest(output, "dem-s3-ci-001", None)
    stored, returned_store = store_s3_manifest_artifacts(
        manifest, output, "test-bucket", "ci", "dem-s3-ci-001", "https://s3.example"
    )

    assert returned_store is store
    assert stored["metadata"]["artifacts_verified"] is True
    assert stored["metadata"]["artifact_storage"] == "tensorphysics.S3ArtifactStore"
    assert {r["metadata"]["filename"] for r in stored["outputs"]} == {"history.csv", "trajectory_3d.pt"}
    assert stored["validation_status"] == "not_run"
    for record in stored["outputs"]:
        filename = record["metadata"]["filename"]
        assert record["uri"] == f"s3://test-bucket/ci/dem-s3-ci-001/{filename}"
        payload, checksum, size = store.objects[record["uri"]]
        assert payload == (output / filename).read_bytes()
        assert record["checksum_sha256"] == checksum == hashlib.sha256(payload).hexdigest()
        assert record["size_bytes"] == size == len(payload)


def test_s3_store_requires_tensorphysics_dependency(tmp_path, monkeypatch):
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "history.csv").write_text("time\n0\n", encoding="utf-8")
    (output / "trajectory.pt").write_bytes(b"trajectory")
    manifest = build_manifest(output, "dem-s3-no-dependency", None)

    def blocked_import(name, *args, **kwargs):
        if name == "tensorphysics":
            raise ImportError("blocked in test")
        return original_import(name, *args, **kwargs)

    import builtins
    original_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "tensorphysics", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match="S3 upload requires TensorPhysics"):
        store_s3_manifest_artifacts(manifest, output, "bucket", "", "dem-s3-no-dependency")
