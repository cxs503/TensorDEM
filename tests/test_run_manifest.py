import hashlib
import json
from pathlib import Path

from tensordem.run_manifest import build_manifest


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
