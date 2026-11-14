"""``python -m tensordem.cli`` must be a real module entry, not a silent no-op.

Defect (found in m19466): ``src/tensordem/cli.py`` defined ``main()`` but had no
``if __name__ == "__main__"`` block, so the repro command documented in
``docs/XVAL_DEM_LIBRARY_BINDING_CONTRACT_3D.md`` (``python -m tensordem.cli
binding-contract ...``) imported the module, printed nothing, wrote no artifact
and still exited 0 -- a silent success that makes "no artifact" look like "pass".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable, "-m", "tensordem.cli", *args],
                          capture_output=True, text=True, cwd=cwd or str(ROOT), env=env, timeout=600)


def test_module_entry_prints_help():
    proc = _run(["--help"])
    assert proc.returncode == 0, proc.stderr
    assert "binding-contract" in proc.stdout


def test_module_entry_runs_the_binding_contract_and_writes_artifacts(tmp_path):
    proc = _run(["binding-contract", "--out", str(tmp_path),
                 "--benchmark-energy-steps", "50", "--benchmark-energy-steps-3d", "20"])
    assert proc.returncode == 0, proc.stderr
    target = tmp_path / "binding_contract_cli.json"
    assert target.is_file(), "module entry exited 0 without writing its artifact"
    report = json.loads(target.read_text())
    assert report["verify"]["status"] == "CONFORMS"
    assert all(p["status"] == "PASS" for p in report["probes"])
    assert report["benchmark"]["verdict"] == "MET"
    assert report["benchmark_3d"]["verdict"] == "MET"
    assert "verify=CONFORMS" in proc.stdout


def test_module_entry_and_package_entry_agree(tmp_path):
    """``python -m tensordem`` and ``python -m tensordem.cli`` must agree on the fingerprint."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    pkg = subprocess.run([sys.executable, "-m", "tensordem", "binding-contract", "--out",
                          str(tmp_path / "a"), "--benchmark-energy-steps", "20",
                          "--benchmark-energy-steps-3d", "10"],
                         capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=600)
    mod = _run(["binding-contract", "--out", str(tmp_path / "b"), "--benchmark-energy-steps", "20",
                "--benchmark-energy-steps-3d", "10"])
    assert pkg.returncode == 0, pkg.stderr
    assert mod.returncode == 0, mod.stderr
    a = json.loads((tmp_path / "a" / "binding_contract_cli.json").read_text())
    b = json.loads((tmp_path / "b" / "binding_contract_cli.json").read_text())
    assert a["contract_fingerprint"] == b["contract_fingerprint"]
    assert a["benchmark"]["worst_rel_error"] == b["benchmark"]["worst_rel_error"]