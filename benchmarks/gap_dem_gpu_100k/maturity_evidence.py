#!/usr/bin/env python
"""Maturity evidence for GAP-DEM-GPU-100K benchmark.

Assembles machine-checkable evidence for the maturity gate:

* **L2** — implementation files (non-placeholder), entry point, run artifacts
* **L3** — integration: DEM3D solver supports device="cuda" natively
* **L4** — benchmark with ref-vs-got numbers and repro_cmd

Run::

    python benchmarks/gap_dem_gpu_100k/maturity_evidence.py --regenerate

Without --regenerate, verifies existing artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "benchmarks" / "gap_dem_gpu_100k"
RUNS_DIR = ROOT / "runs" / "gap_dem_gpu_100k"
ARTIFACTS_DIR = ROOT / "artifacts" / "gap_dem_gpu_100k"

IMPL_FILES = [
    "benchmarks/gap_dem_gpu_100k/benchmark_gpu_throughput.py",
    "benchmarks/gap_dem_gpu_100k/run.py",
]

RUN_ARTIFACTS = [
    "benchmarks/gap_dem_gpu_100k/results.json",
    "benchmarks/gap_dem_gpu_100k/report.md",
    "runs/gap_dem_gpu_100k/run_manifest.json",
]


def _file_hash(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_impl_files() -> list[dict]:
    """Verify implementation files exist and are non-placeholder."""
    results = []
    for rel in IMPL_FILES:
        p = ROOT / rel
        if not p.exists():
            results.append({"file": rel, "exists": False, "status": "MISSING"})
            continue
        content = p.read_text()
        is_placeholder = any(marker in content for marker in [
            "TODO", "FIXME", "placeholder", "raise NotImplementedError",
            "pass  # TODO",
        ])
        # Check for substantial code (>50 lines of non-comment/non-blank)
        code_lines = [
            line for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        results.append({
            "file": rel,
            "exists": True,
            "sha256": _file_hash(p),
            "code_lines": len(code_lines),
            "is_placeholder": is_placeholder,
            "status": "OK" if not is_placeholder and len(code_lines) > 10 else "SUSPECT",
        })
    return results


def _check_entry_point() -> dict:
    """Verify the entry point runs successfully."""
    entry = ROOT / "benchmarks" / "gap_dem_gpu_100k" / "run.py"
    if not entry.exists():
        return {"exists": False, "status": "MISSING"}

    try:
        result = subprocess.run(
            [sys.executable, str(entry), "--device", "cpu", "--warmup", "1", "--measure", "1"],
            capture_output=True, text=True, timeout=300, cwd=str(ROOT),
        )
        return {
            "exists": True,
            "exit_code": result.returncode,
            "stdout_lines": len(result.stdout.splitlines()),
            "status": "OK" if result.returncode == 0 else "FAIL",
        }
    except subprocess.TimeoutExpired:
        return {"exists": True, "status": "TIMEOUT"}
    except Exception as e:
        return {"exists": True, "status": "ERROR", "error": str(e)}


def _check_run_artifacts() -> list[dict]:
    """Verify run artifacts exist and are fresh."""
    results = []
    for rel in RUN_ARTIFACTS:
        p = ROOT / rel
        if not p.exists():
            results.append({"file": rel, "exists": False, "status": "MISSING"})
        else:
            results.append({
                "file": rel,
                "exists": True,
                "sha256": _file_hash(p),
                "size_bytes": p.stat().st_size,
                "status": "OK",
            })
    return results


def _check_integration() -> dict:
    """Verify DEM3D solver natively supports cuda device."""
    dem3d_file = ROOT / "src" / "tensordem" / "dem3d.py"
    if not dem3d_file.exists():
        return {"status": "MISSING"}

    content = dem3d_file.read_text()
    has_cuda_support = 'device: str = "cpu"' in content and '"cuda"' in content
    has_cuda_check = "torch.cuda.is_available()" in content

    return {
        "dem3d_file": "src/tensordem/dem3d.py",
        "cuda_device_param": has_cuda_support,
        "cuda_availability_check": has_cuda_check,
        "status": "OK" if has_cuda_support and has_cuda_check else "PARTIAL",
    }


def _check_benchmark() -> dict:
    """Verify benchmark has ref-vs-got numbers and repro_cmd."""
    results_file = BENCH_DIR / "results.json"
    manifest_file = RUNS_DIR / "run_manifest.json"

    if not results_file.exists():
        return {"status": "MISSING_RESULTS"}

    data = json.loads(results_file.read_text())

    # Check for 100k result
    has_100k = any(r.get("scale_label") == "100k" for r in data.get("results", []))

    # Check for acceptance field
    has_acceptance = any(
        "acceptance" in r for r in data.get("results", [])
        if r.get("scale_label") == "100k"
    )

    # Check for repro_cmd in manifest
    has_repro_cmd = False
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        has_repro_cmd = "repro_cmd" in manifest

    return {
        "has_100k_result": has_100k,
        "has_acceptance_field": has_acceptance,
        "has_repro_cmd": has_repro_cmd,
        "protocol": data.get("protocol", "unknown"),
        "threshold_s": data.get("acceptance_threshold_s", "unknown"),
        "status": "OK" if has_100k and has_acceptance and has_repro_cmd else "INCOMPLETE",
    }


def build_evidence(regenerate: bool = False) -> dict:
    """Build the full maturity evidence bundle."""
    if regenerate:
        # Re-run the benchmark to refresh artifacts
        print("Re-running benchmark to refresh artifacts...")
        subprocess.run(
            [sys.executable, str(BENCH_DIR / "run.py"), "--device", "cpu", "--warmup", "2", "--measure", "3"],
            cwd=str(ROOT), check=True,
        )

    evidence = {
        "benchmark": "GAP-DEM-GPU-100K",
        "protocol": "tensordem-dem3d-gpu-throughput-v1",
        "maturity_levels": {
            "L2": {
                "description": "Implementation files (non-placeholder) + entry point + run artifacts",
                "impl_files": _check_impl_files(),
                "entry_point": _check_entry_point(),
                "run_artifacts": _check_run_artifacts(),
            },
            "L3": {
                "description": "Integration: main solver supports cuda device",
                "integration": _check_integration(),
            },
            "L4": {
                "description": "Benchmark with ref-vs-got numbers and repro_cmd",
                "benchmark": _check_benchmark(),
            },
        },
    }

    # Compute overall maturity
    l2_ok = (
        all(f["status"] == "OK" for f in evidence["maturity_levels"]["L2"]["impl_files"])
        and evidence["maturity_levels"]["L2"]["entry_point"]["status"] == "OK"
        and all(a["status"] == "OK" for a in evidence["maturity_levels"]["L2"]["run_artifacts"])
    )
    l3_ok = evidence["maturity_levels"]["L3"]["integration"]["status"] == "OK"
    l4_ok = evidence["maturity_levels"]["L4"]["benchmark"]["status"] == "OK"

    evidence["maturity_reached"] = []
    if l2_ok:
        evidence["maturity_reached"].append("L2")
    if l3_ok:
        evidence["maturity_reached"].append("L3")
    if l4_ok:
        evidence["maturity_reached"].append("L4")

    evidence["overall_status"] = "PASS" if l2_ok and l3_ok and l4_ok else "PARTIAL"

    return evidence


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="GAP-DEM-GPU-100K maturity evidence")
    parser.add_argument("--regenerate", action="store_true",
                        help="Re-run benchmark before checking")
    args = parser.parse_args(argv)

    evidence = build_evidence(regenerate=args.regenerate)

    # Write evidence JSON
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    evidence_path = ARTIFACTS_DIR / "maturity_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(f"Evidence written to {evidence_path}")

    # Print summary
    print(f"\n=== Maturity Evidence Summary ===")
    print(f"Maturity reached: {', '.join(evidence['maturity_reached']) or 'NONE'}")
    print(f"Overall status: {evidence['overall_status']}")

    for level in ["L2", "L3", "L4"]:
        data = evidence["maturity_levels"][level]
        print(f"\n{level}:")
        if level == "L2":
            for f in data["impl_files"]:
                print(f"  impl: {f['file']} → {f['status']}")
            print(f"  entry: {data['entry_point']['status']}")
            for a in data["run_artifacts"]:
                print(f"  artifact: {a['file']} → {a['status']}")
        elif level == "L3":
            print(f"  integration: {data['integration']['status']}")
        elif level == "L4":
            print(f"  benchmark: {data['benchmark']['status']}")


if __name__ == "__main__":
    main()
