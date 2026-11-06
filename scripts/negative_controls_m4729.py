#!/usr/bin/env python3
"""Negative controls for the m4729 test-expansion mission.

For each new test file this script injects one deliberate defect into the
production module it covers, runs the test file, and requires it to turn RED.
The original source is restored from git afterwards and the run is re-executed
to prove the file is green again.  The full log (mutation, failing test name,
assertion message) is written to ``artifacts/m4729_negative_controls/``.

Usage::

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 scripts/negative_controls_m4729.py
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "m4729_negative_controls"
LOG = OUT / "negative_injection.log"

#: (test file, production file, old snippet, new snippet, what the defect is)
MUTATIONS = [
    (
        "tests/test_shear_convergence_fit.py",
        "src/tensordem/shear_convergence.py",
        "    x = 1.0 / n\n",
        "    x = 1.0 / (2.0 * n)\n",
        "boundary-layer design variable scaled by 1/2",
    ),
    (
        "tests/test_threefield_coupling_units.py",
        "src/tensordem/threefield/coupling.py",
        "    return (6.0 * mass / (math.pi * density)) ** (1.0 / 3.0)\n",
        "    return (6.0 * mass / (math.pi * density)) ** (1.0 / 2.0)\n",
        "sphere diameter exponent 1/3 -> 1/2",
    ),
    (
        "tests/test_buoyancy_verification_reference.py",
        "src/tensordem/buoyancy_verification.py",
        "    return 2.0 * math.sqrt(max(0.0, radius * radius - h * h))\n",
        "    return math.sqrt(max(0.0, radius * radius - h * h))\n",
        "waterplane chord width factor 2 dropped",
    ),
]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, env=env)


def _summary(stdout: str) -> tuple[int, str]:
    """Return (exit status hint, one-line summary) from a pytest run."""
    tail = [line for line in stdout.strip().splitlines() if line.strip()]
    counts = [line for line in tail if re.search(r"\d+ (passed|failed)", line)]
    failures = [line for line in tail if line.startswith("FAILED ")]
    message = counts[-1] if counts else (tail[-1] if tail else "<no output>")
    first = failures[0] if failures else "<no FAILED line>"
    return (len(failures), f"{message} | {first}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["m4729 negative controls: injected defects must turn the new tests red", ""]
    ok = True
    for test_file, src_file, old, new, description in MUTATIONS:
        source = ROOT / src_file
        original = source.read_text()
        if old not in original:
            lines.append(f"SKIP {src_file}: mutation anchor not found")
            ok = False
            continue
        banner = f"=== {test_file} <- {src_file} ({description}) ==="
        print(banner)
        lines.append(banner)
        try:
            source.write_text(original.replace(old, new, 1))
            red = _run([sys.executable, "-m", "pytest", test_file, "-q", "-x"])
            n_fail, red_summary = _summary(red.stdout)
            lines.append(f"command: pytest {test_file} -q -x   (mutated source)")
            lines.append(f"exit_code: {red.returncode}")
            lines.append(f"red summary: {red_summary}")
            assertion = [
                line for line in red.stdout.splitlines()
                if line.strip().startswith(("assert ", "E   "))
            ]
            lines.extend(f"assertion: {line.strip()}" for line in assertion[:4])
            red_ok = red.returncode != 0 and n_fail > 0
        finally:
            source.write_text(original)
        green = _run([sys.executable, "-m", "pytest", test_file, "-q"])
        _, green_summary = _summary(green.stdout)
        lines.append(f"command: pytest {test_file} -q   (restored source)")
        lines.append(f"exit_code: {green.returncode}")
        lines.append(f"green summary: {green_summary}")
        lines.append(f"verdict: {'RED as required, GREEN after restore' if red_ok and green.returncode == 0 else 'FAILED negative control'}")
        lines.append("")
        print(lines[-4])
        print(lines[-3])
        print(lines[-1])
        ok = ok and red_ok and green.returncode == 0
    LOG.write_text("\n".join(lines) + "\n")
    print(f"log written to {LOG.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())