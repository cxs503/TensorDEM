#!/usr/bin/env python3
"""Mutation proof for the 2-D energy/fracture ledger negative controls (m4804).

``tests/test_dem2d_energy_fracture_ledger_negative.py`` asserts that the
ledger balance check *rejects* a deliberately corrupted ledger, and the file
is therefore green.  This script proves those green assertions are not
vacuous by mutating the test module and requiring it to turn RED:

1. tolerance blinded (``TOL`` 0.5 % -> 100 %) — the balance check would accept
   a corrupted ledger, so the negative controls must fail;
2. corruption made a no-op — the "corrupted" ledger equals the pristine one,
   so the negative controls must fail;
3. the fracture channel dropped from the balance identity — the positive
   control must fail (the identity is sensitive in both directions).

The module is restored from its own in-memory copy after every mutation and
re-run to prove it is green again.  The full log (mutation, exit code,
failing test ids, assertion messages) is written to
``artifacts/m4804_negative_controls/negative_injection.log``.

Usage::

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 scripts/negative_controls_m4804.py
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_FILE = "tests/test_dem2d_energy_fracture_ledger_negative.py"
OUT = ROOT / "artifacts" / "m4804_negative_controls"
LOG = OUT / "negative_injection.log"

#: (anchor, replacement, what the defect is, failing test ids required)
MUTATIONS = [
    (
        "TOL = 5.0e-3\n",
        "TOL = 1.0\n",
        "closure tolerance blinded from 0.5 % to 100 %",
        "negative",
    ),
    (
        "    corrupted.accounting[channel] *= factor\n",
        "    corrupted.accounting[channel] *= 1.0\n",
        "channel corruption made a no-op",
        "negative",
    ),
    (
        '        + acc["fracture_release_J"]\n',
        '        + 0.0 * acc["fracture_release_J"]\n',
        "fracture-release channel dropped from the balance identity",
        "positive_control",
    ),
]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, env=env)


def _summary(stdout: str) -> tuple[int, str, list[str]]:
    """Return (number of FAILED lines, pytest count line, failing test ids)."""
    lines = [line for line in stdout.strip().splitlines() if line.strip()]
    counts = [line for line in lines if re.search(r"\d+ (passed|failed)", line)]
    failures = [line.split(" ")[1] for line in lines if line.startswith("FAILED ")]
    message = counts[-1] if counts else (lines[-1] if lines else "<no output>")
    return len(failures), message, failures


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / TEST_FILE
    original = source.read_text()
    lines = [
        "m4804 negative controls: mutating the balance check must turn the "
        "negative-control tests red",
        f"test file: {TEST_FILE}",
        "",
    ]
    records = []
    ok = True
    try:
        green = _run([sys.executable, "-m", "pytest", TEST_FILE, "-q"])
        _, green_summary, _ = _summary(green.stdout)
        lines += [
            "baseline (unmutated)",
            f"  command: pytest {TEST_FILE} -q",
            f"  exit_code: {green.returncode}",
            f"  summary: {green_summary}",
            "",
        ]
        print(f"baseline: exit={green.returncode} {green_summary}")
        ok = ok and green.returncode == 0

        for anchor, replacement, description, expected_selector in MUTATIONS:
            if anchor not in original:
                lines.append(f"SKIP {description}: mutation anchor not found")
                ok = False
                continue
            banner = f"=== mutation: {description} ==="
            print(banner)
            lines.append(banner)
            try:
                source.write_text(original.replace(anchor, replacement, 1))
                red = _run([
                    sys.executable, "-m", "pytest", TEST_FILE, "-q",
                    "-k", expected_selector,
                ])
                full = _run([sys.executable, "-m", "pytest", TEST_FILE, "-q"])
                _, red_summary, failing = _summary(red.stdout)
                _, full_summary, _ = _summary(full.stdout)
                assertions = [
                    line.strip() for line in red.stdout.splitlines()
                    if line.strip().startswith(("E   ", "assert "))
                ]
                detected = red.returncode != 0 and len(failing) > 0
                whole_file_red = full.returncode != 0
                lines += [
                    f"  command: pytest {TEST_FILE} -q -k {expected_selector}   (mutated)",
                    f"  exit_code: {red.returncode}",
                    f"  summary: {red_summary}",
                    f"  failing: {', '.join(failing) if failing else '<none>'}",
                    f"  whole file after mutation: exit {full.returncode} | {full_summary}",
                ]
                lines += [f"  assertion: {item}" for item in assertions[:3]]
                verdict = (
                    "RED as required, GREEN after restore"
                    if detected and whole_file_red else "FAILED negative control"
                )
                lines += [f"  verdict: {verdict}", ""]
                print(f"  {red_summary} | detected={detected}")
                ok = ok and detected and whole_file_red
                records.append(
                    dict(
                        mutation=description,
                        selector=expected_selector,
                        exit_code=red.returncode,
                        summary=red_summary,
                        failing=failing,
                        whole_file_exit_code=full.returncode,
                        detected=bool(detected),
                    )
                )
            finally:
                source.write_text(original)
            restored = _run([sys.executable, "-m", "pytest", TEST_FILE, "-q"])
            _, restored_summary, _ = _summary(restored.stdout)
            lines += [
                f"  restored: exit_code {restored.returncode} | {restored_summary}",
                "",
            ]
            print(f"  restored: exit={restored.returncode} {restored_summary}")
            ok = ok and restored.returncode == 0
    finally:
        source.write_text(original)
    (OUT / "negative_injection.json").write_text(
        json.dumps(dict(test_file=TEST_FILE, mutations=records, ok=bool(ok)), indent=2) + "\n",
        encoding="utf-8",
    )
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"log written to {LOG.relative_to(ROOT)}")
    print("VERDICT:", "all mutations detected" if ok else "negative control FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())