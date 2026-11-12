#!/usr/bin/env python3
"""m16316 -- energy-audit probe for the m16243 elastic-limit shear case.

The audited case reports a 19-25 % relative energy-audit residual and explains it as
"model-level, not an integration error" (dt-independent).  That explanation leaves the
mechanism open; this probe tests one concrete alternative -- that the residual is the
tangential-spring energy retained in the history variable ``xi`` of pairs that have
separated (the kernel only writes ``xi`` back for the *active* pairs, so a broken
contact keeps its stored tangential energy forever, which is never dissipated and
never released through the audit).

It exports the mission branch read-only (``git archive``), copies the kernel, inserts a
single diagnostic print immediately before the summary statistics, and runs one filed
point (kn=1e3, gamma_dot=1, N=400, strain 1.8 from 0.8, dt=0.0025 t_c, seed 20261112),
i.e. exactly the point whose filed tau* is 11.187963375044525.

Usage
-----
    python scripts/audit/m16316_energy_probe.py [--kn 1e3] [--tag audit_probe_kn1e3]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = "agent/m16243"
WORK = Path("/tmp/m16316_probe")
RUNS = ROOT / "runs" / "audit_m16316"
ANCHOR = "    tau_mean = float(np.mean(tau_samples)) if tau_samples else float(\"nan\")"
PROBE = '''    # --- AUDIT PROBE (m16316): split the .energy residual by mechanism ---------
    _contact = delta > 0.0
    _xi_live = 0.5 * kt * float(np.sum(xi_t[_contact] ** 2))
    _xi_dead = 0.5 * kt * float(np.sum(xi_t[~_contact] ** 2))
    _elas_n = 0.5 * kn * float(np.sum(np.clip(delta[_contact], 0.0, None) ** 2))
    if win["started"]:
        _w = work - win["work0"]
        _d = (diss_n + diss_t) - win["diss0"]
        _e_end = e_kin_end + e_el_end
        print("PROBE kn=%.3g work_win=%.6g diss_win=%.6g dE_window=%.6g "
              "residual=%.6g | xi_dead=%.6g (%.2f%% of work) xi_live=%.6g "
              "elastic_n=%.6g rel_defect_excl_dead=%.4f" % (
                  case.kn, _w, _d, _e_end - win["e0"],
                  _w - _d - (_e_end - win["e0"]), _xi_dead,
                  100.0 * _xi_dead / max(abs(_w), 1e-30), _xi_live, _elas_n,
                  abs(_w - _d - (_e_end - win["e0"]) - _xi_dead) / max(abs(_w), 1e-30)))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kn", default="1e3")
    ap.add_argument("--shear-rate", default="1.0")
    ap.add_argument("--n", default="400")
    ap.add_argument("--tag", default="audit_probe_kn1e3")
    args = ap.parse_args()

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    tar = Path("/tmp/m16316_probe.tar")
    subprocess.run(["git", "-C", str(ROOT), "archive", REF, "-o", str(tar)], check=True)
    with tarfile.open(tar) as tf:
        tf.extractall(WORK)

    kernel = WORK / "src" / "tensordem" / "elastic_limit_shear.py"
    s = kernel.read_text()
    if ANCHOR not in s:
        raise SystemExit("anchor not found -- the kernel changed in an incompatible way")
    kernel.write_text(s.replace(ANCHOR, PROBE + ANCHOR, 1))

    RUNS.mkdir(parents=True, exist_ok=True)
    log = RUNS / f"probe_kn{args.kn}.log"
    cmd = [sys.executable,
           str(WORK / "benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py"),
           "--tag", args.tag, "--kn", args.kn, "--shear-rate", args.shear_rate,
           "--n", args.n, "--strain", "1.8", "--strain-from", "0.8",
           "--dt-factor", "0.0025", "--outdir", str(RUNS / "probe_out")]
    with log.open("w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            env={"PYTHONPATH": str(WORK / "src"), "PATH": "/usr/bin:/bin"}).returncode
    print(f"exit={rc} log={log}")
    for line in log.read_text().splitlines():
        if line.startswith("PROBE"):
            print(line)
    return rc


if __name__ == "__main__":
    sys.exit(main())