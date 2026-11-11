"""CLI runner for the chute-flow cross-validation case.

Examples
--------
Generate the shared case files (once; every tool reads them)::

    python -m tensordem.chute_flow.runner gen \
        --case artifacts/xval_chute2001/case_spec/case_input.json \
        --outdir artifacts/xval_chute2001/case_spec

Run our solver to steady state (dumps the steady window for the shared
post-processing)::

    python -m tensordem.chute_flow.runner run --specdir artifacts/xval_chute2001/case_spec \
        --mode steady --max-steps 120000 --out runs/chute_steady.json --dump runs/chute_steady.csv

Arrest (h_stop) run::

    python -m tensordem.chute_flow.runner run --specdir ... --mode arrest --out runs/arrest.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import analysis
from .engine import ChuteEngine
from .spec import (DEFAULT_CASE, ChuteCase, make_base, make_initial_particles,
                   write_case_spec)

__all__ = ["main", "run_steady", "run_arrest"]


def _load(specdir: Path, theta_deg: float | None, n_target: int | None
          ) -> tuple[ChuteCase, np.ndarray, np.ndarray, dict]:
    manifest = json.loads((specdir / "case.json").read_text())
    case = ChuteCase.from_dict(manifest["case_spec"])
    if theta_deg is not None:
        case = case.with_(theta_deg=float(theta_deg), case_id=f"theta{theta_deg:g}")
    if n_target is not None:
        case = case.with_(n_target=int(n_target), case_id=f"n{int(n_target)}")
    base = np.genfromtxt(specdir / "base_spheres.csv", delimiter=",", skip_header=1)
    init = np.genfromtxt(specdir / "particles_init.csv", delimiter=",", skip_header=1)
    if theta_deg is not None or n_target is not None:
        # theta does not change the geometry; a different mass does
        base = make_base(case)
        init = make_initial_particles(case)
    return case, np.atleast_2d(init), np.atleast_2d(base), manifest


def _dumps_to_csv(path: Path, snaps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("t,id,x,y,z,vx,vy,vz\n")
        for k, snap in enumerate(snaps):
            n = snap["z"].size
            ids = np.arange(n)
            for i in range(n):
                fh.write("%.9e,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n"
                         % (snap["t"], ids[i], snap["x"][i], snap["y"][i],
                            snap["z"][i], snap["vx"][i], snap["vy"][i],
                            snap["vz"][i]))


def run_steady(case: ChuteCase, init: np.ndarray, base: np.ndarray, *,
               max_steps: int = 120000, chunk: int = 4000,
               dump_every: int = 2000, window_steps: int = 20000,
               tol: float = 0.01, min_chunks: int = 3) -> dict:
    """Integrate to a steady state, then sample the velocity profile window."""
    eng = ChuteEngine(case, init, base)
    hist = []
    chunk_means: list[float] = []
    steady_at = None
    steps = 0
    while steps < max_steps:
        k = min(chunk, max_steps - steps)
        eng.run(k)
        steps += k
        diag = eng.diagnostics()
        chunk_means.append(diag["vx_mean_over_vref"])
        hist.append(diag)
        if (len(chunk_means) >= min_chunks + 1
                and diag["t"] > 20.0 * case.t_ref):
            ref = np.mean(chunk_means[-min_chunks:])
            recent = chunk_means[-1]
            if ref != 0.0 and abs(recent - ref) / abs(ref) < tol:
                steady_at = steps
                break
    # sampling window: window_steps more steps, dumping every dump_every
    snaps = []
    nwin = max(1, int(window_steps // dump_every))
    for _ in range(nwin):
        eng.run(dump_every)
        snaps.append(eng.snapshot())
    result = {
        "case_id": case.case_id,
        "case_spec": case.summary(),
        "mode": "steady",
        "steps_total": steps + nwin * dump_every,
        "steady_detected_at_steps": steady_at,
        "final_diagnostics": eng.diagnostics(),
        "history": hist,
        "dumps": len(snaps),
    }
    prof = analysis.average_profile_over_window(
        [{"z": s["z"], "vx": s["vx"], "t": s["t"]} for s in snaps], case=case)
    result["profile"] = prof
    return result, snaps


def run_arrest(case: ChuteCase, init: np.ndarray, base: np.ndarray, *,
               max_steps: int = 120000, window: int = 8000) -> dict:
    """Integrate a fixed mass at a fixed inclination and classify arrest/flow."""
    eng = ChuteEngine(case, init, base)
    eng.run(max(1, max_steps - window))
    win = []
    for _ in range(4):
        eng.run(window // 4)
        win.append(eng.diagnostics())
    v_rms = float(np.mean([w["v_rms_over_vref"] for w in win]))
    vx_mean = float(np.mean([w["vx_mean_over_vref"] for w in win]))
    state = analysis.classify_run(v_rms_over_vref=v_rms,
                                  vx_mean_over_vref=vx_mean)
    n = init.shape[0]
    h_vol = n * (math.pi / 6.0) * case.d ** 3 / (case.lx * case.ly)
    return {
        "case_id": case.case_id,
        "case_spec": case.summary(),
        "mode": "arrest",
        "steps_total": max_steps,
        "h_over_d": h_vol / case.d,
        "n_particles": int(n),
        "state": state,
        "v_rms_over_vref": v_rms,
        "vx_mean_over_vref": vx_mean,
        "z95_over_d": float(np.mean([w["z95_over_d"] for w in win])),
        "final_diagnostics": eng.diagnostics(),
    }, eng


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="write the shared case files")
    g.add_argument("--case", type=str, default=None,
                   help="JSON with ChuteCase overrides (optional)")
    g.add_argument("--outdir", type=str, required=True)
    g.add_argument("--theta", type=float, default=None)
    g.add_argument("--n", type=int, default=None)
    g.add_argument("--lx", type=float, default=None)
    g.add_argument("--ly", type=float, default=None)

    r = sub.add_parser("run", help="run one case with our solver")
    r.add_argument("--specdir", type=str, required=True)
    r.add_argument("--mode", choices=("steady", "arrest"), default="steady")
    r.add_argument("--theta", type=float, default=None)
    r.add_argument("--n", type=int, default=None)
    r.add_argument("--max-steps", type=int, default=120000)
    r.add_argument("--dump-every", type=int, default=2000)
    r.add_argument("--window-steps", type=int, default=20000)
    r.add_argument("--out", type=str, required=True)
    r.add_argument("--dump", type=str, default=None)

    a = sub.add_parser("analyze", help="profile + Bagnold fit from a dump CSV")
    a.add_argument("--specdir", type=str, required=True)
    a.add_argument("--dump", type=str, required=True)
    a.add_argument("--theta", type=float, default=None)
    a.add_argument("--n", type=int, default=None)
    a.add_argument("--out", type=str, required=True)

    args = ap.parse_args(argv)

    if args.cmd == "gen":
        payload = dict(DEFAULT_CASE)
        if args.case and Path(args.case).exists():
            payload.update(json.loads(Path(args.case).read_text()))
        if args.theta is not None:
            payload["theta_deg"] = float(args.theta)
        if args.n is not None:
            payload["n_target"] = int(args.n)
        if args.lx is not None:
            payload["lx_d"] = float(args.lx)
        if args.ly is not None:
            payload["ly_d"] = float(args.ly)
        case = ChuteCase.from_dict(payload).with_(
            case_id=f"chute_lx{payload['lx_d']:g}_n{payload['n_target']}_"
                    f"theta{payload['theta_deg']:g}")
        manifest = write_case_spec(case, args.outdir)
        print(json.dumps(manifest, indent=2))
        return 0

    if args.cmd == "run":
        case, init, base, manifest = _load(Path(args.specdir), args.theta, args.n)
        if args.mode == "steady":
            result, snaps = run_steady(
                case, init, base, max_steps=args.max_steps,
                dump_every=args.dump_every, window_steps=args.window_steps)
            if args.dump:
                _dumps_to_csv(Path(args.dump), snaps)
        else:
            result, eng = run_arrest(case, init, base, max_steps=args.max_steps)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result[k] for k in
                          ("case_id", "mode", "steps_total")}, indent=2))
        return 0

    if args.cmd == "analyze":
        case, init, base, manifest = _load(Path(args.specdir), args.theta, args.n)
        arr = analysis.read_dump(args.dump)
        ts = np.unique(arr[:, 0])
        dumps = [
            {"z": arr[arr[:, 0] == t, 4], "vx": arr[arr[:, 0] == t, 5], "t": t}
            for t in ts]
        prof = analysis.average_profile_over_window(dumps, case=case)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(prof, indent=2))
        print(json.dumps({k: prof[k] for k in
                          ("h_over_d", "bagnold_p", "bagnold_r2",
                           "vx_bulk_over_vref")}, indent=2))
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())