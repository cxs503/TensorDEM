"""Command-line entry point of the DEM erosion face (node p176w2-n11).

Three cheap, deterministic sub-commands that make the capability callable from
outside the test suite (a pipeline script, a platform coupling, a reviewer):

.. code-block:: bash

    python -m tensordem.erosion_dem impact --speed 30 --angle 90 --diameter-um 150
    python -m tensordem.erosion_dem stream --angle 20 --n-particles 500
    python -m tensordem.erosion_dem pipe --speed 2.0 --seconds 1.0
    python -m tensordem.erosion_dem verify              # full acceptance study

Every sub-command prints one JSON object on stdout (``--out`` also writes it) so
the numbers can be consumed programmatically; nothing is printed that is not
computed by the library.
"""
from __future__ import annotations

import argparse
import json
import sys

from .crater import single_impact_crater
from .recorder import ImpactErosionRecorder
from .stream import CASE_LAW, CASE_MATERIAL, StreamConfig, run_impact_stream

__all__ = ["main", "build_parser"]


def _jsonify(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj


def _cmd_impact(args: argparse.Namespace) -> dict:
    geo = single_impact_crater(
        args.speed,
        args.angle,
        args.diameter_um * 1e-6,
        args.particle_density,
        CASE_MATERIAL,
        law=CASE_LAW,
    )
    return {
        "command": "impact",
        "material": CASE_MATERIAL.name,
        "inputs": {
            "speed_m_s": args.speed,
            "alpha_deg": args.angle,
            "diameter_m": args.diameter_um * 1e-6,
            "particle_density_kg_m3": args.particle_density,
        },
        "erosion_ratio_kg_kg": geo.removed_mass_kg / (
            args.particle_density * (1.0 / 6.0) * 3.141592653589793
            * (args.diameter_um * 1e-6) ** 3
        ),
        "removed_volume_m3": geo.removed_volume_m3,
        "removed_mass_kg": geo.removed_mass_kg,
        "crater_depth_m": geo.depth_m,
        "crater_radius_m": geo.radius_m,
        "physically_admissible": geo.physically_admissible,
    }


def _cmd_stream(args: argparse.Namespace) -> dict:
    cfg = StreamConfig(n_particles=args.n_particles, angle_mean_deg=args.angle,
                       speed_mean_m_s=args.speed, seed=args.seed)
    out = dict(run_impact_stream(cfg))
    wall = out.pop("wall", None)
    out.pop("records", None)          # per-impact list is for the benchmark, not the CLI
    if wall is not None:
        out["wall_area_m2"] = float(wall.area_m2)
        out["wall_integrated_volume_m3"] = float(wall.integrated_volume_m3)
        out["wall_mean_depth_m"] = float(wall.mean_depth_m)
    out["command"] = "stream"
    out["stream_config"] = {
        "n_particles": cfg.n_particles,
        "angle_mean_deg": cfg.angle_mean_deg,
        "speed_mean_m_s": cfg.speed_mean_m_s,
        "seed": cfg.seed,
    }
    return out


def _cmd_pipe(args: argparse.Namespace) -> dict:
    from tensordem.slurry_transport.pipe_dem import PipeSectionDEMCFD

    rec = ImpactErosionRecorder(material=CASE_MATERIAL, law=CASE_LAW)
    pipe = PipeSectionDEMCFD(D=args.diameter, d=args.particle_diameter,
                             erosion=rec, seed=args.seed)
    pipe.initialize(args.speed)
    res, _hist = pipe.run(t_end=args.seconds, dt=args.dt)
    state = rec.state()
    state["command"] = "pipe"
    state["pipe"] = {
        "D_m": pipe.D,
        "particle_diameter_m": pipe.d,
        "U_mix_m_s": pipe.U_mix,
        "dt_s": args.dt,
        "simulated_s": pipe.t,
        "excess_gradient": res.get("excess_gradient"),
    }
    return state


def _cmd_verify(args: argparse.Namespace) -> dict:
    import pathlib
    import sys as _sys

    root = pathlib.Path(__file__).resolve().parents[3]
    bench = root / "benchmarks" / "erosion_dem_verification.py"
    if not bench.exists():  # installed wheel: benchmark not shipped
        raise SystemExit(f"benchmark not found at {bench}")
    _sys.path.insert(0, str(bench.parent))
    import importlib

    mod = importlib.import_module("erosion_dem_verification")
    outdir = pathlib.Path(args.outdir)
    argv = ["--outdir", str(outdir)]
    if args.quick:
        argv.append("--quick")
    mod.main(argv)
    return json.loads((outdir / "results.json").read_text())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tensordem.erosion_dem",
        description="DEM-side solid-particle erosion face (Finnie/Oka/DNV-RP-O501)",
    )
    p.add_argument("--out", default=None, help="write the JSON result to this path")
    sub = p.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("impact", help="single-impact crater")
    imp.add_argument("--speed", type=float, default=30.0)
    imp.add_argument("--angle", type=float, default=90.0)
    imp.add_argument("--diameter-um", type=float, default=150.0)
    imp.add_argument("--particle-density", type=float, default=2650.0)
    imp.set_defaults(func=_cmd_impact)

    st = sub.add_parser("stream", help="multi-particle ballistic stream")
    st.add_argument("--angle", type=float, default=90.0)
    st.add_argument("--speed", type=float, default=30.0)
    st.add_argument("--n-particles", type=int, default=500)
    st.add_argument("--seed", type=int, default=20261111)
    st.set_defaults(func=_cmd_stream)

    pp = sub.add_parser("pipe", help="slurry pipe DEM with erosion coupling")
    pp.add_argument("--speed", type=float, default=2.0, help="mixture velocity [m/s]")
    pp.add_argument("--seconds", type=float, default=1.0)
    pp.add_argument("--diameter", type=float, default=0.10)
    pp.add_argument("--particle-diameter", type=float, default=0.004)
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--dt", type=float, default=1e-4)
    pp.set_defaults(func=_cmd_pipe)

    vf = sub.add_parser("verify", help="run the full acceptance study")
    vf.add_argument("--outdir", default="artifacts/p176w2-n11")
    vf.add_argument("--quick", action="store_true")
    vf.set_defaults(func=_cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    payload = json.dumps(_jsonify(result), indent=2, default=str)
    print(payload)
    if args.out:
        import pathlib

        pathlib.Path(args.out).write_text(payload + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())