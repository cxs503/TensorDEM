"""One-way FEM -> DEM fracture handoff: driving the dynamic chain from a script.

Physical scenario
-----------------
A 10 m x 10 m, 0.5 m thick level ice sheet floats on a Winkler foundation
(``rho_w g``) and is loaded by two contact patches.  The contact pressure ramps at
``qdot`` Pa/s; the Kirchhoff-plate FEM of wave 1 finds the fracture state
(``sigma_max -> sigma_flexure``), the crack tracer partitions the sheet into
floes, and wave 2 hands the DEM rigid initial conditions:

* position: the floe mass centroid (from its FEM cells),
* mass / area: exactly the floe mass / area (rigid fit conserves both),
* vertical velocity: ``qdot * w_hat`` averaged over the floe (translation),
* tilt rate: the principal-axis rotation rates ``omega_1, omega_2``,
* deformational energy the rigid body cannot carry: reported, not smuggled.

The script is the "platform-side caller": it only uses the public package API.
It writes nothing unless ``--out`` is given.

Usage
-----
    python examples/fem_dem_fracture_dynamic/run.py
    python examples/fem_dem_fracture_dynamic/run.py --out /tmp/handoff
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensordem.fem_dem_fracture import (          # noqa: E402
    IcePlateCase,
    dem_initial_conditions_dynamic,
    dem_intake,
    read_dynamic_failure_faces,
    run_dynamic_handoff,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="directory for the artefact bundle (default: none)")
    parser.add_argument("--pressure-rate", type=float, default=1000.0,
                        help="contact-pressure ramp rate [Pa/s]")
    parser.add_argument("--mesh", type=int, default=40, help="FEM cells per side")
    args = parser.parse_args(argv)

    case = IcePlateCase(nx=args.mesh, ny=args.mesh,
                        length_x=10.0, length_y=10.0)
    if args.out is None:
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        out = tmp.name
    else:
        tmp = None
        out = args.out

    summary = run_dynamic_handoff(case, out, pressure_rate_pa_s=args.pressure_rate,
                                  tag="example", make_figures=False,
                                  heave_periods=2.0)
    dynamic = read_dynamic_failure_faces(summary["artifacts"]["failure_faces_dynamic"])
    ic = dem_initial_conditions_dynamic(dynamic)
    intake = dem_intake(dynamic, heave_periods=3.0)

    print("== FEM -> DEM fracture handoff (dynamic) ==")
    print(f"ice plate          : {case.length_x:.1f} x {case.length_y:.1f} m, "
          f"t = {case.material.thickness:.2f} m, "
          f"sigma_flexure = {case.material.sigma_flexure/1e6:.3f} MPa")
    print(f"pressure ramp      : {args.pressure_rate:g} Pa/s, fracture at "
          f"{summary['fracture_pressure_pa']:.6g} Pa")
    print(f"floes              : {len(dynamic.faces.floes)}")
    print(f"mass  (FEM/DEM)    : {dynamic.ledger.mass_field:.6f} / "
          f"{ic.total_mass:.6f} kg  (residual "
          f"{dynamic.ledger.mass_residual:.3e} kg)")
    print(f"momentum residual  : {dynamic.ledger.momentum_residual:.3e} kg m/s")
    print(f"angular residual   : "
          f"{float(abs(dynamic.ledger.angular_residual).max()):.3e} kg m^2/s")
    print(f"KE split residual  : {dynamic.ledger.ke_sum_residual:.3e} J "
          f"(rigid fraction {dynamic.ledger.rigid_fraction:.9f})")
    print(f"quasi-static check : KE/U = "
          f"{summary['quasi_static']['ke_over_elastic']:.3e} at {args.pressure_rate:g} "
          f"Pa/s, critical rate {summary['quasi_static']['critical_rate_pa_s']:.3g} Pa/s")
    print(f"DEM intake         : {intake['n_bodies']} bodies, seeded mass residual "
          f"{intake['seeded_ledger']['mass_residual_kg']:.3e} kg, heave period error "
          f"{intake['heave']['max_period_error']:.3e}")
    density = case.material.density
    thickness = case.material.thickness
    print("transfer per floe  : area, equivalent diameter, inherited motion")
    for k in dynamic.kinematics:
        print(f"  floe {k.floe_id:2d}  A = {k.mass/(density*thickness):7.3f} m^2  "
              f"d = {k.equivalent_diameter:5.2f} m  v_z = {k.v_z:+.4e} m/s  "
              f"omega_x = {k.omega_x:+.3e} rad/s  omega_y = {k.omega_y:+.3e} rad/s")
    print("\ncriteria:", json.dumps(
        {name: block["pass"] for name, block in summary["criteria"].items()
         if isinstance(block, dict) and "pass" in block}, sort_keys=True))
    print("artefacts:", "written to " + out if args.out else "(temporary)")
    failures = [n for n, b in summary["criteria"].items()
                if isinstance(b, dict) and "pass" in b and not b["pass"]]
    if tmp is not None:
        tmp.cleanup()
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())