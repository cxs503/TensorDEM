"""m16340 -- stability of the viscous rolling-resistance branch (verification harness).

The honest FAIL carried by ``docs/clump_rolling_resistance_calibration.md`` §3.4
says the ``"viscous"`` branch is *numerically unstable* at its own default
``rolling_omega_ref = 1e-3`` rad/s and ``dt = 2e-4`` s, and that the 5.74 deg
"threshold" it produced is an artefact.  This harness replaces that observation
with a quantified, reproducible verification surface.

What is measured
----------------
1. **Guard behaviour** -- the configuration matrix on which the explicit rate
   law is refused (``ContactConfig.rolling_stability_guard``), with the analytic
   window ``dt*M_lim/(I*omega_ref) <= 2``.
2. **Free spin-down** -- an isolated spinning sphere on a flat plane, where the
   rolling rate obeys ``d(omega)/dt = -lambda omega`` with
   ``lambda = mu_r |F_n| r / (omega_ref I)``, so ``omega(t) = omega0 exp(-lambda t)``
   is an exact reference.  First-order convergence of the scheme inside the
   stability window is measured, and the diverging regime is reproduced.
3. **Terminal rolling on an incline** -- the closed form
   ``omega* = omega_ref tan(theta) / mu_r`` (mass and radius cancel) is a DC /
   steady-state anchor; it separates "stable" from "still the declared law".
4. **Energy bookkeeping** -- dissipated rolling work vs the kinetic-energy drop.
   Both branches are compared with the three quadrature rules (rectangle on the
   commit evaluation, trapezoid, model-exact factor), which is what the
   accounting fix in ``ClumpContact._resistance`` rests on.
5. **Stability boundary localisation** -- bisection on ``lambda*dt`` for three
   geometries, checking that the empirical boundary sits at 2 and is independent
   of mass (``I ∝ m`` and ``M_lim ∝ m`` cancel).

Run::

    PYTHONPATH=src python scripts/m16340_viscous_rolling_stability.py
    PYTHONPATH=src python scripts/m16340_viscous_rolling_stability.py --figures-only

Outputs: ``artifacts/m16340/viscous_rolling_stability.json`` and
``artifacts/m16340/viscous_rolling_stability.png``.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
import pathlib

import numpy as np

from tensordem.clump import Clump, SphereMember
from tensordem.clump_dynamics import ClumpBody, ContactConfig, PlaneContact
from tensordem.clump_rolling import viscous_damping_gain, viscous_stability_dt

OUTDIR = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "m16340"
GRAVITY = 9.81


@dataclass(frozen=True)
class Sheet:
    """Sphere-on-plane testbed (one load-bearing member)."""

    radius: float = 0.05
    density: float = 900.0

    @property
    def clump(self) -> Clump:
        return Clump([SphereMember((0.0, 0.0, 0.0), self.radius)], density=self.density)

    def inertia(self) -> float:
        return float(self.clump.inertia_about_com[0, 0])

    def normal_force(self) -> float:
        return float(self.clump.mass * GRAVITY)

    def rate(self, mu_r: float, omega_ref: float) -> float:
        """Analytic rate constant ``lambda = M_lim / (omega_ref I)`` [1/s]."""
        limit = mu_r * self.normal_force() * self.radius
        return limit / (omega_ref * self.inertia())


def _spin_down(sheet: Sheet, *, omega_ref: float, dt: float, steps: int,
               implicit: bool, guard: bool = True, mu_r: float = 0.9,
               omega0: float = 1.0) -> tuple[float, float, float]:
    """Return ``(omega_end, analytic_omega, dissipated_work)`` after ``steps``."""
    clump = sheet.clump
    config = ContactConfig(normal_stiffness=1.0e5, restitution=0.5, friction=0.0,
                          rolling_friction=mu_r, rolling_model="viscous",
                          rolling_omega_ref=omega_ref,
                          rolling_viscous_implicit=implicit,
                          rolling_stability_guard=guard)
    plane = PlaneContact((0.0, 0.0, 1.0), config=config)
    penetration = sheet.normal_force() / 1.0e5
    body = ClumpBody(clump, position=(0.0, 0.0, sheet.radius - penetration),
                     angular_velocity=(omega0, 0.0, 0.0), gravity=(0.0, 0.0, -GRAVITY))
    body.run(steps, dt, (plane,))
    lam = sheet.rate(mu_r, omega_ref)
    return (float(body.angular_velocity[0]), omega0 * math.exp(-lam * steps * dt),
            float(plane.moment_work()))


def _incline(sheet: Sheet, *, theta_deg: float, mu_r: float, omega_ref: float,
             dt: float, steps: int, implicit: bool) -> dict[str, float]:
    """Terminal rolling on an incline; analytic ``omega* = omega_ref tan/mu_r``."""
    theta = math.radians(theta_deg)
    normal = np.array([math.sin(theta), 0.0, math.cos(theta)])
    config = ContactConfig(normal_stiffness=1.0e5, restitution=0.5, friction=0.6,
                          rolling_friction=mu_r, rolling_model="viscous",
                          rolling_omega_ref=omega_ref,
                          rolling_viscous_implicit=implicit)
    plane = PlaneContact(normal, config=config)
    penetration = sheet.normal_force() * math.cos(theta) / 1.0e5
    body = ClumpBody(sheet.clump, position=normal * (sheet.radius - penetration),
                     gravity=(0.0, 0.0, -GRAVITY))
    body.run(steps, dt, (plane,))
    omega = float(body.angular_velocity[1])
    downslope = np.array([math.cos(theta), 0.0, -math.sin(theta)])
    speed = float(body.velocity @ downslope)
    analytic = omega_ref * math.tan(theta) / mu_r
    return {"omega_rad_s": omega, "omega_analytic_rad_s": analytic,
            "omega_rel_error": (omega / analytic - 1.0) if analytic else float("nan"),
            "speed_m_s": speed, "speed_analytic_m_s": analytic * sheet.radius,
            "no_slip_rel_error": (speed / (analytic * sheet.radius) - 1.0) if analytic else float("nan"),
            "lambda_dt": sheet.rate(mu_r, omega_ref) * dt}


def _diverges(sheet: Sheet, *, omega_ref: float, mu_r: float, dt: float,
              steps: int = 400) -> bool:
    """True when the explicit branch leaves the bounded regime within ``steps``."""
    try:
        omega, _, _ = _spin_down(sheet, omega_ref=omega_ref, dt=dt, steps=steps,
                                 implicit=False, guard=False, mu_r=mu_r)
    except (ValueError, OverflowError, FloatingPointError):
        return True
    return (not math.isfinite(omega)) or abs(omega) > 2.0


def boundary_scan(sheet: Sheet, *, mu_r: float = 0.9, omega_ref: float = 0.1) -> dict:
    """Bisect on ``lambda*dt`` for the empirical divergence boundary."""
    lam = sheet.rate(mu_r, omega_ref)
    lo, hi = 1.0, 8.0
    assert not _diverges(sheet, omega_ref=omega_ref, mu_r=mu_r, dt=lo / lam)
    assert _diverges(sheet, omega_ref=omega_ref, mu_r=mu_r, dt=hi / lam)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if _diverges(sheet, omega_ref=omega_ref, mu_r=mu_r, dt=mid / lam):
            hi = mid
        else:
            lo = mid
    return {"lambda_dt_boundary": 0.5 * (lo + hi), "lower_ok": lo, "upper_diverges": hi,
            "radius_m": sheet.radius, "density_kg_m3": sheet.density,
            "mass_kg": float(sheet.clump.mass), "inertia_kg_m2": sheet.inertia()}


def quadrature_errors(xs: list[float]) -> list[dict]:
    """Relative error of the three work quadratures for pure exponential decay.

    Exact: ``W = 0.5 I w0^2 (1 - e^{-2x})``.  The legacy rule integrated the
    *commit* power (the already decayed rate) over the full step, the trapezoid
    uses both evaluations, and the shipped rule uses the model's own one-step
    factor ``F = (1 - x/2)^2``.
    """
    rows = []
    for x in xs:
        exact = 1.0 - math.exp(-2.0 * x)                       # per unit 0.5 I w0^2
        rectangle = 2.0 * x * math.exp(-2.0 * x)
        trapezoid = x * (1.0 + math.exp(-2.0 * x))
        model = 1.0 - (1.0 - 0.5 * x) ** 4
        rows.append({"lambda_dt": x,
                     "rectangle_rel_error": rectangle / exact - 1.0,
                     "trapezoid_rel_error": trapezoid / exact - 1.0,
                     "model_factor_rel_error": model / exact - 1.0})
    return rows


def run_harness() -> dict:
    sheet = Sheet()
    lam = sheet.rate(0.9, 1.0e-3)
    report: dict[str, object] = {
        "case": "viscous rolling resistance, sphere on plane",
        "sheet": {"radius_m": sheet.radius, "density_kg_m3": sheet.density,
                  "mass_kg": float(sheet.clump.mass),
                  "inertia_kg_m2": sheet.inertia(),
                  "normal_force_N": sheet.normal_force()},
        "default_config": {"mu_r": 0.9, "omega_ref_rad_s": 1.0e-3, "dt_s": 2.0e-4,
                           "lambda_1_s": lam, "lambda_dt": lam * 2.0e-4},
    }

    # 1. guard matrix ------------------------------------------------------
    guard_rows = []
    for omega_ref, dt in ((1.0e-3, 2.0e-4), (1.0e-3, 1.0e-4), (0.1, 2.0e-4),
                          (0.1, 1.0e-3), (1.0, 2.0e-4)):
        window = viscous_stability_dt(sheet.inertia(), sheet.normal_force(),
                                      sheet.radius, 0.9, omega_ref)
        try:
            _spin_down(sheet, omega_ref=omega_ref, dt=dt, steps=200, implicit=False)
            verdict = "ran"
        except ValueError as exc:
            verdict = "refused"
            assert "stability window" in str(exc)
        guard_rows.append({"omega_ref_rad_s": omega_ref, "dt_s": dt,
                           "lambda_dt": sheet.rate(0.9, omega_ref) * dt,
                           "guard": verdict,
                           "dt_monotone_s": window["dt_monotone"],
                           "dt_divergence_s": window["dt_divergence"]})
    report["guard_matrix"] = guard_rows

    # 2. free spin-down: convergence inside the window, divergence outside
    decay_rows = []
    for lambda_dt in (0.01, 0.05, 0.1, 0.25, 0.5, 0.9):
        omega_ref = 0.1
        dt = lambda_dt / sheet.rate(0.9, omega_ref)
        steps = max(1, int(round(math.log(1.0 / 0.35) / lambda_dt)))
        for implicit in (False, True):
            omega, analytic, work = _spin_down(sheet, omega_ref=omega_ref, dt=dt,
                                               steps=steps, implicit=implicit)
            decay_rows.append({
                "lambda_dt": lambda_dt, "steps": steps, "implicit": implicit,
                "omega": omega, "omega_analytic": analytic,
                "rel_error": (omega / analytic - 1.0) if analytic > 1e-12 else None,
                "work_J": work,
                "dke_J": 0.5 * sheet.inertia() * (1.0 - omega * omega),
            })
    report["free_decay"] = decay_rows

    divergence_rows = []
    for omega_ref, dt in ((1.0e-3, 2.0e-4), (1.0e-3, 1.0e-4), (0.1, 1.0e-3)):
        row = {"omega_ref_rad_s": omega_ref, "dt_s": dt,
               "lambda_dt": sheet.rate(0.9, omega_ref) * dt,
               "explicit_diverges": _diverges(sheet, omega_ref=omega_ref, mu_r=0.9, dt=dt)}
        try:
            omega, analytic, work = _spin_down(sheet, omega_ref=omega_ref, dt=dt,
                                               steps=200, implicit=True)
            row.update({"implicit_omega": omega, "implicit_work_J": work,
                        "implicit_finite": math.isfinite(omega),
                        "implicit_amplitude": abs(omega)})
        except (ValueError, OverflowError) as exc:  # pragma: no cover - safety net
            row.update({"implicit_error": repr(exc)})
        divergence_rows.append(row)
    report["stiff_regime"] = divergence_rows

    # 3. terminal rolling on an incline (DC anchor)
    incline_rows = []
    for theta_deg, mu_r, omega_ref, dt, steps in ((10.0, 0.3, 0.1, 2.0e-4, 4000),
                                                  (5.0, 0.5, 0.05, 2.0e-4, 4000),
                                                  (10.0, 0.3, 0.1, 2.0e-3, 4000),
                                                  (10.0, 0.3, 1.0e-3, 2.0e-4, 4000)):
        for implicit in (False, True):
            row = {"theta_deg": theta_deg, "mu_r": mu_r, "omega_ref_rad_s": omega_ref,
                   "dt_s": dt, "implicit": implicit}
            try:
                row.update(_incline(sheet, theta_deg=theta_deg, mu_r=mu_r,
                                    omega_ref=omega_ref, dt=dt, steps=steps,
                                    implicit=implicit))
            except ValueError as exc:
                row.update({"refused": "stability window", "detail": str(exc)[:120]})
            incline_rows.append(row)
    report["terminal_rolling"] = incline_rows

    # 4. work quadratures (analytic) + measured consistency
    report["quadrature_errors"] = quadrature_errors([0.01, 0.05, 0.1, 0.5, 0.88])
    consistency = []
    for lambda_dt in (0.01, 0.1, 0.5):
        dt = lambda_dt / sheet.rate(0.9, 0.1)
        steps = max(1, int(round(math.log(1.0 / 0.35) / lambda_dt)))
        omega, _, work = _spin_down(sheet, omega_ref=0.1, dt=dt, steps=steps, implicit=False)
        dke = 0.5 * sheet.inertia() * (1.0 - omega * omega)
        consistency.append({"lambda_dt": lambda_dt, "work_J": work, "dke_J": dke,
                            "rel_error": work / dke - 1.0 if dke > 0 else None})
    report["work_vs_kinetic_energy"] = consistency

    # 5. stability boundary vs geometry
    report["boundary_scan"] = [boundary_scan(Sheet(radius=0.05, density=900.0)),
                               boundary_scan(Sheet(radius=0.02, density=900.0)),
                               boundary_scan(Sheet(radius=0.05, density=2100.0))]

    report["gain_limits"] = {
        "explicit_gain_at_default": viscous_damping_gain(1.0, 2.0e-4, None),
        "implicit_gain_at_default": viscous_damping_gain(1.0, 2.0e-4, sheet.inertia()),
        "note": "implicit gain g = gamma/(1+lambda*dt/2) <= gamma; steady state of "
                "the gain-limited scheme is T/g, i.e. inflated by 1+lambda*dt/2",
    }
    return report


def make_figure(report: dict, path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sheet = Sheet()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    # (a) free decay vs exact, inside the window
    ax = axes[0]
    omega_ref, dt = 0.1, 0.5 / sheet.rate(0.9, 0.1)
    lam = sheet.rate(0.9, omega_ref)
    steps = 40
    times = np.arange(steps + 1) * dt
    ax.plot(times, np.exp(-lam * times), "k-", lw=2.0, label="exact $e^{-\\lambda t}$")
    for implicit, style, label in ((False, "o", "explicit (declared law)"),
                                   (True, "s", "gain-limited (implicit)")):
        series = [1.0]
        clump = sheet.clump
        config = ContactConfig(normal_stiffness=1.0e5, restitution=0.5, friction=0.0,
                              rolling_friction=0.9, rolling_model="viscous",
                              rolling_omega_ref=omega_ref,
                              rolling_viscous_implicit=implicit)
        plane = PlaneContact((0.0, 0.0, 1.0), config=config)
        body = ClumpBody(clump, position=(0.0, 0.0, sheet.radius - sheet.normal_force() / 1.0e5),
                         angular_velocity=(1.0, 0.0, 0.0), gravity=(0.0, 0.0, -GRAVITY))
        for _ in range(steps):
            body.run(1, dt, (plane,))
            series.append(float(body.angular_velocity[0]))
        ax.plot(times, series, style, ms=3.0, label=label)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\omega_x/\omega_0$")
    ax.set_title(f"(a) spin-down, $\\lambda\\Delta t$ = {lam*dt:.2f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) stiff regime: explicit diverges, implicit stays bounded
    ax = axes[1]
    omega_ref, dt = 1.0e-3, 2.0e-4
    for implicit, style, label in ((False, "o", "explicit (guard off)"),
                                   (True, "s", "gain-limited (implicit)")):
        clump = sheet.clump
        config = ContactConfig(normal_stiffness=1.0e5, restitution=0.5, friction=0.0,
                              rolling_friction=0.9, rolling_model="viscous",
                              rolling_omega_ref=omega_ref,
                              rolling_viscous_implicit=implicit,
                              rolling_stability_guard=False)
        plane = PlaneContact((0.0, 0.0, 1.0), config=config)
        body = ClumpBody(clump, position=(0.0, 0.0, sheet.radius - sheet.normal_force() / 1.0e5),
                         angular_velocity=(1.0, 0.0, 0.0), gravity=(0.0, 0.0, -GRAVITY))
        series = [1.0]
        for _ in range(12):
            try:
                body.run(1, dt, (plane,))
                series.append(abs(float(body.angular_velocity[0])))
            except (ValueError, OverflowError):
                series.append(np.nan)
        ax.semilogy(np.arange(len(series)), series, style, ms=3.0, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$|\omega_x|$ [rad/s]")
    ax.set_title(f"(b) default $\\omega_{{ref}}$=1e-3: $\\lambda\\Delta t$ = "
                 f"{Sheet().rate(0.9,1e-3)*2e-4:.0f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) DC (steady-state) error of the gain-limited branch vs lambda*dt
    ax = axes[2]
    xs = np.linspace(0.0, 3.0, 200)
    ax.plot(xs, xs / 2.0 * 100.0, "k-", lw=2.0, label=r"analytic $1+\lambda h$ overshoot")
    rows = [r for r in report["terminal_rolling"] if "omega_rel_error" in r]
    for implicit, style, label in ((False, "o", "explicit"), (True, "s", "gain-limited")):
        pts = [(r["lambda_dt"], 100.0 * r["omega_rel_error"])
               for r in rows if r["implicit"] is implicit]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, ms=5.0, label=label)
    ax.axhspan(-3.0, 3.0, color="green", alpha=0.12, label="3 % band")
    ax.set_xlabel(r"$\lambda\Delta t$ at the terminal state")
    ax.set_ylabel("steady rolling-rate error [%]")
    ax.set_title("(c) incline terminal velocity vs closed form")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    print(f"figure -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        payload = json.loads((OUTDIR / "viscous_rolling_stability.json").read_text())
        make_figure(payload, OUTDIR / "viscous_rolling_stability.png")
        return

    report = run_harness()
    (OUTDIR / "viscous_rolling_stability.json").write_text(json.dumps(report, indent=2))
    print(f"json -> {OUTDIR / 'viscous_rolling_stability.json'}")
    for row in report["guard_matrix"]:
        print(f"  guard  omega_ref={row['omega_ref_rad_s']:<8g} dt={row['dt_s']:<9g} "
              f"lambda*dt={row['lambda_dt']:>9.3g} -> {row['guard']}")
    for row in report["free_decay"]:
        err = row["rel_error"]
        print(f"  decay  lambda*dt={row['lambda_dt']:<6g} implicit={row['implicit']!s:<5} "
              f"rel_err={'n/a' if err is None else f'{err*100:+.3f}%'}")
    for row in report["terminal_rolling"]:
        if "omega_rel_error" in row:
            print(f"  incline theta={row['theta_deg']:g} omega_ref={row['omega_ref_rad_s']:g} "
                  f"dt={row['dt_s']:g} implicit={row['implicit']!s:<5} "
                  f"err={row['omega_rel_error']*100:+.4f}%")
        else:
            print(f"  incline theta={row['theta_deg']:g} omega_ref={row['omega_ref_rad_s']:g} "
                  f"dt={row['dt_s']:g} implicit={row['implicit']!s:<5} refused")
    for row in report["boundary_scan"]:
        print(f"  boundary r={row['radius_m']} rho={row['density_kg_m3']} "
              f"m={row['mass_kg']:.4g} -> lambda*dt = {row['lambda_dt_boundary']:.4f}")
    if not args.json_only:
        make_figure(report, OUTDIR / "viscous_rolling_stability.png")


if __name__ == "__main__":
    main()