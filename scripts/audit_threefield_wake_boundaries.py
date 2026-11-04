"""Audit the bounded-domain boundary model of the D3Q19 propeller-race solver.

Closes the coverage gap left by ``docs/threefield_lbm_transport_verification.md``
(m2949): the interior operators were verified on a periodic box, the *bounded box*
that the ice/race chain actually runs was not.

Checks
------
A. uniform stream is an exact fixed point of collision + streaming + inlet +
   outlet + specular walls (zero drift reference),
B. specular walls are impermeable and frictionless (exact zero references),
C. open-boundary mass books and the fill time in flow-throughs,
D. the published propeller-race station table (`THREEFIELD_ICE_HULL_PROPELLER.md`)
   reproduced with the current code, at the published averaging protocol and at
   2.3 flow-throughs,
E. the ``tau0 = 1.4`` shear-wave honest-FAIL of m2949 re-measured on a finer
   lattice (resolution attribution).

Usage::

    PYTHONPATH=src OMP_NUM_THREADS=8 python scripts/audit_threefield_wake_boundaries.py

Writes ``docs/threefield-wake-boundaries/{audit.json,ledger.csv,figures/*.png}`` and
returns a non-zero exit code if any exactness or tolerance check fails.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

from tensordem.threefield import boundary_verification as bv
from tensordem.threefield.lbm_transport import shear_wave_decay, shear_wave_rate
from tensordem.threefield.wake_lbm import LBMConfig, WakeLBM

OUT = Path("docs/threefield-wake-boundaries")
FIG = OUT / "figures"

# The propeller-race case actually published in docs/THREEFIELD_ICE_HULL_PROPELLER.md
# (scripts/run_threefield_coupling.py::_lbm_config).
PUBLISHED_CASE = dict(
    nx=72,
    ny=36,
    nz=36,
    tau0=0.5005,
    smagorinsky_cs=0.14,
    inflow_velocity=0.04,
    jet_velocity=0.12,
    disk_x=8,
    disk_radius=5.0,
    disk_thickness=3,
    disk_profile="gaussian",
    disk_perturbation=0.15,
    quarter_domain=True,
    dtype="float32",
)
PUBLISHED_STEPS = 2100
PUBLISHED_WARMUP = 1260          # averaging starts at 0.6 * total_steps
PUBLISHED_STATIONS = (21, 31, 41)
PUBLISHED_DOC = {
    # station: (sigma_fit, R^2 core, excess momentum flux, printed Uc)
    "21": {"sigma_fit": 2.63, "r_squared": 0.9986, "excess_momentum_flux": 0.0842, "Uc": 0.130},
    "31": {"sigma_fit": 2.53, "r_squared": 0.9990, "excess_momentum_flux": 0.0770, "Uc": 0.127},
    "41": {"sigma_fit": 2.40, "r_squared": 0.9948, "excess_momentum_flux": 0.0753, "Uc": 0.124},
}
PUBLISHED_THRUST = 0.0707

LITERATURE_SPREADING_RATE = 0.085


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _station_record(inst: WakeLBM, stations: tuple[int, ...], label: str) -> dict:
    """Station table: resolved width, core fit quality, mass and excess momentum flux."""
    thrust = inst.thrust_lattice()
    rows = {}
    for station in stations:
        profile = inst.axial_velocity_profile(station)
        fit = inst.gaussian_fit(station)
        excess = inst.momentum_flux_excess(station)
        rows[str(station)] = {
            "steps": inst.step_count,
            "averaged_samples": inst._samples,
            "mass_flux": inst.mass_flux(station),
            "excess_momentum_flux": excess,
            "excess_over_thrust": excess / thrust if thrust else float("nan"),
            "thrust": thrust,
            "sigma_fit": fit["sigma"],
            "log_r_squared": fit["log_r_squared"],
            "centreline_velocity": max(profile["mean_axial_velocity"]),
            "axial_rms_velocity": max(profile["axial_rms_velocity"]),
            "sigma_model_literature": LITERATURE_SPREADING_RATE
            * (station - (PUBLISHED_CASE["disk_x"] + PUBLISHED_CASE["disk_thickness"]))
            + 0.5 * PUBLISHED_CASE["disk_radius"],
        }
    return {"label": label, "stations": rows}


def run_station_protocol() -> tuple[dict, dict, dict]:
    """Published protocol (2100 steps) plus a 2.3-flow-through re-audit (4200 steps)."""
    config = LBMConfig(**PUBLISHED_CASE)
    inst = WakeLBM(config)
    inst.step(PUBLISHED_WARMUP)
    inst.step(PUBLISHED_STEPS - PUBLISHED_WARMUP, accumulate=True)
    published = _station_record(inst, PUBLISHED_STATIONS, "published protocol (2100 steps)")
    walls_published = asdict(bv.wall_slip_audit(inst))
    # Keep the same averaging-window length but start it at 3360 = 1.87 flow-throughs
    # so the 4200-step record is not contaminated by the published window.
    inst._mean_u.zero_()
    inst._mean_u2.zero_()
    inst._samples = 0
    inst.step(3360 - PUBLISHED_STEPS)
    inst.step(4200 - 3360, accumulate=True)
    extended = _station_record(inst, PUBLISHED_STATIONS, "2.3 flow-throughs (4200 steps)")
    walls_extended = asdict(bv.wall_slip_audit(inst))
    return published, extended, {"published": walls_published, "extended": walls_extended}


def shear_wave_resolution() -> dict:
    """Re-measure the m2949 tau0 = 1.4 honest-FAIL on a finer lattice."""
    sizes = (16, 32)
    taus = (0.55, 0.6, 0.7, 0.8, 1.0, 1.4)
    rows = []
    for size in sizes:
        for tau0 in taus:
            result = shear_wave_decay(tau0=tau0, size=size, mode=1, steps=400, amplitude=1e-3)
            rate_cont, rate_disc = shear_wave_rate(tau0, 2.0 * math.pi / size)
            rows.append(
                {
                    "size": size,
                    "tau0": tau0,
                    "nu_nominal": (tau0 - 0.5) / 3.0,
                    "wavenumber": 2.0 * math.pi / size,
                    "rate_measured": result.rate_measured,
                    "rate_continuum": rate_cont,
                    "rate_discrete": rate_disc,
                    "error_vs_continuum_pct": 100.0 * (result.rate_measured / rate_cont - 1.0),
                    "error_vs_discrete_pct": 100.0 * (result.rate_measured / rate_disc - 1.0),
                }
            )
    return {"rows": rows}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(ledger_payload: dict, published: dict, extended: dict, shear: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    written = []

    samples = ledger_payload["samples"]
    transit = ledger_payload["transit_steps"]
    steps = [row["step"] / transit for row in samples]
    ratio = [row["flux_out"] / row["flux_in"] for row in samples]
    mass = [row["mass"] / samples[0]["mass"] - 1.0 for row in samples]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].plot(steps, ratio, lw=1.2)
    axes[0].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[0].axvline(1.0, color="tab:red", ls=":", lw=0.8,
                    label="published run length (1.17 transits)")
    axes[0].set_xlabel("flow-throughs  $t U_{in}/L$")
    axes[0].set_ylabel("$J_{out}/J_{in}$")
    axes[0].set_title("inlet/outlet mass-flux balance")
    axes[0].legend(fontsize=7)
    axes[1].plot(steps, mass, lw=1.2, color="tab:green")
    axes[1].set_xlabel("flow-throughs  $t U_{in}/L$")
    axes[1].set_ylabel("$M(t)/M_0 - 1$")
    axes[1].set_title("box mass (open-boundary books)")
    fig.tight_layout()
    path = FIG / "flux_ledger.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    stations = sorted(int(key) for key in published["stations"])
    for record, colour, marker in ((published, "tab:blue", "o"), (extended, "tab:orange", "s")):
        rows = record["stations"]
        axes[0].plot(stations, [rows[str(s)]["excess_over_thrust"] for s in stations],
                     marker=marker, color=colour, label=record["label"])
        reference = [rows[str(s)]["mass_flux"] for s in stations]
        axes[1].plot(stations, [100.0 * (value / reference[0] - 1.0) for value in reference],
                     marker=marker, color=colour, label=record["label"])
    axes[0].axhline(1.0, color="k", ls="--", lw=0.8, label="thrust = excess momentum flux")
    axes[0].set_xlabel("station $x$ [lattice units]")
    axes[0].set_ylabel("excess momentum flux / thrust")
    axes[0].set_title("momentum book at the race stations")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0.0, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("station $x$ [lattice units]")
    axes[1].set_ylabel("mass flux deviation from station 21 [%]")
    axes[1].set_title("axial mass-flux constancy")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    path = FIG / "station_audit.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    rows = shear["rows"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for size, colour in ((16, "tab:blue"), (32, "tab:orange")):
        subset = [row for row in rows if row["size"] == size]
        ax.plot([row["tau0"] for row in subset],
                [row["error_vs_continuum_pct"] for row in subset],
                marker="o", color=colour, label=f"N = {size}")
    ax.axhspan(-3.0, 3.0, color="tab:green", alpha=0.12, label="3 % band")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"relaxation time $\tau_0$")
    ax.set_ylabel(r"decay-rate error vs $\nu k^2$ [%]")
    ax.set_title("shear-wave diffusivity: lattice-resolution error")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIG / "shear_wave_resolution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))
    return written


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict = {"published_case": PUBLISHED_CASE, "published_steps": PUBLISHED_STEPS}
    failures: list[str] = []

    # A. uniform stream fixed point
    uniform = []
    for label, tau0, cs in (
        ("laminar tau0=0.6", 0.6, 0.0),
        ("sgs tau0=0.5005 Cs=0.14", 0.5005, 0.14),
    ):
        result = bv.uniform_stream_check(tau0=tau0, smagorinsky_cs=cs, steps=200)
        record = asdict(result)
        record["label"] = label
        record["exact"] = result.exact
        uniform.append(record)
        if not result.exact:
            failures.append(f"uniform stream drift at {label}: {record}")
    payload["uniform_stream"] = uniform
    print("A uniform-stream drift:",
          [(row["label"], row["velocity_drift"]) for row in uniform])

    # B/C. published case: station protocol + open-boundary ledger
    published, extended, walls = run_station_protocol()
    payload["stations"] = {"published": published, "extended": extended}
    payload["walls"] = walls
    # The published case runs in float32, whose round-off floor is 1.19e-7; the wall
    # residuals are dimensionless ratios, so the tolerance has to follow the dtype.
    wall_tolerance = 1e-6 if PUBLISHED_CASE["dtype"] == "float32" else 1e-13
    for label, record in walls.items():
        if record["normal_velocity"] != 0.0:
            failures.append(f"wall normal velocity at {label}: {record['normal_velocity']}")
        if record["mass_crossing_mismatch"] >= wall_tolerance:
            failures.append(f"wall mass mismatch at {label}: {record}")
        if record["axial_momentum_exchange"] >= wall_tolerance:
            failures.append(f"wall axial momentum exchange at {label}: {record}")
        if record["reflection_tangential_mismatch"] != 0.0:
            failures.append(f"reflection table at {label}: {record}")
    print("B walls:", walls)

    config = LBMConfig(**PUBLISHED_CASE)
    ledger = bv.open_boundary_ledger(config, steps=4200, sample_every=10)
    transit = config.nx / config.inflow_velocity
    fill = {
        f"{int(tol * 100)}pct": bv.fill_time_in_flow_throughs(ledger, tolerance=tol)
        for tol in (0.01, 0.02, 0.05)
    }
    ledger_payload = {
        "steps": ledger.samples[-1].step,
        "transit_steps": transit,
        "thrust_rate": ledger.thrust_rate,
        "mass_drift_rate_relative": ledger.mass_drift_rate,
        "flux_imbalance": ledger.flux_imbalance,
        "flux_imbalance_relative": ledger.flux_imbalance_relative,
        "flux_balance_final": ledger.flux_balance,
        "fill": fill,
        "samples": [asdict(sample) for sample in ledger.samples],
    }
    payload["ledger"] = {key: value for key, value in ledger_payload.items() if key != "samples"}
    if abs(ledger.flux_imbalance_relative) > 0.05:
        failures.append(
            f"inlet/outlet mass-flux imbalance {ledger.flux_imbalance_relative:.3%} > 5 %"
        )
    if abs(ledger.mass_drift_rate) * 4200 > 1e-3:
        failures.append(f"box mass drift {ledger.mass_drift_rate:.3e} per step over 4200 steps")
    print("C ledger:", payload["ledger"])

    (OUT / "ledger.csv").write_text(
        "step,mass,momentum_x,flux_in,flux_out,thrust_cumulative\n"
        + "\n".join(
            f"{s.step},{s.mass:.10g},{s.momentum_x:.10g},{s.flux_in:.10g},"
            f"{s.flux_out:.10g},{s.thrust_cumulative:.10g}"
            for s in ledger.samples
        )
        + "\n"
    )

    # D. compare the published station columns with the reproduction
    comparison = {}
    for station, doc in PUBLISHED_DOC.items():
        row = published["stations"][station]
        extended_row = extended["stations"][station]
        comparison[station] = {
            "doc_sigma_fit": doc["sigma_fit"],
            "reproduced_sigma_fit": row["sigma_fit"],
            "sigma_fit_deviation_pct": 100.0 * (row["sigma_fit"] / doc["sigma_fit"] - 1.0),
            "doc_r_squared": doc["r_squared"],
            "reproduced_r_squared": row["log_r_squared"],
            "doc_excess_momentum_flux": doc["excess_momentum_flux"],
            "reproduced_excess_momentum_flux": row["excess_momentum_flux"],
            "excess_deviation_pct": 100.0
            * (row["excess_momentum_flux"] / doc["excess_momentum_flux"] - 1.0),
            "excess_over_thrust_published_window": row["excess_over_thrust"],
            "excess_over_thrust_extended_window": extended_row["excess_over_thrust"],
        }
    payload["doc_comparison"] = comparison
    print("D reproduction:", json.dumps(comparison, indent=1))

    # E. tau0 = 1.4 resolution close-out
    shear = shear_wave_resolution()
    payload["shear_wave"] = shear
    worst = {
        size: max(
            (abs(row["error_vs_continuum_pct"]) for row in shear["rows"] if row["size"] == size)
        )
        for size in (16, 32)
    }
    payload["shear_wave_worst_error_pct"] = worst
    print("E shear-wave worst error:", worst)

    figures = make_figures(ledger_payload, published, extended, shear)
    payload["figures"] = figures
    payload["wall_tolerance"] = wall_tolerance
    payload["failures"] = failures
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("wrote", OUT / "audit.json")
    if failures:
        print("FAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())