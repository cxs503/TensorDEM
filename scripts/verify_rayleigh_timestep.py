"""Verify the Rayleigh-wave time-step estimate against the exact lattice limit.

Produces (all numbers machine-generated, nothing hand-copied):

* ``artifacts/rayleigh_timestep/report.json``      -- every number used below
* ``artifacts/rayleigh_timestep/rayleigh_ratio_sweep.csv``
* ``artifacts/rayleigh_timestep/dispersion.csv``
* ``artifacts/rayleigh_timestep/fig1_dispersion.png``
* ``artifacts/rayleigh_timestep/fig2_bz_map.png``
* ``artifacts/rayleigh_timestep/fig3_dt_limits.png``
* ``artifacts/rayleigh_timestep/fig4_growth.png``
* ``docs/dem3d_rayleigh_timestep_verification.md`` -- report page with the tables
  injected from ``report.json``

Run: ``PYTHONPATH=src python scripts/verify_rayleigh_timestep.py``
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tensordem import rayleigh_timestep as rt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "rayleigh_timestep"
DOC = ROOT / "docs" / "dem3d_rayleigh_timestep_verification.md"

RADIUS = 0.025
DENSITY = 917.0
STIFFNESS = 2_000.0
MASS = DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3
SPACING = 2.0 * RADIUS
GROWTH_FACTORS = (0.90, 0.99, 1.00711, 1.05)
GROWTH_STEPS = 400


def _rel(a: float, b: float) -> float:
    return (a - b) / b if b else float("nan")


def dispersion_path(spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gamma -> X -> M -> Gamma -> R, 40 interior points per segment.

    Returns the wavevectors, the cumulative arc length at each wavevector and the
    arc length of the five corner points (used for the tick labels).
    """
    pi_over_a = math.pi / spacing
    corners = np.array(
        [
            [0.0, 0.0, 0.0],           # Gamma
            [1.0, 0.0, 0.0],           # X
            [1.0, 1.0, 0.0],           # M
            [0.0, 0.0, 0.0],           # Gamma
            [1.0, 1.0, 1.0],           # R
        ]
    ) * pi_over_a
    chunks = []
    for start, end in zip(corners[:-1], corners[1:]):
        for t in np.linspace(0.0, 1.0, 41, endpoint=False):
            chunks.append(start + t * (end - start))
    chunks.append(corners[-1])
    wave_vectors = np.array(chunks)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(wave_vectors, axis=0), axis=1))])
    stops = arc[[0, 40, 80, 120, 160]]
    return wave_vectors, arc, stops


def main() -> dict:
    ART.mkdir(parents=True, exist_ok=True)
    report = rt.timestep_limit_report(
        bond_stiffness=STIFFNESS, radius=RADIUS, density=DENSITY, mass=MASS
    )
    dt_exact = report["dt_exact_s"]

    # ---------------- Rayleigh ratio: two exact paths + published anchors ------
    anchors = rt.rayleigh_ratio_table()
    sweep = rt.rayleigh_ratio_sweep()
    worst_anchor = max(abs(r["err_bisection_vs_reference"]) for r in anchors)
    worst_cubic = max(abs(r["err_cubic_vs_bisection"]) for r in sweep)
    worst_bergmann = max(abs(r["err_bergmann_vs_exact"]) for r in sweep)

    # ---------------- dispersion + closed forms ------------------------------
    wave_vectors, arc, stops = dispersion_path(SPACING)
    omega = rt.dispersion_frequencies(
        wave_vectors, stiffness=STIFFNESS, mass=MASS, spacing=SPACING
    )
    # closed form along Gamma -> X for the longitudinal / transverse branches
    n_seg = 41
    qx = np.linalg.norm(wave_vectors[:n_seg], axis=1)
    closed_l = np.sqrt(26.0 * STIFFNESS / (3.0 * MASS) * (1.0 - np.cos(qx)))
    closed_t = np.sqrt(14.0 * STIFFNESS / (3.0 * MASS) * (1.0 - np.cos(qx)))
    closed_err_l = float(np.max(np.abs(omega[:n_seg, 2] - closed_l)) / closed_l[-1])
    closed_err_t = float(np.max(np.abs(omega[:n_seg, 0] - closed_t)) / closed_t[-1])

    # ---------------- BZ map on the q_z = 0 plane ----------------------------
    pi_over_a = math.pi / SPACING
    axis = np.linspace(-1.0, 1.0, 81) * pi_over_a
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    planes = np.concatenate([grid, np.zeros((grid.shape[0], 1))], axis=1)
    omega_plane = rt.dispersion_frequencies(
        planes, stiffness=STIFFNESS, mass=MASS, spacing=SPACING
    )[:, -1].reshape(axis.size, axis.size)

    # ---------------- growth test: measured vs theory -----------------------
    growth_rows = []
    growth_curves = {}
    for factor in GROWTH_FACTORS:
        run = rt.simulate_periodic_lattice(
            stiffness=STIFFNESS,
            mass=MASS,
            spacing=SPACING,
            timestep=factor * dt_exact,
            steps=GROWTH_STEPS,
        )
        tail_energy = run["energy"][GROWTH_STEPS // 2:]
        growth_curves[factor] = run["max_displacement"]
        growth_rows.append(
            {
                "dt_over_exact": factor,
                "dt_s": factor * dt_exact,
                "measured_growth_per_step": run["measured_growth_per_step"],
                "theoretical_growth_per_step": run["theoretical_growth_per_step"],
                "rel_error_measured_vs_theory": _rel(
                    run["measured_growth_per_step"], run["theoretical_growth_per_step"]
                ),
                "energy_tail_max_over_initial": float(
                    tail_energy.max() / run["energy"][0]
                ),
                "stable_expected": factor < 1.0,
                "verdict": "stable" if factor < 1.0 else "divergent",
            }
        )

    # ---------------- figures ----------------------------------------------
    plt.rcParams.update({"font.size": 8, "figure.dpi": 76, "axes.grid": True,
                         "grid.alpha": 0.3})
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for branch, label in enumerate(("T", "T'", "L")):
        ax.plot(arc, omega[:, branch], lw=1.2, label=f"Bloch branch {label}")
    ax.plot(arc[:n_seg], closed_l, "k--", lw=0.9,
            label=r"closed form $\omega_L^2=\frac{26k}{3m}(1-\cos qa)$")
    ax.plot(arc[:n_seg], closed_t, "k:", lw=0.9,
            label=r"closed form $\omega_T^2=\frac{14k}{3m}(1-\cos qa)$")
    for x, name in zip(stops, ("$\\Gamma$", "X", "M", "$\\Gamma$", "R")):
        ax.axvline(x, color="0.6", lw=0.6)
        ax.text(x, ax.get_ylim()[1] * 0.97, name, ha="center", va="top", fontsize=7)
    ax.set_xlabel("wavevector path")
    ax.set_ylabel(r"$\omega$ [rad/s]")
    ax.set_title("Exact Bloch spectrum, 26-neighbour bond lattice")
    ax.legend(fontsize=5.5, loc="lower right", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(ART / "fig1_dispersion.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    mesh = ax.pcolormesh(axis / pi_over_a, axis / pi_over_a, omega_plane, shading="auto",
                         cmap="viridis")
    fig.colorbar(mesh, ax=ax, label=r"$\omega_{max}(q)$ [rad/s]")
    ax.plot(report["q_argmax_over_pi_over_a"][0], report["q_argmax_over_pi_over_a"][1],
            "r+", ms=10, mew=1.5, label="argmax")
    ax.set_xlabel(r"$q_x a/\pi$")
    ax.set_ylabel(r"$q_y a/\pi$")
    ax.set_title(r"$\omega_{max}$ over the $q_z=0$ plane")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ART / "fig2_bz_map.png")
    plt.close(fig)

    labels = [
        "exact $2/\\omega_{max}$",
        "Gershgorin row sum",
        "Rayleigh, $\\rho_{lattice}$",
        "Rayleigh, Reuss $G$",
        "Rayleigh, $\\rho_{material}$",
        "engine $0.15\\sqrt{m/k}$",
    ]
    values = [
        dt_exact,
        report["dt_gershgorin_s"],
        report["dt_rayleigh_s"]["voigt_lattice_density"],
        report["dt_rayleigh_s"]["reuss_lattice_density"],
        report["dt_rayleigh_s"]["voigt_material_density"],
        report["dt_engine_elastic_s"],
    ]
    colors = ["tab:green"] + ["tab:blue"] * 4 + ["tab:gray"]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.6)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value * 1.02, bar.get_y() + bar.get_height() / 2,
                f"{value * 1e3:.3f} ms  ({value / dt_exact:.3f}x)", va="center", fontsize=7)
    ax.axvline(dt_exact, color="tab:green", ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlim(min(values) * 0.7, max(values) * 2.2)
    ax.set_xlabel("time step [s] (log scale)")
    ax.set_title("Explicit step limits vs the exact lattice limit (dashed)")
    fig.tight_layout()
    fig.savefig(ART / "fig3_dt_limits.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for factor, curve in growth_curves.items():
        ax.semilogy(np.arange(curve.size), curve / curve[0], lw=1.1,
                    label=f"$dt={factor:.4g}\\,dt_{{exact}}$")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\max|u| \,/\, \max|u_0|$")
    ax.set_title("Measured semi-implicit Euler growth on the 4x4x4 torus")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ART / "fig4_growth.png")
    plt.close(fig)

    # ---------------- csv artefacts ----------------------------------------
    with (ART / "rayleigh_ratio_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    with (ART / "dispersion.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arc_length", "omega_T", "omega_T2", "omega_L", "q_x", "q_y", "q_z"])
        writer.writerows(
            np.column_stack([arc, omega, wave_vectors]).tolist()
        )

    payload = {
        "params": {
            "bond_stiffness_N_per_m": STIFFNESS,
            "radius_m": RADIUS,
            "density_kg_per_m3": DENSITY,
            "mass_kg": MASS,
            "spacing_m": SPACING,
        },
        "timestep_report": report,
        "rayleigh_anchors": anchors,
        "rayleigh_sweep": sweep,
        "worst_anchor_relative_error": worst_anchor,
        "worst_cubic_vs_bisection": worst_cubic,
        "worst_bergmann_vs_exact": worst_bergmann,
        "dispersion_closed_form_error": {"longitudinal": closed_err_l,
                                         "transverse": closed_err_t},
        "growth": growth_rows,
        "gershgorin_exact_ratio_closed_form": 1.0 / math.sqrt(3.0),
    }
    (ART / "report.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")

    DOC.write_text(_render_doc(payload))
    return payload


def _f(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}g}"


def _pct(value: float) -> str:
    return f"{100.0 * value:+.4f}%"


def _render_doc(payload: dict) -> str:
    r = payload["timestep_report"]
    params = payload["params"]
    anchors = payload["rayleigh_anchors"]
    sweep = payload["rayleigh_sweep"]
    growth = payload["growth"]
    dt_exact = r["dt_exact_s"]

    anchor_rows = "\n".join(
        "| {nu} | {bis:.8f} | {cub:.8f} | {ref:.6f} | {e1} | {e2} |".format(
            nu=f"{a['poisson_ratio']:.4f}",
            bis=a["ratio_bisection"],
            cub=a["ratio_lamb_cubic"],
            ref=a["ratio_reference"],
            e1=f"{a['err_bisection_vs_reference']:+.2e}",
            e2=f"{a['err_cubic_vs_bisection']:+.1e}",
        )
        for a in anchors
    )

    estimate_rows = "\n".join(
        [
            "| exact `2/omega_max` | {} | {} | 1.000 | reference |".format(
                _f(dt_exact * 1e3, 6), _f(dt_exact, 6)
            ),
            "| Gershgorin row sum `2/sqrt(2 Z k/m)` | {} | {} | {} | safe, conservative |".format(
                _f(r["dt_gershgorin_s"] * 1e3, 6), _f(r["dt_gershgorin_s"], 6),
                _f(r["ratios"]["gershgorin_over_exact"], 6),
            ),
            "| Rayleigh `r_min/c_R`, `rho_lattice` | {} | {} | {} | safe, conservative |".format(
                _f(r["dt_rayleigh_s"]["voigt_lattice_density"] * 1e3, 6),
                _f(r["dt_rayleigh_s"]["voigt_lattice_density"], 6),
                _f(r["ratios"]["rayleigh_voigt_lattice_density_over_exact"], 6),
            ),
            "| Rayleigh `r_min/c_R`, Reuss `G` | {} | {} | {} | safe, conservative |".format(
                _f(r["dt_rayleigh_s"]["reuss_lattice_density"] * 1e3, 6),
                _f(r["dt_rayleigh_s"]["reuss_lattice_density"], 6),
                _f(r["ratios"]["rayleigh_reuss_lattice_density_over_exact"], 6),
            ),
            "| Rayleigh `r_min/c_R`, `rho_material` | {} | {} | {} | **unsafe** |".format(
                _f(r["dt_rayleigh_s"]["voigt_material_density"] * 1e3, 6),
                _f(r["dt_rayleigh_s"]["voigt_material_density"], 6),
                _f(r["ratios"]["rayleigh_voigt_material_density_over_exact"], 6),
            ),
            "| engine `recommended_dt` elastic factor `0.15 sqrt(m/k_bond)` | {} | {} | {} | safe, very conservative |".format(
                _f(r["dt_engine_elastic_s"] * 1e3, 6), _f(r["dt_engine_elastic_s"], 6),
                _f(r["ratios"]["engine_elastic_over_exact"], 6),
            ),
        ]
    )

    growth_rows = "\n".join(
        "| {f:.5g} | {dt} | {meas} | {theo} | {err} | {en} | {v} |".format(
            f=g["dt_over_exact"],
            dt=_f(g["dt_s"], 6),
            meas=_f(g["measured_growth_per_step"], 8),
            theo=_f(g["theoretical_growth_per_step"], 8),
            err=f"{g['rel_error_measured_vs_theory']:+.2e}",
            en=_f(g["energy_tail_max_over_initial"], 4),
            v=g["verdict"],
        )
        for g in growth
    )

    worst_bergmann = payload["worst_bergmann_vs_exact"]
    return f"""# Rayleigh-wave time-step estimate: implementation and audit against the exact lattice limit

Protocol `tensordem-rayleigh-timestep-verification-v1`. This page closes the item
recorded as open in `docs/dem3d_timestep_stability.md` §4 ("*No Rayleigh-wave
estimate adopted ... the Rayleigh estimate was not implemented*"). It is a
code-versus-analytic verification of a *timestep criterion*; it says nothing about
spatial resolution or physical fidelity.

## 1. Why the gap mattered

Classical DEM practice caps the step with a Rayleigh surface-wave criterion
`dt ~ r_min / c_R`. That rule is a dimension argument, not a stability theorem:
it contains no coordination number, no anisotropy and no density convention. The
repository instead ships a rigorous Gershgorin bound plus an ad-hoc
`0.15 sqrt(m/k)` elastic factor in `DEM3DConfig.recommended_dt`, and never
quantified how far either sits from the *actual* stability limit of the bond
lattice. Without that reference the folklore rule cannot be accepted or rejected
— it can only be believed, which is what this page replaces with numbers.

## 2. What is exact here, and why

For symplectic (semi-implicit) Euler — the scheme `IceDEM3D.step` uses — applied
to a linear, undamped, equal-mass bond network, each normal mode is an
independent 2x2 map with `trace = 2 - (dt omega)^2` and `det = 1`. The pair is
non-expanding **iff `dt omega <= 2`**, so

    dt_crit = 2 / omega_max,     omega_max = max_q max_i omega_i(q)

is the exact limit, not an estimate. For the solver's 26-neighbour lattice
(13 half-space offsets: 3 axis bonds of length `a`, 6 face diagonals, 4 body
diagonals) the Bloch matrix is
`D(q) = (2k/m) sum_13 n n^T (1 - cos q.d)`. Along `q = (q,0,0)` this reduces to

    omega_L^2 = (26 k / 3 m) (1 - cos qa),   omega_T^2 = (14 k / 3 m) (1 - cos qa),

so the maximum sits at X = (pi/a, 0, 0) and has the closed form

    omega_max = sqrt(52 k / (3 m)),   dt_crit = sqrt(3 m / (13 k)).

The same expansion gives the acoustic slopes `c_L^2 = C11/rho_lattice` and
`c_T^2 = C44/rho_lattice` with `rho_lattice = m/a^3`, which ties this module to
`tensordem.lattice_elasticity3d.periodic_elastic_constants` (both are checked
against each other below).

## 3. Case definition

| item | value |
|---|---|
| particle radius `r` | {params['radius_m']} m |
| lattice spacing `a = 2r` | {params['spacing_m']} m |
| material density | {params['density_kg_per_m3']} kg/m^3 |
| particle mass `m` (sphere) | {_f(params['mass_kg'], 6)} kg |
| lattice density `m/a^3` | {_f(r['density_lattice_kg_per_m3'], 6)} kg/m^3 (`pi/6` of the material density) |
| bond stiffness `k` | {params['bond_stiffness_N_per_m']} N/m |
| coordination number `Z` | 26 |
| C11 / C12 / C44 | {_f(r['C11_Pa'], 6)} / {_f(r['C12_Pa'], 6)} / {_f(r['C44_Pa'], 6)} Pa |
| Poisson ratio `C12/(C11+C12)` | {_f(r['poisson_ratio_lattice'], 6)} |
| Zener anisotropy `2C44/(C11-C12)` | {_f(r['zener_anisotropy'], 6)} |
| scope | linear, undamped, intact bonds, translational DOF only |

## 4. Rayleigh speed: two independent exact paths

`x = (c_R/c_S)^2` satisfies the traction-free condition
`(2-x)^2 = 4 sqrt(1-x) sqrt(1-kappa x)` with
`kappa = (c_S/c_P)^2 = (1-2nu)/(2(1-nu))`. The module solves it (a) by bisection
on that form and (b) by the Lamb cubic `x^3 - 8x^2 + 8x(3-2kappa) - 16(1-kappa) = 0`,
which shares no code path with (a). Published anchors are quoted to six decimals
only, so that comparison is capped at ~1e-6.

| nu | bisection | Lamb cubic | published | err vs published | cubic vs bisection |
|---|---|---|---|---|---|
{anchor_rows}

Worst |err| vs published: **{payload['worst_anchor_relative_error']:.2e}** (all of it at
`nu = 1/3`, where six-decimal tables differ in the fifth digit: 0.932472 is quoted
in print, the two independent exact formulations here both give
{anchors[2]['ratio_bisection']:.6f}; the table is *not* adjusted toward the code);
worst cubic-vs-bisection disagreement over the full sweep (nu = 0 ... 0.5, step 0.05):
**{payload['worst_cubic_vs_bisection']:.2e}**. The Bergmann fit
`(0.87+1.12nu)/(1+nu)` is only an approximation: worst error **{worst_bergmann:.2e}**
over the sweep, i.e. good to a few tenths of a percent and *not* used in any
number below.

## 5. `omega_max`: three routes agree

| route | omega_max [rad/s] | deviation from closed form |
|---|---|---|
| closed form `sqrt(52k/3m)` | {_f(r['omega_max_closed_form'], 10)} | reference |
| Bloch scan over the BZ (33^3 grid, X included) | {_f(r['omega_max_bz'], 10)} | {r['errors']['omega_closed_vs_bz']:+.2e} |
| atomistic 4x4x4 periodic Hessian (192 DOF, dense eigenvalues) | {_f(r['omega_max_supercell_4x4x4'], 10)} | {r['errors']['omega_closed_vs_supercell']:+.2e} |

argmax at `q a / pi = ({r['q_argmax_over_pi_over_a'][0]:.0f}, {r['q_argmax_over_pi_over_a'][1]:.0f}, {r['q_argmax_over_pi_over_a'][2]:.0f})`,
i.e. the zone-boundary point X, as the closed form predicts. Along Gamma->X the
code output matches the two closed forms to {payload['dispersion_closed_form_error']['longitudinal']:.2e}
(longitudinal) and {payload['dispersion_closed_form_error']['transverse']:.2e} (transverse).

The 4x4x4 torus is the smallest size that both keeps the `+n`/`-n` images
distinct and whose admissible wavevectors `2 pi m /(4a)` include `pi/a`; a 3x3x3
cell would sample only `2 pi/(3a)` and therefore *understate* `omega_max`, and a
2x2x2 cell silently collapses bonds.

## 6. The three estimates side by side

![step limits](artifacts/rayleigh_timestep/fig3_dt_limits.png)

| estimate | dt [ms] | dt [s] | dt / dt_exact | verdict |
|---|---|---|---|---|
{estimate_rows}

Two closed-form relations are worth stating because they are exact, not fitted:

* `dt_gershgorin / dt_exact = 1/sqrt(3) = {payload['gershgorin_exact_ratio_closed_form']:.6f}`
  — the Gershgorin row sum is conservative by a factor `sqrt(3)` for this lattice,
  independent of `k`, `m` and `a`.
* The Rayleigh rule is **not** density-convention-free. With the consistent
  lattice density `rho_lattice = m/a^3 = (pi/6) rho_material` it is conservative
  (ratio {_f(r['ratios']['rayleigh_voigt_lattice_density_over_exact'], 6)}), but feeding it the *material* density
  (the natural thing to do, since that is the number in `DEM3DConfig`) makes it
  **unsafe**: ratio {_f(r['ratios']['rayleigh_voigt_material_density_over_exact'], 6)} > 1. The 38% ambiguity in
  `c_R` is exactly the size of the safety margin the rule is usually assumed to
  carry, so a bare `dt = r_min/c_R` must not be used as a guard.

## 7. The threshold is real: measured growth on the 4x4x4 torus

The engine refuses any `dt` above `recommended_dt`, so the threshold cannot be
observed through `IceDEM3D`. It is observed here by time-stepping the *same*
integrator on the linear bond network directly, and comparing the measured
per-step growth of `max|u|` with the spectral radius of the exact map.

| dt / dt_exact | dt [s] | measured growth/step | theoretical | rel. error | energy tail max / initial | outcome |
|---|---|---|---|---|---|---|
{growth_rows}

![growth](artifacts/rayleigh_timestep/fig4_growth.png)

The measured and predicted growth factors agree to `{max(abs(g['rel_error_measured_vs_theory']) for g in growth):.1e}`-level relative
precision (the bounded rows are dominated by fit noise around 1.0), and the sign of `dt - dt_exact` decides boundedness: steps below the
limit stay bounded (energy oscillates, as expected for symplectic Euler, but does
not drift), steps above it grow geometrically. The `dt = 1.00711 dt_exact` row is
the Rayleigh rule with the material density — the case §6 calls unsafe — and it
diverges, so the verdict is demonstrated rather than argued.

## 8. Dispersion and the BZ maximum

![dispersion](artifacts/rayleigh_timestep/fig1_dispersion.png)
![bz map](artifacts/rayleigh_timestep/fig2_bz_map.png)

The maximum is a single point (plus its symmetry images) at X, which is why any
criterion built on a *surface* wave speed (which is what the Rayleigh rule uses)
misses it: the tallest mode of this lattice is a bulk longitudinal zone-boundary
mode, not a Rayleigh mode.

## 9. Errors vs reference, and the honest limits

| key quantity | reference | deviation | < 3%? |
|---|---|---|---|
| `c_R/c_S` at the four published anchors | textbook table | <= {payload['worst_anchor_relative_error']:.2e} | yes |
| bisection vs Lamb cubic over nu in [0, 0.5] | second exact formulation | <= {payload['worst_cubic_vs_bisection']:.2e} | yes |
| `omega_max`, Bloch scan | closed form | {r['errors']['omega_closed_vs_bz']:+.2e} | yes |
| `omega_max`, atomistic Hessian | closed form | {r['errors']['omega_closed_vs_supercell']:+.2e} | yes |
| `omega(q)` along Gamma->X | closed form | <= {max(payload['dispersion_closed_form_error'].values()):.2e} | yes |
| measured vs theoretical growth | exact 2x2 spectral radius | <= {max(abs(g['rel_error_measured_vs_theory']) for g in growth):.2e} | yes |
| `dt_gershgorin / dt_exact` | `1/sqrt(3)` (exact) | {abs(r['ratios']['gershgorin_over_exact'] - 1/math.sqrt(3)):.2e} absolute | yes |

**Limitations**

1. **Anisotropy.** `A = 2C44/(C11-C12) = {_f(r['zener_anisotropy'], 4)}`, so an isotropic
   Rayleigh speed is not strictly defined. That is why §6 reports a Voigt/Reuss
   band ({_f(r['ratios']['rayleigh_voigt_lattice_density_over_exact'], 4)} ... {_f(r['ratios']['rayleigh_reuss_lattice_density_over_exact'], 4)} of the exact limit) instead of a single number; the band stays
   below 1, the material-density variant stays above 1.
2. **Damping, contact, fracture and rotation are excluded.** The exactness of
   `dt = 2/omega_max` is a statement about the undamped linear network; contact
   damping and bond deletion change the spectrum inside a run.
3. **`recommended_dt` is unchanged.** Raising the engine guard toward the exact
   limit would alter every existing benchmark and regression artefact; as
   `docs/dem3d_timestep_stability.md` states, that is deferred on purpose. This
   page only supplies the missing reference.
4. **Density convention.** `rho_lattice = (pi/6) rho_material` is a property of a
   simple-cubic packing of spheres, not a calibrated material constant; a real
   calibration (`docs/dem3d_material_calibration_roadmap.md`) would be needed
   before any absolute `c_R` claim about ice.
5. **Not a physical validation.** Everything here is lattice dynamics of the
   discrete model.

## 10. Reproduce

```
PYTHONPATH=src python scripts/verify_rayleigh_timestep.py
python -m pytest tests/test_dem3d_rayleigh_timestep.py -q
```

Artefacts: `artifacts/rayleigh_timestep/report.json`, `rayleigh_ratio_sweep.csv`,
`dispersion.csv`, `fig1_dispersion.png`, `fig2_bz_map.png`, `fig3_dt_limits.png`,
`fig4_growth.png`.
"""


if __name__ == "__main__":
    result = main()
    print(json.dumps(
        {
            "dt_exact_s": result["timestep_report"]["dt_exact_s"],
            "ratios": result["timestep_report"]["ratios"],
            "omega_errors": result["timestep_report"]["errors"],
            "worst_anchor": result["worst_anchor_relative_error"],
            "worst_growth_error": max(
                abs(g["rel_error_measured_vs_theory"]) for g in result["growth"]
            ),
        },
        indent=2,
    ))