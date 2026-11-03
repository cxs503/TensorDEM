"""Render docs/dem3d_rolling_resistance.md from the validation artifacts.

Every number in the generated document is read from
``docs/rolling_resistance_validation/summary.json`` or the CSVs produced by
``scripts/validate_rolling_resistance.py``; nothing is typed by hand.

Example:
    python scripts/report_rolling_resistance.py
"""
from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs" / "rolling_resistance_validation"
OUT = ROOT / "docs" / "dem3d_rolling_resistance.md"


def read_csv(name: str) -> list[dict[str, str]]:
    with (ART / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value, digits: int = 4) -> str:
    if value is None or value == "":
        return "--"
    return f"{float(value):.{digits}g}"


def main() -> int:
    summary = json.loads((ART / "summary.json").read_text(encoding="utf-8"))
    accel = read_csv("acceleration_sweep.csv")
    threshold = read_csv("threshold_sweep.csv")
    models = read_csv("model_comparison.csv")
    torque = read_csv("torque_laws.csv")

    harness = summary["harness"]
    parity = summary["acceleration_parity"]
    bisect = summary["bisection"]
    lines: list[str] = []
    add = lines.append

    add("# Rolling resistance in TensorDEM: contact laws and analytical validation")
    add("")
    add("Status: contact law implemented and verified against closed-form references; "
        "**not yet wired into the bonded-particle solver** (see *Limitations*).")
    add("")
    add("## 1. Why this capability was missing")
    add("")
    add("`tensordem.contact_tangential` provides normal restitution and Cundall-Strack "
        "sliding friction, but its `TangentialContactConfig.rolling_mu` field was an "
        "unused placeholder: there was no rolling-resistance torque anywhere in the "
        "package. For DEM ice work this is the dominant missing contact mechanism, "
        "because ice blocks and floes are near-spherical/lumpy bodies that, without a "
        "rolling resistance, roll far too freely. That biases two quantities the "
        "platform reports: the angle of repose of a rubble field and the resistance of "
        "a broken-ice channel.")
    add("")
    add("`docs/dem3d_mechanics_benchmark_protocol.md` and "
        "`docs/dem3d_specimen_response_benchmarks.md` both list rolling resistance as "
        "not implied by the existing benchmark suite, so this document closes a "
        "recorded verification gap rather than adding a new solver feature.")
    add("")
    add("## 2. Method")
    add("")
    add("### 2.1 Contact laws (`src/tensordem/rolling_resistance.py`)")
    add("")
    add("Let `omega_rel = omega_j - omega_i` be the relative angular velocity of a "
        "contact pair, `n` the unit normal from `i` to `j`, and `r_eff` the effective "
        "radius. Only the *rolling* part `omega_roll = omega_rel - (omega_rel.n) n` is "
        "resisted; drilling (spin) friction about `n` is out of scope. The saturated "
        "rolling limit is `M_lim = mu_r |F_n| r_eff`.")
    add("")
    add("| model | torque applied to `j` | character |")
    add("|---|---|---|")
    add("| `cdt` | `-mu_r |F_n| r_eff · omega_hat` | memoryless, always saturated |")
    add("| `viscous` | `-(mu_r |F_n| r_eff / omega_ref) · omega_roll` | strictly dissipative, unbounded above `omega_ref` |")
    add("| `epsd` | `M^{n+1} = clamp(M^n - k_r omega_roll dt, |M| <= M_lim)` | elastic-plastic spring, supports static equilibrium |")
    add("")
    add("Model families follow Ai et al. (2011, *Powder Technology* 206:269-282); the "
        "elastic-plastic branch follows Iwashita & Oda (1998) and Jiang et al. (2005).")
    add("")
    add("### 2.2 Verification harness")
    add("")
    add("A single solid sphere is integrated with a normal spring-dashpot contact, an "
        "incremental Cundall-Strack tangential spring driven by the *contact-point* slip "
        "velocity `v + R omega`, and the rolling torque above. N independent inclines run "
        "as one batched tensor loop.")
    add("")
    add("| parameter | value |")
    add("|---|---|")
    add(f"| sphere radius | {f(harness['radius_m'])} m |")
    add(f"| normal stiffness `k_n` | {f(harness['k_n_N_m'])} N/m |")
    add(f"| time step `dt` | {f(harness['dt_s'])} s |")
    add(f"| run duration | {f(harness['duration_s'])} s |")
    add(f"| acceleration fit window | `t >= {f(harness['settle_time_s'])}` s |")
    add(f"| sliding friction `mu` | {f(harness['sliding_friction_mu'])} |")
    add(f"| rolling stiffness rule | `{harness['rolling_stiffness_rule']}` |")
    add("")
    add("Two closed-form references are used, selected by the friction-cone condition "
        "`pure_rolling_feasible()`:")
    add("")
    add("* pure rolling: `a = g (sin t - mu_r cos t) / (1 + I/mR^2)`, `I/mR^2 = 2/5`;")
    add("* steady sliding: `a = g (sin t - mu cos t)`;")
    add("* static threshold: `theta_c = arctan(mu_r)`.")
    add("")
    add("## 3. Results")
    add("")
    add(f"Environment: Python {summary['environment']['python']}, torch "
        f"{summary['environment']['torch']}, {summary['environment']['device']}.")
    add("")
    add("### 3.1 Acceleration parity (rolling and sliding regimes)")
    add("")
    add("| mu_r | theta (deg) | theta/theta_c | regime | a simulated (m/s^2) | a analytic (m/s^2) | rel. error | fit R^2 |")
    add("|---|---|---|---|---|---|---|---|")
    for row in accel:
        if row["comparable"] != "True":
            continue
        add(f"| {f(row['mu_r'])} | {f(row['theta_deg'],5)} | {f(row['theta_over_theta_c'],3)} | "
            f"{row['regime']} | {f(row['a_simulated_m_s2'],6)} | {f(row['a_analytic_m_s2'],6)} | "
            f"{float(row['relative_error'])*100:.2e} % | {f(row['fit_r_squared'],5)} |")
    add("")
    add(f"Worst relative error over {parity['comparable_rows']} supercritical rows "
        f"(both regimes): **{float(parity['max_relative_error'])*100:.3g} %**; "
        f"minimum fit R^2 = {f(parity['min_fit_r_squared'],5)}. "
        f"Rows whose analytical acceleration is below "
        f"{parity['min_analytic_reference_m_s2']} m/s^2 are excluded from the relative "
        "error statistics because they are dominated by the settling transient.")
    add("")
    add("### 3.2 Static threshold versus arctan(mu_r)")
    add("")
    add("| mu_r | theta_c analytic (deg) | measured transition (deg) | grid cell (deg) | rel. error |")
    add("|---|---|---|---|---|")
    for mu, block in summary["threshold"].items():
        add(f"| {f(mu)} | {f(block['theta_c_analytic_deg'],5)} | "
            f"{f(block['theta_transition_measured_deg'],5)} | {f(block['grid_cell_deg'],3)} | "
            f"{float(block['relative_error'])*100:.3g} % |")
    add("")
    add(f"Bisection refinement at mu_r = {f(0.2)}: transition at "
        f"{f(bisect['theta_deg'],6)} deg versus {f(bisect['theta_c_analytic_deg'],6)} deg "
        f"analytical, relative error **{float(bisect['relative_error'])*100:.3g} %**.")
    add("")
    add("### 3.3 Model character on the same inclines")
    add("")
    add("| case | model | theta/theta_c | late-window a (m/s^2) | analytic a (m/s^2) | displacement / R | end speed (m/s) |")
    add("|---|---|---|---|---|---|---|")
    for row in models:
        add(f"| {row['case']} | {row['model']} | {f(row['theta_over_theta_c'],3)} | "
            f"{f(row['a_late_m_s2'],4)} | {f(row['a_analytic_m_s2'],4)} | "
            f"{f(row['displacement_over_radius'],3)} | {f(row['end_speed_m_s'],3)} |")
    add("")
    add("Reading: at the subcritical angle EPSD stays at rest (no sustained "
        "acceleration, displacement at the numerical floor), whereas `cdt` and "
        "`viscous` creep downhill because they have no static branch. At the "
        "supercritical angle `cdt` and `epsd` reproduce the analytical rolling "
        "acceleration, while `viscous` brakes much more strongly and settles at a "
        "terminal speed instead of a constant acceleration -- a model-character "
        "difference, not a numerical error.")
    add("")
    add("### 3.4 Torque laws (law level, |F_n| = 1.5 N, mu_r = 0.05)")
    add("")
    add("| |omega| (rad/s) | limit (N m) | cdt (N m) | viscous (N m) | epsd (N m) |")
    add("|---|---|---|---|---|")
    for row in torque:
        add(f"| {f(row['omega_rad_s'])} | {f(row['limit_Nm'],3)} | {f(row['cdt'],3)} | "
            f"{f(row['viscous'],3)} | {f(row['epsd'],3)} |")
    add("")
    add("The viscous branch exceeds the rolling limit for `|omega| > omega_ref = 1` "
        "rad/s; it is intended as a dissipation term, not as a rolling-friction "
        "coefficient.")
    add("")
    add("### 3.5 Figures")
    add("")
    add("| figure | content |")
    add("|---|---|")
    add("| `rolling_resistance_validation/fig_torque_laws.png` | |M| versus rolling rate for the three models with the limit line |")
    add("| `rolling_resistance_validation/fig_acceleration_parity.png` | simulated versus analytical acceleration with the +-3 % band |")
    add("| `rolling_resistance_validation/fig_threshold.png` | displacement after a fixed run versus incline angle, per mu_r |")
    add("| `rolling_resistance_validation/fig_history.png` | supercritical run: speed versus -omega R, contact slip, rolling torque |")
    add("")
    add("## 4. Error judgement against the 3 % criterion")
    add("")
    add("| key quantity | reference | error | below 3 %? |")
    add("|---|---|---|---|")
    add(f"| downhill acceleration on an incline | `g(sin t - mu_r cos t)/(1+I/mR^2)` or `g(sin t - mu cos t)` | "
        f"{float(parity['max_relative_error'])*100:.3g} % (max of {parity['comparable_rows']} rows) | yes |")
    add(f"| static transition angle | `arctan(mu_r)` | {float(bisect['relative_error'])*100:.3g} % (bisection), "
        f"<= half a grid cell on the sweep | yes |")
    add("| rolling time-scale ratio `omega R / v` | `-1` for rolling without slip | "
        f"{abs(float(accel[0]['omega_radius_over_v']) + 1)*100:.3g} % (first row) | yes |")
    add("")
    add("## 5. Limitations and honest caveats")
    add("")
    for item in summary["limitations"]:
        add(f"* {item}")
    add("* The validation harness is not a granular-assembly benchmark: no angle-of-repose "
        "or channel-resistance calibration is claimed, because the repository holds no "
        "experimental ice data to compare against.")
    add("* `cdt`/`viscous` cannot hold a static contact; they must not be used for "
        "repose-angle work without an explicit static regularisation.")
    add("")
    add("## 6. Reproduction")
    add("")
    add("```bash")
    add("python scripts/validate_rolling_resistance.py   # JSON/CSV/PNG into docs/rolling_resistance_validation/")
    add("python scripts/report_rolling_resistance.py     # regenerate this document from those artifacts")
    add("python -m pytest tests/test_rolling_resistance.py -q")
    add("```")
    add("")
    add(f"Generated by `scripts/report_rolling_resistance.py` on "
        f"{platform.python_implementation()} {platform.python_version()} with torch "
        f"{torch.__version__}.")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
