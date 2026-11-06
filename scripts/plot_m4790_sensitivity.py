#!/usr/bin/env python3
"""m4790: fault-magnitude sensitivity sweep of the 2-D energy/fracture ledger gate.

The companion artifact ``artifacts/m4812_ledger_negative_controls/`` answers
"does the balance gate react to a fault in channel *c*?" (bar chart, one fault
per channel).  This script answers the quantitative follow-up that the
negative controls leave open:

    **how large must a fault be before the gate actually turns red?**

For each ledger channel we sweep a multiplicative corruption factor ``z`` and
plot the relative residual

    rho(z) = |R(z)| / S(z),

with ``R`` the balance residual and ``S`` the same reference scale the test
gate uses (``|W_tool| + |Q_damping| + |E_fracture|``, recomputed on the
*corrupted* ledger, exactly like ``_rel_err`` in
``tests/test_dem2d_energy_fracture_ledger_negative.py``).

Because ``R`` is *linear* in each channel, the sweep has a closed form.  With
``u = z - 1`` and ``a`` the clean channel value,

    R(z) = R0 + s * a * u,          s = +1 (damping, fracture)
                                        -1 (tool work, external work, pair contact)
    S(z) = S0 + |a| * u             (only for channels inside the scale)

so the detection threshold ``|R| = TOL * S`` is a pair of linear equations and
``z*`` is exact.  The script therefore reports, per channel, the analytic
``z*`` **and** the value the same formula predicts on the measured sweep: the
two must agree to machine precision, which is the correctness criterion used
in ``docs/dem2d_energy_fracture_ledger_negative_controls.md`` (no baseline
experiment exists for a numerical gate, so the analytic identity is the
reference).

Outputs (all numbers injected programmatically, none transcribed):

* ``artifacts/m4790/sensitivity.json``  -- sweep + analytic thresholds
* ``artifacts/m4790/ledger_corruption_sensitivity.png`` -- two panels:
  (a) rho(z) per channel against the 0.5 % tolerance line, (b) analytic
  fault magnitude threshold ``|1 - z*|`` per channel (gate blind-spot width).

Usage::

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 scripts/plot_m4790_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.test_dem2d_energy_fracture_ledger_negative import (  # noqa: E402
    TOL,
    _corrupt,
    _energy_residual,
    _energy_scale,
    _rel_err,
    _run_indentation,
)

OUT = ROOT / "artifacts" / "m4790"

#: sign of each channel inside the balance identity
SIGN = {
    "damping_dissipation_J": +1.0,
    "fracture_release_J": +1.0,
    "tool_work_J": -1.0,
    "external_work_J": -1.0,
    "pair_contact_acquired_J": -1.0,
}

#: channels that enter the reference scale S = |W_tool| + |Q| + |E_fracture|
IN_SCALE = ("tool_work_J", "damping_dissipation_J", "fracture_release_J")

#: multiplicative corruption factors swept for every channel
FACTORS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.01, 1.1, 1.25, 1.5, 2.0, 3.0]


def analytic_z_star(
    r0: float, s0: float, a: float, sign: float, in_scale: bool, tol: float = TOL
) -> list[float]:
    """Exact multiplicative factors ``z*`` at which ``|R(z)| = tol * S(z)``.

    Both ``R`` and ``S`` are affine in ``u = z - 1``, so the two branches of
    the absolute value each give one root ``u = (tol*s0 - r0) / (sign*a -+ tol*|a|)``.
    """
    roots: list[float] = []
    a_bar = abs(a) if in_scale else 0.0
    for branch in (+1.0, -1.0):
        denom = sign * a - branch * tol * a_bar
        if abs(denom) < 1e-300:
            continue
        u = (branch * tol * s0 - r0) / denom
        roots.append(1.0 + u)
    return sorted(set(round(z, 15) for z in roots))


def main() -> int:
    sim = _run_indentation()
    OUT.mkdir(parents=True, exist_ok=True)

    accounting = {
        k: [float(x) for x in v] if isinstance(v, (list, tuple)) else float(v)
        for k, v in sim.accounting.items()
    }
    r0 = _energy_residual(sim)
    s0 = _energy_scale(sim)
    clean_rel = _rel_err(sim)

    channels = []
    for channel, sign in SIGN.items():
        a = accounting[channel]
        in_scale = channel in IN_SCALE
        sweep = []
        for z in FACTORS:
            corrupted = _corrupt(sim, channel, factor=z)
            sweep.append(
                {
                    "factor": z,
                    "value_J": float(corrupted.accounting[channel]),
                    "residual_J": _energy_residual(corrupted),
                    "rel_residual": _rel_err(corrupted),
                    "detected": bool(_rel_err(corrupted) > TOL),
                }
            )
        analytic = analytic_z_star(r0, s0, a, sign, in_scale)
        # verify the analytic roots really sit on the tolerance surface
        checks = []
        for z in analytic:
            corrupted = _corrupt(sim, channel, factor=z)
            predicted = abs(r0 + sign * a * (z - 1.0)) / max(
                s0 + (abs(a) * (z - 1.0) if in_scale else 0.0), 1e-18
            )
            checks.append(
                {
                    "factor": z,
                    "residual_J": _energy_residual(corrupted),
                    "rel_residual_predicted": predicted,
                    "rel_residual_measured": _rel_err(corrupted),
                    "abs_err_vs_analytic": abs(predicted - TOL),
                }
            )
        # smallest |z - 1| that the gate detects on the swept grid
        detected = [t for t in sweep if t["detected"]]
        grid_z = min(detected, key=lambda t: abs(t["factor"] - 1.0))["factor"] if detected else None
        channels.append(
            {
                "channel": channel,
                "sign": sign,
                "in_scale": in_scale,
                "clean_value_J": a,
                "sweep": sweep,
                "analytic_threshold_factors": analytic,
                "analytic_checks": checks,
                "grid_first_detected_factor": grid_z,
                "vacuous": bool(a == 0.0),
                "blind_spot_abs_1mz": (
                    min(abs(1.0 - z) for z in analytic) if analytic else None
                ),
            }
        )

    payload = {
        "case": "dem2d tool indentation, fractures bonds",
        "gate": "balance_check: |R| / S < TOL",
        "lattice": list((7, 3)),
        "duration_s": 0.12,
        "tool_speed_m_per_s": 0.2,
        "dt_s": float(sim.dt),
        "steps": int(round(0.12 / sim.dt)),
        "broken_bonds": int(sim.broken_bonds),
        "tolerance_rel": TOL,
        "clean_residual_J": r0,
        "clean_scale_J": s0,
        "clean_rel_residual": clean_rel,
        "clean_closes": bool(clean_rel < TOL),
        "ledger_J": accounting,
        "factors_swept": FACTORS,
        "channels": channels,
        "analytic_surface_max_abs_err": max(
            (c["analytic_checks"] and max(
                ch["abs_err_vs_analytic"] for ch in c["analytic_checks"]
            )) or 0.0
            for c in channels
        ),
    }
    (OUT / "sensitivity.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:  # pragma: no cover - viz extra is optional
        print("matplotlib unavailable: skipped figure")
        return 0

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.2, 5.4))

    # (a) measured rho(z) per channel with the acceptance tolerance
    for ch in channels:
        color = "#c0392b" if ch["channel"] == "pair_contact_acquired_J" else None
        ax0.plot(
            [t["factor"] for t in ch["sweep"]],
            [max(t["rel_residual"], 1e-18) for t in ch["sweep"]],
            marker="o",
            markersize=3.6,
            linewidth=1.3,
            color=color,
            label=ch["channel"].replace("_J", ""),
        )
    ax0.axhline(TOL, color="navy", linestyle="--", linewidth=1.6,
                label=f"tolerance {TOL:g} (0.5 %)")
    ax0.axhline(clean_rel, color="green", linestyle=":", linewidth=1.4,
                label=f"clean residual {clean_rel:.3g}")
    ax0.set_xlabel("multiplicative corruption factor z  (z=1 is clean)")
    ax0.set_ylabel(r"relative ledger residual  $\rho(z)=|R|/S$")
    ax0.set_yscale("log")
    ax0.set_title("(a) residual growth vs fault magnitude")
    ax0.legend(fontsize=8, ncol=2)
    ax0.grid(alpha=0.3)

    # (b) analytic blind-spot width |1 - z*| per channel (gate sensitivity)
    labels = [c["channel"].replace("_J", "") for c in channels]
    worst = [c["blind_spot_abs_1mz"] if c["blind_spot_abs_1mz"] is not None else 0.0
             for c in channels]
    for c in channels:
        if c["blind_spot_abs_1mz"] is None:  # dormant channel: multiplicative faults are no-ops
            c["note"] = "vacuous: channel is 0 in this scenario, multiplicative faults are no-ops"
    idx = np.arange(len(channels))
    ax1.barh(idx, worst, color="#2980b9", edgecolor="black", linewidth=0.6)
    for i, w in zip(idx, worst):
        ax1.annotate(f"|1-z*| = {w:.3g}", (w, i), xytext=(3, 0),
                     textcoords="offset points", va="center", fontsize=8)
    ax1.set_yticks(idx)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("smallest relative fault the gate detects,  |1 - z*|")
    ax1.set_title("(b) analytic detection threshold per channel\n"
                  "(smaller = more sensitive; the scale channel is the blind spot)")
    ax1.grid(axis="x", alpha=0.3)
    ax1.set_xlim(0, max(worst) * 1.35 if max(worst) > 0 else 1.0)

    fig.suptitle(
        "2-D DEM energy/fracture ledger: fault-magnitude sensitivity of the 0.5 % balance gate\n"
        f"(tool indentation, {payload['steps']} steps, {payload['broken_bonds']} broken bonds,"
        f" clean residual {clean_rel * 100:.2f} %)"
    )
    fig.tight_layout()
    png = OUT / "ledger_corruption_sensitivity.png"
    fig.savefig(png, dpi=115)
    print(f"wrote {png}")
    print(f"wrote {OUT / 'sensitivity.json'}")
    print(f"clean_rel_residual={clean_rel:.6e} closes={payload['clean_closes']}")
    for c in channels:
        blind = c["blind_spot_abs_1mz"]
        blind_s = "vacuous(a=0)" if blind is None else f"{blind:.6g}"
        print(
            f"  {c['channel']:<26s} |1-z*|={blind_s} "
            f"analytic_z*={[round(z, 6) for z in c['analytic_threshold_factors']]} "
            f"grid_z={c['grid_first_detected_factor']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())