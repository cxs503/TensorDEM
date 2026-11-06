#!/usr/bin/env python3
"""Produce the evidence bundle for the 2-D energy/fracture negative controls.

Runs the tool-indentation scenario once, then measures the ledger balance
residual for (a) the clean ledger and (b) one deliberately corrupted channel at
a time.  Writes:

    artifacts/m4812_ledger_negative_controls/ledger_negative_controls.json
    artifacts/m4812_ledger_negative_controls/ledger_negative_controls.png
    artifacts/m4812_ledger_negative_controls/ledger_negative_controls.log

Every number in the accompanying document is injected from the JSON above; no
value is transcribed by hand.
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
    _energy_scale,
    _rel_err,
    _run_indentation,
)

OUT = ROOT / "artifacts" / "m4812_ledger_negative_controls"

#: (channel, factor, offset) — one fault per control
FAULTS = [
    ("fracture_release_J", 0.0, None),
    ("fracture_release_J", 3.0, None),
    ("pair_contact_acquired_J", 0.0, None),
    ("pair_contact_acquired_J", 3.0, None),
    ("tool_work_J", 0.0, None),
    ("damping_dissipation_J", 0.0, None),
    ("external_work_J", None, "tool_work_J"),  # additive fault (dormant channel)
]


def main() -> int:
    sim = _run_indentation()
    OUT.mkdir(parents=True, exist_ok=True)
    clean_err = _rel_err(sim)
    ledger = {
        k: (float(sum(v)) if isinstance(v, (list, tuple)) else float(v))
        for k, v in sim.accounting.items()
    }
    rows = []
    for channel, factor, offset_channel in FAULTS:
        offset = None if offset_channel is None else float(sim.accounting[offset_channel])
        corrupted = _corrupt(sim, channel, factor=factor, offset=offset)
        err = _rel_err(corrupted)
        rows.append(
            {
                "channel": channel,
                "factor": factor,
                "offset_J": offset,
                "value_clean_J": float(sim.accounting[channel]),
                "value_corrupted_J": float(corrupted.accounting[channel]),
                "rel_residual": err,
                "detected": bool(err > TOL),
                "margin_over_tol": err / TOL,
            }
        )
    payload = {
        "case": "dem2d tool indentation, fractures bonds",
        "lattice": list((7, 3)),
        "duration_s": 0.12,
        "tool_speed_m_per_s": 0.2,
        "steps": int(round(0.12 / sim.dt)),
        "dt_s": float(sim.dt),
        "broken_bonds": int(sim.broken_bonds),
        "tolerance_rel": TOL,
        "clean_rel_residual": clean_err,
        "clean_closes": bool(clean_err < TOL),
        "energy_scale_J": _energy_scale(sim),
        "ledger_J": ledger,
        "faults": rows,
        "all_faults_detected": bool(all(r["detected"] for r in rows)),
    }
    (OUT / "ledger_negative_controls.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        f"case={payload['case']} lattice={tuple(payload['lattice'])} "
        f"steps={payload['steps']} dt={payload['dt_s']:.6e}",
        f"broken_bonds={payload['broken_bonds']} tolerance={TOL}",
        f"clean_rel_residual={clean_err:.6e} closes={payload['clean_closes']}",
        "channel,fault,value_clean,value_corrupted,rel_residual,detected",
    ]
    for r in rows:
        fault = f"x{r['factor']}" if r["factor"] is not None else f"+{r['offset_J']:.6e}"
        lines.append(
            f"{r['channel']},{fault},{r['value_clean_J']:.6e},"
            f"{r['value_corrupted_J']:.6e},{r['rel_residual']:.6e},{r['detected']}"
        )
    (OUT / "ledger_negative_controls.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:  # pragma: no cover - viz extra is optional
        print("matplotlib unavailable: skipped figure")
        return 0

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.0, 5.2))

    # (a) ledger channel magnitudes: clean vs corrupted
    channels = [k for k in ledger if k.endswith("_J")]
    clean_vals = [max(ledger[k], 1e-18) for k in channels]
    x0 = np.arange(len(channels))
    ax0.bar(x0 - 0.2, clean_vals, width=0.4, color="#2980b9",
            edgecolor="black", linewidth=0.5, label="clean ledger")
    worst = {}
    for r in rows:
        worst[r["channel"]] = max(worst.get(r["channel"], 0.0), abs(r["value_corrupted_J"]))
    corr_vals = [max(worst.get(k, ledger[k]), 1e-18) for k in channels]
    ax0.bar(x0 + 0.2, corr_vals, width=0.4, color="#e67e22",
            edgecolor="black", linewidth=0.5, label="worst corrupted variant")
    ax0.set_yscale("log")
    ax0.set_xticks(x0)
    ax0.set_xticklabels([c.replace("_J", "") for c in channels],
                        rotation=18, fontsize=8, ha="right")
    ax0.set_ylabel("channel magnitude [J]")
    ax0.set_title("(a) energy ledger channels, clean vs corrupted")
    ax0.legend(fontsize=9)
    ax0.grid(axis="y", which="both", alpha=0.3)

    # (b) relative residual per injected fault, against the acceptance tolerance
    labels = [f"{r['channel']}\n" + (f"x{r['factor']:g}" if r["factor"] is not None else "additive fault")
              for r in rows]
    values = [r["rel_residual"] for r in rows]
    idx = np.arange(len(rows))
    colors = ["#c0392b" if r["detected"] else "#7f8c8d" for r in rows]
    ax1.bar(idx, values, color=colors, edgecolor="black", linewidth=0.6)
    ax1.set_yscale("log")
    ax1.axhline(TOL, color="navy", linestyle="--", linewidth=1.6,
               label=f"acceptance tolerance {TOL:g} (0.5 %)")
    ax1.axhline(clean_err, color="green", linestyle=":", linewidth=1.6,
               label=f"clean ledger residual {clean_err:.3g}")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(labels, fontsize=7.0)
    ax1.set_ylabel("relative ledger residual |R| / scale")
    ax1.set_title("(b) single-channel fault detection (all above tolerance)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(axis="y", which="both", alpha=0.3)

    fig.suptitle(
        "2-D DEM energy/fracture ledger: negative controls that detect every channel\n"
        f"(tool indentation, {payload['steps']} steps, {payload['broken_bonds']} broken bonds)"
    )
    fig.tight_layout()
    fig.savefig(OUT / "ledger_negative_controls.png", dpi=115)
    print(f"wrote {OUT / 'ledger_negative_controls.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())