#!/usr/bin/env python3
"""Plot the m4804 negative-control margins from the JSON produced by the runner.

Reads ``artifacts/m4804_negative_control_margins/margins.json`` (written by
``scripts/negative_controls_m4804.py``) and draws

    docs/figures/m4804_negative_control_margins.png

panel (a): margin ``M = injected fault / smallest detectable fault`` for every
           shipped negative control, log x-axis, with the ``M = 1`` resolution
           limit drawn as a vertical line;
panel (b): additive resolution ``d*`` (J) against each channel's clean value, log
           x-axis, which shows how much headroom each channel's magnitude has
           over the gate's resolution.

No number is typed in this script: every bar is read from the JSON.

Usage::

    python3 scripts/plot_m4804_ledger_negative_margins.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "m4804_negative_control_margins" / "margins.json"
OUT = ROOT / "docs" / "figures" / "m4804_negative_control_margins.png"


def control_label(control: dict) -> str:
    """Short human label for one shipped negative control."""
    if control["mode"] == "multiplicative":
        if control["factor"] == 0.0:
            kind = "zeroed"
        elif control["factor"] == 3.0:
            kind = "tripled"
        else:
            kind = f"x{control['factor']:g}"
    else:
        kind = "additive"
    return f"{control['channel']} ({kind})"


def main() -> int:
    report = json.loads(DATA.read_text())
    controls = report["controls"]
    channels = report["channels"]

    fig, (ax_margin, ax_res) = plt.subplots(
        1, 2, figsize=(13.0, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    # ---- panel (a): control margin over the gate resolution -----------------
    labels = [control_label(c) for c in controls]
    margins = [c["margin"] for c in controls]
    y = list(range(len(controls)))
    bars = ax_margin.barh(y, margins, color="#3b7dd8", alpha=0.85)
    worst = min(range(len(controls)), key=lambda i: margins[i])
    bars[worst].set_color("#d1495b")
    ax_margin.set_yticks(y)
    ax_margin.set_yticklabels(labels, fontsize=8.5)
    ax_margin.invert_yaxis()
    ax_margin.set_xscale("log")
    ax_margin.axvline(
        1.0, color="k", linestyle="--", linewidth=1.2,
        label="M = 1: gate resolution limit",
    )
    for yi, m in zip(y, margins):
        ax_margin.text(m * 1.06, yi, f"{m:.1f}x", va="center", fontsize=8)
    ax_margin.set_xlim(0.5, max(margins) * 3.0)
    ax_margin.xaxis.set_major_locator(LogLocator(base=10))
    ax_margin.set_xlabel("margin  M = injected fault / smallest detectable fault")
    ax_margin.set_title(
        "(a) every shipped negative control clears the gate resolution",
        fontsize=10.5,
    )
    ax_margin.grid(axis="x", alpha=0.25, which="both")
    ax_margin.legend(loc="lower right", fontsize=8)

    # ---- panel (b): resolution against channel magnitude --------------------
    res_y = list(range(len(channels)))
    values = [max(c["value_J"], 1e-30) for c in channels]
    resolutions = [c["additive_resolution_J"] for c in channels]
    height = 0.38
    ax_res.barh(
        [i - height / 2 for i in res_y], values, height=height,
        color="#7fb3d5", label="clean channel value $x_0$",
    )
    ax_res.barh(
        [i + height / 2 for i in res_y], resolutions, height=height,
        color="#c0392b", label="additive resolution $d^*$",
    )
    ax_res.set_yticks(res_y)
    ax_res.set_yticklabels(
        [c["channel"] + (" (dormant)" if c["value_J"] == 0.0 else "") for c in channels],
        fontsize=8.5,
    )
    ax_res.invert_yaxis()
    ax_res.set_xscale("log")
    ax_res.xaxis.set_major_locator(LogLocator(base=10))
    ax_res.set_xlabel("energy (J), log scale")
    ax_res.set_title(
        "(b) channel magnitude vs the gate's additive resolution",
        fontsize=10.5,
    )
    ax_res.grid(axis="x", alpha=0.25, which="both")
    ax_res.legend(loc="lower right", fontsize=8)

    headroom = report["gate"]["headroom_J"]
    fig.suptitle(
        "2-D energy/fracture ledger: negative-control margins "
        f"(tolerance {report['gate']['tolerance_rel']:.3g}, "
        f"headroom $d^*$ = {headroom:.3e} J, "
        f"{report['broken_bonds']} broken bonds)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    print(
        "min margin {:.3f} ({}), max margin {:.3f}".format(
            report["checks"]["min_control_margin"],
            report["checks"]["min_control_margin_channel"],
            max(margins),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())