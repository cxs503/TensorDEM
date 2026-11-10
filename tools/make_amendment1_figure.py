#!/usr/bin/env python
"""Amendment-1 evidence: the paired sheet-thrust comparison, parsed and plotted.

Reads the raw run log ``artifacts/rubble_accumulation/amendment1_evidence/
exp_thrust2.log`` (8 cases: 2 x C_i, 2 x v_i, thrust 0 vs 20 kPa at
D = 1 m flume) and writes

* ``artifacts/rubble_accumulation/amendment1_evidence/exp_thrust2.json`` -- the
  parsed table (no hand copying),
* ``artifacts/rubble_accumulation/fig6_thrust_decision.png`` -- the paired
  comparison that the pre-registered decision rule of amendment 1 was applied to.

Usage::

    python tools/make_amendment1_figure.py
"""
from __future__ import annotations

import json
import os
import re

EVID = os.path.join("artifacts", "rubble_accumulation", "amendment1_evidence")
LOG = os.path.join(EVID, "exp_thrust2.log")
OUT_JSON = os.path.join(EVID, "exp_thrust2.json")
OUT_FIG = os.path.join("artifacts", "rubble_accumulation", "fig6_thrust_decision.png")

PATTERN = re.compile(
    r"C=(?P<C>[\d.]+) v=(?P<v>[\d.]+) sig=\s*(?P<sigma>[\d.]+) r=(?P<r>[\d.]+) "
    r"L=(?P<L>[\d.]+) B=(?P<B>[\d.]+) H=(?P<H>[\d.]+) H/2r=(?P<Hb>[\d.]+) "
    r"pile=\s*(?P<pile>[\d.]+) p=\s*(?P<p>[\d.]+) "
    r"r2\(L,B,H\)=\((?P<r2L>[-\d.]+),(?P<r2B>[-\d.]+),(?P<r2H>[-\d.]+)\) "
    r"Hmax=(?P<Hmax>[\d.]+)")


def parse() -> list[dict]:
    rows = []
    with open(LOG) as fh:
        for line in fh:
            m = PATTERN.search(line)
            if not m:
                continue
            d = {k: float(v) for k, v in m.groupdict().items()}
            d["C"] = float(m.group("C"))
            rows.append(d)
    return rows


def main() -> int:
    rows = parse()
    if not rows:
        raise SystemExit(f"no rows parsed from {LOG}")
    with open(OUT_JSON, "w") as fh:
        json.dump(rows, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = {}
    for r in rows:
        pairs.setdefault((r["C"], r["v"]), {})[r["sigma"]] = r
    labels, h0, h1, p0, p1, m0, m1 = [], [], [], [], [], [], []
    for (C, v), d in sorted(pairs.items()):
        if 0.0 not in d or 20000.0 not in d:
            continue
        labels.append(f"C_i={C:g}\nv_i={v:g} m/s")
        h0.append(d[0.0]["Hb"])
        h1.append(d[20000.0]["Hb"])
        p0.append(d[0.0]["p"])
        p1.append(d[20000.0]["p"])
        m0.append(d[0.0]["pile"])
        m1.append(d[20000.0]["pile"])

    import numpy as np
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for ax, a, b, lab, thr in (
            (axes[0], h0, h1, "H_p / (2r)  [-]", 1.5),
            (axes[1], p0, p1, "p_front [Pa]", None),
            (axes[2], m0, m1, "pile mass [kg]", None)):
        ax.bar(x - 0.2, a, 0.4, label="thrust off (reference drive)")
        ax.bar(x + 0.2, b, 0.4, label="thrust 20 kPa")
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, axis="y")
        if thr is not None:
            ax.axhline(thr, color="k", ls=":", lw=0.9)
            ax.text(0.02, thr, " decision threshold 1.5", fontsize=6, va="bottom")
    axes[0].set_title("sail height is not produced", fontsize=9)
    axes[1].set_title("the thrust raises the load", fontsize=9)
    axes[2].set_title("... but thins the pile", fontsize=9)
    axes[0].legend(fontsize=6)
    fig.suptitle("amendment-1 decision: sheet-thrust boundary (REJECT as a sail mechanism)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_JSON} ({len(rows)} rows) and {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())