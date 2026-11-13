#!/usr/bin/env python3
"""m17588 board-audit figure — m17500's got-vs-ref evidence revisited.

Panel A: got (contact-proxy I_c / mu_proxy) vs ref (bulk I / mu) for the five
         audited rows, with the y=x identity line and the 15 % band.
Panel B: the three error families per row against the frozen 15 % tolerance,
         plus the reference gate (bulk route vs Jop 2006) that its own prereg
         requires to pass before absolute mu comparison is allowed.

Numbers come straight from the audited results.json — nothing typed by hand.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = "/root/tp-agent/work/m17500/artifacts/xval_mu_i_contact_proxy_m17500/results.json"
OUT = "/root/tp-agent/work/m17588/artifacts/audit_m17588/fig_m17500_audit.png"

d = json.load(open(RES))
rows = d["rows"]
tol = d["tolerance"]
lab = [r["label"] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9))

# ---- Panel A: got vs ref -------------------------------------------------
for key_got, key_ref, mk, col, name in (
        ("i_proxy_fn_weighted", "i_bulk_measured", "o", "#1f77b4", "I_c (proxy) vs I (bulk)"),
        ("mu_proxy_fn_weighted", "mu_bulk_measured", "s", "#d62728", "mu_proxy vs mu = tau/P")):
    got = np.array([r[key_got] for r in rows])
    ref = np.array([r[key_ref] for r in rows])
    ax1.scatter(ref, got, marker=mk, s=52, color=col, label=name, zorder=3)
xx = np.logspace(-2.5, 0.2, 50)
ax1.plot(xx, xx * (1 + tol), "k--", lw=0.9, label="±15 % band")
ax1.plot(xx, xx * (1 - tol), "k--", lw=0.9)
ax1.plot(xx, xx, "k-", lw=0.7, label="identity")
for r in rows:
    ax1.annotate(r["label"].replace("base_", "").replace("sens_", "s:"),
                 (r["i_bulk_measured"], r["i_proxy_fn_weighted"]),
                 textcoords="offset points", xytext=(5, -10), fontsize=7, color="#1f77b4")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("REF: bulk I, mu = tau/P")
ax1.set_ylabel("GOT: contact-proxy I_c, mu(I_c)")
ax1.set_title("A  m17500 got vs ref, 5 audited rows (log-log)")
ax1.legend(fontsize=7.5, loc="upper left")
ax1.grid(alpha=0.3, which="both")

# ---- Panel B: errors vs tolerance ---------------------------------------
x = np.arange(len(rows))
w = 0.26
sets = (("i_mapping_error", "|I_c/I-1|", "#1f77b4"),
        ("mu_error", "|mu_proxy/mu-1|", "#d62728"),
        ("bulk_vs_jop_rel_error", "reference gate: |mu_bulk/mu_Jop-1|", "#7f7f7f"))
for i, (k, name, col) in enumerate(sets):
    ax2.bar(x + (i - 1) * w, [r[k] for r in rows], w, color=col, label=name)
ax2.axhline(tol, color="k", ls="--", lw=1.1)
ax2.text(len(rows) - 0.5, tol * 1.04, f"prereg tolerance {tol:.0%}", ha="right",
         fontsize=8)
ax2.set_xticks(x)
ax2.set_xticklabels(lab, rotation=18, fontsize=8)
ax2.set_ylabel("relative error")
ax2.set_title("B  error families; grey = reference gate that voids absolute mu")
ax2.legend(fontsize=7.5)
ax2.grid(axis="y", alpha=0.3)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, dpi=135)
print(OUT, os.path.getsize(OUT), "bytes")
print(json.dumps({"tol": tol,
                  "worst_i_mapping_error": max(r["i_mapping_error"] for r in rows),
                  "worst_mu_error": max(r["mu_error"] for r in rows),
                  "worst_reference_gate": max(r["bulk_vs_jop_rel_error"] for r in rows)},
                 indent=1))