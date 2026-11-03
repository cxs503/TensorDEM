"""Audit the coupling between the platen contact mask and the restitution rigs.

The platen boundary applies its dashpot term to *every* particle, not only to
the particles that intersect the plane (see
``docs/dem3d_platen_contact_mask_audit.md``). The published fix is a two-line
``torch.where(overlap > 0, ...)`` mask, but landing the mask alone is not
verified-safe: the platen-restitution rig
(``tensordem.restitution.measure_platen_restitution``) currently agrees with the
clamped closed form partly *because* of the fictitious drag. This script
quantifies both effects on whatever ``dem3d.py`` is on disk, so the coupling can
be measured before and after the mask lands.

Usage::

    python scripts/audit_platen_mask_coupling.py --tag baseline
    git stash / patch dem3d.py            # land the mask
    python scripts/audit_platen_mask_coupling.py --tag masked
    python scripts/audit_platen_mask_coupling.py --merge

``--merge`` writes ``artifacts/platen_mask_coupling/{audit.json,audit.png}``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.restitution import (
    measure_platen_restitution,
    restitution_from_damping,
    restitution_from_damping_clamped,
)

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "platen_mask_coupling"

GAMMA_P = 0.5          # platen damping [N s/m]
V_PLATEN = 0.05        # platen closing speed [m/s]
N_SIDE, N_Z = 4, 3     # 48-particle specimen


def fictitious_reaction() -> dict[str, float]:
    """Platens 0.5 m away from a 48-particle block that still has velocity.

    Closed form for the unmasked law: every particle contributes
    ``gamma_p * |v_platen|`` to the reaction, so the reported reaction is
    ``n * gamma_p * |v_platen|`` with zero contacting particles.
    """
    sim = IceDEM3D(DEM3DConfig(
        nx=N_SIDE, ny=N_SIDE, nz=N_Z, radius=0.025, density=917.0,
        bond_stiffness=2000.0, contact_stiffness=5000.0, breaking_strain=0.015,
        platen_stiffness=5000.0, platen_damping=GAMMA_P,
        top_platen_enabled=True, bottom_platen_enabled=True,
        top_platen_velocity=-V_PLATEN, bottom_platen_velocity=V_PLATEN,
        top_platen_gap=0.5, bottom_platen_gap=0.5,
    ))
    sim.tool_start[:] = torch.tensor([1.0e6, 1.0e6, 1.0e6], dtype=sim.dtype)
    sim.tool_velocity.zero_()
    row = sim.diagnostics()
    n = int(sim.positions.shape[0])
    contacting = int((sim.positions[:, 2] + sim.config.radius > sim.top_platen_start_z).sum())
    return {
        "n_particles": n,
        "n_contacting": contacting,
        "reported_reaction_N": float(row["top_platen_reaction_z_N"]),
        "closed_form_unmasked_N": n * GAMMA_P * V_PLATEN,
        "platen_contact_energy_J": float(row["platen_contact_energy_J"]),
    }


def restitution_ladder(dampings=(0.0, 2.0, 5.0, 10.0)) -> list[dict[str, float]]:
    rows = []
    for c in dampings:
        m = measure_platen_restitution(damping=c)
        mass = getattr(m, "particle_mass", 917.0 * 4.0 / 3.0 * math.pi * 0.025**3)
        rows.append({
            "damping": c,
            "e_measured": m.measured_restitution,
            "e_clamped": float(getattr(m, "analytic_restitution_clamped")),
            "e_unclamped": float(getattr(m, "analytic_restitution")),
            "rel_err_clamped": m.relative_error_clamped,
            "rel_err_unclamped": m.relative_error,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if not args.merge:
        payload = {"tag": args.tag, "fictitious": fictitious_reaction(),
                   "restitution": restitution_ladder()}
        Path(f"/tmp/platen_audit_{args.tag}.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return

    base = json.loads(Path("/tmp/platen_audit_baseline.json").read_text())
    masked = json.loads(Path("/tmp/platen_audit_masked.json").read_text())
    worst = max(abs(r["rel_err_clamped"]) for r in base["restitution"])
    worst_masked = max(abs(r["rel_err_clamped"]) for r in masked["restitution"])
    audit = {
        "baseline": base, "masked": masked,
        "summary": {
            "fictitious_reaction_baseline_N": base["fictitious"]["reported_reaction_N"],
            "fictitious_reaction_masked_N": masked["fictitious"]["reported_reaction_N"],
            "closed_form_unmasked_N": base["fictitious"]["closed_form_unmasked_N"],
            "worst_rel_err_clamped_baseline": worst,
            "worst_rel_err_clamped_masked": worst_masked,
            "mask_alone_within_3pct": bool(worst_masked < 0.03),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2))
    _plot(audit)
    print(json.dumps(audit["summary"], indent=2))


def _plot(audit: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for tag, style in (("baseline", "o-"), ("masked", "s--")):
        rows = audit[tag]["restitution"]
        ax.plot([r["damping"] for r in rows], [r["e_measured"] for r in rows],
                style, label=f"{tag} measured")
    rows = audit["masked"]["restitution"]
    ax.plot([r["damping"] for r in rows], [r["e_clamped"] for r in rows], "k-", lw=1,
            label="clamped closed form")
    ax.plot([r["damping"] for r in rows], [r["e_unclamped"] for r in rows], "k:", lw=1,
            label="unclamped closed form")
    ax.set_xlabel("platen damping c [N s/m]")
    ax.set_ylabel("measured restitution e")
    ax.set_title("Platen restitution rig vs closed forms")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = ["closed form\n(unmasked law)", "measured\nbaseline", "measured\nmasked"]
    vals = [audit["baseline"]["fictitious"]["closed_form_unmasked_N"],
            audit["baseline"]["fictitious"]["reported_reaction_N"],
            audit["masked"]["fictitious"]["reported_reaction_N"]]
    bars = ax.bar(labels, vals, color=["#888", "#c44", "#4a4"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("reported top platen reaction [N]")
    ax.set_title("Fictitious reaction, 0 contacting particles")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "audit.png", dpi=140)


if __name__ == "__main__":
    main()
