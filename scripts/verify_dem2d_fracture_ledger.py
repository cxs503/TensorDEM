"""Closed-form audit of the 2-D engine fracture ledger channels.

Report 3817 (m3635) closed the 2-D energy-ledger audit for tool work, damping
dissipation and the balance residual, and named the remaining debt: the
fracture channels ``fracture_release_J`` and ``pair_contact_acquired_J`` are
only covered by the balance identity, which is insensitive to a channel being
scaled by a wrong constant.  This script confronts both channels with
independently coded closed forms:

* A controlled single-bond tensile break at a known extension
  ``xi0 = L0 * (eps_b + delta)``: the ledger must equal ``0.5 * k * xi0**2``
  and the acquired channel must be exactly zero (no overlap in tension).
* A stiffness/extension scaling grid: the ledger must scale as ``k * xi**2``
  (a hard-coded or identity-only channel cannot pass).
* A production tool-indentation run: every break event is re-scored from the
  pre-step geometry (the solver evaluates failure at the pre-step positions),
  so ``released = 0.5 * k * (d - L0)**2`` and
  ``acquired = 0.5 * kc * (2r - d)**2`` (overlap part) are recomputed
  independently of the solver's bookkeeping.
* A sub-threshold negative control: the controlled bond must survive and the
  ledger must stay zero.

Run with ``PYTHONPATH=$PWD/src python3 scripts/verify_dem2d_fracture_ledger.py``.
"""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem import DEMConfig, IceDEM  # noqa: E402

ARTIFACT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "docs", "artifacts", "dem2d_fracture_ledger"
)

R = 0.025
L0 = 2.0 * R
EPS_B = 0.015


def check_provenance() -> str:
    """Return the tensordem source path; fail if it leaks from another tree."""
    import tensordem

    path = os.path.abspath(tensordem.__file__)
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not path.startswith(here + os.sep):
        raise RuntimeError(
            f"provenance failure: tensordem imported from {path}, "
            f"expected under {here}"
        )
    return path


def _single_bond_sim(k: float, xi0: float) -> IceDEM:
    """2x2 grid, one live axial bond (pair 0-1) pre-strained to xi0.

    The tool is parked 1 m above with a 1e-6 m/s speed (one step moves it
    1e-6 m, no contact) and all dissipation is switched off.
    """
    config = DEMConfig(nx=2, ny=2, radius=R, bond_stiffness=k,
                       contact_stiffness=2000.0, breaking_strain=EPS_B,
                       shear_breaking_strain=0.03, contact_damping=0.0,
                       drag=0.0, tool_speed=1e-6, tool_gap=1.0,
                       fix_edges=False)
    sim = IceDEM(config)
    sim.alive = torch.zeros(len(sim.pairs), dtype=torch.bool, device=sim.device)
    mask = (sim.pairs[:, 0] == 0) & (sim.pairs[:, 1] == 1)
    sim.alive[mask] = True
    sim.positions = sim.initial_positions.clone()
    sim.positions[1, 0] += xi0
    sim.velocities.zero_()
    return sim


def _bond01(sim: IceDEM):
    return (sim.pairs[:, 0] == 0) & (sim.pairs[:, 1] == 1)


def case_a_controlled_break(k: float = 2000.0, xi0: float | None = None) -> dict:
    if xi0 is None:
        xi0 = L0 * (EPS_B + 1e-4)
    sim = _single_bond_sim(k, xi0)
    u_before = sim.mechanical_energy()["mechanical_J"]
    ke_before = float(0.5 * sim.config.mass * sim.velocities.square().sum())
    sim.step()
    u_after = sim.mechanical_energy()["mechanical_J"]
    ke_after = float(0.5 * sim.config.mass * sim.velocities.square().sum())
    released = sim.accounting["fracture_release_J"]
    acquired = sim.accounting["pair_contact_acquired_J"]
    closed_released = 0.5 * k * xi0**2
    # Total-energy residual of the single break step (no external work, no
    # damping): R = d(U+KE) + released - acquired.  Engine convention: a bond
    # that breaks in a step exerts no force in that step, so the released
    # elastic energy is dissipated at the break and no KE is generated; the
    # residual must be ~0 (machine precision), not an O((omega dt)^2) term.
    residual = (u_after + ke_after - u_before - ke_before) + released - acquired
    return dict(
        k=k, xi0=xi0, L0=L0, dt=sim.dt,
        controlled_bond_broken=bool(sim.failure_mode[_bond01(sim)].any().item()),
        ledger_released_J=released,
        closed_released_J=closed_released,
        rel_err_released=abs(released - closed_released) / closed_released,
        ledger_acquired_J=acquired,
        recorded_extension_m=float(sim.failure_extension_m[_bond01(sim)].item()),
        energy_residual_J=residual,
        u_before_J=u_before, u_after_J=u_after, ke_after_J=ke_after,
    )


def case_b_scaling() -> list[dict]:
    """Ledger must scale as k * xi^2: the non-identity check."""
    rows = []
    base = 2000.0
    for k in (base, 2.0 * base):
        for xi0 in (0.0008, 0.0016):
            row = case_a_controlled_break(k, xi0)
            row["expected_ratio"] = (k / base) * (xi0 / 0.0008) ** 2
            rows.append(row)
    return rows


def case_c_indentation_events(dt_scale: float = 1.0,
                             duration: float = 0.12) -> dict:
    """Tool indentation; re-score every break event from pre-step geometry."""
    nx, ny = 7, 3
    dt0 = IceDEM(DEMConfig(nx=nx, ny=ny, tool_speed=0.2)).dt
    config = DEMConfig(nx=nx, ny=ny, tool_speed=0.2, dt=dt0 * dt_scale)
    sim = IceDEM(config)
    r = sim.config.radius
    k = sim.config.bond_stiffness
    kc = sim.config.contact_stiffness
    i, j = sim.pairs.T
    events = []
    broken_set: set[int] = set()
    n_steps = int(round(duration / sim.dt))
    for _ in range(n_steps):
        pre = sim.positions.clone()
        d_pre = torch.linalg.vector_norm(pre[j] - pre[i], dim=1)
        sim.step()
        now_broken = {int(x) for x in
                     torch.nonzero(sim.bonded & ~sim.alive).flatten().tolist()}
        for idx in sorted(now_broken - broken_set):
            broken_set.add(idx)
            d = float(d_pre[idx])
            l0 = float(sim.rest_lengths[idx])
            events.append(dict(
                pair=[int(i[idx]), int(j[idx])],
                distance_m=d, rest_length_m=l0,
                extension_m=d - l0,
                overlap_m=max(0.0, 2.0 * r - d),
                closed_released_J=0.5 * k * (d - l0) ** 2,
                closed_acquired_J=0.5 * kc * max(0.0, 2.0 * r - d) ** 2,
                recorded_energy_J=float(sim.failure_energy_J[idx]),
                mode=int(sim.failure_mode[idx]),
            ))
    acc = sim.accounting
    closed_released = sum(e["closed_released_J"] for e in events)
    closed_acquired = sum(e["closed_acquired_J"] for e in events)
    return dict(
        dt=sim.dt, n_steps=n_steps, n_events=len(events),
        n_overlap_events=sum(1 for e in events if e["overlap_m"] > 0.0),
        ledger_released_J=acc["fracture_release_J"],
        closed_released_J=closed_released,
        rel_err_released=(abs(acc["fracture_release_J"] - closed_released)
                          / closed_released if closed_released else 0.0),
        ledger_acquired_J=acc["pair_contact_acquired_J"],
        closed_acquired_J=closed_acquired,
        rel_err_acquired=(abs(acc["pair_contact_acquired_J"] - closed_acquired)
                          / closed_acquired if closed_acquired else 0.0),
        events=events,
    )


def case_d_subthreshold() -> dict:
    """Strain just below the threshold must never break the bond."""
    xi0 = 0.99 * L0 * EPS_B
    sim = _single_bond_sim(2000.0, xi0)
    for _ in range(50):
        sim.step()
    return dict(controlled_bond_alive=bool(sim.alive[_bond01(sim)].item()),
                ledger_released_J=sim.accounting["fracture_release_J"],
                ledger_acquired_J=sim.accounting["pair_contact_acquired_J"])


def main() -> dict:
    src = check_provenance()
    torch.set_num_threads(1)
    a = case_a_controlled_break()
    b = case_b_scaling()
    c = case_c_indentation_events()
    d = case_d_subthreshold()
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    summary = dict(
        provenance=src,
        case_a=a,
        case_b=b,
        case_c={key: val for key, val in c.items() if key != "events"},
        case_d=d,
    )
    with open(os.path.join(ARTIFACT_DIR, "fracture_ledger_closure.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(ARTIFACT_DIR, "fracture_events.csv"), "w") as fh:
        fh.write("pair_i,pair_j,distance_m,rest_length_m,extension_m,overlap_m,"
                 "closed_released_J,closed_acquired_J,recorded_energy_J,mode\n")
        for e in c["events"]:
            fh.write(f"{e['pair'][0]},{e['pair'][1]},{e['distance_m']:.9f},"
                     f"{e['rest_length_m']:.9f},{e['extension_m']:.9f},"
                     f"{e['overlap_m']:.9f},{e['closed_released_J']:.9e},"
                     f"{e['closed_acquired_J']:.9e},{e['recorded_energy_J']:.9e},"
                     f"{e['mode']}\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        ax = axes[0]
        for row in b:
            ax.plot(row["closed_released_J"], row["ledger_released_J"], "o")
        lim = max(row["closed_released_J"] for row in b) * 1.2
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlabel("closed form 0.5 k xi^2 [J]")
        ax.set_ylabel("ledger fracture_release_J [J]")
        ax.set_title("A/B: channel follows k*xi^2")
        ax = axes[1]
        eng = [e["recorded_energy_J"] for e in c["events"]]
        clo = [e["closed_released_J"] for e in c["events"]]
        ax.plot(clo, eng, "o")
        lim = max(max(clo), max(eng)) * 1.2 if clo else 1.0
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlabel("closed form per event [J]")
        ax.set_ylabel("recorded failure_energy_J [J]")
        ax.set_title(f"C: {c['n_events']} events, "
                    f"{c['n_overlap_events']} with overlap")
        ax = axes[2]
        for scale in (1.0, 0.5, 0.25):
            cc = case_c_indentation_events(dt_scale=scale)
            ax.plot(scale, max(cc["rel_err_released"], 1e-18), "o-")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("dt scale")
        ax.set_ylabel("released rel. err")
        ax.set_title("C: event audit vs dt")
        fig.tight_layout()
        fig.savefig(os.path.join(ARTIFACT_DIR, "fracture_ledger_closure.png"),
                   dpi=110)
        plt.close(fig)
    except Exception as exc:  # figures are optional for the numeric audit
        print(f"figure skipped: {exc}")
    print(json.dumps({key: val for key, val in summary.items()
                      if key != "case_b"}, indent=2, default=float))
    return summary


if __name__ == "__main__":
    main()
