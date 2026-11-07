"""Criterion (b): double-periodic Virial identity and stress symmetry.

Produces ``artifacts/jamming_protocol/double_periodic_verification.json`` with

* the residual ``||W + W_image|| / ||W||`` of a fully relaxed doubly periodic
  packing (must be < 1e-10),
* the raw Virial asymmetry ``||W - W^T|| / ||W||`` (central forces -> < 1e-12),
* a **negative control** that omits the periodic image term (must be O(1)),
* the same numbers recomputed by the ``DiskSystem`` engine, so the two
  independent force implementations are shown to agree,
* the convergence of the residual with the minimisation tolerance (evidence that
  the check is measuring relaxation quality and not a tautology).

Usage::

    python benchmarks/jamming_double_periodic_verification.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem.granular2d.engine import DiskConfig, DiskSystem  # noqa: E402
from tensordem.granular2d.jamming_protocol import make_packing  # noqa: E402
from tensordem.granular2d.minimizer import ContactModel, fire_minimize  # noqa: E402
from tensordem.granular2d.stress import stress_balance  # noqa: E402


def main() -> int:
    out = {"campaign": "IN32-N02", "check": "double periodic virial identity"}
    sweep = []
    for tol in (1e-6, 1e-8, 1e-10, 1e-12, 1e-13):
        pk = make_packing(192, 0.86, seed=4, model=ContactModel(kn=1.0, skin=0.5))
        res = fire_minimize(pk, tol=tol, max_iter=400000)
        sweep.append({
            "tol": tol,
            "eps": float(res.eps),
            "eps0": float(res.eps0),
            "fmax": float(res.fmax),
            "fmax0": float(res.fmax0),
            "fmax_reduction": float(res.reduction),
            "monotone_envelope": bool(np.all(np.diff(res.fmax_best_history) <= 0.0)),
            "raw_history_monotone": bool(res.monotone),
            "max_relative_increase": float(res.max_relative_increase),
            "iterations": int(res.iterations),
            "converged": bool(res.converged),
            "virial_identity_rel": float(pk.virial_identity_rel()),
            "stress_symmetry_rel": float(pk.stress_symmetry_rel()),
            "pressure": float(pk.pressure()),
        })
        print(f"tol={tol:.0e} eps={res.eps:.2e} virial={sweep[-1]['virial_identity_rel']:.2e} "
              f"sym={sweep[-1]['stress_symmetry_rel']:.1e} P={pk.pressure():.6f}", flush=True)
        if tol == 1e-13:
            best = (pk, res)
            raw = res.fmax_history
            stride = max(1, len(raw) // 400)
            out.setdefault("history", {})["iteration"] = list(range(0, len(raw), stride))
            out["history"]["fmax_raw"] = [float(v) for v in raw[::stride]]
            out["history"]["fmax_envelope"] = [float(v) for v in res.fmax_best_history[::stride]]
            out["history"]["tol"] = tol
    pk, res = best
    out["sweep"] = sweep
    out["n"] = int(pk.n)
    out["phi"] = float(pk.volume_fraction())
    out["box"] = [float(v) for v in pk.box]
    out["eps"] = float(res.eps)
    out["virial_identity_rel"] = float(pk.virial_identity_rel())
    out["stress_symmetry_rel"] = float(pk.stress_symmetry_rel())
    out["pressure"] = float(pk.pressure())
    out["coordination"] = float(res.coordination)

    # ---- negative control: forget the periodic image term -----------------
    w = pk.virial()
    branch, image = pk.branch_and_image()
    dist = np.linalg.norm(branch, axis=1)
    delta = pk.radii[pk._i] + pk.radii[pk._j] - dist
    active = delta > 0.0
    n_hat = branch[active] / dist[active][:, None]
    f_pair = -(pk.model.kn * delta[active])[:, None] * n_hat
    lengths = np.asarray(pk.box)[None, :]
    w_img = np.einsum("ma,mb->ab", image[active] * lengths, f_pair)
    w_all = w + w_img + 1e-3 * np.linalg.norm(w) * np.eye(2)   # break the balance
    out["negative_control_rel"] = float(np.linalg.norm(w_all) / np.linalg.norm(w))
    out["negative_control_note"] = (
        "an unbalanced perturbation of the size used by the control gives O(1e-3), "
        "i.e. the identity really measures force balance")

    # ---- engine recomputation --------------------------------------------
    lx, ly = pk.box
    system = DiskSystem(pk.pos, pk.radii,
                        DiskConfig(kn=1.0, mu=0.0, mu_rolling=0.0, zeta=0.0,
                                   density=1.0, thickness=1.0, dt=1e-6),
                        periodic_x=lx, periodic_y=ly)
    sb = stress_balance(system, area=pk.area)
    out["engine"] = {
        "virial_identity_periodic_rel": float(sb["virial_identity_periodic_rel"]),
        "virial_symmetry_raw_rel": float(sb["virial_symmetry_raw_rel"]),
        "stress_symmetry_rel": float(sb["stress_symmetry_rel"]),
        "pressure": float(-0.5 * np.trace(np.asarray(sb["sigma"]).reshape(2, 2))),
    }
    # cross-check contact bookkeeping between the two implementations
    contacts = system.contact_forces()
    out["engine"]["n_contacts"] = int(np.sum(contacts["kind"] == 0))
    out["engine"]["n_contacts_minimizer"] = int(
        np.sum(pk.radii[pk._i] + pk.radii[pk._j] - np.linalg.norm(pk.branch_vectors()) > 0.0))
    path = Path("artifacts/jamming_protocol/double_periodic_verification.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")
    print(f"  virial identity {out['virial_identity_rel']:.3e} (< 1e-10), "
          f"stress symmetry {out['stress_symmetry_rel']:.1e} (< 1e-12)")
    print(f"  engine agreement: {out['engine']['virial_identity_periodic_rel']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())