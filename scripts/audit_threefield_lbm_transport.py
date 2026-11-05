"""Audit the transport and the Smagorinsky closure of the D3Q19 core in wake_lbm.

Runs four independent checks and writes ``docs/threefield-lbm-transport/``:
``audit.json`` (machine-readable), ``closure_sweep.csv``, ``shear_wave_sweep.csv``
and ``figures/*.png``.

Checks
------
A. streaming exactness   -- ``WakeLBM._stream`` (padded halo) vs an independent
                            ``torch.roll`` periodic shift: expected bit-exact.
B. momentum diffusivity  -- decaying shear wave ``u_x = A cos(k y)``, an exact
                            Navier-Stokes solution, vs ``nu = (tau0-1/2)/3``.
C. Smagorinsky closure   -- realised ``nu_t`` vs ``Cs^2 |S|`` on synthetic
                            Chapman-Enskog fields (pure shear / normal / mixed).
D. subgrid-free limit    -- ``Cs = 0`` must return ``tau0`` exactly, and a
                            vanishing strain rate must leave ``tau_eff -> tau0``.

Usage: ``python scripts/audit_threefield_lbm_transport.py``
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

from tensordem.threefield.lbm_transport import (  # noqa: E402
    double_symmetric_contraction,
    periodic_stream,
    reduced_symmetric_contraction,
    reference_stream,
    sgs_eddy_viscosity,
    sgs_fixed_point_tau,
    shear_wave_decay,
    strain_invariants,
    synthetic_strain_field,
    target_eddy_viscosity,
)
from tensordem.threefield.wake_lbm import LBMConfig, WakeLBM  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "threefield-lbm-transport")


def _instance(tau0: float = 0.6, cs: float = 0.1, n: int = 12) -> WakeLBM:
    return WakeLBM(LBMConfig(nx=n, ny=n, nz=n, tau0=tau0, smagorinsky_cs=cs,
                             disk_radius=1.0, disk_x=8, disk_thickness=1))


def check_streaming() -> dict:
    inst = _instance()
    torch.manual_seed(0)
    f = torch.rand((inst.nx, inst.ny, inst.nz, 19), dtype=torch.float64)
    err = float((periodic_stream(inst, f) - reference_stream(inst, f)).abs().max())
    return {"max_abs_difference": err, "verdict": "exact" if err == 0.0 else "MISMATCH"}


def check_viscosity() -> list[dict]:
    rows = []
    for tau0 in (0.55, 0.6, 0.7, 0.8, 1.0, 1.4):
        res = shear_wave_decay(tau0=tau0, size=16, mode=1, steps=120)
        rows.append({
            "tau0": tau0,
            "wavenumber": res.wavenumber,
            "rate_measured": res.rate_measured,
            "rate_continuum": res.rate_continuum,
            "rate_discrete": res.rate_discrete,
            "nu_measured": res.viscosity_measured,
            "nu_nominal": res.viscosity_nominal,
            "error_vs_continuum": res.rate_measured / res.rate_continuum - 1.0,
            "error_vs_discrete": res.rate_measured / res.rate_discrete - 1.0,
            "discrete_laplacian_ratio": res.discrete_laplacian,
        })
    return rows


def check_closure() -> list[dict]:
    gradients = {
        "pure_shear_S12": [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "jet_shear_du_x_dy": [[0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "uniaxial_normal": [[1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, -0.5]],
        "plane_strain": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
        "mixed_shear_normal": [[0.3, 0.6, 0.0], [0.0, -0.3, 0.0], [0.0, 0.0, 0.0]],
    }
    rows = []
    for cs in (0.05, 0.1, 0.16):
        inst = _instance(cs=cs)
        tau0 = inst.config.tau0
        for name, gradient in gradients.items():
            inv = strain_invariants(gradient)
            tau_fp = sgs_fixed_point_tau(tau0, cs, inv["strain"])
            f = synthetic_strain_field(inst, gradient, tau=tau_fp)
            rho, u = inst._macroscopic(f)
            f_neq = f - inst._equilibrium(rho.unsqueeze(-1), u)
            pi = torch.einsum("ik,...i->...k", inst._outer, f_neq)
            full = float(double_symmetric_contraction(pi)[0, 0, 0])
            reduced = float(reduced_symmetric_contraction(pi)[0, 0, 0])
            tau_eff = inst._effective_tau(f, rho, u)
            nu_t = float(sgs_eddy_viscosity(float(tau_eff[0, 0, 0]), tau0))
            target = target_eddy_viscosity(cs, inv["strain"])
            rows.append({
                "cs": cs,
                "field": name,
                "strain": inv["strain"],
                "shear_fraction": inv["shear"] / inv["strain"] if inv["strain"] else 0.0,
                "pi_full": full,
                "pi_reduced": reduced,
                "reduced_over_full": reduced / full if full else 0.0,
                "tau_eff": float(tau_eff[0, 0, 0]),
                "tau_fixed_point": tau_fp,
                "nu_t": nu_t,
                "nu_t_target": target,
                "error": nu_t / target - 1.0 if target else 0.0,
            })
    return rows


def check_limits() -> dict:
    inst = _instance(cs=0.0)
    f = inst.f
    rho, u = inst._macroscopic(f)
    tau = inst._effective_tau(f, rho, u)
    zero_cs = float((tau - inst.config.tau0).abs().max())
    inst = _instance(cs=0.16)
    tiny = synthetic_strain_field(inst, [[0.0, 1e-12, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], tau=0.6)
    rho, u = inst._macroscopic(tiny)
    tau = inst._effective_tau(tiny, rho, u)
    vanishing = float((tau - inst.config.tau0).abs().max())
    return {
        "cs_zero_max_deviation": zero_cs,
        "vanishing_strain_max_deviation": vanishing,
        "verdict": "exact" if max(zero_cs, vanishing) < 1e-12 else "NONZERO",
    }


def figures(closure: list[dict], viscosity: list[dict]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"matplotlib unavailable ({exc}); skipping figures", file=sys.stderr)
        return []
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    made = []

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for cs in sorted({row["cs"] for row in closure}):
        rows = [row for row in closure if row["cs"] == cs]
        ax.plot([r["strain"] for r in rows], [100 * r["error"] for r in rows], "o-",
                label=f"Cs = {cs}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-3.0, 3.0, color="0.85", zorder=0, label="+/-3% band")
    ax.set_xlabel("strain-rate invariant |S|")
    ax.set_ylabel("realised nu_t error vs Cs^2|S|  [%]")
    ax.set_title("Smagorinsky closure after the m2949 fix")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "figures", "sgs_closure_error.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot([r["tau0"] for r in viscosity], [r["nu_measured"] for r in viscosity], "o",
            label="measured from shear-wave decay")
    ax.plot([r["tau0"] for r in viscosity], [r["nu_nominal"] for r in viscosity], "-",
            label="(tau0 - 1/2)/3")
    ax.set_xlabel("tau0")
    ax.set_ylabel("kinematic viscosity [lattice units]")
    ax.set_title("D3Q19 BGK momentum diffusivity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "figures", "shear_wave_viscosity.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot([r["tau0"] for r in viscosity], [100 * r["error_vs_continuum"] for r in viscosity],
            "s-", label="vs continuum nu k^2")
    ax.plot([r["tau0"] for r in viscosity], [100 * r["error_vs_discrete"] for r in viscosity],
            "o-", label="vs discrete 2(1-cos k) reference")
    ax.axhspan(-3.0, 3.0, color="0.85", zorder=0, label="+/-3% band")
    ax.set_xlabel("tau0")
    ax.set_ylabel("decay-rate error [%]")
    ax.set_title("Shear-wave decay rate vs analytic references")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, "figures", "shear_wave_rate_error.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path)
    return made


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    streaming = check_streaming()
    viscosity = check_viscosity()
    closure = check_closure()
    limits = check_limits()

    with open(os.path.join(OUT, "shear_wave_sweep.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(viscosity[0]))
        writer.writeheader()
        writer.writerows(viscosity)
    with open(os.path.join(OUT, "closure_sweep.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(closure[0]))
        writer.writeheader()
        writer.writerows(closure)

    made = figures(closure, viscosity)
    worst_closure = max(closure, key=lambda row: abs(row["error"]))
    worst_viscosity = max(viscosity, key=lambda row: abs(row["error_vs_continuum"]))
    payload = {
        "module": "tensordem.threefield.lbm_transport",
        "solver": "tensordem.threefield.wake_lbm.WakeLBM",
        "streaming": streaming,
        "momentum_diffusivity": {
            "rows": viscosity,
            "max_abs_error_vs_continuum": abs(worst_viscosity["error_vs_continuum"]),
            "max_abs_error_vs_discrete": max(abs(row["error_vs_discrete"]) for row in viscosity),
            "worst_tau0": worst_viscosity["tau0"],
        },
        "smagorinsky_closure": {
            "rows": closure,
            "max_abs_error": abs(worst_closure["error"]),
            "worst_field": worst_closure["field"],
            "worst_cs": worst_closure["cs"],
        },
        "limits": limits,
        "figures": [os.path.relpath(path, OUT) for path in made],
        "verdict": {
            "streaming_exact": streaming["verdict"] == "exact",
            "viscosity_within_3pct": abs(worst_viscosity["error_vs_continuum"]) < 0.03,
            "closure_within_3pct": abs(worst_closure["error"]) < 0.03,
            "limits_exact": limits["verdict"] == "exact",
        },
    }
    with open(os.path.join(OUT, "audit.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["verdict"], indent=2))
    print(f"max closure error: {abs(worst_closure['error']):.3e} ({worst_closure['field']}, Cs={worst_closure['cs']})")
    print(f"max viscosity error vs continuum: {abs(worst_viscosity['error_vs_continuum']):.3%} (tau0={worst_viscosity['tau0']})")
    print(f"wrote {OUT}")
    return 0 if all(payload["verdict"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())