"""Verification of the Cundall-Strack Coulomb friction cone against analytic anchors.

Model under test
----------------
A contact carries a stored tangential displacement ``delta_t`` (m).  One time
increment is

    delta_t  <-  delta_t + v_t * dt                 (tangential velocity increment)
    |delta_t| <= delta_s,   delta_s = mu * |F_n| / k_t      (Coulomb cone)
    F_t = k_t * delta_t

Analytic anchors used here (block on a plane, constant normal load ``N``,
constant tangential speed ``v``, no rotation):

* **stick**       ``|F_t| = k_t |delta_t|`` exactly while ``|delta_t| < delta_s``.
* **onset**       sliding starts after tangential travel ``delta_s``,
                  i.e. at ``t_on = delta_s / v``.
* **saturation**  while sliding, ``|F_t| = mu * N`` for every step; never above.
* **reversal**    after sliding has saturated, reversing the velocity re-sticks
                  the contact and the force passes through zero after travel
                  ``delta_s`` and reaches ``-mu*N`` after travel ``2 * delta_s``.
* **dissipation** total friction work from rest to tangential travel ``s`` at
                  saturated sliding is ``k_t*delta_s^2/2 + mu*N*(s - delta_s)``.

The same protocol is replayed with the legacy stateless stepper
(:func:`tensordem.contact_tangential.compute_tangential_force` fed with the
unprojected displacement) so the size of the state-projection error is
quantified rather than asserted.

Usage
-----
    python scripts/verify_coulomb_friction_cone.py --output outputs/friction-cone
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from tensordem.contact_tangential import (
    advance_tangential_contact,
    compute_tangential_force,
    integrate_tangential_displacement,
    saturation_displacement,
)

# Screening threshold for every analytic anchor (relative, dimensionless).
DEFAULT_TOLERANCE = 0.03


@dataclass(frozen=True)
class FrictionConeParams:
    """Constant-normal-load block-on-plane parameters (SI units)."""

    normal_force: float = 100.0      # N, constant normal load on the contact
    k_t: float = 1.0e4               # N/m, tangential stiffness
    mu: float = 0.3                  # -, Coulomb friction coefficient
    dt: float = 1.0e-4               # s, time step
    speed: float = 0.02              # m/s, imposed tangential speed
    slide_time: float = 0.6          # s, duration of each sliding phase

    @property
    def delta_s(self) -> float:
        """Cone radius in displacement space, m."""
        return self.mu * self.normal_force / self.k_t

    @property
    def f_sat(self) -> float:
        """Saturated tangential force magnitude, N."""
        return self.mu * self.normal_force

    def as_dict(self) -> dict[str, float]:
        return {
            "normal_force_N": self.normal_force,
            "k_t_N_per_m": self.k_t,
            "mu": self.mu,
            "dt_s": self.dt,
            "speed_m_per_s": self.speed,
            "slide_time_s": self.slide_time,
            "delta_s_m": self.delta_s,
            "f_sat_N": self.f_sat,
        }


_NORMAL = torch.tensor([[1.0, 0.0]], dtype=torch.float64)


def _step(
    disp: torch.Tensor,
    velocity: torch.Tensor,
    params: FrictionConeParams,
    *,
    legacy: bool,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Advance one step; returns (tangential force, new displacement, sliding)."""
    f_n = torch.tensor([params.normal_force], dtype=torch.float64)
    if legacy:
        disp_raw = integrate_tangential_displacement(disp, velocity, _NORMAL, params.dt)
        force, _raw = compute_tangential_force(_NORMAL, disp_raw, f_n, params.k_t, params.mu)
        return force, disp_raw, False
    force, disp_new, sliding = advance_tangential_contact(
        disp, velocity, _NORMAL, f_n, params.k_t, params.mu, params.dt
    )
    return force, disp_new, bool(sliding[0])


def _travel_at_level(samples: list[tuple[float, float]], level: float) -> float | None:
    """First tangential travel where ``F_y`` reaches ``level`` (linear interp)."""
    for (s0, f0), (s1, f1) in zip(samples, samples[1:]):
        if (f0 - level) * (f1 - level) <= 0.0 and f0 != f1:
            frac = (level - f0) / (f1 - f0)
            return s0 + frac * (s1 - s0)
    return None


def _relative_error(value: float | None, reference: float) -> float | None:
    if value is None or reference == 0.0:
        return None
    return abs(value - reference) / abs(reference)


def run_single_phase(
    params: FrictionConeParams, *, legacy: bool = False
) -> dict[str, object]:
    """Monotonic sliding phase: stick -> onset -> saturation, plus dissipation."""
    disp = torch.zeros((1, 2), dtype=torch.float64)
    velocity = torch.tensor([[0.0, params.speed]], dtype=torch.float64)
    steps = int(round(params.slide_time / params.dt))

    history: list[dict[str, float]] = []
    travel = 0.0
    work = 0.0
    max_ratio = 0.0
    onset_travel: float | None = None
    for k in range(steps):
        force, disp, sliding = _step(disp, velocity, params, legacy=legacy)
        fy = float(force[0, 1])
        work += fy * params.speed * params.dt
        travel += params.speed * params.dt
        max_ratio = max(max_ratio, abs(fy) / params.f_sat)
        if legacy:
            onset_travel = None  # legacy clamps the force only, it has no slide state
        elif onset_travel is None and sliding:
            onset_travel = travel
        history.append(
            {
                "step": k + 1,
                "time_s": (k + 1) * params.dt,
                "travel_m": travel,
                "delta_t_m": float(torch.linalg.vector_norm(disp[0]).item()),
                "F_t_N": fy,
                "sliding": float(sliding),
            }
        )

    s_end = params.speed * params.slide_time
    work_analytic = 0.5 * params.k_t * params.delta_s ** 2 + params.f_sat * (
        s_end - params.delta_s
    )
    reference_onset = params.delta_s if not legacy else None
    case = {
        "case": "monotonic_sliding",
        "stepper": "legacy_stateless" if legacy else "friction_cone_projected",
        "analytic_onset_travel_m": reference_onset,
        "measured_onset_travel_m": onset_travel,
        "onset_travel_rel_error": _relative_error(onset_travel, params.delta_s)
        if not legacy
        else None,
        "max_force_over_saturation": max_ratio,
        "force_ceiling_respected": max_ratio <= 1.0 + 1e-12,
        "final_force_N": history[-1]["F_t_N"],
        "analytic_final_force_N": params.f_sat,
        "final_force_rel_error": _relative_error(history[-1]["F_t_N"], params.f_sat),
        "travel_m": s_end,
        "friction_work_J": work,
        "analytic_friction_work_J": work_analytic,
        "friction_work_rel_error": _relative_error(work, work_analytic),
        "steps": steps,
    }
    return {"case": case, "history": history}


def run_stick_phase(params: FrictionConeParams) -> dict[str, object]:
    """No tangential velocity: the tangential force is exactly k_t * delta_t."""
    disp0 = 1.0e-3
    disp = torch.tensor([[0.0, disp0]], dtype=torch.float64)
    velocity = torch.zeros((1, 2), dtype=torch.float64)
    force, disp_new, sliding = _step(disp, velocity, params, legacy=False)
    analytic = params.k_t * disp0
    measured = float(force[0, 1])
    return {
        "case": "stick_no_slip",
        "analytic_force_N": analytic,
        "measured_force_N": measured,
        "force_rel_error": _relative_error(measured, analytic),
        "displacement_unchanged": bool(
            torch.allclose(disp_new, disp, rtol=0.0, atol=1e-18)
        ),
        "sliding_reported": bool(sliding),
        "below_cone": bool(disp0 < params.delta_s),
    }


def run_reversal_phase(
    params: FrictionConeParams, *, legacy: bool = False
) -> dict[str, object]:
    """Saturate, then reverse: measure re-stick and force-reversal travel."""
    disp = torch.zeros((1, 2), dtype=torch.float64)
    forward = torch.tensor([[0.0, params.speed]], dtype=torch.float64)
    backward = torch.tensor([[0.0, -params.speed]], dtype=torch.float64)
    steps = int(round(params.slide_time / params.dt))

    for _ in range(steps):
        _force, disp, _sliding = _step(disp, forward, params, legacy=legacy)

    samples: list[tuple[float, float]] = [(0.0, float((params.k_t * disp[0, 1]).item()))]
    travel = 0.0
    history: list[dict[str, float]] = []
    for k in range(2 * steps):
        force, disp, sliding = _step(disp, backward, params, legacy=legacy)
        travel += params.speed * params.dt
        fy = float(force[0, 1])
        samples.append((travel, fy))
        history.append(
            {
                "step": k + 1,
                "travel_back_m": travel,
                "F_t_N": fy,
                "delta_t_m": float(torch.linalg.vector_norm(disp[0]).item()),
                "sliding": float(sliding),
            }
        )

    # Level slightly inside -mu*N so the linear-interpolation crossing is robust
    # while staying within 0.1% of the analytic saturation value.
    level = -params.f_sat * (1.0 - 1e-3)
    zero_travel = _travel_at_level(samples, 0.0)
    neg_travel = _travel_at_level(samples, level)
    return {
        "case": "velocity_reversal",
        "stepper": "legacy_stateless" if legacy else "friction_cone_projected",
        "analytic_zero_crossing_travel_m": params.delta_s,
        "measured_zero_crossing_travel_m": zero_travel,
        "zero_crossing_rel_error": _relative_error(zero_travel, params.delta_s),
        "analytic_minus_saturation_travel_m": 2.0 * params.delta_s,
        "measured_minus_saturation_travel_m": neg_travel,
        "minus_saturation_rel_error": _relative_error(neg_travel, 2.0 * params.delta_s),
        "saturation_level_offset": 1e-3,
        "history": history,
        "samples": samples[1:],
    }


def run_study(output_dir: Path | None = None) -> dict[str, object]:
    """Run all verification cases and optionally write JSON/CSV artifacts."""
    params = FrictionConeParams()

    stick = run_stick_phase(params)
    monotonic = run_single_phase(params)
    legacy_monotonic = run_single_phase(params, legacy=True)
    reversal = run_reversal_phase(params)
    legacy_reversal = run_reversal_phase(params, legacy=True)

    checks = {
        "stick_force_rel_error": stick["force_rel_error"],
        "onset_travel_rel_error": monotonic["case"]["onset_travel_rel_error"],
        "final_force_rel_error": monotonic["case"]["final_force_rel_error"],
        "friction_work_rel_error": monotonic["case"]["friction_work_rel_error"],
        "zero_crossing_rel_error": reversal["zero_crossing_rel_error"],
        "minus_saturation_rel_error": reversal["minus_saturation_rel_error"],
    }
    max_error = max(abs(v) for v in checks.values() if v is not None)
    legacy_max_error = max(
        abs(v)
        for v in (
            legacy_monotonic["case"]["final_force_rel_error"],
            legacy_monotonic["case"]["friction_work_rel_error"],
            legacy_reversal["zero_crossing_rel_error"],
            legacy_reversal["minus_saturation_rel_error"],
        )
        if v is not None
    )

    report = {
        "parameters": params.as_dict(),
        "tolerance_relative": DEFAULT_TOLERANCE,
        "cases": {
            "stick_no_slip": stick,
            "monotonic_sliding": monotonic["case"],
            "velocity_reversal": {
                k: v for k, v in reversal.items() if k not in ("history", "samples")
            },
            "legacy_monotonic_sliding": legacy_monotonic["case"],
            "legacy_velocity_reversal": {
                k: v for k, v in legacy_reversal.items() if k not in ("history", "samples")
            },
        },
        "error_budget": checks,
        "max_relative_error": max_error,
        "legacy_max_relative_error": legacy_max_error,
        "verdict": "PASS" if max_error <= DEFAULT_TOLERANCE else "FAIL",
        "reference": (
            "Analytic Coulomb/Cundall-Strack anchors: delta_s = mu*N/k_t, "
            "s_zero = delta_s, s_neg = 2*delta_s, W = k_t*delta_s^2/2 + mu*N*(s-delta_s)"
        ),
    }

    if output_dir is not None:
        _write_artifacts(Path(output_dir), report, monotonic, reversal, legacy_reversal)
    return report


def _write_csv(path: Path, rows: list[dict[str, float]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_artifacts(
    output_dir: Path,
    report: dict[str, object],
    monotonic: dict[str, object],
    reversal: dict[str, object],
    legacy_reversal: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "friction_cone_verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    summary = report["cases"]
    _write_csv(
        output_dir / "friction_cone_summary.csv",
        [dict({"case": name}, **values) for name, values in summary.items()],
        ["case"]
        + sorted({k for values in summary.values() for k in values}),
    )

    _write_csv(
        output_dir / "monotonic_sliding_history.csv",
        monotonic["history"],
        ["step", "time_s", "travel_m", "delta_t_m", "F_t_N", "sliding"],
    )
    for name, payload in (
        ("reversal_history.csv", reversal),
        ("legacy_reversal_history.csv", legacy_reversal),
    ):
        _write_csv(
            output_dir / name,
            payload["history"],
            ["step", "travel_back_m", "F_t_N", "delta_t_m", "sliding"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/friction-cone"),
        help="directory for JSON/CSV artifacts",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="also write a PNG comparing projected vs legacy reversal (needs matplotlib)",
    )
    args = parser.parse_args()

    report = run_study(args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    if report["verdict"] != "PASS":
        raise SystemExit("friction-cone verification FAILED: error above tolerance")
    if args.plot:
        _plot(args.output, report)


def _plot(output_dir: Path, report: dict[str, object]) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = FrictionConeParams()
    reversal = run_reversal_phase(params)
    legacy = run_reversal_phase(params, legacy=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=150)
    for payload, label, style in (
        (reversal, "projected cone state (verified)", "-"),
        (legacy, "legacy unprojected state", "--"),
    ):
        travel = [0.0] + [p[0] for p in payload["samples"]]
        force = [params.f_sat] + [p[1] for p in payload["samples"]]
        axes[0].plot(travel, force, style, label=label)
    axes[0].axhline(-params.f_sat, color="grey", lw=0.8, ls=":")
    axes[0].axhline(params.f_sat, color="grey", lw=0.8, ls=":")
    axes[0].axvline(params.delta_s, color="tab:red", lw=0.8, ls=":")
    axes[0].axvline(2.0 * params.delta_s, color="tab:red", lw=0.8, ls=":")
    axes[0].set_xlabel("backward tangential travel after saturation (m)")
    axes[0].set_ylabel("tangential force F_t (N)")
    axes[0].set_title("Velocity reversal: re-stick travel")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    monotonic = run_single_phase(params)
    steps = [row["travel_m"] for row in monotonic["history"]]
    axes[1].plot(steps, [row["F_t_N"] for row in monotonic["history"]], label="F_t (N)")
    axes[1].axhline(params.f_sat, color="grey", lw=0.8, ls=":")
    axes[1].axvline(params.delta_s, color="tab:red", lw=0.8, ls=":")
    axes[1].set_xlabel("tangential travel (m)")
    axes[1].set_ylabel("tangential force F_t (N)")
    axes[1].set_title("Sliding onset at delta_s = mu N / k_t")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "friction_cone_verification.png")
    plt.close(fig)


if __name__ == "__main__":
    main()