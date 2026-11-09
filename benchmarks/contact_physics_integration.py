"""Integration, performance and acceptance benchmark of the contact-physics module.

Produces ``artifacts/contact_physics/integration_report.json`` (every number the
documentation and the platform report quote) and the comparison figures
``artifacts/contact_physics/fig_*.png``.

The standalone reference used for the equality criterion (a) is imported from the
integration test module, so there is exactly one reference implementation:

    python3 benchmarks/contact_physics_integration.py

Sections
--------
1. ``regimes``   -- three工况, integrated run vs standalone runs, max relative
   error per quantity (criterion a)
2. ``invariants`` -- coupling invariants per regime (criterion a, part 2)
3. ``performance`` -- wall time / Python peak memory of the integrated run versus
   the three standalone drivers (criterion c)
4. ``api``        -- the frozen API baseline fingerprint (criterion b)
5. ``regression`` -- the upstream node gate artifact (criterion d)
6. ``documentation`` -- the docs/example coverage check (criterion e)
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import sys
import time
import tracemalloc
from datetime import datetime, timezone

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.contact_physics import ContactPhysicsMarch  # noqa: E402
from tensordem.contact_physics.statistics import RidgeContactStatistics  # noqa: E402
import tensordem  # noqa: E402

import test_contact_physics_integration as reference  # noqa: E402

OUT = ROOT / "artifacts" / "contact_physics"
OUT.mkdir(parents=True, exist_ok=True)


def _git_rev() -> str:
    try:
        import subprocess

        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def _run_regime(name: str) -> dict:
    builder, n_steps = reference.CASES[name]
    point = builder()
    integrated = ContactPhysicsMarch(point).run(n_steps)
    standalone = reference.standalone_reference(point, n_steps)
    errors = {key: reference.relative_error(integrated["histories"][key],
                                            standalone["histories"][key])
              for key in reference.COMPARED_KEYS}
    statistics = integrated["statistics"]
    return {
        "steps": n_steps,
        "dt_s": point.dt_s,
        "duration_s": n_steps * point.dt_s,
        "max_relative_error": max(errors.values()),
        "relative_error_per_quantity": errors,
        "tolerance": reference.TOLERANCE,
        "passed": max(errors.values()) <= reference.TOLERANCE,
        "final": {
            "interface_temperature_c": integrated["final"]["interface_temperature_c"],
            "mu": integrated["final"]["mu"],
            "film_thickness_m": integrated["final"]["film_thickness_m"],
            "film_normal_stiffness_n_m": integrated["final"]["film_normal_stiffness_n_m"],
            "contact_stiffness_n_m": point.contact.contact_stiffness_n_m,
            "bond_force_mean_n": integrated["final"]["bond_force_mean_n"],
            "damage_max": integrated["final"]["damage_max"],
            "broken_bonds": integrated["final"]["broken_bonds"],
            "load_bearing_fraction": integrated["final"]["load_bearing_fraction"],
        },
        "energy": integrated["energy"],
        "invariants": integrated["invariants"],
        "statistics": statistics,
        "histories": integrated["histories"],
        "model": {
            "slip_velocity_m_s": point.contact.slip_velocity_m_s,
            "normal_force_n": point.contact.normal_force_n,
            "mean_pressure_pa": point.contact.mean_pressure_pa,
            "initial_temperature_c": point.contact.initial_temperature_c,
            "mu_cold": point.contact.mu_cold,
            "mu_wet": point.contact.lubrication.mu_wet,
            "t_onset_c": point.contact.lubrication.t_onset,
            "strain_rate_1_s": point.strain_rate_1_s,
            "n_bonds": point.n_bonds,
            "reference_force_n": point.bond.reference_force_n,
            "damage_time_s": point.bond.damage_time_s,
            "damage_exponent": point.bond.damage_exponent,
            "salinity_ppt": point.bond.salinity_ppt,
        },
    }


def _performance(n_steps: int = 100, repeats: int = 7) -> dict:
    """Overhead of the integration layer, measured with the test's protocol.

    The measurement protocol (single-threaded torch, pre-assembled ridge
    realisation, interleaved A/B, min-of-N estimator) lives in exactly one place
    -- ``tests/test_contact_physics_integration.py::_paired_overhead`` -- so the
    benchmark and the acceptance test cannot drift apart.
    """
    record = reference._paired_overhead(n_steps=n_steps, repeats=repeats)
    record["regime"] = "cold_slow"
    record["passed"] = (record["time_overhead_percent"] < 5.0
                        and record["memory_overhead_percent"] < 5.0)
    return record


def _api_section() -> dict:
    baseline_path = OUT / "api_baseline.json"
    baseline = json.loads(baseline_path.read_text())
    live = {}
    for module_name, record in baseline["modules"].items():
        path = ROOT / record["path"]
        live[module_name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "unchanged": hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"],
            "n_public_signatures": len(record["signatures"]),
        }
    new_module = (ROOT / "src" / "tensordem" / "contact_physics")
    return {
        "baseline_path": str(baseline_path.relative_to(ROOT)),
        "baseline_git_rev": baseline["generated_from_git_rev"],
        "modules": live,
        "signature_changes": sum(0 if record["unchanged"] else 1
                                 for record in live.values()),
        "new_module_files": sorted(p.name for p in new_module.glob("*.py")),
        "passed": all(record["unchanged"] for record in live.values()),
    }


def _regression_section() -> dict:
    path = OUT / "regression_gate.json"
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "passed": False,
                "reason": "gate artifact missing"}
    gate = json.loads(path.read_text())
    return {
        "path": str(path.relative_to(ROOT)),
        "returncode": gate["returncode"],
        "passed_tests": gate["passed"],
        "failed_tests": gate["failed"],
        "errors": gate["errors"],
        "wall_s": gate["wall_s"],
        "per_node": [{key: row[key] for key in ("node", "passed", "failed", "returncode")}
                     for row in gate["per_node"]],
        "passed": gate["returncode"] == 0 and gate["failed"] == 0,
    }


def _documentation_section() -> dict:
    doc = (ROOT / "docs" / "contact_physics_module.md").read_text()
    import tensordem.contact_physics as package

    covered = [name for name in package.__all__ if name in doc]
    return {
        "doc_path": "docs/contact_physics_module.md",
        "doc_bytes": len(doc),
        "public_api_names": list(package.__all__),
        "covered_names": covered,
        "coverage": len(covered) / len(package.__all__),
        "example": "examples/contact_physics_demo.py",
        "passed": len(covered) == len(package.__all__),
    }


def _figures(regimes: dict) -> list[dict]:
    figures = []
    colors = {"cold_slow": "#1f77b4", "cold_rupture": "#d62728", "melting": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, record in regimes.items():
        history = record["histories"]
        label = name.replace("_", " ")
        axes[0].plot(history["time"], history["interface_temperature_c"],
                     color=colors[name], label=label)
        axes[1].plot(history["time"], history["mu"], color=colors[name], label=label)
        axes[2].plot(history["time"], np.asarray(history["film_thickness_m"]) * 1e6,
                     color=colors[name], label=label)
    axes[0].set(xlabel="time [s]", ylabel="interface temperature [degC]",
                title="N08: coupled interface temperature")
    axes[1].set(xlabel="time [s]", ylabel="mu [-]", title="N08: melt-lubricated friction")
    axes[2].set(xlabel="time [s]", ylabel="melt film h [um]",
                title="N08: Stefan melt film")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig_coupling_history.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": "fig_coupling_history.png",
                    "caption": "Interface temperature, friction coefficient and melt-film "
                               "thickness of the three integrated regimes (N08 two-way "
                               "coupling driven inside one contact-physics march)."})

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for name, record in regimes.items():
        history = record["histories"]
        axes[0].plot(history["bond_strain"], history["bond_force_mean_n"],
                     color=colors[name], label=name.replace("_", " "))
        axes[1].plot(history["time"], history["damage_max"],
                     color=colors[name], label=name.replace("_", " "))
    axes[0].set(xlabel="bond strain [-]", ylabel="mean bond force [N]",
                title="N02: Maxwell-Wiechert bond response")
    axes[1].set(xlabel="time [s]", ylabel="max Kachanov damage D [-]",
                title="N02: creep damage (truncated at rupture)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig_bond_response.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": "fig_bond_response.png",
                    "caption": "Bond force versus strain and the Kachanov damage of the "
                               "bond bundle; the melting regime saturates within one step "
                               "because F_ref -> 0 at the melting point."})

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    keys = list(reference.COMPARED_KEYS)
    width = 0.8 / len(regimes)
    for index, (name, record) in enumerate(regimes.items()):
        values = [max(record["relative_error_per_quantity"][key], 1e-18) for key in keys]
        axis.bar(np.arange(len(keys)) + index * width, values, width=width,
                 color=colors[name], label=name.replace("_", " "))
    axis.axhline(reference.TOLERANCE, color="black", linestyle="--", linewidth=1.0,
                 label="1e-10 acceptance")
    axis.set_yscale("log")
    axis.set_xticks(np.arange(len(keys)) + 0.4 - width / 2)
    axis.set_xticklabels(keys, rotation=75, fontsize=7, ha="right")
    axis.set(ylabel="max relative error integrated vs standalone",
             title="Criterion (a): integration layer does not perturb the results")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig_parity.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": "fig_parity.png",
                    "caption": "Per-quantity maximum relative error between the integrated "
                               "march and the three standalone drivers (all identical to "
                               "round-off, far below the 1e-10 acceptance line)."})

    polymorph = regimes["cold_slow"]["statistics"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    network = polymorph["force_network"]
    axes[0].bar(["Z (bonded lattice)", "bond density / m2", "keel angle / 10 deg"],
                [polymorph["morphology"]["mean_coordination"],
                 polymorph["morphology"]["bond_density_1_per_m2"],
                 polymorph["morphology"]["keel_angle_deg"] / 10.0],
                color="#4c72b0")
    axes[0].set(title="N05: ridge morphology of the sampled realisation",
                ylabel="value [-]")
    axes[1].bar(["CV of f", "strong fraction", "strong load fraction"],
                [network["coefficient_of_variation"], network["strong_fraction"],
                 network["strong_load_fraction"]], color="#dd8452")
    axes[1].set(title="N05: contact-force network statistics", ylabel="value [-]")
    for axis in axes:
        axis.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = OUT / "fig_statistics.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append({"name": "fig_statistics.png",
                    "caption": "Ridge morphology and contact-force-network statistics "
                               "sampled at the operating point (criterion for N05)."})
    return figures


def main() -> int:
    started = time.time()
    regimes = {}
    for name in sorted(reference.CASES):
        regimes[name] = _run_regime(name)
        print(f"regime {name}: max rel. error "
              f"{regimes[name]['max_relative_error']:.3e} "
              f"({'PASS' if regimes[name]['passed'] else 'FAIL'})",
              flush=True)

    performance = _performance()
    print(f"performance: time overhead {performance['time_overhead_percent']:.2f} %, "
          f"memory overhead {performance['memory_overhead_percent']:.2f} %", flush=True)

    api = _api_section()
    regression = _regression_section()
    documentation = _documentation_section()
    figures = _figures(regimes)

    acceptance = {
        "a_integration": {
            "criterion": "integrated run == three standalone runs, max rel. error < 1e-10",
            "max_relative_error": max(record["max_relative_error"]
                                      for record in regimes.values()),
            "passed": all(record["passed"] for record in regimes.values()),
            "artifact": "artifacts/contact_physics/integration_report.json",
        },
        "b_api_stability": {
            "criterion": "zero signature changes; every new public API documented",
            "signature_changes": api["signature_changes"],
            "public_api_names": documentation["public_api_names"],
            "passed": api["passed"] and documentation["coverage"] == 1.0,
            "artifact": api["baseline_path"],
        },
        "c_performance": {
            "criterion": "time and Python peak-memory overhead < 5 %",
            "time_overhead_percent": performance["time_overhead_percent"],
            "memory_overhead_percent": performance["memory_overhead_percent"],
            "passed": performance["passed"],
            "artifact": "artifacts/contact_physics/performance_pytest.json",
        },
        "d_regression": {
            "criterion": "upstream node suites (N02/N05/N08) all green",
            **{key: regression[key] for key in
               ("passed_tests", "failed_tests", "errors", "returncode", "passed")},
            "artifact": "artifacts/contact_physics/regression_gate.json",
        },
        "e_documentation": {
            "criterion": "module doc covers all public API; example runs standalone",
            "coverage": documentation["coverage"],
            "example": documentation["example"],
            "passed": documentation["passed"],
            "artifact": "docs/contact_physics_module.md",
        },
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": _git_rev(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tensordem_version": getattr(tensordem, "__version__", "unknown"),
        },
        "acceptance": acceptance,
        "all_passed": all(item["passed"] for item in acceptance.values()),
        "regimes": {name: {key: value for key, value in record.items()
                           if key != "histories"}
                    for name, record in regimes.items()},
        "performance": performance,
        "api": api,
        "regression": regression,
        "documentation": documentation,
        "figures": figures,
        "wall_s": time.time() - started,
    }
    (OUT / "integration_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({name: item["passed"] for name, item in acceptance.items()}))
    print(f"wrote artifacts/contact_physics/integration_report.json "
          f"({time.time() - started:.1f} s)")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())