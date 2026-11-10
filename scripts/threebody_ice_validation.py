"""Runner: three-body collision + broken-ice resistance decomposition (node p262w1-N04).

Sharded execution (the manifest is fixed before any number is produced)::

    python scripts/threebody_ice_validation.py --shard 0/14     # one shard
    python scripts/threebody_ice_validation.py --merge          # criteria + figures

``--merge`` reads every ``artifacts/threebody_ice/parts/*.json``, evaluates the
predeclared criteria, writes ``artifacts/threebody_ice/validation.json`` (with a
SHA-256 over its own numeric payload), draws the three required figures and
regenerates ``docs/P262W1_N04_THREEBODY_ICE.md``.

All numbers in the artifact and in the markdown are computed here; nothing is
typed by hand.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402  (import after sys.path is extended)

from tensordem.threebody_ice.model import parameter_ledger_sha256  # noqa: E402
from tensordem.threebody_ice.validation import (CASE_MANIFEST, THRESHOLDS,  # noqa: E402
                                                artifact_dir, evaluate, run_case)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ARTIFACT = os.path.join(ROOT, "artifacts", "threebody_ice")
DOC = os.path.join(ROOT, "docs", "P262W1_N04_THREEBODY_ICE.md")


def _parts(root: str = ARTIFACT):
    paths = sorted(glob.glob(os.path.join(root, "parts", "*.json")))
    return [json.load(open(path, encoding="utf-8")) for path in paths]


def _run_list(keys, *, root: str = ARTIFACT) -> None:
    for key, kind, params in CASE_MANIFEST():
        if keys is not None and key not in keys:
            continue
        print(f"[case] {key} ({kind}) ...", flush=True)
        run_case(key, kind, params, root=root)
        print(f"[case] {key} done", flush=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def figure_force_distribution(validation, parts, path: str) -> str:
    plt = _mpl()
    import numpy as np

    matrices = validation["three_body_matrix"]
    keys = [p["key"] for p in parts]
    trace_key = "three_body_drag_d1"
    trace = next((p for p in parts if p["key"] == trace_key), None)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))

    if trace is not None and trace["row"].get("series"):
        series = trace["row"]["series"]
        t = np.array(series["t"])
        axes[0].plot(t, np.array(series["f_1_x"]) / 1e3, label="$F_{11}$ (near floe)", lw=1.0)
        axes[0].plot(t, np.array(series["f_2_x"]) / 1e3, label="$F_{12}$ (far floe)", lw=1.0)
        axes[0].plot(t, np.array(series["f_total_x"]) / 1e3, label="$F_{total}$", lw=1.4,
                     color="k", alpha=0.7)
        axes[0].set_xlabel("time [s]")
        axes[0].set_ylabel("hull contact force [kN]")
        axes[0].set_title("(a) tandem impact, d/D = 1 (with water drag)")
        axes[0].legend(fontsize=8, loc="upper right")

    for drag, marker, label in ((False, "o", "drag-free impact"),
                                (True, "s", "sustained (drag on)")):
        rows = sorted([m for m in matrices if m["drag"] == drag],
                      key=lambda m: m["spacing_ratio"])
        x = [m["spacing_ratio"] for m in rows]
        axes[1].plot(x, [m["impulse_share_1"] for m in rows], marker + "-", label=label)
        axes[2].plot(x, [m["simultaneous_contact_fraction"] for m in rows],
                     marker + "-", label=label)
    axes[1].set_xlabel("clear floe spacing  d / D [-]")
    axes[1].set_ylabel("impulse share of the near floe $I_{11}/I_{total}$")
    axes[1].set_title("(b) load allocation vs spacing")
    axes[1].axhline(0.5, ls=":", color="grey", lw=0.8)
    axes[1].legend(fontsize=8)
    axes[2].set_xlabel("clear floe spacing  d / D [-]")
    axes[2].set_ylabel("fraction of contact time with both floes loaded")
    axes[2].set_title("(c) simultaneity of the two hull contacts")
    axes[2].legend(fontsize=8)
    fig.suptitle("F_total = F_11 + F_12 : force partition of the hull-floe-floe contact", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_resistance_decomposition(validation, parts, path: str) -> str:
    plt = _mpl()
    import numpy as np

    rows = validation["field_matrix"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))

    base = sorted([r for r in rows if abs(r["concentration"] - 0.5) < 1e-9],
                  key=lambda r: r["speed_m_s"])
    speeds = np.array([r["speed_m_s"] for r in base])
    components = (("R_collision_N", "collision", "tab:red"),
                  ("R_submerged_drag_N", "submerged drag", "tab:blue"),
                  ("R_channel_clearing_N", "channel clearing", "tab:green"))
    bottom = np.zeros_like(speeds)
    for key, label, colour in components:
        values = np.array([r[key] for r in base])
        axes[0].bar(speeds, values, bottom=bottom, width=0.55, label=label, color=colour)
        bottom += values
    axes[0].plot(speeds, [r["R_ice_N"] for r in base], "k^--", label="$R_{ice}$ (measured)")
    axes[0].set_xlabel("ship speed V [m/s]")
    axes[0].set_ylabel("resistance [N]")
    axes[0].set_title("(a) decomposition at $c_{ice}$ = 0.5")
    axes[0].legend(fontsize=8)

    exponents = validation["velocity_exponents"]
    key_c = "c0.5"
    if key_c in exponents:
        for name, label, colour in (("R_collision", "collision", "tab:red"),
                                    ("R_submerged_drag", "submerged drag", "tab:blue"),
                                    ("R_channel_clearing", "channel clearing", "tab:green"),
                                    ("R_ice", "$R_{ice}$", "k")):
            fit = exponents[key_c][name]
            values = [r[{"R_collision": "R_collision_N",
                         "R_submerged_drag": "R_submerged_drag_N",
                         "R_channel_clearing": "R_channel_clearing_N",
                         "R_ice": "R_ice_N"}[name]] for r in base]
            axes[1].loglog(speeds, values, "o", color=colour, ms=4)
            axes[1].loglog(speeds, [fit["K"] * s ** fit["exponent"] for s in speeds], "-",
                           color=colour, lw=1.0,
                           label=f"{label}: V^{fit['exponent']:.2f}")
        axes[1].set_xlabel("ship speed V [m/s]")
        axes[1].set_ylabel("resistance [N]")
        axes[1].set_title("(b) measured velocity exponents ($c_{ice}$ = 0.5)")
        axes[1].legend(fontsize=8)

    for concentration in sorted({r["concentration"] for r in rows}):
        subset = sorted([r for r in rows if abs(r["concentration"] - concentration) < 1e-9],
                        key=lambda r: r["speed_m_s"])
        axes[2].plot([r["speed_m_s"] for r in subset],
                     [r["R_ice_N"] for r in subset], "o-",
                     label=f"$c_{{ice}}$ = {concentration:g}")
    axes[2].set_xlabel("ship speed V [m/s]")
    axes[2].set_ylabel("$R_{ice}$ [N]")
    axes[2].set_title("(c) concentration dependence")
    axes[2].legend(fontsize=8)
    fig.suptitle("R_ice = R_collision + R_submerged_drag + R_channel_clearing "
                 "(measured energy ledger)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_energy_dissipation(validation, parts, path: str) -> str:
    plt = _mpl()
    import numpy as np

    matrices = validation["three_body_matrix"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))

    rows = sorted([m for m in matrices if not m["drag"]], key=lambda m: m["spacing_ratio"])
    x = [m["spacing_ratio"] for m in rows]
    axes[0].plot(x, [m["dissipation_fraction_of_input"] for m in rows], "o-",
                 label="$E_{diss}/W_{hull}$")
    axes[0].plot(x, [m["dissipation_over_ke_gain"] for m in rows], "s--",
                 label="$E_{diss}/\\Delta E_k$")
    band = validation["thresholds"]
    axes[0].axhspan(band["dissipation_band_low"], band["dissipation_band_high"],
                    color="tab:green", alpha=0.12, label="node band [0.3, 0.8]")
    analytic = next((p["calibration"]["analytic_single_collision"]
                     for p in parts if p["key"] == "reference_calibration"), None)
    if analytic:
        axes[0].axhline(analytic["dissipation_fraction_of_input"], ls=":", color="k",
                        lw=0.9, label="analytic single collision")
    axes[0].set_xlabel("clear floe spacing  d / D [-]")
    axes[0].set_ylabel("dissipated fraction of the collision energy")
    axes[0].set_title("(a) dissipation band, drag-free impacts")
    axes[0].legend(fontsize=8)

    elastic = next((p["elastic"] for p in parts if p["key"] == "reference_elastic"), None)
    if elastic and elastic.get("series"):
        series = elastic["series"]
        axes[1].plot(series["t"], series["kinetic"], label="kinetic")
        axes[1].plot(series["t"], series["spring"], label="contact spring")
        axes[1].plot(series["t"], series["total"], "k--", lw=1.2, label="total")
        axes[1].set_xlabel("time [s]")
        axes[1].set_ylabel("energy [J]")
        axes[1].set_title("(b) elastic reference: $E_{kin}+E_{spring}$ conservation")
        axes[1].legend(fontsize=8)

    for concentration in sorted({r["concentration"] for r in validation["field_matrix"]}):
        subset = sorted([r for r in validation["field_matrix"]
                         if abs(r["concentration"] - concentration) < 1e-9],
                        key=lambda r: r["speed_m_s"])
        axes[2].plot([r["speed_m_s"] for r in subset],
                     [100.0 * r["closure"] for r in subset], "o-",
                     label=f"$c_{{ice}}$ = {concentration:g}")
    axes[2].axhline(100.0 * band["decomposition_closure_relative"], ls=":", color="r",
                    lw=1.0, label="5 % criterion")
    axes[2].set_xlabel("ship speed V [m/s]")
    axes[2].set_ylabel("decomposition closure error [%]")
    axes[2].set_title("(c) energy-ledger closure of the retirements")
    axes[2].legend(fontsize=8)
    fig.suptitle("Collision energy dissipation and ledger closure", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


FIGURE_BUILDERS = (
    ("force_distribution.png", figure_force_distribution),
    ("resistance_decomposition.png", figure_resistance_decomposition),
    ("energy_dissipation.png", figure_energy_dissipation),
)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(*, pytest_count: int = 0, source: str = "") -> dict:
    parts = _parts()
    keys_done = {p["key"] for p in parts}
    missing = [k for k, _, _ in CASE_MANIFEST() if k not in keys_done]
    validation = evaluate(parts, pytest_count=pytest_count or None, figures=[])
    figure_paths = []
    for name, builder in FIGURE_BUILDERS:
        path = os.path.join(ARTIFACT, name)
        figure_paths.append(builder(validation, parts, path))
    validation = evaluate(parts, pytest_count=pytest_count or None,
                          figures=[os.path.basename(p) for p in figure_paths])
    payload = {
        "node": "p262w1-N04-dem-collision-brash",
        "repo": "TensorDEM",
        "module": "src/tensordem/threebody_ice",
        "source": source or ",".join(sorted(keys_done)),
        "missing_cases": missing,
        "parameter_ledger_sha256": parameter_ledger_sha256(),
        "parallel_fallback_used": any(
            bool(p.get("row", {}).get("spec", {}).get("parallel_fallback"))
            for p in parts),
        **validation,
        "figures": [os.path.basename(p) for p in figure_paths],
        "cases": {p["key"]: {"kind": p["kind"], "params": p.get("params", {})}
                  for p in parts},
    }
    numeric = json.dumps({k: v for k, v in payload.items() if k != "cases"},
                         sort_keys=True, default=str)
    payload["artifact_sha256"] = hashlib.sha256(numeric.encode()).hexdigest()
    with open(os.path.join(ARTIFACT, "validation.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=str)
    from tensordem.threebody_ice.report import build_markdown
    markdown = build_markdown(payload, parts)
    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    with open(DOC, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"[merge] {payload['n_passed']}/{payload['n_criteria']} criteria passed; "
          f"missing={missing}")
    print(f"[merge] wrote {os.path.join(ARTIFACT, 'validation.json')} and {DOC}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="three-body ice collision validation")
    parser.add_argument("--shard", default="", help="i/n : run one shard of the manifest")
    parser.add_argument("--merge", action="store_true", help="evaluate + figures + report")
    parser.add_argument("--all", action="store_true", help="run the whole manifest locally")
    parser.add_argument("--list", action="store_true", help="print the manifest")
    parser.add_argument("--pytest-count", type=int, default=0)
    parser.add_argument("--source", default="")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))

    if args.list:
        for key, kind, params in CASE_MANIFEST():
            print(key, kind, params)
        return 0
    if args.shard:
        index, total = (int(x) for x in args.shard.split("/"))
        cases = CASE_MANIFEST()
        assigned = [cases[i] for i in range(len(cases)) if i % total == index]
        keys = {key for key, _, _ in assigned}
        print(f"[shard {index}/{total}] {sorted(keys)}", flush=True)
        _run_list(keys)
        return 0
    if args.all:
        _run_list(None)
    if args.merge or args.all:
        merge(pytest_count=args.pytest_count, source=args.source)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())