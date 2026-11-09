"""Run the granular Hertz-Mindlin wave-speed study and write the artefacts.

Usage
-----
    python scripts/run_granular_wave_speed.py [--out artifacts/granular_wave]

Writes ``wave_speed_study.json`` (all numbers, no hand editing) and, when
matplotlib is available, ``p_vs_solid_fraction.png`` and ``speed_vs_pressure.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensordem.granular_wave import GranularWaveConfig, WaveStudyConfig, run_wave_speed_study  # noqa: E402


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _figure(report: dict, out: pathlib.Path) -> list[dict]:
    """Two report figures; returns the figure descriptors (name/caption)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is optional
        print(f"[warn] matplotlib unavailable ({exc}); skipping figures")
        return []
    records = report["records"]
    sf = [r["solid_fraction"] for r in records]
    p = [r["state"]["pressure_mean_Pa"] for r in records]
    pngs: list[dict] = []

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.semilogy([100.0 * s for s in sf], p, "o-", label="measured $p$ (platen reactions)")
    ax.set_xlabel("solid fraction  [%]")
    ax.set_ylabel("confining pressure $p$  [Pa]")
    ax.set_title("Confined Hertz-Mindlin packing: $p$ vs density")
    ax.grid(True, which="both", alpha=0.3)
    for r in records:
        ax.annotate(f"Z={r['state']['coordination_number']:.1f}",
                    (100.0 * r["solid_fraction"], r["state"]["pressure_mean_Pa"]),
                    textcoords="offset points", xytext=(4, 6), fontsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "p_vs_solid_fraction.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    pngs.append({"name": path.name,
                 "caption": "Confining pressure against solid fraction for the six "
                            "constant-volume states (coordination number annotated)."})

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    vp = [r["derived"].get("static_v_p_m_s") for r in records]
    vs = [r["derived"].get("static_v_s_m_s") for r in records]
    vpd = [r["digby"]["v_p_m_s"] for r in records]
    vsd = [r["digby"]["v_s_m_s"] for r in records]
    good = [(a, b, c, d, e) for a, b, c, d, e in zip(p, vp, vs, vpd, vsd)
            if _finite(b) and _finite(c)]
    if good:
        pp = [g[0] for g in good]
        ax.loglog(pp, [g[1] for g in good], "o-", label="$v_p$ from static moduli")
        ax.loglog(pp, [g[2] for g in good], "s-", label="$v_s$ from static moduli")
        ax.loglog(pp, [g[3] for g in good], "o--", mfc="none",
                  label="$v_p$ Digby (1981)")
        ax.loglog(pp, [g[4] for g in good], "s--", mfc="none",
                  label="$v_s$ Digby (1981)")
        slopes = report["acceptance"]["slopes"]
        txt = "\n".join(f"slope({k}) = {v:.4f}" for k, v in slopes.items())
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(fc="white", ec="0.7", alpha=0.9))
    ax.set_xlabel("confining pressure $p$  [Pa]")
    ax.set_ylabel("wave speed  [m/s]")
    ax.set_title("Wave speeds vs pressure: measured and Hertz-Mindlin prediction")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out / "speed_vs_pressure.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    pngs.append({"name": path.name,
                 "caption": "P/S wave speeds against confining pressure from the static "
                            "moduli route (solid) and the Digby (1981) Hertz-Mindlin "
                            "effective medium (dashed); the fitted log-log slopes are "
                            "annotated in the panel."})
    return pngs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/granular_wave")
    ap.add_argument("--fast", action="store_true", help="smaller packing (smoke run)")
    ap.add_argument("--no-dynamic", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base = GranularWaveConfig()
    if args.fast:
        base = GranularWaveConfig(nx=3, ny=3, nz=16)
    cfg = WaveStudyConfig(base=base, run_dynamic=not args.no_dynamic)
    if args.fast:
        cfg = WaveStudyConfig(base=base, run_dynamic=not args.no_dynamic,
                              solid_fractions=(0.530, 0.550, 0.575, 0.605),
                              relax_steps=400, run_size_test=False)
    report = run_wave_speed_study(cfg)
    report["figures"] = _figure(report, out)
    path = out / "wave_speed_study.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {path}")
    checks = report["acceptance"]["checks"]
    for k, v in checks.items():
        if isinstance(v, dict):
            val = v.get("value")
            print(f"  {k:26s} value={val if not isinstance(val, float) else round(val, 5)}"
                  f"  pass={v.get('pass')}")
    print(f"  all_pass={checks['all_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())