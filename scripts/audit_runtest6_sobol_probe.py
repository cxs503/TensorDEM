#!/usr/bin/env python3
"""AUDIT-RUNTEST-6 (m14498): does the Sobol acceptance criterion (c) test anything?

Claim under audit
-----------------
``src/tensordem/brash_ice_rheology/fullscale_validation.py`` documents acceptance
criterion (c) as *"Sobol ΣS_Ti = 1.0 ± 0.01"* and the committed artifact
(``benchmarks/verified/dem_brash_ice_fullscale/results.json``, campaign node
p472w2-N04, capability upgrade WITHHELD -> REAL) reports it as PASS with
ΣS_Ti = 1.0000.

Two independent concerns are measured here:

1. **Tautology.**  ``sobol_sensitivity_analysis`` divides ``ST`` by its own sum
   before returning it, so the reported ``ST_sum`` is 1.0 by construction and
   criterion (c) cannot fail for any sample count.  Probe: rerun with
   deliberately starved sampling and show the criterion still reports PASS while
   the indices themselves are meaningless.

2. **Physical consistency.**  For independent inputs the total-order index must
   satisfy ``S_Ti >= S1_i``.  The committed artifact reports
   ``S1[mu_ii] = 0.4797 > ST[mu_ii] = 0.4089``.  The probe recomputes the *raw*
   (unnormalised) Jansen estimator and quantifies the violation and the raw
   ΣST deviation from unity, i.e. the quantity criterion (c) was supposed to
   check.

Run: ``PYTHONPATH=src python3 scripts/audit_runtest6_sobol_probe.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.brash_ice_rheology.fullscale_validation import (  # noqa: E402
    MT_UIKKU_HULL,
    OperatingPoint,
    dem_resistance,
    evaluate_acceptance_criteria,
    sobol_sensitivity_analysis,
    three_way_comparison,
)

BOUNDS = {
    "mu_ii": (0.15, 0.45),
    "mu_is": (0.05, 0.25),
    "sigma_b": (0.5e6, 2.0e6),
    "rho_ice": (880.0, 950.0),
}


def raw_sobol(n_samples: int, seed: int = 42):
    """Recompute the library's Saltelli/Jansen estimators *without* normalising."""
    point = OperatingPoint("BASELINE", MT_UIKKU_HULL, 1.0, 0.5, 3.0, 520, 55)
    rng = np.random.RandomState(seed)
    names = list(BOUNDS)
    k = len(names)
    A, B = rng.rand(n_samples, k), rng.rand(n_samples, k)

    def scale(m):
        out = np.zeros_like(m)
        for i, name in enumerate(names):
            lo, hi = BOUNDS[name]
            out[:, i] = m[:, i] * (hi - lo) + lo
        return out

    A_s, B_s = scale(A), scale(B)
    y_A = np.array([dem_resistance(point, *p) for p in A_s])
    y_B = np.array([dem_resistance(point, *p) for p in B_s])
    y_AB = np.zeros((k, n_samples))
    for i in range(k):
        AB = A.copy()
        AB[:, i] = B[:, i]
        y_AB[i] = np.array([dem_resistance(point, *p) for p in scale(AB)])

    var = np.var(np.concatenate([y_A, y_B]))
    S1 = np.array([np.mean(y_B * (y_AB[i] - y_A)) / var for i in range(k)])
    ST = np.array([np.mean((y_A - y_AB[i]) ** 2) / (2 * var) for i in range(k)])
    return names, S1, ST, float(var)


def main() -> int:
    report: dict = {"claim": "acceptance criterion (c) Sobol sum(S_Ti) = 1.0 +- 0.01"}

    # --- 1. raw estimators at the production sample count ---------------
    names, S1, ST, var = raw_sobol(500)
    st_sum = float(ST.sum())
    violations = [
        {"param": n, "S1": float(S1[i]), "ST_raw": float(ST[i]),
         "ST_minus_S1": float(ST[i] - S1[i])}
        for i, n in enumerate(names) if ST[i] < S1[i]
    ]
    normalized = ST / st_sum if st_sum > 0 else ST
    report["production_n500"] = {
        "S1": {n: float(S1[i]) for i, n in enumerate(names)},
        "ST_raw": {n: float(ST[i]) for i, n in enumerate(names)},
        "ST_normalized_as_reported": {n: float(normalized[i]) for i, n in enumerate(names)},
        "raw_ST_sum": st_sum,
        "raw_ST_sum_deviation_from_1": abs(st_sum - 1.0),
        "S1_sum": float(S1.sum()),
        "variance_of_R_kN2": var,
        "ST_lt_S1_violations": violations,
    }
    print(f"[raw] n=500  sum(ST)={st_sum:.6f}  |sum-1|={abs(st_sum-1):.4f}  "
          f"sum(S1)={S1.sum():.4f}")
    print(f"[raw] ST<S1 violations: {violations}")

    # --- 1b. estimator convergence: seed spread and sample-size trend ---
    scan = []
    for n, seed in [(500, 1), (500, 2), (500, 7), (2000, 42), (4000, 42)]:
        nm, s1, st, v = raw_sobol(n, seed=seed)
        scan.append({
            "n_samples": n, "seed": seed,
            "raw_ST_sum": float(st.sum()), "S1_sum": float(s1.sum()),
            "max_S1_minus_ST": float((s1 - st).max()),
            "var_of_R_kN2": v,
        })
        print(f"[scan] n={n:4d} seed={seed:2d}  sum(ST)={st.sum():.6f}  "
              f"sum(S1)={s1.sum():.4f}  max(S1-ST)={float((s1-st).max()):+.4f}  "
              f"var={v:.1f}")
    report["estimator_convergence"] = scan

    # --- 2. does criterion (c) ever fail? starve the sampling ----------
    starved = {}
    for n in (31, 101, 500):
        sobol = sobol_sensitivity_analysis(n_samples=n, seed=42)
        comparison = three_way_comparison()
        criteria = evaluate_acceptance_criteria(comparison, sobol, [])
        starved[n] = {
            "reported_ST_sum": sobol["ST_sum"],
            "criterion_c_verdict": criteria["c_sobol_sum"]["verdict"],
            "reported_ST": sobol["ST"],
            "reported_S1": sobol["S1"],
            "max_ST_lt_S1_gap": max(
                (sobol["S1"][k] - sobol["ST"][k] for k in sobol["S1"]), default=0.0),
        }
        print(f"[starved] n={n:4d} reported sum(ST)={sobol['ST_sum']:.6f} "
              f"criterion(c)={criteria['c_sobol_sum']['verdict']} "
              f"max(S1-ST)={starved[n]['max_ST_lt_S1_gap']:+.4f}")
    report["criterion_c_starved_samples"] = starved
    report["finding"] = (
        "criterion (c) is tautological: ST is normalised by its own sum inside "
        "sobol_sensitivity_analysis, so sum(ST)=1.0 exactly for every sample count "
        "including n=31; the criterion cannot detect unconverged indices. "
        "ST>=S1 is violated for mu_ii at the production sample count."
    )
    print("\nFINDING:", report["finding"])

    out = ROOT / "runs" / "sobol_probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", out / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())