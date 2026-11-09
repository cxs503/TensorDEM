"""Downstream consumer of the ``r_u(T_v)`` consolidation table (IN99-N08, C4).

This is a *separate* script on purpose: it is the proof that the exported
interface is usable without importing the consolidation internals.  It reads
the JSON table, refuses it when the applicability verdict is negative
(fail-closed), interpolates ``r_u`` along a storm consolidation history, feeds
the value into the existing liquefaction criterion
(:class:`tensordem.cargo_liquefaction_effstress.CargoMatrix`) and asserts the
end-to-end behaviour.

The liquefaction criterion used here is the one already in the repository:
effective (Terzaghi) friction angle
``tan phi_eff = (1 - r_u) [c(MR)/sigma_n + tan phi_0]``.  A bed is declared
liquefaction-prone while ``phi_eff`` stays below ``phi_crit``; the script solves
for the ``r_u`` that makes ``phi_eff = phi_crit``, converts it to a time factor
through the table and cross-checks that value against the independent Terzaghi
anchor (``r_u = 1 - U``).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.cargo_liquefaction_effstress import CargoMatrix  # noqa: E402
from tensordem.consolidation import (  # noqa: E402
    Tv_of_U,
    U_series,
    r_u_at,
    r_u_profile_at,
    read_ru_table,
)


def phi_eff_deg(matrix: CargoMatrix, moisture_ratio: float, r_u: float, sigma_n: float) -> float:
    return matrix.effective_friction_angle(
        moisture_ratio, pore_pressure_ratio=r_u, sigma_n=sigma_n
    )["phi_eff_deg"]


def solve_r_u_for_phi(
    matrix: CargoMatrix, moisture_ratio: float, sigma_n: float, phi_crit: float
) -> float:
    """``r_u`` at which the effective friction angle equals ``phi_crit``."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if phi_eff_deg(matrix, moisture_ratio, mid, sigma_n) > phi_crit:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=str(REPO / "artifacts" / "in99-n08" / "ru_table.json"))
    parser.add_argument("--out", default=str(REPO / "artifacts" / "in99-n08" / "consumer_assertions.json"))
    parser.add_argument("--moisture-ratio", type=float, default=None,
                        help="bulk moisture ratio; default = full moisture point")
    parser.add_argument("--sigma-n", type=float, default=10.0e3)
    parser.add_argument("--phi-crit-deg", type=float, default=20.0)
    args = parser.parse_args(argv)

    table_path = Path(args.table)
    if not table_path.exists():
        print(f"FAIL: table not found: {table_path}", file=sys.stderr)
        return 2
    table = read_ru_table(table_path)  # validates schema, monotonicity, hash

    verdict = table.get("applicability", {})
    if not verdict.get("applicable", False):
        print("REFUSED (fail-closed): table applicability verdict is negative: "
              f"{verdict.get('reasons')}", file=sys.stderr)
        return 3

    matrix = CargoMatrix()
    moisture_ratio = (matrix.full_moisture_point
                      if args.moisture_ratio is None else args.moisture_ratio)
    sigma_n = args.sigma_n
    phi_crit = args.phi_crit_deg

    Tv = [float(t) for t in table["T_v"]]
    U = [float(u) for u in table["U_analytic"]]
    ru_avg = [float(r) for r in table["r_u_average"]]
    ru_mid = [float(r) for r in table["r_u_midplane"]]

    assertions: list[dict] = []

    def check(name: str, passed: bool, value=None, expected=None, note: str = "") -> None:
        assertions.append({"name": name, "passed": bool(passed), "value": value,
                           "expected": expected, "note": note})

    # A1 mass consistency r_u = 1 - U
    check("r_u_average_is_1_minus_U",
          all(abs((1.0 - u) - r) <= 1e-12 for u, r in zip(U, ru_avg)),
          float(max(abs((1.0 - u) - r) for u, r in zip(U, ru_avg))), "<= 1e-12")

    # A2 bounds
    check("r_u_in_unit_interval", all(-1e-12 <= r <= 1.0 + 1e-12 for r in ru_avg + ru_mid),
          [min(ru_avg + ru_mid), max(ru_avg + ru_mid)], "[0, 1]")

    # A3 monotonicity (dissipation only removes excess pressure)
    check("r_u_monotone_decreasing", all(b <= a + 1e-12 for a, b in zip(ru_avg, ru_avg[1:])))

    # A4 the impermeable mid-plane lags the depth average
    check("midplane_lags_average", all(m >= a - 1e-12 for a, m in zip(ru_avg, ru_mid)),
          float(max(a - m for a, m in zip(ru_avg, ru_mid))), "<= 0")

    # A5 the exported C_v is usable to convert a time factor into a time
    C_v = float(table["C_v_m2_s"])
    H_dr = float(table["definition"]["drainage_path_m"])
    t_end = Tv[-1] * H_dr ** 2 / C_v
    check("C_v_positive_and_time_axis_finite",
          C_v > 0.0 and math.isfinite(t_end) and t_end > 0.0,
          {"C_v": C_v, "t_at_Tv_max_s": t_end}, "finite positive")

    # A6 end-to-end: the liquefaction threshold time factor read from the table
    #     agrees with the independent Terzaghi anchor to < 2 %.
    r_u_crit = solve_r_u_for_phi(matrix, moisture_ratio, sigma_n, phi_crit)
    phi_at_crit = phi_eff_deg(matrix, moisture_ratio, r_u_crit, sigma_n)
    Tv_from_table = None
    for a, b, ra, rb in zip(Tv, Tv[1:], ru_avg, ru_avg[1:]):
        if rb <= r_u_crit <= ra:
            w = (ra - r_u_crit) / (ra - rb) if ra > rb else 0.0
            Tv_from_table = a + w * (b - a)
            break
    if Tv_from_table is None:
        Tv_from_table = 0.0 if r_u_crit >= ru_avg[0] else Tv[-1]
    Tv_anchor = Tv_of_U(1.0 - r_u_crit)
    rel = abs(Tv_from_table - Tv_anchor) / Tv_anchor * 100.0
    check("liquefaction_window_vs_terzaghi_anchor_le2pct", rel <= 2.0,
          {"phi_crit_deg": phi_crit, "phi_eff_at_solution_deg": phi_at_crit,
           "r_u_crit": r_u_crit, "T_v_from_table": Tv_from_table,
           "T_v_from_series": Tv_anchor, "relative_pct": rel}, "<= 2 %")

    # A7 physical direction: effective strength recovers as the pore pressure
    #     dissipates (monotone increasing phi_eff along the table)
    phis = [phi_eff_deg(matrix, moisture_ratio, r, sigma_n) for r in ru_avg]
    check("phi_eff_recovers_monotonically",
          all(b >= a - 1e-12 for a, b in zip(phis, phis[1:])),
          {"phi_first_deg": phis[0], "phi_last_deg": phis[-1]})

    # A8 end-to-end: at t -> 0+ the bed is fully liquefied (r_u -> 1) and at the
    #     last tabulated time the criterion has switched off
    r0 = r_u_at(table, Tv[0])
    rN = r_u_at(table, Tv[-1])
    check("liquefied_early_and_recovered_late",
          r0 > r_u_crit > rN,
          {"r_u_at_first_Tv": r0, "r_u_crit": r_u_crit, "r_u_at_last_Tv": rN})

    # A9 depth-resolved use of the exported profile
    depth, prof = r_u_profile_at(table, 0.2)
    check("profile_export_consumable",
          len(depth) == len(prof) and prof[0] <= 1e-9 and prof[-1] >= prof[1],
          {"depth": list(depth), "r_u": list(prof)},
          "drained face r_u=0, mid-plane maximum")

    all_pass = all(a["passed"] for a in assertions)
    payload = {
        "schema": "tensordem.consolidation.consumer_assertions/1",
        "table": {
            "path": str(table_path.relative_to(REPO) if table_path.is_relative_to(REPO) else table_path),
            "sha256": table.get("integrity", {}).get("sha256_of_content"),
            "schema": table.get("schema"),
        },
        "liquefaction_model": {
            "criterion": "tan phi_eff = (1 - r_u) [c(MR)/sigma_n + tan phi_0]",
            "moisture_ratio": moisture_ratio,
            "full_moisture_point": matrix.full_moisture_point,
            "sigma_n_pa": sigma_n,
            "phi_crit_deg": phi_crit,
        },
        "liquefaction_window_T_v": Tv_from_table if Tv_from_table is not None else None,
        "assertions": assertions,
        "all_pass": all_pass,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    for a in assertions:
        print(f"  [{'PASS' if a['passed'] else 'FAIL'}] {a['name']}")
    print(f"consumer: {sum(a['passed'] for a in assertions)}/{len(assertions)} assertions pass")
    print(f"liquefaction window closes at T_v = {Tv_from_table}")
    print(f"wrote {out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())