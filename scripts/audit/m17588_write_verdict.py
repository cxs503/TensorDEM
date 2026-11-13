#!/usr/bin/env python3
"""m17588 board audit — write the maturity verdict for m17500.

The evidence blob is *assembled from the machine-checked audit output*
(artifacts/audit_m17588/evidence.json + optin_bitwise_probe.json), not typed by
hand: every number in `benchmark` is copied out of the audited results.json.

Declared level = 2 (independent functional implementation).
Why not 3/4 is recorded in `not_reached`, and the gate re-checks each key itself.
"""
from __future__ import annotations

import json
import os

ENV = "/root/tensor-ecosystem/TensorPlatform/.env"
HERE = os.path.dirname(os.path.abspath(__file__))
W88 = "/root/tp-agent/work/m17588"
W00 = "/root/tp-agent/work/m17500"
ANCHOR = "XVAL-DEM-MU-I-CONTACT-PROXY-NEGATIVE"
MISSION = 17588          # the audit mission id (this task)
ORIG = 17500             # the audited mission
LEVEL = 2


def dsn():
    for line in open(ENV):
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().replace(
                "postgresql+psycopg://", "postgresql://")
    raise SystemExit("no DATABASE_URL")


def build_evidence():
    ev = json.load(open(os.path.join(W88, "artifacts/audit_m17588/evidence.json")))
    probe = json.load(open(os.path.join(W88, "artifacts/audit_m17588/optin_bitwise_probe.json")))
    res = json.load(open(os.path.join(W00, "artifacts/xval_mu_i_contact_proxy_m17500/results.json")))
    rows = {r["label"]: r for r in res["rows"]}

    got_vs_ref = {
        lab: {
            "i_bulk_ref": round(r["i_bulk_measured"], 6),
            "i_c_got": round(r["i_proxy_fn_weighted"], 6),
            "i_mapping_error": round(r["i_mapping_error"], 6),
            "mu_bulk_ref": round(r["mu_bulk_measured"], 6),
            "mu_proxy_got": round(r["mu_proxy_fn_weighted"], 6),
            "mu_error": round(r["mu_error"], 6),
            "shortfall": round(r["shortfall"], 6),
            "servo_valid": r["servo_valid"],
            "tolerance": res["tolerance"],
        } for lab, r in rows.items()
    }

    return {
        "impl_files": [
            f"{W00}/src/tensordem/mu_i_contact_proxy.py",
            f"{W00}/src/tensordem/mu_i_rheology.py",
        ],
        "impl_files_detail": (
            "both files exist only on branch agent/m17500 @68f1622c (270 + 25 added "
            "lines); they are NOT in main f2e227f6 — main-checkout path "
            "/root/tensor-ecosystem/TensorDEM has no mu_i_contact_proxy.py"),
        "entry": [
            f"{W00}/scripts/xval_mu_i_contact_proxy_m17500.py",
            f"{W00}/docs/xval_mu_i_contact_proxy_m17500.md",
            f"{W00}/prereg/xval_mu_i_contact_proxy_m17500.md",
        ],
        "run_artifacts": [
            f"{W00}/artifacts/xval_mu_i_contact_proxy_m17500/results.json",
            f"{W00}/artifacts/xval_mu_i_contact_proxy_m17500/fig_mu_vs_I.png",
            f"{W00}/artifacts/xval_mu_i_contact_proxy_m17500/fig_errors.png",
            f"{W88}/artifacts/audit_m17588/evidence.json",
            f"{W88}/artifacts/audit_m17588/optin_bitwise_probe.json",
        ],
        "run_artifacts_sha256": {
            k: v["sha256"] for k, v in ev["artifacts"].items()},
        "tests": [
            "tests/test_xval_mu_i_contact_proxy.py::test_contact_gauge_reduces_to_bulk_inertial_number",
            "tests/test_xval_mu_i_contact_proxy.py::test_slip_speed_rigid_translation_and_rotation",
            "tests/test_xval_mu_i_contact_proxy.py::test_contact_proxy_stats_fn_weighting_matches_hand_calculation",
            "tests/test_xval_mu_i_contact_proxy.py::test_contact_proxy_stats_degenerate_inputs",
            "tests/test_xval_mu_i_contact_proxy.py::test_shortfall_is_driven_by_the_dense_band",
            "tests/test_xval_mu_i_contact_proxy.py::test_aggregate_proxy_uses_final_window_only",
            "tests/test_xval_mu_i_contact_proxy.py::test_xval_row_and_summary_verdicts",
            "tests/test_xval_mu_i_contact_proxy.py::test_device_diagnostic_is_opt_in",
        ],
        "tests_run_by_auditor": {
            "cmd": "cd /root/tp-agent/work/m17500 && PYTHONPATH=src python3 -m "
                   "pytest tests/test_xval_mu_i_contact_proxy.py -q",
            "result": "8 passed in 5.15s (auditor re-run, m17588)",
        },
        "integration_callers": [
            f"{W00}/src/tensordem/mu_i_rheology.py:1077",
            f"{W00}/src/tensordem/mu_i_rheology.py:1269",
        ],
        "integration_callers_detail": (
            "simulate_servo_shear() imports contact_proxy_stats/aggregate_proxy under "
            "`if case.contact_proxy:`; opt-in diagnostic, bit-identical when off "
            "(auditor probe: 8/8 quantities bit-equal). Caller reference exists ONLY on "
            "branch agent/m17500 -> L3 not reached in the ecosystem checkout"),
        "benchmark": {
            "registered_referee_rows_for_anchor": ev["benchmark_cases_for_anchor"],
            "tensordem_registered_cases_total": ev["tensordem_registered_cases"],
            "auditor_note": "tensor_benchmark_cases has 0 rows for this anchor -> no "
                            "platform referee, L4 unverifiable by rule; the in-repo "
                            "prereg (DEFAULT_TOL=0.15) is the only tolerance source",
            "got_vs_ref": got_vs_ref,
            "prereg_check": ev["prereg_check"],
            "report_vs_json_consistent": all(
                r["map_pct_agrees_0p1"] and r["mu_pct_agrees_0p1"]
                for r in ev["report_vs_json"] if r.get("in_report_table")),
            "arith_recomputed_ok": all(
                a["i_match_1e-9"] and a["mu_match_1e-9"] and a["shortfall_match_1e-9"]
                for a in ev["arithmetic"]),
        },
        "repro_cmd": ("cd /root/tp-agent/work/m17500 && PYTHONPATH=src python3 -m pytest "
                      "tests/test_xval_mu_i_contact_proxy.py -q"),
        "external_anchor": [
            f"{W00}/prereg/xval_mu_i_contact_proxy_m17500.md",
            "oss_solutions#421 (tag mu-i-contact-proxy-negative, repo TensorDEM) — the "
            "negative ledger entry this run reproduces and quantifies",
            "Jop/Forterre/Pouliquen 2006 Nature 441:727-730 (mu(I) law, mu_s=0.36, "
            "mu_2=0.84, I_0=0.30) — cited source of the closure constants",
        ],
        "not_reached": {
            "L3": "integration caller lives only on unmerged branch agent/m17500 "
                  "(not pushed to origin, not in main f2e227f6); not shipped in the "
                  "ecosystem checkout -> integration not demonstrable to the gate",
            "L4": "tensor_benchmark_cases: 0 rows for anchor "
                  "XVAL-DEM-MU-I-CONTACT-PROXY-NEGATIVE -> no registered referee; "
                  "additionally the run's own reference gate FAILED "
                  "(bulk-vs-Jop worst 32.98% > 15%), which by prereg voids the absolute "
                  "friction comparison",
            "L5": "no external experimental anchor beyond the OSS ledger entry; "
                  "3/5 cases servo_valid=False (steady state not certified)",
        },
        "audit_derived": {
            "servo_invalid_rows": [r["label"] for r in res["rows"] if not r["servo_valid"]],
            "n_servo_invalid": sum(1 for r in res["rows"] if not r["servo_valid"]),
            "worst_pressure_control_error": min(
                r["pressure_control_error_final"] for r in res["rows"]),
            "ic_over_ibulk": {r["label"]: round(
                r["i_proxy_fn_weighted"] / r["i_bulk_measured"], 4) for r in res["rows"]},
            "mission_text_claims_vs_artifacts": {
                "claims_4_of_5_servo_invalid": "4/5",
                "artifact_servo_invalid": "see n_servo_invalid (3/5)",
                "claims_worst_pressure_err": "-7.4%",
                "artifact_worst_pressure_err": round(min(
                    r["pressure_control_error_final"] for r in res["rows"]), 4),
                "claims_ic_over_ibulk_first": 2.54,
                "artifact_ic_over_ibulk_first": round(
                    res["rows"][0]["i_proxy_fn_weighted"]
                    / res["rows"][0]["i_bulk_measured"], 4),
            },
            "bit_identity_probe": {
                "integrator_bit_identical": probe["integrator_bit_identical"],
                "diag_key_absent_when_off": probe["diag_key_absent_when_off"],
                "n_quantities_compared": len(probe["per_quantity"]),
            },
        },
        "findings": [
            "F1 (blocking, delivery): deliverables 68f1622c sit on local branch "
            "agent/m17500 only; `git ls-remote origin agent/m17500` = empty and "
            "merge-base --is-ancestor agent/m17500 main = 1 (rc=1) -> nothing of this "
            "mission is in main, so no CI/main reproducibility",
            "F2 (blocking, referee): no tensor_benchmark_cases row for the anchor "
            "(8 TensorDEM cases exist for other anchors) -> 基准裁判 absent",
            "F3 (major, science): prereg reference gate FAILED (bulk vs Jop 22.88% / "
            "32.98% / 24.03%) so per its own clause the absolute mu comparison is void; "
            "the headline still quotes mu_error 35.8% as a result",
            "F4 (major, validity + mis-statement): 3/5 rows servo_valid=False with "
            "pressure_control_error_final -5.3%/-7.4%/-19.3%; the mission text claims "
            "'4/5 rows invalid, worst -7.4%' -> one row over-counted and the worst "
            "deviation understated by 12 pp",
            "F5 (minor, closed by auditor): the bit-identity claim was only asserted "
            "structurally (test_device_diagnostic_is_opt_in checks the dataclass flag); "
            "auditor probe re-ran one case off/on and got 8/8 bit-equal integrator "
            "quantities -> claim holds",
            "F7 (minor, numeral drift): report says I_c/I_bulk=2.54 for the first row; "
            "artifact gives 0.024108/0.009544 = 2.526 (0.5% slip); other two (0.576, "
            "0.407) are exact",
            "F6 (minor, data hygiene): results.json contains literal NaN (n_contacts, "
            "kappa) -> not strict JSON",
        ],
        "audit_checks_run": [
            "pytest 8/8 in the deliverable worktree (auditor re-run)",
            "recomputed i_mapping_error / mu_error / shortfall from primitives, all 5 "
            "rows match stored to 1e-9",
            "report #16473 markdown table vs results.json: 5/5 rows agree to 0.1 pp; "
            "six required sections all present",
            "opt-in bit-identity probe (off vs on, 8 integrator quantities)",
            "git ancestry / origin reachability of the mission branch",
        ],
        "audit_checks_not_run": [
            "full 346 s 5-case regeneration of the DEM sweep (repro_cmd executable for "
            "the unit tests; the sweep was not re-executed inside this audit budget — "
            "not_run, not passed)",
            "pressure-control certification of the 3 servo_valid=False cases",
        ],
    }


def main():
    import psycopg
    ev = build_evidence()
    scope = f"m{ORIG} {ANCHOR}"
    payload = json.dumps(ev, ensure_ascii=False, default=str)
    with psycopg.connect(dsn()) as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO tensor_maturity_verdicts "
            "(repo, scope, mission_id, level, evidence, verdict_by) "
            "VALUES ('TensorDEM', %s, %s, %s, %s::jsonb, 'board-audit') "
            "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET "
            "level=EXCLUDED.level, evidence=EXCLUDED.evidence, "
            "mission_id=EXCLUDED.mission_id "
            "RETURNING id, level, gate_state, gate_level",
            (scope, MISSION, LEVEL, payload))
        vid, lvl, gs, gl = cur.fetchone()
        c.commit()
    print(f"verdict id={vid} level={lvl} gate_state={gs} gate_level={gl}")


if __name__ == "__main__":
    main()