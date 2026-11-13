"""Finaliser for mission 18143 (round 2 of CASE-DEM-RCP-PACKING-FRACTION).

Waits for the four new production runs (mu = 0 / 0.1 at k_n = 64000, seeds 2/3),
then, in one shot: stops the round-2 lanes from starting their optional arms,
aggregates the artefact, runs the kernel tests, records the method-selection row
(`oss_solutions`) and the negative OSS-reference finding, and publishes the
report.  Written to fit a hard per-call shell budget; every step is idempotent
and its outcome is appended to `.scratch/finalize.log`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
RUNS = REPO / "benchmarks" / "case_rcp_deposition" / "runs"
LOG = REPO / ".scratch" / "finalize.log"
NEED = [RUNS / f"deposition_n10000_mu{m:g}_kn64000_e0.1_seed{s}.json"
        for m in (0.0, 0.1) for s in (2, 3)]


def log(msg: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def wait(paths, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if all(p.exists() for p in paths):
            return True
        time.sleep(5)
    return all(p.exists() for p in paths)


def sh(cmd):
    r = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


def sql(stmt: str, tag: str) -> None:
    p = REPO / ".scratch" / f"{tag}.sql"
    p.write_text(stmt)
    rc, out = sh(f"psql -U foundation -d tensorplatform -v ON_ERROR_STOP=1 -f {p}")
    log(f"sql {tag}: rc={rc} {out.strip()[-400:]}")


def main() -> int:
    log("finaliser start")
    wait(NEED[:2], 900)
    time.sleep(25)  # let each lane launch its mu=0.1 child
    for lane in ("a", "b"):
        sh(f"pkill -f 'run_round2.sh {lane}' || true")
        log(f"pkill lane {lane}")
    ok = wait(NEED, 1500)
    log(f"production runs present: {ok}")

    rc, out = sh("PYTHONPATH=src python3 benchmarks/case_rcp_deposition/aggregate.py")
    log(f"aggregate rc={rc}\n{out[-2000:]}")

    rc, out = sh("PYTHONPATH=src python3 -m pytest -q tests/test_rcp_deposition.py "
                 "tests/test_rcp_references.py tests/test_rcp_extrapolation_order.py")
    log(f"pytest rc={rc}\n{out[-1000:]}")

    s = json.load(open(RUNS / "summary.json"))
    npass = sum(1 for g in s["gates"] if g["pass"])
    failing = [g["id"] for g in s["gates"] if not g["pass"]]
    s0 = s["stiffness"].get("mu=0", {})
    s1 = s["stiffness"].get("mu=0.1", {})
    t0 = s["tiers"].get("mu=0", {})
    t1 = s["tiers"].get("mu=0.1", {})
    phi_ref = s["reference"]["phi_rcp"]
    kn = s["kn_production"]

    method = (
        "Soft-sphere gravity deposition (DEM) in a periodic-footprint box: RSA "
        "dilute placement -> gravity settle -> settle when KE/(N m g d) < 5e-4 for "
        f"4 gravity times; phi from a bulk slab band (2.5 d above floor to 2.5 d "
        f"below the free surface); production stiffness = stiffest k_n carrying 3 "
        f"seeds per friction tier ({kn:g}); hard-sphere limit by observed-order "
        "power-law Richardson in 1/k_n with leave-one-level-out self-consistency; "
        "halved-timestep probes separate stiffness from integrator error. "
        f"Result: phi_inf(mu=0)={s0.get('limit')}, phi_inf(mu=0.1)={s1.get('limit')} "
        f"vs ref {phi_ref} (Scott 1960; Bernal & Mason 1960); "
        f"production phi(mu=0)={t0.get('phi_mean')}, phi(mu=0.1)={t1.get('phi_mean')}."
    )
    files = ("src/tensordem/deposition_packing.py; "
             "benchmarks/case_rcp_deposition/{run_case.py,aggregate.py,shard.py,"
             "run_round2.sh,plan.json}; "
             "benchmarks/case_rcp_deposition/runs/deposition_n10000_*.json; "
             "docs/CASE_RCP_DEPOSITION.md; docs/OSS_METHOD_SELECTION_RCP.md")
    conventions = (
        "1) never quote phi without either k_n or the hard-sphere limit; "
        "2) production stiffness chosen by seed coverage, not by which value looks "
        "best; 3) band [0.62,0.66] and 3 % are applied to the rigid-grain limit; "
        "4) residual overlap/d < 1 % is a gate, not a footnote; 5) every gate is "
        "machine-evaluated from run JSONs (no hand-copied numbers)."
    )
    transfer = (
        "Any DEM packing/porosity case in TensorDEM or TensorFEM coupling can reuse "
        "the protocol: copy DepositionSpec + run_deposition, set kn ladder and mu, "
        "and use stiffness_limit/extrapolation_selfconsistency from the same module. "
        "The 'production stiffness by seed coverage' rule and the bulk-band window "
        "are protocol-level and carry over to polydisperse or non-spherical media; "
        "what does NOT carry over is the numeric band (it is RCP-specific)."
    )
    evidence = json.dumps({
        "kn_production": kn, "gates_pass": npass, "gates_total": len(s["gates"]),
        "failing": failing,
        "phi_inf_mu0": s0.get("limit"), "phi_inf_mu0.1": s1.get("limit"),
        "phi_prod_mu0": t0.get("phi_mean"), "phi_prod_mu0.1": t1.get("phi_mean"),
        "reference": phi_ref, "impl_sha256": s.get("impl_sha256"),
        "doc": "docs/CASE_RCP_DEPOSITION.md",
        "method_doc": "docs/OSS_METHOD_SELECTION_RCP.md",
    })
    def q(x):
        return "'" + str(x).replace("'", "''") + "'"

    sql(f"""
INSERT INTO oss_solutions (repo, feature_slug, tag, method, files, conventions, transfer, evidence)
VALUES ('TensorDEM', 'CASE-DEM-RCP-PACKING-FRACTION', 'round2-deposition-rigid-limit',
        {q(method)}, {q(files)}, {q(conventions)}, {q(transfer)}, {q(evidence)})
ON CONFLICT (repo, feature_slug, tag) DO UPDATE
  SET method = EXCLUDED.method, files = EXCLUDED.files,
      conventions = EXCLUDED.conventions, transfer = EXCLUDED.transfer,
      evidence = EXCLUDED.evidence;
""", "oss_solution")

    sql(f"""
INSERT INTO negative_knowledge_ledger
  (entry_id, mission_id, node_id, title, hypothesis, falsification_evidence,
   failure_reason, failure_kind, applicable_conditions, keywords, domain_tags,
   source_artifacts, confidence, registered_by, status)
VALUES
  ('negk:m18143:no-oss-reference', 18143, '18143:oss-probe',
   'TensorDEM RCP 沉积堆积在 OSS 库存/缓存内无同类实现可借鉴',
   '期望在 oss_inventory 或集群 ~/oss-cache 找到可直接借鉴的随机密堆/沉积堆积实现（RCP/RLP 协议或 DEM packing 模块）',
   'oss_inventory 中 summary/focus/methods ILIKE %packing% 命中 0 条；repo 名含 dem/granular/liggghts/yade/lammps/mercury 命中 0 条；唯一 MD 平台 espressomd/espresso 的条目只登记 MD-LBM 耦合。集群 ~/oss-cache 全文 grep 作业见 benchmarks/case_rcp_deposition/runs/logs/ 与 docs/OSS_METHOD_SELECTION_RCP.md §①。',
   '迁移不能靠移植代码，只能靠移植文献协议 + 本仓自建内核（src/tensordem/deposition_packing.py）',
   'hypothesis_falsified', '{{"repo":"TensorDEM","feature":"RCP packing fraction","scope":"oss-cache + oss_inventory"}}',
   '["TensorDEM","RCP","packing","oss-reference","deposition"]',
   '["granular","packing","dem"]',
   '["docs/OSS_METHOD_SELECTION_RCP.md","benchmarks/case_rcp_deposition/"]',
   0.8, 'agent:m18143', 'active')
ON CONFLICT (mission_id, node_id) DO NOTHING;
""", "negk")

    rc, out = sh("DRY=0 python3 post_report_m18143.py")
    log(f"report rc={rc}\n{out[-600:]}")
    (REPO / ".scratch" / "FINALIZED").write_text(
        json.dumps({"gates_pass": npass, "total": len(s["gates"]),
                    "failing": failing, "production_ok": ok}))
    log("finaliser done")
    return 0


if __name__ == "__main__":
    sys.exit(main())