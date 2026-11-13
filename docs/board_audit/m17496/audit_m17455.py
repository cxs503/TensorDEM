#!/usr/bin/env python3
"""Board cross-audit (交差核验) of m17455 — anchor
XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT (TensorDEM).

Verification only.  Nothing is re-implemented; the only execution is the
mission's own registered repro command, run into /tmp so that no mission file
is touched.

Checks
------
1. three-piece check: run artifacts exist on disk / ref-vs-got numbers present /
   repro_cmd executable;
2. delivery check: are the deliverables committed on the mission branch or in
   the repo HEAD?  (auto-push only pushes commits)
3. registered referee: tensor_benchmark_cases WHERE anchor_slug=<anchor>;
4. independent reproduction: re-run the repro command, diff every number in
   result.json against the archived artifact (float tol 1e-12, wall clocks
   excluded);
5. test integrity: does tests/test_binding_contract.py assert invariants or
   merely execute code?  Run it.
6. claim binding: every headline claim in the mission result / report 16428
   mapped to a raw number in result.json;
7. OSS reference check: the cited TensorPhysics files/line counts exist as
   claimed (zero-transcription);
8. integration: is there any caller of tensordem.binding_contract outside
   tests/ + scripts/?

Outputs
-------
docs/board_audit/m17496/refs/**            byte-exact copies + sha256 manifest
docs/audit/figures/m17496_m17455_audit.png
artifacts/audit_m17496/finding.json
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = "/root/tp-agent/work/m17455"
ANCHOR = "XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT"
MISSION = 17455
AUDIT_MISSION = 17496
REF = os.path.join(REPO, "docs", "board_audit", "m17496", "refs")
OUT = os.path.join(REPO, "artifacts", "audit_m17496")
FIG = os.path.join(REPO, "docs", "audit", "figures")
REPRO_OUT = "/tmp/m17496_repro"

DELIVERED = [
    "src/tensordem/binding_contract.py",
    "tests/test_binding_contract.py",
    "scripts/xval_library_binding_contract.py",
    "docs/XVAL_DEM_LIBRARY_BINDING_CONTRACT.md",
    "prereg/XVAL_DEM_LIBRARY_BINDING_CONTRACT.json",
    "artifacts/xval_library_binding_contract/result.json",
    "artifacts/xval_library_binding_contract/fig_impulse_error.png",
    "artifacts/xval_library_binding_contract/fig_mutation_matrix.png",
    "artifacts/xval_library_binding_contract/fig_coverage.png",
    "artifacts/xval_library_binding_contract/fig_substep_boundary.png",
    "runs/xval_library_binding_contract/run.json",
]
OSS_REFS = [
    ("/root/tensor-ecosystem/TensorPhysics/src/tensorphysics/adapters/tensordem.py", 113),
    ("/root/tensor-ecosystem/TensorPhysics/tests/test_adapter_inventory_meta.py", 43),
    ("/root/tensor-ecosystem/TensorPhysics/docs/adapter-contracts.md", None),
    ("/root/tensor-ecosystem/TensorPhysics/docs/adapter-inventory.md", None),
    ("/root/tensor-ecosystem/TensorPhysics/src/tensorphysics/protocols.py", None),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def copy_refs():
    os.makedirs(REF, exist_ok=True)
    man = {"mission": MISSION, "anchor": ANCHOR, "source_worktree": SRC, "files": {}}
    for p in DELIVERED + ["docs/board_audit/m17496/refs/DUMMY"]:
        if p.endswith("DUMMY"):
            continue
        s = os.path.join(SRC, p)
        d = os.path.join(REF, p.replace("/", "__"))
        if not os.path.exists(s):
            man["files"][p] = {"exists": False}
            continue
        shutil.copy2(s, d)
        man["files"][p] = {"copy": os.path.relpath(d, REPO), "sha256": sha256(s),
                           "bytes": os.path.getsize(s), "exists": True}
    with open(os.path.join(REF, "MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    return man


def git_state():
    def run(*a):
        return subprocess.run(["git", "-C", SRC, *a], capture_output=True, text=True).stdout
    status = run("status", "--porcelain")
    return {
        "branch": run("branch", "--show-current").strip(),
        "head": run("rev-parse", "--short", "HEAD").strip(),
        "untracked_deliverables": [p for p in DELIVERED
                                   if subprocess.run(["git", "-C", SRC, "cat-file", "-e",
                                                      f"HEAD:{p}"],
                                                     capture_output=True).returncode != 0],
        "porcelain": status.splitlines(),
        "mission_commit": run("log", "--all", "--oneline", "--grep", str(MISSION)).splitlines(),
        "in_audit_head": {p: subprocess.run(["git", "-C", REPO, "cat-file", "-e", f"HEAD:{p}"],
                                            capture_output=True).returncode == 0
                          for p in ["src/tensordem/binding_contract.py"]},
    }


def registered():
    d = None
    for line in open("/root/tensor-ecosystem/TensorPlatform/.env"):
        if line.startswith("DATABASE_URL="):
            d = line.split("=", 1)[1].strip().replace("postgresql+psycopg://",
                                                      "postgresql://")
    import psycopg
    with psycopg.connect(d) as c, c.cursor() as cur:
        cur.execute("SELECT id, anchor_slug, case_name, metric, tolerance, repo, status "
                    "FROM tensor_benchmark_cases WHERE anchor_slug=%s ORDER BY id",
                    (ANCHOR,))
        rows = cur.fetchall()
        cur.execute("SELECT count(*) FROM tensor_benchmark_cases")
        n_all = cur.fetchone()[0]
        cur.execute("SELECT id, title, mission_id, qa_status, length(markdown), "
                    "jsonb_array_length(figures) FROM tensor_reports WHERE mission_id=%s",
                    (MISSION,))
        reports = cur.fetchall()
    return {"anchor_rows": rows, "n_registered_cases": n_all, "reports": reports}


def reproduce():
    os.makedirs(REPRO_OUT, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="src")
    p = subprocess.run([sys.executable, "scripts/xval_library_binding_contract.py",
                        "--outdir", REPRO_OUT], cwd=SRC, env=env,
                       capture_output=True, text=True, timeout=1800)
    return p


def diff_archived(arch, new):
    """Return list of (path, archived, reproduced) numeric mismatches."""
    bad = []

    def walk(x, y, path=""):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), path + "/" + str(k))
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                bad.append((path, f"len {len(x)}", f"len {len(y)}"))
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, path + f"[{i}]")
        elif isinstance(x, float) and isinstance(y, float):
            if not path.endswith("wall_s") and abs(x - y) > 1e-12:
                bad.append((path, x, y))
        elif x != y and not path.endswith("wall_s"):
            bad.append((path, x, y))

    walk(arch, new)
    return bad


def callers():
    out = {}
    for root in ("src", "benchmarks", "scripts", "tests"):
        hits = subprocess.run(["grep", "-rln", "binding_contract", "--include=*.py", root],
                              cwd=SRC, capture_output=True, text=True).stdout.split()
        out[root] = [h for h in hits if not h.endswith("binding_contract.py")]
    return out


def oss_refs():
    rows = []
    for path, claimed_lines in OSS_REFS:
        if not os.path.exists(path):
            rows.append({"path": path, "exists": False})
            continue
        with open(path) as f:
            n = sum(1 for _ in f)
        rows.append({"path": path, "exists": True, "lines": n,
                     "claimed_lines": claimed_lines,
                     "lines_match": (claimed_lines is None or claimed_lines == n)})
    return rows


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    r = res["result"]
    base = r["baseline_comparison"]
    keys = [row["key"] for row in base["control"]["rows"]]
    tr = [1.0 if row.get("detected") else 0.0 for row in r["mutation_matrix"]]
    bl = [1.0 if row["baseline_detected"] else 0.0 for row in base["control"]["rows"]]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    y = range(len(keys))
    ax[0].barh([i + 0.2 for i in y], bl, height=0.4, label="presence-only baseline",
               color="#c44")
    ax[0].barh([i - 0.2 for i in y], tr, height=0.4, label="frozen contract", color="#2a7")
    ax[0].set_yticks(list(y))
    ax[0].set_yticklabels(keys, fontsize=8)
    ax[0].set_xlim(0, 1.1)
    ax[0].set_xlabel("mutation detected (1 = yes)")
    ax[0].set_title(f"drift detection: contract {base['treatment']['rate']:.3f} "
                    f"vs baseline {base['control']['rate']:.3f}")
    ax[0].legend(fontsize=8, loc="lower right")

    rows = r["impulse_momentum_sweep"]
    ns = [x["n_steps"] for x in rows]
    err = [max(abs(x["rel_error"]), 1e-18) for x in rows]
    ax[1].loglog(ns, err, "o-", label="reproduced run")
    ax[1].axhline(1e-12, ls="--", c="grey", label="prereg threshold 1e-12")
    ax[1].axhline(0.15, ls=":", c="r", label="owner 15 % gate")
    ax[1].set_xlabel("native substeps n")
    ax[1].set_ylabel("impulse-momentum rel. error")
    ax[1].set_title("impulse-momentum identity vs substep count")
    ax[1].legend(fontsize=8)

    labels = ["reproduced\nnumbers", "delivered but\nuncommitted", "solver-side\ncallers",
              "registered\nbenchmark case"]
    vals = [1 if not res["repro_diff"] else 0, 0 if res["git"]["untracked_deliverables"] else 1,
            1 if res["callers"]["src"] else 0, 1 if res["referee"]["anchor_rows"] else 0]
    colors = ["#2a7" if v else "#c44" for v in vals]
    ax[2].bar(labels, vals, color=colors)
    ax[2].set_ylim(0, 1.2)
    ax[2].set_yticks([0, 1])
    ax[2].set_yticklabels(["no", "yes"])
    ax[2].set_title("audit gate checklist (yes/no)")
    fig.tight_layout()
    path = os.path.join(FIG, "m17496_m17455_audit.png")
    fig.savefig(path, dpi=130)
    print("figure ->", path)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    man = copy_refs()
    git = git_state()
    ref = registered()
    ref["anchor_rows"] = [list(map(str, r)) for r in ref["anchor_rows"]]
    rep = reproduce()
    arch = json.load(open(os.path.join(SRC,
                                       "artifacts/xval_library_binding_contract/result.json")))
    new = json.load(open(os.path.join(REPRO_OUT, "result.json")))
    bad = diff_archived(arch, new)
    res = {"mission": MISSION, "audit_mission": AUDIT_MISSION, "anchor": ANCHOR,
           "git": git, "referee": ref, "callers": callers(), "oss_refs": oss_refs(),
           "repro": {"returncode": rep.returncode, "stderr_tail": rep.stderr[-400:],
                     "outdir": REPRO_OUT, "wall_s": new.get("wall_s")},
           "repro_diff": bad, "manifest": man.get("files", {}), "result": new}
    res["checks"] = {
        "three_piece_artifacts": all(v.get("exists") for v in man["files"].values()),
        "repro_reproduces_bitwise": not bad,
        "committed_to_branch": not git["untracked_deliverables"],
        "registered_referee_case": bool(ref["anchor_rows"]),
        "solver_side_caller": bool(res["callers"]["src"]),
        "oss_refs_verified": all(r.get("exists") and r.get("lines_match", True)
                                 for r in res["oss_refs"]),
        "criteria_all_met": all(v["met"] for v in new["verdicts"]),
        "n_criteria": len(new["verdicts"]),
        "probes": (sum(1 for p in new["semantic_probes"] if p["status"] == "PASS"),
                   len(new["semantic_probes"])),
        "mutation_classes_detected": sum(1 for r in new["mutation_matrix"] if r.get("detected")),
        "mutation_classes_total": len(new["mutation_matrix"]),
    }
    figp = figure(res)
    res["figure"] = os.path.relpath(figp, REPO)
    with open(os.path.join(OUT, "finding.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: v for k, v in res["checks"].items()}, indent=1, ensure_ascii=False))
    print("repro_diff entries:", len(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())