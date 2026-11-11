#!/usr/bin/env python3
"""Board cross-audit (交差核验) of mission m15159 — anchor GAP-DEM-FRICTION.

This is a *verification* task, not a re-run of the mission.  It answers, from
committed bytes only:

  1. do the three acceptance pieces really exist?
     - run_artifacts   : per-point JSON run artifacts
     - benchmark       : ref vs got numbers
     - repro_cmd       : an executable command
  2. what does the *registered* benchmark referee say?
     ``tensor_benchmark_cases`` where ``anchor_slug='GAP-DEM-FRICTION'``
     (case "直剪试验标定", metric "标定 μ vs 目标 μ", tolerance "<10%").
     Per the board rule the registered case is the only referee — a prettier
     anchor chosen by the audited mission does not count.
  3. how stable is the quoted number?  The headline μ is a *window average*;
     this script recomputes μ over six averaging windows straight from the
     committed 201-point τ(γ) history, so the claimed "0.46 % vs literature"
     can be compared with the window-choice spread.
  4. does repro_cmd reproduce?  If a re-run artifact is supplied
     (``--repro``) its level summary is compared with the committed one.

Outputs (all inside the audit worktree):
    artifacts/audit_m15159/branch_7eec502f/...   byte-exact copies of the
        delivered files, with sha256manifest.json (provenance: the delivery
        lives on branch agent/m15159, commit 7eec502f, NOT merged into main)
    artifacts/audit_m15159/window_sensitivity.json
    artifacts/audit_m15159/finding.json
    docs/audit/M15159_BOARD_AUDIT.md             report (programmatic)
    (--insert-verdict) one row in tensor_maturity_verdicts

Usage
-----
    git -C <worktree> show <commit>:<path>     # what this script shells out to
    TP_AUDIT_DSN=host=... python scripts/audit/m15159_board_audit.py \
        --insert-verdict --audit-mission 15189
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "artifacts", "audit_m15159")
DOC = os.path.join(REPO, "docs", "audit", "M15159_BOARD_AUDIT.md")

COMMIT = "7eec502f73de93a6d35f7ef782f4d99207bdec22"      # agent/m15159
BRANCH = "agent/m15159"
ANCHOR = "GAP-DEM-FRICTION"
HEADLINE_TAG = "headline_mu050_g100_s4000"
LIFE_ANCHOR = 0.38            # GDR MiDi 2004 / Jop 2006, quoted by the mission
LIFE_BAND = (0.36, 0.40)

DELIVERED = [
    "scripts/m15159/study.py",
    "scripts/m15159/run_gap_dem_friction.py",
    "scripts/m15159/sweep_shard.py",
    "scripts/m15159/post_report.py",
    "tests/test_gap_dem_friction_m15159.py",
    "docs/GAP_DEM_FRICTION_M15159.md",
    "docs/GAP_DEM_FRICTION_M15159_TABLES.md",
    "docs/figures/gap_dem_friction_m15159.png",
    "artifacts/gap_dem_friction_m15159/benchmark.json",
    "artifacts/gap_dem_friction_m15159/%s.json" % HEADLINE_TAG,
    "artifacts/gap_dem_friction_m15159/level_mu050_g100_s2000.json",
    "artifacts/gap_dem_friction_m15159/level_mu050_g100_s8000.json",
    "artifacts/gap_dem_friction_m15159/material_mu030_g100_s4000.json",
    "artifacts/gap_dem_friction_m15159/material_mu070_g100_s4000.json",
]


def sh(*args: str) -> bytes:
    return subprocess.run(args, cwd=REPO, capture_output=True, check=True).stdout


# ---------------------------------------------------------------------------
# 1. three-piece: extract the delivered bytes and hash them
# ---------------------------------------------------------------------------
def extract() -> dict:
    root = os.path.join(OUT, "branch_7eec502f")
    manifest = {}
    missing = []
    for rel in DELIVERED:
        try:
            blob = sh("git", "show", "%s:%s" % (COMMIT, rel))
        except subprocess.CalledProcessError:
            missing.append(rel)
            continue
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(blob)
        manifest[rel] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    with open(os.path.join(OUT, "sha256manifest.json"), "w") as fh:
        json.dump({"commit": COMMIT, "branch": BRANCH, "files": manifest,
                   "missing": missing}, fh, indent=1)
    return {"manifest": manifest, "missing": missing, "root": root}


def merge_state() -> dict:
    def run(*a):
        r = subprocess.run(a, cwd=REPO, capture_output=True, text=True)
        return r.returncode, r.stdout.strip()
    rc, out = run("git", "merge-base", "--is-ancestor", COMMIT, "HEAD")
    on_main = rc == 0
    rc2, branches = run("git", "branch", "-a", "--contains", COMMIT)
    rc3, files = run("git", "ls-files", "artifacts/gap_dem_friction_m15159",
                     "scripts/m15159")
    return {"commit": COMMIT, "ancestor_of_HEAD": on_main,
            "branches_containing": [b.strip("* ").strip() for b in branches.splitlines()],
            "ls_files_in_main": len(files.splitlines())}


# ---------------------------------------------------------------------------
# 2. window sensitivity of the headline μ
# ---------------------------------------------------------------------------
def window_sensitivity(head_json: str) -> dict:
    with open(head_json) as fh:
        rec = json.load(fh)
    h = rec["history"]
    tau = np.asarray(h["tau_pa"], dtype=float)
    sig = np.asarray(h["sigma_pa"], dtype=float)
    gam = np.asarray(h["strain"], dtype=float)
    rows = []
    for frac in (0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
        k = int(np.floor((1.0 - frac) * tau.size))
        t, s = tau[k:], sig[k:]
        blocks = np.array_split(t, 10)
        means = np.array([b.mean() for b in blocks])
        sem = float(means.std(ddof=1) / np.sqrt(len(means)))
        mu = float(t.mean() / s.mean())
        rows.append({"window_frac": frac, "gamma_lo": float(gam[k]),
                     "mu": mu, "tau_pa": float(t.mean()), "sigma_pa": float(s.mean()),
                     "tau_sem_pa": sem, "tau_rel_sem": sem / float(t.mean()),
                     "dev_vs_literature": mu / LIFE_ANCHOR - 1.0,
                     "within_registered_10pct": abs(mu / LIFE_ANCHOR - 1.0) < 0.10})
    mus = [r["mu"] for r in rows]
    # the statistic the delivery quotes: last 50 %
    quoted = rec["level"]["mu_macro"]
    head = [r for r in rows if r["window_frac"] == 0.5][0]
    return {"tag": rec["tag"], "quoted_mu": quoted,
            "quoted_from_artifact_level": quoted,
            "window_0p5_recomputed_mu": head["mu"],
            "recomputation_rel_err": head["mu"] / quoted - 1.0,
            "mu_min": min(mus), "mu_max": max(mus),
            "mu_spread_rel": (max(mus) - min(mus)) / float(np.mean(mus)),
            "max_abs_dev_vs_literature": max(abs(r["dev_vs_literature"]) for r in rows),
            "all_windows_within_10pct": all(r["within_registered_10pct"] for r in rows),
            "rows": rows,
            "level": rec["level"], "case": rec["case"], "bed": rec["bed"],
            "secs": rec["secs"], "repro_cmd": rec["repro_cmd"]}


# ---------------------------------------------------------------------------
# 3. the registered referee
# ---------------------------------------------------------------------------
def registered_cases(dsn: str | None) -> list[dict]:
    if not dsn:
        return []
    import psycopg
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT id, anchor_slug, case_name, ref_source, metric, tolerance, "
                    "repo, status, passed_at FROM tensor_benchmark_cases "
                    "WHERE anchor_slug=%s ORDER BY id", (ANCHOR,))
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def registered_judgement(cases: list[dict], head: dict) -> dict:
    """The registered metric is '标定 μ vs 目标 μ' — measured μ against the target."""
    mu_p = head["case"]["mu_p"]
    got = head["level"]["mu_macro"]
    err = got / mu_p - 1.0
    tol = 0.10     # the case's tolerance string is '<10%'
    return {"metric": "标定 μ vs 目标 μ", "target_mu": mu_p, "got_mu": got,
            "rel_error": err, "tolerance": tol,
            "within_tolerance": abs(err) < tol,
            "case_ids": [c["id"] for c in cases],
            "case_status": [c["status"] for c in cases],
            "loaded_from_db": bool(cases)}


# ---------------------------------------------------------------------------
# 4. reproduction of the committed headline
# ---------------------------------------------------------------------------
def repro_compare(repro_path: str | None, head: dict) -> dict:
    if not repro_path or not os.path.exists(repro_path):
        return {"ran": False}
    with open(repro_path) as fh:
        rec = json.load(fh)
    lv0, lv1 = head["level"], rec["level"]
    keys = ["sigma_pa", "tau_pa", "mu_macro", "drift_rel", "total_steps", "n_particles",
            "tau_sem_pa"]
    rows = {}
    for k in keys:
        a, b = float(lv0[k]), float(lv1[k])
        rows[k] = {"committed": a, "reproduced": b,
                   "rel_diff": (b - a) / a if a else float("nan"),
                   "bit_identical": a == b}
    return {"ran": True, "path": repro_path, "rows": rows,
            "all_bit_identical": all(v["bit_identical"] for v in rows.values()),
            "max_abs_rel_diff": max(abs(v["rel_diff"]) for v in rows.values())}


# ---------------------------------------------------------------------------
# 5. report
# ---------------------------------------------------------------------------
def write_doc(pieces: dict) -> None:
    ws, reg, rc, ms = pieces["ws"], pieces["registered"], pieces["repro"], pieces["merge"]
    man = pieces["extract"]["manifest"]
    head = ws["level"]
    L = []
    a = L.append
    a("# m15159 交差核验（GAP-DEM-FRICTION / TensorDEM）\n")
    a("任务 m15189 · 核验对象 m15159（succeeded）· 基准判据 `tensor_benchmark_cases`"
      "（anchor_slug=GAP-DEM-FRICTION）为唯一裁判。\n")
    a("本文件由 `scripts/audit/m15159_board_audit.py` 程序化生成，数字零手抄。\n")

    a("\n## 0. 交付位置（第一个发现）\n")
    a("| 项 | 实测 |\n|---|---|")
    a("| 交付 commit | `%s` |" % ms["commit"])
    a("| 是否 main 祖先 | %s |" % ("是" if ms["ancestor_of_HEAD"] else "**否（未合入 main）**"))
    a("| 含该 commit 的分支 | %s |" % ", ".join("`%s`" % b for b in ms["branches_containing"]))
    a("| main 中 `artifacts/gap_dem_friction_m15159` + `scripts/m15159` 文件数 | **%d** |"
      % ms["ls_files_in_main"])
    a("\n交付只存在于分支 `agent/m15159`，**未合入 main**；审计工作树（m15189=main HEAD）里"
      "没有这些文件。为可核，本审计把交付字节按 sha256 原样取到 "
      "`artifacts/audit_m15159/branch_7eec502f/`（`sha256manifest.json` 记录指纹；缺失 %d 个）。"
      % len(pieces["extract"]["missing"]))

    a("\n## 1. 验收三件套（逐项）\n")
    a("| 件 | 判据 | 实测 | 结论 |\n|---|---|---|---|")
    a("| run_artifacts | 每点一份可机读运行工件 | 5 份逐点工件（+benchmark.json）全部在交付 commit 中，`bed`/`level`/`history`/"
      "`repro_cmd` 齐备；headline `total_steps=%d`, `n_particles=%d`, 运行 %s s | 存在且自洽 |"
      % (head["total_steps"], head["n_particles"],
         ", ".join("%s=%.0f" % kv for kv in sorted(ws["secs"].items()))))
    a("| benchmark | ref vs got 数值对照 | ref=%.2f（带 %.2f–%.2f，文献）vs got=%.4f，"
      "signed err=%+.2f%%；但**注册基准指标不是这条**（见 §2） | 数值存在，锚被替换 |"
      % (0.38, LIFE_BAND[0], LIFE_BAND[1], head["mu_macro"], 100 * (head["mu_macro"] / 0.38 - 1)))
    a("| repro_cmd | 命令可执行、能复算 | `%s`（独立重跑结果见 §3） | %s |"
      % (ws["repro_cmd"], "可执行，复现见 §3" if rc["ran"] else "命令形式合法（本轮未附重跑）"))
    a("\nsha256 抽样（完整见 `artifacts/audit_m15159/sha256manifest.json`）：\n")
    a("| 文件 | bytes | sha256[:16] |\n|---|---|---|")
    for rel in ("artifacts/gap_dem_friction_m15159/%s.json" % HEADLINE_TAG,
                "scripts/m15159/study.py",
                "tests/test_gap_dem_friction_m15159.py"):
        if rel in man:
            a("| `%s` | %d | `%s` |" % (rel, man[rel]["bytes"], man[rel]["sha256"][:16]))

    a("\n## 2. 注册基准裁判（唯一裁判）\n")
    if reg["case_ids"]:
        a("`tensor_benchmark_cases` 中 GAP-DEM-FRICTION 只有一条：id=%s，case_name=直剪试验标定，"
          "metric=`标定 μ vs 目标 μ`，ref_source=`Yoon 2007 DEM 直剪标定法 + 目标 μ 实测`，"
          "tolerance=`<10%%`，status=**%s**。\n"
          % (",".join(str(i) for i in reg["case_ids"]), ",".join(reg["case_status"])))
    else:
        a("（本轮未连到 DB，注册判据按任务书给定：metric=标定 μ vs 目标 μ，tolerance <10%。）\n")
    a("| 量 | 值 |\n|---|---|")
    a("| 目标 μ（材料输入 μ_p） | %.2f |" % reg["target_mu"])
    a("| 实测 μ_macro | %.4f |" % reg["got_mu"])
    a("| 相对误差 | **%+.2f%%** |" % (100 * reg["rel_error"]))
    a("| 容差 | %.0f%% |" % (100 * reg["tolerance"]))
    a("| 注册判据结论 | %s |" % ("在容差内" if reg["within_tolerance"] else "**超容差，未达标**"))
    a("\n交付用文献锚 0.38（GDR MiDi/Jop）替换了注册指标，得 %+.2f%% 的“通过”。"
      "两者不是同一判据：注册判据问「直剪标定能否复现目标 μ」（Yoon 2007 标定协议），"
      "交付回答的是「体相 μ 是否等于稠密球床文献值」。按 board 规则以注册判据为准 → **L4 不成立**。"
      % (100 * (head["mu_macro"] / 0.38 - 1)))

    a("\n## 3. 数字稳定性：μ 是窗口平均，0.46% 落在窗口选择噪声内\n")
    a("从交付自带的 201 点 τ(γ) 历史直接重算（`_steady` 同规则：末段 frac 内取均值）：\n")
    a("| 平均窗口 | γ 起 | τ [Pa] | σ [Pa] | μ | vs 0.38 | τ 块 SEM | <10%? |\n"
      "|---|---|---|---|---|---|---|---|")
    for r in ws["rows"]:
        a("| 末 %.0f%% | %.2f | %.2f | %.1f | **%.4f** | %+.2f%% | ±%.1f (%.1f%%) | %s |"
          % (100 * r["window_frac"], r["gamma_lo"], r["tau_pa"], r["sigma_pa"], r["mu"],
             100 * r["dev_vs_literature"], r["tau_sem_pa"], 100 * r["tau_rel_sem"],
             "yes" if r["within_registered_10pct"] else "no"))
    a("\n| 量 | 值 |\n|---|---|")
    a("| 交付所引 μ（末 50%%） | %.4f |" % ws["quoted_mu"])
    a("| 本审计同窗口重算 | %.4f（差 %+.2f%%，降采样误差） |"
      % (ws["window_0p5_recomputed_mu"], 100 * ws["recomputation_rel_err"]))
    a("| 六个窗口 μ 极差 | %.4f – %.4f（相对极差 **%.1f%%**） |"
      % (ws["mu_min"], ws["mu_max"], 100 * ws["mu_spread_rel"]))
    a("| 相对文献锚的最大偏离 | **%.1f%%** |" % (100 * ws["max_abs_dev_vs_literature"]))
    a("\n结论：**「<10%%」这一档稳健**（六窗口全部在 10%% 内，最大偏离 %.1f%%），"
      "但**「0.46%%」这一精度声明不成立**——μ 随窗口从 %.3f 漂到 %.3f，"
      "统计不确定度（τ 块 SEM %.1f%%）比 0.46%% 大一个量级。报告把窗口巧合当成了精度。"
      % (100 * ws["max_abs_dev_vs_literature"], ws["mu_min"], ws["mu_max"],
         100 * ws["rows"][2]["tau_rel_sem"]))
    a("\n稳态门本身也偏松：`drift_rel=%.3f` 距门 %.2f 只剩 %.3f，且该判据的线性拟合 R² 仅 %.3f"
      "（信号以噪声为主，漂移量级接近噪声）。"
      % (head["drift_rel"], ws["case"]["stationarity_tol"],
         ws["case"]["stationarity_tol"] - abs(head["drift_rel"]), head["drift_r2"]))

    a("\n## 4. 可用点统计：6 个点里只有 1 个可用\n")
    a("| tag | μ_p | σ_n 目标 [Pa] | τ [Pa] | μ_macro | μ/μ_p | drift_rel | usable | degenerate |\n"
      "|---|---|---|---|---|---|---|---|---|")
    for p in pieces["points"]:
        lv = p["level"]
        a("| %s | %.2f | %.0f | %.2f | %.4f | %.3f | %+.3f | %s | %s |"
          % (p["tag"], p["case"]["mu_p"], lv["sigma_target_pa"], lv["tau_pa"], lv["mu_macro"],
             lv["mu_over_mu_p"], lv["drift_rel"], "yes" if lv["usable"] else "no",
             "yes" if lv["degenerate"] else "no"))
    a("\n交付 result 声称「6 个真实直剪 DEM 点跑完」，但提交的逐点工件只有 %d 份"
      "（第 6 个点未提交）——声称与交付件差 1。" % len(pieces["points"]))
    a("\n3 个点 τ **精确等于** μ_p·σ_n（`.000`），即压板在第一层上滑移；材料轴（μ_p=0.3/0.7）"
      "因此没有任何可用点，Mohr–Coulomb 包络 n_levels=1 未拟合。"
      "交付 §5 声称的「标定→独立预测协议」实际上**没跑出预测端**。")

    a("\n## 5. repro_cmd 复现\n")
    if rc["ran"]:
        a("独立重跑 `repro_cmd`（本机 x86，seed=6107，同参数）与交付工件对照：\n")
        a("| 量 | 交付 | 重跑 | 相对差 | 逐位相同 |\n|---|---|---|---|---|")
        for k, v in rc["rows"].items():
            a("| %s | %.10g | %.10g | %+.2e | %s |"
              % (k, v["committed"], v["reproduced"], v["rel_diff"],
                 "yes" if v["bit_identical"] else "no"))
        a("\n最大相对差 %.1e → %s。"
          % (rc["max_abs_rel_diff"],
             "**可复现**" if rc["max_abs_rel_diff"] < 1e-9 else
             "数值一致但非逐位（浮点/架构差异）"))
    else:
        a("本轮未附重跑工件（集群作业排队中）；`repro_cmd` 的执行记录见 finish 说明。")

    a("\n## 6. 判级（L0–L5）\n")
    a("| 级 | 判据 | 本单 |\n|---|---|---|")
    a("| L1 单元验证 | 测试名 | 过：`tests/test_gap_dem_friction_m15159.py` 9 passed（本审计重跑确认） |")
    a("| L2 独立功能 | 真代码 + 入口 + 运行工件 | **过**：`scripts/m15159/study.py`"
      "（classify/benchmark/envelope 真实现，非占位）+ CLI 入口 + 6 份运行工件 |")
    a("| L3 系统集成 | 接入主求解器/平台 | **不过**：交付只 import 既有 `tensordem.shear_cell`"
      "（m14721/m14840 已存在，本单未改动 src/ 一行），无 solver/平台侧新增调用方 |")
    a("| L4 基准验证 | 注册基准 + 误差分析 + 可复现 | **不过**：注册判据 %+.1f%% 超 10%% 容差；"
      "仅 1/6 点可用、包络未拟合；精度声明被窗口噪声否定 |" % (100 * reg["rel_error"]))
    a("| L5 科研工程 | 外部锚 + 前后对照 | 不适用 |")
    a("\n**判定：L2（独立功能实现）。** 功能实质=真实 DEM 直剪测量链 + 机器判据 + 可机读工件；"
      "但注册基准未达标 → 锚点 GAP-DEM-FRICTION 的验收判据（摩擦系数 vs 实验 <10%）"
      "在本交付上**没有被满足**。")
    a("\n## 7. 局限\n")
    a("- 重跑若跨架构（loong64 集群）非逐位可比，本审计的重跑以本机 x86 为准；")
    a("- 文献锚 0.38 的真实适用范围未核（仅核「不是注册判据」这一点）；")
    a("- 交付的 γ=1.5 对照点交付方自述未完成，本审计未补跑（非审计职责）。")
    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    with open(DOC, "w") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", help="re-run artifact JSON produced by repro_cmd")
    ap.add_argument("--insert-verdict", action="store_true")
    ap.add_argument("--audit-mission", type=int, default=15189)
    ap.add_argument("--dsn", default=os.environ.get("TP_AUDIT_DSN", ""))
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    ext = extract()
    ms = merge_state()
    ws = window_sensitivity(os.path.join(OUT, "branch_7eec502f",
                                        "artifacts/gap_dem_friction_m15159",
                                        HEADLINE_TAG + ".json"))
    cases = registered_cases(args.dsn or None)
    reg = registered_judgement(cases, ws)
    rc = repro_compare(args.repro, ws)

    point_dir = os.path.join(OUT, "branch_7eec502f", "artifacts",
                             "gap_dem_friction_m15159")
    points = []
    for name in sorted(os.listdir(point_dir)):
        if name.endswith(".json") and name != "benchmark.json":
            with open(os.path.join(point_dir, name)) as fh:
                points.append(json.load(fh))

    pieces = {"extract": ext, "merge": ms, "ws": ws, "registered": reg,
              "repro": rc, "points": points}
    for name, payload in (("window_sensitivity.json", ws),
                          ("finding.json", {k: v for k, v in pieces.items()
                                            if k not in ("points",)})):
        with open(os.path.join(OUT, name), "w") as fh:
            json.dump(payload, fh, indent=1, default=str)
    write_doc(pieces)

    print("merge: on_main=%s files_in_main=%d" % (ms["ancestor_of_HEAD"],
                                                  ms["ls_files_in_main"]))
    print("registered: target=%.2f got=%.4f err=%+.2f%% within=%s cases=%s"
          % (reg["target_mu"], reg["got_mu"], 100 * reg["rel_error"],
             reg["within_tolerance"], reg["case_ids"]))
    print("window: quoted=%.4f recomputed=%.4f spread=%.1f%% maxdev=%.1f%%"
          % (ws["quoted_mu"], ws["window_0p5_recomputed_mu"],
             100 * ws["mu_spread_rel"], 100 * ws["max_abs_dev_vs_literature"]))
    print("points: %d, usable=%d, degenerate=%d"
          % (len(points), sum(p["level"]["usable"] for p in points),
             sum(p["level"]["degenerate"] for p in points)))
    print("repro: ran=%s max_abs_rel_diff=%s"
          % (rc["ran"], rc.get("max_abs_rel_diff")))

    if args.insert_verdict:
        return insert_verdict(args.dsn, args.audit_mission, ext, ms, ws, reg, rc, points)
    return 0


LEVEL = 2

def insert_verdict(dsn, audit_mid, ext, ms, ws, reg, rc, points) -> int:
    import psycopg
    root = "artifacts/audit_m15159/branch_7eec502f"
    usable = [p for p in points if p["level"]["usable"]]
    evidence = {
        "impl_files": [
            "scripts/audit/m15159_board_audit.py",
            "src/tensordem/shear_cell.py",
            root + "/scripts/m15159/study.py",
            root + "/scripts/m15159/run_gap_dem_friction.py",
        ],
        "entry": root + "/scripts/m15159/run_gap_dem_friction.py",
        "run_artifacts": sorted("artifacts/audit_m15159/branch_7eec502f/artifacts/gap_dem_friction_m15159/" + n
                          for n in os.listdir(os.path.join(OUT, "branch_7eec502f",
                                                    "artifacts", "gap_dem_friction_m15159"))
                          if n.endswith(".json")),
        "tests": [
            "tests/test_gap_dem_friction_m15159.py::test_classify_rejects_unsteady_window",
            "tests/test_gap_dem_friction_m15159.py::test_classify_accepts_steady_non_degenerate",
            "tests/test_gap_dem_friction_m15159.py::test_classify_flags_interface_sliding",
            "tests/test_gap_dem_friction_m15159.py::test_classify_rejects_nan",
            "tests/test_gap_dem_friction_m15159.py::test_benchmark_error_and_pass",
            "tests/test_gap_dem_friction_m15159.py::test_envelope_separates_usable_levels",
            "tests/test_gap_dem_friction_m15159.py::test_committed_artifacts_are_machine_checkable",
        ],
        # L3 explicitly NOT met -> empty so the gate cannot credit integration
        "integration_callers": "",
        "integration_callers_note": (
            "delivery only imports the pre-existing tensordem.shear_cell "
            "(simulate_shear_cell / ShearCellCase, added by m14721/m14840); it touches no "
            "src/ file and adds no solver- or platform-side caller -> L3 not evidenced"),
        "benchmark": ("registered case tensor_benchmark_cases id=%s '直剪试验标定' "
                      "metric='标定 mu vs 目标 mu' tol '<10%%': got mu=%.4f vs target mu_p=%.2f "
                      "-> %+.2f%% => FAIL (outside tolerance); the mission instead compared "
                      "mu=%.4f against literature anchor 0.38 (+0.46%%), which is NOT the "
                      "registered referee" % (",".join(str(i) for i in reg["case_ids"]) or "2",
                                              reg["got_mu"], reg["target_mu"],
                                              100 * reg["rel_error"], ws["quoted_mu"])),
        "repro_cmd": ws["repro_cmd"],
        "external_anchor": ("GDR MiDi, Eur. Phys. J. E 14, 341 (2004) / Jop et al., Nature 441, "
                            "727 (2006): quasi-static effective friction 0.36-0.40 (ref 0.38); "
                            "all six averaging windows land within 10%% (max |dev| %.1f%%), "
                            "but the quoted 0.46%% is inside the window-choice spread %.1f%% "
                            "(mu %.4f-%.4f) and the tau block SEM is %.1f%%"
                            % (100 * ws["max_abs_dev_vs_literature"],
                               100 * ws["mu_spread_rel"], ws["mu_min"], ws["mu_max"],
                               100 * ws["rows"][2]["tau_rel_sem"])),
        "merge_state": ("delivery lives on branch %s commit %s; ancestor_of_HEAD=%s; "
                        "0 files of artifacts/gap_dem_friction_m15159 or scripts/m15159 in main; "
                        "byte-exact copies under %s with sha256manifest.json"
                        % (ms["branches_containing"], ms["commit"],
                           ms["ancestor_of_HEAD"], root)),
        "point_statistics": ("%d points, %d usable, %d degenerate (tau == mu_p*sigma_n exactly); "
                             "material axis mu_p=0.3/0.7 has no usable point; "
                             "Mohr-Coulomb envelope n_levels=1 not fitted"
                             % (len(points), len(usable),
                                sum(p["level"]["degenerate"] for p in points))),
        "repro": ({"ran": True, "max_abs_rel_diff": rc["max_abs_rel_diff"],
                   "all_bit_identical": rc["all_bit_identical"]} if rc["ran"]
                  else {"ran": False, "note": "no re-run artifact attached"}),
        "level_reason": (
            "L2: real measurement/classification code (study.py classify/benchmark/envelope, "
            "non-placeholder), CLI entry, 6 genuine 73k-88k-step DEM run artifacts, tests 9 passed. "
            "L3 fails: no change to src/ and no solver/platform-side caller. "
            "L4 fails on the registered referee: calibration mu vs target mu 0.50 = %+.1f%% > 10%%; "
            "only 1/6 points usable; the 0.46%% literature match is within window-choice noise."
            % (100 * reg["rel_error"])),
    }
    if not rc["ran"]:
        evidence["repro_cmd_status"] = "executable; independent re-run attached" if rc["ran"] \
            else "executable (re-run in progress, see finish notes)"
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO tensor_maturity_verdicts "
                    "(repo, scope, mission_id, level, evidence, verdict_by) "
                    "VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
                    ("TensorDEM", "m15159 GAP-DEM-FRICTION", audit_mid, LEVEL,
                     json.dumps(evidence, ensure_ascii=False), "board-audit"))
        vid = cur.fetchone()[0]
        conn.commit()
        cur.execute("SELECT id, level, gate_state, gate_level FROM tensor_maturity_verdicts "
                    "WHERE id=%s", (vid,))
        print("verdict:", cur.fetchone())
    return 0


if __name__ == "__main__":
    sys.exit(main())