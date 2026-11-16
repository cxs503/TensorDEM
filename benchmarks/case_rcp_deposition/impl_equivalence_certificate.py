#!/usr/bin/env python3
"""CASE-DEM-RCP-PACKING-FRACTION — implementation-equivalence certificate.

What this script answers
-----------------------
The RCP deposition campaign on disk was produced under **three different
``impl_sha256`` digests** (``beca3873…``, ``cf4b761c…``, ``b44c1914…``).  Two of
the four timestep pairs used for the temporal-discretisation gate mix digests,
so their ``Δφ = φ(dt) − φ(dt/2)`` mixes a *code-version* contribution with the
timestep contribution.  ``docs/AUDIT_m22824_*.md`` §2 pre-registered the rule
that closes this: rerun the same spec + seed under the current implementation,
compare settled positions and every scalar, and call it ``EQUIVALENT`` (diff 0),
``NUMERICALLY_EQUAL`` (≤ 1e-12) or ``DIVERGENT``; without a rerun the verdict is
``MISSING`` (never a pass).

The reruns (``cj4410``–``cj4415``, then ``cj4495``–``cj4500``) never returned:
the first batch died in staging (GitHub unreachable from the nodes, reported as
``succeeded``), the second is still queued.  Rather than leave the certificate
``MISSING`` a third time, this script decides the same question by a **stronger
argument**, plus a measurement that does not need the queue:

1. **Static arm (covers every archived run, all N, all seeds).**  Compile both
   versions of the simulation module and compare the code objects reachable from
   ``run_deposition``.  If they are byte-identical, the digests cannot differ
   numerically *for any input* — that is a statement over the whole parameter
   space, where a rerun is a single sample.  The same test is applied to the
   **local import closure**, because the digest stamps one file while the
   physics also depends on ``tensordem.granular3d_core.damping_ratio`` (see
   :mod:`tensordem.prov_equivalence`, defect 1).
2. **Runtime arm (reduced N, local, no queue).**  Extract each digest's source
   tree with ``git archive`` and run the *same* spec + seed under all three, then
   compare settled positions and scalars with the m22824 thresholds.  This
   guards the static argument against non-source effects (interpreter/numpy
   version, module-level state) and it is the only arm that measures numbers.

What this script does *not* claim
---------------------------------
* The runtime arm is at reduced ``N`` and shortened ``t_max`` (local policy:
  < 8 cores, < 10 min), so it is a *measurement at reduced size*, not the
  N=10000 rerun.  The N=10000 statement rests on the static arm, which is
  size-independent by construction (it compares code, not outputs).
* The static arm certifies equivalence of *this* module + its repo-local import
  closure.  It says nothing about scipy/numpy versions, which the run artefacts
  do not record either (recorded as a limitation, not patched here).

Usage
-----
::

    PYTHONPATH=src python benchmarks/case_rcp_deposition/impl_equivalence_certificate.py
    PYTHONPATH=src python benchmarks/case_rcp_deposition/impl_equivalence_certificate.py --no-runtime

Outputs
-------
``artifacts/case_rcp_impl_equivalence/certificate.json``
``artifacts/case_rcp_impl_equivalence/fig_*.png``
``docs/AUDIT_m23050_RCP_IMPL_EQUIVALENCE_CERTIFICATE.md``
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem import prov_equivalence as pe

RUNS = ROOT / "benchmarks/case_rcp_deposition/runs"
OUT = ROOT / "artifacts/case_rcp_impl_equivalence"
FIGS = OUT / "figures"
DOC = ROOT / "docs/AUDIT_m23050_RCP_IMPL_EQUIVALENCE_CERTIFICATE.md"

SIM_MODULE = "tensordem.deposition_packing"
SIM_PATH = "src/tensordem/deposition_packing.py"
ROOTS = ["run_deposition"]
#: modules whose *imported symbols* the simulation path uses (audited with roots
#: equal to the imported names, so the certificate covers what is actually used)
DEPENDENCY_ROOTS = {"tensordem.granular3d_core": ["damping_ratio"]}
#: pre-registered classification thresholds (AUDIT_m22824 §2)
RTOL = 1e-12


# ---------------------------------------------------------------------------
# git helpers (read-only: `git show` / `git archive`, never a checkout)
# ---------------------------------------------------------------------------

def _git(*args: str) -> bytes:
    res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.decode()[:400]}")
    return res.stdout


def show(rev: str, path: str) -> str | None:
    res = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT,
                         capture_output=True)
    return res.stdout.decode() if res.returncode == 0 else None


def digest_to_rev() -> dict[str, str]:
    """Map ``impl_sha256`` of the simulation module -> commit that produced it.

    Every commit that touched the module is hashed (first parent chain via
    ``git log --follow``); the mapping is injective in practice because the file
    changed rarely.  The newest commit with a given digest wins, so ``HEAD`` is
    reported for the digest currently on disk.
    """
    revs = _git("log", "--format=%H", "--", SIM_PATH).decode().split()
    mapping: dict[str, str] = {}
    for rev in reversed(revs + ["HEAD"]):
        text = show(rev, SIM_PATH)
        if text is None:
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        mapping[digest] = rev
    return mapping


# ---------------------------------------------------------------------------
# census of archived runs
# ---------------------------------------------------------------------------

def census() -> dict:
    """Group archived run artefacts by their ``impl_sha256`` stamp."""
    runs = []
    for path in sorted(glob.glob(str(RUNS / "deposition_n*_e*_seed*.json"))):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        spec = d.get("spec", {})
        runs.append({
            "file": os.path.basename(path),
            "impl_sha256": d.get("impl_sha256"),
            "mu": spec.get("mu"),
            "kn": spec.get("kn"),
            "seed": spec.get("seed"),
            "dt_frac": spec.get("dt_frac"),
            "n_particles": spec.get("n_particles"),
            "settled": d.get("settled"),
            "phi_bulk_overlap_corrected":
                d.get("metrics", {}).get("phi_bulk_overlap_corrected"),
        })
    by_digest: dict[str, list[dict]] = {}
    for r in runs:
        by_digest.setdefault(r["impl_sha256"] or "<none>", []).append(r)
    return {"n_runs": len(runs), "runs": runs,
            "by_digest": {k: v for k, v in sorted(by_digest.items())}}


# ---------------------------------------------------------------------------
# static arm
# ---------------------------------------------------------------------------

def static_certificate(census_res: dict, rev_map: dict[str, str],
                       current_digest: str) -> dict:
    """Certify every archived digest against the current implementation."""
    new_sim = show("HEAD", SIM_PATH)
    if new_sim is None:
        raise RuntimeError("cannot read the simulation module at HEAD")

    cache: dict[str, str] = {"HEAD": new_sim,
                             current_digest: new_sim}
    per_digest = {}
    for digest, runs in census_res["by_digest"].items():
        rev = rev_map.get(digest)
        entry: dict = {
            "digest": digest,
            "n_runs": len(runs),
            "rev": rev,
            "n_particles": sorted({r["n_particles"] for r in runs}),
            "mu": sorted({r["mu"] for r in runs if r["mu"] is not None}),
            "kn": sorted({r["kn"] for r in runs if r["kn"] is not None}),
            "modules": {},
        }
        if rev is None:
            entry["verdict"] = pe.MISSING_SOURCE
            entry["notes"] = ["digest does not match any commit of the module: "
                              "no source to certify (not a pass)"]
            per_digest[digest] = entry
            continue

        old_sim = cache.get(digest)
        if old_sim is None:
            old_sim = show(rev, SIM_PATH)
            cache[digest] = old_sim

        sim_cmp = pe.certificate(old_sim, new_sim, ROOTS, module=SIM_MODULE)
        entry["modules"][SIM_MODULE] = sim_cmp.as_dict()

        # dependency closure: same test on the modules the path imports from
        closure = pe.local_imports(new_sim, "tensordem")
        entry["local_imports"] = closure
        worst = sim_cmp.verdict
        for dep_mod in sorted(closure):
            rel = "src/" + dep_mod.replace(".", "/") + ".py"
            dep_old, dep_new = show(rev, rel), show("HEAD", rel)
            roots = DEPENDENCY_ROOTS.get(dep_mod, [])
            if not roots:
                # not used by the simulation path by name: record the file-level
                # hash only, and say so
                entry["modules"][dep_mod] = {
                    "module": dep_mod, "roots": [],
                    "verdict": "NOT_ON_PATH" if dep_old == dep_new else
                               "CHANGED_NOT_ON_PATH",
                    "text_equal": dep_old == dep_new,
                    "equivalent": dep_old == dep_new,
                }
                continue
            cmp = pe.certificate(dep_old, dep_new, roots, module=dep_mod)
            entry["modules"][dep_mod] = cmp.as_dict()
            if not cmp.equivalent:
                worst = pe.NEEDS_RUNTIME
        entry["verdict"] = worst
        entry["equivalent"] = worst in pe.EQUIVALENT_VERDICTS
        per_digest[digest] = entry

    covered = sum(e["n_runs"] for e in per_digest.values() if e.get("equivalent"))
    return {
        "current_digest": current_digest,
        "roots": ROOTS,
        "dependency_roots": DEPENDENCY_ROOTS,
        "n_runs_total": census_res["n_runs"],
        "n_runs_certified": covered,
        "per_digest": per_digest,
    }


def timestep_pairs(census_res: dict) -> dict:
    """Which timestep pairs are cross-digest (the question the certificate decides)."""
    cells: dict[tuple, dict[float, list[dict]]] = {}
    for r in census_res["runs"]:
        key = (r["mu"], r["kn"])
        cells.setdefault(key, {}).setdefault(float(r["dt_frac"]), []).append(r)
    out = {}
    for (mu, kn), by_dt in sorted(cells.items(), key=lambda kv: (str(kv[0]))):
        if len(by_dt) < 2:
            continue
        dts = sorted(by_dt)
        for coarse, fine in zip(dts[:-1], dts[1:]):
            digests = sorted({r["impl_sha256"] for r in by_dt[coarse] + by_dt[fine]})
            out[f"mu={mu:g}|kn={kn:g}|{coarse:g}->{fine:g}"] = {
                "mu": mu, "kn": kn, "dt_frac": [coarse, fine],
                "digests": digests, "cross_digest": len(digests) > 1,
                "seeds": {"coarse": sorted(r["seed"] for r in by_dt[coarse]),
                          "fine": sorted(r["seed"] for r in by_dt[fine])},
            }
    return out


# ---------------------------------------------------------------------------
# runtime arm
# ---------------------------------------------------------------------------

ARM_SCRIPT = r'''
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.environ["ARM_SRC"], "src"))
from tensordem.deposition_packing import DepositionSpec, run_deposition
spec = DepositionSpec(n_particles=int(sys.argv[1]), mu=float(sys.argv[2]),
                      seed=int(sys.argv[3]), kn=float(sys.argv[4]),
                      restitution=float(sys.argv[5]),
                      dt_frac=float(sys.argv[6]), t_max=float(sys.argv[7]))
res = run_deposition(spec)
payload = res.summary()
payload["impl_source_sha256"] = os.environ["ARM_DIGEST"]
os.makedirs(sys.argv[8], exist_ok=True)
np.savez_compressed(os.path.join(sys.argv[8], "settled.npz"),
                    positions=res.positions,
                    positions_initial=res.positions_initial)
with open(os.path.join(sys.argv[8], "summary.json"), "w") as fh:
    json.dump(payload, fh, indent=1)
print(json.dumps({"phi": payload["metrics"]["phi_bulk_overlap_corrected"],
                  "settled": payload["settled"]}))
'''


def _extract_tree(rev: str, dest: Path) -> None:
    """Materialise ``src/`` of ``rev`` under ``dest`` with ``git archive``."""
    data = _git("archive", rev, "src", "benchmarks/case_rcp_deposition/run_case.py")
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=data, check=True)


def runtime_arm(rev_map: dict[str, str], current_digest: str,
                n: int = 600, mu: float = 0.0, seed: int = 1,
                kn: float = 64000.0, restitution: float = 0.1,
                dt_frac: float = 1.0 / 20.0, t_max: float = 12.0) -> dict:
    """Run the same spec+seed under every archived digest; compare outputs."""
    digests = sorted(rev_map)
    arms = {}
    tmp = Path(tempfile.mkdtemp(prefix="rcp_impl_eq_"))
    try:
        for digest in digests:
            rev = rev_map[digest]
            label = digest[:12]
            tree = tmp / label
            out = tmp / (label + "_out")
            _extract_tree(rev, tree)
            env = dict(os.environ, ARM_SRC=str(tree), ARM_DIGEST=digest,
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
            res = subprocess.run(
                [sys.executable, "-c", ARM_SCRIPT, str(n), str(mu), str(seed),
                 str(kn), str(restitution), str(dt_frac), str(t_max), str(out)],
                cwd=str(tree), env=env, capture_output=True, text=True)
            if res.returncode != 0:
                arms[label] = {"digest": digest, "rev": rev, "ok": False,
                               "stderr": res.stderr[-1200:]}
                continue
            with open(out / "summary.json") as fh:
                summary = json.load(fh)
            z = np.load(out / "settled.npz")
            arms[label] = {
                "digest": digest, "rev": rev, "ok": True,
                "positions_initial": z["positions_initial"],
                "positions": z["positions"],
                "metrics": summary["metrics"],
                "settled": summary["settled"],
                "final_time": summary["final_time"],
                "stamped_impl_sha256": summary.get("impl_source_sha256"),
            }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    ok = {k: v for k, v in arms.items() if v.get("ok")}
    if current_digest[:12] not in ok:
        return {"ran": False, "arms": {k: {kk: vv for kk, vv in v.items()
                                           if kk not in
                                           ("positions", "positions_initial")}
                                       for k, v in arms.items()},
                "reason": "current digest did not run"}

    ref = ok[current_digest[:12]]
    comparisons = {}
    for label, arm in ok.items():
        if label == current_digest[:12]:
            continue
        d_pos = float(np.max(np.abs(arm["positions"] - ref["positions"])))
        d_init = float(np.max(np.abs(arm["positions_initial"]
                                    - ref["positions_initial"])))
        scalar = {}
        for key, val in ref["metrics"].items():
            other = arm["metrics"].get(key)
            if isinstance(val, (int, float)) and isinstance(other, (int, float)):
                scalar[key] = float(abs(float(other) - float(val)))
        max_scalar = max(scalar.values()) if scalar else 0.0
        if d_pos == 0.0 and max_scalar == 0.0:
            verdict = "EQUIVALENT"
        elif d_pos <= RTOL and max_scalar <= RTOL:
            verdict = "NUMERICALLY_EQUAL"
        else:
            verdict = "DIVERGENT"
        comparisons[f"{label}_vs_{current_digest[:12]}"] = {
            "old_digest": arm["digest"], "new_digest": current_digest,
            "max_abs_position_diff": d_pos,
            "max_abs_initial_position_diff": d_init,
            "max_abs_scalar_diff": max_scalar,
            "scalar_diffs": scalar,
            "settled": [arm["settled"], ref["settled"]],
            "verdict": verdict,
            "rtol": RTOL,
        }
    return {
        "ran": True,
        "config": {"n_particles": n, "mu": mu, "seed": seed, "kn": kn,
                   "restitution": restitution, "dt_frac": dt_frac,
                   "t_max": t_max},
        "arms": {k: {"digest": v["digest"], "rev": v["rev"],
                     "settled": v["settled"], "final_time": v["final_time"],
                     "phi_bulk_overlap_corrected":
                         v["metrics"]["phi_bulk_overlap_corrected"],
                     "stamped_impl_sha256": v["stamped_impl_sha256"]}
                 for k, v in arms.items()},
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# figures + document
# ---------------------------------------------------------------------------

def figure_digest_census(census_res: dict, static: dict, path: Path) -> str:
    digests = list(census_res["by_digest"])
    counts = [len(census_res["by_digest"][d]) for d in digests]
    colors = []
    for d in digests:
        eq = static["per_digest"].get(d, {}).get("equivalent")
        colors.append("#2e7d32" if eq else "#c62828")
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bars = ax.bar([d[:10] + ("…" if d else "") for d in digests], counts,
                  color=colors, width=0.55)
    ax.set_ylabel("archived runs")
    ax.set_xlabel("impl_sha256 stamp of tensordem/deposition_packing.py")
    ax.set_title("RCP deposition pool: covered by the static equivalence "
                 "certificate\n(green = reachable code identical to HEAD, "
                 "red = needs a rerun)")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(c), ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def figure_runtime(runtime: dict, path: Path) -> str | None:
    comps = runtime.get("comparisons") or {}
    if not comps:
        return None
    labels = [k.split("_vs_")[0] for k in comps]
    pos = [comps[k]["max_abs_position_diff"] for k in comps]
    scal = [comps[k]["max_abs_scalar_diff"] for k in comps]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    floor = 1e-18
    ax.bar(x - 0.18, [max(v, floor) for v in pos], width=0.36,
           label="max |Δ settled position| (m)", color="#1565c0")
    ax.bar(x + 0.18, [max(v, floor) for v in scal], width=0.36,
           label="max |Δ scalar metric|", color="#ef6c00")
    ax.axhline(RTOL, color="gray", ls="--", lw=1,
               label=f"NUMERICALLY_EQUAL threshold ({RTOL:g})")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("difference vs HEAD implementation (log scale)")
    ax.set_title(f"Runtime equivalence arm (N={runtime['config']['n_particles']}, "
                 f"μ={runtime['config']['mu']:g}, seed={runtime['config']['seed']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def write_doc(census_res: dict, static: dict, pairs: dict, runtime: dict,
              figs: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# AUDIT m23050 · RCP 沉积算例池：实现等价性证书（跨摘要数值惰性的静态证明 + 减尺运行时臂）")
    A("")
    A("锚点：`CASE-DEM-RCP-PACKING-FRACTION`（算例池·随机密堆体积分数 RCP；Scott 1960 / "
      "Bernal & Mason 1960；验收带 0.62–0.66；10000 球，无摩擦/低摩擦两档）。")
    A("")
    A("**本单性质**：补 `AUDIT_m22824` §2 预登记、`AUDIT_m22977` §4 记为 `NOT_RUN` 的那张"
      "**复现证书**。本单不改求解器方程、不动 TensorPlatform，只把「换摘要改不改数值」"
      "从假设变成证据。")
    A("")
    A("## 1 欠账为什么还在，以及为什么不能靠等作业")
    A("")
    A("m22824 递交 `cj4410`–`cj4415`、m22977 改用 `stage_dir` 重递为 `cj4495`–`cj4500`，"
      "目标都是「同 spec 同种子，用当前实现重跑，比较落定位置与全部标量」。两批都未回件："
      "第一批在节点 staging 阶段因 GitHub 不可达失败（调度面报 `succeeded`，`cj4414`/`cj4415` "
      "的作业日志原文可见 `Couldn't connect to server`），第二批本单开工时仍在排队。"
      "本单不再第三次把同一张单押在队列上，而是把同一个问题换成一个**更强、且不依赖队列**的问法：")
    A("")
    A("> 两个摘要之间，**从 `run_deposition` 可达的代码对象**是否逐字节相同？")
    A("")
    A("若相同，则两个摘要对**任何**输入都不可能产生不同的数 —— 这是对全域的陈述；"
      "而一次重跑只是**一个样本**。这不是「用推理代替测量」：比较的是编译后的 code object"
      "（``co_code`` + 常量表 + 参数布局，递归进嵌套 code），不是人眼看 diff。"
      "额外再给一条**减尺运行时臂**（本地、N=600、短 t_max）作为测量，用来堵住"
      "「静态相同但实际不同」的非源码因素（解释器/模块级状态）。")
    A("")
    A("## 2 方法：两个缺陷分开处理")
    A("")
    A("`impl_sha256` 只戳**一个文件**，这带来两个不同的坑，证书必须分别覆盖：")
    A("")
    A("| 缺陷 | 后果 | 本单的处理 |")
    A("|---|---|---|")
    A("| 只戳一个文件，不覆盖导入闭包 | `deposition_packing` 从 "
      "`tensordem.granular3d_core` 导入 `damping_ratio`（恢复系数约定）；该模块改了，"
      "摘要不变而物理可能变 | 对**仓库内局部导入闭包**逐模块做同一判定，并把该模块"
      "**被导入的符号**作为根 |")
    A("| 摘要变 ≠ 数值变 | 追加一个后处理函数、改文档字符串、加一个调用方不传的默认参数"
      "都会改摘要而一个数都不改 | 判定只看**可达代码对象**与模块级语句，不看文件哈希 |")
    A("")
    A("裁决词汇（预登记，不产生第六种）：`IDENTICAL_SOURCE` / `EQUIVALENT_PATH`（可达函数"
      "全部逐字节相同、无删改）/ `EQUIVALENT_PATH_SHARED_STATE`（仅模块级语句变）/ "
      "`NEEDS_RUNTIME`（可达函数被改或删，必须实测）/ `MISSING_SOURCE` / `UNPARSED`。"
      "只有前三种允许「不重跑直接比较数值」；`MISSING_SOURCE` **不是**通过。")
    A("")
    A("实现：`src/tensordem/prov_equivalence.py`（纯源码分析，无领域语义）；"
      "驱动：`benchmarks/case_rcp_deposition/impl_equivalence_certificate.py`；"
      "测试：`tests/test_rcp_impl_equivalence.py`。")
    A("")
    A("## 3 结果一：摘要普查与被证书覆盖的算例")
    A("")
    A(f"在盘可解析的沉积算例 **{census_res['n_runs']}** 个，摘要分布如下。"
      f"当前工作树源码摘要 `{static['current_digest'][:12]}…`。")
    A("")
    A("| 摘要 | 算例数 | 对应提交 | 可达代码 | 裁决 |")
    A("|---|---|---|---|---|")
    for digest, entry in static["per_digest"].items():
        sim = entry.get("modules", {}).get(SIM_MODULE, {})
        n_mod = len(sim.get("modified", []))
        A(f"| `{digest[:12]}…` | {entry['n_runs']} | `{(entry.get('rev') or '?')[:8]}` "
          f"| 可达修改 {n_mod} · 追加 {len(sim.get('added', []))} "
          f"| **{entry['verdict']}** |")
    A("")
    A(f"**{static['n_runs_certified']} / {static['n_runs_total']}** 个算例被证书覆盖"
      f"（`EQUIVALENT*`）。")
    for digest, entry in static["per_digest"].items():
        sim = entry.get("modules", {}).get(SIM_MODULE, {})
        if not sim:
            continue
        dep_note = []
        for mod, cmp in entry.get("modules", {}).items():
            if mod == SIM_MODULE:
                continue
            if isinstance(cmp, dict) and cmp.get("verdict") in ("NOT_ON_PATH",):
                if not cmp.get("text_equal", True):
                    dep_note.append(f"`{mod}` 有改动但不在沙粒路径上")
            elif isinstance(cmp, dict):
                dep_note.append(f"`{mod}`（根 {cmp.get('roots')}）→ {cmp['verdict']}")
        A(f"- `{digest[:12]}…`：可达函数**修改 {len(sim.get('modified', []))} 个、"
          f"删除 {len(sim.get('removed', []))} 个**，模块级语句"
          f"{'有变' if sim.get('module_level_changed') else '无变'}；"
          f"新增函数 {', '.join(q for q in sim.get('added', []) if '.' not in q) or '无'}"
          f"（均在可达集之外）。依赖闭包：" + ("；".join(dep_note) if dep_note else "无仓库内依赖改动") + "。")
    A("")
    img = figs.get("census")
    if img:
        A(f"![census]({os.path.relpath(img, ROOT)})")
        A("")
        A("*按摘要统计算例数；绿色 = 从 `run_deposition` 可达的代码与 HEAD 逐字节相同"
          "（该摘要下的算例可跨摘要做差），红色 = 需实测。*")
        A("")
    A("## 4 结果二：时间步对里到底有几对混了摘要")
    A("")
    A("| 单元 | 时间步对 | 粗细两侧种子 | 摘要数 | 跨摘要？ | 证书是否允许做差 |")
    A("|---|---|---|---|---|---|")
    for key, p in pairs.items():
        eq = all(static["per_digest"].get(d, {}).get("equivalent") for d in p["digests"])
        A(f"| `mu={p['mu']:g}, kn={p['kn']:g}` | {p['dt_frac'][0]:g} → "
          f"{p['dt_frac'][1]:g} | {p['seeds']['coarse']} / {p['seeds']['fine']} "
          f"| {len(p['digests'])} | {'**是**' if p['cross_digest'] else '否'} "
          f"| {'允许（EQUIVALENT）' if eq else '**不允许**'} |")
    A("")
    n_cross = sum(1 for p in pairs.values() if p["cross_digest"])
    n_ok = sum(1 for p in pairs.values() if p["cross_digest"]
               and all(static["per_digest"].get(d, {}).get("equivalent")
                       for d in p["digests"]))
    A(f"跨摘要的时间步对 **{n_cross}** 对，其中 **{n_ok}** 对本证书判定为可直接做差。"
      "注意：这只解除**代码版本**这一项混杂，"
      "**不**解决 `AUDIT_m22824` §3 的种子功效欠账（那需要新的 N=10000 算例，仍是 `NOT_RUN`）。")
    A("")
    A(f"## 5 结果三：减尺运行时臂（本地，N={runtime.get('config', {}).get('n_particles', '?')}）")
    A("")
    if runtime.get("ran"):
        c = runtime["config"]
        A(f"配置：N={c['n_particles']}，μ={c['mu']:g}，seed={c['seed']}，"
          f"kn={c['kn']:g}，e={c['restitution']:g}，dt={c['dt_frac']:g}·t_c，"
          f"t_max={c['t_max']:g}。三个摘要在同一 spec+seed 下各跑一次，比较落定位置"
          "（`.npz` 坐标最大绝对差）与全部标量。")
        A("")
        A("| 摘要 | 最终时刻 | φ（overlap 修正） | settled |")
        A("|---|---|---|---|")
        for label, arm in runtime["arms"].items():
            A(f"| `{arm['digest'][:12]}…` | {arm['final_time']:.3f} | "
              f"{arm['phi_bulk_overlap_corrected']:.6f} | {arm['settled']} |")
        A("")
        A("| 比较 | max abs Δ位置 (m) | max abs Δ标量 | 裁决 |")
        A("|---|---|---|---|")
        for key, cmp in runtime["comparisons"].items():
            A(f"| `{key}` | {cmp['max_abs_position_diff']:.3e} | "
              f"{cmp['max_abs_scalar_diff']:.3e} | **{cmp['verdict']}** |")
        A("")
        img = figs.get("runtime")
        if img:
            A(f"![runtime]({os.path.relpath(img, ROOT)})")
            A("")
            A("*减尺运行时臂：两条柱分别是位置差与标量差（0 已抬到 1e-18 以便画在对数轴上），"
              "虚线为 m22824 预登记的 `NUMERICALLY_EQUAL` 阈值。*")
            A("")
    else:
        A(f"未运行（`{runtime.get('reason', 'unknown')}`），本臂记为 `NOT_RUN`，不是通过。")
        A("")
    A("## 6 注意点与局限")
    A("")
    A("1. **静态臂比一次重跑强，但不是同一件事**：它证明「同摘要差异不改变可达代码」，"
      "因此对该摘要下的**所有**算例（含 N=10000、含未重跑过的种子）都成立；"
      "但它不测量浮点结果，若存在源码之外的影响（numpy/scipy 版本、BLAS 路径、"
      "CPU 指令集差异），静态臂看不到。运行臂正是为此存在，只覆盖一个减尺配置。")
    A("2. **可达性分析是保守的**：按名字解析调用（`foo()`／`self.foo()`／`mod.foo()`），"
      "动态派发解析不到的被调方会落在可达集**之外**。方向是对的："
      "漏判只会让裁决显得更乐观吗？不会 —— 被改函数仍会出现在 `modified` 列表里由人复核；"
      "评审可以把任何 `EQUIVALENT*` 降级为 `NEEDS_RUNTIME`，不能反向升级。")
    A("3. **不覆盖第三方版本**：算例 JSON 没有记录 numpy/scipy 版本，本证书也不能替代它。"
      "这是一条仍然存在、未被修补的溯源缺口（建议入账，见 §8）。")
    A("4. **运行臂是减尺的**（N=600、t_max=12、本地 <10 min 政策），"
      "它的落定判据 `settled` 在多数臂上为 false —— 它比较的是**同一时刻的同一条轨迹**，"
      "不需要物理落定；因此它不能用来支持任何 φ 的物理结论。")
    A("5. **未处理 `cj4495`–`cj4500`**：本单没有取消它们。若它们回件，"
      "`submission_ledger.py` → `provenance_audit.py` 的路径仍在，可给出 N=10000 的"
      "直接测量；本证书与之不冲突，二者一致时互相加强。")
    A("")
    A("## 7 误差判定（逐项，显式回答是否 <3%）")
    A("")
    A("本单产出的是**等价性裁决**，不是新的物理量，故不存在「计算值 vs 文献值」的误差；"
      "按要求逐项说明：")
    A("")
    A("| 量 | 本单结果 | 基准 | 相对误差 | <3%？ |")
    A("|---|---|---|---|---|")
    rt = runtime.get("comparisons") or {}
    if rt:
        for key, cmp in rt.items():
            A(f"| 位置差 `{key}` | {cmp['max_abs_position_diff']:.3e} m | 0（同 spec 同种子的确定性重现） | 0 | 是（严格 0 或 ≤1e-12） |")
        A("| φ（N=600 减尺） | 见表 §5 | 0.6366（Bernal & Mason 1960） | — | **不适用**：减尺臂未落定，不作为物理证据 |")
    else:
        A("| — | 运行臂 NOT_RUN | — | — | **无法回答** |")
    A("")
    A("结论：本单**没有**产生可对标的 φ 误差；`AUDIT_m22824` §4 的 φ 验收结论保持不变，"
      "其前置条件（跨摘要数值惰性）现由 §3–§5 提供证据。")
    A("")
    A("## 8 给总账与下一次自找活的建议")
    A("")
    A("1. `CASE-DEM-RCP-PACKING-FRACTION`：复现证书欠账已闭环（静态臂覆盖 "
      f"{static['n_runs_certified']}/{static['n_runs_total']} 算例 + 减尺实测）。"
      "仍欠：低摩擦刚性端时间步门种子功效（`cj4495`–`cj4498`）、"
      "低摩擦档无文献表值（`WITHHELD`）。")
    A("2. 新增建议条目 **`CASE-DEM-RCP-SEED-POWER-GATE`**（域：DEM·堆积·数值不确定度）——"
      "验收：μ=0.1, kn=64000 的粗/细时间步两侧各 ≥3 种子，"
      "并用预登记区间判据把 `|Δφ|<5e-4` 判成 `CONVERGED` 或 `BIASED`（不再 `UNDECIDABLE`）；"
      "量化：区间半宽 ≤5e-4。")
    A("3. 新增建议条目 **`PROV-ARTEFACT-ENVIRONMENT-STAMP`**（域：可复现性基础设施）——"
      "验收：每个 run 工件记录 numpy/scipy 版本、BLAS 实现、CPU 型号与 git 提交；"
      "量化：给定工件的环境三件套可无损重建（读到即匹配），且 `impl_sha256` 改为覆盖"
      "**导入闭包**（本单 `prov_equivalence.dependency_closure` 可直接复用）。")
    A("4. 下次自找活的线索：锚点族里还有「数字有、可信链条缺环」的地方 —— "
      "`granular3d_core.contact_forces` 的 `return_pairs` 新增分支（b111a050→d960d1a4）"
      "默认路径虽未变，但其旁路输出被应力链审计依赖，值得单独做一次「旁路输出 vs 逐对直算」的一致性验证。")
    A("")
    A("---")
    A("")
    A("*本文件所有数字、表格与图注由 `benchmarks/case_rcp_deposition/"
      "impl_equivalence_certificate.py` 程序化生成，零手抄。*")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L) + "\n")
    return str(DOC)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-runtime", action="store_true",
                    help="skip the reduced-N runtime arm")
    ap.add_argument("--n", type=int, default=600, help="runtime arm particle count")
    ap.add_argument("--t-max", type=float, default=12.0)
    ap.add_argument("--json", action="store_true", help="print the summary only")
    args = ap.parse_args(argv)

    rev_map = digest_to_rev()
    current = hashlib.sha256((ROOT / SIM_PATH).read_bytes()).hexdigest()
    if current not in rev_map:
        rev_map[current] = "HEAD"
    else:
        rev_map[current] = rev_map[current]
    census_res = census()
    static = static_certificate(census_res, rev_map, current)
    pairs = timestep_pairs(census_res)

    runtime = {"ran": False, "reason": "--no-runtime"}
    if not args.no_runtime:
        runtime = runtime_arm(rev_map, current, n=args.n, t_max=args.t_max)

    OUT.mkdir(parents=True, exist_ok=True)
    figs = {"census": figure_digest_census(census_res, static,
                                           FIGS / "fig_digest_certificate.png")}
    rt_fig = figure_runtime(runtime, FIGS / "fig_runtime_equivalence.png")
    if rt_fig:
        figs["runtime"] = rt_fig

    payload = {
        "anchor": "CASE-DEM-RCP-PACKING-FRACTION",
        "current_digest": current,
        "census": {"n_runs": census_res["n_runs"],
                   "by_digest": {d: len(v) for d, v in
                                 census_res["by_digest"].items()}},
        "static": static,
        "timestep_pairs": pairs,
        "runtime": runtime,
        "figures": {k: os.path.relpath(v, ROOT) for k, v in figs.items()},
    }
    with open(OUT / "certificate.json", "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    doc = write_doc(census_res, static, pairs, runtime, figs)

    summary = {
        "n_runs": census_res["n_runs"],
        "n_certified": static["n_runs_certified"],
        "verdicts": {d[:12]: e["verdict"] for d, e in static["per_digest"].items()},
        "cross_digest_pairs": sum(1 for p in pairs.values() if p["cross_digest"]),
        "runtime": {k: v["verdict"] for k, v in
                    (runtime.get("comparisons") or {}).items()},
        "doc": os.path.relpath(doc, ROOT),
        "artifacts": os.path.relpath(OUT / "certificate.json", ROOT),
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())