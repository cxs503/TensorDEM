"""Publish the m21886 report (disordered-route verdict re-adjudication)."""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUNDLE = os.path.join(ROOT, "docs", "artifacts",
                      "column_collapse_disordered_m20811")


def b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def load_mod():
    path = os.path.join(HERE, "analyze_column_collapse_disordered_m20811.py")
    spec = importlib.util.spec_from_file_location("m21886_a", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    mod = load_mod()
    summary = json.load(open(os.path.join(BUNDLE, "summary.json")))
    rows = mod.load_bundled_cases()
    old = json.loads(subprocess.check_output(
        ["git", "-C", ROOT, "show",
         "HEAD~1:docs/artifacts/column_collapse_disordered_m20811/summary.json"],
        text=True))
    st = summary["verdict_states"]
    cov = summary["evidence_coverage"]

    vrows = "\n".join(
        f"| `{k}` | {old['verdicts'].get(k)} | **{st[k]}** | "
        f"{summary['verdicts'][k]['reason'][:110]} |" for k in st)
    drows = "\n".join(
        f"| {r['case']} | {r['aspect_nominal']:g} | {r['aspect_measured']:.4f} | "
        f"{r['n_particles']} | {r['packing_fraction']:.4f} | "
        f"{r['max_overlap_frac']:.4f} | {r['runout_ratio']:.4f} | "
        f"{mod.lube_2004_runout_ratio(r['aspect_measured']):.4f} | "
        f"{100 * r['err']:+.2f} % | {'是' if r['admissible'] else '**否**'} |"
        for r in rows)
    band = "; ".join(
        f"$a$={r['aspect_nominal']:g}: {100 * r['err']:+.2f} %"
        f"{'（可容许）' if r['admissible'] else '（**不可容许**）'}" for r in rows)

    md = f"""# CASE-COLUMN-COLLAPSE：无序路线判据的三值重裁（m21886）

## ① 案例介绍
锚点 `CASE-COLUMN-COLLAPSE`（ref: Lube et al. 2004 *JFM* 508:175-199，容差 <15 %）。
圆柱颗粒柱 $h_i=a r_i$ 释放后坍塌，判据为 $\\rho_\\infty = r_\\infty/r_i$ 对
$1+1.24a\\ (a<1.7)$ / $1+1.6\\sqrt a\\ (a>1.7)$ 的相对误差 <15 %。
锚点的证据链用两条制备路线：**晶格控制族** 与 **无序族**（m20811，沉降弛豫后释放）。
本单审计后者——它在 m20811 里报出 `C_DISORDER=true`、`C_SPREAD=false`。

**参数字（全部来自入库 CSV，无一手抄）**：状态数 {cov['n_rows']}（成员 {cov['n_members']}），
可容许 {summary['n_admissible']}；每长宽比成员数 {cov['members_per_aspect']}；
$\\sigma/\\mu$ 可评估长宽比 {cov['spread_evaluable'] or '无'}；无序度可评估长宽比 {cov['disorder_evaluable'] or '无'}。

## ② 怎么做的
1. **判据三值化**：`pass/fail/unevaluable` + 理由 + 证据计数。原实现按长宽比做
   `all(gen)` 聚合，`gen` 为空时**空真** —— 这正是 `C_DISORDER=true` 的来源
   （两个长宽比的晶格控制六角度都是 `nan`）；`C_SPREAD` 把 `n=1` 时无定义的
   `sigma/mean` 报成 `fail`。
2. **只在入库证据上重算**：新增 `load_bundled_cases()`，用 `aspect_measured` 重算
   Lube 参考值与 `err`，不一致 >1e-9 直接 `ValueError` 拒绝；新增 `--source bundle`，
   让归约在无 `runs/`（不入库）的干净检出里可复现。
3. **补上预注册却被漏实现**的判据 `B_LATTICE`（本波内重跑晶格控制）。
4. **逐成员可容许性**：`C_EQ`（静止 + 残余重叠 ≤0.02 $r_p$）下放到成员级。
5. **修失效守门测试**：原 `test_bundled_csv_regression` 断言的是不入库的 `runs/` JSON
   （干净检出里必然失败），从未校验入库 CSV；改为校验 CSV 自洽 + `summary.json`
   与 CSV 的 sha256 绑定 + 空真/未定义反例。该文件 13 项（12 绿 +1 恒失败）→ 21 项全绿。

## ③ 计算结果

| case | $a_{{nom}}$ | $a_{{meas}}$ | N | $\\phi_0$ | 残余重叠/$r_p$ | $\\rho_\\infty$ | 参考 | 误差 | 可容许 |
|---|---|---|---|---|---|---|---|---|---|
{drows}

| 判据 | m20811 原报 | 本单 | 依据 |
|---|---|---|---|
{vrows}

**对锚点的直接结论**：`C_DISORDER=true` **撤回**（无同长宽比晶格控制 → `unevaluable`）；
`C_SPREAD=false` 改判 `unevaluable`（$n=1$，散布无定义）；
`C_EQ=fail` 是**真实失败**（$a$=1.0 的释放态残余重叠 0.0244 $r_p$ > 0.02 $r_p$），
该成员虽在 15 % 带内（−5.65 %）却是**不可容许态**，因此可容许成员上的带内命中只剩
$a$=0.5 一点（−12.09 %）。工件包里的数值一个未改。

## ④ 结果图
![verdict](fig1) 左：证据覆盖（每长宽比成员数 vs 散布门槛 2）；右：判据三值裁定。
![band](fig2) 逐状态对 Lube 律的带内余量。

## ⑤ 注意点
1. **空真是这类判据的头号坑**：按类别聚合时缺基线会静默变真；聚合前必须先数证据。
2. **本单没有重跑 DEM**：`unevaluable` 严格表示「现有工件包不支持」，
   不等于「物理上不成立」。补跑 1 个晶格控制 + 每长宽比 ≥2 成员即可把两个判据变可裁定。
3. **只能验证内部一致性**：CSV 的 `file` 列指向不入库的 `runs/...json`，
   无法证明这些数确由该配置算出（需入库每例 JSON 或配置哈希）。
4. **判据阈值一个没动**（0.52–0.68 / 0.02 $r_p$ / 5 % / 0.8 / 15 %），只改裁定语义。
5. 该算例池大量波次是「每长宽比 1 个确定性成员」，其上 `C_SPREAD` 永远 `unevaluable`；
   报 `fail` 比不报更误导。

## ⑥ 误差判定（显式回答「是否 <3 %」）
关键量 $\\rho_\\infty$ vs Lube 2004（按 $a_{{meas}}$ 求值）：{band}。
- 是否 <3 %：**否**（两个状态都在 −5.65 % 与 −12.09 %）。
- 是否 <15 %：两个状态都在带内，但 $a$=1.0 那个违反 `C_EQ`，
  故**可容许成员上的带内命中只有 1 例**。
- 本单不改变锚点的通过/不通过，只把「证据够不够判」说清楚：无序族在现有工件包上
  不足以支撑任何关于零散度或无序度的结论（honest unevaluable，不是 FAIL）。
"""
    payload = {
        "title": "CASE-COLUMN-COLLAPSE：无序路线判据的三值重裁（守门测试修复 + 空真撤回）",
        "summary": ("m20811 无序工件包的 C_DISORDER=true 撤回到 unevaluable（空集 all() 空真）、"
                    "C_SPREAD 由 fail 改判 unevaluable（n=1 散布无定义）、补实现漏掉的 B_LATTICE=fail；"
                    "新增仅凭入库 CSV 的归约与 sha256 绑定，修复必然失败的守门测试（21 项全绿）；"
                    "工件包数值未改，可容许成员上的 15 % 带内命中只剩 a=0.5 一点。"),
        "scenario_key": "CASE-COLUMN-COLLAPSE",
        "mission_id": 21886,
        "tags": ["CASE-COLUMN-COLLAPSE", "TensorDEM", "granular-column", "Lube2004",
                 "verification-debt", "adjudication-semantics", "m21886"],
        "markdown": md,
        "figures": [
            {"name": "fig_verdict_evidence.png",
             "caption": "左：入库工件包的证据覆盖（每长宽比成员数，虚线为散布可评估门槛 2）；"
                        "右：预注册判据的三值裁定（绿 pass / 红 fail / 灰 unevaluable）",
             "png_base64": b64(os.path.join(BUNDLE, "fig_verdict_evidence.png"))},
            {"name": "fig_err_band.png",
             "caption": "逐状态对 Lube (2004) 律（按 a_meas 求值）的带内余量",
             "png_base64": b64(os.path.join(BUNDLE, "fig_err_band.png"))},
        ],
    }
    with open("/tmp/report_m21886.json", "w") as fh:
        json.dump(payload, fh)

    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    if not base.endswith("/api/v1"):
        base = base.rstrip("/") + "/api/v1"
    login = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=login,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        tok = json.loads(resp.read()).get("access_token")
    req = urllib.request.Request(base + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        print(resp.read().decode()[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())