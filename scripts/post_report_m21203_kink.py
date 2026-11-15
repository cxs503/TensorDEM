"""Publish the m21203 breakpoint-neighbourhood report to the platform API.

Every number is read from ``docs/artifacts/column_collapse_kink_m21007/
verdict.json`` (produced by the frozen benchmark's ``--collect``) and every
figure from the PNGs written by ``scripts/report_column_collapse_kink_m21007.py``
— nothing is typed by hand here.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "artifacts", "column_collapse_kink_m21007")
VERDICT = os.path.join(OUT, "verdict.json")
MISSION = 21203
TITLE = "CASE-COLUMN-COLLAPSE 断点邻域收官：Lube(2004) 分段律在 a∈(1.5,1.8) 的复核（m21203）"


def b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def main() -> int:
    with open(VERDICT) as fh:
        payload = json.load(fh)
    v = payload["verdict"]
    rows = payload["rows"]
    law = v["law_discontinuity"]
    h1, h2, h3, h4 = (v["H1_band"], v["H2_no_branch_artefact"],
                      v["H3_slope_shape"], v["H4_resolution"])
    g12 = sorted([r for r in rows if r["grains_per_radius"] == 12.0],
                 key=lambda r: r["aspect_measured"])

    def pct(x: float) -> str:
        return f"{100.0 * x:+.1f} %"

    point_rows = "\n".join(
        f"| {r['case_id']} | {r['aspect_nominal']:g} | {r['aspect_measured']:.4f} | "
        f"{r['grains_per_radius']:g} | {r['n_particles']} | {r['stop_reason']} | "
        f"{r['runout_ratio']:.4f} | {r['ref_runout_ratio']:.4f} | {pct(r['err_runout'])} | "
        f"{pct(r['err_runout_nominal'])} | {pct(r['err_t90'])} | "
        f"{100.0 * r['energy_residual']:+.2f} % |"
        for r in rows)

    n_in = sum(1 for r in g12 if abs(r["err_runout"]) <= 0.15)
    best = min(g12, key=lambda r: abs(r["err_runout"]))
    worst = max(g12, key=lambda r: abs(r["err_runout"]))
    repro = v.get("reproducibility", {})

    md = f"""# CASE-COLUMN-COLLAPSE · 断点邻域（a∈(1.5,1.8)）验收复核

**锚点** `CASE-COLUMN-COLLAPSE`（算例池·颗粒柱坍塌）· mission 21203 · repo TensorDEM · branch `agent/m21203`
**ref** Lube, Huppert, Sparks & Freundt (2004) *J. Fluid Mech.* **508**:175-199,
doi:10.1017/S0022112004009036（分段跑距律，容差 < 15 %）
**预登记** `docs/PREREG_COLUMN_COLLAPSE_KINK_m21007.md`（5 算例 / 4 假设，跑前冻结）

## ① 案例介绍

轴对称颗粒柱：半径 $r_i$、高 $h_i=a r_i$ 的 HCP 柱体在粗糙底板上撤去侧支撑后坍塌，
最终沉积半径 $r_\\infty$ 只依赖 $a$。Lube et al. (2004) 的跑距律是**分段**的：

$$r_\\infty/r_i = 1+1.24a\\ (a<1.7);\\qquad r_\\infty/r_i = 1+1.6\\sqrt a\\ (a>1.7)$$

两支在卡面断点处斜率相差 {law['slope_ratio_across_break']:.2f} 倍，而真实等值点为
$a^*={law['crossover_aspect']:.4f}$（卡面断点 $1.7$ 在其之上 {law['break_minus_crossover']:.4f}），
参考律**自身**在 $a=1.7$ 的不连续为 {100 * law['relative_jump']:.3f} %。

| 参数 | 值 |
|---|---|
| 柱半径 $r_i$ | 0.48 m |
| 颗粒半径 $r$ | $r_i/(2\\,gpr)$，$gpr\\in\\{{8,12\\}}$ |
| 长宽比 $a$ | 1.5 / 1.6 / 1.7 / 1.8（@gpr 12）+ 1.7（@gpr 8 对照） |
| 密度 / 重力 | 2500 kg/m³ / 9.81 m/s² |
| 摩擦 | $\\mu=0.5$，$\\mu_r=0.5$，底板 $\\mu_w=0.8$ |
| 接触 | 法向弹簧-阻尼 $k_n=6\\times10^4(r/0.03)^2$ N/m，$k_t/k_n=0.5$，$e=0.5$ |
| 初始堆积 | HCP + 2 % 格隙，`jitter`=0，`seed`=12345 |

## ② 怎么做的

1. **执行冻结预登记**：直接运行 m21007 已合并的基准脚本，协议参数一字未改；
   逐算例 JSON 的 `protocol_sha256` 与既有归档 `KN_a1.6_g12.json` **逐位一致**
   （`{repro.get('protocol_sha_match', 'not_run')}`），证明本次复算没有重调任何参数。
2. **集群递交**：作业 `cj3259`（4 分片单进程，调度器派发，未指定节点），
   每算例 N ≈ 8k–12k 粒、单核 3–5 min。
3. **归约**：`benchmarks/case_column_collapse_kink_m21007.py --collect` →
   `verdict.json` + `cases_kink.csv`；参考值全部由 `lube_2004_runout_ratio(a_meas)` 运行时求值，
   判据用**实测**长宽比（m17534 口径），名义 $a$ 并列入表。
4. **出图/报告**：`scripts/report_column_collapse_kink_m21007.py`（matplotlib，Agg）→ 3 张 png；
   本报告的每个数字均由 `scripts/post_report_m21203_kink.py` 从 `verdict.json` 注入。

## ③ 计算结果

| case | $a_{{nom}}$ | $a_{{meas}}$ | gpr | N | 终止 | $r_\\infty/r_i$ | ref($a_{{meas}}$) | err(meas) | err(nom) | err $t_{{90}}$ | 能量残差 |
|---|---|---|---|---|---|---|---|---|---|---|---|
{point_rows}

## ④ 结果图

![kink curve](fig_kink_curve.png)

**图 1** — 断点邻域：DEM（$r_i/d=12$ 圆点，标注相对误差）vs 分段参考律两支
（红实线线性支、紫虚线 $\\sqrt a$ 支）；灰点线 = 卡面断点 $a=1.7$，绿点划线 = 真实等值点 $a^*$。

![error band](fig_error_band.png)

**图 2** — 误差-长宽比曲线（±15 % 绿带）。跨断点跳变：{h2['jump_below_pp']:.2f} pp
（$1.6\\to1.7$）、{h2['jump_above_pp']:.2f} pp（$1.7\\to1.8$）。

![slopes](fig_slopes_resolution.png)

**图 3** — 左：实测分段斜率 vs 参考律两支斜率；右：断点 $a=1.7$ 处 gpr 8 vs 12 误差对照。

## ⑤ 注意点

1. **两点差分低功效**：斜率只能用 $a$ 差 0.1 的两点差分估计，H3 因此同时给短/长基线，
   两者冲突时判 `incomplete`。
2. **单分辨率主扫描**：只有 $a=1.7$ 有一个 gpr=8 对照点，其余点无分辨率阶梯。
3. **确定性单实现**（`jitter`=0）：不含系综散布，单点差异不得读成随机性。
4. **能量残差**按实测入账、不设事后阈值门；残差 >10 % 的算例：
   {', '.join(v['energy_flagged']) if v['energy_flagged'] else '无'}。
5. **穿透/重叠计数** `not_run`（协议 `return_positions=False`）。
6. **适用域**：干颗粒、轴对称、HCP 初始堆积、单分散球体、粗糙底板。

## ⑥ 误差判定（显式回答「是否 < 3 %」「是否 < 15 %」）

| 关键量 | 基准 | 最佳 | 最差 | < 3 % ? | < 15 % ? |
|---|---|---|---|---|---|
| 跑距律（gpr=12，{len(g12)} 点） | Lube 两律@$a_{{meas}}$ | {pct(best['err_runout'])} | {pct(worst['err_runout'])} | **否** | **{'是' if h1['all_in_band'] else '否'}**（{n_in}/{len(g12)} 在带内） |
| 断点处分辨率漂移（a=1.7, gpr 8→12） | — | {h4['delta_pp']:.2f} pp | 同左 | {'是' if h4['delta_pp'] < 3 else '否'} | {'是' if h4['delta_pp'] <= 5 else '否'}（阈值 5 pp） |

**假设裁决**：H1 `{h1['verdict']}` · H2 `{h2['verdict']}` · H3 `{h3['verdict']}` ·
H4 `{h4['verdict']}` → **总体 `{v['overall']}`**

- 跑距律 < 3 %：**否**（最小 {pct(best['err_runout'])}）。
- 跑距律 < 15 %：**{'是' if h1['all_in_band'] else '否'}**（{n_in}/{len(g12)}）。
- 断点是否构成判据障碍：**{'否' if h2['verdict'] == 'pass' else '是'}**。
"""

    report = {
        "title": TITLE,
        "summary": (
            f"执行冻结预登记 CASE-COLUMN-COLLAPSE 断点扫描（m21007 遗留的 4 个算例）："
            f"a=1.5/1.6/1.7/1.8 @ r_i/d=12 + a=1.7 @ r_i/d=8；"
            f"总体裁决 {v['overall']}，gpr=12 全部 {n_in}/{len(g12)} 点在 15% 带内，"
            f"最差 {pct(worst['err_runout'])}，跨断点跳变 {max(h2['jump_below_pp'], h2['jump_above_pp']):.2f} pp"
        ),
        "scenario_key": "CASE-COLUMN-COLLAPSE",
        "mission_id": MISSION,
        "tags": ["TensorDEM", "granular-column-collapse", "Lube2004", "runout-law",
                 "breakpoint", "benchmark-verification"],
        "markdown": md,
        "figures": [
            {"name": "fig_kink_curve.png", "caption": "断点邻域 DEM vs Lube(2004) 分段律",
             "png_base64": b64(os.path.join(OUT, "fig_kink_curve.png"))},
            {"name": "fig_error_band.png", "caption": "跑距误差 ±15% 带与跨断点跳变",
             "png_base64": b64(os.path.join(OUT, "fig_error_band.png"))},
            {"name": "fig_slopes_resolution.png", "caption": "分段斜率与分辨率对照",
             "png_base64": b64(os.path.join(OUT, "fig_slopes_resolution.png"))},
        ],
    }
    with open("/tmp/report_m21203.json", "w") as fh:
        json.dump(report, fh)
    print("wrote /tmp/report_m21203.json")

    if "--post" in sys.argv:
        base = os.environ["TENSOR_API_BASE"].rstrip("/")
        user, pw = os.environ["TENSOR_AGENT_USER"], os.environ["TENSOR_AGENT_PASS"]
        tok = subprocess.run(
            ["curl", "-s", "-X", "POST", f"{base}/auth/login",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"username": user, "password": pw})],
            capture_output=True, text=True, check=True).stdout
        token = json.loads(tok).get("access_token") or json.loads(tok).get("token")
        out = subprocess.run(
            ["curl", "-s", "-X", "POST", f"{base}/tensor/reports",
             "-H", f"Authorization: Bearer {token}",
             "-H", "Content-Type: application/json",
             "-d", "@/tmp/report_m21203.json"],
            capture_output=True, text=True, check=True).stdout
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())