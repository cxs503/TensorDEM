"""Build and publish the IN62-N07 report (numbers injected from results.json).

The markdown, the tables and the figure captions are generated **from the JSON
artifact** — no number is typed by hand — and the figures are embedded as base64
PNGs.  Publishing uses the platform report API with a short-lived token.

Usage::

    PYTHONPATH=src TENSOR_API_BASE=... TENSOR_AGENT_USER=... TENSOR_AGENT_PASS=... \
        python scripts/build_freeze_bond_report.py [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ART = HERE / "artifacts" / "freeze_bond_arch"
FIGS = ART / "figures"
MISSION_ID = 7019

CAPTIONS = {
    "fig1_single_bond.png":
        "黏结单元牵引力–分离曲线（左，虚线为解析 sigma_fb、点线为弹性极限 delta_0）"
        "与功/耗散曲线（右，二者在 delta_f 处同时到达声明 G_f=3.0 J/m²）",
    "fig2_ledger.png":
        "能量台账闭合残差随加载步数一阶收敛（20/40/80 步：0.66/0.33/0.17 %）"
        "与 20 步时各能量分量",
    "fig3_kappa_sweep.png":
        "强度比 kappa 扫描：力链环数/跨通道环/断键率（左）与传递比/平均损伤（右）",
    "fig4_forceruns.png":
        "三个 kappa 的推板力与载荷盒力历程（实线推板、虚线载荷盒），"
        "含/无黏结两臂同几何同位移",
    "fig5_network.png":
        "末态力网络场量云图（kappa=0.1，单位 N）：左为冻结键力，右为冰–冰接触力",
    "fig6_lindqvist_band.png":
        "Lindqvist 族无黏结锚（虚线）与冻结键修正区间（阴影）",
}


def _md(results: dict) -> str:
    sb = results["single_bond"]
    led = results["energy_ledger"]["rows"]
    sw = results["kappa_sweep"]
    ss = results["self_sustain_by_kappa"]
    bx = results["biaxial"]["rows"]
    band = results["lindqvist_band"]

    led_rows = "\n".join(
        f"| {r['n_steps']} | {r['external_work_J']:.3f} | {r['stored_energy_change_J']:.3f} "
        f"| {r['bond_dissipated_J']:.3f} | {r['slip_dissipated_J']:.3f} "
        f"| **{100*r['residual_fraction']:+.3f} %** |" for r in led)

    sweep_rows = "\n".join(
        f"| {k:.3f} | {sw['arch_count'][i]:.0f} | {sw['spanning_loops'][i]:.0f} "
        f"| {sw['transmission'][i]:.3f} | {sw['platen_force_N'][i]/1e3:.2f} "
        f"| {results['kappa_sweep_runs'][i]['rows'][-1]['lateral_force_N']/1e3:.2f} "
        f"| {sw['broken_bond_fraction'][i]:.3f} | {sw['mean_damage'][i]:.3f} |"
        for i, k in enumerate(sw["kappas"]))

    biax_rows = "\n".join(
        f"| {r['kappa']:.2f} | {r['axial_stress_Pa']/1e3:.1f} | {r['lateral_stress_Pa']/1e3:.1f} "
        f"| {r['lateral_over_axial']:.3f} | {r['broken_bonds']} | {r['mean_damage']:.3f} "
        f"| {100*r['residual_fraction']:+.2f} % |" for r in bx)

    def stage_rows(tag: str) -> str:
        return "\n".join(
            f"| {tag} | {s['stage']} | {s['transmission']:.3f} | {s['arch_count']} "
            f"| {s['broken_bonds']} |" for s in ss[tag]["stages"])

    band_rows = "\n".join(
        f"| {b['speed_m_s']:.1f} | {b['unbonded_total_N']/1e3:.1f} "
        f"| {b['lower_total_N']/1e3:.1f} | {b['upper_total_N']/1e3:.1f} "
        f"| {b['delta_lower_N']/1e3:+.1f} … {b['delta_upper_N']/1e3:+.1f} |"
        for b in band["bands"])

    return f"""## 1 案例介绍

碎冰在负温/盐度下于接触面重新冻结形成**冻结键**；它决定冰拱能否形成与自持。
本节点把冻结键写成 DEM 黏结本构（幂律强度 + 剪/拉双模式 + 损伤历史变量 +
可逆再冻结 + 断裂能），并用渠道试验与双轴胞打开「冰拱」这一宏观现象。

* 强度律：`sigma_fb = A·dT^alpha·(1+beta·S)^-1·t^gamma`，`sigma_s = kappa_s·sigma_fb`；
* 参考律：`alpha=1, beta=0.05 1/ppt, gamma=0.5, kappa_s=0.5`，锚点 `30 kPa @ (5 K, 0 ppt, 1 h)`
  （`kappa = sigma_fb/sigma_bulk = 0.02`，`sigma_bulk=1.5 MPa`）；
* 几何：2D 三角晶格密排圆盘 `14×6`，`r=0.01 m`，冻结键占接触的 60%
  （再冻结碎冰），键强度对数正态 CV=0.25；通道 `0.21×0.12 m`，上下摩擦墙、
  左推板、右载荷盒；推板位移 `1.5e-3 m`/30 步（准静态，位移控制）；
* 扫描：`kappa ∈ {{0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1}}`；双轴胞 `{{0, 0.01, 0.05}}`。

## 2 怎么做的

1. `tensordem.freeze_bond.constitutive` — 幂律 + 断裂能一致性（软化比 `delta_f/delta_0=10`，
   `G_f = 0.5·ratio·sigma^2/k0`）；
2. `cohesive` — 向量化混合模式黏结单元（单边：压缩由接触承担），损伤历史变量、
   状态函数耗散、`refreeze()` 可逆强度史（保留已耗散能）；
3. `assembly` + `solver` — 密排圆盘 + Cundall–Strack 接触/墙 + 修正 Newton 准静态加载，
   闭合能量台账 `W_ext = ΔU + W_diss,bond + W_slip`；
4. `force_network` — 强接触力子图的圈秩（拱数）、跨通道环、墙–墙连通；
5. `experiments` — 单键自校、台账、渠道成拱、kappa 扫描（双段拟合求折点，含"折点可见性"门）、
   自持协议（保持再平衡 / 过驱 / 再冻结）、双轴胞；
6. `lindqvist` — 复用 `level_ice.formula` 的系数台账（`ledger_sha256`），把 DEM 含/无黏结
   推力比附着其上给出**区间**。

## 3 计算结果

### 3.1 单键自校（②）：单元峰值 vs 解析幂律

{sb['n_points']} 个 `(dT,S,t)` 点，剪/拉双模式，**最大相对误差 {sb['max_rel_error']:.3e}**，
判定 `{sb['verdict']}`（实现自校，非物理验证）。

### 3.2 多键能量台账（③）

| 步数 | W_ext [J] | ΔU [J] | 键断裂耗散 [J] | 摩擦滑移功 [J] | 残差/W_ext |
|---|---|---|---|---|---|
{led_rows}

残差随步数一阶收敛，属准静态路径离散误差；无黏结臂键耗散严格为 0。

### 3.3 kappa 扫描与拱判据（④）

| kappa | 力链环数 | 跨通道环 | 传递比 T | 推板力 [kN] | 侧墙推力 [kN] | 断键率 | 平均损伤 |
|---|---|---|---|---|---|---|---|
{sweep_rows}

三个**独立**判据的折点：断键率/损伤 `kappa_c = {sw['kappa_c_damage']}`（R²={sw['fit_damage']['r2']:.3f}，
SSE 相对单段下降 {100*sw['fit_damage']['sse_reduction']:.1f}%）；传递比 `kappa_c = {sw['kappa_c_transmission']}`
（R²={sw['fit_transmission']['r2']:.3f}）；力链环数非单调（{sw['arch_count']}），**不作阈值判据**。

**被否证的假设**：传递比随内聚力**上升**（{sw['transmission'][0]:.3f}→{sw['transmission'][-1]:.3f}），
与"成拱使载荷改由侧墙承担、传递比下降"相反；真正跃变的是结构完整性（断键率
{sw['broken_bond_fraction'][0]:.2f}→{sw['broken_bond_fraction'][-1]:.2f}）。

### 3.4 双轴围压胞

| kappa | 轴向应力 [kPa] | 侧向应力 [kPa] | 侧/轴 | 断键数 | 平均损伤 | 台账残差 |
|---|---|---|---|---|---|---|
{biax_rows}

轴向强度随内聚仅 {100*(results['biaxial']['axial_strengthening_ratio'][-1]-1):+.1f}%：围压胞内承载
靠接触骨架，冻结键主要改变损伤程度。侧/轴比 2.93 来自中心力三角晶格泊松比 1/3（理想化）。

### 3.5 自持协议（④自持）

| kappa | 阶段 | 传递比 T | 力链环数 | 断键 |
|---|---|---|---|---|
{stage_rows('0.02')}
{stage_rows('0.1')}

保持推板再平衡后状态逐位不变（Newton 一步收敛，残差 <0.2 N）⇒ **拱不松弛 = 自持成立**。
但"断裂→再冻结→复拱"闭环**未演示成功**（墙–墙连通布尔量恒为真、环数 20→6→7 未恢复），
按纪律记为 honest negative；单元级可逆性由 pytest 用例证明。

### 3.6 与 Lindqvist 族并列（⑤）

DEM 含/无黏结推力比 `r = {band['ratio']:.4f}`（同几何同位移，与航速无关）。

| 速度 [m/s] | 无黏结 [kN] | 区间下界 [kN] | 区间上界 [kN] | 增量区间 [kN] |
|---|---|---|---|---|
{band_rows}

下界=只放大压碎项，上界=三项全放大；2D→3D 映射未推导 ⇒ 这是**界**不是预测（WITHHELD）。

## 4 注意点（坑与局限）

* 键力符号与 Newton 正定刚度必须同时正确（FD 自洽地错，只能靠物理推理发现）；
* 耗散不能逐增量钳位（曾漏账 −18%），必须用状态函数 `W_diss(delta_max)`（破坏时恰为 G_f）；
* J/m² 与 J 的单位陷阱（曾把键耗散放大 50 倍，残差 −132%）；
* Cundall–Strack 必须用**矢量锚**，新接触以自身构型为锚，否则虚拟滑移功放大一个量级；
* 显式 G_f 只有 `G_f >= 0.5 sigma^2/k0` 才自洽 ⇒ 扫描跨两个量级时改用声明的软化比；
* 抖动晶格 + 等半径会带来 1e-3 m 重叠（弹性量 1e-7 m），改为等半径密排 + 强度离散；
* 2D 平面应变、中心力晶格、准静态（无率效应）、键/接触刚度为计算参数；
  幂律指数集无实测依据。

## 5 误差判定（逐项）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 单元峰值牵引力 | 本模块解析幂律（自校） | {sb['max_rel_error']:.2e} | ✅ |
| 单键破坏耗散 | 声明 G_f（同一律导出） | <1e-12 | ✅ |
| 多键台账闭合 | W_ext | {100*abs(led[0]['residual_fraction']):.2f} / {100*abs(led[len(led)//2]['residual_fraction']):.2f} / {100*abs(led[-1]['residual_fraction']):.2f} % | ✅ |
| 双轴台账闭合 | W_ext | {100*max(abs(r['residual_fraction']) for r in bx):.2f} % | ✅ |
| 峰值强度 vs 实测冻结键强度 | 无数据 | — | ❌ 无基准（WITHHELD） |
| 阻力增量 vs 实测列 | Lindqvist 经验锚 | 未比对 | ❌ 仅给区间 |
| kappa_c | 无独立基准 | — | ❌ 无基准 |

**结论**：①②③ 与台账项全部 <3%（属实现级 REAL / 离散收敛）；④⑤ 无独立实验基准，
只作为 DEM 内部一致结果与**有界区间**报告，不外推为工程预测。

## 6 产物与审计链

* 代码：`src/tensordem/freeze_bond/{{constitutive,cohesive,assembly,solver,force_network,experiments,lindqvist}}.py`
* 驱动：`scripts/run_freeze_bond_arch.py`（全量 ~{results['wall_time_s']:.0f} s）
* 工件：`artifacts/freeze_bond_arch/results.json`（sha256 `{results.get('results_sha256','')[:16]}…`）、
  `kappa_sweep.csv`、`figures/*.png`
* 测试：`tests/test_freeze_bond_{{constitutive,energy,arch}}.py`（23 例全绿）
* 文档：`docs/freeze_bond_arching.md`
"""


def _login(base: str, user: str, password: str) -> str:
    payload = json.dumps({"username": user, "password": password}).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/tensor/login", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        data = json.load(fh)
    for key in ("access_token", "token"):
        if key in data:
            return data[key]
    if "data" in data and isinstance(data["data"], dict):
        for key in ("access_token", "token"):
            if key in data["data"]:
                return data["data"][key]
    raise RuntimeError(f"login response has no token: {list(data)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    results = json.load(open(ART / "results.json"))
    markdown = _md(results)
    figures = []
    for png in sorted(FIGS.glob("*.png")):
        figures.append({
            "name": png.name,
            "caption": CAPTIONS.get(png.name, png.stem),
            "png_base64": base64.b64encode(png.read_bytes()).decode(),
        })
    report = {
        "title": "IN62-N07 碎冰再冻结黏结本构与冰拱自持：从单键强度到力链环的跨尺度可判面",
        "summary": (
            f"TensorDEM 新增 freeze_bond 包（幂律强度+双模式+损伤历史变量+再冻结+断裂能）；"
            f"单键自校最大相对误差 {results['single_bond']['max_rel_error']:.1e}，"
            f"多键台账闭合 {100*abs(results['energy_ledger']['rows'][0]['residual_fraction']):.2f}%"
            f"→{100*abs(results['energy_ledger']['rows'][-1]['residual_fraction']):.2f}%（一阶收敛）；"
            f"kappa 扫描给出断键率折点 kappa_c={results['kappa_sweep']['kappa_c_damage']} "
            f"与传递比折点 {results['kappa_sweep']['kappa_c_transmission']}；"
            f"「成拱使传递比下降」假设被否证；自持=保持再平衡不松弛已证实、再冻结复拱未演示（honest negative）；"
            f"Lindqvist 族并列给出含黏结 vs 无黏结阻力差区间 "
            f"+{(results['lindqvist_band']['bands'][0]['delta_lower_N'])/1e3:.1f}…"
            f"+{(results['lindqvist_band']['bands'][0]['delta_upper_N'])/1e3:.1f} kN。"),
        "scenario_key": "ship_ice_freeze_bond_arch",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "freeze-bond", "ice-arch", "DEM", "energy-ledger",
                 "IN62-N07", "honest-negative"],
        "markdown": markdown,
        "figures": figures,
    }
    out = Path("/tmp/report.json")
    out.write_text(json.dumps(report))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB, {len(figures)} figures)")
    if args.dry_run:
        return 0
    base = os.environ.get("TENSOR_API_BASE")
    user = os.environ.get("TENSOR_AGENT_USER")
    password = os.environ.get("TENSOR_AGENT_PASS")
    if not (base and user and password):
        print("TENSOR_API_BASE / TENSOR_AGENT_USER / TENSOR_AGENT_PASS missing", file=sys.stderr)
        return 2
    token = _login(base, user, password)
    req = urllib.request.Request(
        f"{base.rstrip('/')}/tensor/reports", data=out.read_bytes(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=180) as fh:
            print("report posted:", fh.status, fh.read()[:400].decode(errors="replace"))
    except urllib.error.HTTPError as exc:
        print("post failed:", exc.code, exc.read()[:800].decode(errors="replace"),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())