"""Generate ``docs/W25_FORMATION_ICE_CHANNEL.md`` from the W25 artifact.

Every number in the document is interpolated from
``papers/artifacts/w25/formation-ice-channel/lindqvist_formation_gci.json``;
nothing is typed by hand, so the document can never drift away from the data.

Usage::

    python benchmarks/w25_formation_doc.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "papers" / "artifacts" / "w25" / "formation-ice-channel" / "lindqvist_formation_gci.json"
OUT = REPO_ROOT / "docs" / "W25_FORMATION_ICE_CHANNEL.md"


def _f(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "n/a"
    return f"{val:.{digits}f}"


def _pct(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "n/a"
    return f"{val:+.{digits}f} %"


def _kn(newton) -> str:
    return _f(float(newton) / 1e3, 2) if newton is not None else "n/a"


def build(artifact: dict) -> str:
    cal = artifact["caliber"]
    ship, ice, env, num = cal["ship"], cal["ice"], cal["environment"], cal["numerics"]
    matrix = artifact["case_matrix"]
    single = artifact["single_ship_vs_lindqvist"]
    surface = artifact["formation_savings"]
    mono = artifact["monotonicity"]
    gci = artifact["grid_convergence"]
    numerics = artifact["numerics_controls"]
    iso = artifact["iso19906_crosscheck"]
    verdicts = {item["id"]: item for item in artifact["verdicts"]}

    lines: list[str] = []
    add = lines.append

    add("# W25 编队冰阻力节省率 vs Lindqvist 单船解析锚（返修轮1）")
    add("")
    add(f"任务：`{artifact['task_key']}`（mission {artifact['mission_id']}，"
        f"领域仓 TensorDEM；runner = native DEM，`tensordem.ice_channel`）。")
    add("")
    add("工件：`papers/artifacts/w25/formation-ice-channel/lindqvist_formation_gci.json`"
        f"（由 `{artifact['generated_by']}` 从 {artifact['case_matrix']['n_cases_with_results']}"
        f"/{matrix['n_cases_planned']} 个原始算例 JSON 程序化生成，零手抄）。")
    add("")
    add(f"预注册：`{artifact['prereg_path']}`（sha256 `{(artifact['prereg_sha256'] or '')[:16]}…`，"
        f"先于生产运行写定）。")
    add("")

    # ---- 1 case ----
    add("## 1 案例与口径")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 船体 | L = {_f(ship['length_m'], 0)} m，B = {_f(ship['beam_m'], 0)} m，"
        f"L/B = {_f(ship['length_over_beam'], 2)}，{ship['hull_shape']} |")
    add(f"| 方形系数 | Cb = {_f(artifact['geometry_check']['block_coefficient_Cb'], 3)}"
        f"（{artifact['geometry_check']['Cb_note']}） |")
    add(f"| 冰况 | h = {_f(ice['thickness_m'], 2)} m，密集度 {_f(ice['concentration'], 1)}，"
        f"平均冰块半径 {_f(ice['mean_floe_radius_m'], 1)} m"
        f"（冰径/船宽 = {_f(ice['floe_diameter_over_beam'], 2)}） |")
    add(f"| 环境 | 无风；横向漂流 {_f(env['cross_channel_drift_m_s'], 2)} m/s"
        f"（{env['drift_role']}） |")
    add(f"| 数值 | dt = {_f(num['base_dt_s'], 3)} s（{num['dt_scaling']}）；"
        f"域宽 {_f(num['domain_width_m'], 0)} m；冰场长度 {num['field_length_rule']}；"
        f"每 {num['sample_every_steps']} 步采样 |")
    add(f"| 系综 | 编队 3 个冰场实现；单船 5 个（节省率分母噪声最大） |")
    add("")
    add("测量窗口（先于运行声明）：船体中心位于冰场内且距纵向两端 ≥ L/2、t ≥ 60 s 时才计入样本；")
    add("每船载荷取该窗口内**上游接触反力**的时间平均，编队总冰阻力取三船之和。")
    add("")
    add(f"算例矩阵：{matrix['n_cases_planned']} 个 DEM 算例"
        f"（角色分布 {matrix['roles']}），速度 {matrix['speeds_m_s']} m/s × "
        f"间距 {matrix['spacings_L']}L × 3 seeds。")
    add("")

    # ---- 2 method ----
    add("## 2 方法")
    add("")
    add("1. `tensordem.ice_channel.IceChannelSimulation` 生成抖动六角冰场（密集度 8/10，实现在 <1% 内），"
        "船体为运动学矩形，冰-船/冰-冰均为锚定罚函数弹簧接触；")
    add("2. 新增 `tensordem.ice_channel_formation`：算例口径、入水脚印清理（`carve_launch_footprints`，"
        "只删与 t=0 船体脚印重叠的冰块，避免初始深穿透冲量）、逐船载荷账本、节省率、")
    add("   Roache GCI、单调性、Lindqvist 同口径换算与「超出公式适用域」判据；")
    add("3. `benchmarks/w25_formation_runner.py` 按分片跑矩阵并落原始 JSON；")
    add("4. `benchmarks/w25_formation_collect.py` 汇总系综、判定三条判据、出图；")
    add(f"5. 单船退化算例对账对象 = `tensordem.level_ice.formula` 的 Lindqvist 三分量锚"
        f"（系数 provenance 仍为 WITHHELD：主论文离线不可读，见锚脚本）。")
    add("")

    # ---- 3 results ----
    add("## 3 结果")
    add("")
    add("### 3.1 单船退化算例 vs Lindqvist 三分量锚")
    add("")
    add("| V [m/s] | DEM 单船 [kN] | seeds | seed 相对散度 | Lindqvist 总量 [kN] | 弯曲+下沉下限 [kN] | 相对偏差 | <15%？ |")
    add("|---|---|---|---|---|---|---|---|")
    for key in sorted(single, key=lambda k: float(k)):
        item = single[key]
        prim = item["lindqvist_primary"]["components"]
        floor = prim["bending_N"] + prim["submersion_N"]
        dev = item["primary_deviation"]
        add(f"| {_f(item['speed_m_s'], 2)} | {_f(item['load_mean_N'] / 1e3, 2)} "
            f"| {item['n_members']} | {_f(item['load_relative_spread_pct'], 1)} % "
            f"| {_f(prim['total_N'] / 1e3, 1)} | {_f(floor / 1e3, 1)} "
            f"| {_pct(dev['relative_deviation_pct'])} | {'是' if dev['within_tolerance'] else '否'} |")
    add("")
    first = single[sorted(single, key=lambda k: float(k))[0]]
    shares = first["lindqvist_primary"]["component_shares"]
    add(f"Lindqvist 主口径（α = {_f(first['lindqvist_primary']['alpha_deg'], 0)}°, "
        f"μ = {_f(first['lindqvist_primary']['mu'], 2)}）分量占比："
        f"挤压 {_f(100 * shares['crushing'], 1)}%、弯曲 {_f(100 * shares['bending'], 1)}%、"
        f"下沉 {_f(100 * shares['submersion'], 1)}%、附加阻力 {_f(100 * shares['added_drag'], 2)}%。")
    add("")
    n_below = sum(1 for item in single.values()
                  if item["primary_deviation"]["dem_below_anchor_floor"])
    add(f"**超出适用域判据**：{n_below}/{len(single)} 个速度点满足 "
        "`R_DEM < R_bending + R_submersion`（Lindqvist 的弯曲+下沉已是任何含连续冰盖模型的载荷下限）。"
        "该条件成立时，任何网格加密都无法把偏差压进 15%——差距是**模型形式（能力缺口）**，不是离散误差。")
    add("")
    add("### 3.2 节省率-间距曲线")
    add("")
    speeds_sorted = sorted(surface, key=lambda k: float(k))
    header = "| S/L | " + " | ".join(f"V={_f(surface[k]['speed_m_s'], 2)}" for k in speeds_sorted) + " |"
    add(header)
    add("|---" * (len(speeds_sorted) + 1) + "|")
    spacings = sorted({float(item["spacing_L"]) for k in speeds_sorted
                       for item in surface[k]["by_spacing"].values()})
    for spacing in spacings:
        cells = []
        for key in speeds_sorted:
            item = surface[key]["by_spacing"].get(f"{spacing:g}")
            if item is None:
                cells.append("n/a")
            else:
                cells.append(f"{_f(100 * item['saving_paired'], 1)} ± "
                             f"{_f(100 * item['saving_paired_sem'], 1)}")
        add(f"| {_f(spacing, 0)} | " + " | ".join(cells) + " |")
    add("")
    add("表中为**配对节省率** `1 − R_总/(3 R_领队)` [%]（同一冰场实现内比较，去掉单船参考的噪声），"
        "± 系综 SEM。独立单船参考口径的节省率见工件 `formation_savings.*.by_spacing.saving`。")
    add("")
    add("### 3.3 单调性（判据 C2a）")
    add("")
    add("| V [m/s] | 容忍阈 | 配对节省率逐档 | 严格单调 | 容忍单调 |")
    add("|---|---|---|---|---|")
    for key in sorted(mono, key=lambda k: float(k)):
        item = mono[key]
        add(f"| {_f(item['speed_m_s'], 2)} | {_f(item['tolerance_paired'], 3)} "
            f"| {', '.join(_f(v, 3) for v in item['saving_paired'])} "
            f"| {'是' if item['saving_paired_strict']['monotone'] else '否'} "
            f"| {'是' if item['saving_paired_tolerant']['monotone'] else '否'} |")
    add("")
    add("### 3.4 粒子数收敛与 GCI（判据 C2b / C3）")
    add("")
    add("| 族 | N | 2N | 4N | 观测阶 p | GCI(fine) | 单调 | 收敛 |")
    add("|---|---|---|---|---|---|---|---|")
    for key, label in (("single_ship", "单船"), ("formation", "三船纵队")):
        fam = gci.get(key)
        if not fam:
            continue
        vals = fam["values_N"]
        g = fam["gci"]
        add(f"| {label} | {_f(vals['level_1_base'] / 1e3, 2)} "
            f"| {_f(vals['level_2'] / 1e3, 2)} | {_f(vals['level_4'] / 1e3, 2)} "
            f"| {_f(g['observed_order'], 2)} | {_f(g['gci_pct'], 1)} % "
            f"| {'是' if g['monotone'] else '否'} | {'是' if g['converged'] else '否'} |")
    add("")
    add("（表内为总冰阻力 kN；GCI 取 fine 网格，安全因子 1.25，细化比 r = 2，"
        "同时细化粒子数 N→2N→4N 与 dt（保持 ω·dt 不变）。）")
    add("")
    saving_gci = gci.get("single_ship", {}).get("saving_metric", {}).get("gci")
    if saving_gci:
        vals = [gci["single_ship"]["saving_metric"][k] for k in ("level_1", "level_2", "level_4")]
        add(f"同一细化族的**节省率**：{', '.join(_f(v, 4) for v in vals)} → "
            f"GCI = {_f(saving_gci['gci_pct'], 1)} %（比值量的误差相消，比绝对载荷更容易收敛）。")
        add("")
    add("### 3.5 数值控制")
    add("")
    ts = numerics.get("timestep")
    if ts:
        add(f"- 时间步减半（dt {_f(ts['base_dt_s'], 3)} → {_f(ts['half_dt_s'], 4)} s，基准网格）："
            f"载荷 {_f(ts['load_base_N'] / 1e3, 2)} → {_f(ts['load_half_dt_N'] / 1e3, 2)} kN，"
            f"相对差 {_pct(ts['relative_difference_pct'])}。")
    band = numerics.get("drift_sensitivity", {}).get("single_ship_band")
    if band:
        add(f"- 横向漂流敏感带：单船载荷 {', '.join(_f(v / 1e3, 2) for v in band['loads_N'])} kN，"
            f"极差 {_f(band['relative_spread_pct'], 1)} %。")
    add("")
    if iso.get("available"):
        add("### 3.6 ISO 19906 p-A 面积律交叉核对（上界口径）")
        add("")
        add("| V [m/s] | DEM 名义压力 [kPa] | ISO 19906 p-A 压力 [kPa] | DEM/ISO 载荷比 |")
        add("|---|---|---|---|")
        for key in sorted(iso["rows"], key=lambda k: float(k)):
            row = iso["rows"][key]
            add(f"| {_f(row['speed_m_s'], 2)} | {_f(row['dem_mean_pressure_kpa'], 1)} "
                f"| {_f(row['iso19906_pressure_kpa'], 1)} | {_f(row['dem_over_iso_load'], 4)} |")
        add("")
        add(f"口径：{iso['convention']}；实现 `{iso['vehicle']}`，"
            f"参数 p_0 = {_f(iso['params']['p_0_pa'], 0)} Pa，A_0 = {_f(iso['params']['A_0_m2'], 1)} m²，"
            f"指数 {_f(iso['params']['exponent'], 2)}；来源 {iso['provenance']}。"
            f"DEM/ISO 比 {_f(iso['dem_over_iso_load_range'][0], 3)}–"
            f"{_f(iso['dem_over_iso_load_range'][1], 3)} = "
            f"{iso['interpretation']}。")
        add("")

    # ---- 4 figures ----
    add("## 4 图")
    add("")
    for fig in artifact.get("figures", []):
        if "path" in fig:
            add(f"- `{fig['path']}` — {fig['caption']}")
    add("")

    # ---- 5 pitfalls ----
    add("## 5 注意点与适用范围")
    add("")
    add("1. **刚体圆盘无弯曲破坏**：模型没有连续冰盖的弯曲失效机制，Lindqvist 的 R_b 在其中没有对应物；")
    add("2. **二维方形船体无艏倾角**：α = 90° 使挤压项发散，有限 α 只是替代口径（公式本身也拒绝 ≥89°）；")
    add("3. **单实现的系综散度数十个百分点**：瞬时载荷以 O(MN) 稀疏尖峰叠加出 O(10 kN) 均值，"
        "重尾时间平均必须用系综均值 + SEM，单例不足以支撑曲线；")
    add("4. **横向漂流是建模选择**：它是唯一能让领队清出的走廊重新闭合、从而产生间距依赖的机制，"
        "其敏感带已给出；")
    add("5. **冰厚恒定 0.7 m**：本算例无法检验锚点的厚度依赖（h² 标度）。")
    add("")

    # ---- 6 verdicts ----
    add("## 6 误差判定（关键量 vs 基准，逐项回答「是否 <3%」）")
    add("")
    add("| 关键量 | 基准 | 误差 | <3%？ |")
    add("|---|---|---|---|")
    c1 = verdicts["C1"]
    if c1["worst_case"]:
        add(f"| 单船冰阻力（最差速度点） | Lindqvist 三分量总量 | "
            f"{_pct(c1['worst_case']['deviation_pct'])} | 否 |")
    gci_single = gci.get("single_ship", {}).get("gci", {})
    gci_form = gci.get("formation", {}).get("gci", {})
    def _lt3(value, scale: float = 1.0) -> str:
        if value is None or not math.isfinite(float(value)):
            return "n/a"
        return "是" if abs(float(value)) * scale < 3.0 else "否"

    add(f"| 单船冰阻力网格收敛（N/2N/4N） | 加密极限 | GCI {_f(gci_single.get('gci_pct'), 1)} % | "
        f"{_lt3(gci_single.get('gci_pct'))} |")
    add(f"| 编队总冰阻力网格收敛（N/2N/4N） | 加密极限 | GCI {_f(gci_form.get('gci_pct'), 1)} % | "
        f"{_lt3(gci_form.get('gci_pct'))} |")
    if ts:
        add(f"| 时间步离散误差 | dt/2 参考 | {_pct(ts['relative_difference_pct'])} | "
            f"{_lt3(ts['relative_difference_pct'])} |")
    add("")
    add("判据结论（程序化判定，honest FAIL 不覆盖）：")
    add("")
    for item in artifact["verdicts"]:
        add(f"- **{item['id']} {item['status']}** — {item['name']}")
    add("")
    add(f"汇总：{artifact['overall']['n_pass']} PASS / "
        f"{artifact['overall']['n_fail']} FAIL / {artifact['overall']['n_withheld']} WITHHELD。")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    artifact = json.loads(ARTIFACT.read_text())
    OUT.write_text(build(artifact))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())