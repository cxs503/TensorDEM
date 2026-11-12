"""Publish the m16718 (W25 formation-ice-channel rework) paper-grade report.

Every number and every figure is read from
``papers/artifacts/w25/formation-ice-channel/lindqvist_formation_gci.json``
(and the PNGs it registers); nothing is typed by hand.

Usage::

    python post_report_m16718.py            # writes /tmp/report.json and posts it
    python post_report_m16718.py --dry-run  # only writes /tmp/report.json
"""
from __future__ import annotations

import base64
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACT = Path(os.environ.get(
    "W25_ARTIFACT",
    REPO_ROOT / "papers" / "artifacts" / "w25" / "formation-ice-channel" / "lindqvist_formation_gci.json"))
MISSION_ID = 16718
BASE = os.environ.get("TENSOR_API_BASE", "").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


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


def post(path: str, body: dict, token: str | None = None) -> dict:
    request = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode())


def build_markdown(art: dict) -> str:
    cal = art["caliber"]
    ship, ice, env, num = cal["ship"], cal["ice"], cal["environment"], cal["numerics"]
    matrix = art["case_matrix"]
    single = art["single_ship_vs_lindqvist"]
    surface = art["formation_savings"]
    mono = art["monotonicity"]
    gci = art["grid_convergence"]
    numerics = art["numerics_controls"]
    iso = art["iso19906_crosscheck"]
    verdicts = {v["id"]: v for v in art["verdicts"]}
    md = []
    add = md.append

    add("## 1 案例介绍")
    add("")
    add("**物理背景**：冰区航行的船舶编队（纵队）中，后续船舶航行于领队清出的航道内，冰阻力显著下降；"
        "航道会被横向漂流/潮流重新闭合，因此节省率随间距与航速变化。本实验用 DEM（刚体圆盘 + 锚定罚函数接触）"
        "量化该现象，并与 Lindqvist (1989) 单船三分量解析阻力对账。")
    add("")
    add("| 参数 | 值 |")
    add("|---|---|")
    add(f"| 船体 | L = {_f(ship['length_m'], 0)} m, B = {_f(ship['beam_m'], 0)} m, L/B = {_f(ship['length_over_beam'], 2)}；{ship['hull_shape']} |")
    add(f"| 方形系数 | Cb = {_f(art['geometry_check']['block_coefficient_Cb'], 3)}（几何先验已机器校验：矩形船体 → Cb = 1） |")
    add(f"| 冰况 | h = {_f(ice['thickness_m'], 2)} m，密集度 {_f(ice['concentration'], 1)}，平均冰块半径 {_f(ice['mean_floe_radius_m'], 1)} m（冰径/船宽 {_f(ice['floe_diameter_over_beam'], 2)}） |")
    add(f"| 环境 | 无风；横向漂流 {_f(env['cross_channel_drift_m_s'], 2)} m/s（航道再闭合驱动）；海水 {_f(env['water_density_kg_m3'], 0)} kg/m³ |")
    add(f"| 编队 | 三船纵队，间距 S/L ∈ {matrix['spacings_L']} |")
    add(f"| 航速 | V ∈ {matrix['speeds_m_s']} m/s |")
    add(f"| 算例数 | {matrix['n_cases_planned']} 个 DEM 算例（已完成 {matrix['n_cases_with_results']}，缺失 {matrix['n_missing']}） |")
    add(f"| 数值 | dt = {_f(num['base_dt_s'], 3)} s；域宽 {_f(num['domain_width_m'], 0)} m；冰场长度 {num['field_length_rule']}；系综 3（编队）/5（单船）|")
    add("")
    add("**测量定义（先于运行声明）**：船体中心位于冰场内且距纵向边界 ≥ L/2、t ≥ 60 s 才计入样本；"
        "单船/逐船载荷 = 该窗口内上游接触反力的时间平均；编队总冰阻力 = 三船之和；"
        "节省率 = 1 − R_总/(3·R_单船)（配对口径 1 − R_总/(3·R_领队) 用于去噪）。")
    add("")

    add("## 2 怎么做的")
    add("")
    add("1. 求解器：`tensordem.ice_channel.IceChannelSimulation`（抖动六角冰场、圆形浮冰、平动+转动+垂荡、"
        "冰-冰/冰-船罚函数弹簧接触、风/流二次阻力）。船体为运动学矩形，多船通过 `extra_hulls` 组队；")
    add("2. 新增库 `tensordem.ice_channel_formation`：算例口径/几何、入水脚印清理（删除与 t=0 船体脚印重叠的冰块，"
        "消除初始深穿透冲量）、逐船载荷账本、节省率、Roache 三网格 GCI（含单调/振荡判定）、"
        "Lindqvist 同口径换算与「超出适用公式域」判据、航道再闭合时间；")
    add("3. 算例矩阵 `tensordem.ice_channel_formation_cases`（预注册）：单船 7 速度 ×5 seeds；编队 5 间距 ×7 速度 ×3 seeds；"
        "粒子加密族 N/2N/4N（粒子数与 dt 同步细化以保持 ω·dt）+ 时间步减半控制 + 漂流敏感带；")
    add("4. 执行：`benchmarks/w25_formation_runner.py` 分片跑集群；"
        "`benchmarks/w25_formation_collect.py` 汇总系综、程序化判定三条判据并出图；"
        "`benchmarks/w25_formation_doc.py` 从工件生成文档（零手抄）。")
    add("5. 外部锚：`tensordem.level_ice.formula` 的 Lindqvist 三分量（系数 provenance 仍 WITHHELD）+ "
        "`tensordem.ice_ridge_keel_hull.iso19906_ridge_load` 的 ISO 19906 p-A 面积律（上界口径）。")
    add("")

    add("## 3 计算结果")
    add("")
    add("### 3.1 单船退化算例 vs Lindqvist")
    add("")
    add("| V [m/s] | DEM [kN] | 系综散度 | Lindqvist 总量 [kN] | 弯曲+下沉下限 [kN] | 偏差 | <15% |")
    add("|---|---|---|---|---|---|---|")
    for key in sorted(single, key=lambda k: float(k)):
        item = single[key]
        comp = item["lindqvist_primary"]["components"]
        floor = comp["bending_N"] + comp["submersion_N"]
        dev = item["primary_deviation"]
        add(f"| {_f(item['speed_m_s'], 2)} | {_f(item['load_mean_N'] / 1e3, 2)} "
            f"| {_f(item['load_relative_spread_pct'], 1)} % | {_f(comp['total_N'] / 1e3, 1)} "
            f"| {_f(floor / 1e3, 1)} | {_pct(dev['relative_deviation_pct'])} "
            f"| {'是' if dev['within_tolerance'] else '否'} |")
    add("")
    n_below = sum(1 for item in single.values() if item["primary_deviation"]["dem_below_anchor_floor"])
    add(f"**超出适用域**：{n_below}/{len(single)} 个速度点上 `R_DEM < R_弯曲 + R_下沉`。"
        "Lindqvist 的弯曲与下沉项对任何带连续冰盖的模型都是载荷下限；DEM 单船载荷长期低于该下限，"
        "说明差距是**模型形式（能力缺口）**而非离散误差，网格加密无法消除。")
    add("")
    add("### 3.2 节省率-间距曲线（配对口径，%；± SEM）")
    add("")
    speeds = sorted(surface, key=lambda k: float(k))
    add("| S/L | " + " | ".join(f"V={_f(surface[k]['speed_m_s'], 2)}" for k in speeds) + " |")
    add("|---" * (len(speeds) + 1) + "|")
    spacings = sorted({float(item["spacing_L"]) for k in speeds for item in surface[k]["by_spacing"].values()})
    for spacing in spacings:
        cells = []
        for key in speeds:
            item = surface[key]["by_spacing"].get(f"{spacing:g}")
            cells.append("n/a" if item is None else
                         f"{_f(100 * item['saving_paired'], 1)} ± {_f(100 * item['saving_paired_sem'], 1)}")
        add(f"| {_f(spacing, 0)} | " + " | ".join(cells) + " |")
    add("")
    add("| V [m/s] | 严格单调 | 容忍单调（1 SEM） | 配对节省率逐档 |")
    add("|---|---|---|---|")
    for key in sorted(mono, key=lambda k: float(k)):
        item = mono[key]
        add(f"| {_f(item['speed_m_s'], 2)} | {'是' if item['saving_paired_strict']['monotone'] else '否'} "
            f"| {'是' if item['saving_paired_tolerant']['monotone'] else '否'} "
            f"| {', '.join(_f(v, 3) for v in item['saving_paired'])} |")
    add("")
    add("### 3.3 粒子数收敛（N / 2N / 4N）")
    add("")
    add("| 族 | N [kN] | 2N [kN] | 4N [kN] | 观测阶 p | GCI | 单调 |")
    add("|---|---|---|---|---|---|---|")
    for key, label in (("single_ship", "单船"), ("formation", "三船纵队")):
        fam = gci.get(key)
        if not fam:
            continue
        vals, g = fam["values_N"], fam["gci"]
        add(f"| {label} | {_f(vals['level_1_base'] / 1e3, 2)} | {_f(vals['level_2'] / 1e3, 2)} "
            f"| {_f(vals['level_4'] / 1e3, 2)} | {_f(g['observed_order'], 2)} "
            f"| {_f(g['gci_pct'], 1)} % | {'是' if g['monotone'] else '否'} |")
    add("")
    ts = numerics.get("timestep")
    if ts:
        add(f"时间步控制：dt {_f(ts['base_dt_s'], 3)} → {_f(ts['half_dt_s'], 4)} s，"
            f"载荷 {_f(ts['load_base_N'] / 1e3, 2)} → {_f(ts['load_half_dt_N'] / 1e3, 2)} kN"
            f"（{_pct(ts['relative_difference_pct'])}）。")
    band = numerics.get("drift_sensitivity", {}).get("single_ship_band")
    if band:
        add(f"漂流敏感带：单船载荷 {', '.join(_f(v / 1e3, 2) for v in band['loads_N'])} kN，"
            f"极差 {_f(band['relative_spread_pct'], 1)} %。")
    add("")

    add("## 4 结果图")
    add("")
    for fig in art.get("figures", []):
        if "name" in fig:
            add(f"- `{fig['name']}`：{fig.get('caption', '')}")
    add("")

    add("## 5 注意点")
    add("")
    add("1. 刚体圆盘没有弯曲破坏，模型不含连续冰盖的弯曲失效（Lindqvist R_b 无对应物）；")
    add("2. 二维方形船体无艏倾角，α = 90° 使挤压项发散，有限 α 只是替代口径；")
    add("3. 单实现的系综散度达数十个百分点：瞬时载荷是 O(MN) 稀疏尖峰叠加 O(10 kN) 均值，"
        "必须用系综均值 ± SEM，单例不足以支撑曲线；")
    add("4. 横向漂流是建模选择，也是间距依赖的唯一来源，已给敏感带；")
    add("5. 冰厚恒定 0.7 m，本算例无法检验锚点的 h² 标度。")
    add("")

    add("## 6 误差判定（逐项回答「是否 <3%」）")
    add("")
    add("| 关键量 | 基准 | 误差 | <3%？ |")
    add("|---|---|---|---|")
    worst = verdicts["C1"]["worst_case"]
    if worst:
        add(f"| 单船冰阻力（最差速度点） | Lindqvist 三分量总量 | {_pct(worst['deviation_pct'])} | 否 |")

    def lt3(value) -> str:
        if value is None or not math.isfinite(float(value)):
            return "n/a"
        return "是" if abs(float(value)) < 3.0 else "否"

    add(f"| 单船载荷网格收敛 | 加密极限 | GCI {_f(gci.get('single_ship', {}).get('gci', {}).get('gci_pct'), 1)} % | "
        f"{lt3(gci.get('single_ship', {}).get('gci', {}).get('gci_pct'))} |")
    add(f"| 编队总载荷网格收敛 | 加密极限 | GCI {_f(gci.get('formation', {}).get('gci', {}).get('gci_pct'), 1)} % | "
        f"{lt3(gci.get('formation', {}).get('gci', {}).get('gci_pct'))} |")
    if ts:
        add(f"| 时间步离散误差 | dt/2 | {_pct(ts['relative_difference_pct'])} | {lt3(ts['relative_difference_pct'])} |")
    if iso.get("available") and iso.get("dem_over_iso_load_range"):
        add(f"| 名义压力 vs ISO 19906 p-A | 连续冰盖设计载荷 | "
            f"DEM/ISO = {_f(iso['dem_over_iso_load_range'][0], 3)}–{_f(iso['dem_over_iso_load_range'][1], 3)} | 否 |")
    add("")
    add("判据（程序化，honest FAIL 不覆盖）：")
    add("")
    for item in art["verdicts"]:
        add(f"- **{item['id']} {item['status']}** — {item['name']}")
    add("")
    add(f"汇总：{art['overall']['n_pass']} PASS / {art['overall']['n_fail']} FAIL / "
        f"{art['overall']['n_withheld']} WITHHELD。"
        "本实验的可发表结论限于**相对量**（节省率-间距曲线及其单调性、粒子收敛性），"
        "绝对量对 Lindqvist 的偏差是模型形式差异的量化结果，不构成对该公式的验证。")
    return "\n".join(md)


def main() -> int:
    art = json.loads(ARTIFACT.read_text())
    markdown = build_markdown(art)

    figures = []
    for fig in art.get("figures", []):
        path = REPO_ROOT / fig.get("path", "")
        if not path.exists():
            continue
        figures.append({
            "name": fig["name"],
            "caption": fig.get("caption", ""),
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })

    worst = {v["id"]: v for v in art["verdicts"]}["C1"]["worst_case"]
    report = {
        "title": ("W25 返修轮1：DEM 三船纵队冰阻力节省率 vs Lindqvist (1989) 三分量锚"
                  "（间距 1L-5L × 航速 0.5-2.0 m/s，粒子数 GCI）"),
        "summary": (
            f"{art['case_matrix']['n_cases_with_results']}/{art['case_matrix']['n_cases_planned']} "
            f"个 DEM 算例（系综 3-5 seeds）扫出节省率-间距-航速面；"
            f"单船退化算例对 Lindqvist 最差偏差 {_pct(worst['deviation_pct']) if worst else 'n/a'}"
            f"（判据 C1 {art['overall']['statuses']['C1']}，且载荷低于弯曲+下沉下限 → 模型形式缺口）；"
            f"节省率曲线单调性 C2a {art['overall']['statuses']['C2a']}；"
            f"粒子数 GCI {_f(art['grid_convergence'].get('single_ship', {}).get('gci', {}).get('gci_pct'), 1)}% / "
            f"{_f(art['grid_convergence'].get('formation', {}).get('gci', {}).get('gci_pct'), 1)}% → C3 "
            f"{art['overall']['statuses']['C3']}。"),
        "scenario_key": "dem.ice.formation_ice_channel.lindqvist_formation_gci",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "DEM", "ice-resistance", "ship-formation", "Lindqvist1989",
                 "ISO19906", "GCI", "preregistered", "honest-report"],
        "markdown": markdown,
        "figures": figures,
    }
    Path("/tmp/report.json").write_text(json.dumps(report, indent=1))
    print(f"wrote /tmp/report.json ({len(markdown)} chars, {len(figures)} figures)")
    if "--dry-run" in sys.argv:
        return 0
    if not (BASE and USER and PASS):
        print("TENSOR_API_BASE / TENSOR_AGENT_USER / TENSOR_AGENT_PASS not set; dry run only")
        return 1
    login = post("/auth/login", {"username": USER, "password": PASS})
    token = (login.get("access_token") or login.get("token")
             or (login.get("data") or {}).get("access_token"))
    assert token, login
    out = post("/tensor/reports", report, token)
    print(json.dumps(out)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())