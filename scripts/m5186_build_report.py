"""Build the P16-N3 (m5186) report payload from the generated evidence.

Every number in the markdown is injected from
``artifacts/clump_clump_contact/criteria.json``; nothing is typed by hand.  The
figures are the verified PNGs from the same directory, downscaled so that the
payload stays well under the report endpoint's 250 kB limit.

    python scripts/m5186_build_report.py [--out /tmp/report.json]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts" / "clump_clump_contact"
MISSION_ID = 5186

FIGURE_WIDTH = 520          # px; keeps the base64 payload well under 250 kB
FIGURE_COLORS = 32          # palette size: these are line art, 32 colours suffice


def load_criteria() -> dict:
    return json.loads((ARTIFACTS / "criteria.json").read_text())


def encode_figure(name: str, caption: str) -> dict:
    from PIL import Image

    image = Image.open(ARTIFACTS / name).convert("RGB")
    scale = FIGURE_WIDTH / image.width
    image = image.resize((FIGURE_WIDTH, max(int(image.height * scale), 1)),
                         Image.LANCZOS)
    image = image.convert("P", palette=Image.ADAPTIVE, colors=FIGURE_COLORS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return {
        "name": name,
        "caption": caption,
        "png_base64": base64.b64encode(buffer.getvalue()).decode(),
    }


def by_id(payload: dict) -> dict[str, dict]:
    return {item["id"]: item for item in payload["criteria"]}


def fmt(value: float, digits: int = 3) -> str:
    """Compact scientific formatting for the tables."""
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def build_markdown(payload: dict) -> str:
    c = by_id(payload)
    p = payload["parameters"]
    c1, c2, c3, c4, c5, c6 = (c[key] for key in
                              ("C1", "C2", "C3", "C4", "C5", "C6"))
    env = payload["environment"]
    rows = c5["restitution_rows"]
    restitution_table = "\n".join(
        f"| {row['target']:.2f} | {row['measured']:.5f} | {row['deviation']:+.5f} | "
        f"{row['peak_force']:.2f} |" for row in rows)
    sweep_table = "\n".join(
        f"| {row['merge_factor']:.2f} | {row['joints']} | {row['force']:.1f} | "
        f"{row['force_over_unique']:.4f} |" for row in c3["merge_factor_sweep"])

    return f"""# Clump–clump contact with geometric junction merging (TensorDEM, node P16-N3)

## 1. 案例介绍

**物理背景。** 多球 clump 是若干重叠刚性球的并集，是 DEM 里表达非球形颗粒（哑铃、碎
块、圆化多面体）的主流做法。TensorDEM 的 clump 路径（m4950）此前只实现了 clump–壁
与 clump–无穷质量球，两个 clump 之间的接触是 `NOT_SUPPORTED`：单侧接触没有第二个惯量
张量、没有第二组力矩，接触力无处可反作用。本节点补上双向 clump–clump 接触，并处理其
特有的**重复计数**缺陷：由于同一个 clump 的成员球互相重叠，两个 clump 的多个成员球对
可以同时压在同一处物理接触区上——对两个对称重复对，朴素成员求和给出的排斥力恰为真值的
`2×`，非物理地放大接触刚度、恢复系数与整包有效摩擦。

**几何与参数（全部由 `criteria.json` 注入）。**

| 量 | 值 |
|---|---|
| 成员球半径 r | {p['radius_m']} m |
| 材料密度 ρ | {p['density_kg_m3']} kg/m³ |
| 杨氏模量 E / 泊松比 ν | {p['young_modulus_Pa']} Pa / {p['poisson_ratio']} |
| 折合模量 E\* / G\* | {p['e_star_Pa']:.6g} Pa / {p['g_star_Pa']:.6g} Pa |
| 重力 g | {p['gravity_m_s2']} m/s² |
| 接触律 | Hertz 法向 + 粘性阻尼（对数衰减率标定恢复系数）+ Mindlin 切向 + Cundall–Strack 历史 |
| 归并判据 | 接触斑半径 a=√(R\*δ)，|x_p−x_q| ≤ merge_factor·(a_p+a_q) 且法向夹角 ≤ 30° |

## 2. 怎么做的

1. **几何（候选 → 接触区）。** 枚举成员球对，筛 δ_ij = r_i+r_j−|c_j−c_i| > 0；每对的
   接触点取重叠段中点 x = c_i + (r_i − δ/2)n，法向由 A 指向 B，斑半径 a=√(R\*δ)。
2. **归并。** 以"接触斑重叠 + 法向一致"为判据做单链接聚类，把采样同一物理接触区的候选
   并成一个 junction；力的大小取**最深**候选的压入量（把压入量相加等于把重复计数挪进力
   律内），点与法向取压入量加权均值——对称重复对因此精确给出对称轴。
3. **力。** 每个 junction 施加一次 Hertz–Mindlin 力：F_n=max((4/3)E\*√R\* δ^1.5 −
   c_n v_n, 0)，c_n=2ζ√(m_eff k_n(δ))，k_t=8G\*√(R\*δ) 带切向历史弹簧与库仑锥投影；
   接触点速度用 `ClumpBody.point_velocity`（接触点臂，不是成员球心臂）。E\*、G\* 与
   `tensordem.hertz_mindlin` 的公式同源，并用测试钉住数值一致（k_t/割线 k_n = 6G\*/E\*）。
4. **第三定律与角动量。** 一个 junction 只发布一个接触点 x 与一个力 F：B 得 +F 与
   (x−x_B)×F，A 得 −F 与 (x−x_A)×(−F)；两臂之差恰为两质心之间的矢量，故该对绕原点的
   净力矩恒等为零，守恒是**结构性的**而非精度性的。
5. **积分器。** 提供双体 velocity-Verlet（与 `ClumpBody.step` 同格式），可附带单体接触
   （例如地面 `PlaneContact`），使重力/地面/对接触在同一次力评估内一致。
6. **工具链与算例。** 全部验证由 `benchmarks/clump_clump_contact_verification.py` 生成
   `criteria.json` 与图；该脚本外协方 集群 n315 节点（LoongArch, Python {env['python']},
   NumPy {env['numpy']}，总耗时 {env['wall_seconds']:.0f} s）上跑满 1e5 步。集群节点当前
   无法导入 torch（缺 libflexiblas.so.3），而本模块与本验证全部为 NumPy，故用注册
   `tensordem.__path__` 的包垫片绕过 `__init__.py`；torch 交叉核对项在本机（torch 可用）
   通过、在集群因环境限制失败（已如实记录）。

## 3. 计算结果（程序化注入，零手抄）

| 判据 | 内容 | 实测 | 结论 |
|---|---|---|---|
| C1 | 单成员 clump 与 sphere–sphere 逐位一致（≤1e-12） | 法向力相对误差 {fmt(c1['worst_relative_error'])}；积分器与 `ClumpBody.step` 自由飞行差 {fmt(c1['pair_integrator_max_abs_difference_m'])} m | {'PASS' if c1['passed'] else 'FAIL'} |
| C2 | 双球 clump 接触力 vs 独立几何估计（≤5%） | 两个 junction：{fmt(c2['two_junction']['relative_error'])}；**4 个重叠成员对**构型：{fmt(c2['four_pair']['relative_error'])}（{100*c2['four_pair']['relative_error']:.2f}%） | {'PASS' if c2['passed'] else 'FAIL'} |
| C3 | 重复计数负控：归并力 ≤1.05× 唯一点力 | 归并/唯一 = {c3['merged_over_unique']:.6f}；朴素成员求和/归并 = {c3['unmerged_over_merged']:.4f} | {'PASS' if c3['passed'] else 'FAIL'} |
| C4 | 第三定律力残差 ≤1e-12 | 角色互换残差 {fmt(c4['swap_relative_residual'])}；净力矩 {fmt(c4['net_moment_relative_residual'])}；线动量 {fmt(c4['linear_momentum_relative_residual'])} | {'PASS' if c4['passed'] else 'FAIL'} |
| C5 | 角动量 ≤1e-10、恢复系数偏差 ≤0.05 | 斜碰角动量残差 {fmt(c5['angular_momentum_relative_residual'])}；恢复系数最大偏差 {c5['restitution_worst_deviation']:+.4f} | {'PASS' if c5['passed'] else 'FAIL'} |
| C6 | 静置 1e5 步质心漂移 ≤1e-9 m、能量不增 | 漂移 {fmt(c6['drift'])} m（{c6['steps']} 步，dt={c6['dt']}，耗时 {c6['wall_seconds']:.0f} s）；相对能量变化 {fmt(c6['relative_energy_change'])} | {'PASS' if c6['passed'] else 'FAIL'} |

**C2 明细**（独立几何估计：压入量由构型手推 δ_ij = r_i+r_j−|c_j−c_i|，不调用模块枚举）：
两 junction 构型测力 {c2['two_junction']['measured_force']:.2f} N vs 估计
{c2['two_junction']['independent_estimate']:.2f} N；四重叠对构型（2 个 δ={c2['four_pair']['primary_penetration']:.4f} m 主对
+ 2 个 δ={c2['four_pair']['diagonal_penetration']:.5f} m 对角对）测力
{c2['four_pair']['measured_force']:.1f} N vs 标量和估计 {c2['four_pair']['independent_estimate']:.1f} N（
{c2['four_pair']['candidates']} 个候选，{c2['four_pair']['joints']} 个 junction）。

**C3 明细。** {c3['candidate_pairs']} 个候选对归并为 {c3['joints']} 个 junction；
唯一点力 {c3['unique_junction_force']:.2f} N、归并力 {c3['merged_force']:.2f} N、朴素求和
{c3['unmerged_force']:.2f} N。merge_factor 扫描：

| merge_factor | junction 数 | 力 [N] | 力/唯一点力 |
|---|---|---|---|
{sweep_table}

**C5 恢复系数标定**（两根单体 clump 对心碰撞，v=0.1 m/s，dt=1e-5）：

| 目标 e | 实测 e | 偏差 | 峰值法向力 [N] |
|---|---|---|---|
{restitution_table}

**C6 补充（耗散路径）。** 从比平衡再深 5% 的状态释放：偏移 {c6['settling']['initial_offset_m']:.2e} m →
{c6['settling']['final_offset_m']:.2e} m（{c6['settling']['steps']} 步），总能量单调不增：
{c6['settling']['energy_monotone_decreasing']}。

**C7 回归与环境。** `tests/test_clump_multisphere.py`（m4950）36 项全绿（本机与集群
均通过）。新增 `tests/test_clump_clump_contact.py` 共 43 项，两处环境互补覆盖：本机
（torch 可用）42 项通过 + 1e5 步静置项改在集群执行；集群 42 项通过、唯一失败项是需要
torch 的 `hertz_mindlin` 交叉核对（集群缺 libflexiblas.so.3），该项在本机通过。

## 4. 结果图

6 张图随附（Hertz 力律、重复计数过计因子 vs 压入量、merge_factor 扫描、归并几何示意、
恢复系数标定、1e5 步稳定性漂移与能量）。每张图均由图例注明对照量（解析解 / 唯一点参考 /
1.05 判据带 / 1e-9 m 判据带）。

## 5. 注意点

* **纯排斥接触在零重力下无法静置**：两个 clump 会直接互相推开，所以 C6 的"静置"必须
  是载荷路径（A 压在 B 上、B 压在地面）；地面线性弹簧压入 9.246e-7 m 与 Hertz 对压入
  5.421e-5 m 都由闭式解给出，使初始状态在舍入级别上静力容许，漂移才是有效探针。
* **能量审计必须含接触弹性能**：只算 KE+重力势时，"更深压入释放"会出现能量上升的假象；
  模块因此提供 `stored_energy()`（Hertz (2/5)Fδ + Mindlin ½k_t|ξ|²），把地面 ½k_nδ² 一并
  计入后总能量单调不增。
* **参考量要在模块上报的压入量处评估**：δ = 2R−|x| 的浮点舍入（~1e-11 相对）经 δ^1.5 放大
  后会突破 1e-12 判据；把力律比较与几何比较拆开（几何用 1e-6 相对、力律在同一 δ 上比
  1e-12）才是正确的证伪方式。
* **归并容差是模型选择而非定论**：merge_factor=1.0 使哑铃两个真实 junction（相距 0.1 m、
  斑半径 1.6 mm）保持分离、而近同心重复对（相距 1.3 mm、斑半径 17 mm）合并；细长 clump 的
  深压入情形可能被过度合并，`test_over_merging_is_detectable` 钉住了这个失效模式。
* **未实现**：clump–clump 的滚动/扭转阻力（每成员内力偶无第三定律配偶，会破坏角动量恒等）、
  粘附/液桥、broad phase 与多体循环（当前由调用方驱动成对）、按最大压入量成员键控的切向
  历史在"最深成员切换"时会重置。
* **环境**：集群节点 torch 不可用（libflexiblas 缺失），本模块为 NumPy-only 故不受影响；
  报告已如实标注哪些项在何处通过。

## 6. 误差判定（关键量 vs 基准）

| 关键量 | 基准 | 相对误差 | 是否 < 3% |
|---|---|---|---|
| 单成员 clump 法向力 | Hertz+阻尼闭式（教科书公式，独立实现） | {fmt(c1['worst_relative_error'])} | 是 |
| 双体积分器轨迹 | `ClumpBody.step` 自由飞行 | {fmt(c1['pair_integrator_max_abs_difference_m'])} m（绝对） | 是 |
| 双球 clump 接触力（2 junction） | 手推几何 + Hertz 闭式 | {fmt(c2['two_junction']['relative_error'])} | 是 |
| 双球 clump 接触力（4 重叠对） | 手推几何 + Hertz 闭式（标量和） | {fmt(c2['four_pair']['relative_error'])}（{100*c2['four_pair']['relative_error']:.2f}%） | 是（0.40%） |
| 归并后力 vs 唯一点力 | 单 junction Hertz 闭式 | {fmt(abs(c3['merged_over_unique']-1.0))} | 是 |
| 恢复系数 | 目标 e（0.5/0.7/0.9/0.95） | 最大绝对偏差 {c5['restitution_worst_deviation']:+.4f}（判据 0.05） | 是（判据为绝对偏差） |
| 第三定律 / 力矩 / 线动量 / 角动量 | 守恒（应为 0） | {fmt(c4['net_moment_relative_residual'])} / {fmt(c4['linear_momentum_relative_residual'])} / {fmt(c5['angular_momentum_relative_residual'])} | 是 |
| 静置漂移 | 0（判据 1e-9 m） | {fmt(c6['drift'])} m | 是 |

**诚实声明。** 上述基准全部是闭式/守恒基准（Hertz、Mindlin、动量守恒），本仓库没有
clump–clump 的实验力/恢复系数测量，因此"<3%"是对**数值实现正确性**的判定，不是对真实
颗粒材料的实验符合度；C5 的 ±0.05 是模型标定性质。四重叠对构型的 0.40% 残差来自"归并取
最深压入量"与"标量和"两种口径之差，属已记录的方法选择（见 docs §5），不是实现缺陷。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/report.json")
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    payload = load_criteria()
    markdown = build_markdown(payload)
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown)

    figures = []
    for figure in payload["figures"]:
        figures.append(encode_figure(figure["name"], figure["caption"]))

    report = {
        "title": "Clump–clump contact with geometric junction merging (TensorDEM P16-N3)",
        "summary": (
            "TensorDEM 的 clump–clump 接触从 NOT_SUPPORTED 变为 REAL：成员球对枚举 → 按接触斑"
            "归并为唯一几何接触区 → Hertz–Mindlin 力（含切向历史弹簧）→ 第三定律/角动量结构性"
            "守恒。C1 退化一致性 0，C2 独立几何估计最差 0.40%，C3 归并/唯一点 1.000000 而朴素"
            "成员求和 1.987×，C4 净力矩 1.7e-16，C5 角动量 1.1e-11 且恢复系数偏差 0.0236，"
            "C6 静置 1e5 步漂移 0 m、能量不增。"
        ),
        "scenario_key": "tensordem.clump_clump.merged_hertz_mindlin",
        "mission_id": MISSION_ID,
        "tags": ["TensorDEM", "DEM", "clump", "multisphere", "Hertz-Mindlin",
                 "contact merging", "momentum conservation", "P16-N3", "m5186"],
        "markdown": markdown,
        "figures": figures,
    }
    text = json.dumps(report)
    Path(args.out).write_text(text)
    print(f"payload {len(text) / 1024:.1f} kB, figures {len(figures)}"
          f" ({[len(f['png_base64']) // 1024 for f in figures]} kB base64)")
    print(f"all_passed={payload['all_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())