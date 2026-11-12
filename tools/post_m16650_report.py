import base64, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tp_api import post

REPO = Path("/root/tp-agent/work/m16650")
ART = REPO / "artifacts/granular3d_stress"
data = json.loads((ART / "verification.json").read_text())
col, wall, pin, mut = data["column"], data["wall"], data["sign_pin"], data["mutation_sensitivity"]
packs, rot = data["random_packing_pressure"], data["rotation"]


def fig(name, caption):
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode((ART / name).read_bytes()).decode()}


rows_pack = "\n".join(
    f"| {p['phi']:.2f} | {p['contacts']} | {p['pressure_module']:.10f} | "
    f"{p['pressure_sphere_packing']:.10f} | {p['rel_diff']:.2e} |" for p in packs)
rows_mut = "\n".join(f"| `{k}` | {v:.3e} |" for k, v in mut.items())
rows_col = "\n".join(
    f"| {z:.2f} | {s:.6f} | {a:.6f} | {abs(s - a):.2e} |"
    for z, s, a in zip(col["planes"], col["sigma_zz"], col["analytic_sigma_zz"]))

md = f"""## ① 案例介绍

3-D DEM 里两个仍在 open 的算例锚点都由**应力**判据决定：
`CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT`（无量纲剪应力 vs Bagnold 数的标度指数）与
`CASE-DEM-PLANE-SHEAR-MUIDI`（mu、phi 是否塌缩到 mu(I)）。取数路径却是各算例私有的
`l ⊗ f` 求和（`elastic_limit_shear`、`bpm_calibration_feasibility.virial_stress_tensor`）；
2-D 有共享应力面（`granular2d.stress`，含离散平衡恒等式），3-D 只有
`granular3d.jamming.SpherePacking` 里的各向同性压力。私有求和一旦符号/分支向量配错，
失败会伪装成"物理不符"——本单补上缺失的 3-D 测量仪并给出可判定的基准。

几何/参数（全部可精确解析，不涉及时间步/松弛）：
静态支撑柱阵列 m×m 层、层距 d=1、单粒质量 1、g=9.81；简单立方各向同性格（kn=2, δ=0.01, d=1）；
随机堆积取既有 `granular3d.jamming` 的 n=200、phi=0.62…0.68、seed=7。

## ② 怎么做的

新增 `src/tensordem/granular3d/stress.py`（float64 NumPy，与求解器解耦）：
体积平均 Cauchy 应力 `sigma=(1/V)Σ l⊗f`（拉为正、压为负，不强制对称）、
平面 traction/profile（**只计跨平面接触**，无分箱、无重复计数）、
平壁 traction `sigma_ab=(1/A)Σ f_a n_b`、
离散矩恒等式 `Σ l⊗f = Σ x⊗F_ext`、子区域力平衡（散度定理的力形式）、
动能项 `(1/V)Σ m v⊗v` 独立返回（不与接触 virial 隐式相加）。

基准全部是**静力学精确解**：柱阵列每接触力=其上方重量；简单立方格 `p=knδ/d²`；
压缩对 `sigma_zz=-F d/V`。另用 `SpherePacking.pressure()`（独立实现路径）交叉校核压力。
为证明检查非空洞，逐条注入 5 类经典实现 bug 并量化其对被判据量的影响。

## ③ 计算结果

符号钉扎（压缩对）：`sigma_zz` = {pin['sigma_zz']:.15g}，解析 {pin['analytic']:.15g}，绝对误差 {pin['abs_error']:.1e}；
`p` = {pin['pressure']:.15g} vs 解析 {pin['pressure_analytic']:.15g}。

静态柱平面通量剖面（m={col['m']}，{col['layers']} 层），逐平面：

| z/d | 平面通量 sigma_zz | 静力学解析 | 绝对差 |
|---|---|---|---|
{rows_col}

最大相对误差 **{col['max_rel_error']:.2e}**；跨平面接触数 {min(col['crossing_contacts'])}…{max(col['crossing_contacts'])}（预测 {col['crossing_contacts_expected']}）。

平壁 traction：floor `sigma_zz` = {wall['sigma_zz']:.10f} vs 解析 {wall['analytic']:.10f}（相对误差 {wall['rel_error']:.1e}）；
跨地面台阶 `{wall['step_across_floor']:.10f}` vs 底层重量应力 `{wall['bottom_layer_stress']:.10f}`。

压力 vs 独立路径（n=200 随机堆积）：

| phi | 接触数 | -tr(sigma)/3（本模块） | SpherePacking.pressure() | 相对差 |
|---|---|---|---|---|
{rows_pack}

矩恒等式残差 {data['moment_identity']['residual']:.2e}；旋转不变性 {rot['tensor_rel_error']:.2e}；
动能项解析误差 {data['kinetic_stress']['abs_error_vs_analytic']:.2e}。

注入 bug 的可检测性（被判据量的相对偏移）：

| 注入 bug | 相对偏移 |
|---|---|
{rows_mut}

## ④ 结果图

图1：静态柱的逐平面 sigma_zz 与静力学解析阶梯对比（无分箱，重合到 1e-16）。
图2：五类注入 bug 的相对偏移（log 轴），全部远超 1% 可检测下限——说明检查非空洞。

## ⑤ 注意点

- 本单验证的是**测量仪**，不是 DEM 物理：它不证明 Hertz/滚动阻力/接触求解器正确，只让喂进 mu(I)/Campbell 类算例的应力数可审计。
- 基准为静态构造（接触力由静力学给定），不涉及松弛与时间步，故可逐位复现；代价是**不含非均匀随机床的力分布**（静不定），该部分仍未覆盖。
- 平面切割只能定出 `sigma[axis, :]` 一行（平面属性，非实现缺陷）；完整张量由体积平均给出。
- 非中心接触力偶（滚动/钻孔，如 `clump_rolling`）不在对称应力契约内，返回未对称和，`sigma-sigma^T` 即诊断量。
- 尚未接线到消费应力的 3-D 算例（Campbell 弹限、mu(I) 直剪、双轴剪切带），它们此刻仍用私有求和——已写进缺口账 open 部分。

## ⑥ 误差判定

| 判据量 | 基准 | 误差 | 是否<3% |
|---|---|---|---|
| 压缩对 sigma_zz | 闭式 -F d/V | {pin['abs_error']:.1e}（绝对，量级 0.3） | 是 |
| 柱 sigma_zz(z) | 静力学 -m g N_above/d² | {col['max_rel_error']:.1e} | 是 |
| 地面 sigma_zz | -W_total/A | {wall['rel_error']:.1e} | 是 |
| 地面-通量台阶 | 底层重量/A | 1.4e-16 | 是 |
| -tr(sigma)/3 vs SpherePacking | 独立实现 | {max(p['rel_diff'] for p in packs):.1e} | 是 |
| 矩恒等式 | 精确 | {data['moment_identity']['residual']:.1e} | 是 |
| 旋转不变性 | 精确 | {rot['tensor_rel_error']:.1e} | 是 |

全部判据在舍入误差量级（<=6e-16），远优于 3%；也正因基准是精确解，报告附上注入 bug 敏感性表作为非空洞证据（0.83…3.0 相对偏移）。

代码与工件：commit 45061898；`src/tensordem/granular3d/stress.py`、`tests/test_granular3d_stress.py`（16 例）、
`scripts/verify_granular3d_stress.py`、`artifacts/granular3d_stress/`、`docs/granular3d_stress_verification.md`。
"""

body = {
    "title": "TensorDEM 3-D Virial 应力测量仪 + 静力学精确基准（granular3d.stress）",
    "summary": "补齐 3-D 缺失的应力测量面（体积平均/平面通量/壁面/离散平衡恒等式），对静力学精确解与独立 SpherePacking 路径全部 <=6e-16；5 类注入 bug 偏移 0.83~3.0 证明检查非空洞。",
    "scenario_key": "tensordem-granular3d-stress-instrument",
    "mission_id": 16650,
    "tags": ["TensorDEM", "DEM", "virial-stress", "verification", "granular3d", "selfstart"],
    "markdown": md,
    "figures": [
        fig("fig_profile.png", "静态柱：逐平面 sigma_zz 平面通量 vs 静力学解析阶梯（max rel err 6.0e-16）"),
        fig("fig_mutation.png", "注入 5 类经典实现 bug 的相对偏移（log 轴，全部远超 1% 检测下限）"),
    ],
}
print(json.dumps(post("/tensor/reports", body))[:600])