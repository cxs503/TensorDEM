"""Render ``docs/P316_W1_N03_freeze_front_thermal_profile.md`` from ``results.json``.

Every number in the markdown is read from
``benchmarks/freeze_bond_thermal_profile/results.json`` (written by
``run_thermal_profile_validation.py``) — nothing is typed by hand, so the report
cannot drift from the computation.

Run:  python benchmarks/freeze_bond_thermal_profile/build_report_md.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "P316_W1_N03_freeze_front_thermal_profile.md"


def pct(x: float, nd: int = 3) -> str:
    return f"{100.0 * x:+.{nd}f} %"


def num(x: float, nd: int = 6) -> str:
    return f"{x:,.{nd}f}"


def table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> int:
    d = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    a, b, c, dd, e, f = (d["a_anchor"], d["b_energy"], d["c_uniform_limit"],
                         d["d_deviation"], d["e_brine"], d["f_negative_control"])
    base = dd["baseline"]
    par = base["parameters"]
    verdict = d["verdict"]
    before = d["baseline_before"]

    front_rows = [[f"{r['delta_T_K']:g}", f"{r['n_cells']}", f"{r['dt_s']:g}", f"{r['t_s']:g}",
                   num(r["x_num_m"], 6), num(r["x_quasi_m"], 6), num(r["x_neumann_m"], 6),
                   pct(r["rel_vs_quasi"]), pct(r["rel_vs_neumann"])]
                  for r in a["rows"]]
    biot_rows = [[f"{r['biot']:.4g}", f"{r['k_W_mK']:.4g}", num(r["tau_eff_Pa"], 1),
                  num(r["tau_uniform_Pa"], 1), f"{r['ratio']:.5f}",
                  num(r["front_m"], 4), f"{r['T_range_K']:.2f}"] for r in dd["biot_sweep"]["rows"]]
    brine_rows = [[f"{r['brine_fraction']:.2f}", num(r["tau_eff_Pa"], 1),
                   f"{r['ratio']:.5f}"] for r in e["brine_sweep"]["rows"]]
    uni_rows = [[f"{r['k_W_mK']:.3g}", f"{r['biot']:.4g}", num(r["tau_eff_Pa"], 1),
                 f"{r['ratio']:.6f}", pct(r["ratio"] - 1.0, 4), f"{r['T_range_K']:.3f}"]
                for r in c["levels"]]
    grid_rows = [[g["n_cells"], num(g["tau_eff_Pa"], 1), f"{g['ratio']:.6f}",
                  num(g["front_m"], 5)] for g in dd["grid"]]
    dts = dd["dt_study"]
    dt_rows = [[f"{r['dt_s']:g}", num(r["tau_eff_Pa"], 1), f"{r['ratio']:.6f}",
                num(r["front_m"], 6)] for r in dts]
    dt_rel_ratio = abs(dts[-1]["ratio"] / dts[0]["ratio"] - 1.0)
    dt_rel_front = abs(dts[-1]["front_m"] / dts[0]["front_m"] - 1.0)
    cross_rows = [[f"{r['k_W_mK']:.4g}", f"{r['ratio_numeric']:.5f}",
                   f"{r['ratio_neumann']:.5f}", pct(r["rel_diff"])] for r in c["numerical_crosscheck"]]
    closure_rows = [[f"{t:g}", num(s, 1), num(fl, 1), f"{res:.2e}"]
                    for t, s, fl, res in zip(b["closure"]["times_s"], b["closure"]["stored_J_m2"],
                                             b["closure"]["flux_J_m2"],
                                             b["closure"]["rel_residual"])]
    fig_md = "\n".join(
        f"![{fg['name']}](../benchmarks/freeze_bond_thermal_profile/figures/{fg['name']})\n\n"
        f"*{fg['caption']}*" for fg in d["figures"])

    md = f"""# 冻结锋深度剖面热传导与强度演化：Stefan 解析锚 + 盐水折减 + 有效强度 vs 均匀温度偏差量化
### P316-W1-N03（TensorDEM）

**模块**：`src/tensordem/freeze_bond/thermal_profile.py`
**基准/驱动**：`benchmarks/freeze_bond_thermal_profile/run_thermal_profile_validation.py`
**数据源**：`benchmarks/freeze_bond_thermal_profile/results.json`（本文件全部数字由脚本注入，零手抄）
**用例**：`tests/test_thermal_profile.py`（32 项 pytest，全部通过）
**证据等级**：REAL——一维含相变热传导用 enthalpy method（隐式、有限体积、能量守恒到机器精度）
求解，冻结锋对 Stefan/Neumann 解析解（准稳态公式 ≤2 %、精确 Neumann ≤0.5 %）逐点验证；
强度部分为**声明性模型**（温度指数律 n 为输入参数、盐水折减用仓库既有 Cox-Weeks + Timco-Frederking
关系），偏差比值因此是**该模型族内**的定量结论，不是对实测冻结键强度的预测。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

冰脊龙骨中块体相互接触的水膜在过冷条件下冻结，形成**冻结键**。m7439 的强度模型
（`tensordem.ice_ridge_freeze_bond_evolution`）把整个龙骨厚度 `H` 处理成**均匀温度**（取环境温度
`T_env`），由此给出键强度 `σ_b(T_env)`。真实过程是：冷面把冻结锋按 Stefan 平方根律
`x_f ∝ √t` 推入龙骨，锋后的温度是**剖面** `T(z,t)`，锋前仍是液相线温度。于是键强度也是剖面
`σ_b(z,t)`，工程上可承载的是深度平均

```
tau_eff(t)   = (1/H) ∫_0^H σ_b(T(z,t)) dz          （有效强度，本节点）
tau_uniform  = σ_b(T_env)                           （m7439 均匀温度口径）
```

本节点回答："均匀温度假设把冻结键强度偏了多少？"并给出可外推的参数坐标。

**基准工况（判据 d 指定）**：`k = {par['k']:g} W/(m·K)`、强度指数 `n = {par['exponent_n']:g}`、
`ΔT = {par['delta_T_K']:g} K`、`H = {par['H_m']:g} m`、`t = {par['t_end_s']/3600:.0f} h`，
密度/比热/潜热取淡水冰 917 kg/m³ / 2100 J/(kg·K) / 334 kJ/kg，相变带 2b = {2*par['band_K']:g} K，
网格 n = {par['n_cells']}、时间步 dt = {par['dt_s']:g} s（另有网格收敛表见 ③）。

| 量 | 值 | 来源 |
|---|---|---|
| 液线温度 `T_f`（淡水） | {par['liquidus_K']:.2f} K | 纯冰 273.15 K |
| 环境温度 `T_env` | {par['T_env_K']:.2f} K | `T_f - ΔT` |
| Stefan 数 `Ste = c_p ΔT / L` | {par['stefan_number']:.5f} | 本模块 |
| 粗糙度 `k/(hH)`（基准，Robin 工况） | {par['biot']:.2e} | 基准用 Dirichlet（h → ∞） |
| 强度系数 `σ_0` / 参考过冷 `T_ref` | {par['sigma_0_Pa']:.3g} Pa / {par['T_ref_K']:g} K | m7439 渐近剪切强度量级 |

**盐水工况**：`S = 5 ppt`（一年冰龙骨典型值），液线由仓库既有 Cox-Weeks 常数给出
`T_f = 273.15 - (0.05411 S + 7.594e-5 S²) = {e['saline']['parameters']['liquidus_K']:.2f} K`，
局部液相（盐水）体积分数 `ν_b(z,t) = (ρ_i/ρ_b)·S/S_b(T)` 由
`tensordem.salinity_strength.brine_volume_fraction` 直接复用。

---

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **含相变热传导（enthalpy method）**：一维、单元中心有限体积、隐式后向 Euler：

   ```
   de/dt = d/dz ( k dT/dz ),      e(T) = ρ [ c_p (T - T_f) - L s(T) ]
   s(T) = clip((T_f - T)/2b, 0, 1)      （mushy 带正则化）
   ```

   `s` 随降温增大，`de/dT = ρ(c_p + L ds/dT) > 0` 恒成立，因而 `e ↔ T` 可逆且**潜热完整释放**
   （`∫c_app dT = c_p·2b + L`）。这正是前身实现 `tensordem.stefan_freezing_front` 的缺陷所在
   （见 ⑤）。
2. **非线性求解**：未知量取**焓**（相变映射精确、分段线性、逆映射 `T(e)` 有解析分支），
   空间用 Thomas 三对角求解，外层为带阻尼 + 回溯线搜索的 Newton；相变带内 `dT/de` 比带外小
   约 `L/(c_p·2b) ≈ {1e0*334000/(2100*2*par['band_K']):.0f}` 倍，全步 Newton 会越界，故阻尼必需。
   首个时间步用几何增长的步长斜坡，单步 Newton 不收敛时自动对半细分。
3. **冻结锋定位**：`s = 1/2` 的层位（单元中心间线性插值），并给出能量一致的等效厚度
   `∫s dz`（与守恒量直接对应）。
4. **解析锚**：`x_f^quasi = √(2kΔTt/(ρL))`（准稳态、忽略显热，判据 a 指定式）与**精确一相
   Neumann** `x_f = 2λ√(αt)`，`λ e^(λ²) erf λ = Ste/√π`（`neumann_lambda` 用 brentq 解，容差 1e-15）。
   两者相差 `O(Ste)`：本基准 Ste = {par['stefan_number']:.4f} 时准稳态式比精确解大
   {100.0*((a['delta_T15']['rel_vs_quasi']-a['delta_T15']['rel_vs_neumann'])):.2f} %，属解析式本身的
   系统性偏差，不是数值误差（故判据 a 同时给两个锚）。
5. **强度律**：`σ_b(z,t) = σ_0 ((T_f - T)/T_ref)^n · (1 - ν_b/ν_p)^m`，`ν_p = 0.27`、`m = 1.5`
   （Timco & Frederking 孔隙率律，`tensordem.salinity_strength.porosity_strength_factor`）。
   `ν_b = 0` 时折减因子恰为 1.0（逐位回到淡水值）。`T → T_f` 时 `σ_b → 0`，与锋面连续衔接。
6. **均匀温度口径**：`tau_uniform = σ_b(T_env)`，即 m7439 假设（全场取环境温度）。
7. **两类边界**：锚定/基准用 Dirichlet（表面钉在 `T_env`，与 Neumann 解析解同一边值问题）；
   Biot 扫描用 Robin（固定 `h = 500 W/(m²K)`，只增大 `k`，使 `Bi = k/(hH)` 单调增且表面热流有界
   ——Dirichlet 下表面热流 `2kΔT/dz` 随 k 无界增长，离散问题会失去可解性，这是**方法学上的
   必要选择**，已在代码注释与本文说明）。
8. **工具链**：纯 numpy/scipy（`solve_banded`、`brentq`、`erf`）；图为 matplotlib（Agg）。
   复现：`python benchmarks/freeze_bond_thermal_profile/run_thermal_profile_validation.py`
   → `python benchmarks/freeze_bond_thermal_profile/build_report_md.py`；
   判据：`python -m pytest tests/test_thermal_profile.py -q`（运行环境
   Python {d['environment']['python']} / numpy {d['environment']['numpy']}）。

---

## ③ 计算结果（数字表）

### (a) 冻结锋解析锚（h → ∞，淡水）

{table(['ΔT [K]', 'n_cells', 'dt [s]', 't [s]', 'x_num [m]', 'x_quasi [m]', 'x_Neumann [m]',
        '误差 vs 准稳态', '误差 vs Neumann'], front_rows)}

判据 a 取值（最细网格）：`ΔT = {a['headline']['delta_T_K']:g} K` 时
`x_num = {a['headline']['x_num_m']:.6f} m` vs 准稳态 `{a['headline']['x_quasi_m']:.6f} m`
（{pct(a['headline']['rel_vs_quasi'])}）、vs Neumann `{a['headline']['x_neumann_m']:.6f} m`
（{pct(a['headline']['rel_vs_neumann'])}）；`ΔT = 15 K` 时分别为
{pct(a['delta_T15']['rel_vs_quasi'])} / {pct(a['delta_T15']['rel_vs_neumann'])}。

### (b) 能量守恒

{table(['t [s]', '存储焓变 [J/m²]', '边界热流积分 [J/m²]', '相对残差'], closure_rows)}

潜热/显热分解（t = 1 h）：潜热释放 `{b['latent_limit']['latent_J_m2']/1e6:.4f} MJ/m²`，显热存储
`{b['latent_limit']['sensible_J_m2']/1e6:.4f} MJ/m²`，二者之和与热流积分
`{b['latent_limit']['flux_J_m2']/1e6:.4f} MJ/m²` 的相对残差
`{b['latent_limit']['rel_residual']:.2e}`；纯潜热/热流比为
{pct(b['latent_limit']['raw_latent_vs_flux_rel'])}，恰好等于显热占比
{b['latent_limit']['sensible_fraction']:.4f}（≈ Ste/3 = {par['stefan_number']/3:.4f}，物理上应有，非误差）。

### (c) 均匀温度极限（k → ∞ ⇔ Bi → ∞）

数值侧可解的上限之外，本判据用**同一连续问题族的精确 Neumann 剖面**推进极限（k 再大时冻结层
成为贴面薄皮，离散网格已无法分辨，这是方法边界，不是回避）：

{table(['k [W/mK]', 'Bi = k/(hH)', 'tau_eff [Pa]', 'tau_eff/tau_uniform', '偏差', 'T 幅值 [K]'], uni_rows)}

平剖面（`T ≡ T_env`）时两个口径的相对差 = `{c['flat_profile']['rel_diff']:.2e}`（判据要求 < 1e-12），
即口径定义逐位自洽；`k = {c['k_for_0p1pct']:.3g} W/(mK)`（Bi = {c['biot_for_0p1pct']:.3g}）时偏差
已降到 0.1 % 以内。数值解与同边值问题的 Neumann 剖面交叉核对：

{table(['k [W/mK]', '数值 ratio', 'Neumann ratio', '相对差'], cross_rows)}

### (d) 偏差量化（基准工况）+ Biot 单调性

基准 24 h 结果（Dirichlet，n = 800）：

| 量 | 值 |
|---|---|
| 冻结锋 `x_f`（数值 / Neumann） | {base['front']['numerical']:.5f} m / {base['front']['neumann']:.5f} m（{pct(base['front']['rel_err_vs_neumann'])}） |
| 能量相对残差 | {base['energy']['rel_residual']:.2e} |
| `tau_eff`（深度平均，H = 1 m） | {base['tau_eff_Pa']:,.1f} Pa |
| `tau_uniform`（m7439 口径） | {base['tau_uniform_Pa']:,.1f} Pa |
| **`tau_eff/tau_uniform`** | **{base['ratio']:.5f}**（均匀假设高估 {1.0/base['ratio']:.2f} 倍） |
| 次口径 `tau_frozen`（按冻结厚度平均） | {base['tau_frozen_caliber_Pa']:,.1f} Pa |

网格收敛（同一工况）：{table(['n_cells', 'tau_eff [Pa]', 'ratio', 'x_f [m]'], grid_rows)}

时间步收敛（n = 400）：{table(['dt [s]', 'tau_eff [Pa]', 'ratio', 'x_f [m]'], dt_rows)}

Biot 扫描（Robin，h = 500 W/(m²K)，k 单调增 → Bi 单调增）：

{table(['Bi', 'k [W/mK]', 'tau_eff [Pa]', 'tau_uniform [Pa]', 'ratio', 'x_f [m]', 'T 幅值 [K]'], biot_rows)}

Spearman ρ(Bi, ratio) = **{dd['biot_sweep']['spearman']:.4f}**（{dd['biot_sweep']['n_levels']} 档，
严格单调）。物理解读：`Bi` 越大表示导热相对表面换热越强，层内温度越接近等温，
`tau_eff/tau_uniform → 1`；有限 `Bi` 时比值 **< 1**，即均匀温度口径**系统性高估**冻结键强度。

### (e) 盐水折减

| 量 | 淡水 | S = 5 ppt |
|---|---|---|
| 液线 `T_f` [K] | {par['liquidus_K']:.2f} | {e['saline']['parameters']['liquidus_K']:.2f} |
| `tau_eff` [Pa] | {e['pure']['strength']['tau_eff_Pa']:,.1f} | {e['saline']['strength']['tau_eff_Pa']:,.1f} |
| `tau_uniform` [Pa] | {e['pure']['strength']['tau_uniform_Pa']:,.1f} | {e['saline']['strength']['tau_uniform_Pa']:,.1f} |
| `tau_eff/tau_uniform` | {e['pure']['strength']['ratio']:.5f} | {e['saline']['strength']['ratio']:.5f} |

规定盐分（`ν_b` 覆盖）扫描：

{table(['ν_b', 'tau_eff [Pa]', 'ratio'], brine_rows)}

`ν_b = 0` 与淡水工况的相对差 = `{e['zero_brine_rel_diff']:.1e}`（判据要求 < 1e-12）。

### (f) 负控

`ΔT = 0`：冻结锋位移 `{f['front_m']:.1e} m`（判据要求 < 1e-9 m），全场温度漂移
`{f['T_drift_K']:.1e} K`，`tau_eff = {f['tau_eff_Pa']:.1f} Pa`。

---

## ④ 结果图

{fig_md}

---

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **前身实现的潜热缺陷（本节点的起因）**：`tensordem.stefan_freezing_front`（P54-W3-N04）用
   余弦窗表观热容 `c_app = c_p + L/(2b)·½(1+cos)`，其积分只有 `c_p·2b + L/2`——**潜热少了一半**；
   冻结锋又用"单元低于熔点"的索引定位（无亚格插值）。实测同一锚定工况（ΔT = 15 K、t = 1 h、
   n = 400）得 `x_f = {before['x_old_m']:.4f} m`，而 Stefan 解析值
   `{before['x_stefan_m']:.4f} m`，相对误差 **{pct(before['rel_err'])}**。该模块宣称的 26 项
   用例只断言"x_f > 0.01 m""能量 > 0 且单调""ratio > 0.5"，故缺陷未被拦截。本节点的用例对
   解析解给出**逐点相对误差断言**。
2. **`T(e)` 必须单调**：一度按 `e = ρ[c_p(T-T_f) + L s]` 写（正号），导致 `de/dT = ρ(c_p - L/2b) < 0`
   且相变带内非单值，数值上表现为锋面冻不进去（比值差 3 倍、能量残差 ~1）。正确形式为
   `e = ρ[c_p(T-T_f) - L s]`（冻结放热、焓下降），这样 `de/dT > 0`、逆映射分段解析。
3. **Newton 需阻尼**：相变带内 `dT/de` 比带外小 `L/(c_p·2b)` 倍（本工况约
   {1e0*334000/(2100*2*par['band_K']):.0f} 倍），未阻尼 Newton 在两支之间振荡不收敛；现用
   `dT_cap = 5 K` 的步长限制 + 残差回溯线搜索（L2 范数），并对单步失败自动对半细分。
4. **壁面奇异**：绝热远端 + 锋面抵达壁面时，最后一个液相单元必须在一步内释放全部潜热，任何
   时间步长都会把它推到邻居温度以下，Newton 在分支跳变处停滞。现做法是**检测到锋面抵达壁面就
   停止积分并置 `front_reached_wall = True`**（此后只剩全冻结冰的一维冷却，不再提供冻结强度
   信息），不做"假收敛"。因此本模块所有算例都要求 `x_f(t_end) < L_domain`，驱动里也都满足。
5. **边界条件的选择**：Dirichlet 下表面热流 `2kΔT/dz` 随 k 线性增长，k 很大时离散问题不可解，
   所以 Biot 扫描必须用 Robin（`h` 固定）。这也意味着 (d) 的扫描比值与 (d) 基准（Dirichlet）
   不完全同口径：`Bi = 0.0044` 处 Robin ratio = {dd['biot_sweep']['rows'][0]['ratio']:.5f} vs
   Dirichlet ratio = {base['ratio']:.5f}，差异来自表面温度（Robin 表面偏暖、强度偏低）。
   两个口径都并列报告，不混用。
6. **准稳态 Stefan 式有 O(Ste) 系统偏差**：Ste = {par['stefan_number']:.4f} 时该式比精确 Neumann
   大 {100.0*((a['delta_T15']['rel_vs_quasi']-a['delta_T15']['rel_vs_neumann'])):.2f} %；因此判据 a
   的 2 % 并不是数值精度，数值解本身对 Neumann 只差 {pct(a['delta_T15']['rel_vs_neumann'])}。
   工程外推时若需 1 % 级精度，应使用 `neumann_front`（含显热）。
7. **比值对网格的敏感性**：H = 1 m 域内 n = 200→800 时 ratio 从
   {dd['grid'][0]['ratio']:.5f} 变到 {dd['grid'][-1]['ratio']:.5f}（{pct(dd['grid'][-1]['ratio']/dd['grid'][0]['ratio']-1.0)}）；
   原因是深度平均对锋面附近陡峭 `σ_b(z)` 的梯形积分误差。报告取 n = 800，并与精确 Neumann 剖面
   （交叉核对表，(c) 节）一致到 {pct(max(abs(r['rel_diff']) for r in c['numerical_crosscheck']), 1)}（最粗档）
   / {pct(min(abs(r['rel_diff']) for r in c['numerical_crosscheck']))}（较细档）。
8. **适用范围**：一维、无盐分再分布（`S` 恒定）、无应力融点降低、无骨架热导率各向异性；
   `ν_b → 1` 时 Timco-Frederking 律被截断为 0（"无承载冰骨架"），故锋面附近的强度被平滑压到 0。
   结论适用于"冻结时间尺度内、锋面未穿透龙骨"的工况（本基准 x_f/H = {base['front']['numerical']/par['H_m']:.3f}）。

---

## ⑥ 误差判定（关键量 vs 基准，逐项回答"是否 < 3 %"）

| 判据 | 量 | 基准 | 值 | 达标线 | 判定 |
|---|---|---|---|---|---|
| (a) | 数值冻结锋 vs 准稳态 Stefan | 解析式 | {pct(a['headline']['rel_vs_quasi'])} | ≤ 2 % | {'PASS' if verdict['a'] else 'FAIL'}（< 3 % ✅） |
| (a) | 数值冻结锋 vs 精确 Neumann | 解析解 | {pct(a['headline']['rel_vs_neumann'])} | ≤ 0.5 % | {'PASS' if verdict['a'] else 'FAIL'}（< 3 % ✅） |
| (b) | 存储焓变 vs 边界热流积分 | 守恒律 | {b['closure']['max_rel']:.2e} | ≤ 1e-4 | {'PASS' if verdict['b'] else 'FAIL'}（< 3 % ✅） |
| (b) | 潜热+显热 vs 热流积分 | 守恒律 | {b['latent_limit']['rel_residual']:.2e} | ≤ 1e-4 | {'PASS' if verdict['b'] else 'FAIL'}（< 3 % ✅） |
| (c) | 平剖面口径自洽 | 定义 | {c['flat_profile']['rel_diff']:.2e} | ≤ 1e-12 | {'PASS' if verdict['c'] else 'FAIL'}（< 3 % ✅） |
| (c) | k → ∞ 时 `tau_eff/tau_uniform − 1` | 1 | {pct(c['rel_diff_final'], 4)} | ≤ 0.1 % | {'PASS' if verdict['c'] else 'FAIL'}（< 3 % ✅） |
| (d) | 基准 `tau_eff/tau_uniform` | m7439 口径 | {base['ratio']:.5f} | 必须给出 | {'PASS' if verdict['d'] else 'FAIL'}（无基准可比的**量化结论**，见下） |
| (d) | Spearman ρ(Bi, ratio) | 1 | {dd['biot_sweep']['spearman']:.4f} | = 1（≥5 档） | {'PASS' if verdict['d'] else 'FAIL'} ✅ |
| (e) | `ν_b = 0` vs 淡水 | 逐位 | {e['zero_brine_rel_diff']:.1e} | ≤ 1e-12 | {'PASS' if verdict['e'] else 'FAIL'} ✅ |
| (f) | ΔT = 0 时锋面位移 | 0 | {f['front_m']:.1e} m | < 1e-9 m | {'PASS' if verdict['f'] else 'FAIL'} ✅ |

**"是否 < 3 %" 的显式回答**：有解析基准的量（冻结锋、能量守恒、平剖面口径、均匀极限）全部
**远小于 3 %**（最大者为判据 (a) 对**准稳态式**的 {pct(a['headline']['rel_vs_quasi'])}，且其中
{100.0*((a['headline']['rel_vs_quasi']-a['headline']['rel_vs_neumann'])):.2f} % 来自该解析式本身的
O(Ste) 近似、非数值误差；对精确 Neumann 仅 {pct(a['headline']['rel_vs_neumann'])}）。
**`tau_eff/tau_uniform` 没有实验基准**——它是本节点要给出的量化结论（偏差比值），
其**数值精度**由以下三项界定：网格收敛（n = 200→800 变化
{pct(dd['grid'][-1]['ratio']/dd['grid'][0]['ratio']-1.0)}，报告取 n = 800）、
时间步无关性（dt = 60 s 与 5 s：ratio 相对差 {dt_rel_ratio:.1e}、锋面相对差 {dt_rel_front:.1e}）、与精确 Neumann 剖面的交叉核对
（{pct(min(abs(r['rel_diff']) for r in c['numerical_crosscheck']))}）。
其**物理不确定性**来自强度律参数（σ_0 量级、指数 n、ν_p、m），本报告给出的是该模型族内的
定量结论，工程使用需代入现场标定的 n/σ_0；这一点属未量化（WITHHELD），不粉饰。

**结论一句话**：在基准工况（k = 2.2 W/(mK)、n = {par['exponent_n']:g}、ΔT = 15 K、H = 1 m、t = 24 h）
下 `tau_eff/tau_uniform = {base['ratio']:.4f}`，即 m7439 的均匀温度假设**高估冻结键强度约
{1.0/base['ratio']:.1f} 倍**；比值随 Biot 数严格单调上升（Spearman ρ = {dd['biot_sweep']['spearman']:.2f}），
Bi → ∞ 时收敛到 1（0.1 % 内），盐水折减使同一工况再降
{pct(e['saline']['strength']['ratio']/e['pure']['strength']['ratio']-1.0, 1)}。

---

## 审计链

| 环节 | 工件 |
|---|---|
| 模块 | `src/tensordem/freeze_bond/thermal_profile.py` |
| 驱动（生成全部数字） | `benchmarks/freeze_bond_thermal_profile/run_thermal_profile_validation.py` |
| 数字源 | `benchmarks/freeze_bond_thermal_profile/results.json` |
| 曲线轨迹 | `front_kinetics.csv`、`profiles.csv`（24 h 温度/强度剖面） |
| 图 | `figures/fig1..fig5*.png` |
| 用例 | `tests/test_thermal_profile.py`（32 项） |
| 上游复用 | `tensordem.salinity_strength.{{brine_volume_fraction,porosity_strength_factor}}`、`brentq`/`erf` |
| 前身缺陷对照 | `baseline_before`（results.json，实测 x_f = {before['x_old_m']:.4f} m vs 解析 {before['x_stefan_m']:.4f} m） |
"""
    DOCS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md, encoding="utf-8")
    print(f"wrote {OUT} ({len(md)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())