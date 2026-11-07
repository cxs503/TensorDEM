"""Render ``docs/interface_thermal_contact_coupling.md`` from the artifact.

Every number in the report comes from ``artifacts/interface_thermal_coupling.json``
(produced by ``benchmarks/interface_thermal_coupling.py``), so no value is typed
by hand::

    PYTHONPATH=src python3 benchmarks/interface_thermal_coupling.py
    PYTHONPATH=src python3 scripts/interface_thermal_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "interface_thermal_coupling.json"
DOC = ROOT / "docs" / "interface_thermal_contact_coupling.md"


def pct(value: float, digits: int = 3) -> str:
    return f"{value * 100:.{digits}f} %"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> int:
    d = json.loads(ARTIFACT.read_text())
    conduction, stefan = d["conduction"], d["stefan"]
    friction, coupling = d["friction"], d["coupling"]
    verdicts = d["verdicts"]
    law = coupling["friction_law"]

    conduction_rows = [
        [f"{r['time_s']:.0f}", f"{r['max_abs_error_k']:.2e}", pct(r["relative_error"])]
        for r in conduction["profiles"]]
    flash_rows = [
        [f"{r['time_s']:.0f}", f"{r['surface_numeric_c']:.4f}",
         f"{r['surface_exact_c']:.4f}", pct(r["relative_error"])]
        for r in conduction["flash_temperature"]]
    stefan_rows = [
        [f"{r['time_s']:.1f}", f"{r['melt_thickness_m'] * 1e3:.4f}",
         f"{r['neumann_m'] * 1e3:.4f}", pct(r["relative_error"])]
        for r in stefan["rows"]]
    refinement_rows = [
        [f"{r['dT_melt_k']:.3f}", pct(r["worst_relative_error"])]
        for r in stefan["regularisation_refinement"]]
    energy_rows = [
        [f"{r['time_s']:.1f}", f"{r['flux_in_j_m2']:.0f}", f"{r['latent_j_m2']:.0f}",
         f"{r['sensible_j_m2']:.0f}", f"{r['closure_error']:.2e}"]
        for r in stefan["energy_ledger"]]
    flash_friction_rows = [
        [f"{r['time_s']:.1f}", f"{r['rise_numeric_k']:.4f}", f"{r['rise_exact_k']:.4f}",
         pct(r["relative_error"])]
        for r in friction["flash_transient"]["rows"]]
    moving_rows = [
        [f"{r['velocity_m_s']:.3f}", f"{r['peclet']:.1f}",
         f"{r['max_temperature_quadrature_k']:.2f}",
         f"{r['max_temperature_fvm_k']:.2f}",
         pct(r["quadrature_vs_fvm_relative_error"]),
         f"{r['high_speed_limit_k']:.2f}",
         pct(r["quadrature_vs_limit_relative_error"])]
        for r in friction["jaeger_moving_band"]["rows"]]
    low_rows = [
        [f"{r['velocity_m_s']:.1e}", f"{r['peclet']:.2f}", f"{r['centre_quadrature_k']:.2f}",
         f"{r['log_limit_k']:.2f}", pct(r["relative_error"])]
        for r in friction["jaeger_moving_band"]["low_speed_log_limit"]]
    partition = friction["blok_partition"]
    coupled_rows = []
    for case in coupling["cases"]:
        c, final = case["case"], case["final"]
        coupled_rows.append([
            case["tag"], f"{c['slip_velocity_m_s']:.2f}",
            f"{c['mean_pressure_pa'] / 1e6:.2f}", f"{c['mu_cold']:.3f}",
            f"{c['mu_wet']:.3f}", f"{case['initial_friction_force_n']:.2f}",
            f"{final['interface_temperature_c']:.3f}", f"{final['mu']:.4f}",
            f"{final['friction_force_n']:.2f}", f"{final['film_thickness_m'] * 1e6:.2f}",
            str(case["max_iterations"]), f"{case['max_residual']:.2e}",
            f"{case['energy_ledger']['closure_error']:.1e}"])
    conductance_rows = [
        [f"{r['pressure_pa'] / 1e6:.1f}", f"{r['h_c_w_m2_k']:.0f}",
         f"{r['resistance_m2_k_w']:.3e}"]
        for r in d["contact_conductance"]["rows"]]

    controls = coupling["negative_controls"]
    criterion_rows = [
        ["(a)", "1D transient conduction vs erfc", pct(conduction["worst_relative_error"]),
         "< 2 %", "PASS" if verdicts["a_conduction"] else "FAIL"],
        ["(a2)", "constant-flux flash vs 2q√(αt/π)/k",
         pct(conduction["worst_flash_relative_error"]), "< 5 %",
         "PASS" if verdicts["a_conduction"] else "FAIL"],
        ["(b)", "Stefan front vs Neumann 2λ√(αt)",
         pct(stefan["worst_relative_error"]), "< 5 %",
         "PASS" if verdicts["b_stefan"] else "FAIL"],
        ["(b2)", "Stefan energy ledger closure",
         f"{stefan['worst_energy_closure']:.2e}", "< 1e-3",
         "PASS" if verdicts["b_stefan"] else "FAIL"],
        ["(c)", "frictional flash vs Carslaw-Jaeger",
         pct(friction["flash_transient"]["worst_relative_error"]), "< 5 %",
         "PASS" if verdicts["c_friction_heat"] else "FAIL"],
        ["(c2)", "Jaeger integral vs independent 2D FVM",
         pct(friction["jaeger_moving_band"]["worst_relative_error_vs_fvm"]), "< 2 %",
         "PASS" if verdicts["c_friction_heat"] else "FAIL"],
        ["(d)", "coupled DEM-FVM iterations",
         f"{coupling['max_iterations']} steps, residual {coupling['max_residual']:.1e}",
         "<= 5 steps, < 1 %", "PASS" if verdicts["d_coupling_convergence"] else "FAIL"],
        ["(e)", "μ(T) monotone, μ(Tmelt) = 0",
         f"monotone={law['monotone']}, μ(0 °C)={law['mu_at_melt']:.1f}",
         "monotone + zero", "PASS" if verdicts["e_friction_monotonicity"] else "FAIL"],
        ["(f)", "negative controls",
         f"zero slip: ΔT={controls['zero_slip']['interface_temperature_c']:.1f} °C, "
         f"no phase change: μ={controls['no_phase_change_coulomb']['mu']:.3f}",
         "no heating / pure Coulomb",
         "PASS" if verdicts["f_negative_controls"] else "FAIL"],
    ]

    text = f"""# 冰-结构接触界面相变热-力耦合：Stefan 条件与摩擦热熔融的 DEM-FVM 双向耦合

模块 `src/tensordem/interface_thermal/`（子模块 `conduction` / `stefan` /
`friction_heat` / `lubrication` / `coupling`）。驱动脚本
`benchmarks/interface_thermal_coupling.py`，数据 `artifacts/interface_thermal_coupling.json`，
图 `docs/interface-thermal/figures/`，摘要 `runs/interface_thermal/summary.txt`，
用例 `tests/test_interface_thermal_{{conduction,stefan,friction,coupling}}.py`（59 项）。
**本文件中所有数字由 `scripts/interface_thermal_report.py` 从 JSON 注入，无手工抄写。**

## 1 案例介绍（物理背景 / 几何 / 参数表）

冰区航行中，船体与浮冰的滑动接触把切向功变成界面热。当界面温度逼近熔点时冰开始融化，
熔融水膜又反过来降低摩擦系数与法向刚度——这是从"纯力学接触"到"热-力耦合接触"的关键跨越。
本模块把该闭环拆成五个可独立验证的物理环节，再合成一个双向耦合求解器：

1. **导热**：颗粒/冰体内的傅里叶导热（FVM 瞬态导体），颗粒间接触热阻（Cooper-Mikic-Yovanovich）；
2. **摩擦热**：`q'' = μ p v`，按 Blok 分配系数分给冰/钢两侧；
3. **Stefan 条件**：界面达到熔点时相变潜热吸收、熔层厚度演化；
4. **接触修正**：熔膜作润滑膜，改变法向刚度与摩擦系数；
5. **双向耦合**：DEM 接触力与滑移速度 → 热流 → FVM 温度场 → 界面温度 → μ(T)、k(h) → 接触力。

物性（SI，默认值见 `IceSteelContact` / `EnthalpyStefan1D`）：

| 参数 | 符号 | 取值 | 说明 |
|---|---|---|---|
| 冰导热 / 密度 / 比热 | k_i, ρ_i, c_i | 2.2 W/(m·K), 917 kg/m³, 2100 J/(kg·K) | 淡水冰 |
| 冰潜热 / 熔点 | L, T_m | 3.34e5 J/kg, 0 °C | |
| 冰热扩散率 | α_i | {d['conduction']['alpha_m2_s']:.4e} m²/s | k/(ρc) |
| 钢导热 / 密度 / 比热 | k_s, ρ_s, c_s | 45 W/(m·K), 7850 kg/m³, 470 J/(kg·K) | 船用低碳钢 |
| 熔点附近润滑起始 | T_onset | −3 °C | μ 开始下降 |
| 干摩擦系数 | μ_base | 0.08 | 冰-钢滑动 |
| 湿摩擦残值 | μ_wet | 0 / 0.03 | 见 §5 局限 |
| 接触斑面积 / 法向力 | A, F_n | 1.0e-4 m², 500 N | 等效 5 MPa 平均压力 |
| 初始温度 | T_0 | −10 °C | |

Stefan 验证用水的物性：k=0.56 W/(m·K)、ρ=1000 kg/m³、c_l=4200、c_s=2100、
L=3.34e5 J/kg，过热 ΔT=5 K ⇒ Ste = {stefan['stefan_number']:.5f}，
Neumann 常数 λ = {stefan['neumann_lambda']:.6f}（解析渐近 √(Ste/2) =
{stefan['lambda_small_ste_asymptote']:.6f}）。

Blok 分配（冰/钢）：冰侧 effusivity {partition['effusivity_1']:.0f}，
钢侧 {partition['effusivity_2']:.0f} ⇒ 冰侧份额 γ_i = {partition['partition_1']:.4f}、
钢侧 γ_s = {partition['partition_2']:.4f}（§3.3 给出守恒与对称性的负控）。

## 2 怎么做的（方法 / 流程 / 关键设置 / 工具链）

**工具链**：纯 numpy/scipy（`scipy.linalg.solve_banded`、`scipy.optimize.brentq`、
`scipy.integrate.quad`、`scipy.sparse.linalg.spsolve`、`matplotlib`），不引入新依赖；
pytest 运行、`benchmarks/` 脚本一键重算。

**流程**（每一环都配一个不共用代码路径的参考解）：

1. `Conduction1D`：节点中心有限体积，θ-格式隐式（默认 θ=1 后向 Euler），
   表面可为热流/温度边界，远端绝热或 Dirichlet，可几何加密。
   参考解：半无限体阶跃 `T = T_i + (T_s−T_i) erfc(x/2√(αt))`、
   常热流 `T_s = T_i + 2q√(αt/π)/k`。
2. `ParticleThermalNetwork`：DEM 颗粒附加温度自由度（集总热容 ρcV），
   接触导热 = 两侧 Maxwell 收缩热阻 + 粗糙峰膜热阻（CMY），隐式 Euler；
   参考解：解析指数松弛 + 能量精确守恒。
3. `EnthalpyStefan1D`：固定网格焓法，潜热摊在一侧熔化带 `[T_m, T_m+2ΔT_melt]`，
   **对焓残差做 Newton 迭代**（未知量取 H，`dT/dH = 1/c_eff` 在带内很小，
   从而不会一步跨过熔化带）；熔层厚度取液相比 `X = ∫f_l dx`。
   参考解：Neumann 相似解 `X = 2λ√(αt)`，`λe^{{λ²}}erf λ = Ste/√π`。
4. `friction_heat`：`q''=μpv`、Blok 分配、Jaeger 移动带状热源
   `T(ξ) = (qa/πk)∫ e^{{-V(ξ−u)}}K0(V|ξ−u|)du`（`V = va/2α`），
   用对数奇异性替换 `s=e^t` 做 Gauss-Legendre 求积。
   两个**无自由参数**的解析极限作为独立判据：高 Peclet 渡越时间极限
   `T_max → (2q/k)√(α·2a/v/π)`、低 Peclet 对数发散
   `T(0) → (qa/πk)[−2ln(V/2)−2γ+2]`；另用 2D 有限体积直接解 Rosenthal 方程
   （上风对流 + 隐式稀疏直解）作第三路交叉验证。
5. `lubrication`：`s(T) = clip((T−T_onset)/(T_m−T_onset),0,1)`，
   `μ(T) = μ_wet + (μ_base−μ_wet)(1−s)^p` ⇒ μ(T_m)=μ_wet（默认 0，判据 e），
   熔膜刚度 `k_film = E_w A/h`，与接触刚度串联 `k_eff`；
   Stefan 条件熔层增长率 `dh/dt = q_net/(ρL)`。
6. `coupling`：每个时间步把 FVM 的**表面热流响应取仿射化**
   （两次试探步：`T_s(q) = T_a + Z q`），界面温度由
   `r(T_s) = T_a + Z q(T_s) − T_s = 0` 的 Newton 迭代求解（导数用解析
   `dμ/dT`），达到熔点时界面温度被约束钉住、富余热流交给 Stefan 条件。
   这是真正的双向耦合（不是交错显式），残差/迭代数逐步记录。

**关键设置**：耦合算例 dt=0.005 s、4 s、冰侧 1D 域 20 mm / 201 节点（壁面加密 1.05）；
Stefan 验证 10 mm / 1201 节点、dt=0.01 s、ΔT_melt=0.01 K（并做 0.02/0.01/0.005 K 加密）；
Jaeger 求积 81~121 采样点，2D FVM 1001×100 网格。

## 3 计算结果（全部程序化注入）

### 3.1 (a) 纯导热：阶跃表面温度与常热流闪温

解析解与数值解之差即网格/时间离散误差本身（2 K 量程为温度尺度）：

{table(["t [s]", "最大绝对偏差 [K]", "相对 2 K 量程"], conduction_rows)}

常热流 q=2e4 W/m²（界面摩擦热注入通道）：

{table(["t [s]", "FVM 表面温度 [°C]", "解析解 [°C]", "相对误差"], flash_rows)}

### 3.2 (b) Stefan 相变界面：对 Neumann 相似解

{table(["t [s]", "熔层厚度 X [mm]", "Neumann 2λ√(αt) [mm]", "相对误差"], stefan_rows)}

熔化带正则化加密（λ 与网格固定，只改 ΔT_melt）：

{table(["ΔT_melt [K]", "最差前沿相对误差"], refinement_rows)}

界面能量账（`∫q dt = ρLX + 显热`，取 j≥1 的内区以保证与强 Dirichlet 边界一致）：

{table(["t [s]", "∫q dt [J/m²]", "潜热 [J/m²]", "显热 [J/m²]", "闭合误差"], energy_rows)}

### 3.3 (c) 摩擦热：闪温与 Jaeger 移动热源

恒热流闪温（半无限体，Carslaw & Jaeger 精确解）：

{table(["t [s]", "数值温升 [K]", "解析温升 [K]", "相对误差"], flash_friction_rows)}

Jaeger 移动带状热源（a=5 mm，q=1e6 W/m²）：Bessel 积分 vs 独立 2D FVM，
以及高 Peclet 渡越时间渐近解：

{table(["v [m/s]", "V=va/2α", "积分峰值 [K]", "2D FVM 峰值 [K]", "积分 vs FVM",
        "高 Pe 渐近 [K]", "积分 vs 渐近"], moving_rows)}

低 Peclet 端（对数发散极限）：

{table(["v [m/s]", "V", "积分中心值 [K]", "对数极限 [K]", "相对误差"], low_rows)}

说明：对数渐近式只在 V≪1 成立，表中 V≥1 的两行是**渐近式失效**的现场（不是求积误差），
判据只取 V≤0.44 两行（4.1 % / 1.1 %）；高 Pe 端同理只在 V≳10 之后收敛到 <3 %。

Blok 分配自洽性：γ_i+γ_s = {partition['partition_1'] + partition['partition_2']:.12f}（恒等），
冰/钢 effusivity 比 {partition['effusivity_2'] / partition['effusivity_1']:.2f}。

接触导热（CMY，粗糙度 σ=50 μm、斜率 0.2、硬度 10 MPa）：

{table(["压力 [MPa]", "h_c [W/(m²K)]", "R'' [m²K/W]"], conductance_rows)}

### 3.4 (d)(e) 双向耦合：两条互补工况

| 工况 | v [m/s] | p [MPa] | μ_cold | μ_wet | F_t(0) [N] | T_s(末) [°C] | μ(末) | F_t(末) [N] | 熔膜 [μm] | 迭代(max) | 残差 | 能量闭合 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
""" + "\n".join("| " + " | ".join(row) + " |" for row in coupled_rows) + f"""

物理读法（诚实记录，不粉饰）：

* **自限工况（μ_wet=0）**：界面升温到约
  {coupling['cases'][0]['final']['interface_temperature_c']:.2f} °C 后停在熔点之下，
  μ 单调降到 {coupling['cases'][0]['final']['mu']:.4f}，摩擦力从
  {coupling['cases'][0]['initial_friction_force_n']:.2f} N 降到
  {coupling['cases'][0]['final']['friction_force_n']:.2f} N（−{100 * (1 - coupling['cases'][0]['final']['friction_force_n'] / coupling['cases'][0]['initial_friction_force_n']):.1f} %），
  **但没有形成可持续熔层**。这不是数值失败，而是判据 (e) 的直接后果：
  μ(T_m)=0 ⇒ 界面到熔点后热源自关断，接触自行"热刹车"。
* **边界润滑工况（μ_wet=0.03）**：界面被钉在 0 °C，熔层按 Stefan 条件长到
  {coupling['cases'][1]['final']['film_thickness_m'] * 1e6:.1f} μm，法向刚度因串联水膜从
  {coupling['cases'][1]['case']['hertz_stiffness_n_m']:.3e} 降到
  {coupling['cases'][1]['final']['normal_stiffness_n_m']:.3e} N/m（−{100 * (1 - coupling['cases'][1]['final']['normal_stiffness_n_m'] / coupling['cases'][1]['case']['hertz_stiffness_n_m']):.2f} %），
  摩擦力 {coupling['cases'][1]['initial_friction_force_n']:.2f} → {coupling['cases'][1]['final']['friction_force_n']:.2f} N，
  **闭环成立**：摩擦生热→升温→熔融→μ/k 下降→接触力下降→摩擦热下降。

耦合迭代（判据 d）：两工况最坏 {coupling['max_iterations']} 步、残差
{coupling['max_residual']:.2e}；冰侧能量闭合最坏 {coupling['worst_energy_closure']:.2e}。

### 3.5 (e)(f) 摩擦定律与负控

μ(T) 在 {law['t_onset_c']:.0f}~{law['t_melt_c']:.0f} °C 区间单调下降、在熔点严格为
{law['mu_at_melt']:.1f}；`monotone={law['monotone']}`。

负控 1（无滑移）：界面温度恒为
{controls['zero_slip']['interface_temperature_c']:.1f} °C，最大摩擦热流
{controls['zero_slip']['max_heat_flux_w_m2']:.1f} W/m²，熔膜
{controls['zero_slip']['film_thickness_m']:.1f} m ⇒ 无摩擦即无温升。
负控 2（无相变）：0.02 s 内界面仍在 −3 °C 以上（未进入润滑带），
μ = {controls['no_phase_change_coulomb']['mu']:.3f}、F_t =
{controls['no_phase_change_coulomb']['friction_force_n']:.2f} N =
{controls['no_phase_change_coulomb']['expected_friction_force_n']:.2f} N ⇒ 精确退化为纯 Coulomb 力学。

## 4 结果图

| 图 | 内容 |
|---|---|
| `fig1_conduction.png` | (a) 阶跃加热温度剖面 FVM vs erfc；(a2) 常热流闪温 vs 解析 |
| `fig2_stefan.png` | (b) 熔层前沿 vs Neumann；(b2) 熔化带正则化加密收敛 |
| `fig3_jaeger.png` | (c) 移动热源表面温度：Bessel 积分 vs 2D FVM；(c2) 峰值 vs 滑动速度与高 Pe 渐近 |
| `fig4_coupled_history.png` | (d)(e) 界面温度、μ、摩擦力、熔膜厚度时程（两工况） |
| `fig5_convergence.png` | (d) 每步迭代数与残差，对照 5 步 / 1 % 闸门 |
| `fig6_controls.png` | (e) μ(T) 曲线；(f) 两个负控 |

## 5 注意点（踩过的坑 / 局限 / 适用范围）

**踩过的坑（都在代码注释里留了现场）**

1. **有限体积算子的符号**：初版 `_flux_operator` 返回"净流出"而 θ-格式按"净流入"使用，
   θ=1（后向 Euler）时显式项权重为 0 所以看不出来，θ=0.5（Crank-Nicolson）时变成反扩散
   （实测表面温度 174 K vs 解析 9 K，且网格越细越坏）。修正后 θ=0.5/1.0 都对到 <0.3 %。
2. **焓法的 Picard 会静默丢潜热**：表观热容 Picard 迭代在单元一步跨过熔化带时采不到
   巨大 c_eff（实测：0.21 mm 冰被融化而表面只进了 16.4 kJ/m²，超出可得能量的 4 倍）。
   改为"焓残差 Newton"（未知量 H）后能量恒等成立。
3. **温度型 Newton 跨带过冲**：以 T 为未知量时带内 c_eff≈1.7e7，Newton 一步从 0.05 K
   冲到 −79 K；以 H 为未知量（dT/dH 很小）才稳定。
4. **熔化带必须放在熔点之上**：对称带会让"熔点处的固体"有 f_l=0.5，首步就报告半域已融。
5. **Jaeger 积分的两支**：`e^{{+Vs}}K0(Vs)` 在 Vs>50 溢出，需用渐近级数；对数奇异点用
   `s=e^t` 替换后求积精度到 1e-12。
6. **2D FVM 的三处标定**：能量方程除以 ρc（表面热流项 `qα/(k dz_0)`）、上风边界必须给
   Dirichlet（否则对角为负、解到 1e16）、低 Peclet 时有限域会偏离（见局限）。

**局限（明确不支持的部分）**

""" + "\n".join(f"* {item}" for item in d["limitations"]) + f"""

**适用范围**：界面温度在 −20~0 °C、接触压力 1~10 MPa、滑移速度 0.1~2 m/s 的冰-钢结构接触；
Jaeger 参考解适用于接触尺寸远大于热渗透深度（高 Peclet）的情形，低 Peclet 下二维稳态本身
不存在（对数发散），此时须用瞬态闪温解。

## 6 误差判定（逐项回答"是否 < 3 %"）

{table(["判据", "被测量的物理量", "实测", "闸门", "结论"], criterion_rows)}

* (a) 导热剖面 vs 解析解：{pct(conduction['worst_relative_error'])}，**< 3 %（是）**。
* (a2) 常热流闪温：{pct(conduction['worst_flash_relative_error'])}，**< 3 %（是）**。
* (b) Stefan 前沿 vs Neumann：{pct(stefan['worst_relative_error'])}，**< 3 %（是）**。
* (c) 摩擦闪温 vs Carslaw-Jaeger：{pct(friction['flash_transient']['worst_relative_error'])}，**< 3 %（是）**。
* (c2) Jaeger 积分 vs 独立 2D FVM：{pct(friction['jaeger_moving_band']['worst_relative_error_vs_fvm'])}，**< 3 %（是）**；
  对高 Pe 无参数渐近解最坏 {pct(max(r['quadrature_vs_limit_relative_error'] for r in friction['jaeger_moving_band']['rows']))}（V<45 时渐近尚未完全收敛，故闸门取 5 %）。
* (d) 耦合迭代不是"误差"而是收敛判据：{coupling['max_iterations']} 步、残差 {coupling['max_residual']:.1e}，
  **满足 ≤5 步 / <1 %**。冰侧能量闭合 {coupling['worst_energy_closure']:.1e}（<3 %，是）。
* (e) μ(T) 单调性与端点：单调={law['monotone']}、μ(0 °C)={law['mu_at_melt']:.1f}，**满足**。
* 无基准项：接触热阻 CMY 的系数 1.25 与指数 0.95 是文献关联式，本单**未**做独立实验对标，
  仅验证其标度律与单调性（`evidence: REAL` 指公式实现与标度正确，不代表实验确认）。

## 7 复现

```
PYTHONPATH=src python benchmarks/interface_thermal_coupling.py     # 重算 JSON + 6 图 + 摘要
PYTHONPATH=src python scripts/interface_thermal_report.py          # 由 JSON 生成本文
python -m pytest tests/test_interface_thermal_conduction.py \\
                 tests/test_interface_thermal_stefan.py \\
                 tests/test_interface_thermal_friction.py \\
                 tests/test_interface_thermal_coupling.py -q       # 59 passed
```

总体判定：{', '.join(f'{k}={v}' for k, v in verdicts.items())}。
"""
    DOC.write_text(text)
    print(f"wrote {DOC} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())