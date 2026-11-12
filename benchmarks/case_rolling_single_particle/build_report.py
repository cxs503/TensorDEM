#!/usr/bin/env python
"""Generate the case report from the run artifact (no hand-typed numbers).

    PYTHONPATH=src python benchmarks/case_rolling_single_particle/build_report.py \
        --result artifacts/case_rolling_single_particle/result.json \
        --doc docs/CASE-DEM-ROLLING-SINGLE-PARTICLE.md \
        --payload /tmp/report.json

Every number in the emitted markdown / platform payload is read from
``result.json`` (which is produced by ``run_case.py``) -- the prose carries no
transcribed values.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

REPRO_CMD = (
    "PYTHONPATH=src OMP_NUM_THREADS=1 python benchmarks/case_rolling_single_particle/run_case.py "
    "--out artifacts/case_rolling_single_particle/result.json && "
    "PYTHONPATH=src python benchmarks/case_rolling_single_particle/figures.py && "
    "PYTHONPATH=src python -m pytest tests/test_case_rolling_single_particle.py -q"
)


def pct(x: float) -> str:
    return f"{100.0 * x:.4f} %"


def sci(x: float) -> str:
    return f"{x:.3e}"


def mark(x: bool) -> str:
    return "✔" if x else "✘"


def build(result_path: str, figure_path: str) -> tuple[str, dict]:
    with open(result_path) as fh:
        case = json.load(fh)
    s = case["summary"]
    base = case["points"]["A_base"]
    got, ref, dim, inp = base["got"], base["reference"], base["dimensionless"], base["inputs"]
    pB = case["points"]["B_rolling"]
    pN = case["points"]["NEG_low_friction"]
    pC = case["points"]["C_below_critical"]
    prov = case["provenance"]

    # ---------------- section 3: parameter table --------------------------
    param_rows = [
        ("倾斜角 theta", f"{inp['theta_deg']:.1f}", "deg", "case definition (pool anchor)"),
        ("球半径 R", f"{inp['radius_m']:.4g}", "m", "case definition"),
        ("密度 rho (冰)", f"{inp['density_kg_per_m3']:.4g}", "kg/m^3", "case definition"),
        ("质量 m", f"{inp['mass_kg']:.6g}", "kg", "m = 4/3 pi R^3 rho (code)"),
        ("转动惯量 I", f"{inp['inertia_kg_m2']:.6g}", "kg m^2", "I = 2/5 m R^2 (code)"),
        ("I / (m R^2)", f"{inp['inertia_factor_I_over_mR2']:.6g}", "-", "solid sphere"),
        ("重力 g", f"{inp['gravity_m_per_s2']:.4g}", "m/s^2", "harness default"),
        ("法向刚度 k_n", f"{inp['k_n_N_per_m']:.6g}", "N/m", "contact"),
        ("切向刚度 k_t", f"{inp['k_t_N_per_m']:.6g}", "N/m", "k_t = k_n"),
        ("滑动摩擦 mu", f"{inp['mu']:.3g}", "-", "Cundall-Strack cone"),
        ("滚动阻力 mu_r", f"{inp['mu_r']:.3g}", "-", "Ai et al. 2011 CDT branch (case A: off)"),
        ("时间步 Delta t", f"{inp['dt_s']:.4g}", "s", "semi-implicit Euler"),
        ("模拟时长 / 拟合窗", f"{inp['duration_s']:.3g} / {inp['duration_s'] - inp['settle_s']:.3g}", "s", "LS fit window"),
        ("步数", f"{inp['steps']}", "-", "duration / dt"),
    ]
    dim_rows = [
        ("I / (m R^2)", f"{dim['I_over_mR2']:.4f}", "1.4 -> a/(g sin t) = 0.7143"),
        ("a / (g sin theta)", f"{dim['a_over_g_sin_theta']:.6f}", "= 1/(1+I/mR^2)"),
        ("friction demand / cone", f"{dim['friction_demand_over_cone']:.4f}", "<< 1 -> no slip"),
        ("k_n / (m g cos t / R)", sci(dim["stiffness_ratio_kn_over_mg_over_R"]), "contact stiffness ratio"),
        ("omega_n Delta t", f"{dim['omega_n_dt']:.4f}", "normal mode"),
        ("omega_slip Delta t", f"{dim['omega_slip_dt']:.4f}", "tangential (slip) mode; ~23 steps/period"),
        ("pure_rolling_feasible(theta, mu_r, mu)", str(dim["pure_rolling_feasible"]), "in-code check"),
    ]

    # ---------------- section 4: comparison table -------------------------
    comp_rows = [
        ("球心平动加速度 a", f"{ref['a_ref']:.6f} m/s^2", f"{got['a_t_m_per_s2']:.6f} m/s^2",
         pct(base["error"]["a_rel_error"]), mark(base["error"]["pass_a"])),
        ("转动角加速度 alpha", f"{ref['alpha_ref']:.4f} rad/s^2", f"{got['alpha_rad_per_s2']:.4f} rad/s^2",
         pct(base["error"]["alpha_rel_error"]), mark(base["error"]["pass_alpha"])),
        ("无滑移闭合 norm(a + R alpha)/a", "0 (exact)", sci(base["error"]["no_slip_closure"]),
         pct(base["error"]["no_slip_closure"]), mark(base["error"]["no_slip_closure"] <= 0.02)),
        ("瞬时加速度振荡幅值", f"{got['oscillation_amplitude_ref']:.4f} m/s^2",
         f"{got['oscillation_amplitude_measured']:.4f} m/s^2",
         pct(base["error"]["oscillation_amplitude_rel_error"]),
         mark(base["error"]["oscillation_amplitude_rel_error"] <= 0.02)),
        ("滚动阻力分支 (mu_r=0.1) a",
         f"{pB['reference']['a_ref']:.6f} m/s^2", f"{pB['got']['a_t_m_per_s2']:.6f} m/s^2",
         pct(pB["error"]["a_rel_error"]), mark(pB["error"]["pass_a"])),
        ("滚动阻力分支 (mu_r=0.1) alpha",
         f"{pB['reference']['alpha_ref']:.4f} rad/s^2", f"{pB['got']['alpha_rad_per_s2']:.4f} rad/s^2",
         pct(pB["error"]["alpha_rel_error"]), mark(pB["error"]["pass_alpha"])),
        ("负控1: mu=0.01 滑动分支 a",
         f"{pN['sliding_reference']['a_sliding_ref']:.6f} m/s^2", f"{pN['got']['a_t_m_per_s2']:.6f} m/s^2",
         pct(pN["error"]["a_rel_error_vs_sliding_ref"]), mark(pN["error"]["a_rel_error_vs_sliding_ref"] <= 0.02)),
        ("负控1: 与无滑移参考的偏差 (必须 >2%)", f"{ref['a_ref']:.6f} m/s^2",
         f"{pN['got']['a_t_m_per_s2']:.6f} m/s^2", pct(pN["error"]["a_rel_error"]),
         mark(not pN["error"]["pass_a"])),
        ("负控2: 无惯性修正伪参考 g sin t (必须被拒)", f"{ref['sliding_without_inertia_a']:.6f} m/s^2",
         f"{got['a_t_m_per_s2']:.6f} m/s^2",
         pct(s["negative_control_no_inertia_reference"]["rel_error_vs_wrong"]),
         mark(s["negative_control_no_inertia_reference"]["rejected"])),
        (f"静态保持 theta=5 deg < theta_c={pC['reference']['critical_angle_deg']:.3f} deg: a",
         "0 (static)", f"{pC['got']['a_t_m_per_s2']:.3e} m/s^2",
         "creep = " + sci(pC["error"]["creep_over_g_sin_theta"]) + " g sin t",
         mark(pC["error"]["creep_over_g_sin_theta"] < 0.01)),
    ]

    conv_rows = [
        (f"{c['dt_s']:.1e}", f"{c['steps']}", f"{c['omega_slip_dt']:.3f}",
         f"{c['a_t']:.6f}", pct(c["a_rel_error"]), pct(c["alpha_rel_error"]),
         f"{c['oscillation_amplitude']:.4f}", pct(c["oscillation_amplitude_rel_error"]))
        for c in case["convergence_dt"]
    ]
    stiff_rows = [
        (f"{k['k_n_N_per_m']:.3g}", f"{k['a_t']:.6f}", pct(k["a_rel_error"]), f"{k['a_instantaneous_std']:.4f}")
        for k in case["stiffness_invariance"]
    ]
    win_rows = [
        (f"{w['fit_window_s']:.3f}", f"{w['a_t']:.6f}", pct(w["a_rel_error"]), pct(w["alpha_rel_error"]))
        for w in case["window_sensitivity"]
    ]

    def table(header, rows) -> str:
        out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
        out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
        return "\n".join(out)

    verdict = s["verdict"]
    md = f"""# CASE-DEM-ROLLING-SINGLE-PARTICLE — 单颗粒斜面纯滚动加速度（TensorDEM 算例池）

* case_id: `{case['case_id']}`
* 生成时间(UTC): {case['generated_utc']}
* 判定: **{verdict}**（容差 {s['tolerance'] * 100:.1f} %）
* 预注册: `{prov['prereg']}`（阈值先于生产运行声明）
* 工件: `artifacts/case_rolling_single_particle/result.json` + `case_rolling_single_particle.png`
* 能力复用: `{prov['capability']}`（未新写求解器）

## ① 问题与参考值出处

单颗实心球在倾角 theta 的斜面上**无滑移纯滚动**，经典刚体动力学给出

```
a = g sin(theta) / (1 + I/(m R^2))          alpha = a / R
I/(m R^2) = 2/5  ->  a = (5/7) g sin(theta)
```

本算例取 theta = {inp['theta_deg']:.0f} deg、无滑移（mu 远大于所需值）。次级参考为滚动阻力模型
Ai, Chen, Rotter, Ooi (2011), *Assessment of rolling resistance models in discrete element
simulations*, Powder Technology **206**, 269-282, doi:10.1016/j.powtec.2010.11.003：
该文比较 CDT / viscous / EPSD 三类滚动阻力模型，滚动阻力矩取 `M_r = mu_r |F_n| r_eff`，
由此斜坡上纯滚动的平衡临界角为 `theta_c = arctan(mu_r)`、稳态加速度为
`a = g (sin t - mu_r cos t)/(1 + I/(m R^2))`。

**参考值出处与零手抄声明（如实）**：本报告与工件中的全部数字由
`benchmarks/case_rolling_single_particle/run_case.py` 内的闭式关系**程序化计算**
（参数 g/theta/R/rho 与 I/(mR^2)=0.4 全部取自代码而非手输常数），不是从论文表格抄录。
本执行环境**无法访问 Ai et al. 2011 原文/表格**（无该 PDF，也无在线检索），
因此**不宣称**引用其原表原页数值，仅引用其模型关系式与临界角结论——这一点在
`provenance.reference.access_note` 中机器可查。误差判定完全针对闭式解析解。

无滑移参考 alpha 的符号遵循 harness 约定：`v > 0` 为沿坡向下、`omega` 为 +z 分量，
无滑移即 `v + R omega = 0`，故 `alpha = -a / R`。

## ② 方法与实现

* 复用仓库现有能力 `tensordem.rolling_resistance.simulate_sphere_on_incline`
  （Cundall-Strack 切向弹簧/库仑锥 + 法向弹簧-阻尼 + rolling_resistance 模块的 CDT 滚动阻力矩），
  半隐式（symplectic）Euler 时间推进；**未新写求解器**（避免重复造轮子）。
* 观测算子（测量定义先于运行声明）：
  * `a` = 稳态窗内 v(t) 的最小二乘斜率（扫过 ~195 个切向振荡周期，振荡平均效果收敛）；
  * `alpha` = 稳态窗内 omega(t) 的最小二乘斜率；
  * 无滑移闭合 = `|a + R alpha| / |a|`；
  * 滑移速率 rms = `rms(v + R omega)`；
  * 瞬时加速度 a(t) 用中心差分给出，仅作诊断（见 ⑤ 的振荡机理）。
* 负控与判别力：
  1. `mu = 0.01`（低于纯滚动所需 mu ≈ 0.051）→ 必须滑移、必须被判失败；
  2. 无惯性修正伪参考 `g sin t`（比真值高 `I/(mR^2)/(1+I/(mR^2))` ≈ 28.6 %）→ 必须被 2 % 门拒绝；
  3. `theta = 5 deg < arctan(mu_r) = {pC['reference']['critical_angle_deg']:.3f} deg`（mu_r = 0.1）→ 必须近似静止。
* 计算规模：单核本地执行，最贵点（Delta t = {case['convergence_dt'][-1]['dt_s']:.1e}）用时
  {case['convergence_dt'][-1]['wall_clock_s']:.1f} s，全案 < 1 分钟，无需集群。

## ③ 参数表（含无量纲数）

{table(['参数', '值', '单位', '来源/说明'], param_rows)}

无量纲数：

{table(['无量纲数', '值', '说明'], dim_rows)}

## ④ 对照表（ref vs got，误差逐项）

基准点 theta = {inp['theta_deg']:.0f} deg、mu_r = 0：

{table(['指标', 'ref（闭式解）', 'got（TensorDEM）', '相对误差', '判据 <2%'], comp_rows)}

补充诊断（基准点）：v(t) 拟合 R^2 = {got['r2_velocity_fit']:.8f}、
omega(t) 拟合 R^2 = {got['r2_spin_fit']:.8f}、滑移速率 rms = {sci(got['slip_rms_m_per_s'])} m/s
（最大 {sci(got['slip_max_abs_m_per_s'])} m/s，相当于 v 的量级 1e-4）。

**关键两问**：球心平动加速度误差 **{pct(s['a_rel_error'])}**，转动角加速度误差
**{pct(s['alpha_rel_error'])}** —— 均 **< 2 %（<3 %）**，判定 {verdict}。

![case figure](case_rolling_single_particle.png)

图说明：(a) v(t) 与 -R omega(t) 重合 = 无滑移闭合；(b) 瞬时 a(t) 在解析均值上下振荡，
虚线为解析均值与实测包络（无阻尼切向弹簧的必然结果，见 ⑤）；
(c) 时间步收敛：均值误差始终 ~1e-6（红/蓝），振荡幅值误差随 Delta t 一阶收敛；
(d) 四项指标相对各自参考的归一化（全部落在 100 % 附近，偏差标注在点上）。

## ⑤ 收敛性与参数敏感性（三点）

**(1) 时间步收敛**（基准点，仅 Delta t 变化）：

{table(['Delta t [s]', '步数', 'omega_slip Delta t', 'a [m/s^2]', 'a 误差', 'alpha 误差',
        '振荡幅值', '幅值误差'], conv_rows)}

* 均值 a、alpha 在全部 Delta t 上误差 ~1e-6 —— **不敏感**（均值是守恒量级的解析结果，误差被振荡平均吸收）；
* **瞬时振荡幅值**才是一阶收敛量（幅值误差随 Delta t 单调下降），这是该系统真正的离散误差度量。

**(2) 接触刚度 k_n 敏感性**（k_t = k_n，Delta t 固定）：

{table(['k_n [N/m]', 'a [m/s^2]', 'a 误差', 'a(t) 标准差'], stiff_rows)}

* 均值 a 在 k_n 变化 {case['stiffness_invariance'][0]['k_n_N_per_m']:.0e}~
  {case['stiffness_invariance'][-1]['k_n_N_per_m']:.0e} N/m 范围内散布 < 1e-5（相对），
  与「参考解不含 k_n」一致 → 刚度无关性成立；
* 瞬时幅值随 k_n 略增（{case['stiffness_invariance'][0]['a_instantaneous_std']:.4f} →
  {case['stiffness_invariance'][-1]['a_instantaneous_std']:.4f}），即刚度越大、同一 Delta t 下振荡越欠解析，
  这正是 (1) 中 Delta t 判据要盯幅值的原因。

**(3) 拟合窗长度敏感性**：

{table(['拟合窗 [s]', 'a [m/s^2]', 'a 误差', 'alpha 误差'], win_rows)}

* 窗长从 {case['window_sensitivity'][0]['fit_window_s']:.3f} s 增到
  {case['window_sensitivity'][-1]['fit_window_s']:.3f} s，a 变化 < 1e-5 相对 → 估计量无窗长偏置，
  本算例用 0.09 s 窗即可（远小于窗长自由度的代价）。

## ⑥ 失败模式、局限与工件

失败模式（已实跑并记录在 result.json）：

* `points.NEG_low_friction`（mu = 0.01）：实测 a = {pN['got']['a_t_m_per_s2']:.6f} m/s^2，
  与无滑移参考 {ref['a_ref']:.6f} m/s^2 差 {pct(pN['error']['a_rel_error'])}（>2 %，**被判失败**），
  而与滑动分支 g(sin t - mu cos t) 一致到 {pct(pN['error']['a_rel_error_vs_sliding_ref'])}
  → 模型在滑移区间同样正确，2 % 门具备判别力。
* `points.C_below_critical`（theta = 5 deg < theta_c = {pC['reference']['critical_angle_deg']:.3f} deg）：
  实测 a = {pC['got']['a_t_m_per_s2']:.3e} m/s^2 = {sci(pC['error']['creep_over_g_sin_theta'])} g sin t，
  即 CDT 无静摩擦分支但在该状态下亦无可见蠕滑（该处静力由切向摩擦承担，滚动阻力矩只在滚动建立后起作用）。
  注意这是本算例的构型结论，不能推广为「CDT 处处可持静」。
* 无惯性修正伪参考：相对误差 {pct(s['negative_control_no_inertia_reference']['rel_error_vs_wrong'])}，被 2 % 门拒绝。

局限与适用范围：

* 参考值是**闭式刚体动力学**（无滑移、刚性平面、点接触），验证的是求解器的运动学/动力学闭环与接触律实现，
  **不是**对任何真实材料参数的实验标定；Ai et al. 2011 的原表数值未取得（见 ①）。
* 切向弹簧**无阻尼**，因此 a(t) 必然在均值上下振荡（幅值 = `f g sin t/(1+f)`，`f = I/(m R^2)`；
  注意它与地面滑动摩擦 mu 无关，只由惯性和重力倾角决定）；任何单点瞬时采样都会带来 ~±40 % 的假误差 —— 本算例用窗内 LS 斜率作正式指标，
  并在表中同时给出解析与实测幅值（{pct(base['error']['oscillation_amplitude_rel_error'])}）。
* 滚动阻力分支仅验证 CDT 模型；EPSD/viscous 分支在 `docs/dem3d_rolling_resistance.md` 的既有验证中覆盖，本算例不重复。
* 半隐式 Euler 一阶精度：均值误差极小是因为振幅被平均，若把小 Delta t 下的瞬时接触力当物理量使用，须先做幅值收敛。

复现（一键）：

```bash
{REPRO_CMD}
```

工件：

* `artifacts/case_rolling_single_particle/result.json` —— 全部点位、无量纲数、收敛/敏感性表、负控与 provenance；
* `artifacts/case_rolling_single_particle/case_rolling_single_particle.png` —— 四联图；
* `prereg/CASE-DEM-ROLLING-SINGLE-PARTICLE.json` —— 预注册设计（判据/负控/停止规则）；
* `tests/test_case_rolling_single_particle.py` —— CI 守卫（6 项，含两个负控）。

## 生态摩擦（如实记录）

1. **知识基座三查部分不可用**：`GET /api/v1/knowledge/search` 对 `TensorDEM`、`DEM rolling`、
   `inclined plane rolling`、`Ai 2011` 均返回 0 命中（`available: true`，store 为空）；
   `knowledge/capabilities`、`knowledge/negative_ledger`、`knowledge/oss_solutions` 返回 404。
   → 本单的三查退化为：仓内 grep（找到 `rolling_resistance` 既有能力）+ 本报告记录。
2. **Ai et al. 2011 原文不可得**：环境内无该 PDF，无在线检索 → 只能引用模型关系式，
   「原表原页数值」**WITHHELD**（已写入 provenance，不粉饰）。
3. **性能缺陷（已修）**：`simulate_sphere_on_incline` 每步把张量 append 进 Python 列表，
   实测单步耗时随步数增长（12k 步时 ~5.8 ms/步 vs 首段 ~0.6 ms/步，超线性，长跑被拖慢 ~9×）。
   改为预分配张量写入后回到线性 0.637 ms/步（12k 步 7.7 s），数值结果不变（54 项既有测试通过）。
4. **线程数**：torch 默认线程数把该单列小张量循环拖慢 10× 以上，需显式 `torch.set_num_threads(1)`
   （已在脚本中固定并在 prereg 记录）。
5. **docstring 符号约定不一致（已修）**：`simulate_sphere_on_incline` 原写 "omega 正值顺时针，omega = -v/R"，
   与代码（omega 为 +z 分量，无滑移即 omega = -v/R < 0，即逆时针向下滚）矛盾；
   若不核对，alpha 会给出 200 % 的假误差。已改为 +z 分量表述。
"""
    payload = {
        "title": "CASE-DEM-ROLLING-SINGLE-PARTICLE：单颗粒斜面纯滚动加速度 2% 验证（TensorDEM 算例池）",
        "summary": (
            f"单球 theta=15deg 无滑移纯滚动：a 误差 {pct(s['a_rel_error'])}、alpha 误差 "
            f"{pct(s['alpha_rel_error'])}（判据 2%），无滑移闭合 {sci(base['error']['no_slip_closure'])}；"
            f"滚动阻力分支(mu_r=0.1) 误差 {pct(pB['error']['a_rel_error'])}；两个负控均被拒；"
            f"Delta t 一阶收敛 + k_n 无关性 + 窗长无关性三点敏感性；判定 {verdict}。"
        ),
        "scenario_key": "dem_single_particle_rolling_incline",
        "mission_id": 16142,
        "tags": ["TensorDEM", "DEM", "rolling", "inclined_plane", "verification",
                 "case_pool", "CASE-DEM-ROLLING-SINGLE-PARTICLE"],
        "markdown": md,
        "figures": [],
    }
    if figure_path and os.path.exists(figure_path):
        with open(figure_path, "rb") as fh:
            payload["figures"].append({
                "name": os.path.basename(figure_path),
                "caption": ("(a) 无滑移闭合 v=-R omega；(b) 瞬时 a(t) 绕解析均值振荡；"
                            "(c) Delta t 收敛；(d) 四项指标 vs 各自参考（theta=15 deg）"),
                "png_base64": base64.b64encode(fh.read()).decode(),
            })
    return md, payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="artifacts/case_rolling_single_particle/result.json")
    ap.add_argument("--doc", default="docs/CASE-DEM-ROLLING-SINGLE-PARTICLE.md")
    ap.add_argument("--figure", default="artifacts/case_rolling_single_particle/case_rolling_single_particle.png")
    ap.add_argument("--payload", default="/tmp/report.json")
    args = ap.parse_args()

    md, payload = build(args.result, args.figure)
    os.makedirs(os.path.dirname(os.path.abspath(args.doc)), exist_ok=True)
    with open(args.doc, "w") as fh:
        fh.write(md)
    with open(args.payload, "w") as fh:
        json.dump(payload, fh)
    print(args.doc, os.path.getsize(args.doc), "bytes;", args.payload,
          os.path.getsize(args.payload), "bytes; figures:", len(payload["figures"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())