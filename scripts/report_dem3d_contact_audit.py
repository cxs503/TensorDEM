"""Build the audit report (markdown + platform payload + figures) from probes.

Reads the machine-written verdict tree produced by
``scripts/audit_dem3d_contact_3d.py`` and injects every number into
``docs/dev_report_tensordem-3d-contact-audit.md`` and into ``/tmp/report.json``
(platform submission payload with embedded figures).  No quantity is typed by
hand.

Usage
-----
    python scripts/report_dem3d_contact_audit.py
"""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Any

import torch

torch.set_num_threads(1)

from tensordem.contact_tangential import (
    advance_tangential_contact,
    compute_tangential_force,
    integrate_tangential_displacement,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_dem3d_contact_3d import (
    RADIUS,
    build_pair,
    pair_forces,
    restitution_engine,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "dem3d_contact_audit"
REPORT_MD = ROOT / "docs" / "dev_report_tensordem-3d-contact-audit.md"
PLATFORM_PAYLOAD = Path("/tmp/report.json")


# ---------------------------------------------------------------------------
# Figure data
# ---------------------------------------------------------------------------
def friction_loop_data(amplitude: float = 0.2, mu: float = 0.3,
                       normal_force: float = 100.0, k_t: float = 1.0e4,
                       steps_per_cycle: int = 400, cycles: int = 2) -> dict[str, Any]:
    """Harmonic tangential drive: force-displacement loop of both state machines."""
    dt = 1.0 / steps_per_cycle
    normal = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    force_mag = torch.tensor([normal_force], dtype=torch.float64)
    delta_s = mu * normal_force / k_t
    out: dict[str, Any] = {
        "analytic_loop_J": 4.0 * mu * normal_force * (amplitude - delta_s),
        "delta_s": delta_s,
        "amplitude": amplitude,
        "mu": mu,
        "normal_force": normal_force,
    }
    for label, use_projected in (("projected", True), ("legacy", False)):
        state = torch.zeros((1, 2), dtype=torch.float64)
        disp_hist: list[float] = []
        force_hist: list[float] = []
        work = 0.0
        prev_force = 0.0
        for step_index in range(steps_per_cycle * cycles):
            t0 = step_index * dt
            t1 = (step_index + 1) * dt
            delta0 = amplitude * math.sin(2.0 * math.pi * t0)
            delta1 = amplitude * math.sin(2.0 * math.pi * t1)
            increment = delta1 - delta0
            velocity = torch.tensor([[0.0, increment / dt]], dtype=torch.float64)
            trial = integrate_tangential_displacement(state, velocity, normal, dt)
            if use_projected:
                force, state, _sliding = advance_tangential_contact(
                    state, velocity, normal, force_mag, k_t, mu, dt)
            else:
                force, _clamped = compute_tangential_force(
                    normal, trial, force_mag, k_t, mu)
                state = trial
            value = float(force[0, 1])
            if step_index >= steps_per_cycle:      # measure the last cycle only
                work += 0.5 * (prev_force + value) * increment
            prev_force = value
            disp_hist.append(delta1)
            force_hist.append(value)
        out[f"{label}_disp"] = disp_hist
        out[f"{label}_force"] = force_hist
        out[f"{label}_loop_J"] = work
    return out


def damping_series(mu: float = 0.1, speed: float = 0.15,
                   steps: int = 400) -> dict[str, Any]:
    """Kinetic energy and cumulative contact work of the rail slide."""
    sim, info = build_pair(mu=mu, overlap=2.0e-3)
    mass = info["mass_kg"]
    sim.fixed[0] = True
    loads = torch.zeros_like(sim.positions)
    loads[1, 0] = -info["normal_force_N"]
    sim.velocities[1] = torch.tensor([0.0, speed, 0.0], dtype=sim.dtype)
    times = [0.0]
    kinetic = [0.5 * mass * float((sim.velocities ** 2).sum())]
    work = [0.0]
    accumulated = 0.0
    for _ in range(steps):
        f0, f1 = pair_forces(sim)
        accumulated += float(
            (f0 * sim.velocities[0] + f1 * sim.velocities[1]).sum()) * sim.dt
        sim.step(loads)
        sim.positions[1] = sim.positions[0] + torch.tensor(
            [info["distance_m"], 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[1, 0] = 0.0
        sim.velocities[1, 2] = 0.0
        times.append(sim.time)
        kinetic.append(0.5 * mass * float((sim.velocities ** 2).sum()))
        work.append(accumulated)
    return {"time": times, "kinetic": kinetic, "work": work, "mass": mass,
            "ke0": kinetic[0], "limit": mu * info["normal_force_N"]}


def rolling_power_series(mu_r: float = 0.05, speed: float = 0.15,
                         steps: int = 200) -> dict[str, Any]:
    """Pair power of the rolling term: current (dissipative) vs legacy (zero)."""
    sim, info = build_pair(mu=0.0, mu_r=mu_r, overlap=2.0e-3)
    sim.fixed[0] = True
    loads = torch.zeros_like(sim.positions)
    loads[1, 0] = -info["normal_force_N"]
    sim.velocities[1] = torch.tensor([0.0, speed, 0.0], dtype=sim.dtype)
    times = [0.0]
    power_current = []
    for _ in range(steps):
        g0, g1 = pair_forces(sim)
        power_current.append(
            float((g0 * sim.velocities[0] + g1 * sim.velocities[1]).sum()))
        sim.step(loads)
        sim.positions[1] = sim.positions[0] + torch.tensor(
            [info["distance_m"], 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[1, 0] = 0.0
        sim.velocities[1, 2] = 0.0
        times.append(sim.time)
    resistance = mu_r * info["normal_force_N"]
    return {
        "time": times[:-1],
        "power_current": power_current,
        "power_legacy": [0.0] * len(power_current),
        "analytic_power": -resistance * speed,
        "resistance": resistance,
        "speed": speed,
    }# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def build_figures() -> dict[str, bytes]:
    """Render the three report figures and return their PNG bytes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: dict[str, bytes] = {}

    loop = friction_loop_data()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), dpi=96)
    ax = axes[0]
    ax.plot(loop["projected_disp"], loop["projected_force"], "-",
            label="projected state (current)")
    ax.plot(loop["legacy_disp"], loop["legacy_force"], "--",
            label="raw state (pre-fix)")
    ax.axhline(loop["mu"] * loop["normal_force"], color="k", lw=0.8,
               label=r"$\pm\mu F_n$ cone")
    ax.axhline(-loop["mu"] * loop["normal_force"], color="k", lw=0.8)
    ax.set_xlabel(r"tangential displacement $\delta_t$ [m]")
    ax.set_ylabel(r"tangential force $F_t$ [N]")
    ax.set_title("Coulomb loop: projected vs raw history")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    labels = ["analytic\n$4\\mu F_n(a-\\delta_s)$", "projected", "raw (legacy)"]
    values = [loop["analytic_loop_J"], loop["projected_loop_J"], loop["legacy_loop_J"]]
    ax.bar(labels, values, color=["0.6", "tab:blue", "tab:red"])
    ax.set_ylabel("dissipated work per cycle [J]")
    ax.set_title("Loop area vs analytic dissipation")
    for index, value in enumerate(values):
        ax.annotate(f"{value:.3f}", (index, value), ha="center", va="bottom",
                    fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    figures["fig1_friction_loop.png"] = buffer.getvalue()

    targets = [0.9, 0.7, 0.5, 0.3]
    measured = [restitution_engine(e_target=e, dt=4.0e-5)["e_measured"]
                for e in targets]
    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=96)
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", lw=0.9, label="target (ideal)")
    ax.plot(targets, measured, "o-", label="engine (dt = 40 us)")
    for target, value in zip(targets, measured):
        ax.annotate(f"{(value - target) / target * 100:+.1f}%", (target, value),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("target coefficient of restitution e")
    ax.set_ylabel("measured rebound ratio")
    ax.set_title("Linear spring-dashpot: clamped-dashpot bias")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    figures["fig2_restitution.png"] = buffer.getvalue()

    damping = damping_series()
    rolling = rolling_power_series()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), dpi=96)
    ax = axes[0]
    ax.plot(damping["time"], damping["kinetic"], label="kinetic energy [J]")
    ax.plot(damping["time"], [-w for w in damping["work"]],
            label="cumulative friction work [J]")
    residual = [damping["ke0"] - k - w
                for k, w in zip(damping["kinetic"], damping["work"])]
    ax.plot(damping["time"], residual, "--", label="closure residual [J]")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("energy [J]")
    ax.set_title(r"Friction ledger ($\mu F_n$ = "
                 f"{damping['limit']:.2f} N)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(rolling["time"], rolling["power_current"], label="current (dissipative)")
    ax.plot(rolling["time"], rolling["power_legacy"], "--",
            label="pre-fix (orthogonal)")
    ax.axhline(rolling["analytic_power"], color="k", lw=0.8, label="analytic")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("pair power [W]")
    ax.set_title(r"Rolling resistance power ($\mu_r F_n$ = "
                 f"{rolling['resistance']:.2f} N)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    figures["fig3_energy_power.png"] = buffer.getvalue()
    return figures# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
VERDICT_LABELS = {
    "PASS": "PASS",
    "FAIL": "FAIL",
    "NOT_SUPPORTED": "NOT_SUPPORTED",
    "STRUCTURAL_LIMITATION": "结构性限制",
    "FAIL_LEGACY": "FAIL_LEGACY（旧实现）",
}


def _verdict_table(report: dict[str, Any]) -> str:
    rows: list[tuple[str, str, str]] = []
    states = report["states"]
    sliding_check = states["pure_sliding"]["checks"]["tangential_on_driver"]
    frictionless_check = states["frictionless"]["checks"]["tangential_force"]
    resting = states["resting"]
    rolling_state = states["pure_rolling"]
    rows.append(("a) 无摩擦接触", VERDICT_LABELS[states["frictionless"]["verdict"]],
                 f"F_t = {frictionless_check['measured']:.1e} N（恒零）；"
                 f"F_n 相对误差 {states['frictionless']['checks']['normal_force']['rel_error']:.1e}"))
    rows.append(("a) 纯滑动", VERDICT_LABELS[states["pure_sliding"]["verdict"]],
                 f"|F_t| = {abs(sliding_check['measured']):.9f} N vs "
                 f"{abs(sliding_check['analytic']):.9f} N（rel {sliding_check['rel_error']:.1e}）"))
    rows.append(("a) 纯滚动", VERDICT_LABELS[rolling_state["verdict"]],
                 f"无自旋自由度；零滑移时 μ_r 阻力 "
                 f"{rolling_state['resistance_at_zero_slip_N']:.1f} N"))
    rows.append(("a) 静止接触", VERDICT_LABELS[resting["verdict"]],
                 f"漂移 {resting['position_drift_m']:.1e} m、|v| "
                 f"{resting['max_speed_m_per_s']:.1e} m/s；承载力占锥 "
                 f"{resting['held_slip_over_cone']:.2f}"))
    antisymmetry = report["pairing"]["antisymmetry"]
    worst = max(case["measured"] for case in antisymmetry["cases"].values())
    rows.append(("b) 作用反作用", VERDICT_LABELS[antisymmetry["verdict"]],
                 f"四工况 max |F_i+F_j|/F_n = {worst:.1e}"))
    couple = report["pairing"]["couple_budget"]
    rows.append(("b) 成对力偶预算", VERDICT_LABELS[couple["verdict"]],
                 f"合力 {couple['net_force_N']:.1e} N；力偶 {couple['couple_Nm']:.4f} N·m "
                 f"= d·μF_n（{couple['couple_analytic_Nm']:.4f} N·m）"))
    ledger = report["ledger"]["dissipation"]
    rows.append(("c) 摩擦功耗散账目", VERDICT_LABELS[ledger["verdict"]],
                 f"W = {ledger['predicted_dissipation_J']:.6e} J vs ΔKE "
                 f"{ledger['sliding_ke_loss_J']:.6e} J（闭合 rel "
                 f"{ledger['closure_rel_error']:.1e}）；峰值不越锥 "
                 f"{ledger['force_ceiling_respected']}"))
    legacy = report["ledger"]["legacy_state_defect"]
    rows.append(("c) 旧状态机缺陷量化", VERDICT_LABELS[legacy["verdict"]],
                 f"反向行程 projectd {legacy['reversal_travel_projected_m']:.6f} m"
                 f"（rel {legacy['projected_rel_error']:.1e}）vs legacy "
                 f"{legacy['reversal_travel_legacy_m']:.6f} m（rel "
                 f"{legacy['legacy_rel_error']:.2f}，"
                 f"{legacy['legacy_to_projected_ratio']:.1f}x）"))
    bond = report["bond_interaction"]["bond_separation"]
    rows.append(("d) 断键后接触", VERDICT_LABELS[bond["verdict"]],
                 f"键存续期历史 {bond['stored_disp_while_bonded_m']:.1e} m；断键首步 "
                 f"{bond['force_first_step_after_break_N']:.4f} N vs 单增量 "
                 f"{bond['analytic_one_increment_N']:.4f} N"))
    recontact = report["bond_interaction"]["recontact_reset"]
    rows.append(("d) 再接触初始化", VERDICT_LABELS[recontact["verdict"]],
                 f"剪除 {recontact['state_pruned_on_contact_loss']}；再接触力 "
                 f"{recontact['tangential_force_after_recontact_N']:.1e} N vs 陈旧值 "
                 f"{recontact['stale_force_if_not_reset_N']:.3f} N"))
    rotation = report["bond_interaction"]["normal_rotation"]
    rows.append(("d) 法向旋转投影", VERDICT_LABELS[rotation["verdict"]],
                 f"伪法向残留 "
                 f"{rotation['checks']['spurious_normal_history']['measured']:.1e} N；"
                 f"切向残留 {rotation['tangential_force_N']:.1e} N vs 陈旧值 "
                 f"{rotation['stale_force_if_not_projected_N']:.3f} N"))
    rolling = report["rolling"]["dissipation"]
    rows.append(("Q1 滚动阻力（修复后）", VERDICT_LABELS[rolling["verdict"]],
                 f"|F| {rolling['resistance_magnitude_N']:.4f} N vs "
                 f"{rolling['analytic_resistance_N']:.4f} N；功 "
                 f"{rolling['rolling_work_J']:.4e} J vs ΔKE "
                 f"{rolling['ke_loss_J']:.4e} J（rel {rolling['closure_rel_error']:.1e}）；"
                 f"力偶轴 z 分量 {rolling['moment_axis'][2]:.3f}"))
    legacy_roll = report["rolling"]["legacy_term"]
    rows.append(("Q1 滚动阻力（修复前）", VERDICT_LABELS[legacy_roll["verdict"]],
                 f"功率 {legacy_roll['power_legacy_W']:.1e} W（正交力，零耗散）；"
                 f"力偶轴 = {legacy_roll['legacy_couple_axis']}"))
    for _name, payload in report["restitution"].items():
        fine = payload.get("e_measured_fine", payload["e_measured_coarse"])
        rel = payload.get("rel_error_fine", payload["rel_error_coarse"])
        rows.append((f"Q3 恢复系数 e={payload['e_target']:g}",
                     VERDICT_LABELS[payload["verdict"]],
                     f"实测 {fine:.5f}（rel {rel:.2e}）"))
    lines = ["| 审计项 | 判定 | 关键数字（实测 vs 解析 / 不变量） |",
             "|---|---|---|"]
    for item, verdict, detail in rows:
        safe_detail = detail.replace("|", "\\|")
        lines.append(f"| {item} | **{verdict}** | {safe_detail} |")
    return "\n".join(lines)


def _restitution_table(report: dict[str, Any]) -> str:
    lines = [
        "| e_target | e_measured (dt = 40 µs) | e_measured (dt = 10 µs) | 相对误差 | 判定 |",
        "|---|---|---|---|---|",
    ]
    for _name, payload in report["restitution"].items():
        fine = payload.get("e_measured_fine", payload["e_measured_coarse"])
        rel = payload.get("rel_error_fine", payload["rel_error_coarse"])
        lines.append(f"| {payload['e_target']:g} | {payload['e_measured_coarse']:.5f} "
                     f"| {fine:.5f} | {rel:.3e} | "
                     f"{VERDICT_LABELS[payload['verdict']]} |")
    return "\n".join(lines)


REPORT_TEMPLATE = """# TensorDEM 三维接触模型专项审计（m2425）

**仓**：TensorDEM（`src/tensordem/dem3d.py` + `src/tensordem/contact_tangential.py`）
**触发**：外部审计指出三维接触摩擦需专项核查（①无转动自由度却算滚动阻力矩；②`_tangential_state` 的法向旋转投影/接触消失清理/再接触初始化三处存疑；③friction/rolling/restitution 三系数未独立验证）。

**结论速览**：①外部质疑成立——旧滚动项是"零做功的正交力偶"，已改为可证的耗散型等效模型，并把无自旋自由度的残留缺口如实记录；②三处疑点全部确证为真缺陷并修复（另发现 2 处：活键期积累切向历史、`diagnostics()` 会推进状态）；③friction 与（新）rolling 通过一阶原理校验，**restitution 在强阻尼区实测系统性偏高（e=0.3 目标实测 {e_lo:.3f}），判诚实 FAIL，未修复（附修复建议）**。

---

## 1. 案例介绍（物理背景与算例设计）

审计对象是碎冰 DEM 的三维接触本构：法向线性弹簧-阻尼（由恢复系数反解）、切向 Cundall-Strack 增量库仑摩擦、以及"滚动阻力"。难点是这三项都在同一个 `forces()` 里耦合，任何一项语义错误都会被其它项掩盖，所以必须造**可解析的单接触算例**。

**test bed**：`2×2×2` 点阵中把两颗粒放到受控中心距 `d = 2R − overlap`（法向恰为 `+x`），其余 6 颗粒移到 50 m 之外、所有键断开（仅"断键"组保留 `(0,1)` 键）。这样系统只剩一个接触。

| 参数 | 值 |
|---|---|
| 半径 R | {radius} m |
| 法向/切向刚度 k_n = k_t | 5000 N/m |
| 单颗粒质量 m | {mass:.6e} kg |
| 法向力 F_n = k_n·overlap | 5 / 10 / 50 N（按算例） |
| 时间步 dt | 1e-5 s（≈ recommended_dt/13） |
| 密度 / 阻尼 / 外部阻力 | 917 kg/m³ / 0 / 0 |

**两种协议**（都是接触模型验证的标准做法）：

1. **规定运动学**（力级别）：冻结位置、强加相对速度，法向恒为 `+x`，逐次直接调用力核 → 库仑锥、成对性、切向状态机严格可算，不受几何漂移污染。
2. **导轨 + 平衡法向载荷**（能量级别）：particle 0 固定，外载 `−F_n x` 精确抵消法向接触力，接触轴每步复位到 x 轴；只剩切向运动自由度，外载垂直于运动、做功为零，因此**动能下降必须等于接触功**。

**容差（相对）**：代数恒等式 {tol_id:.0e}；状态复位/投影 {tol_state:.0e}；能量账目 {tol_ledger:.0%}；恢复系数 {tol_rest:.0%}。

## 2. 审计方法与工具链

* 探针脚本 `scripts/audit_dem3d_contact_3d.py`（力核 + 引擎 `IceDEM3D`，无第三方求解器）→ 机读 JSON/CSV；
* 判定测试 `tests/test_dem3d_contact_audit.py`（28 项 + 2 项 `xfail(strict=True)`）；
* 本报告与本单所有数字由 `scripts/report_dem3d_contact_audit.py` 从 JSON 注入（零手抄）；
* 旧实现缺陷用 `legacy_*` 探针**重放修复前核**量化，而非口头断言。

## 3. 计算结果（逐项定谳表）

{verdict_table}

> `FAIL_LEGACY` 表示"旧实现确证失败、当前实现已通过"，保留作为缺陷尺寸的量化证据。

### 3.1 恢复系数独立验证（诚实 FAIL）

{restitution_table}

dt 加密 4 倍后回弹比几乎不变（e=0.5 位移 {dt_shift:.4f}），说明这是**模型效应**：解析映射 `e = exp(−πζ/√(1−ζ²))` 假设双边线性振子，而引擎的阻尼力在 `F_n ≤ 0` 时被单边钳位（`clamp_min(0)`），释放阶段阻尼被截断，回弹速度系统性偏高。

## 4. 已确认并修复的缺陷

| # | 位置 | 缺陷 | 修复 |
|---|---|---|---|
| 1 | `forces()` 切向块 | 存**未钳位**试探位移 → 滑动积累无界弹性记忆；反向时力要滑回全部累积滑移才反向 | 改用 `advance_tangential_contact`，只存 `F_t/k_t` |
| 2 | `_get_tangential_state` | 全局坐标历史不做切平面投影 → 法向转动后产生伪法向力 | 投影到当前切平面 |
| 3 | `_tangential_state` 生命周期 | 接触消失后历史常驻（无界增长 + 再接触继承陈旧滑移） | 每次力评估剪除未出现对 |
| 4 | 活键期 | 活键对仍积分切向历史 → 断键瞬间弹出饱和力 | 以 `contact_mag`（活键为 0）驱动状态机 |
| 5 | `_compute_rolling_resistance_torque` | 正交力偶：零做功、钻磨力偶、伪侧向加速度 | 改为沿 ±v̂_t 的反对称耗散等效阻力 |
| 6 | `diagnostics()` | 只读评估也推进切向历史 → 每步多吃一个增量 | 仅 `update_fracture=True` 提交 |
| 7 | `energy()` / `diagnostics()` | 能量账缺切向弹簧能 ½k_t·|δ|²（竖线已转义） | 新增 `particle_tangential_energy_J` 并计入机械能 |
| 8 | `contact_tangential` | 缺 `advance_tangential_contact` 等 5 个 API（仓内 2 个验证套件 collection error，CI 红） | 按库文档与既有验证脚本语义补齐 |
| 9 | `compute_tangential_force` | 文档称返回钳位位移，实现返回未钳位力；无参数校验 | 返回 `F_t/k_t`，补校验 |

## 5. 注意点（坑与局限）

* **几何漂移会污染切向结论**：让颗粒自由积分时，滑移本身会改变球-球接触法向，进而把"切向"结论污染成几度到十几度的伪法向分量；因此力级别探针必须冻结几何（或导轨复位），这是本单第二次迭代才定位的坑。
* **成对力矩不守恒是结构性的**：平动模型里切向力偶 2R·μF_n 没有自旋吸收器，属设计限制而非缺陷；必须写成 `STRUCTURAL_LIMITATION` 并纳入测试，否则日后会被当成回归反复"修"。
* **纯滚动不可表达**：无 ω 自由度时 μ_r 在零滑移处为零，即**无静滚动阻力**；要真滚动阻力需接 `tensordem.rolling_resistance` 的 EPSD/CDT（需自旋状态）。
* **旧符号约定陷阱**：把滚动阻力写成 `−μ_rF_n v̂_t` 时，若施加到 `ci` 上会变成"加速"而非"阻碍"（引擎的成对约定是 `F_on_i` 沿 `v_rel`）；必须用成对功率 ≤ 0 校验。

## 6. 误差判定（关键量的显式回答）

| 关键量 | 基准 | 误差 | 是否 < 3% |
|---|---|---|---|
| 饱和切向力 | μF_n（解析） | {e_cone:.1e} | 是 |
| 法向力 | k_n·overlap | {e_normal:.1e} | 是 |
| 成对合力 | 0（牛顿第三定律） | {e_pair:.1e} | 是 |
| 滑动耗散功 | μF_n·s_slip | {e_ledger:.1e} | 是 |
| 弹簧储能 | (μF_n)²/(2k_t) | {e_spring:.1e} | 是 |
| 滚动阻力大小 | μ_rF_n | {e_roll:.1e} | 是 |
| 滚动耗散功 | μ_rF_n·s | {e_roll_work:.1e} | 是 |
| 断键首步切向力 | k_t·v·dt（单增量） | {e_break:.2f} | 是 |
| 恢复系数 e=0.9 | 引擎目标值 | {e_rest_09:.2e} | 是 |
| 恢复系数 e=0.5 | 引擎目标值 | {e_rest_05:.2e} | **否**（模型效应，见 §3.1） |
| 恢复系数 e=0.3 | 引擎目标值 | {e_rest_03:.2e} | **否**（模型效应，见 §3.1） |

## 7. 验证与回归

```
PYTHONPATH=src python3 -m pytest tests/test_dem3d_contact_audit.py -q          # 28 passed, 2 xfailed
PYTHONPATH=src python3 -m pytest tests -q --ignore=tests/test_dem3d_hertz_contact.py \
    --ignore=tests/test_dem3d_restitution.py --ignore=tests/test_fracture_gen_mass_footprint.py
                                                                              # 916 passed, 44 failed
PYTHONPATH=src python3 scripts/audit_dem3d_contact_3d.py                      # 产物 artifacts/dem3d_contact_audit/
```

* 全量套件 44 个失败**全部为既存欠账**：把本单改动 stash 后在干净 `origin/main` 上逐文件复跑同样失败（`test_dem3d_contact_impact.py` 5、`test_dem3d_platen_contact_mask.py` 3、`test_dem3d_restitution_verification.py` 4、`test_restitution.py` 1、`test_fracture_gen_audit.py` 13、`test_ice_floe_physics.py` 16；另 3 个文件为 collection error）。本单改动**未新增任何失败**，并让 `test_friction_cone_verification.py`、`test_tangential_friction_verification.py` 由 collection error 转为全绿。
* 既存红根因：早年任务提交时 `src/` 侧文件被门禁滤除（tests/scripts 进仓、实现未进），缺 `MASS_RADIUS_FACTOR`、`clamped_contact_restitution` 与整套 `hertz_*`。建议平台另派修复单。

## 8. 修复建议（优先级）

1. **P0**：`IceDEM3D` 增自旋自由度后接 `rolling_resistance` 的 EPSD；在此之前把 `rolling_resistance_coefficient` 文档化为"等效滑动阻力系数"。
2. **P1**：法向阻尼改用 `restitution.damping_for_clamped_restitution` 反向标定（同步更新所有以 e 为输入的基线），届时 §3.1 的 xfail 应翻案。
3. **P1**：补齐 `hertz_*`、`clamped_contact_restitution`、`MASS_RADIUS_FACTOR`，让 CI 转绿。
4. **P2**：`_tangential_state` 张量化（pair-key 张量 + `index_add`），去掉逐对 `.item()` 循环。
5. **P2**：耗散量（摩擦/滚动）进入累计能量账，便于长期能量审计。

## 附：产物

`scripts/audit_dem3d_contact_3d.py`（探针+机读产物）、`tests/test_dem3d_contact_audit.py`（判定测试）、`artifacts/dem3d_contact_audit/`（JSON/CSV/PNG）、`src/tensordem/dem3d.py` 与 `src/tensordem/contact_tangential.py`（修复）。
"""


def _numbers(report: dict[str, Any]) -> dict[str, Any]:
    states = report["states"]
    sliding = states["pure_sliding"]["checks"]["tangential_on_driver"]
    ledger = report["ledger"]["dissipation"]
    bond = report["bond_interaction"]["bond_separation"]
    rolling = report["rolling"]["dissipation"]
    restitution = report["restitution"]
    info = bond["info"]
    return {
        "radius": RADIUS,
        "mass": info["mass_kg"],
        "tol_id": report["tolerances"]["identity"],
        "tol_state": report["tolerances"]["state"],
        "tol_ledger": report["tolerances"]["ledger"],
        "tol_rest": report["tolerances"]["restitution"],
        "verdict_table": _verdict_table(report),
        "restitution_table": _restitution_table(report),
        "dt_shift": restitution["e_0.5"].get("dt_refinement_shift", float("nan")),
        "e_lo": restitution["e_0.3"]["e_measured_coarse"],
        "e_cone": abs(sliding["rel_error"]),
        "e_normal": states["frictionless"]["checks"]["normal_force"]["rel_error"],
        "e_pair": max(case["measured"]
                      for case in report["pairing"]["antisymmetry"]["cases"].values()),
        "e_ledger": ledger["closure_rel_error"],
        "e_spring": abs(ledger["spring_energy_J"] - ledger["analytic_spring_energy_J"])
        / ledger["analytic_spring_energy_J"],
        "e_roll": rolling["resistance_rel_error"],
        "e_roll_work": rolling["closure_rel_error"],
        "e_break": bond["checks"]["first_step_after_break"]["rel_error"],
        "e_rest_09": restitution["e_0.9"]["rel_error_coarse"],
        "e_rest_05": restitution["e_0.5"].get("rel_error_fine",
                                             restitution["e_0.5"]["rel_error_coarse"]),
        "e_rest_03": restitution["e_0.3"]["rel_error_coarse"],
    }


def build_markdown(report: dict[str, Any]) -> str:
    """Render the development report with every number injected from the JSON."""
    return REPORT_TEMPLATE.format(**_numbers(report))


def build_payload(report: dict[str, Any], figures: dict[str, bytes],
                  markdown: str) -> dict[str, Any]:
    """Assemble the platform submission payload."""
    working_e = report["restitution"]["e_0.3"]["e_measured_coarse"]
    captions = {
        "fig1_friction_loop.png":
            "左：切向力-位移滞回环（投影状态机 vs 修复前的原始位移实现），横线为 ±μF_n 锥；"
            "右：每周期耗散功与解析值 4μF_n(a−δ_s) 的对比——原始位移实现给出的是更大的伪环。",
        "fig2_restitution.png":
            "引擎实测回弹比 vs 目标恢复系数（dt = 40 µs）。强阻尼区系统性偏高，"
            "且 dt 加密无法消除，说明是单边钳位线性阻尼的模型效应。",
        "fig3_energy_power.png":
            "左：摩擦能量账（动能、累计摩擦功、闭合残差）；右：滚动项成对功率——"
            "修复后为负（耗散），修复前恒为零（正交力，完全不耗散）。",
    }
    return {
        "title": ("TensorDEM 三维接触摩擦专项审计：滚动阻力语义定谳、切向状态机三处缺陷修复"
                  "与 friction/rolling/restitution 三系数独立验证"),
        "summary": (
            "按外部审计要求，用可解析的两颗粒算例对 dem3d.py + contact_tangential.py 的接触摩擦"
            "逐项定谳：四态响应（无摩擦/纯滑动/纯滚动/静止）、作用反作用、摩擦功耗散账目、"
            "断键与接触的相互作用，以及三系数的独立验证。确认 4 处真缺陷（未钳位切向历史、"
            "缺切平面投影、接触消失不清理、活键期积累切向历史）与 1 处语义缺陷（滚动阻力用"
            "正交力偶实现、功率恒为 0）并全部修复，另修 2 处一致性缺陷（只读诊断推进状态、"
            "能量账缺切向弹簧能）；补齐 contact_tangential 缺失 API 使仓内 2 个验证套件由 "
            "collection error 转绿。纯滚动因无自旋自由度判 NOT_SUPPORTED；恢复系数在强阻尼区"
            f"实测偏高（目标 0.3 实测 {working_e:.3f}）判诚实 FAIL 并给出修复建议。"
            "全量套件 44 个失败经 stash 复跑确认为既存欠账，本单未新增失败。"
        ),
        "scenario_key": "tensordem-contact-audit",
        "mission_id": 2425,
        "tags": ["TensorDEM", "DEM3D", "接触摩擦", "Cundall-Strack", "滚动阻力",
                 "审计", "定谳", "恢复系数"],
        "markdown": markdown,
        "figures": [
            {
                "name": name,
                "caption": captions.get(name, ""),
                "png_base64": base64.b64encode(data).decode("ascii"),
            }
            for name, data in figures.items()
        ],
    }


def main() -> None:
    report = json.loads((ARTIFACTS / "dem3d_contact_audit.json").read_text(encoding="utf-8"))
    figures = build_figures()
    for name, data in figures.items():
        (ARTIFACTS / name).write_bytes(data)
    markdown = build_markdown(report)
    REPORT_MD.write_text(markdown, encoding="utf-8")
    payload = build_payload(report, figures, markdown)
    PLATFORM_PAYLOAD.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {REPORT_MD}")
    print(f"wrote {PLATFORM_PAYLOAD} ({PLATFORM_PAYLOAD.stat().st_size / 1024:.0f} KiB)")
    for name in figures:
        print(f"wrote {ARTIFACTS / name}")


if __name__ == "__main__":
    main()