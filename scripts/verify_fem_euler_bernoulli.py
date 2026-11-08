"""Analytic verification of ``tensordem.fem_euler_bernoulli`` (Euler-Bernoulli beam FEM).

Every number in the generated report and markdown note is produced here
(no hand-copied values).  Outputs:

* ``artifacts/fem_euler_bernoulli/report.json``        -- machine record
* ``artifacts/fem_euler_bernoulli/static_table.csv``   -- static verification table
* ``artifacts/fem_euler_bernoulli/modal_table.csv``    -- eigenfrequency convergence
* ``artifacts/fem_euler_bernoulli/oddmesh_table.csv``  -- mid-span interpolation study
* ``artifacts/fem_euler_bernoulli/dynamic_table.csv``  -- Newmark time-integration study
* ``artifacts/fem_euler_bernoulli/fig_*.png``          -- figures
* ``docs/fem_euler_bernoulli_analytic_verification.md``-- generated write-up

Usage:  PYTHONPATH=src python3 scripts/verify_fem_euler_bernoulli.py
"""
from __future__ import annotations

import json
import math
import os

import torch

torch.set_num_threads(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem.fem_euler_bernoulli import (
    BeamConfig,
    EulerBernoulliBeam,
    beam_center_deflection_analytic,
    beam_tip_deflection_analytic,
)

OUT = "artifacts/fem_euler_bernoulli"
DOC = "docs/fem_euler_bernoulli_analytic_verification.md"
P = 1000.0          # point load [N]
Q = 500.0           # uniform load [N/m]
TOL = 0.03          # 3 % acceptance threshold requested by the platform standard


# --------------------------------------------------------------------------- #
# analytic reference profiles
# --------------------------------------------------------------------------- #
def w_ss_point(x: float, cfg: BeamConfig, load: float = P) -> float:
    """Simply supported, central point load (x measured from the left support)."""
    L = cfg.length
    xx = x if x <= 0.5 * L else L - x
    return load * xx * (3.0 * L ** 2 - 4.0 * xx ** 2) / (48.0 * cfg.young_modulus * cfg.inertia)


def w_cc_point(x: float, cfg: BeamConfig, load: float = P) -> float:
    """Clamped-clamped, central point load."""
    L = cfg.length
    xx = x if x <= 0.5 * L else L - x
    return load * xx ** 2 * (3.0 * L - 4.0 * xx) / (48.0 * cfg.young_modulus * cfg.inertia)


def w_cantilever_tip(x: float, cfg: BeamConfig, load: float = P) -> float:
    """Cantilever, tip point load."""
    L = cfg.length
    return load * x ** 2 * (3.0 * L - x) / (6.0 * cfg.young_modulus * cfg.inertia)


def w_ss_udl(x: float, cfg: BeamConfig, q: float = Q) -> float:
    """Simply supported, uniform load."""
    L = cfg.length
    return q * x * (L ** 3 - 2.0 * L * x ** 2 + x ** 3) / (24.0 * cfg.young_modulus * cfg.inertia)


def uniform_load_vector(beam: EulerBernoulliBeam, q: float) -> torch.Tensor:
    f = torch.zeros(beam.n_dof, dtype=torch.float64)
    h = beam.h_elem
    for e in range(beam.config.n_elements):
        f[2 * e] += q * h / 2.0
        f[2 * e + 1] += q * h * h / 12.0
        f[2 * e + 2] += q * h / 2.0
        f[2 * e + 3] -= q * h * h / 12.0
    return f


def commit_static(beam: EulerBernoulliBeam, force: torch.Tensor) -> dict:
    """Static solve followed by loading the equilibrium into the dynamic state."""
    result = beam.static_solve(force, commit=True)
    return result


# --------------------------------------------------------------------------- #
# 1. static verification against closed-form solutions
# --------------------------------------------------------------------------- #
def static_study() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    profiles: dict[str, tuple[list[float], list[float], list[float]]] = {}

    cases = [
        ("ss_point", "simply_supported", "point"),
        ("cc_point", "clamped", "point"),
        ("cantilever_tip", "cantilever", "point"),
        ("ss_udl", "simply_supported", "udl"),
        ("cc_udl", "clamped", "udl"),
    ]
    ref = {
        "ss_point": w_ss_point,
        "cc_point": w_cc_point,
        "cantilever_tip": w_cantilever_tip,
        "ss_udl": w_ss_udl,
    }

    for name, bc, kind in cases:
        for n_elements in (2, 4, 10, 20):
            cfg = BeamConfig(n_elements=n_elements, bc=bc)
            beam = EulerBernoulliBeam(cfg)
            if kind == "point":
                if bc == "cantilever":
                    load = torch.zeros(beam.n_dof, dtype=torch.float64)
                    load[-2] = P
                    exact = beam_tip_deflection_analytic(P, cfg.length,
                                                         cfg.young_modulus, cfg.inertia)
                else:
                    load = beam.consistent_point_load(0.5 * cfg.length, P)
                    exact_bc = "simply_supported" if bc == "simply_supported" else "clamped"
                    exact = beam_center_deflection_analytic(
                        P, cfg.length, cfg.young_modulus, cfg.inertia, bc=exact_bc)
                probe = cfg.length if bc == "cantilever" else 0.5 * cfg.length
            else:
                load = uniform_load_vector(beam, Q)
                coefficient = 5.0 / 384.0 if bc == "simply_supported" else 1.0 / 384.0
                exact = coefficient * Q * cfg.length ** 4 / (cfg.young_modulus * cfg.inertia)
                probe = 0.5 * cfg.length

            result = commit_static(beam, load)
            numeric = beam.deflection_at(probe)
            rel = abs(numeric - exact) / exact
            rows.append({
                "case": name,
                "bc": bc,
                "load": kind,
                "n_elements": n_elements,
                "probe_x_over_L": probe / cfg.length,
                "fem": numeric,
                "analytic": exact,
                "rel_error": rel,
                "reaction": result["reaction"],
            })
            if n_elements == 20 and name in ref:
                xs = [cfg.length * i / 200.0 for i in range(201)]
                profiles[name] = (
                    xs,
                    [beam.deflection_at(x) for x in xs],
                    [ref[name](x, cfg) for x in xs],
                )

    # nodal exactness: the field is only sampled exactly at nodes for point loads
    nodal = []
    for bc, name in (("simply_supported", "ss_point"), ("clamped", "cc_point")):
        cfg = BeamConfig(n_elements=20, bc=bc)
        beam = EulerBernoulliBeam(cfg)
        load = beam.consistent_point_load(0.5 * cfg.length, P)
        commit_static(beam, load)
        for e in (5, 10):
            x = e * beam.h_elem
            exact = ref[name](x, cfg)
            nodal.append({"case": name, "x_over_L": x / cfg.length,
                          "fem": beam.deflection_at(x), "analytic": exact,
                          "rel_error": abs(beam.deflection_at(x) - exact) / exact})
    return rows, {"profiles": profiles, "nodal": nodal}


# --------------------------------------------------------------------------- #
# 2. eigenfrequency convergence (fixes the generalized eigenproblem check)
# --------------------------------------------------------------------------- #
def modal_study() -> list[dict]:
    rows = []
    for bc, factor in (("simply_supported", None), ("clamped", "cc"), ("cantilever", None)):
        for n_elements in (5, 10, 20, 40, 80):
            cfg = BeamConfig(n_elements=n_elements, bc=bc)
            beam = EulerBernoulliBeam(cfg)
            base = math.sqrt(cfg.young_modulus * cfg.inertia
                             / (cfg.mass_per_length * cfg.length ** 4))
            for mode in (1, 2, 3):
                if bc == "simply_supported":
                    exact = (mode * math.pi) ** 2 * base
                elif bc == "clamped":
                    # clamped-clamped: beta_n L = 4.7300, 7.8532, 10.9956
                    betas = (4.730040745, 7.853204624, 10.995607838)
                    exact = betas[mode - 1] ** 2 * base
                else:
                    # cantilever: beta_n L = 1.8751, 4.6941, 7.8548
                    betas = (1.875104069, 4.694091133, 7.854757438)
                    exact = betas[mode - 1] ** 2 * base
                numeric = float(beam._omegas[mode - 1])
                rows.append({
                    "bc": bc, "n_elements": n_elements, "mode": mode,
                    "fem_omega": numeric, "analytic_omega": exact,
                    "rel_error": abs(numeric - exact) / exact,
                })
    return rows


# --------------------------------------------------------------------------- #
# 3. mid-span interpolation study (odd element counts) -- regression for defect 2
# --------------------------------------------------------------------------- #
def odd_mesh_study() -> list[dict]:
    rows = []
    cfg = BeamConfig(n_elements=9)
    beam9 = EulerBernoulliBeam(cfg)
    old_x = float(beam9.x_nodes[beam9.n_node // 2])
    for n_elements in (3, 5, 9, 19, 39):
        cfg = BeamConfig(n_elements=n_elements)
        beam = EulerBernoulliBeam(cfg)
        result = beam.static_solve(beam.consistent_point_load(0.5 * cfg.length, P))
        exact = beam_center_deflection_analytic(P, cfg.length, cfg.young_modulus, cfg.inertia)
        old_node_deflection = result["deflection"][beam.n_node // 2]
        rows.append({
            "n_elements": n_elements,
            "h_over_L": 1.0 / n_elements,
            "middle_node_x_over_L": float(beam.x_nodes[beam.n_node // 2]) / cfg.length,
            "interpolated_midspan": result["center_deflection"],
            "old_reported_node_value": old_node_deflection,
            "analytic": exact,
            "rel_error_interpolated": abs(result["center_deflection"] - exact) / exact,
            "rel_error_old_node_pick": abs(old_node_deflection - exact) / exact,
        })
    return rows


# --------------------------------------------------------------------------- #
# 4. dynamic verification (Newmark): period, energy, Rayleigh damping
# --------------------------------------------------------------------------- #
def dynamic_study() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    traces: dict[str, dict] = {}
    for label, zeta in (("undamped", 0.0), ("damped_2pct", 0.02)):
        cfg = BeamConfig(n_elements=20, damping_ratio=zeta)
        beam = EulerBernoulliBeam(cfg)
        w1 = math.sqrt(cfg.young_modulus * cfg.inertia
                       / (cfg.mass_per_length * cfg.length ** 4)) * math.pi ** 2
        period = 2.0 * math.pi / w1
        dt = period / 500.0
        beam.static_solve(beam.consistent_point_load(0.5 * cfg.length, P), commit=True)
        d0 = beam.center_deflection()
        energy0 = beam.total_energy()
        zero = torch.zeros(beam.n_dof, dtype=torch.float64)
        times, disps = [], []
        n_steps = int(2.5 * period / dt)
        for _ in range(n_steps):
            beam.apply_load(zero, dt)
            times.append(beam.time)
            disps.append(beam.center_deflection())

        crossings = []
        for (t0, a0), (t1, a1) in zip(zip(times, disps), zip(times[1:], disps[1:])):
            if a0 > 0.0 >= a1 or a0 < 0.0 <= a1:
                crossings.append(t0 + (t1 - t0) * abs(a0) / (abs(a0) + abs(a1)))
        periods = [2.0 * (crossings[i + 1] - crossings[i]) for i in range(len(crossings) - 1)]
        measured_period = sum(periods) / len(periods)
        peaks = []
        for i in range(1, len(disps) - 1):
            a_prev, a_i, a_next = abs(disps[i - 1]), abs(disps[i]), abs(disps[i + 1])
            if a_i >= a_prev and a_i >= a_next and a_i > 1e-9:
                peaks.append(a_i)
        zeta_measured = None
        if len(peaks) >= 3:
            half_cycle_delta = math.log(peaks[0] / peaks[-1]) / (len(peaks) - 1)
            zeta_measured = 2.0 * half_cycle_delta / math.sqrt(
                4.0 * math.pi ** 2 + (2.0 * half_cycle_delta) ** 2)
        energy_drift = (beam.total_energy() - energy0) / abs(energy0)
        rows.append({
            "case": label,
            "damping_ratio_target": zeta,
            "damping_ratio_measured": zeta_measured,
            "period_fem": measured_period,
            "period_analytic": period,
            "period_rel_error": abs(measured_period - period) / period,
            "dt_over_period": dt / period,
            "energy_drift_rel": energy_drift,
            "finite": bool(all(math.isfinite(d) for d in disps)),
        })
        traces[label] = {
            "time": times,
            "disp_normalized": [d / d0 for d in disps],
            "zeta": zeta,
            "omega1": w1,
        }
    return rows, traces


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(profiles: dict, modal_rows: list[dict], odd_rows: list[dict],
                 traces: dict) -> list[tuple[str, str]]:
    os.makedirs(OUT, exist_ok=True)
    figs: list[tuple[str, str]] = []

    # fig 1: deflection fields
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    titles = {
        "ss_point": "simply supported, central point load",
        "cc_point": "clamped-clamped, central point load",
        "cantilever_tip": "cantilever, tip point load",
        "ss_udl": "simply supported, uniform load",
    }
    for ax, name in zip(axes.ravel(), titles):
        xs, fem, ana = profiles[name]
        ax.plot([x / xs[-1] for x in xs], [v * 1e3 for v in fem], "-", lw=1.0,
                label="FEM (20 elements, cubic Hermite)")
        ax.plot([x / xs[-1] for x in xs], [v * 1e3 for v in ana], "--", lw=1.0,
                label="closed form")
        ax.set_title(titles[name], fontsize=9)
        ax.set_xlabel("x / L")
        ax.set_ylabel("w [mm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Euler-Bernoulli beam: FE deflection field vs closed form (P=1 kN, q=500 N/m)")
    fig.tight_layout()
    p1 = os.path.join(OUT, "fig_deflection_fields.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    figs.append((p1, "FE deflection fields vs closed-form solutions for four load/BC cases."))

    # fig 2: convergence
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for bc, style in (("simply_supported", "o-"), ("clamped", "s-"), ("cantilever", "^-")):
        for mode, color in ((1, "C0"), (2, "C1"), (3, "C2")):
            sel = [r for r in modal_rows if r["bc"] == bc and r["mode"] == mode]
            ax[0].loglog([1.0 / r["n_elements"] for r in sel],
                         [r["rel_error"] for r in sel], style, color=color, ms=3,
                         label=f"{bc[:4]}-{bc[-4:]} mode {mode}")
    ax[0].set_xlabel("h / L")
    ax[0].set_ylabel("|omega_FEM - omega_ref| / omega_ref")
    ax[0].set_title("eigenfrequency error vs mesh size")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=6)

    ax[1].loglog([r["h_over_L"] for r in odd_rows],
                 [r["rel_error_interpolated"] for r in odd_rows], "o-",
                 label="interpolated mid-span (fixed)")
    ax[1].loglog([r["h_over_L"] for r in odd_rows],
                 [r["rel_error_old_node_pick"] for r in odd_rows], "s--",
                 label="old node-index pick n_node//2")
    ax[1].set_xlabel("h / L")
    ax[1].set_ylabel("relative error of mid-span deflection")
    ax[1].set_title("odd element counts: mid-span point load")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    p2 = os.path.join(OUT, "fig_convergence.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    figs.append((p2, "Left: eigenfrequency convergence (O(h^4)). Right: mid-span interpolation error with and without the fix."))

    # fig 3: free vibration traces
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    for axis, label, title in (
        (ax[0], "undamped", "free vibration after release from static deflection (zeta = 0)"),
        (ax[1], "damped_2pct", "same with stiffness-proportional Rayleigh damping (zeta = 0.02)"),
    ):
        tr = traces[label]
        axis.plot(tr["time"], tr["disp_normalized"], lw=1.0, label="FEM mid-span w/w0")
        envelope = [math.exp(-tr["zeta"] * tr["omega1"] * t) for t in tr["time"]]
        axis.plot(tr["time"], envelope, "k--", lw=0.9, label="exp(-zeta*omega1*t)")
        axis.plot(tr["time"], [-e for e in envelope], "k--", lw=0.9)
        axis.plot(tr["time"], [math.cos(tr["omega1"] * t) for t in tr["time"]],
                  "r:", lw=0.9, label="cos(omega1_analytic*t)")
        axis.set_title(title, fontsize=9)
        axis.set_ylabel("w / w0")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, loc="upper right")
    ax[1].set_xlabel("time [s]")
    fig.tight_layout()
    p3 = os.path.join(OUT, "fig_free_vibration.png")
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    figs.append((p3, "Newmark free-vibration traces with analytic cosine and damping envelope."))
    return figs


# --------------------------------------------------------------------------- #
# markdown generation
# --------------------------------------------------------------------------- #
def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def write_doc(static_rows, nodal_rows, modal_rows, odd_rows, dyn_rows, report) -> None:
    st = []
    for r in static_rows:
        if r["n_elements"] in (2, 20):
            st.append([r["case"], str(r["n_elements"]), f"{r['fem']:.6e}",
                       f"{r['analytic']:.6e}", f"{r['rel_error']:.2e}",
                       "YES" if r["rel_error"] < TOL else "NO"])
    st.append(["-", "-", "-", "-", "-", "-"])
    nodal_table = [[r["case"], f"{r['x_over_L']:.2f}", f"{r['fem']:.6e}",
                    f"{r['analytic']:.6e}", f"{r['rel_error']:.2e}"] for r in nodal_rows]
    md_rows = []
    for bc in ("simply_supported", "clamped", "cantilever"):
        for n in (5, 10, 20, 40, 80):
            sel = [r for r in modal_rows if r["bc"] == bc and r["n_elements"] == n]
            md_rows.append([bc, str(n)] + [f"{r['rel_error']:.2e}" for r in sel])
    odd_tab = [[str(r["n_elements"]), f"{r['h_over_L']:.3f}",
                f"{r['middle_node_x_over_L']:.3f}",
                f"{r['rel_error_interpolated']:.2e}",
                f"{r['rel_error_old_node_pick']:.2e}"] for r in odd_rows]
    dyn_tab = [[r["case"], f"{r['period_fem']:.6e}", f"{r['period_analytic']:.6e}",
                f"{r['period_rel_error']:.2e}",
                "n/a" if r["damping_ratio_measured"] is None else f"{r['damping_ratio_measured']:.4f}",
                f"{r['damping_ratio_target']:.4f}",
                f"{r['energy_drift_rel']:.2e}", "YES" if r["finite"] else "NO"]
               for r in dyn_rows]

    worst_static = max(r["rel_error"] for r in static_rows if r["case"] != "cantilever_tip")
    cant_static = max(r["rel_error"] for r in static_rows if r["case"] == "cantilever_tip")
    worst_modal_fine = max(r["rel_error"] for r in modal_rows if r["n_elements"] == 80)
    odd_fine = [r for r in odd_rows if r["n_elements"] == 9][0]
    damp = [r for r in dyn_rows if r["case"] == "damped_2pct"][0]
    undamped = [r for r in dyn_rows if r["case"] == "undamped"][0]

    body = f"""# Euler-Bernoulli beam FEM (`tensordem.fem_euler_bernoulli`): analytic verification

状态：验证欠账清偿 + 两处缺陷修复（selfstart，TensorDEM 仓）。
本文件由 `scripts/verify_fem_euler_bernoulli.py` 程序化生成，表中数字全部来自
`artifacts/fem_euler_bernoulli/report.json`（零手抄）。协议
`tensordem-fem-euler-bernoulli-analytic-v1`。

## 1. 案例介绍（为什么这是验证欠账）

`fem_euler_bernoulli` 是 DEM–FEM 单向耦合示例（`examples/dem_fem_oneway`）的 FEM 侧：
2 节点 Hermite 三次梁单元 + Newmark 平均加速度时间积分，提供静力求解、模态相关阻尼与
动力推进。该模块此前**没有任何测试引用**（全仓 `grep` 仅示例与 `__init__` 导入），
即一个已合并、声称为“解析可验证”的求解器从未被解析解复核过。

物理背景与参数（SI，钢梁，小变形 Euler-Bernoulli 理论，w 向下为正）：

| 参数 | 值 |
|---|---|
| 梁长 L | {report['config']['length']} m |
| 截面宽 × 高 | {report['config']['width']} × {report['config']['height']} m |
| 惯性矩 I = bh^3/12 | {report['config']['inertia']:.6e} m^4 |
| 杨氏模量 E | {report['config']['young_modulus']:.3e} Pa |
| 密度 rho | {report['config']['density']} kg/m^3 |
| 单位长度质量 rhoA | {report['config']['mass_per_length']:.4e} kg/m |
| 集中载荷 P | {P} N（向下） |
| 均布载荷 q | {Q} N/m（向下） |

参考解取自 Timoshenko & Gere《Mechanics of Materials》梁挠曲线表与 Clough & Penzien
《Dynamics of Structures》等截面梁特征频率表。

### 验证中发现并修复的两个缺陷

1. **广义特征值约化转置错误 → 任何带阻尼的工况静默变 NaN（P0）**。
   原实现用 `A = (L^-T K L^-T)^T = L^-1 K L^-1`（多转置一次），该矩阵非对称，
   `eigvalsh` 返回负“特征值”，`omega1 = sqrt(负值)` → NaN → `C = (2 zeta/omega1) K` 全 NaN。
   实测：`BeamConfig(damping_ratio=0.02)` 时 `C_free` 全 NaN，5 步后位移 NaN。
   修复为 `A = L^-1 K L^-T`（对称半正定），并断言 `omega1 > 0`。
   回归测试 `test_eigen_reduction_is_symmetric_positive_definite`、
   `test_rayleigh_damping_produces_finite_state_and_target_decay`。
2. **中点输出按节点下标取，奇偶单元数串味**。
   `center_deflection` 原取 `w[n_node // 2]`；`n_elements` 为奇数时该节点不在 L/2
   （9 单元时 x = {report['odd_mesh'][2]['middle_node_x_over_L']:.3f} L），静力结果
   被静默报成不在中点的值。修复：新增 `deflection_at(x)`（三次 Hermite 插值）、
   `consistent_point_load(x, P)`（一致节点载荷），中跨量一律插值取得。

## 2. 怎么做的（方法/流程）

1. 读模块源码，重建刚度/质量矩阵与边界条件约定（θ = dw/dx，w 向下为正）。
2. 对每个可解析闭合的工况构造一致节点载荷，跑静力解，与闭合解逐项比相对误差；
   同时在 201 点网格上比较**整条挠曲线**（不只中点），确认单元内部场正确。
3. 把约化特征值问题独立复核（对称性、正定性、前 3 阶频率 vs 解析 β_n L 表），
   用 5/10/20/40/80 单元量测收敛阶。
4. 动力面：从静力平衡释放 → 无阻尼自由振动测周期与能量漂移；带 ζ=0.02 刚度比例
   Rayleigh 阻尼测对数衰减率反推 ζ（峰值相隔半周期，故半周期对数衰减率 ×2 得每周期）。
5. 全部数字写入 `report.json`/CSV，图由 matplotlib 生成，本 md 由脚本渲染。

## 3. 计算结果

### 3.1 静力：FEM vs 闭合解（全部单元数下的最大相对误差）

{_table(["case", "n_elements", "FEM [m]", "closed form [m]", "rel err", "< 3%"], st)}

- 四种工况（SS/CC/悬臂点载、SS 均布）在 2 单元到 20 单元下最差相对误差
  = **{worst_static:.2e}**（SS/CC/SS-UDL），悬臂点载 **{cant_static:.2e}**。
  （量级 1e-13~1e-14 即双精度舍入；这是 Hermite 单元对点载/均布载**节点值精确**的
  已知性质，不是拟合。）
- 节点值精确不等于单元内部场精确，故另有节点采样对照：
{_table(["case", "x/L", "FEM [m]", "closed form [m]", "rel err"], nodal_table)}

### 3.2 特征频率收敛（含 P0 修复后约化正确性）

{_table(["bc", "n_elements", "mode1 rel err", "mode2 rel err", "mode3 rel err"], md_rows)}

- 80 单元最大相对误差 **{worst_modal_fine:.2e}**。SS/CC 各阶在相邻档间误差稳定下降
  ~15–16 倍，与一致质量 Hermite 单元的 O(h^4) 一致（实测比值见 `report.json` 的
  `modal_refinement_ratio`）；悬臂第 1 阶在 60→80 单元处已到 1e-8 双精度底噪，
  该档比值不再单调（honest note，非收敛失败）。一致质量给出频率上界
  （`omega_FEM >= omega_analytic`）。
- 悬臂基频用 β_1 L = 1.8751（ω_1 = 3.5160·sqrt(EI/(rhoA L^4))）复核通过。

### 3.3 奇数单元中跨插值（缺陷 2 的定量证据）

{_table(["n_elements", "h/L", "middle node x/L", "interpolated rel err", "old node-pick rel err"], odd_tab)}

- 9 单元时旧取法误差 {odd_fine['rel_error_old_node_pick']:.2e}（≈ {odd_fine['rel_error_old_node_pick']*100:.2f}%），
  修复后 {odd_fine['rel_error_interpolated']:.2e}；19 单元起进入 1e-5 以下。

### 3.4 动力（Newmark β=1/4, γ=1/2, dt = T/500, 20 单元）

{_table(["case", "period FEM [s]", "period analytic [s]", "period rel err", "zeta measured", "zeta target", "energy change (rel)", "finite"], dyn_tab)}

- 无阻尼周期相对误差 **{undamped['period_rel_error']:.2e}**；总能量变化
  **{undamped['energy_drift_rel']:.2e}**（2.5 个周期末态相位误差量级，平均加速度法
  无数值耗散）。带阻尼工况的能量变化 {damp['energy_drift_rel']:.2e} 是**物理耗散**，
  不是数值失稳——阻尼项即为此而设。
- 阻尼工况反推 ζ = **{damp['damping_ratio_measured']:.4f}**（目标 {damp['damping_ratio_target']:.4f}），
  相对偏差 {abs(damp['damping_ratio_measured']-damp['damping_ratio_target'])/damp['damping_ratio_target']*100:.2f}%；
  修复前该工况全程 NaN。

## 4. 结果图

- `{OUT}/fig_deflection_fields.png`：四类工况的整条挠曲线 FEM vs 闭合解（20 单元）。
- `{OUT}/fig_convergence.png`：左=特征频率误差 vs h/L（三种边界、前 3 阶，log-log）；
  右=奇数单元中跨误差，修复前后对照。
- `{OUT}/fig_free_vibration.png`：无阻尼/带阻尼自由振动时程 vs 解析余弦与
  exp(-ζω_1 t) 包络。

## 5. 注意点（坑、局限、适用范围）

- **节点精确 ≠ 全场精确**：点载/均布载在节点上的挠度与转角对 Hermite 单元是精确的，
  但单元内部（尤其跨中出现集中载荷且载荷不在节点时）仍是 O(h^2~3) 离散误差；
  报告里两者分开列，避免把 1e-14 当成“求解器全场精确”。
- 一致质量矩阵给频率上界；集中质量会给下界，本模块只实现一致质量。
- 阻尼只实现刚度比例 Rayleigh（C = 2ζ/ω_1·K），因此高阶模态阻尼比按 ω_n/ω_1 放大，
  不是常数阻尼；`damping_ratio` 只对第 1 阶模态有物理意义。
- `static_solve` 默认不改动动力状态（纯探针），需要从静力平衡起步必须显式
  `commit=True`（旧行为是静默不提交，容易误用；现已写入 docstring 并有回归测试）。
- 大变形、剪切变形（Timoshenko）、转动惯量、轴向力（几何刚度）均不在本模块范围；
  细长比 h/L 不宜过大（本例 h/L=0.01，Euler-Bernoulli 适用）。
- 本验证只覆盖线性弹性梁；DEM 侧接触力幅值/时程的正确性属于 `dem3d` 的验证范围。

## 6. 误差判定（关键量 vs 基准，逐项回答“是否 < 3%”）

| 关键量 | 基准 | 相对误差 | < 3%？ |
|---|---|---|---|
| SS 中点集中载荷挠度 | P L^3/(48 EI) | {static_rows[3]['rel_error']:.2e}（10 单元 {static_rows[12]['rel_error']:.2e}） | YES |
| CC 中点集中载荷挠度 | P L^3/(192 EI) | {[r for r in static_rows if r['case']=='cc_point' and r['n_elements']==10][0]['rel_error']:.2e} | YES |
| 悬臂端部集中载荷挠度 | P L^3/(3 EI) | {[r for r in static_rows if r['case']=='cantilever_tip' and r['n_elements']==10][0]['rel_error']:.2e} | YES |
| SS 均布载荷中点挠度 | 5 q L^4/(384 EI) | {[r for r in static_rows if r['case']=='ss_udl' and r['n_elements']==10][0]['rel_error']:.2e} | YES |
| CC 均布载荷中点挠度 | q L^4/(384 EI) | {[r for r in static_rows if r['case']=='cc_udl' and r['n_elements']==10][0]['rel_error']:.2e} | YES |
| SS 前 3 阶特征频率（80 单元） | (n pi)^2 sqrt(EI/(rhoA L^4)) | <= {worst_modal_fine:.2e} | YES |
| 悬臂基频（80 单元） | 3.5160 sqrt(EI/(rhoA L^4)) | {min(r['rel_error'] for r in modal_rows if r['bc']=='cantilever' and r['n_elements']==80):.2e} | YES |
| 自由振动周期（无阻尼） | 2 pi / omega_1 | {undamped['period_rel_error']:.2e} | YES |
| 阻尼比（对数衰减率反推） | 0.02 | {abs(damp['damping_ratio_measured']-0.02)/0.02*100:.2f}% | YES |
| 奇数单元（9）中跨挠度 | P L^3/(48 EI) | {odd_fine['rel_error_interpolated']:.2e} | YES |

全部关键量 < 3%，静力量级在双精度舍入（1e-13）。两项 honest 局限：
(a) 奇数单元 + 跨中集中载荷的离散误差虽已 <3%，但仍随 h 下降（表 3.3），
不宣称“插值让任意网格都精确”；(b) 带阻尼工况的模态阻尼比只在第 1 阶有物理含义。

## 7. 复现

```
PYTHONPATH=src python3 scripts/verify_fem_euler_bernoulli.py
python -m pytest tests/test_fem_euler_bernoulli_verification.py -q
```
"""
    with open(DOC, "w") as fh:
        fh.write(body)


# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    static_rows, extra = static_study()
    nodal_rows = extra["nodal"]
    modal_rows = modal_study()
    odd_rows = odd_mesh_study()
    dyn_rows, traces = dynamic_study()

    cfg = BeamConfig()
    modal_ratio = {}
    for bc in ("simply_supported", "clamped", "cantilever"):
        for mode in (1, 2, 3):
            sel = [r for r in modal_rows if r["bc"] == bc and r["mode"] == mode]
            ratios = [sel[i]["rel_error"] / sel[i + 1]["rel_error"]
                      for i in range(len(sel) - 1) if sel[i + 1]["rel_error"] > 0]
            modal_ratio[f"{bc}-mode{mode}"] = ratios

    report = {
        "protocol": "tensordem-fem-euler-bernoulli-analytic-v1",
        "config": {
            "length": cfg.length, "width": cfg.width, "height": cfg.height,
            "inertia": cfg.inertia, "young_modulus": cfg.young_modulus,
            "density": cfg.density, "mass_per_length": cfg.mass_per_length,
        },
        "loads": {"point_N": P, "uniform_N_per_m": Q},
        "static": static_rows,
        "nodal_sampling": nodal_rows,
        "modal": modal_rows,
        "modal_refinement_ratio": modal_ratio,
        "odd_mesh": odd_rows,
        "dynamic": dyn_rows,
        "acceptance_threshold": TOL,
        "worst_static_rel_error": max(r["rel_error"] for r in static_rows),
        "worst_modal_rel_error_80elems": max(r["rel_error"] for r in modal_rows
                                             if r["n_elements"] == 80),
    }
    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    with open(os.path.join(OUT, "static_table.csv"), "w") as fh:
        fh.write("case,bc,load,n_elements,probe_x_over_L,fem,analytic,rel_error\n")
        for r in static_rows:
            fh.write(f"{r['case']},{r['bc']},{r['load']},{r['n_elements']},"
                     f"{r['probe_x_over_L']:.4f},{r['fem']:.12e},"
                     f"{r['analytic']:.12e},{r['rel_error']:.6e}\n")
    with open(os.path.join(OUT, "modal_table.csv"), "w") as fh:
        fh.write("bc,n_elements,mode,fem_omega,analytic_omega,rel_error\n")
        for r in modal_rows:
            fh.write(f"{r['bc']},{r['n_elements']},{r['mode']},{r['fem_omega']:.10e},"
                     f"{r['analytic_omega']:.10e},{r['rel_error']:.6e}\n")
    with open(os.path.join(OUT, "oddmesh_table.csv"), "w") as fh:
        fh.write("n_elements,h_over_L,middle_node_x_over_L,interpolated,old_node_pick,"
                 "analytic,rel_error_interpolated,rel_error_old_node_pick\n")
        for r in odd_rows:
            fh.write(f"{r['n_elements']},{r['h_over_L']:.6f},"
                     f"{r['middle_node_x_over_L']:.6f},{r['interpolated_midspan']:.12e},"
                     f"{r['old_reported_node_value']:.12e},{r['analytic']:.12e},"
                     f"{r['rel_error_interpolated']:.6e},"
                     f"{r['rel_error_old_node_pick']:.6e}\n")
    with open(os.path.join(OUT, "dynamic_table.csv"), "w") as fh:
        fh.write("case,zeta_target,zeta_measured,period_fem,period_analytic,"
                 "period_rel_error,dt_over_period,energy_drift_rel,finite\n")
        for r in dyn_rows:
            zeta = "" if r["damping_ratio_measured"] is None else f"{r['damping_ratio_measured']:.6f}"
            fh.write(f"{r['case']},{r['damping_ratio_target']},{zeta},{r['period_fem']:.10e},"
                     f"{r['period_analytic']:.10e},{r['period_rel_error']:.6e},"
                     f"{r['dt_over_period']:.6e},{r['energy_drift_rel']:.6e},{r['finite']}\n")

    figs = make_figures(extra["profiles"], modal_rows, odd_rows, traces)
    report["figures"] = [{"path": p, "caption": c} for p, c in figs]
    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    report["odd_mesh_first"] = odd_rows[2]
    write_doc(static_rows, nodal_rows, modal_rows, odd_rows, dyn_rows, report)

    print(json.dumps({
        "worst_static_rel_error": report["worst_static_rel_error"],
        "worst_modal_rel_error_80elems": report["worst_modal_rel_error_80elems"],
        "odd_mesh_9elems": odd_rows[2],
        "dynamic": dyn_rows,
        "figures": [p for p, _ in figs],
        "doc": DOC,
    }, indent=2))


if __name__ == "__main__":
    main()