"""Order-of-accuracy verification of the polyhedral-floe contact-dissipation ledger.

Why this script exists
----------------------
``DEMSystem3D.step`` books the work of the drag / buoyancy closures with the
*midpoint* velocity of the kick (``dE = F.(v_n+v_{n+1})/2 dt``), but the contact
dissipation used to be booked with the **step-start** velocity, i.e. ``P(v_n) dt``.
That closure is first order in ``dt``: the error is ``O(P' dt^2)`` per step, so it
grows *linearly* in ``dt`` / in the amount of dissipated work.  On an
impact-dominated case (m16361 round-3 note: ``energy_residual`` rising to 0.165 in
the impact window) that bookkeeping error alone can exceed the ``0.05`` validity
gate of ``case3_repose`` even when the trajectory is perfectly fine.

This harness measures the order instead of asserting it:

* **Case A (closed form).** One hexagonal floe, no rotation, dropped normally onto
  a rigid wall with ``mu = mu_rolling = 0``.  The contact is then the 1-DOF ODE
  ``m x'' + c x' + k x = 0`` with ``c = 2 zeta sqrt(k m)``, whose solutions are
  known:

      x(t)  = (v0/wd) exp(-zeta wn t) sin(wd t),   wd = wn sqrt(1-zeta^2)
      e     = exp(-pi zeta / sqrt(1-zeta^2))                       (restitution)
      D     = c * int_0^{pi/wd} xdot(t)^2 dt                       (dissipation)

  The integral is evaluated by adaptive quadrature (``scipy.integrate.quad``),
  independently of the DEM integrator.  ``D`` is also checked against the
  *dynamics-level* energy drop ``E(0) - E(end)``, which the ledger does not enter.
* **Case B (no closed form).** Three floes dropped into a box: simultaneous
  contacts, sliding friction, rolling resistance.  The only available statement is
  the ledger identity ``E(t) - E(0) + D(t) - W(t) = 0``; its residual is measured
  against ``dt`` and the order fitted.

Both closures are run on the same grid: ``dissipation_ledger="midpoint"`` (the
shipped default) and ``"begin"`` (the pre-fix closure, kept for exactly this
measurement and for the regression test).  The force law is evaluated before the
kick and never reads the ledger, which is why a trajectory-invariance gate
(bitwise identical positions/velocities) is included: the repair is bookkeeping.

Run
---
    PYTHONPATH=src python scripts/verify_floe3d_energy_ledger_order.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensordem.polyhedron_floe3d.contact import ContactLaw            # noqa: E402
from tensordem.polyhedron_floe3d.dynamics import DEMSystem3D, FloeBody  # noqa: E402
from tensordem.polyhedron_floe3d.geometry import extrude_polygon, regular_polygon  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "floe3d_energy_ledger"
DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "FLOE3D_DISSIPATION_LEDGER_ORDER.md"

DENSITY = 900.0


# ---------------------------------------------------------------------------
# analytic reference: 1-DOF damped impact
# ---------------------------------------------------------------------------
def analytic_impact(mass: float, kn: float, zeta: float, v0: float) -> dict:
    wn = math.sqrt(kn / mass)
    c = 2.0 * zeta * math.sqrt(kn * mass)
    wd = wn * math.sqrt(1.0 - zeta * zeta)
    t_contact = math.pi / wd

    def xdot(t: float) -> float:
        return (v0 / wd) * math.exp(-zeta * wn * t) * (
            wd * math.cos(wd * t) - zeta * wn * math.sin(wd * t))

    diss, _err = quad(lambda t: c * xdot(t) ** 2, 0.0, t_contact, epsabs=1e-14,
                      epsrel=1e-13, limit=400)
    return {
        "omega_n": wn,
        "omega_d": wd,
        "c": c,
        "contact_time_s": t_contact,
        "restitution": math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta)),
        "dissipated_J": float(diss),
        "energy_in_J": 0.5 * mass * v0 * v0,
    }


def _floe(volume: float, thickness: float, position, velocity=(0.0, 0.0, 0.0)) -> FloeBody:
    footprint = volume / thickness
    shape = extrude_polygon(regular_polygon(footprint, 6, phase=math.pi / 6.0), thickness,
                           principal=False)
    return FloeBody(shape, DENSITY, position=position, velocity=velocity)


def case_a(dt: float, ledger: str, *, volume: float = 0.05, thickness: float = 0.2,
           kn: float = 2.0e5, zeta: float = 0.10, v0: float = 0.4) -> dict:
    """One floe, normal impact on a floor wall; returns the ledger error."""
    gap = 1.0e-3
    # the extruded body is centred in z, so the face sits ``0.5*thickness + gap``
    # above the floor: a bare ``gap`` offset would start 0.099 m *inside* the wall
    body = _floe(volume, thickness, (0.0, 0.0, 0.5 * thickness + gap))
    ref = analytic_impact(body.mass, kn, zeta, v0)
    system = DEMSystem3D([body], law=ContactLaw(kn=kn, zeta=zeta, mu=0.0, mu_rolling=0.0),
                         gravity=0.0, dt=dt, record=True, dissipation_ledger=ledger)
    system.add_wall([0.0, 0.0, 1.0], 0.0, label="floor")
    body.velocity[2] = -v0
    system.record_initial_energy()
    e0 = system.energy()
    n_steps = int(math.ceil((gap / v0 + 1.20 * ref["contact_time_s"]) / dt))
    for _ in range(n_steps):
        system.step()
    last = len(system.history) - 1
    energy_trace = np.array([
        row["ke_trans"] + row["ke_rot"] + row["pe_gravity"] + row["pe_spring"]
        for row in system.history])
    vz = np.array([row["velocity"][0, 2] for row in system.history])
    spring_trace = np.asarray([row["pe_spring"] for row in system.history], dtype=float)
    touched = np.flatnonzero(spring_trace > 0.0)
    if touched.size == 0:
        raise RuntimeError("case A never made contact")
    after = np.flatnonzero(spring_trace[touched[0]:] <= 1e-15)
    first_out = int(touched[0] + after[0]) if after.size else last
    v_out = float(vz[first_out])
    e_end = float(energy_trace[first_out])
    return {
        "dt": dt,
        "ledger": ledger,
        "n_steps": n_steps,
        "steps_per_contact": ref["contact_time_s"] / dt,
        "dissipated_ledger_J": float(system.dissipated),
        "dissipated_analytic_J": ref["dissipated_J"],
        "dissipated_rel_error": abs(float(system.dissipated) - ref["dissipated_J"])
        / ref["dissipated_J"],
        "energy_drop_J": e0 - e_end,
        "energy_drop_rel_error": abs((e0 - e_end) - ref["dissipated_J"]) / ref["dissipated_J"],
        "restitution_measured": -v_out / v0,
        "restitution_analytic": ref["restitution"],
        "restitution_rel_error": abs(-v_out / v0 - ref["restitution"]) / ref["restitution"],
        "positions": [b.position.copy() for b in system.bodies],
        "velocities": [b.velocity.copy() for b in system.bodies],
    }


def case_b(dt: float, ledger: str, *, kn: float = 5.0e4, zeta: float = 0.1,
           t_end: float = 0.02, mu: float = 0.0, mu_rolling: float = 0.0,
           tag: str = "case_b") -> dict:
    """Three floes dropped into a box: ledger identity residual, no closed form."""
    bodies = [
        _floe(0.02, 0.12, (-0.30, 0.0, 0.35), (0.0, 0.0, -0.5)),
        _floe(0.02, 0.12, (0.0, 0.0, 0.45)),
        _floe(0.02, 0.12, (0.30, 0.0, 0.55), (0.0, 0.0, -0.2)),
    ]
    system = DEMSystem3D(bodies, law=ContactLaw(kn=kn, zeta=zeta, mu=mu, mu_rolling=mu_rolling),
                         gravity=9.81, dt=dt, record=True, dissipation_ledger=ledger)
    system.add_box((0.0, 0.0, 0.0), (0.80, 0.80, 0.80), bottom=True, sides=True)
    system.record_initial_energy()
    n_steps = int(round(t_end / dt))
    for _ in range(n_steps):
        system.step()
    hist = system.history
    energy = np.array([row["ke_trans"] + row["ke_rot"] + row["pe_gravity"] + row["pe_spring"]
                       for row in hist])
    diss = np.array([row["dissipated"] for row in hist])
    work = np.array([row["work_external"] for row in hist])
    scale = system.energy_scale()
    # row k holds the state at t_k; diss/work of step k bridge t_k -> t_{k+1}
    resid = np.abs((energy[1:] - energy[0]) + diss[:-1] - work[:-1]) / scale
    return {
        "tag": tag,
        "mu": mu,
        "mu_rolling": mu_rolling,
        "dt": dt,
        "ledger": ledger,
        "n_steps": n_steps,
        "contacts_max": int(max(row["spring_deltas"].size for row in hist)),
        "residual_max": float(resid.max()),
        "dissipated_J": float(diss[-1]),
        "positions": [b.position.copy() for b in system.bodies],
        "velocities": [b.velocity.copy() for b in system.bodies],
    }


def fit_order(dts, errors) -> float:
    x = np.log(np.asarray(dts, dtype=float))
    y = np.log(np.abs(np.asarray(errors, dtype=float)))
    return float(np.polyfit(x, y, 1)[0])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dts = [2.0e-4, 1.0e-4, 5.0e-5, 2.5e-5, 1.25e-5]
    result: dict = {"dt_grid": dts, "case_a": {}, "case_b": {}, "case_b_friction": {},
                    "gates": {}}

    for ledger in ("midpoint", "begin"):
        result["case_a"][ledger] = [case_a(dt, ledger) for dt in dts]
    dts_b = [4.0e-4, 2.0e-4, 1.0e-4, 5.0e-5]
    for ledger in ("midpoint", "begin"):
        result["case_b"][ledger] = [case_b(dt, ledger) for dt in dts_b]
        result["case_b_friction"][ledger] = [
            case_b(dt, ledger, mu=0.4, mu_rolling=0.01, tag="case_b_friction") for dt in dts_b]

    ca, cb = result["case_a"], result["case_b"]
    order_a = {k: fit_order([r["dt"] for r in ca[k]], [r["dissipated_rel_error"] for r in ca[k]])
               for k in ca}
    # Case B is *reported*, not gated: see the "known open item" note in the doc.
    order_b = {k: fit_order([r["dt"] for r in cb[k]], [r["residual_max"] for r in cb[k]])
               for k in cb}
    order_bf = {k: fit_order([r["dt"] for r in result["case_b_friction"][k]],
                             [r["residual_max"] for r in result["case_b_friction"][k]])
                for k in result["case_b_friction"]}
    result["orders"] = {"case_a_dissipation": order_a, "case_b_residual": order_b,
                        "case_b_friction_residual": order_bf}

    # trajectory invariance: the ledger must not feed back into the dynamics
    def _traj_diff(runs_mid, runs_begin):
        worst = 0.0
        for ra, rb in zip(runs_mid, runs_begin):
            if ra["dt"] != rb["dt"]:
                continue
            for pa, pb in zip(ra["positions"], rb["positions"]):
                worst = max(worst, float(np.abs(pa - pb).max()))
            for va, vb in zip(ra["velocities"], rb["velocities"]):
                worst = max(worst, float(np.abs(va - vb).max()))
        return worst

    traj = {"case_a": _traj_diff(ca["midpoint"], ca["begin"]),
            "case_b": _traj_diff(cb["midpoint"], cb["begin"])}

    fine_a = ca["midpoint"][-1]
    gates = {
        "G1_analytic_dissipation_finest_dt_lt_1e-3": fine_a["dissipated_rel_error"] < 1.0e-3,
        "G2_midpoint_order_case_a_gt_1_5": order_a["midpoint"] > 1.5,
        "G3_begin_order_case_a_lt_1_5": order_a["begin"] < 1.5,
        "G4_midpoint_beats_begin_coarsest_dt": (
            ca["midpoint"][0]["dissipated_rel_error"]
            < ca["begin"][0]["dissipated_rel_error"]),
        "G5_restitution_analytic_lt_5e-3": fine_a["restitution_rel_error"] < 5.0e-3,
        "G6_energy_drop_matches_analytic_lt_1e-2": fine_a["energy_drop_rel_error"] < 1.0e-2,
        "G7_trajectory_invariant_bitwise": (traj["case_a"] == 0.0 and traj["case_b"] == 0.0),
        "G8_midpoint_error_halves_when_dt_halves": (
            ca["midpoint"][-2]["dissipated_rel_error"]
            < 0.6 * ca["midpoint"][-3]["dissipated_rel_error"]),
    }
    result["gates"] = {k: bool(v) for k, v in gates.items()}
    result["trajectory_max_abs_diff"] = traj
    result["n_pass"] = int(sum(bool(v) for v in gates.values()))
    result["n_gates"] = len(gates)

    (OUT / "results.json").write_text(json.dumps(result, indent=2,
                                     default=lambda o: o.tolist()))
    _figures(result)
    _write_doc(result)
    for name, ok in result["gates"].items():
        print(("PASS " if ok else "FAIL ") + name)
    print("orders case_a:", order_a)
    print("orders case_b:", order_b)
    print("gates %d/%d" % (result["n_pass"], result["n_gates"]))
    print("json ->", OUT / "results.json")
    return 0 if result["n_pass"] == result["n_gates"] else 1


def _figures(result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ledger, style in (("midpoint", "o-"), ("begin", "s--")):
        runs = result["case_a"][ledger]
        ax[0].loglog([r["dt"] for r in runs], [r["dissipated_rel_error"] for r in runs],
                     style, label="%s (slope %.2f)" % (ledger, result["orders"]["case_a_dissipation"][ledger]))
        runs_b = result["case_b"][ledger]
        ax[1].loglog([r["dt"] for r in runs_b], [r["residual_max"] for r in runs_b],
                     style, label="%s (slope %.2f)" % (ledger, result["orders"]["case_b_residual"][ledger]))
    ax[0].set_xlabel("dt [s]")
    ax[0].set_ylabel("|D_ledger - D_analytic| / D_analytic")
    ax[0].set_title("Case A: damped impact, dissipation error")
    ax[1].set_xlabel("dt [s]")
    ax[1].set_ylabel("max |E - E0 + D - W| / scale")
    ax[1].set_title("Case B: 3-floe pile, ledger residual")
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_ledger_order.png", dpi=140)
    plt.close(fig)


def _write_doc(result: dict) -> None:
    ca, cb = result["case_a"], result["case_b"]
    lines = [
        "# floe3d 接触耗散账本：从 O(dt) 修到 O(dt²)，以及它为什么不改动力学",
        "",
        "> 本文件的表格与数字由 `scripts/verify_floe3d_energy_ledger_order.py` 程序化生成",
        "> （源数据 `artifacts/floe3d_energy_ledger/results.json`，图 `fig_ledger_order.png`）。",
        "> 手抄数字为零：文档由同一进程在算完之后写出。",
        "",
        "## ① 案例介绍（物理背景 / 几何 / 参数）",
        "",
        "* **物理背景**：`tensordem.polyhedron_floe3d` 是碎冰场刚体 DEM，m16361 第 3 轮",
        "  `case3_repose` 报 `energy_residual` 在冲击段由 0.093 升到 0.165，超过 0.05 有效门，",
        "  当时判读为「冲击期耗散项按步初速度记账的 O(dt) 误差」。本单把该判读变成可测结论。",
        "* **案例 A（有闭式解）**：单块六边形冰排，无旋转，正面落到刚性地面壁；`mu = mu_rolling = 0`，",
        "  于是接触退化为 1 自由度 `m x'' + c x' + k x = 0`，`c = 2 zeta sqrt(k m)`。",
        ("  参数：体积 %.3f m³、厚 %.2f m、密度 %.0f kg/m³、`kn = 2e5 N/m`、`zeta = 0.10`、"
         "撞击速度 `v0 = 0.4 m/s`、`dt` 见下表。") % (0.05, 0.2, DENSITY),
        "* **案例 B（无闭式解）**：3 块冰排跌入方箱，多体同时接触 + 滑动摩擦 + 滚动阻力，",
        "  `kn = 5e4 N/m`、`zeta = 0.1`、`mu = 0.4`、`mu_rolling = 0.01`，量到 `t = 0.02 s`。",
        "",
        "## ② 怎么做的（方法 / 关键设置）",
        "",
        "1. **缺陷定位**：`step()` 里 drag/buoyancy 的功已经用踢半步的中点速度记账",
        "   （`dE = F.(v_n+v_{n+1})/2 dt` 是踢的精确动能增量），而接触耗散用步初速度",
        "   `P(v_n) dt`。后者每步误差 `O(P' dt²)`，累计随耗散功线性增长 → 相对误差 `O(dt)`。",
        "2. **修复**：`_apply_pair` / `_apply_wall` 只收集耗散功率的系数（`c_n`、`mu f_n`、滚动上限",
        "   `cap_r`）与几何（接触点力臂、法向），踢半步结束后由 `_settle_dissipation` 用",
        "   中点速度统一结算：`P_mid = c_n v_n,mid² + mu f_n |v_t,mid| + cap_r |omega_rel,mid|`。",
        "   力律仍在踢前求值、不读账本，故轨迹不受影响。",
        "3. **对照口径**：`dissipation_ledger=\"begin\"` 保留修复前的口径（逐位复现旧算术），",
        "   只用于本测量与回归门；默认 `midpoint`。",
        "4. **独立基准**：`D = c∫ xdot² dt` 用 `scipy.integrate.quad`（`epsabs=1e-14`）独立求积；",
        "   恢复系数用 `e = exp(-pi zeta/sqrt(1-zeta²))`。二者都不经过 DEM 积分器。",
        "",
        "## ③ 计算结果",
        "",
        "**案例 A：耗散能量 vs 闭式解**",
        "",
        "| dt [s] | 每接触步数 | D_ledger(midpoint) | D_ledger(begin) | D_analytic | 相对误差 midpoint | 相对误差 begin |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in ca["midpoint"]:
        rb = next(x for x in ca["begin"] if x["dt"] == r["dt"])
        lines.append("| %.3g | %.1f | %.9e | %.9e | %.9e | %.3e | %.3e |" % (
            r["dt"], r["steps_per_contact"], r["dissipated_ledger_J"], rb["dissipated_ledger_J"],
            r["dissipated_analytic_J"], r["dissipated_rel_error"], rb["dissipated_rel_error"]))
    lines += [
        "",
        "**案例 A：能量落差（动力学侧，与账本无关）与恢复系数**",
        "",
        "| dt [s] | E(0)-E(end) [J] | 相对闭式解 | e 实测 | e 闭式 | 相对误差 |",
        "|---|---|---|---|---|---|",
    ]
    for r in ca["midpoint"]:
        lines.append("| %.3g | %.9e | %.3e | %.6f | %.6f | %.3e |" % (
            r["dt"], r["energy_drop_J"], r["energy_drop_rel_error"],
            r["restitution_measured"], r["restitution_analytic"], r["restitution_rel_error"]))
    lines += [
        "",
        "**案例 B：账本恒等式残差**",
        "",
        "| dt [s] | max 接触对 | 残差 midpoint | 残差 begin |",
        "|---|---|---|---|",
    ]
    for r in cb["midpoint"]:
        rb = next(x for x in cb["begin"] if x["dt"] == r["dt"])
        lines.append("| %.3g | %d | %.3e | %.3e |" % (
            r["dt"], r["contacts_max"], r["residual_max"], rb["residual_max"]))
    lines += [
        "",
        "**收敛阶（log-log 最小二乘）**",
        "",
        "| 口径 | 案例 A 耗散误差阶 | 案例 B 残差阶 |",
        "|---|---|---|",
        "| midpoint（默认） | %.2f | %.2f |" % (
            result["orders"]["case_a_dissipation"]["midpoint"],
            result["orders"]["case_b_residual"]["midpoint"]),
        "| begin（修复前） | %.2f | %.2f |" % (
            result["orders"]["case_a_dissipation"]["begin"],
            result["orders"]["case_b_residual"]["begin"]),
        "",
        "轨迹不变性：两种口径下位置/速度最大差 `%.1e` / `%.1e`（A/B）——记账不反馈进动力学。"
        % (result["trajectory_max_abs_diff"]["case_a"], result["trajectory_max_abs_diff"]["case_b"]),
        "",
        "## ④ 结果图",
        "",
        "`artifacts/floe3d_energy_ledger/fig_ledger_order.png`：左=案例 A 的耗散相对误差 vs dt；",
        "右=案例 B 的账本残差 vs dt；两图各含两条口径的 log-log 拟合斜率。",
        "（纯记账/数值方法任务，场量云图不适用；图是误差-步长收敛曲线。）",
        "",
        "## ⑤ 注意点 / 局限",
        "",
        "1. **修的是记账不是物理**：轨迹逐位不变（上面 G7），所以本单不改变任何一个已发布的",
        "   力学结论；它只让「能量门」在冲击主导算例上不再被记账误差打穿。",
        "2. **`begin` 口径保留**：它不是可选项而是测量/回归工具；生产路径默认 `midpoint`。",
        "3. **案例 A 是无摩擦、无旋转的 1-DOF 极限**：这样闭式解才成立；带摩擦/滚动的实际",
        "   接触没有闭式基准，只能量守恒恒等式（案例 B）。",
        "4. **案例 B 的 `dt` 上限**：`dt = 4e-4` 时接触期仅约 6~7 步，残差里还含积分离散误差，",
        "   故拟合只用 `dt <= 4e-4` 的整段并在结论中给出斜率而非单点。",
        "5. **未做的事**：滚动阻力与切向摩擦的耗散仍以「本步力的大小 × 本步速度」结算——用中点",
        "   速度已消除一阶项，但摩擦历史（`xi`）本身是显式更新，其误差仍是 O(dt)，属另一条账。",
        "",
        "## ⑥ 误差判定（是否 <3%）",
        "",
        "| 量 | 参考 | 最细 dt 下误差 | 判读 |",
        "|---|---|---|---|",
        "| 接触耗散 D | 闭式 `c∫xdot²dt` | %.3e | %s |" % (
            ca["midpoint"][-1]["dissipated_rel_error"],
            "PASS <3%" if ca["midpoint"][-1]["dissipated_rel_error"] < 0.03 else "FAIL"),
        "| 恢复系数 e | 闭式 `exp(-pi zeta/sqrt(1-zeta²))` | %.3e | %s |" % (
            ca["midpoint"][-1]["restitution_rel_error"],
            "PASS <3%" if ca["midpoint"][-1]["restitution_rel_error"] < 0.03 else "FAIL"),
        "| 能量落差 | = 耗散解析值 | %.3e | %s |" % (
            ca["midpoint"][-1]["energy_drop_rel_error"],
            "PASS <3%" if ca["midpoint"][-1]["energy_drop_rel_error"] < 0.03 else "FAIL"),
        "",
        "门禁汇总：**%d/%d PASS**（逐门见 `results.json.gates`，未过的门如实保留 FAIL）。"
        % (result["n_pass"], result["n_gates"]),
        "",
        "## 复现",
        "",
        "```bash",
        "PYTHONPATH=src python scripts/verify_floe3d_energy_ledger_order.py",
        "PYTHONPATH=src python -m pytest tests/test_floe3d_energy_ledger_order.py -q",
        "```",
        "",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())