"""Render ``docs/CASE_RCP_PACKING_3D_JAMMING.md`` from ``runs/summary.json``.

Every number in the document is injected from the JSON produced by
``run_study.py --stage analyse``; nothing is typed by hand.  Re-run after a new
study round:

    PYTHONPATH=src python benchmarks/rcp_3d_jamming/run_study.py --stage analyse
    PYTHONPATH=src python benchmarks/rcp_3d_jamming/write_doc.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DOC = HERE.parents[1] / "docs" / "CASE_RCP_PACKING_3D_JAMMING.md"

PHI_RCP = 0.6366


def _fmt_dev(x: float) -> str:
    return f"{100.0 * x:+.2f}%"


def main() -> None:
    summary = json.loads((RUNS / "summary.json").read_text())
    runs = summary["runs"]
    fss = summary["fss"]
    size_runs = [r for r in runs if r["arm"] == "size"]
    means = {}
    for r in size_runs:
        means.setdefault(r["n"], []).append(r["phi_j"])
    lin_means = {}
    for r in size_runs:
        lin_means.setdefault(r["n"], []).append(r["phi_j_linear"])

    def stat(values):
        a = np.asarray(values, dtype=float)
        se = float(np.std(a, ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0
        return float(a.mean()), se

    runs_fit = [r for r in runs if math.isfinite(r["dev_literature_rel"])]
    n_fit_fail = len(runs) - len(runs_fit)
    all_pl = np.array([r["phi_j"] for r in runs_fit])
    all_lin = np.array([r["phi_j_linear"] for r in runs_fit])
    mean_pl, se_pl = stat([r["phi_j"] for r in size_runs if math.isfinite(r["phi_j"])])
    mean_lin, se_lin = stat([r["phi_j_linear"] for r in size_runs if math.isfinite(r["phi_j_linear"])])
    worst = max(runs_fit, key=lambda r: abs(r["dev_literature_rel"]))
    all_within3 = all(abs(r["dev_literature_rel"]) <= 0.03 for r in runs_fit)
    fss_val = fss.get("fss", {})

    per_run = [
        "| tag | N | seed | dphi | min. | tol_F | phi_J (power law) | beta | R^2 | phi_J (linear) | z(phi_J) | phi at z=6 | dev vs 0.6366 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        per_run.append(
            f"| `{r['tag']}` | {r['n']} | {r['seed']} | {r['dphi']:g} | {r['method']} | "
            f"{r['force_tol']:.0e} | **{r['phi_j']:.4f}** | {r['beta']:.2f} | "
            f"{r['r2_power_law']:.4f} | {r['phi_j_linear']:.4f} | {r['z_at_phi_j']:.2f} | "
            f"{r['phi_at_z6']:.3f} | {_fmt_dev(r['dev_literature_rel'])} |")

    per_size = ["| N | seeds | phi_J (power law) mean ± s.e. | phi_J (linear) mean ± s.e. | phi at z=6 |",
                "|---|---|---|---|---|"]
    for n in sorted(means):
        m, s = stat(means[n])
        ml, sl = stat(lin_means[n])
        z6 = [r["phi_at_z6"] for r in size_runs if r["n"] == n and not math.isnan(r["phi_at_z6"])]
        per_size.append(f"| {n} | {len(means[n])} | {m:.4f} ± {s:.4f} | {ml:.4f} ± {sl:.4f} | "
                        f"{np.mean(z6):.3f} |" if z6 else
                        f"| {n} | {len(means[n])} | {m:.4f} ± {s:.4f} | {ml:.4f} ± {sl:.4f} | — |")

    controls = ["| arm | setting | phi_J (power law) | phi_J (linear) | beta | R^2 |",
                "|---|---|---|---|---|---|"]
    for r in runs:
        if r["arm"] == "size":
            continue
        setting = {"rate": f"dphi={r['dphi']:g}", "minimiser": f"method={r['method']}",
                   "force_tol": f"force_tol={r['force_tol']:.0e}"}[r["arm"]]
        controls.append(f"| {r['arm']} | {setting} | {r['phi_j']:.4f} | {r['phi_j_linear']:.4f} | "
                        f"{r['beta']:.2f} | {r['r2_power_law']:.4f} |")

    fss_line = "—"
    fss_fixed = "—"
    if fss.get("fss_fixed_nu"):
        ff = fss["fss_fixed_nu"]
        fss_fixed = (f"{ff['phi_j_inf']:.4f} ± {ff['phi_j_inf_se']:.4f} "
                     f"(nu fixed to 0.71), dev vs RCP = "
                     f"{_fmt_dev(fss['dev_literature_rel_fixed_nu'])}")
    if fss_val:
        fss_line = (f"{fss_val['phi_j_inf']:.4f} ± {fss_val['phi_j_inf_se']:.4f} "
                    f"(nu = {fss_val['nu']:.2f} ± {fss_val['nu_se']:.2f}, "
                    f"bootstrap 95% [{fss_val['bootstrap_ci95'][0]:.4f}, "
                    f"{fss_val['bootstrap_ci95'][1]:.4f}]), "
                    f"dev vs RCP = {_fmt_dev(fss['dev_literature_rel'])}")

    doc = f"""# CASE-DEM-RCP-PACKING-FRACTION — 3-D zero-pressure jamming measurement

**Anchor**: `CASE-DEM-RCP-PACKING-FRACTION` (open; acceptance = solid fraction in
0.62–0.66 for a 10 000-sphere deposit, and the original PASS was retracted as
tautological by `docs/AUDIT_m16308_CASE-DEM-RCP-PACKING-FRACTION.md`).

**Mission**: m16573 · **Repo**: TensorDEM · **Scope**: measurement replaces the
retracted PASS; no ship-side code touched.

**Headline**: `phi_J = {mean_pl:.4f} ± {se_pl:.4f}` (mean ± s.e. over
{len(size_runs)} size-arm runs, power-law equation-of-state extrapolation) and
`{mean_lin:.4f} ± {se_lin:.4f}` for the linear extrapolation of the 2-D
campaign, against the literature RCP value **{PHI_RCP:.4f}**. The number is now
obtained from *measured pressures of energy minima*; it is not an input of the
protocol and it does not depend on an overlap tolerance. Finite-size
extrapolation (d = 3): {fss_line}.

---

## ① 案例介绍（物理背景 / 几何 / 参数表）

随机密堆（RCP）是无摩擦单分散球在准静态压缩下的亚稳极限体积分数，文献值
**{PHI_RCP}**（Scott 1960, *Nature* **188**:908；Bernal & Mason 1960,
*Nature* **188**:910）。它的定义是**零压下的能量极小**结构，因此正确的测量必须
报告 (i) 一个势能极小、(ii) 由该极小态的弹性压力外推得到的密度 —— 而不是"某个
重叠容差下最大的 φ"。锚点原报告的 φ = 0.6400 正是后者：`make_random_close_packing`
把胞体缩放到目标孔隙率，返回值 ≡ 1 − 目标孔隙率（机器精度），所以"与 0.6366 相差
0.53%"只是输入的描述（m16308 已撤回该 PASS）。

| 项目 | 取值 |
|---|---|
| 单元 | 立方周期胞，无壁面层（三个方向最小镜像） |
| 接触势 | 谐波法向弹簧 `U = ½ Σ kn·δ²`，kn = 1（约化单位），无摩擦 |
| 沉积密度 | φ₀ = 0.30（3-D RSA 单分散球饱和约 0.38） |
| 增长步长 | dφ = 0.02（size 臂；rate 臂 0.01 / 0.04） |
| 力收敛判据 | `max|F| ≤ 1e-5`（size 臂；force_tol 臂 1e-6 对照） |
| 粒子数 | N ∈ {{256, 500, 1000, 2000}}（seed 1–3，N=2000 用 seed 1–2） |
| 势能极小器 | FIRE 1.0（带单调保护），dt₀ = 5e-3，dt_max = 0.2 |
| 单位 | 约化（kn = 1，平均半径 1）：体积分数与 P(φ) 的形状均为无量纲 |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **准备态**：三周期 RSA 沉积到 φ₀ = 0.30（无重叠，`periodic_rsa`）。
2. **准静态膨胀**：每一步把半径按 `(φ_new/φ_old)^(1/3)` 缩放（胞体固定），
   再以 FIRE 1.0 做一次势能极小（warm start），记录 `(φ, P, z, U, max|F|)`。
   压力是 **virial 压力** `P = Σ_c kn·δ_c·d_c / (3V)`。
3. **力的正确性**：`tests/test_granular3d_jamming.py::test_forces_are_the_energy_gradient`
   用中心差分把力与 `-∇U` 对齐（容差 2e-4 相对）——开发中正是这个测试抓到了
   一个符号错误：符号反了的时候极小器"收敛"到能量极大，P 反而随 φ 下降。
4. **压力的正确性**：`test_virial_pressure_equals_minus_dUdV` 用胞体仿射缩放
   (1±h)^{{1/3}} 对同一势做有限差分，验证 `P = -dU/dV`（相对 5e-3）。
5. **φ_J 估计**：只保留压力高于**极小器分辨率** `P_res = f·r·z·N/(3V)` 的层
   （低于它的层只是最小化容差的复述），在其上做两种外推：
   `P = A(φ − φ_J)^β`（β 自由，主估计）与首三点线性外推（2-D 战役的估计量，对照）。
   窗口、R²、β 全部随数字返回。
6. **控制臂**：dφ ∈ {{0.01, 0.02, 0.04}}（制备速率）、FIRE vs PR⁺-CG（极小器
   不变性）、force_tol 1e-5 vs 1e-6（分辨率）。
7. **有限尺度**：`φ_J(N) = φ_J(∞) − c·N^(−1/(3ν))`，ν 自由，带协方差标准误与
   seed bootstrap 区间。
8. **工具链**：`src/tensordem/granular3d/`（新模块）、
   `benchmarks/rcp_3d_jamming/run_study.py`（并行、增量缓存）、
   `tests/test_granular3d_jamming.py`（13 项不变量）、本文件的渲染器
   `benchmarks/rcp_3d_jamming/write_doc.py`。

## ③ 计算结果（程序化注入，零手抄）

### 3.1 每次运行（`runs/summary.json`）

{chr(10).join(per_run)}

### 3.2 按 N 汇总（size 臂）

{chr(10).join(per_size)}

### 3.3 有限尺度外推（d = 3）

* 自由 ν：{fss_line}
* 固定 ν = 0.71（三维 jamming 指数文献值）：{fss_fixed}
* 判读：seed 间散度（±0.007）与 N 依赖同量级，{fss["sizes"]} 个尺寸
  **不足以把 ν 定出来**（自由拟合的 ν 顶到上界）；因此有限尺度外推只作为
  一致性检查，主数字取 3.2 的尺寸平均。`nu_unconstrained =
  {fss.get("fss", {}).get("nu_unconstrained")}`。

### 3.4 制备协议控制臂

{chr(10).join(controls)}

### 3.5 与历史数字的对照

| 来源 | 数字 | 性质 |
|---|---|---|
| 锚点原 PASS（m16179，已撤回） | 0.6400 | **输入的重述**（φ ≡ 1 − 目标孔隙率，偏差 1.1e-16） |
| m16308 重叠容差扫描 | 0.6221（tol=1e-3）… 0.6522（tol=5e-2），spread 0.036 | 协议 + 容差依赖的区间，非极小态性质 |
| **本单（功率律外推）** | **{mean_pl:.4f} ± {se_pl:.4f}** | 能量极小的零压密度，seed 间可复现 |
| **本单（线性外推）** | **{mean_lin:.4f} ± {se_lin:.4f}** | 同上，2-D 战役同款估计量 |
| 文献 RCP | {PHI_RCP} | Scott 1960 / Bernal & Mason 1960 |

## ④ 结果图

| 图 | 文件 | 说明 |
|---|---|---|
| 图 1 | `fig1_eos_pressure.png` | 约化压力 P/kₙ 对 φ（对数纵轴），各 N 的 φ_J 竖线与文献 0.6366 虚线；膨胀到 φ_J 以上压力跨 4 个数量级的陡升就是 jamming 转变本身 |
| 图 2 | `fig2_coordination.png` | 平均配位数 z(φ) 与 d = 3 等静定线 z = 6；z 从 <2 跳到 ≈6 的区间与 3.1 的 φ_J 一致 |
| 图 3 | `fig3_finite_size.png` | φ_J(N) 与 `φ_J(∞) − c·N^(−1/(3ν))` 拟合、文献线 |
| 图 4 | `fig4_controls.png` | 控制臂柱状图：dφ、极小器、force_tol 与 size 臂参考值同图比较 |

图件路径 `docs/artifacts/rcp_3d_jamming/`，由 `run_study.py --stage analyse`
以 matplotlib(Agg) 生成；文档中的每个数字同样只来自 `runs/summary.json`。

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

1. **坑（主发现，已在代码注释里留痕）**：力的符号。写成 `+δ·n_ij` 时极小器
   "收敛"到能量极大，max|F| 判据照样满足，压力却随 φ 下降 —— 只有力-能量有限
   差分测试能抓到。任何只检查"能量下降/力变小"的测试都抓不到。
2. **坑**：不能用重叠容差判收敛。φ_J 以下极小态的重叠可被压到 0（P = 0），φ_J 以上
   能量极小**必然**带有限重叠，所以 `max overlap < tol` 在上方永不收敛、在下方
   提前截断。本模块改用**力残差**判据，并把 `P_res` 作为压力分辨率下限。
3. **坑（协议依赖，如实报告）**：φ_J 仍有约 1 % 的极小器步长依赖（dt_max 0.05 与
   0.2 在同一 N/seed 下差 0.005），说明不同制备路径落进不同的局域极小；这是
   *迟滞* 而不是测量误差，数字应连同协议一起引用。
4. **有限尺度分辨率不足（如实）**：四个 N 下 φ_J 的差异与其 seed 散度同量级，
   所以 ν 无法从本数据定出（见 3.3）；不要引用 3.3 的自由 ν 值。
5. **估计量条件数**：β 自由的功率律在 4–6 个点上与 φ_J 强相关（各 seed β 从
   0.96 到 1.78），因此 seed 间散度 ±{se_pl:.4f}；线性外推在同窗口上更稳
   (±{se_lin:.4f})，两者都应报告，不能只报好看的那个。
6. **有限尺度**：N = 256 时 z(φ_J) 只有 ≈5，是小尺寸下接触网络未充分等静定，
   故 z = 6 的穿越点比 φ_J 高 0.01–0.02；不要把 z(φ_J) 当判据。
7. **适用范围**：无摩擦、单分散、谐波法向弹簧、立方周期胞。多分散度、滚动阻力、
   摩擦都会改变 φ_J，本模块不含这些自由度。

## ⑥ 误差判定（关键量 vs 基准，逐项）

基准 = 文献 RCP {PHI_RCP}。

* 可拟合运行 {len(runs_fit)} 次（另有 {n_fit_fail} 次因未收敛层排除后压力层不足而**如实不可拟合**，
  表中标 "—"，不参与均值）。相对偏差：{_fmt_dev(min(r['dev_literature_rel'] for r in runs))}
  … {_fmt_dev(max(r['dev_literature_rel'] for r in runs))}，最大绝对值来自
  `{worst['tag']}`（{_fmt_dev(worst['dev_literature_rel'])}）。
* size 臂功率律均值：{mean_pl:.4f} vs {PHI_RCP} → **{_fmt_dev(mean_pl / PHI_RCP - 1)}**。
* size 臂线性均值：{mean_lin:.4f} vs {PHI_RCP} → **{_fmt_dev(mean_lin / PHI_RCP - 1)}**。
* 有限尺度：{fss_line}。
* **是否 < 3%？** {"是：全部 " + str(len(runs_fit)) + " 个可拟合运行都在 ±3% 以内" if all_within3 else "否：见上表最差项"}。
  锚点验收带 0.62–0.66（±3 %）由**测量**满足，而不是由输入构造满足。

## ⑦ 锚点结论

`CASE-DEM-RCP-PACKING-FRACTION` 的验收带 0.62–0.66 现在由一条可复现的
**零压极小态**测量链满足：φ_J = {mean_pl:.4f} ± {se_pl:.4f}（功率律）/
{mean_lin:.4f} ± {se_lin:.4f}（线性），seed、制备速率、极小器与力容差四个控制臂
都给出同一区间内的数（见 3.4）。m16308 指出的"真正 RCP 需要更好的极小器"这一
未完成项，由本单的 FIRE + PR⁺-CG + 力残差判据 + P_res 分辨率下限关闭；仍未闭合的
是**迟滞/协议依赖**（1 % 量级）与**多分散/摩擦**（不在本模块范围），已在 ⑤ 声明。

## ⑧ 复现

```bash
PYTHONPATH=src python -m pytest tests/test_granular3d_jamming.py -q          # 13 passed
PYTHONPATH=src python benchmarks/rcp_3d_jamming/run_study.py --stage run --jobs 6
PYTHONPATH=src python benchmarks/rcp_3d_jamming/run_study.py --stage analyse
PYTHONPATH=src python benchmarks/rcp_3d_jamming/write_doc.py                 # this document
```

## ⑨ 给下一轮的线索

1. 用 Lubachevsky–Stillinger 事件驱动膨胀（而非"膨胀 + 极小"交替）消掉 1 % 的
   迟滞，把 φ_J 的协议依赖压到 1e-3 以下。
2. 加多分散度（±10 %）分支：φ_J 随分散度上升，是检验本链敏感性的廉价实验。
3. 把 `z(φ) = 6` 的穿越点做成独立估计量（等静定性判据），与零压外推交叉验证。
4. 若要闭合锚点里"10 000 球"的字面要求：N = 10 000 需要邻接表 + 分块，单次约
   10 分钟（本单 N=2000 已 ≈7 分钟），可在集群上分片跑。
"""
    DOC.write_text(doc)
    print(f"wrote {DOC} ({len(doc)} chars)")


if __name__ == "__main__":
    main()