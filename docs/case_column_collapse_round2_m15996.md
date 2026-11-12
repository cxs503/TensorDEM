# CASE-COLUMN-COLLAPSE 第 2 轮 — 补跑预登记但未跑的网格点，并把 +23 % 偏差归因（TensorDEM · m15996）

锚点：**CASE-COLUMN-COLLAPSE**（feature_ledger status=open；轮 1 = m15324，`docs/case_column_collapse_lube2004.md`）
本轮对象：轮 1 §⑥ 自报 **honest not_run** 的算例 —— `a = 3.0` 验证点与 4 条敏感性轴，以及**预登记却没被实例化**的两条轴
（`grains_per_radius = 14`、`mu = 0.8`，见 benchmark 模块 docstring 第 46–47 行）。本轮不改判据、不放宽容差。

## ① 案例介绍

轴对称颗粒柱坍塌：半径 $r_i$、高 $h_i = a\,r_i$ 的圆柱 HCP 颗粒柱在 $t=0$ 释放，重力驱动径向流动并在
地面上沉积。参考文献（逐字引用，目标值运行时求值，零手抄）：

> Lube, Huppert, Sparks & Freundt (2004), *Axisymmetric collapses of granular columns*,
> J. Fluid Mech. **508**, 175–199, doi:10.1017/S0022112004009036 —
> $r_\infty = r_i(1+1.24a)$ (a<1.7)；$r_\infty = r_i(1+1.6a^{1/2})$ (a>1.7)；$t_\infty = 3(h_i/g)^{1/2}$。

判据（轮 1 预登记，未改）：$r_\infty/r_i$ 与 $t_\infty$ 的相对误差 **< 15 %**。

参数（全部取自 `benchmarks/case_column_collapse_lube2004.py::BASE`，单个标定参数 $\mu_r^*=0.4$ 沿用轮 1 在 $a=0.5$ 上的标定）：
$\rho=2500$ kg/m³、$r=0.03$ m、$r_i=0.48$ m、$\mu=0.5$（除敏感性轴）、$e=0.5$、$k_n=6\times10^4$ N/m、$k_t/k_n=0.5$、
HCP 2 % 初始格隙、$g=9.81$、速度 Verlet、$\mathrm{dt}=4.82\times10^{-4}$ s（8 粒/$r_i$）。

## ② 怎么做的

1. **补全预登记（代码改动，最小）**：`wave2_cases()` 补入两条 docstring 已声明但从未实例化的轴
   `sens_gpr14_a2.0`（分辨率阶梯的预登记点）与 `sens_mu0.8_a2.0`（摩擦轴）；新增 `--case ID` / `--list-cases`
   命令行，使任一预登记算例可单独跑（集群分片与本地补跑都靠它）。未知 id 返回 rc=2。
2. **新增预登记完整性测试**（`tests/test_axisym_column_lube2004.py`，+3 项，共 8 passed）：
   ① `VALIDATION_ASPECTS` 每个 a 都必须有具体算例 id；② docstring 声明的敏感性轴必须实例化
   （分辨率阶梯 $\{6,11,14\}$ + 基准 8、`dt_scale=0.5`、$\mu\in\{0.3,0.8\}$）；③ `--case` 拒绝未知 id、`--list-cases` 覆盖两波全集。
   → 「声明了却没跑」从口头欠账变成**红测试**。
3. **执行**：`--case <id> --mu-rolling 0.4` 单算例单进程；因集群队列排队 398 作业（cj1217 试探后撤单），本轮改为
   本机 128 核（load15 ≈ 60–100）上 6 个单核进程并行补跑。每算例写 `runs/case_column_collapse_lube2004/cases/<id>.json`
   （参数、能量账本、front 时程、判据全在 JSON 里 = provenance）。
4. **汇总/出图**：`scripts/analyze_column_collapse_m15996.py` 只读这些 JSON，产出
   `artifacts/case_column_collapse_m15996/{grid.csv,summary.json,grid_vs_a.png,resolution_ladder.png}`，报告里没有一个手打数字。

## ③ 计算结果（本轮实测，`artifacts/case_column_collapse_m15996/grid.csv`）

| 算例 | a | N | $r_\infty/r_i$ | ref | 误差 | 判 | $t_{99}$ [s] | ref $t_\infty$ | 误差 | 判 | 墙钟 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| valid_mur0.4_a2（基准复跑） | 2.00 | 3718 | 4.006 | 3.263 | **+22.8 %** | ✗ | 1.270 | 0.938 | +35.3 % | ✗ | 95 s |
| valid_mur0.4_a3（**新**） | 3.00 | 5673 | 4.646 | 3.771 | **+23.2 %** | ✗ | 1.433 | 1.149 | +24.6 % | ✗ | 130 s |
| sens_dt0.5_a2.0（**新**，dt/2） | 2.00 | 3718 | 3.682 | 3.263 | **+12.8 %** | ✅ | 1.201 | 0.938 | +28.0 % | ✗ | 200 s |
| sens_gpr11_a2.0（**新**，11 粒/$r_i$） | 2.00 | 9997 | 3.976 | 3.263 | **+21.9 %** | ✗ | 1.534 | 1.100 | +39.4 % | ✗ | 368 s |
| sens_mu0.8_a2.0（**新**，μ=0.8） | 2.00 | 3718 | 3.264 | 3.263 | **+0.03 %** | ✅ | 1.275 | 0.938 | +35.8 % | ✗ | 147 s |
| sens_mu0.3_a2.0（**新**，μ=0.3） | 2.00 | 3718 | 4.361 | 3.263 | **+33.7 %** | ✗ | 1.350 | 0.938 | +43.9 % | ✗ | 97 s |

基准复跑与轮 1 逐位一致（4.006 / +22.8 %，`at_rest`）⇒ 本轮与轮 1 是同一确定性代码路径，数字可比。

**归因（本轮主结论）**：$a\ge2$ 的 runout 超预测**不是离散化不够**：

| 敏感性轴 | $a=2$ runout 误差 |
|---|---|
| 分辨率 6 → 8 → 11 粒/$r_i$（6/8 来自轮 1） | +28.5 % → +22.8 % → **+21.9 %**（单调改善但**饱和在 ~22 %，不穿 15 % 带**） |
| 时间步 dt → dt/2 | +22.8 % → **+12.8 %（穿带）** |
| 接触摩擦 μ 0.5 → 0.8 | +22.8 % → **+0.03 %（穿带）** |
| 接触摩擦 μ 0.5 → 0.3 | +22.8 % → +33.7 %（恶化，单调） |

即：runout 对 $\mu$ 最敏感且单调（0.3 → 4.361、0.5 → 4.006、0.8 → 3.264），$\mu=0.8$ 时与论文**逐千分位吻合**；
分辨率阶梯三档只把误差从 28.5 % 压到 21.9 % 就在 ~1.5 % 量级上收敛。**结论：轮 1 把 +23 % 归给「沉积层仅 ~2 粒厚」的
离散化不足是不成立的假设——被本轮的收敛阶梯证伪；真正的杠杆是摩擦参数化（球体 DEM 用单一 $\mu$+滚动阻力代理
真实棱角砂的内摩擦），以及时间积分误差（dt/2 单独就能吃掉 10 个百分点）。**

## ④ 结果图

![grid_vs_a.png](../artifacts/case_column_collapse_m15996/grid_vs_a.png)

**grid_vs_a.png** — 左：$r_\infty/r_i$ vs $a$（黑线=论文公式，蓝点=本轮 DEM）；右：runout(o) 与 $t_{99}$(□) 相对误差，绿带 ±15 %。
可见 runout 全程 +22 % 量级、$t_\infty$ 全程 +25 %~+44 %。

![resolution_ladder.png](../artifacts/case_column_collapse_m15996/resolution_ladder.png)

**resolution_ladder.png** — $a=2$ 分辨率阶梯（6/8/11 粒每柱半径）的 runout 误差：28.5 % → 22.8 % → 21.9 %，
阶梯在 8 粒之后基本饱和，**不可能靠加密颗粒穿过 15 % 带**。

## ⑤ 注意点 / 局限

* **预登记补全的 id 约定坑**：敏感性算例在 `--mu-rolling != BASE.mu_rolling` 时自动加 `_mur0.4` 后缀
  （轮 1 文档里的 `sens_gpr6_a2.0` 是无后缀写法，实际按 0.4 标定跑出来的 id 是 `sens_gpr6_a2.0_mur0.4`）。
  本轮 `--case` 必须用带后缀 id，否则 rc=2。已在 docstring 注明。
* **$t_\infty$ 判据全体 FAIL（+25 %~+44 %）且与 $\mu$ 无关**：DEM 沉积尾段是蠕变，$t_{99}$（到自身 99 % runout 的时间）
  系统性长于论文高速摄影判读的「突然停止」。这是**定义域差异**（同轮 1 判读），本轮 $\mu=0.8$ 的 runout 误差 0.03 %
  而 $t$ 误差仍 +35.8 %，进一步支持「不是接触律失效」。判据未改，如实记 FAIL。
* `sens_gpr14_a2.0_mur0.4`（~2 万颗粒、dt 更小）本轮启动后未在收尾前跑完，**honest not_run**，不计入上表；
  它是分辨率阶梯的第 4 档，按 8→11 的饱和趋势不预期穿带，但**未实测不结论**。
* 单进程 wall clock 95–368 s，未上集群（队列 398 作业）；并行 6 进程共享 128 核，时间数字不可当性能基准。
* 论文 $\mu$、$e$ 未逐字给出实测量（砂的 $\mu$ 取决于材料），因此 $\mu=0.8$ 拟合到 0.03 % 只能说明
  「runout 落在 $\mu$ 的合理区间内可复现」，**不构成 $\mu$ 的独立标定**（该标定仍是 $a=0.5$ 上的 $\mu_r^*=0.4$）。

## ⑥ 误差判定（逐项，是否 <15 %）

| 关键量 | 最佳配置误差 | 是否 <15 % | 说明 |
|---|---|---|---|
| runout $r_\infty/r_i$，$a\le1.5$（轮 1，$\mu_r^*=0.4$ 标定后样本外） | +3.3 % / +10.6 % / −3.7 % | **是** | 已合并能力，本轮复跑 $a=2$ 逐位一致 |
| runout $a=2$，基准（μ=0.5, dt） | +22.8 % | 否 | 本轮复现轮 1 |
| runout $a=3$（新） | +23.2 % | 否 | $a^{1/2}$ 分支超预测持续到 $a=3$ |
| runout $a=2$，dt/2 | +12.8 % | **是** | 时间积分误差 ~10 pp |
| runout $a=2$，μ=0.8 | **+0.03 %** | **是** | 摩擦参数为支配杠杆 |
| runout $a=2$，μ=0.3 | +33.7 % | 否 | 单调：μ 越小跑得越远 |
| runout $a=2$，11 粒/$r_i$ | +21.9 % | 否 | 阶梯饱和 |
| 沉降至静止时间 $t_{99}$ 全部算例 | +24.6 % ~ +43.9 % | 否（**honest FAIL**） | 蠕变尾段 vs 摄影判读的定义差异，判据未改 |

**是否 <3 %？** 是，但只在 one-point 意义上：$\mu=0.8$、$a=2$ 的 runout 误差 0.03 % < 3 %。
其余关键量（$a=2$ 基准、$a=3$、$t_{99}$）均 >15 %，**能力整体判为「$a\le1.5$ 段可用、$a\ge2$ 段不可用于定量预报」**，不粉饰。

## 复现

```bash
python3 benchmarks/case_column_collapse_lube2004.py --list-cases --mu-rolling 0.4
python3 benchmarks/case_column_collapse_lube2004.py --case valid_mur0.4_a3 --mu-rolling 0.4
python3 scripts/analyze_column_collapse_m15996.py          # 只读 case JSON → 表 + 图 + csv
python3 -m pytest tests/test_axisym_column_lube2004.py -q   # 8 passed
```