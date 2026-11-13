# XVAL-DEM-SANDPILE-REPOSE-CALIBRATION-ANCHOR · TensorDEM vs Zhou et al. (1999)

跨仓/跨实现交叉对标（XVAL）：本仓冻结的沙堆休止角协议
`tensordem.protocols.sandpile`（`sandpile-v1`）对标外部标定锚

> Zhou, Y.C., Wright, B.D., Yang, R.Y., Xu, B.H., Yu, A.B. (1999).
> *Rolling friction in the dynamic simulation of sandpile formation.*
> Physica A **269**, 536–553.

## 0 参考值的来源（零手抄）

参考数字**不写在本文件里**，运行时从仓内文献记录解析：

| 来源 | 取什么 | 方式 |
|---|---|---|
| `docs/GAP_DEM_CLUMP_ROUND2.md`（GAP-DEM-CLUMP round-2 写入的 `LIT` 锚行） | Zhou-1999 球基线休止角带 + 非球形带 | 正则 `(\d+)\s*[–\-~]\s*(\d+)\s*°`，文件 sha256 入 `verdict.json` |
| `src/tensordem/protocols/sandpile.py` 模块 docstring | 书目行 | 正则 `Zhou,\s*Y\.C\..*?Physica A\s*269,\s*536-553\.` |

解析结果（`--aggregate` 输出，报告 §① 原文引用）：球 **[22, 25]°**（中心 23.5°），
角砾/椭球 [35, 45]°。

**证据边界（如实记）**：本生态的知识引擎（`:8021`）本单期间离线、
`oss_solutions` / `knowledge_base` 表对本 DB 用户 `permission denied`，
因此 Zhou-1999 的**算法细节（源码级）无法取得**，XVAL 只能做**带级对标** +
**机制级（滚动摩擦符号）对标**，不能做实现级 diff。这一条是 WITHHELD，不是"没差"。

## 1 本仓两条路线（同一接触物理，不同边界条件）

| 路线 | 驱动 | 边界条件 | 用途 |
|---|---|---|---|
| A | `protocols.sandpile`（`sandpile-v1`） | 喷嘴撒料 → 自由落体 → 平底摩擦板静置 | 本单主对标对象 |
| B | `repose_calibration` | 六方柱体装填 → 提筒塌落（lifted-cylinder，**与 Zhou 的 lifted-box 同族**） | 交叉一致性腿 |

`cross_route.py` 用**同一 `(μ_s=0.50, μ_r)`** 跑 B 路线，`μ_r ∈ {0.20, 0.05}`。

## 2 差异清单：本仓（路线 A）vs Zhou-1999（据仓内锚行可得部分）

| 项 | 本仓 `sandpile-v1` | Zhou-1999（仓内记录可得） | 影响/取舍 |
|---|---|---|---|
| 成堆方式（边界条件） | 喷嘴**撒料**，释放平面固定在 `heap_height_estimate + drop`，落高随堆高衰减；出口速度由流量守恒 `v0=q V_g/(0.55π r_n²)` 定 | **lifted-box**（箱体抬起成堆） | 撒料把每球动能（落高≈堆高，本单 H/R≈16）带进堆体，系统性压低坡角；lifted-box 无冲击能 |
| 测角算子 | 环带宽度 2R、扇区取**最大**高程、25–85 % 趾半径最小二乘直线拟合 | 论文给了趋势与带，算子细节不可得（WITHHELD） | 同族算子（B 路线用 20–85 %），不可做逐项 diff |
| 摩擦先验 | **冻结** `μ_s=0.5, μ_r=0.05`，只能经营白名单负控制偏离（`mu_r_zero`/`mu_s_zero`） | 论文以 `μ_r` 为扫描变量标定 | 本仓**不做 μ_r 标定**（防标定守则），故不能复现论文的 μ_r 曲线，只能测 `μ_r=0` 端点 |
| 滚动摩擦模型 | EPSD，上限 `μ_r r_eff |F_n|`（`contact_tangential.rolling_resistance_si3`） | 滚动摩擦是主变量；模型族细节不可得 | 机制方向可比（有/无滚动），幅值不可比 |
| 基底/侧壁 | 平底摩擦板，无侧壁 | 不可得 | — |
| 刚度/时间步 | `k_n` 为数值刚度（本单锚点 case 1500 N/m），`dt=0.1 t_c` | 不可得 | 本单已验证 dt、N、静置时间不敏感（D5） |
| 目标量 | 休止角 θ + 配位数 Z + 探针孔隙率 φ | 休止角带 | Z/φ 作"这是不是一堆"的判据，论文不带 |

## 3 移植取舍

1. **采纳（adapt）**：把 [22,25]° 作为**接受带**（`D2`）+ 15 % 相对误差（`D1`）；
   把"去掉滚动摩擦 → 堆体撑不住坡"作为**机制判据**（`D3`），这是本仓负控制
   `mu_r_zero` 与论文主结论的公共交点。
2. **不采纳（reference-not-adopted）**：论文的 `μ_r` 扫描**不移植**——本仓
   `assert_priors` 在第一步之前就拒绝非先验摩擦系数，标定必须走既有的
   `repose_calibration` 标定面（`μ→θ` 表），与协议分离。理由：协议一旦可标定，
   "passed to the stepper == parametrised" 就不可机器核验。
3. **不移植（method-mismatch）**：lifted-box 几何不合并进 `sandpile-v1`；
   它由路线 B 承担。两条路线的分歧本身就是本单的产出之一（见报告 §⑤）。
4. **待办（两轮法接棒）**：若要在路线 A 上闭合带级偏差，正确做法是改**边界条件**
   （释放平面跟随当前堆顶、限幅），而不是调 `μ_r`；这需要重过协议冻结与
   先验守则，属下一轮工作量。

## 4 产物与复现

```bash
# 单 case / 矩阵（每 case 单线程，可 shard 到集群）
python benchmarks/xval_sandpile_zhou1999/xval_zhou1999.py --list
python benchmarks/xval_sandpile_zhou1999/xval_zhou1999.py --run gentle_anchor
python benchmarks/xval_sandpile_zhou1999/xval_zhou1999.py --run-all --jobs 8
# 交叉路线腿
python benchmarks/xval_sandpile_zhou1999/cross_route.py
# 判定 + 图 + 报告（docs/XVAL_SANDPILE_ZHOU1999.md）
python benchmarks/xval_sandpile_zhou1999/xval_zhou1999.py --aggregate
```

工件：`artifacts/XVAL-SANDPILE-ZHOU1999/{cases/*.json, cross_route.json,
verdict.json, figures/*.png}`。判据 D1–D6 与阈值写死在脚本常量里（跑前登记，跑后未改）。