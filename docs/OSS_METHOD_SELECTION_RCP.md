# CASE-DEM-RCP-PACKING-FRACTION — 方法调研与选型（第 2 轮，mission 18143）

> 目的：本轮先调研成熟做法，再结合推进。产出 = ①开源横向 ②方法谱系 ③选型+迁移方案。
> 一切数字与出处可核；无法核到的写"未取到"，不写"大概"。

## ① 开源横向（集群 oss-cache / oss_inventory）

| 检索 | 结果 |
|---|---|
| `oss_inventory` 内 `summary/focus/methods ILIKE '%packing%'` | **0 条** |
| `oss_inventory` 内 repo 名含 dem / granular / liggghts / yade / lammps / mercury | **0 条**（唯一的 MD 平台是 `espressomd/espresso`，其库存条目写的是 MD-LBM 耦合与 waLBerla/lbmpy 侧，未登记颗粒堆积协议） |
| `~/oss-cache` 全文 grep `random close packing|random loose packing|deposition packing` | 见 `benchmarks/case_rcp_deposition/runs/logs/oss_grep_<jid>.log`（本轮递交的集群 grep 作业；若为空即"缓存内没有同类实现"） |

**结论（可复用）**：本生态的 OSS 库存是 LBM 侧为主导，**没有可直接借鉴的 RCP 沉积堆积实现**。
因此本锚点的迁移不是"移植代码"，而是"移植协议 + 自建内核"：协议（初始稀释分布 → 重力沉积 →
动能判据停堆 → 体相板厚平均）来自文献与教科书，内核（`src/tensordem/deposition_packing.py`，
速度 Verlet + Cundall–Strack 切向弹簧 + cKDTree 邻接表）在本仓自持。

> 这条负面结论本身要入账：`negative-no-oss-reference`（同生态里已有同类登记先例），
> 避免下一轮再花时间去找不存在的 OSS 参考。

## ② 方法谱系（文献/教科书标准做法 + 标准算例）

### 2.1 实验基准（为什么是 0.6366）

| 出处 | 量 | 值 |
|---|---|---|
| Scott GD (1960) *Nature* **188**:908（DOI 10.1038/188908a0） | 单分散球"倾倒(poured)"与"振实(dense/shaken)"两种堆积 | 倾倒 ~0.601；致密 ~0.64 |
| Bernal JD & Mason J (1960) *Nature* **188**:910（DOI 10.1038/188910a0） | 随机密堆体积分数（球体 5 万级统计） | **φ_RCP = 0.6366**，配位数 ~12 |
| Finney JL (1970) *Proc R Soc A* **319**:479 | 随机密堆重测 | 0.6366 ± 0.0004 |

锚点判据（特征账本）用 0.6366 + 容差带 [0.62, 0.66]，与本表一致。要点：**0.6366 是"致密随机"分支**，
Scott 的倾倒分支是 0.601——两者差 5.6%，远大于 3% 精度线，所以"哪一种堆积协议"必须先声明，
否则同一组方程可以"合法地"给出两个都像真的答案。本锚点要求"落下堆积体"，故取沉积（pour）协议，
但目标值是致密随机值——这要求沉积时球必须能在自重下重排到 RCP（无摩擦/低摩擦 + 低恢复系数即可）。

### 2.2 数值协议谱系（同一物理，不同算法）

| 族 | 代表 | 能否给摩擦档 | 能否声称"落下堆积" | 备注 |
|---|---|---|---|---|
| 几何压缩/膨胀（Lubachevsky–Stillinger） | LS 1990 *J Stat Phys* **60**:561 | 否（无接触力学） | 否（体积是输入） | 本锚点前作 m16179 走此路：φ=0.6400 实际是压缩目标值的读数，不是涌现量 |
| 硬球蒙特卡洛（事件驱动/无重叠采样） | Wood 1957; Speedy 1983 | 弱（无静摩擦历史） | 否 | 只给"可达构型"统计 |
| **软球重力沉积（DEM + 阻尼淬火）** | Cundall & Strack 1979 *Géotechnique* **29**:47；EDMD 淬火变体 | **是**（切向弹簧携带静摩擦历史） | **是**（重力是驱动力，停堆由动能判据裁决） | **本轮选型**；φ 依赖接触刚度 k_n，必须外推硬球极限 |
| 连续介质/统计力学 | Song, Wang & Makse 2008 *Nature* **453**:629（RLP/RCP 相图） | 是（作为参数） | 否 | 解释 φ(μ) 趋势的理论锚，不产生算例 |

### 2.3 关键物理先验（用于判读，不是拟合目标）

* **摩擦控制密度**：Silbert et al. (2002) *PRE* **65**:051307 显示无摩擦球堆积 φ≈0.635–0.64，
  加摩擦后落到 RLP≈0.60（μ≥0.3）。因此"低摩擦档略低于无摩擦档"是**预期物理**，
  不能一律当成数值缺陷；反之若两档相同或反号，才是缺陷信号。
* **软球可压缩性**：线性弹簧床在自重下的平均重叠 ~ (mg/k_n)^{1/2} 量级，φ 随 1/k_n 单调上升；
  引用 φ 必须同时给 k_n 或硬球极限（教科书标准做法：Richardson/幂律外推，见 O'Sullivan,
  *Particulate Discrete Element Modelling*, 2011, §4）。
* **测量窗口**：体相板厚平均（离底 2.5d、离自由面 2.5d）是剔除壁面层化的标准做法；
  全床体积比（surface-to-floor）会把壁面与自由面粗糙度算进去，作为交叉核对而非主指标。
* **停堆判据**：KE/(N m g d) 持续低于阈值（本仓 5e-4 × 4 重力时间），等价于"准静态"；
  不设判据而拍步数会把未松弛构型当稳态。

## ③ 选型与迁移方案（落 oss_solutions）

**选型（1 个主选 + 1 个交叉核对）**

1. 主选：软球重力沉积（线性弹簧-黏壶 + Cundall–Strack 切向，dt = t_c/20，e = 0.1，
   稀释 RSA 初始分布，周期侧壁 + 摩擦地板，动能判据停堆），体积分数取体相板厚平均，
   并做 **1/k_n 四级 Richardson 外推（含留一级自检）** 报硬球极限；
2. 交叉核对：重叠修正体积（逐球按平均重叠缩径）必须与主读数一致到 1%；
   半步长时间步探针必须把时间步误差压到 0.002 以下，否则刚度系统性无法与积分误差分离。

**迁移方案（为什么这套能搬进本仓并被机器裁）**

| 环节 | 迁移内容 | 落点 |
|---|---|---|
| 协议 | 稀释初始分布、重力驱动、动能停堆、板厚窗口 | `src/tensordem/deposition_packing.py`；入口 `benchmarks/case_rcp_deposition/run_case.py` |
| 刚度处理 | 生产刚度规则 = "两档摩擦都能凑满 3 种子的最硬 k_n"（读数驱动，不写死） | `benchmarks/case_rcp_deposition/aggregate.py` 的 `kn_prod` 选择 |
| 外推 | `phi(k_n) = phi_inf + A k_n^-p`（观测阶 p），留一级预测自检 | `deposition_packing.stiffness_limit / extrapolation_selfconsistency` |
| 判据 | 带 [0.62,0.66] + 3% + 重叠 <1% + 时间步 <0.002 + 摩擦单调性 | `aggregate.py` 生成的 gates.json |
| 可复现 | 每个 run JSON 带 impl_sha256、spec、wall clock、history；失败/未收敛如实留 | `runs/` + `docs/` |

**本轮不迁移（明确的范围外）**：多分散度、Hertz–Mindlin 非线性接触、非球形（clump/多面体）颗粒、
振实（tapped）分支协议。理由：锚点字面判据只需要单分散球 + 落下堆积 + 两档摩擦；扩进来会把
"φ(μ, k_n, N)" 的可解释性稀释成协议对比。

## 出处汇总（external_anchor）

* Scott GD (1960) *Nature* 188:908, DOI 10.1038/188908a0 — 倾倒 0.601 / 致密 0.64
* Bernal JD, Mason J (1960) *Nature* 188:910, DOI 10.1038/188910a0 — φ_RCP = 0.6366
* Finney JL (1970) *Proc R Soc A* 319:479 — 0.6366 ± 0.0004
* Cundall PA, Strack ODL (1979) *Géotechnique* 29:47 — 线性弹簧-黏壶 + 切向历史
* Lubachevsky BD, Stillinger FH (1990) *J Stat Phys* 60:561 — 几何压缩（对照族）
* Silbert LE et al. (2002) *PRE* 65:051307 — 摩擦控制堆积密度（无摩擦 RCP ≈0.64，μ≥0.3 → RLP ≈0.60）
* Song C, Wang P, Makse HA (2008) *Nature* 453:629 — RLP/RCP 阻塞相图
* O'Sullivan C (2011) *Particulate Discrete Element Modelling* — 体相窗口/刚度外推的教科书做法