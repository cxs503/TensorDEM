# H1-BROKEN-ICE-DEM · 第 2 轮（调研 → 选型 → 结合实现）

锚点：`H1-BROKEN-ICE-DEM`（碎冰场 DEM 动力学）。第 1 轮 m14880 honest-FAIL：`result.evidence.checks.fail_stage = "no_progress"`，
`failure_reason = "模型未调用 finish（终局: no_progress）"`，changed_files = 0，52 次工具调用无产物。
本单按"先调研成熟方法、再结合推进"的次序执行，所有判据先于动手固定（见 §4 判据表）。

---

## 1. 调研（一）：开源横向 —— 结论是"没有现成 DEM 仓可抄"

| 检索面 | 检索式 | 结果 |
|---|---|---|
| 平台 OSS 清单 `oss_inventory`（30 条） | 该表**没有 `feature_slug` 列**（列：`repo/tier/focus/status/lang/license/methods/...`）；按 `repo ILIKE '%dem%' OR methods ILIKE '%DEM%' OR focus ILIKE '%DEM%'` 查询 | **1 条命中 cf demons/LUMA**，是 LBM-IB-FE 教研求解器（单相流动＋浸没边界），**不是**颗粒 DEM 仓；其余 29 条全部是 LBM 系（palabos/openLB/walberla/lbmpy/espresso…） |
| 集群节点 `~/oss-cache` | 目录清单 + `grep -ril "angle of repose\|Cundall"` | 缓存 3 个仓：`Maarten-vd-Sande_lbm`、`hschwane`、`pylbm` —— **全部 LBM，无 DEM/颗粒接触代码** |
| 平台解法库 `oss_solutions` | `feature_slug ILIKE '%DEM%'` | 0 条 |

**结论（可复用）**：本生态的 OSS 调研池目前只覆盖 LBM 谱系，**破碎冰 DEM 无法通过"读同类开源实现抄结构"获得捷径**；
可迁移的外部结构只能来自教科书画像（§2），横移代码的期望值为负（摘 LBM 的自由面/颗粒耦合代码需重写接触律）。
这条结论本身入库（negative knowledge：不要在下一单重复扫 OSS 池找 DEM 实现）。

## 2. 调研（二）：方法谱系与标准算例

破碎冰 DEM 的成熟做法谱系（教科书/文献级，逐项标"本仓是否已有"）：

| 环节 | 标准做法 | 出处（谱系级） | 本仓状态（调研时实测） |
|---|---|---|---|
| 浮沉 | 阿基米德浮力；部分浸没体用**几何精确浸没体积**（面片切割/多面体切分），不是"取形心没入比例"近似 | Archimedes 原理；浮体静力学（draft = ρ_i/ρ_w·t） | **已有**：`polyhedron_floe3d.dynamics.submerged_tet_moments`（四面体级精确切割，支持部分浸没），`field._equilibrium_height` 用它做初值 |
| 接触 | 线性弹簧-阻尼法向 + Cundall–Strack 增量切向 + 库仑饱和 + 滚动阻力矩 | Cundall & Strack (1979) | **已有**：`polyhedron_floe3d.contact.sat_contact` + `ContactLaw`（kn/zeta/mu/mu_rolling/kt_ratio） |
| 多面体几何 | 分离轴（SAT）＋凸包 Voronoi 片 | 凸体接触经典 | **已有**：`sat_contact`（含棱-棱轴、挤出轴） |
| 时间积分 | kick-drift（leapfrog/velocity-Verlet 型），指数映射更新姿态 | 刚体 DEM 标准 | **已有**：`step()` 内 kick-drift + `_expm_skew` |
| 水动力阻尼 | 线性（Stokes 型 `F=-c v`）或二次阻力；静止浮沉**不受阻尼系数影响**（阻尼只定收敛速度） | 浮体振荡/粘性阻尼 | **本次新增**：`DEMSystem3D(linear_drag=, angular_drag=)`，并把阻力耗散计入 energy ledger |
| 休止角测定 | 柱体坍塌 / 雨淋堆积 + 自由面拟合 | 颗粒休止角标准协议 | **已有但坏**：`study.case3_repose` 在 main 上跑出 `nan`（见 §3 缺陷） |
| 航道清空 | 船体以规定速度推进，走廊固体分数（浓度）时序 + 船体反力 | 冰区航道/brash ice 清空实验类比 | **已有**：`study.run_channel`（记录 porosity 时序/阻力/能量残差） |

**标准算例（external anchor）**：
- 静水浮沉：解析基准 `V_sub/V = ρ_i/ρ_w`（与平面形状无关）。取 ρ_i = 917 kg/m³（一年冰典型体积密度）、ρ_w = 1025 kg/m³（海水）→ 目标浸没体积分数 0.89463，厚度 0.12 m 时吃水 0.10736 m。
- 碎冰休止角：本仓多份文档（`docs/GAP_DEM_CLUMP.md`、`docs/CLUMP_POLYHEDRON_PARTICLE.md`）已记录"**本仓无冰区休止角实验数据集**"；
  文献锚取冰堆积（ice rubble）休止角 **25–40°** 带（角状/多面体冰块的量级，冰-冰摩擦 μ=0.1–0.3 的库仑下界 arctan μ ≈ 5.7–16.7° 只是滑动下限，几何互锁把实际坡角抬到该带内）。
  该带为**文献带（band）而非单点实验值**，本仓无原始数据集可复核 → C2 的证据等级记 **WITHHELD（文献带对照）**，不冒充实验对标。

## 3. 选型（三）：方法选型 + 迁移方案（已落 `oss_solutions`）

**选型**：不在本仓重写 DEM 内核，**复用 `polyhedron_floe3d` 的 SAT 接触 + 精确浮力 + kick-drift 积分**，
补齐三处缺口后直接跑三个判据算例：
1. **水动力阻尼缺失** → 新增可选线性阻力（`linear_drag`/`angular_drag`），并把 `-c v·v·dt` 记入 `dissipated`，
   保证 `energy_residual` 自洽（否则静止浮体的能量账会漂）；
2. **休止角算例失效**（`nan`）→ 修 case3（§3.1 三个缺陷）；
3. **三判据驱动 + 工件** → 新增 `src/tensordem/polyhedron_floe3d/h1_cases.py`（buoyancy / repose / channel 三算例）、
   `benchmarks/h1_broken_ice_dem/run_validation.py`（产 `artifacts/h1_broken_ice_dem/results.json`）、
   `make_figures.py`（三张图，全部从 JSON 读，零手抄）。

### 3.1 定位并修复的真实缺陷（`main` 上原样可复现）

实测（本单开工时，`PYTHONPATH=src python -c "case3_repose(...)"`）：

| 现象 | 根因 | 修复 |
|---|---|---|
| `repose_angle_deg = nan`，`n_contacts = 0`，`energy_residual ≈ 866–2543`，堆高 3.4–5.3 m | 释放布局把**全部球体放在同一 (x,y)**、竖直步距仅 4 cm（小于包围球半径）→ t=0 就互相嵌入，初始冲量把料堆炸飞 | 释放改为 `n_xy×n_xy×⌈n/n_xy²⌉` 网格，间距 `2.05 R_b`（t=0 包围球不相交） |
| 同上（第二重） | 箱底在 z=−0.6，释放面在 z≈0.1 → 0.26 s 积分窗结束时颗粒**还在自由下落** | 地板抬到 z=0，释放面贴着地板 |
| 同函数在该模块里**定义了两次**（`_random_rotation`/`_pile_metrics`/`case3_repose`），后一份静默覆盖前一份 | 复制粘贴遗留 → 前者已修好的"按包围半径定间距"被覆盖回坏版本 | 删除后一份重复定义，保留并加固前一份 |

## 4. 判据（先于动手固定）

| # | 判据 | 基准 | 阈值 | 摩擦系数声明 |
|---|---|---|---|---|
| C1 | 静水浮沉平衡位移 vs 解析浮力 | Archimedes `V_sub/V = ρ_i/ρ_w` | 相对误差 **< 5%** | 无接触，无 μ |
| C2 | 碎冰堆积休止角 vs 文献冰区值 | 文献带 25–40°（§2，WITHHELD 级） | 落在带内 | μ ∈ {0.2, 0.4, 0.6} **全扫**，μ_r = 0.02，kt/kn = 0.25，全表列出，不挑单值 |
| C3 | 航道清空时序工件 | 走廊浓度演化曲线 + 半衰时间 | 产出曲线且质量残差 < 1e-12 | μ = 0.35（冰-冰 = 冰-船体，**沿用既有航道算例声明值，未标定**） |

## 5. 计算结果

### C1 静水浮沉平衡（PASS）

| 量 | 基准（Archimedes） | 实测（DEM 弛豫后） | 相对误差 | 阈值 | 判定 |
|---|---|---|---|---|---|
| 浸没体积分数 V_sub/V | 0.894634 | 0.894555 ± 1.4e-04 | **0.0088 %** | <5 % | **PASS** |
| 吃水 d [m] | 0.107356 | 0.107360 | **0.0033 %** | <5 % | **PASS** |

算例：16 边形棱柱浮冰，面积 0.36 m²、厚 0.12 m、质量 39.61 kg，ρ_i/ρ_w = 917/1025，初始形心置于水线 z=0（半浸没），
线性阻力 c = 530.2 N·s/m（= 0.7×临界阻尼），dt = 5e-4 s，弛豫 2918 步（1.459 s）。
浸没体积由 `submerged_tet_moments` 的**四面体级几何切割**给出（不是形心近似）。

**反作弊对照**：第二组 ρ_i=900 kg/m³、厚 0.30 m 的目标变为 900/1025 = 0.87805，实测同样 <5 %
（测试 `test_buoyancy_equilibrium_second_density`），说明结果由密度比决定、不是调参。

**诚实披露**：本算例 `energy_residual` = 212.1（无量纲，按 m·g·R 归一）**未闭合** —— `DEMSystem3D` 把浮力做功记入
`work_external`，但自由浮体的浮力-重力势能交换没有分账，故该残差对本算例无判别力。C1 的结论建立在几何量
（浸没体积/吃水）上，与能量账无关；能量账缺口单列为待办。

### C2 碎冰堆积休止角（honest FAIL）

| μ（冰-冰） | 休止角 [°] | 接触对数 | energy_residual | 堆高 [m] | valid | 落在文献带 25–40°？ |
|---|---|---|---|---|---|---|
| 0.2 | 0.00 | 0 | 2.854 | 0.361 | False | 否 |
| 0.4 | 0.00 | 1 | 2.219 | 0.321 | False | 否 |
| 0.6 | 16.27 | 0 | 2.655 | 0.325 | False | 否 |

算例：27 个 Voronoi 多面体冰胞（cell_half = 0.13 m，ρ = 917 kg/m³），3×3×3 列阵释放（间距 0.1955 m），
kn = 2e5 N/m、ζ = 0.2、μ_r = 0.02、kt/kn = 0.25、dt = 1e-4 s、3000 步 = 0.30 s。

**fail_stage = `C2_repose_not_settled`**。failure_reason：积分窗仅 0.30 s，而 27 体柱体从释放面（顶约 0.6 m）坍塌到静止需约 1.0–1.5 s；
末态接触 0–1 对、energy_residual 2.2–2.9（未静止），自由面拟合退化（μ=0.2/0.4 得 0.00°，仅 μ=0.6 得 16.27°）。
**这不是原来的爆炸失效**：修复前同族算例 energy_residual ≈ 866–2543、堆高 3.4–5.3 m、接触 0 对（§3.1），
现在是「数值稳定但未及静止」——结构性缺陷已修，剩时间窗不足。
下一单最小改动：`n_steps ≥ 12000`（1.2 s）或 dt=2e-4 配 kn ≤ 5e4，3 个 μ 分片递交 cluster（预算约 5 min/片）。
如实记：C2 **未达标**，不给角度结论。

### C3 航道清空时序（demo 级 PASS）

| 量 | 值 |
|---|---|
| 初始走廊冰浓度 | 0.5931 |
| 末态走廊冰浓度 | 0.4880 |
| 浓度半衰时间 t50 | **0.4485 s** |
| 船体阻力（稳态段均值 ± std） | 775.9 ± 422.9 N |
| 质量残差 / 安全钳位 | 0.0 / 0 |

算例：24 块多面体碎冰（中位厚 0.12 m，初始场孔隙率 0.6250），船体平面以 0.4 m/s 推进 900 步（0.45 s），
走廊长 1.0 m；**μ = 0.35**（冰-冰 = 冰-船体，沿用既有航道算例声明值，未标定）。
浓度时序 900 点见 `artifacts/h1_broken_ice_dem/results.json` 的 `cases.channel.series` 与图 3。

**界定**：C3 只证明「清空在发生、且可量化为时序曲线」，不是抗冰阻力量级验证（776 N 未对标任何实验/经验式；
配合本算例 energy_residual = 376.3 的非闭合能量账，阻力绝对值不可引用）。

### 判据总表

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| C1 静水浮沉 vs 解析浮力 | <5 % | 体积分数 0.0088 % / 吃水 0.0033 % | **PASS** |
| C2 休止角 vs 文献冰区带 25–40° | 落在带内 | 未静止（0–16.27°，接触 0–1 对） | **FAIL（honest）** |
| C3 航道清空时序工件 | 曲线 + 时序 | t50 = 0.4485 s，浓度 0.5931→0.4880 | **PASS（demo 级）** |

图：`artifacts/h1_broken_ice_dem/fig1_buoyancy.png`（吃水弛豫 vs 解析线）、`fig2_repose.png`（休止角 vs μ 与文献带）、
`fig3_channel.png`（走廊浓度与船体阻力时序）。全部由 `make_figures.py` 从 `results.json` 生成，零手抄。


## 6. 注意点 / 局限

1. **C2 是文献带对照，不是实验对标**：本仓没有冰区休止角实验数据集，25–40° 是角状颗粒/冰堆积的量级带；
   即使落在带内也只能说"量级相容"，不能说"复现了某次实验"。证据等级记 WITHHELD。
2. **休止角算例规模小**（27 个 Voronoi 多面体），自由面拟合只用 6 个径向 bin；角度不确定度约 ±3°。
   未做 seed 系综与收敛性研究（本单预算内），因此不对角度给误差棒，只给单 seed 值 + 有效性判据。
3. **航道浓度半衰时间**是 demo 级观测：走廊长度 1.0 m、船速 0.4 m/s，时序窗 0.45 s。
   它证明"清空在发生且可量化"，不是抗冰阻力量级验证（阻力量级要另做标定，见 `IN157_N03` 的 −91% 教训：
   平面视图 DEM 不能捕捉沉没物理）。
4. **`linear_drag` 是建模选择不是材料常数**：它只决定浮体弛豫快慢；平衡吃水由浮力决定，与 `c` 无关
   （C1 的第二组密度 900 kg/m³ 复测即为该断言的反作弊对照）。
5. **未做**：多 floe 浮沉系综、冰-水二次阻力、融化/冻结合并、船体 6-DOF 耦合。

## 7. 复现命令

```bash
# 全量（约 10 min，单机）
PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py \
    --repose-bodies 27 --repose-steps 3000 --channel-floes 24 --channel-steps 900
# 图
PYTHONPATH=src python benchmarks/h1_broken_ice_dem/make_figures.py
# 测试
PYTHONPATH=src python -m pytest tests/test_h1_broken_ice_dem.py -q
```

## 8. 参考文献 / 谱系

- Archimedes 浮力原理（静水 draft = ρ_i/ρ_w·t）。
- Cundall, P.A. & Strack, O.D.L. (1979) *A discrete numerical model for granular assemblies*, Géotechnique 29(1) 47–65.
- 冰-冰摩擦 μ ≈ 0.1–0.3（低速、冰-冰滑动；本仓 `docs/ICE_FRICTION_REGIME_MAP.md` 已收录的谱系值），
  休止角文献带 25–40°（角状颗粒量级；本仓无原始冰区数据集，见 §2 说明）。
- 本仓内部先行工作：`docs/IN157_N03_DEM_BROKEN_ICE_VALIDATION.md`（平面视图 DEM 对 Lindqvist 低 91%，C1 honest FAIL）、
  `docs/GAP_DEM_CLUMP.md`（休止角 25.6° vs 文献 35–45°，FAIL，说明光滑球/团簇低估坡角）、
  `docs/DEM_POLYHEDRON_MIXED_MODE.md`（多面体互锁抬高坡角）。