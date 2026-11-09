# TensorDEM

**Tensor ecosystem vision:** [AI-era ecosystem vision](https://github.com/cxs503/TensorPlatform/blob/main/docs/architecture/TENSOR_ECOSYSTEM_AI_ERA_VISION.md).

**Project charter:** [docs/PROJECT_CHARTER.md](docs/PROJECT_CHARTER.md) — mission, ownership, non-goals, integration, quality gates, and agent rules.

基于 **PyTorch** 的二维黏结颗粒离散元（DEM）破冰原型，支持 CPU / CUDA。
圆形刚性破冰头向下压入两端固定的冰层，黏结可因轴向拉伸或客观剪切应变超阈值而不可逆断裂；
断裂后的碎冰仍通过接触力相互作用。所有力和积分均使用 PyTorch 张量。

**System-wide overview:** [Tensor Ecosystem System Overview](https://github.com/cxs503/TensorPhysics/blob/main/docs/SYSTEM_OVERVIEW.md) — architecture, repository boundaries, shared contracts, end-to-end workflow and icebreaker application roadmap.


## 冰脊触底与海底管道（ridge keel grounding）

`tensordem.ridge_dem` / `ridge_geotech` / `ridge_grounding` 组成二维平面应变
DEM–Mohr-Coulomb 地基–Winkler 弹性地基梁耦合流程：冰脊龙骨参数化（龙骨深度、翼缘宽度、
孔隙率、冻结黏结率）、管道刚体圆柱接触、海床 Mohr-Coulomb/Drucker-Prager 等效承载力与
峰后残余弹簧、管道竖向自由度与土体动力平衡，输出载荷–位移曲线、极限承载力包络、
管道弯矩/应力分布与 DNV-OS-F101 组合载荷利用率。方法、验证判据与局限见
`docs/ice_ridge_pipeline_grounding.md`；工件见 `artifacts/ridge_grounding/`；
运行 `python scripts/run_ice_ridge_grounding.py`。

## 冰–船体水线带冲蚀磨损（ice-hull abrasion wear）

`tensordem.ice_abrasion` 把 DEM 接触统计（Hertz 接触 + 冰压碎强度截断 +
Cundall–Strack 切向饱和，输出局部压力 `p_c` 与滑移距离 `L_slip` 的联合分布）
接上 Archard 磨损律 `V = K W L / H`（等价于比磨损率 `k = K/H`）与 Finnie
高冲击角修正，投影到船体水线带网格得到磨损深度场，并与 `threefield.resistance`
的摩擦功做能量守恒闭环（`V H / W_f = K / mu`），输出涂层/钢基体寿命 P10/P50/P90。
几何先过 Cb/L/B 校验再算磨损。绝对量级由单一参数 `k` 对文献锚点标定，
明确标注为 CALIBRATED；方法、判据与局限见 `docs/ice_abrasion_archard_wear.md`，
工件见 `artifacts/ice_abrasion/`，API 合约 `ice_abrasion_calculation`
对应 `POST /api/v1/calculations/structure/ice_abrasion`。

## 冰压带相关长度 λ_x 的标定估计量（`tensordem.ice_pressure_band`）

冰–结构压力带沿面相关长度 λ_x 通常以文献锚定值 λ_x/h = 1.5–3 直接引用。本模块把它改成
**可审计的估计量**：三种估计器（ACF 积分尺度 / 指数拟合 / 洛伦兹谱拟合，另加仓内既有的
e 折遗留法做对照）的有限窗偏差在已知解析 ACF 的合成场上被**测量**并拟合修正
（`ρ(K)=1+a/K+b/K²` 等预注册形式族按留一交叉验证选取，`K=L/λ`），修正为自洽方程
`λ_corr = λ̂/ρ(L/λ_corr)`（向量化求根）；记录长度充分性门槛 `K*` 用留一残差偏差的
「连续充分尾」定义（|bias|<5 % 且所有更大的 K 都满足）；方差标度 `Var∝λ/L` 由两条独立
扫描检验；白噪声与严格周期记录被 regime 诊断**拒绝**而不是给出数字。长域 DEM 压力带
（`ice_contact.sim.IceWallCrush`，B/h = 40–60）给出原生 λ_x 与移动块 bootstrap 95 % 区间，
并以「粗化阶梯」检验所报长度是否只是沉积核尺度。输出契约为
`schemas/lambda_x_estimate.v1.json`，扁平视图供 prop-1104 消费。

运行 `python scripts/pressure_band_lambda_calibration.py --skip-dem`（合成标定）与
`--reuse-dem`（合成长域 DEM 结果）；方法、判据数字与局限见
`docs/ice_pressure_band_lambda_x.md`，工件见 `artifacts/pressure_band_lambda/`。

### 成熟化（p48w3-n02）：级交叉估计量 · 方差标度律 · 记录长度门槛

在上一节点（P204-N08）的估计量链路上，本节点新增级交叉估计量 `crossing`（ACF 穿越
`R = 0.6` 的滞后反演，条件数 ~3），并修复两处可复现缺陷：

* `run_variance_family` 名义上「固定记录长度」，实际固定的是 `L/lam`（`n = K/dx`），斜率按
  **构造恒为 0**，无法检验标度律；现固定绝对 `L` 并返回 `R^2`；
* bootstrap 的百分位区间会出现**不含点估计**的格点；现给出以点估计为中心的 basic 区间并
  逐格报告 `contains_point`（同时修掉 `models=None` 时全部重采样为 NaN 的静默失效）。

结果（全部由 `scripts/pressure_band_maturation.py` 程序生成，判据落
`artifacts/pressure_band_lambda_maturation/criteria.json`）：

* 方差标度 `Var(λ̂/λ) ∝ λ/L` 在二维 `(L, λ)` 网格上成立：`crossing` 斜率 1.01/1.04、
  `R² = 0.97/0.94`（ar1/spectral），`lagslope` 0.94/0.93、`R² = 0.98`；首零点截断的
  `integral` 不成立（`R² = 0.01/0.26`）——上一节点的 honest FAIL 由此定位到**估计量选择**，
  修法是换估计量而不是把曲线拟合得更用力；
* 记录长度门槛：严格 5 % 规则下三法全达标 `K* = 12.6`，工程建议 `K ≥ 25`（单条记录相对
  sd 在 K=25 仍为 33 %–66 %，10 % 不确定度需 `K ~ O(10²)`）；
* 长域 DEM 压力带（`B/h = 40–60`，`d50/h = 0.125/0.25/0.40`）原生 `λ_x/h = 0.058`
  （0.041–0.082），仍落在文献带 1.5–3 之外：`λ_x/dy = 0.66–1.31` 个网格、对 `d50` 的对数
  斜率仅 0.21（`R² = 0.20`）、自诊断全部判 `white_noise`、打乱对照 = 0.50 网格、分辨率阶梯
  恒为 `O(1)` —— **所报长度是压力沉积核尺度，不是连续介质斑块尺度**（honest FAIL，机制量化）。

方法、数字表、图与误差判定见 `docs/ice_pressure_band_lambda_x_maturation.md`；
脚本 `scripts/pressure_band_maturation.py`＋`scripts/pressure_band_maturation_report.py`；
测试 `tests/test_ice_pressure_band_maturation.py`。

## 面向 Agent 科研工厂的职责

在以 Agent 为科研主体的黑灯科研工厂中，TensorDEM 是一个**可编程、可观测、可验证的离散元数值实验设备**，负责颗粒运动、接触、键合与断裂等数值内核及专项验证。TensorPhysics 负责实验编排与跨求解器协调；TensorPlatform 负责数据、产物、研究记录和模型资产治理。

Agent 可以提出参数扫描、断裂假设和下一轮实验建议，但结果只有在可复现实验、独立基准、敏感性分析及适用范围检查后，才能作为相应等级的科研证据。运行结束、动画合理或接口能够交换载荷，都不等于工程物理预测已经验证。

**系统架构：** [科研工厂架构](https://github.com/cxs503/TensorPhysics/blob/main/docs/SCIENCE_FACTORY_ARCHITECTURE.md) · [系统总览](https://github.com/cxs503/TensorPhysics/blob/main/docs/SYSTEM_OVERVIEW.md)

## 在 Tensor 系统中的职责

TensorDEM 是 Tensor 工程仿真体系中的**离散元与颗粒断裂求解器**，负责颗粒运动、接触、键合与断裂等数值内核及其独立验证。TensorPhysics 负责工作流编排、与其他求解器的数据交换和运行追溯；TensorPlatform 负责共性数据、实验和模型生命周期服务。

破冰应用的目标链路是：冰层 DEM → 船体/结构 FEM → 流体 TensorLBM 或 TensorFVM → TensorPhysics 协调耦合 → TensorPlatform 归档运行与数据 → 经验证的数据用于 AI 代理模型。每一步都必须先通过各自的数值验收和接口守恒检查。

**成熟度说明：** 本仓库描述的二维黏结颗粒模型是研究原型，不应称为已经验证的工程破冰预测器。晶格方向偏差、材料标定、分辨率敏感性、三维物理验证和流固耦合准确性仍是重要限制；外载荷 API 本身不等于已完成耦合。

## 安装与运行

**新手从这里开始：** [教程：从零到一次「可验证」的冰格 DEM 运行](docs/tutorials/first_verified_run.md) —— 15 分钟、全程带断言的 runnable 教程（建格 → 平衡态 → Cauchy–Born 刚度 → 各向同性诊断 → 压入能量账），其代码块由 `tests/test_tutorial_first_verified_run.py` 原样执行并做负例控制。

需要 Python 3.10+。在项目根目录执行：

```bash
python -m pip install -e .
python -m tensordem --steps 2000 --output results
# 有 CUDA 环境时：
python -m tensordem --device cuda --output results/cuda
```

也可使用安装后的 `tensordem` 命令。小规模 CPU 演示通常比 GPU 更合适。
默认生成 21 × 5 个颗粒，并自动选择保守时间步长。

```bash
python -m tensordem --nx 31 --ny 5 --speed 0.5 --breaking-strain 0.01 \
    --shear-breaking-strain 0.02 --steps 5000 --save-every 25 --output results/slow
python -m unittest discover -s tests -v
```

步数与实际时间的关系为 `总时间 = steps × dt`。增大粒子数会减小自动时间步长，
因此可能需要增加步数才能让破冰头接触冰层。`--dt` 可指定更小步长，
超过保守上限会被拒绝；完整选项见 `python -m tensordem --help`。
Python API 中 `DEMConfig` 可调整粒径、密度、厚度、刚度、阻尼及边界条件，
`IceDEM.step()` 推进一步，`IceDEM.diagnostics()` 返回当前状态指标。

## 模型与单位

- SI 单位：m、s、kg、N；二维圆盘质量为 `density × π × radius² × thickness`，
  `thickness` 是面外厚度。默认密度 917 kg/m³，其他默认值为演示参数而非标定冰材参数。
- 初始正方格点的水平、竖直和对角邻居建立中心力弹簧黏结。
  当前键长为 `l`、初始键长为 `l0`，轴向力为 `bond_stiffness × (l - l0)`。
  当轴向拉伸应变超过 `breaking_strain`，或基于局部变形梯度计算的客观等效剪切应变超过 `shear_breaking_strain` 时不可逆断键；纯压缩不会触发轴向拉伸断裂。剪切指标使用 Green-Lagrange 应变，刚体旋转不应触发剪切断裂。
  中心力模型在刚体平移、旋转时不产生虚假应变。
- 未黏结或已断键颗粒接触：重叠量 `δ = max(2r - l, 0)`，
  排斥力大小为 `max(contact_stiffness × δ - contact_damping × v_normal, 0)`，
  只在重叠时生效，阻尼不会产生接触吸引力。完整黏结不叠加接触力。
- 圆形破冰头采用同样的罚接触，与所有颗粒检测碰撞，速度恒定向下。
  CSV 中 `reaction_y` 为冰对破冰头的反力（向上为正），不是边界支反力。
- 使用半隐式 Euler、线性环境阻力 `-drag × velocity` 和双精度浮点数。
  左右边缘位置严格固定；默认不考虑重力，以避免未建模浮力导致冰层先行下落。
  时间步长综合所有可能的接触刚度、阻尼和破冰头行进距离给出保守限制，
  仍应通过减小步长做收敛验证。

## 输出

`history.csv`：采样时刻、破冰头位置、两个方向反力、累计断键数和动能。
包含初始帧、每 `save-every` 步的帧及最终帧。

`trajectory.pt`：配置、实际时间步长、采样时间、颗粒位置和速度、
初始黏结的端点索引及各帧存活掩码、固定颗粒掩码和破冰头位置。
可用 `torch.load("results/trajectory.pt", weights_only=True)` 读取，
绘制颗粒及存活键或分析破冰力曲线。仅加载可信来源的数据。

## 外部载荷接口（为后续 TensorLBM 耦合准备）

IceDEM.forces(external_forces=loads) 和 IceDEM.step(external_forces=loads) 接受形状为 (N, 2) 的 PyTorch 张量，表示每个颗粒在全局 x/y 坐标系中的瞬时外力，单位 N。每个时间步都应重新传入载荷；求解器不会自动累加或缓存外力。输入必须为有限数值，求解器会转换到自身设备和双精度类型。diagnostics() 另外输出 boundary_reaction_x/y，表示固定边界施加给颗粒的合支反力（按固定颗粒未约束力残差取反）。

该接口仅是载荷入口，不代表已完成流固耦合。TensorLBM 侧应先将流体牵引力按对应表面积分成颗粒节点力，并验证总力/力矩传递守恒。

## 两向 LBM–DEM 曳力闭合（`tensordem.lbmdem`）

仓库内自带的 D3Q19 流体求解器与两种浸没边界，用于把颗粒–流体曳力闭合跑到解析锚上，
口径与实测见 [docs/tensordem_coupling_p1.md](docs/tensordem_coupling_p1.md)，
机器可读判据在 `benchmarks/verified/lbm_dem_drag_calibration/{result.json,gates.json}`
（由 `run.py --merge` 程序化重算，门槛不接受手抄）。

**当前自陈（2026-11，P55-N08 实测，不夸大）**：Stokes 极限（Re≈0.05，d=8，L/d=8）
的曳力在扣除周期阵列解析修正后与 $3\pi\mu dU$ 相差 **+0.98%**（≤2% 门通过）；
同一算例对 Smagorinsky 常数（Cs=0/0.5/1/2）散布 **1.19%**、对域长（L/d=8→16）散布
**0.36%**（≤2% / ≤1% 门通过）；全局动量恒等式残差 **6.6e-12**（≤1e-10 门通过）；
自由沉降的曳力–有效重力平衡残差 **0.40%**。
未通过（如实记录）：中低 Re 的 $C_D$（Re=20 在两档分辨率下为 −27.5%/+17.5%，门 ≤5%）、
MEM 与直接强迫两种口径的一致性（+24%，门 ≤2%）、二球串联比值（+12.1%，门 ≤5%，
周期像污染）、沉降终速（+9.2%，门 ≤3%）与沉降启动瞬态的负阻伪迹。
因此 `gates.json.all_recomputed_pass = false`：**曳力闭合目前只在 Stokes 极限定量可用**，
中低 Re、移动边界与二体相互作用仍需更高分辨率、多次强迫迭代与非周期远场。


## 网格/分辨率敏感性扫描

可用独立扫描工具比较不同颗粒间距下的压头反力历史：

```bash
python -m tensordem.sensitivity --nx-levels 7,11,15 --duration 0.12 --speed 0.2 --output sensitivity-results
```

工具尽量保持目标冰层宽度和高度不变，逐级改变颗粒半径；由于离散行数必须为整数，实际高度会有轻微偏差，汇总表会记录每组实际几何、时间步长、峰值反力、反力冲量、断键数和动能。每组的逐步数据写入 `history_nx_*.csv`，指标汇总写入 `summary.csv`；新增的 `comparison_to_finest.csv` 将各组的反力历史按重叠时间区间插值到共同采样时刻，并报告相对最细网格的反力 RMS 差异和峰值差异。该指标用于比较敏感性，不是正式的收敛阶估计。

该扫描用于暴露离散分辨率敏感性，不自动证明收敛。当前中心力晶格存在方向偏差，且各向同性材料响应尚未校准；跨分辨率比较前仍需合理标定本构参数，并对齐物理几何、时间范围和边界条件。默认扫描计算成本会随粒子数近似按平方增长，可先用较小的 `--nx-levels` 做试跑。

## 适用范围

这是教学/研究起步用的二维压入断裂演示，并非完整船舶破冰预测软件：
没有颗粒转动自由度、显式剪切键力/力矩、切向摩擦、弯曲黏结、压碎、海水浮力/流固耦合，
也没有三维船体几何。正方格点与中心力网络有各向异性，刚度与断裂阈值
需要实验标定，不能直接用于工程载荷预测。
全粒子对检测的时间和内存复杂度为 O(N²)，轨迹保存内存随采样帧数增长，
不适合大规模冰场；扩大模型前需要空间邻居搜索与流式轨迹存储。
## Prescribed polyline hull mode (prototype)

The circular indenter remains the default. To prescribe a moving 2-D hull, provide
a CSV of local hull vertices in metres with x,y headers. Vertices are joined
in file order; the local origin is translated by hull-start-x/y and the hull
moves at the prescribed hull-velocity-x/y. The included examples/hull_profile.csv
is only a small synthetic wedge for exercising the API, not a digitized Glacier
profile or validated vessel geometry.

```bash
python -m tensordem --hull-profile examples/hull_profile.csv \
  --hull-start-x -0.2 --hull-start-y 0.05 \
  --hull-velocity-x 0.2 --hull-velocity-y 0 \
  --speed 0.2 --steps 2000 --save-every 10 --output results/hull
```

history.csv records tool_x, tool_y, reaction_x, and reaction_y in SI units.
The reported reaction is the ice-on-prescribed-hull contact reaction; the hull has
no dynamic degrees of freedom and does not respond to that force. Choose an initial
position that avoids unintended initial overlap, and use a conservative speed bound
when selecting --dt. This mode is a prescribed-contact prototype, not a calibrated
ship resistance prediction.


## 3-D bonded-sphere solver (new prototype)

The 2-D API and command remain unchanged. A separate 3-D solver is available with
spherical particles, a prescribed spherical indenter, irreversible tensile/shear
bond failure, post-fracture sphere contact, fixed x-side boundaries, instantaneous
nodal external loads, and fracture-event arrays in the saved trajectory.

```bash
python -m tensordem.dem3d_cli --nx 9 --ny 5 --nz 3 --steps 1000 --save-every 10 --output results-3d
# Or after installation:
tensordem-3d --nx 9 --ny 5 --nz 3 --steps 1000 --device cuda --output results-3d-cuda
python -m unittest discover -s tests -v
```

Python API: `DEM3DConfig` and `IceDEM3D` from `tensordem`. The state uses
`(N, 3)` positions, velocities and external forces (N). The z-axis is vertical;
the indenter reaction is reported as `reaction_x/y/z`. Outputs are
`history.csv`, `trajectory_3d.pt`, and `checkpoint_3d.pt`. The CSV includes
kinetic energy, surviving-bond elastic energy, particle/tool contact potential,
recorded bond-release energy, and the instantaneous mechanical-energy estimate.

This is a first 3-D verification baseline, not yet a validated ship-ice engineering
solver. It uses a simple-cubic central-force network, a prescribed spherical tool,
no particle rotational degrees of freedom or tangential friction. Permanent local
bond topology is constructed in O(N), and current contact candidates are generated
with a uniform spatial cell list rather than a full all-pairs distance matrix.
The cell hash currently runs on CPU, causing host-device synchronization for CUDA;
benchmark before using large GPU cases. The central-force lattice is directionally
biased; calibrate the constitutive response, verify time-step and particle-resolution
convergence, and validate against experiments before interpreting resistance
quantitatively. Large domains still need optimized device-side neighbor search,
performance benchmarking, and streaming trajectory output.


### Checkpoint / restart (3-D)

The 3-D solver now supports strict state restoration for reproducible continuation:

```python
from tensordem import DEM3DConfig, IceDEM3D

cfg = DEM3DConfig(nx=9, ny=5, nz=3)
sim = IceDEM3D(cfg)
for _ in range(500):
    sim.step()

checkpoint = sim.state_dict()
# Persist only trusted checkpoints, e.g. with torch.save(checkpoint, "checkpoint.pt")

resumed = IceDEM3D(cfg)
resumed.load_state_dict(checkpoint)
for _ in range(500):
    resumed.step()
```

Loading validates the dimensionality, exact solver configuration and time step,
array shapes, pair/bond topology, fixed-boundary mask, and finite floating-point
state before mutating the simulation. The CLI saves a checkpoint at each recorded
output and can continue a run from one:

```bash
python -m tensordem.dem3d_cli --steps 1000 --save-every 20 --output results-3d
python -m tensordem.dem3d_cli --restart results-3d/checkpoint_3d.pt \\
  --steps 1000 --save-every 20 --output results-3d
```

The restart command restores physical parameters from the checkpoint; `--device`
may be used to select CPU or CUDA. If the existing `history.csv` is in the output
directory, resumed rows are appended. `trajectory_3d.pt` contains the latest run
segment; the checkpoint is the authoritative full restart state. The energy
columns are instantaneous diagnostics, not a complete energy-conservation audit:
damping work, external work, and numerical integration error are not yet closed
in a cumulative balance. Only load checkpoint files from trusted sources.


### Observation and checkpoint validation correction

The 3-D diagnostics now evaluate the stored bond state without committing new fracture events or changing the stored last reaction. Fracture is committed by explicit force updates (the existing forces default) or stepping. Output sampling therefore does not advance fracture history.

Checkpoint tensors must retain their declared storage dtype before device transfer. Invalid floating masks or fractional topology indices are rejected before conversion and before any state mutation. This does not establish a complete physical admissibility or energy audit of a checkpoint.

Regression evidence and the retained pre-fix reproduction are in [the observation audit](docs/dem3d-observation-audit/README.md).


## DEM3D benchmark paper/report workflow

The paper-style benchmark catalogue, reproducible run commands, figure plan, reporting template, and verification/validation distinctions are documented in [docs/DEM3D_BENCHMARK_PAPER.md](docs/DEM3D_BENCHMARK_PAPER.md).

After running the benchmark cases, assemble comparison figures and an index of generated/skipped plots with:

```bash
python -m pip install -e ".[test,viz]"
python scripts/plot_dem3d_benchmark_paper.py --input-root . --output results-dem3d-paper-figures
python -m unittest discover -s tests -p test_dem3d_paper_figures.py
```

The figure script consumes existing CSV outputs only. It never fabricates absent results; consult `paper_figure_manifest.json` and `paper_figures_index.md` to see which plots were actually generated. Benchmark PASS status means only that the configured software checks passed, not that physical ice properties or full-scale ship resistance have been validated.

## Tensor ecosystem integration

This repository's role, interface boundaries, data-exchange requirements and recommended integration path are documented in [Ecosystem Integration](docs/ecosystem-integration.md).

## Run provenance and optional TensorPlatform registration

After a completed 2-D or 3-D run has produced `history.csv` and its trajectory (`trajectory.pt` for 2-D, `trajectory_3d.pt` for 3-D), create a compact run manifest:

```bash
python -m tensordem.run_manifest --output results --run-id dem-run-001
```

The manifest records SHA-256 checksums and sizes for both outputs. By default the URIs are local `file://` paths and the run is explicitly marked `validation_status=not_run`.

For platform registration, first upload both exact files to shared storage, then register with their common URI prefix:

```bash
python -m tensordem.run_manifest --output results --run-id dem-run-001 \
  --artifact-uri-prefix s3://your-bucket/dem-run-001 --register-platform
```

Set `TENSORPLATFORM_URL`, `TENSORPLATFORM_API_TOKEN`, and `TENSORPLATFORM_ORGANIZATION_ID`; install TensorPhysics in the same Python environment. This command registers metadata only and does not upload or verify remote objects. Set `TENSORDEM_GIT_REVISION` to the actual source commit when running from a Git checkout. This is provenance plumbing, not physical validation.
