# TensorDEM

基于 **PyTorch** 的二维黏结颗粒离散元（DEM）破冰原型，支持 CPU / CUDA。
圆形刚性破冰头向下压入两端固定的冰层，黏结可因轴向拉伸或客观剪切应变超阈值而不可逆断裂；
断裂后的碎冰仍通过接触力相互作用。所有力和积分均使用 PyTorch 张量。

## 安装与运行

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
`history.csv` and `trajectory_3d.pt`, including 3-D particle coordinates,
bond survival masks and per-pair fracture metadata.

This is a first 3-D verification baseline, not yet a validated ship-ice engineering
solver. It uses a simple-cubic central-force network, a prescribed spherical tool,
no particle rotational degrees of freedom or tangential friction, and O(N²) pair
search. The central-force lattice is directionally biased; calibrate the constitutive
response, verify time-step and particle-resolution convergence, and validate
against experiments before interpreting resistance quantitatively. Large domains
need a spatial neighbor list and streaming trajectory output.
