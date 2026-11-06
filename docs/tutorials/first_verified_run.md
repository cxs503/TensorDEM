# 教程：从零到一次「可验证」的冰格 DEM 运行

**读者**：第一次接触 TensorDEM 的研究者/工程师，会一点 Python 和材料力学。
**时长**：约 15 分钟（CPU 单核算，全部断言合计 < 10 s）。
**产出**：你亲手跑通一条链路——**建格 → 平衡态自检 → 解析刚度（Cauchy–Born）→ 引擎自算刚度 → 各向异性诊断与修复 → 一次动态压入 + 能量账收敛阶**，
每一步都带**可执行断言**，任何一步物理写错都会立刻变红。

> 这篇教程的每个 ```python 代码块都是顺序可执行的同一段脚本的一部分。
> 仓库里 `tests/test_tutorial_first_verified_run.py` 会把它们**原样抽取并按序执行**，
> 所以本教程不可能"写完就烂"（文档腐烂门）。你也可以直接跑：
>
> ```bash
> TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest tests/test_tutorial_first_verified_run.py -q
> ```

**为什么要先做这种"没意思"的静态验证？**
DEM 破冰算例里的 `bond_stiffness` 是一个**自由建模参数**，它既不是杨氏模量也不对应任何实测冰参数。
如果不知道它等于多少 Pa，你画出来的载荷–位移曲线就只是"某个数"。
本教程第 4–6 步把 `bond_stiffness` 与工程模量**闭式接通**，并量化一个真实陷阱：
方形八邻域中心力格点**不是各向同性介质**，45° 方向比 0° 方向硬 60%。

---

## 第 0 步：环境自检

```python
import math
import sys

import torch

assert sys.version_info >= (3, 10), "TensorDEM 需要 Python 3.10+"
torch_major, torch_minor = (int(part) for part in torch.__version__.split(".")[:2])
assert (torch_major, torch_minor) >= (2, 0), f"torch too old: {torch.__version__}"
print(f"python {sys.version.split()[0]} / torch {torch.__version__}")
```

## 第 1 步：建一个最小冰格，并**手工**核对拓扑

我们建 7 × 5 个半径为 `R` 的圆盘，间距 `a = 2R`，只给截断半径内（即方形 8 邻域）的粒子对建键。
`fix_edges=True` 把最左/最右两列粒子固定（相当于夹持端）。

```python
from tensordem.dem import DEMConfig, IceDEM

RADIUS = 1.0e-3        # m，冰盘半径
K_BOND = 2000.0        # N/m，单键刚度（自由参数，见第 7 步）
THICKNESS = 0.2        # m，二维平面应变假定下的出平面厚度
DENSITY = 917.0        # kg/m^3
SPACING = 2.0 * RADIUS # m，格点间距

config = DEMConfig(
    nx=7, ny=5, radius=RADIUS, thickness=THICKNESS, density=DENSITY,
    bond_stiffness=K_BOND, contact_stiffness=K_BOND, fix_edges=True,
)
sim = IceDEM(config)
```

拓扑必须能被**手算**复现，否则后面所有模量都是错的：

```python
n_particles = config.nx * config.ny
hand_bonds = (
    (config.nx - 1) * config.ny        # 水平键
    + config.nx * (config.ny - 1)      # 竖直键
    + 2 * (config.nx - 1) * (config.ny - 1)   # 两条对角线
)
assert n_particles == 35
assert hand_bonds == 106
assert int(sim.alive.sum()) == hand_bonds, "引擎建键数与手算不一致"

mass_ref = DENSITY * math.pi * RADIUS**2 * THICKNESS
assert math.isclose(config.mass, mass_ref, rel_tol=1e-12)

# 每个格点的邻居数：内部 8，边界更少；内部粒子数 = (nx-2)*(ny-2)
dist = torch.cdist(sim.initial_positions, sim.initial_positions)
cutoff = SPACING * math.sqrt(2.0) * (1.0 + 1e-12)
coordination = ((dist > 0) & (dist < cutoff)).sum(dim=1)
assert int(coordination.max()) == 8
assert int((coordination == 8).sum()) == (config.nx - 2) * (config.ny - 2) == 15
print(f"particles={n_particles} bonds={int(sim.alive.sum())} interior={int((coordination == 8).sum())}")
```

> **陷阱 1**：引擎的截断是 `sqrt(2) * diameter`，只含 8 邻域而**不含第二近邻**。
> 若你按"全部粒子对"数键，键数会变成 C(35,2)=595，刚度会偏大数倍。

## 第 2 步：平衡态——不加载时格子不该动

没有任何外载、也没有初速度时，解析平衡解就是**位移严格为零**。这是最便宜的自检：

```python
max_drift = 0.0
for _ in range(50):
    sim.step()
    max_drift = max(max_drift, float((sim.positions - sim.initial_positions).abs().max()))

assert max_drift == 0.0, f"未加载格子发生了漂移：{max_drift}"
assert sim.positions.dtype == torch.float64, "引擎状态必须是 float64"
assert sim.diagnostics()["broken_bonds"] == 0
kinetic = sim.mechanical_energy()["kinetic_J"]
assert kinetic == 0.0
print(f"50 步后最大位移={max_drift} m，动能={kinetic} J，dt={sim.dt:.3e} s")
```

注意 `sim.dt` 不是随便取的：`DEMConfig.recommended_dt` 由**弹性、阻尼、工具行程**三条上限取最小值，
引擎在 `dt > recommended_dt` 时直接抛 `ValueError`——这是稳定性门，不要绕过。

## 第 3 步：从 `bond_stiffness` 到刚度张量（Cauchy–Born，手推）

中心力键网络在小应变（仿射）假定下，刚度张量是

$$ C_{ijkl} = \frac{1}{A}\sum_b \tfrac{1}{2} k_b l_b^2 n_i n_j n_k n_l $$

其中 `A = a^2` 是每个粒子分摊的面积，`1/2` 是「一根键被两个粒子共享」的权重。
把 8 个方向代进去，得到本格型的闭式：

$$ C_{11}=C_{22}=2k,\quad C_{12}=C_{66}=k $$

```python
from tensordem import lattice_elasticity2d as le

neighbour_steps = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
area_per_particle = SPACING**2
C_hand = torch.zeros(2, 2, 2, 2, dtype=torch.float64)
for step_x, step_y in neighbour_steps:
    norm = math.hypot(step_x, step_y)
    length = norm * SPACING
    direction = torch.tensor([step_x / norm, step_y / norm], dtype=torch.float64)
    weight = 0.5 * K_BOND * length**2 / area_per_particle   # 1/2 = 键共享
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    C_hand[i, j, k, l] += weight * direction[i] * direction[j] * direction[k] * direction[l]

assert math.isclose(float(C_hand[0, 0, 0, 0]), 2.0 * K_BOND, rel_tol=1e-12)
assert math.isclose(float(C_hand[0, 0, 1, 1]), K_BOND, rel_tol=1e-12)
assert math.isclose(float(C_hand[0, 1, 0, 1]), K_BOND, rel_tol=1e-12)

C_ref = le.affine_stiffness_tensor(K_BOND, spacing=SPACING)
assert le.tensor_error(C_hand, C_ref) < 1e-12, "手推求和与模块闭式不一致"
print(f"手推 C11={float(C_hand[0, 0, 0, 0]):.6g} C12={float(C_hand[0, 0, 1, 1]):.6g} "
      f"C66={float(C_hand[0, 1, 0, 1]):.6g} N/m，与模块相对误差={le.tensor_error(C_hand, C_ref):.2e}")
```

单位提醒：平面应变/平面应力下 `C` 的量纲是 **N/m**（每单位出平面厚度），
所以 `C / thickness` 才是 Pa 意义上的模量。

## 第 4 步：让引擎自己算刚度（有限差分），并解释 23% 的表面亏损

上一步是"我推的公式"，这一步是"引擎实际的键能"。`stiffness_from_strain_energy`
对 `sim` 自己的键势能 `mechanical_energy()['bond_J']` 做中心差分：

```python
measured = le.stiffness_from_strain_energy(sim)
rel_err = le.tensor_error(measured, C_ref)
assert rel_err < 1e-5, f"引擎键能与 Cauchy–Born 不符：{rel_err}"
print(f"引擎自算刚度 vs 闭式：相对误差={rel_err:.2e}")
```

**但如果你偷懒只对有限自由板施加仿射应变，直接拿总能除以体积，会低估 ~23%**：

```python
strain = torch.tensor([[1e-4, 0.0], [0.0, 0.0]], dtype=torch.float64)
bulk_density = 0.5 * float(torch.einsum("ij,ijkl,kl->", strain, C_ref, strain))  # N/m

original = sim.positions.clone()
sim.positions = sim.initial_positions @ (torch.eye(2, dtype=torch.float64) + strain).T
raw_density = float(sim.mechanical_energy()["bond_J"]) / (n_particles * area_per_particle)
sim.positions = original

assert 0.75 < raw_density / bulk_density < 0.80, raw_density / bulk_density   # ≈0.771
corrected = le.applied_strain_energy_density(sim, strain)                      # 补上缺失表面键
assert abs(corrected / bulk_density - 1.0) < 1e-4
print(f"裸有限板密度/体密度={raw_density / bulk_density:.4f}，补表面键后={corrected / bulk_density:.6f}")
```

> **陷阱 2**：7 × 5 的板里 20 个粒子在自由表面（占 57%），缺失的键**不在**引擎的键表里，
> 所以"总能/体积"永远小于体模量。做标定时要么用足够大的板做外推，要么显式补亏（模块的 `include_deficit=True`）。

## 第 5 步：方形格点**不是各向同性**——45° 方向硬 60%

两个方向的杨氏模量（平面应力，`E = (C11² − C12²)/C11`）差多少？直接问模块：

```python
directions = le.directional_moduli([0.0, 45.0, 90.0], K_BOND)
young_0 = directions[0]["youngs_modulus_N_per_m"]
young_45 = directions[1]["youngs_modulus_N_per_m"]
young_90 = directions[2]["youngs_modulus_N_per_m"]

assert math.isclose(young_0, 1.5 * K_BOND, rel_tol=1e-12)   # (C11²−C12²)/C11 = 1.5k
assert math.isclose(young_45 / young_0, 1.6, rel_tol=1e-12)   # 偏硬 60%，不是数值噪声
assert math.isclose(young_90, young_0, rel_tol=1e-12)
assert abs(directions[0]["poisson_ratio"] - 0.5) < 1e-12       # 0° 与 45° 泊松比也不同
assert abs(directions[1]["poisson_ratio"] - 0.2) < 1e-12
print(f"E(0°)={young_0:.0f} E(45°)={young_45:.0f} N/m，比值={young_45 / young_0:.3f}")
```

二维中心力网络各向同性的充要条件是 `C11 − C12 = 2·C66`。默认 `k_d = k_a` 时左边 2k、右边 k，
**差一倍**。只需把对角键刚度取一半（`ISOTROPIC_DIAGONAL_SCALE = 0.5`）就能精确补回：

```python
iso = le.stiffness_components(K_BOND, le.ISOTROPIC_DIAGONAL_SCALE)
assert math.isclose(iso["c11"], 1.5 * K_BOND, rel_tol=1e-12)
assert math.isclose(iso["c12"], 0.5 * K_BOND, rel_tol=1e-12)
assert math.isclose(iso["c66"], 0.5 * K_BOND, rel_tol=1e-12)
assert math.isclose((iso["c11"] - iso["c12"]) / (2.0 * iso["c66"]), 1.0, rel_tol=1e-12)

iso_moduli = le.directional_moduli([0.0, 30.0, 45.0, 60.0, 90.0], K_BOND, le.ISOTROPIC_DIAGONAL_SCALE)
spread = max(d["youngs_modulus_N_per_m"] for d in iso_moduli) - min(d["youngs_modulus_N_per_m"] for d in iso_moduli)
assert spread < 1e-9, spread
assert math.isclose(iso_moduli[0]["youngs_modulus_N_per_m"], 4.0 * K_BOND / 3.0, rel_tol=1e-12)
assert abs(iso_moduli[0]["poisson_ratio"] - 1.0 / 3.0) < 1e-12
print(f"修复后 E={iso_moduli[0]['youngs_modulus_N_per_m']:.3f} N/m 各向同性、ν=1/3，方向离散={spread:.1e}")
```

同样的各向同性修正要在**引擎**里成立才算数（改不改引擎代码由你选，这里不动引擎）：

```python
from tensordem.calibrated_dem import CalibratedIceDEM

coefficients = le.isotropic_bond_stiffness(sim)                # 轴键 k，对角键 k/2
assert sorted({float(value) for value in coefficients}) == [0.5 * K_BOND, K_BOND]
calibrated = CalibratedIceDEM(config, bond_stiffness=coefficients)
iso_measured = le.stiffness_from_strain_energy(calibrated)
iso_ref = le.affine_stiffness_tensor(K_BOND, le.ISOTROPIC_DIAGONAL_SCALE, spacing=SPACING)
assert le.tensor_error(iso_measured, iso_ref) < 1e-5
print(f"引擎侧各向同性刚度误差={le.tensor_error(iso_measured, iso_ref):.2e}")
```

## 第 6 步：跑一次真实压入，并审计能量账

现在换成工程尺度的 7 × 5 冰板（`R = 25 mm`），刚性圆形工具以恒速压入。
`tool_gap=0` 时工具初位就在冰板上表面上，接触立刻建立：

```python
tool_radius = 0.01          # m
tool_speed = 0.05           # m/s
impact_config = DEMConfig(
    nx=7, ny=5, radius=0.025, thickness=0.2, density=DENSITY,
    bond_stiffness=K_BOND, contact_stiffness=K_BOND,
    tool_radius=tool_radius, tool_gap=0.0, tool_speed=tool_speed, fix_edges=True,
)
impact = IceDEM(impact_config)
top_row_y = (impact_config.ny - 1) * 2.0 * impact_config.radius
assert math.isclose(float(impact.tool_position[1]), top_row_y + tool_radius + impact_config.radius, rel_tol=1e-12)
assert float(impact.tool_position[1]) > top_row_y
```

跑同样物理时间 `t_end` 的两次仿真，`dt` 差 2 倍，比较能量账残差：

```python
def run_impact(dt_scale: int) -> float:
    """以 dt/recommended_dt = 1/dt_scale 跑固定物理时长，返回相对能量账残差。"""
    cfg = DEMConfig(**{**impact_config.__dict__, "dt": impact_config.recommended_dt / dt_scale})
    sim_ = IceDEM(cfg)
    steps = int(round(400 * impact_config.recommended_dt / cfg.dt))
    for _ in range(steps):
        sim_.step()
    diag = sim_.diagnostics()
    mechanical = sim_.mechanical_energy()["mechanical_J"]
    residual = diag["tool_work_J"] - mechanical - diag["damping_dissipation_J"] - diag["fracture_release_J"]
    run_impact.reaction = diag["reaction_y"]
    run_impact.broken = diag["broken_bonds"]
    assert diag["tool_work_J"] > 0.0
    assert mechanical > 0.0
    assert math.isfinite(mechanical) and math.isfinite(diag["reaction_y"])
    return residual / diag["tool_work_J"]

coarse = run_impact(1)
fine = run_impact(2)
assert -0.01 < coarse < 0.0, coarse         # 残差 ≈ -2e-3：功率取步初量，能量账系统性偏低
assert 0.45 < fine / coarse < 0.55, fine / coarse   # dt 减半 → 残差减半：一阶收敛
assert run_impact.reaction > 0.0             # 冰对工具的反力
assert run_impact.broken == 0                # 弹性区，不该断键
print(f"能量账相对残差 dt={coarse:.3e} → dt/2={fine:.3e}，比值={fine / coarse:.3f}（一阶）")
```

> **为什么残差不是 0？** 半隐式 Euler 用**步初**速度/反力做功率积分，能量账天然带 O(dt) 误差，
> 所以"能量守恒到 0.2%"在离散层面就是**正常值**，而 `残差 ∝ dt` 才是它真的守恒的证据。
> 如果换成 dt/2 残差不下降，那才是 bug（漏项或符号错）。

## 第 7 步：把 `bond_stiffness` 换算成 Pa，并写清适用范围

各向同性版本下 `C11 = 1.5k`、`E_plane = 4k/3`（N/m），除以厚度得到 Pa：

```python
def bond_stiffness_from_young_modulus(young_pa: float, thickness: float) -> float:
    """各向同性格点**轴键**刚度：k_axis = 3/4 * E * thickness（对角键取 k_axis/2）。"""
    return 0.75 * young_pa * thickness

def young_modulus_from_bond_stiffness(k_axis: float, thickness: float) -> float:
    """逆变换：E = 4/3 * k_axis / thickness（前提是 k_diag = k_axis/2 各向同性修正）。"""
    return 4.0 / 3.0 * k_axis / thickness

ice_young = 9.5e9   # Pa，柱状冰/粒间弹性模量的量级参考值
k_axis = bond_stiffness_from_young_modulus(ice_young, THICKNESS)   # 传给 DEMConfig.bond_stiffness
k_diag = le.ISOTROPIC_DIAGONAL_SCALE * k_axis                      # 对角键刚度（各向同性修正）
assert math.isclose(k_diag, 0.5 * k_axis, rel_tol=1e-12)
assert math.isclose(k_axis, 0.75 * 9.5e9 * 0.2, rel_tol=1e-12)
assert math.isclose(young_modulus_from_bond_stiffness(k_axis, THICKNESS), ice_young, rel_tol=1e-12)
# 把这对刚度真的喂回格点模块，三个方向都必须正好 9.5 GPa（不是"其中某个方向"）：
for _row in le.directional_moduli([0.0, 45.0, 90.0], k_axis, le.ISOTROPIC_DIAGONAL_SCALE):
    assert math.isclose(_row["youngs_modulus_N_per_m"] / THICKNESS, ice_young, rel_tol=1e-9)
print(f"E={ice_young/1e9:.2f} GPa @ t={THICKNESS} m → 轴键 k_axis={k_axis:.4g} N/m，"
      f"对角键 k_diag={k_diag:.4g} N/m（喂 DEMConfig.bond_stiffness 的是 k_axis）")
```

> **陷阱 3（差 2 倍，最容易漏）**：`bond_stiffness_from_young_modulus` 返回的是**轴键**刚度
> `k_axis = 3/4 E t`；对角键刚度是它的一半。若把返回值再乘 2 当"轴键"（或把 `k_axis/2` 当轴键）
> 喂给引擎，模量就差 **2 倍**：`k_axis = 1.425e9` → E = 9.5 GPa，而 `2 k_axis = 2.85e9` → E = 19 GPa。
> 若完全不做各向同性修正（对角键 = 轴键 = `k_axis`），0° 方向是 10.69 GPa、45° 方向 17.1 GPa——
> 这就是第 5 步那个 60% 各向异性在"标定"里的直接后果。

**适用范围（本教程验证到哪、没验证到哪）**

1. 本帖只验证**小应变弹性**（`ε ≲ 1e-4`）与**离散能量账**。断裂阈值/剪切破坏/碎冰堆积都不在原位。
2. `C` 是**平面应力**二维化结果；真实冰盖是三维，需用 `dem3d` 的三维 Cauchy–Born 审计（`docs/dem3d_lattice_elastic_constants.md`）。
3. 各向同性修正只改对角键刚度，**不改**晶格拓扑：四重对称的 45° 与 0° 模量差被消掉，但更高阶的方向偏差（例如 22.5°）仍存在，需按 `directional_moduli` 全角度扫描确认。
4. 时间步稳定上限来自 `recommended_dt`；动态压入的载荷–位移曲线还需要单独对标（见 `docs/dem3d_three_point_bending.md`、`docs/indentation_benchmark_protocol.md` 的方法学）。

---

## 常见坑速查

| 症状 | 原因 | 修法 |
| --- | --- | --- |
| 键数比手算多 | 用了全部粒子对而非 8 邻域 | 截断 `sqrt(2)*diameter`，用 `sim.alive` 数键 |
| 刚度比解析小 ~20% | 有限板自由表面缺键 | 补表面亏损或做大板外推 |
| 45° 比 0° 硬 60% | 方形格点四重对称 | `diagonal_scale = 0.5` 或 `isotropic_bond_stiffness` |
| 能量账不闭合 | 半隐式 Euler 的 O(dt) 偏置 | dt 减半验证残差减半；不下降才是 bug |
| `dt exceeds conservative limit` | 手动设了过大 `dt` | 用 `recommended_dt`，先减小 `tool_speed` 再谈加速 |

## 复现

```bash
python -m pip install -e .
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest tests/test_tutorial_first_verified_run.py -q
```

本教程的全部数字都由脚本程序化产出（零手抄）：`python scripts/collect_tutorial_first_verified_run.py`
写出 `artifacts/tutorial_first_verified_run/summary.json`（含 `run_log.txt` 与两张图
`artifacts/tutorial_first_verified_run/figures/directional_modulus.png`、
`artifacts/tutorial_first_verified_run/figures/energy_ledger_convergence.png`）。
`summary.json` 与 `run_log.txt` 随仓库提交（基线），并由门禁 `tests/test_tutorial_first_verified_run.py`
逐字段核对：提交的基线与**重跑一次**的结果必须一致（相对误差 1e-9），正文里写死的数字
（23% 表面亏损、60% 各向异性、C(35,2)=595、≈-2e-3 残差、差 2 倍的标定陷阱）也必须与本次执行一致，
任一处漂移就变红。两张 png 不提交（可用上面的脚本一条命令重建，见 `docs/doc_evidence_ledger.json` 的
`REGENERABLE` 记录）。

学完想继续深入：本篇第 3–5 步背后的完整研究报告（四路径数值验证、方向模量 CSV、
压入后果的定量分析）在 `docs/dem2d_lattice_elasticity.md`，复现脚本 `scripts/validate_lattice_elasticity2d.py`、
模块级测试 `tests/test_lattice_elasticity2d.py`；三维版本见 `docs/dem3d_lattice_elastic_constants.md`。

断言全部通过时你得到的是：**几何/拓扑正确、解析刚度与引擎键能一致（<1e-5）、
各向同性修正有效、动态压入能量账一阶收敛**的冰格 DEM 设备——在这之上谈破冰载荷才有意义。