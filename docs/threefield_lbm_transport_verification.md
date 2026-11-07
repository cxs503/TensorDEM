# TensorDEM D3Q19 BGK + Smagorinsky 核心的输运验证与子网格闭合缺陷（m2949）

状态：验证欠账清偿（verifier 席 selfstart，2026-11-03）。本单只做**复核与判定**，除一处闭合式
修正外不引入新物理模型。产物：`src/tensordem/threefield/lbm_transport.py`、
`scripts/audit_threefield_lbm_transport.py`、`tests/test_threefield_lbm_transport.py`、
`docs/threefield-lbm-transport/{audit.json,closure_sweep.csv,shear_wave_sweep.csv,figures/}`。

## 1 案例介绍：为什么审这个核心

`src/tensordem/threefield/wake_lbm.py` 是仓库里唯一一个**真实流体求解器**：D3Q19 单弛豫
（BGK）格子 Boltzmann + Guo 型螺旋桨作动盘推力 + Smagorinsky 亚格子闭合 + 侧向自由滑移壁面，
用来生成螺旋桨尾流（race），再由 `thermal.py` 的 Ranz–Marshall + Stefan 条件把尾流速度折算成
浮冰融化，最后由 `coupling.py` 把质量账并进冰‑桨‑船耦合账本。也就是说：**尾流的速度尺度与
湍流混合尺度直接进入冰质量损失率**。

而在本单之前，这个核心只有两类测试：平衡态矩恢复（`_equilibrium` 的一阶/二阶矩），以及
「短时间跑起来不炸、通量守恒」。两个关键物理量从未与任何参考值对过：

| 待验证的量 | 之前的证据等级 | 依据 |
|---|---|---|
| 运动学黏性 `nu = (tau0 - 1/2)/3`（代码注释里写着） | WITHHELD | 没有任何衰减/输运测量 |
| Smagorinsky 闭合实际实现的涡黏性 `nu_t` | WITHHELD | `_effective_tau` 无任何测试引用 |
| 周期化后的 streaming 算子正确性 | WITHHELD | 仅有开箱段测试 → 无法做谱/解析对比 |

这正是本席「验证欠账」类活的口径：公式写得对 ≠ 按求解器的方式用它是对的。

### 无因次参数（格子单位 dx=dt=1, cs²=1/3）

| 参数 | 符号 | 取值 |
|---|---|---|
| 松弛时间 | tau0 | 0.55 / 0.6 / 0.7 / 0.8 / 1.0 / 1.4 |
| Smagorinsky 系数 | Cs | 0（层流极限）/ 0.05 / 0.1 / 0.16（默认值） |
| 域 | N³ | 12³（闭合检查）、16³（剪切波） |
| 剪切波幅值 / 波数 | A, k | 1e-3, 2π/16（低 Mach：Ma≈1.7e-3） |

## 2 怎么做的

四项独立检查，全部**复用求解器自身的核**（`_equilibrium` / `_effective_tau` / `_stream`），
参考量一律取闭式解，不做事后拟合：

- **A. streaming 精确性**：`LBMConfig` 造的是有界箱，做谱/解析对比必须先有周期闭合。
  方法：把场沿三个轴各补一层**回绕 halo**，`_stream` 的移位本质是 shape 通用的，
  取内层即得精确周期 streaming。参考实现是 `torch.roll` 的独立加和，两者应**逐位相同**。
- **B. 动量扩散率**：`u_x = A cos(k y)` 是 Navier–Stokes 方程的**精确解**（对流项恒为零），
  衰减率 = `nu k²`；格子上同一模态的离散 Laplacian 参考为 `nu·2(1-cos k)`。用余弦模态投影
  提取幅值（避免 `f=f^eq` 初值不是本征向量所带来的前几步瞬态污染），过 20 步后对 `ln A`
  做过定点最小二乘拟合。
- **C. Smagorinsky 闭合**：构造 Chapman–Enskog 非平衡部分 `f^neq_i = -3 w_i tau Q_iab ∂_a u_b`
  （`Q = c c - cs² I`），使 `Pi_ab = -(2/3) rho tau S_ab` 精确成立，于是闭合的**不动点**
  `tau_eff = tau0 + 3 Cs² |S|`、目标 `nu_t = Cs² |S|` 都有闭式。在纯剪切 / 射流剪切 /
  单轴 / 平面应变 / 剪切+法向混合五种场上分别评。
- **D. 极限**：`Cs=0` 必须逐位返回 `tau0`；应变率趋零时 `tau_eff → tau0`。

## 3 计算结果

### 3.1 发现的缺陷：Smagorinsky 张量收缩少算一倍非对角项

`_outer` 只存六个独立分量 `(00,11,22,01,02,12)`，原实现把六格平方和直接开方当 `|Pi|`：

```python
pi_norm = torch.sqrt((pi * pi).sum(dim=-1))          # 旧：Σ_{a<=b}
```

而 `Pi_ab Pi_ab` 中三个非对角分量各出现**两次**，正确写法是 `sqrt(Σ_diag² + 2Σ_off²)`。
后果是 `|Pi|` 在纯剪切中被低估 `sqrt(2)` 倍。判定它确属缺陷（而非另一种约定）的证据是
**常数 18√2 的自洽性**：用完整 Frobenius 范数时，代码自己给出的更新式在不动点处**解析地**
得到 `nu_t = Cs²|S|`（推导见 `lbm_transport.py` docstring）：

```
36 nu_t² + 12 nu_t tau0 = 18√2 Cs² |Pi|/rho, |Pi| = (√2/3) rho tau |S|
=> 3 nu_t² + nu_t tau0 - 3 Cs²|S| nu_t - Cs² tau0 |S| = 0 => nu_t = Cs²|S|
```

修前实测（Cs=0.1, tau0=0.6, `|S|=1` 纯剪切）：`|Pi|_red/|Pi|_full = 0.707107`，
`nu_t` 误差 **−28.3 %**（解析渐近值 `√2/2 - 1 = −29.29 %`）；而**单轴法向应变误差 0.000 %**——
同一个常数在一个场上精确、在另一个场上差 29 %，只能是范数定义错。修正 = 六个分量按
`(1,1,1,2,2,2)` 加权（`_outer_norm_weight`）。

物理相关性：螺旋桨射流的混合层正是 `∂u_x/∂r` 主导的剪切，因此修正前射流混合层的涡黏性
系统性偏小 29 %，而射流核心区（法向应变）不受影响——这也解释了为什么旧尾流基准看起来
「不炸、守恒、还挺像高斯自相似」，欠账完全藏在这一类无声偏差里。

### 3.2 四项检查结果（`docs/threefield-lbm-transport/audit.json`）

| 检查 | 参考 | 最差误差 | 判定 |
|---|---|---|---|
| A streaming（12³, 19 方向） | `torch.roll` 独立实现 | `max_abs_difference = 0.0` | 逐位相同 |
| B 动量扩散率（tau0 6 档） | `nu k²`，`nu=(tau0-0.5)/3` | +1.68 % … −3.18 % | 5/6 在 3 % 带内 |
| C Smagorinsky `nu_t`（5 场 × Cs 3 档） | `Cs²|S|`（闭合不动点） | 8.2e-15（相对 1e-15 量级） | 机器精度 |
| D 极限（Cs=0 / S→0） | `tau_eff = tau0` | 0.0 / 7.7e-14 | 精确 |

B 的逐档数据（`shear_wave_sweep.csv`）：

| tau0 | nu_meas | nu_nominal | 误差 vs nu k² | 误差 vs 2(1-cos k) 参考 |
|---|---|---|---|---|
| 0.55 | 0.016946 | 0.016667 | +1.676 % | +2.992 % |
| 0.6 | 0.033811 | 0.033333 | +1.434 % | +2.748 % |
| 0.7 | 0.067449 | 0.066667 | +1.173 % | +2.483 % |
| 0.8 | 0.100875 | 0.100000 | +0.875 % | +2.182 % |
| 1.0 | 0.166659 | 0.166667 | −0.004 % | +1.291 % |
| 1.4 | 0.290454 | 0.300000 | −3.182 % | −1.928 % |

图：`figures/sgs_closure_error.png`（修正后闭合误差，全在 ±1e-13）、
`figures/shear_wave_viscosity.png`（实测 vs 名义黏性）、
`figures/shear_wave_rate_error.png`（衰减率误差 vs 两条参考）。

## 4 注意点（踩过的坑 / 局限 / 适用范围）

1. **`_stream` 用实例维度切片**，所以周期化时必须临时把 `inst.nx/ny/nz` 指向补齐后的形状，
   否则 halo 被截断（本单第一版就错在这里，streaming 检查报 MISMATCH）。
2. **`torch.linalg.lstsq` 在同一份数据上给出比两端点比值大两个数量级的斜率**（数据本身是干净
   指数，已用 `history/times` 通道逐点核过）。改为显式正规方程后与端点比值一致到 1 %。这是个
   工具自身的坑：向量化 lstsq 在这里不可信，不能拿它当基准。
3. **初值 `f = f^eq` 不是本征向量**，前若干步含松弛瞬态；用逐点 `max|u_x|` 拟合会把瞬态折进
   斜率（首版实测误差 3353 %）。正解是余弦模态投影 + 跳过前 20 步。
4. **离散 Laplacian 参考并不更准**：`nu·2(1-cos k)` 在 tau0 ≤ 1.0 时误差反而更大
   （+1.3…+3.0 %），说明剩余偏差不是「k²→2(1-cos k)」这一项，而是更高阶的 k⁴/Ma² 与
   BGK 谱曲率项；所以不能拿它当「更精确的参考」把误差抹平。
5. **适用域**：全部结论在格子单位、低 Mach（Ma≈1.7e-3）、周期域下成立；有界箱的入口/出口/
   自由滑移壁面误差**未被本单覆盖**（A 项只保证了算子正确，不保证边界条件精度）。
6. 本次修改会轻微改变已发布尾流基准的数值（射流混合层涡黏性变大 → 混合略强），
   `THREEFIELD_ICE_HULL_PROPELLER.md` 的历史数字应在下一次重跑后更新，本单不重跑（容量纪律：
   load15≈119 ≥ 100，只做轻量活）。

## 5 误差判定（逐项，显式回答是否 < 3 %）

| 关键量 | 基准 | 相对误差 | < 3 % ? |
|---|---|---|---|
| 周期 streaming 输出 | `torch.roll` 独立实现 | 0.0（逐位） | 是（精确） |
| `nu_t` / `Cs²|S|`，纯剪切 Cs=0.16 | 闭合不动点闭式 | ~1e-15 | 是（机器精度） |
| `nu_t` / `Cs²|S|`，单轴法向 Cs=0.1 | 同上 | ~1e-15 | 是（机器精度） |
| `tau_eff` @ Cs=0 | `tau0` | 0.0 | 是（精确） |
| 剪切波衰减率 vs `nu k²`（tau0 0.55–1.0） | 解析 NS 解 | ≤ 1.68 % | 是 |
| 剪切波衰减率 vs `nu k²`（tau0 = 1.4, nu=0.3） | 解析 NS 解 | **−3.18 %** | **否（honest FAIL，单点）** |

> **后续（IN52-N02）**：该 FAIL 已在 `docs/threefield_les_resolution_study.md` 清偿为分辨率问题——
> $N=32$ 上 $-0.736\%$、$N=64$ 上 $-0.181\%$，三级 Richardson 外推极限 $-0.014\%$（拟合阶 2.11）；
> 适用域写入该文档 §3.5（$\tau_0\le1.4$ 需 $N\ge32$，$kh\lesssim0.2$）。本节判据与数字保持原样不改写。

最后一行如实记 FAIL：该点 `nu k² = 0.046/步`，120 步内幅值衰减到 e^-5.5，高波数/高阶项修正
在粗网格（k=2π/16）上不可忽略；偏差随 tau0 单调变化（+1.68 → −3.18 %）是截断误差的指纹，
不是实现缺陷（若实现有错，误差不会在 tau0≈1.0 处穿过零点）。补救方向：τ0>1 时用更细网格
（N=32，k 减半）或更短时间窗重测——属分辨率问题，留作下一步而不是靠放宽判据抹平。

## 6 工具链

`PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/audit_threefield_lbm_transport.py`
（torch 2.7.1 CPU，单线程，全套 < 20 s，fail-fast 返回非零码）；
`PYTHONPATH=src python -m pytest tests/test_threefield_lbm_transport.py -q` → 24 passed。