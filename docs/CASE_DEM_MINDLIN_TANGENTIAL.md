# CASE-DEM-MINDLIN-TANGENTIAL — Mindlin–Deresiewicz 部分滑移切向接触律验证

> 算例池锚点。ref: Mindlin & Deresiewicz (1953) J. Appl. Mech. 20:327,
> DOI 10.1115/1.4010702；Johnson (1985) Contact Mechanics §4.3-4.4（二级来源）。
> 判据：6 个解析锚点全部通过（零手抄，程序化注入）。

## ① 物理背景

两弹性球体在法向力 F_n 压紧后承受切向加载。Mindlin (1949) 证明无滑移
切向响应为刚度 k_t = 8G*a 的弹簧（a = Hertz 接触半径，G* 为等效剪切模量）。
Mindlin & Deresiewicz (1953) 将其推广至摩擦情形：切向牵引力不能超过 μp，
因此从第一个切向增量起接触边缘就形成**滑移环**，内部**粘着圆**半径 c < a
承载载荷。力-位移关系为：

```
F_t = μF_n [1 - (1 - δ/δ_s)^{3/2}],   δ_s = 3μF_n/(2k_t)
c/a = (1 - F_t/(μF_n))^{1/3}
```

线性 Cundall-Strack 弹簧+Coulomb 钳位使用单一刚度直到钳位，完全丢失了
微滑移曲率：在 δ = 0.5δ_s 处两律相差 18.5% 的 μF_n（1 - 2^{-1.5} = 0.6464
vs 0.5）。

## ② 方法与参数

**纯解析验证**（无 DEM 运行，轻量级）：直接调用 `tensordem.mindlin_partial_slip`
模块的闭式函数，与经典公式逐项比对。

| 参数 | 值 | 含义 |
|---|---|---|
| E | 9.0e9 Pa | 冰杨氏模量 |
| ν | 0.33 | 冰泊松比 |
| R | 0.05 m | 球半径 |
| μ | 0.3 | 摩擦系数 |
| F_n | 10.0 N | 法向载荷 |
| a | 3.336e-4 m | Hertz 接触半径 |
| k_t | 5.407e6 N/m | 无滑移切向刚度 |
| δ_s | 8.322e-7 m |  gross-slip 位移 |

## ③ 验证结果（6 锚点）

| 锚点 | 量 | 期望值 | 实测值 | 误差 | 判定 |
|---|---|---|---|---|---|
| zero_at_no_load | F_t(δ=0) | 0 | 0 | 0 | ✓ PASS |
| gross_slip_at_delta_s | F_t(δ=δ_s) | μF_n = 3.0 N | 3.0 N | 0 | ✓ PASS |
| initial_slope_equals_kt | dF_t/dδ\|₀ | k_t = 5.407e6 N/m | 5.407e6 N/m | 6.5e-8 相对 | ✓ PASS |
| stick_radius_at_half_load | c/a at F_t=0.5μF_n | 0.5^{1/3} ≈ 0.7937 | 0.7937 | 闭式恒等 | ✓ PASS |
| reversal_to_negative_gross_slip | F_t(Δδ=2δ_s) | -μF_n = -3.0 N | -3.0 N | 0 | ✓ PASS |
| reversal_initial_slope_equals_kt | -dF_t/d(Δδ)\|₀ | k_t | k_t | 8.3e-8 相对 | ✓ PASS |

**全部 6 锚点通过。**

## ④ 结果图

### 图1: 加载曲线对比

`artifacts/case_dem_mindlin_tangential/fig1_loading_curve.png`

Mindlin 部分滑移律（蓝实线）与 Cundall-Strack 线性弹簧+钳位（红虚线）的
归一化对比。在 δ/δ_s = 0.5 处，Mindlin 给出 F_t/(μF_n) = 0.6464，而
Cundall-Strack 给出 0.5——相差 14.6% 的 μF_n（即 18.5% 的相对偏差）。

### 图2: 滞回环

`artifacts/case_dem_mindlin_tangential/fig2_hysteresis.png`

从 gross-slip 点卸载→反向加载→再加载的完整滞回环。卸载/再加载分支具有
与加载分支相同的初始斜率 k_t，并在完全反向（Δδ = 2δ_s）后达到 -μF_n。

### 图3: 粘着圆半径演化

`artifacts/case_dem_mindlin_tangential/fig3_stick_radius.png`

c/a 随 F_t/(μF_n) 的演化：从 1（纯粘着）单调递减到 0（完全滑移），遵循
(1 - F_t/(μF_n))^{1/3} 律。

## ⑤ 注意点

- **二级来源**：Johnson (1985) Contact Mechanics 为二级来源（1953 原文
  付费墙，未逐字转录）。闭式公式经交叉验证（与 `tensordem.contact_tangential.
  mindlin_backbone` 独立实现一致，见 `test_mindlin_partial_slip.py` 第 7 项）。
- **准静态假设**：本律为率无关准静态模型，不含切向阻尼（求解器自行添加）。
- **多反转路径**：任意部分滑移转折点的通用多反转路径需要带运动学锚点的
  状态机；本模块仅覆盖单调加载和从 gross-slip 的反转。
- **与 DEM 求解器的集成**：`IceDEM3D` 支持 `tangential_model="mindlin_partial_slip"`
  选项，已在 `test_dem3d_mindlin_branch_reproduces_the_analytic_curve` 中验证
  求解器分支复现解析曲线（10 项测试全部通过）。

## ⑥ 误差判定

6 个解析锚点全部通过，最大相对误差 8.3e-8（数值微分引入），远低于 1% 阈值。
**结论：Mindlin–Deresiewicz 部分滑移律实现正确。**

## 工件

- `src/tensordem/verify_mindlin_partial_slip.py` — 验证脚本（可复现）
- `artifacts/case_dem_mindlin_tangential/verification.json` — 机读结果
- `artifacts/case_dem_mindlin_tangential/fig{1,2,3}.png` — 结果图
- `tests/test_mindlin_partial_slip.py` — 10 项 pytest 测试（全部通过）

复现：`PYTHONPATH=src python src/tensordem/verify_mindlin_partial_slip.py`
