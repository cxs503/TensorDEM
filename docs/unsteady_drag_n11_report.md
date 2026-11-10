# 非定常曳力最小闭合面成熟化（IN71-N11）
## 1. 案例介绍
颗粒-流体两相耦合中，碎冰/泥沙颗粒的瞬态水动力不能只用准稳态阻力描述。按 Maxey–Riley 方程把作用在球上的水动力写成**四项可分离账本**：

$$F_{\rm total}=F_{\rm qs}(\text{Schiller–Naumann})+\underbrace{\tfrac12\rho_f V\frac{d(u_f-u_p)}{dt}}_{\text{附加质量}}+\underbrace{6a^2\sqrt{\pi\rho\mu}\int_0^t\frac{\dot{u}_{slip}}{\sqrt{t-\tau}}d\tau}_{\text{Basset 历史力}}+\underbrace{\rho_f V\frac{du_f}{dt}}_{\text{压力梯度}}$$

参数表：水 $\rho_f=1000.0$ kg/m³、$\mu=0.001$ Pa·s（$\nu=10^{-6}$ m²/s）；球径 $d=0.002$ m；密度比 $\rho_p/\rho_f\in\{0.9, 1.05, 1.5, 7.8\}$；振荡流 $\varepsilon=\omega a^2/\nu\in\{0.1, 1.0, 10.0\}$。单向耦合（流场给定），线性 Stokes 区 + 中等 Re 的 Schiller–Naumann；不含涡脱落横向力。
## 2. 怎么做的
- **三项分离的对照实验**：稳态匀速（只有准稳态项）、突然加速球（只测附加质量）、正弦振荡流（历史力主导）。
- **解析锚**：Landau–Lifshitz §24 精确非定常 Stokes 解 $\hat F/(6\pi\mu a\hat W)=1+\lambda+\lambda^2/9$（滑移力）、定球在振荡流中 $1+\lambda+\lambda^2/3$（多出的 $2\lambda^2/9$ 即 Froude–Krylov 压力梯度）、Stokes 脉冲启动 $F(t)=6\pi\mu aU(1+a/\sqrt{\pi\nu t})$。
- **Basset 核**：参考实现＝分段常数精确求积（自区间奇异项解析积分）$w(m)=2\sqrt{\Delta t}(\sqrt m-\sqrt{m-1})$，O(N²)；O(N) 递推用指数和近似 $w(m)\approx\sum_k c_k\rho_k^{m-1}$（相对误差加权最小二乘，松弛率对数网格 $s_k\in[1/(2N\Delta t),10/\Delta t]$，vH–Winckelmans 思路），每项一个状态变量；长序列用分块 BLAS/`np.convolve` 实现（块内 Toeplitz、块间状态），与逐标量递推逐位一致。
- **振荡流驱动**：$u_f(y,t)=U_0e^{-\kappa y}\cos(\omega t-\kappa y)$（Stokes 第二问题精确解，$\kappa=\sqrt{\omega/2\nu}$）作为局部流场输入。
- **API**：`POST /api/v1/dem/unsteady_drag`（纯函数 handler + FastAPI router 适配，6 种 mode）。
- 工具链：Python 3.11 + numpy/scipy(最小二乘)/matplotlib；所有数字由 `benchmarks/unsteady_drag_n11_validation.py` 程序化注入本报告，零手抄。
## 3. 计算结果（判据逐项）
| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| (a) 附加质量系数 $A=F_{am}/(\rho_fV\dot u)$ | 5 % | 0.000 % | ✅ |
| (a) 高频渐近 $A_{total}(\alpha\to\infty)\to1.5$ | 5 % | 0.501 %（$\alpha=300$） | ✅ |
| (b) 稳态总力 vs Schiller–Naumann | 1 % | 0.00e+00 % | ✅ |
| (b) 稳态 Basset 残留 $/|F_{qs}|$ | $<10^{-3}$ | 0.00e+00 | ✅ |
| (b) 缓升-保持后历史力 $\propto1/\sqrt t$（实测/解析） | 10 % | 4.99 % | ✅ |
| (b) 脉冲启动 vs Stokes $1/\sqrt{\pi\nu t}$（平均） | 5 % | 0.13 %（最大 10.1 %） | ✅ |
| (c) 振荡流幅值 vs LL §24 | 10 % | max 0.61 % | ✅ |
| (c) 振荡流相位滞后 vs LL §24 | 5° | max 0.14° | ✅ |
| (d) 递推 vs 直接卷积全时程 L2（$n_{exp}=16$） | 2 % | 0.0063 % | ✅ |
| (d) 耗时比（10⁴ 步，CPU 时间，取 3 次最小） | ≥10× | 37.5× | ✅ |
| (d) 三系数递推是否达标（对照） | 不应达标（证明 $n_{exp}$ 必要） | L2 = 34.8 % | ✅ |
| (e) 附加质量对密度比不变 | 5 % | 0.000 % | ✅ |
| (e) $\rho_p\approx\rho_f$ 时历史力占比（滑移力口径） | >20 % | 70.6 % | ✅ |
| (f) 多球 $\phi\to0$ 收敛到单球（最小 $\phi$） | 10 % | 3.56 %（$\phi=1e-06$） | ✅ |
| (f) $\phi^{1/3}$ 外推截距 | 10 % | 0.021 %（截距 0.9998） | ✅ |
| (g) 时间步减半递推误差不增长 | ≤1.5× | 比值 1.000 | ✅ |

### Basset 递推项数扫描（10⁴ 步，$\Delta t=10^{-5}$ s）

| $n_{exp}$ | 核相对 L2 | 全时程 L2 | 最大相对误差 | 递推 CPU 时间 [s] | 相对 O(N²) 加速 |
|---|---|---|---|---|---|
| 3 | 2.88e-01 | 3.48e-01 | 3.44e-01 | 0.0038 | 45.3× |
| 8 | 7.62e-03 | 6.93e-03 | 6.20e-03 | 0.0053 | 32.2× |
| 16 | 4.66e-05 | 6.27e-05 | 5.18e-05 | 0.0046 | 37.5× |
| 24 | 4.26e-06 | 1.65e-06 | 1.32e-06 | 0.0058 | 29.5× |

O(N²) 参考（向量化内积）CPU 时间：0.1721 s。

### 振荡流三点（$\varepsilon\in\{0.1,1,10\}$）

| $\varepsilon$ | $\alpha$ | $Re_{amp}$ | 幅值实测 [N] | 幅值解析 [N] | 幅值误差 | 相位滞后实测 | 相位解析 | 相位误差 | 历史力占比 |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.224 | 2.00e-02 | 2.3712e-10 | 2.3567e-10 | 0.61 % | 11.87° | 11.86° | 0.01° | 25.2 % |
| 1.0 | 0.707 | 2.00e-02 | 3.7802e-10 | 3.7684e-10 | 0.31 % | 31.44° | 31.36° | 0.08° | 49.9 % |
| 10.0 | 2.236 | 2.00e-02 | 1.2164e-09 | 1.2142e-09 | 0.18 % | 59.98° | 59.84° | 0.14° | 49.1 % |

### 密度比扫描（振荡流 $\varepsilon=1$，自由响应颗粒）

| $\rho_p/\rho_f$ | 滑移/驱动速度 | 附加质量系数 | 历史力占滑移力 | 历史力占总力 |
|---|---|---|---|---|
| 0.9 | 0.0082 | 0.5000 | 91.8 % | 35.7 % |
| 1.05 | 0.0044 | 0.5000 | 95.4 % | 35.7 % |
| 1.5 | 0.0387 | 0.5000 | 70.6 % | 31.7 % |
| 7.8 | 0.3515 | 0.5000 | 49.8 % | 39.1 % |

### 多球稀释极限

| $\phi$ | 盒长 [m] | $F(\phi)/F_{\rm single}$ |
|---|---|---|
| 1e-06 | 0.3224 | 1.0356 |
| 1e-05 | 0.1496 | 1.0768 |
| 1e-04 | 0.0695 | 1.1655 |
| 1e-03 | 0.0322 | 1.3572 |

## 4. 结果图
Fig. 1 – four-term ledger at $\varepsilon=1$ (left) and amplitude/phase agreement with the exact unsteady Stokes solution (right).

![fig1](../artifacts/unsteady_drag_n11/fig1_oscillatory_ledger.png)
Fig. 2 – O(N) recursion vs the O(N²) reference (left) and the kernel-term sweep with the 2 % / 10× targets (right).

![fig2](../artifacts/unsteady_drag_n11/fig2_basset_recursion.png)
Fig. 3 – per-sphere force vs $\phi^{1/3}$ (left) and the $1/\sqrt{t}$ history decay after a ramp (right).

![fig3](../artifacts/unsteady_drag_n11/fig3_dilute_and_decay.png)
Fig. 4 – added-mass coefficient (flat) and history-force share vs density ratio.

![fig4](../artifacts/unsteady_drag_n11/fig4_density_ratio.png)

## 5. 注意点（踩坑 / 局限 / 适用范围）
1. **解析参考的相位符号**：`F(t)=Re[F̂e^{iωt}]` 下用拟合 `F≈a cosωt + b sinωt`，滞后角必须是 `-atan2(b,a)` 与解析 `atan2(Im F̂, Re F̂)` 同号；此前的实现在这里差一个符号，导致相位判据看似 60° 偏差。
2. **定球 vs 滑移**：定球在振荡流中总力是 $1+\lambda+\lambda^2/3$，只算滑移力是 $1+\lambda+\lambda^2/9$，两者的差就是 Froude–Krylov 项 $2\lambda^2/9$。若混用，幅值误差可达 2 倍。
3. **准稳态非线性**：Schiller–Naumann 在 $Re=20$ 时相对 Stokes 放大 2.1 倍，线性解析锚只在 $Re\ll1$ 可比；非 Stokes 区必须把 SN 因子乘到准稳态项再比。
4. **自区间奇异项**：直接卷积必须对 $j=n$ 的 $\int_{(0)}^{\Delta t}\tau^{-1/2}d\tau$ 解析积分，丢掉它会同时偏幅值和相位。
5. **速度阶跃的差分格式**：中心差分会把一个速度阶跃摊到两个样本，多出一个半脉冲，历史上力虚高约 50 %；阶跃场景必须用后向差分（模块 `derivative='backward'`）。
6. **BLAS 线程陷阱**：128 核机器上 256×256 的 `@` 单次约 70 ms（线程池开销），比它要替代的 `np.convolve` 慢三个数量级；分块实现因此用 `np.convolve`，且计时取 3 次最小 CPU 时间来抑制共享机器的噪声。
7. **局限**：单向耦合（流场给定）；历史力核为线性 Stokes 核，未做非线性/有限 Re 修正；多球只叠加速度扰动（Stokeslet+dipole）与准稳态阻力，未包含多球历史力耦合；不含涡脱落横向升力与尾流记忆；$\rho_p=\rho_f$ 时滑移恒为零（线性模型给出的解析结论），历史力占比只在近中性（$\rho_p/\rho_f\approx1$）有意义。

## 6. 误差判定（逐项，显式回答是否 <3%）
> 口径声明：账本里的 $F_{am}=\tfrac12\rho_fV\dot u_{slip}$ 是**定义式**，所以「$A$ 相对误差 = 0」只证明实现与定义一致（结构性检查），真正独立于该定义的是**高频渐近锚**——精确非定常 Stokes 解在 $\alpha\to\infty$ 时总力系数必为 $1.5(=1_{\rm PG}+0.5_{\rm AM})$，实测偏差 0.501 %，这才是有内容的一致性检验。
- (a) 附加质量 $A$ 相对误差 0.000 % → **<3% ✅**（高频渐近 0.501 % → **<3% ✅**）。
- (b) 稳态总力 vs SN：0.00e+00 % → **<3% ✅**；Basset 残留 0 → **<3% ✅**；缓升后历史力 vs 解析 $1/\sqrt t$：4.99 % → **满足战役判据(10 %) ✅，但未达 3 % ❌**（来源：缓升折点被分段常数求积抹平，误差 $O(\Delta t)$，随 $\Delta t$ 减小收敛；该量是“离散核 vs 解析核”的一致性，非物理模型误差）；脉冲启动平均误差 0.13 % → **<3% ✅**（最大 10.1 %，出现在 $t\le3\Delta t$ 的阶跃分辨区，非渐近区，如实列出）。
- (c) 振荡流幅值误差 max 0.61 % → **<3% ✅**；相位误差 max 0.14° → **<3% ✅**（判据 5°）。
- (d) 递推 vs 直接卷积 L2 = 0.0063 % → **<3% ✅**（这是同模型两种离散的差异，非物理误差）；对照：三系数核 34.8 % → **>3% ❌**，故取 16 项。
- (e) 附加质量对密度比：0.000 % → **<3% ✅**；历史力占比 70.6 %（是不等式判据，非误差量）。
- (f) 多球→单球：最小 $\phi$ 直接比 3.56 % → **<3% ❌**（$\phi=10^{-6}$ 时为 3.56 %，判据是 10 %，**满足判据 ✅**，但未达 3 %）；$\phi^{1/3}$ 外推截距误差 0.021 % → **<3% ✅**。
- 说明：本表的“误差”分两类——与**解析解/参考实现**的误差（应 <3 %）与**收敛性/统计量**判据（(f) 收敛、(e) 占比），后者按战役判据口径给出，不强行套 3 %。

## 7. 审计链
- 代码：`src/tensordem/unsteady_drag.py`（闭合面 + 递推核 + 场景）、`src/tensordem/unsteady_drag_api.py`（endpoint handler）。
- 用例：`tests/test_unsteady_drag.py`（46 项：含两条负控与 ω→0 退化一致性）。
- 本脚本：`benchmarks/unsteady_drag_n11_validation.py`；数字源：`validation.json`。
- 复现：`PYTHONPATH=src python -m pytest tests/test_unsteady_drag.py -q`；`PYTHONPATH=src python benchmarks/unsteady_drag_n11_validation.py`。
