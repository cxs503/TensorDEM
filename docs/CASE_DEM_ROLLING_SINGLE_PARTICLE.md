# CASE DEM-ROLLING-SINGLE-PARTICLE — 单颗粒斜面纯滚动加速度（TensorDEM 算例池）

**池条目**：`DEM-ROLLING-SINGLE-PARTICLE`（tensor_case_pool，tier=core，tol `<2%`）
**锚点**：`NEW:CASE-DEM-ROLLING-SINGLE-PARTICLE`（case-pool pending 项，无 feature_ledger 锚）
**引用**：经典刚体动力学 `a = g sinθ/(1+I/mR²) = (5/7) g sinθ`（均质球，无滑移）；滚动阻力分支 Ai et al. 2011 Powder Technol. 206:269, DOI 10.1016/j.powtec.2010.09.030
**引擎**：`tensordem.tilt_plane_friction`（多面体 DEM + 罚函数接触 + 库仑摩擦，`shape="sphere"`, `rotation_locked=False`）
**工件**：`benchmarks/dem_rolling_single_particle.py`、`artifacts/dem_rolling_single_particle/{results.json,figure.png}`、`tests/test_dem_rolling_single_particle.py`

## ① 案例介绍
边长 2 mm 的均质球（ρ=1500 kg/m³，I=2/5 mR²）静止释放于倾角 θ 的刚性斜面，接触
k_n=k_t=1e5 N/m、γ_n=20、μ_s=0.4，重力 9.81 m/s²。无滑移滚动条件 tanθ < 3.5μ_s（对 μ_s=0.4 为 54.5°），
故验收角 θ=15° 处于纯滚动区。核心量：球心平动加速度 a 与角加速度 α=a/R。

## ② 怎么做的
1. `TiltPlaneCase(shape="sphere", rotation_locked=False, mu_s=0.4, size=2e-3, t_end=0.06 s)`，
   引擎按弹簧周期自动取 dt（`DT_SAFETY=0.01`，每周期 100 步），确定性初速 1e-6 m/s；
2. `dynamic_friction()` 对质心位移做线性最小二乘拟合得到 a（返回 `acceleration_r2`）并给出滑移速度与"朴素反演 μ"；
3. 曲线：θ ∈ {10,15,20,30}° 逐点对 `(5/7) g sinθ` 求相对误差；滑移起始扫描：μ_s ∈ {0.2,0.4}，
   分别在 `atan(3.5μ_s) ∓ 3°` 两侧各跑一次，检查下方复现解析律、上方出现宏观滑移（a/a_analytic > 1）；
4. 单次测量约 15 s（单核），全案 6 次测量 + 出图 < 3 min，未占用集群。

## ③ 计算结果（程序化注入，零手抄）
验收点：a_DEM = 1.813582 m/s²，a_解析 = 1.813582 m/s²，相对误差 **0.0000%**（R² = 1.000000，滑移速度 7.3e-05 m/s）

| θ [deg] | a_DEM [m/s²] | a_解析 [m/s²] | 相对误差 [%] | 滑移速度 [m/s] |
|---|---|---|---|---|
| 10 | 1.216778 | 1.216778 | 0.0000 | 4.88e-05 |
| 15 | 1.813582 | 1.813582 | 0.0000 | 7.26e-05 |
| 20 | 2.396584 | 2.396584 | 0.0000 | 7.73e-05 |
| 30 | 3.503572 | 3.503571 | 0.0000 | 4.81e-05 |

曲线最大相对误差 0.0000%。

滑移起始括号（判据 tanθ_c = 3.5μ_s）：

| μ_s | 极限角 [deg] | 下方角 [deg] | 下方滚动误差 [%] | 上方角 [deg] | 上方 a_DEM/a_解析 |
|---|---|---|---|---|---|
| 0.20 | 35.0 | 32.0 | 0.0000 | 38.0 | 1.042 |
| 0.40 | 54.5 | 51.5 | 0.0000 | 57.5 | 1.043 |

滚动阻力分支的解析参考（本代码路径**不支持**，见 ⑤）：

| μ_r | a = (5/7)g(sinθ−μ_r cosθ) 于 θ=15° [m/s²] |
|---|---|
| 0.0 | 1.813582 |
| 0.01 | 1.745898 |
| 0.02 | 1.678214 |
| 0.05 | 1.475163 |

## ④ 结果图
`artifacts/dem_rolling_single_particle/figure.png`：DEM 点 vs 解析曲线 `(5/7)g sinθ`（实线）
以及带滚动阻力的解析族（μ_r=0.02 虚线、0.05 点线，标注 NOT_SUPPORTED），θ=15° 验收点标注误差。

## ⑤ 注意点 / 局限
* **滚动阻力分支 NOT_SUPPORTED 于此路径**：`TiltPlaneCase` 只有 `mu_s/mu_k`，未把
  `rolling_resistance` 的 μ_r 力矩接入多面体斜面引擎；因此池引用里 Ai et al. 2011 的那一半只给了
  解析参考线，未作 DEM 验证（滚动律本身在 `tensordem.rolling_resistance` 有独立验证面）。
  这是本案例诚实声明的缺口。
* **本案例不是摩擦标定**：自由滚动球的"朴素反演 μ"= (2/7)tanθ（θ=20° 时 0.052），远低于接触 μ_s=0.4；
  测试 `test_naive_inversion_returns_two_sevenths_of_tan` 固化这一点，防止把滚动陷阱当摩擦系数。
* 罚函数刚度 k_n 与 dt 由引擎按弹簧周期取，误差为舍入级（~1e-4 %），故 <2% 门在本算例上极宽松；
  真正的价值在**该门的机器可裁判形式**与滑移起始括号，而非"通过"本身。
* 只测了球（`shape="sphere"`）；立方体/多面体的滚动/翻滚不在本池条目范围。

## ⑥ 误差判定
| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| a（θ=15°） | (5/7)g sinθ = 1.813582 m/s² | 0.0000% | **是** |
| α=a/R（θ=15°） | 906.791 rad/s² | 同 a（R 为常量） | **是** |
| a(θ) 曲线（4 点） | 同上解析 | 最大 0.0000% | **是** |
| 滑移起始角 | atan(3.5μ_s) | 括号内（下方滚动误差 0.00%，上方 a 超出解析 4.2-4.3%） | 是（括号级判定） |
| 滚动阻力分支 | Ai et al. 2011 | **未验证（NOT_SUPPORTED）** | 否（缺口已声明） |

## 复现
```bash
PYTHONPATH=src python benchmarks/dem_rolling_single_particle.py            # 全案 ~2-3 min
PYTHONPATH=src python benchmarks/dem_rolling_single_particle.py --fast     # 仅验收点 ~15 s
PYTHONPATH=src python -m pytest tests/test_dem_rolling_single_particle.py -q
```
pytest 结果（本次运行）：....
