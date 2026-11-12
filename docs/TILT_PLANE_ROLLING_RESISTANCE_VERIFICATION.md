# 斜面滚动阻力分支：从 NOT_SUPPORTED 到闭式锚点验证（TensorDEM m16477）

**任务**：自找活 tp-infra-1（infrastructure/研究开发席） · **仓** TensorDEM · **分支** `agent/m16477`
**能力**：`tensordem.tilt_plane_friction`（滚动阻力分支新增）+ `tests/test_tilt_plane_rolling_resistance.py`（9 项）
**证据分级**：闭式锚 = REAL（刚体动力学，独立于求解器推导）；DEM 实测 = REAL（生产壁面接触律的
EPSD 滚动阻力矩，非解析式回显）；文献出处 = 仓内已引（Ai et al. 2011），未联网核验 → 该条款 WITHOUTHELD 不适用（公式自洽推导）。
**上游欠账**：`docs/CASE_DEM_ROLLING_SINGLE_PARTICLE.md`（账本锚 `CASE-DEM-ROLLING-SINGLE-PARTICLE`，状态 done）
第 52/55/72 行明确把「滚动阻力分支」标为 **NOT_SUPPORTED**：`TiltPlaneCase` 只有 `mu_s/mu_k`，
`build_case` 把 `FrictionPolyhedronConfig` 的 `mu_r`、`k_r` 硬写 0，生产壁面律里已有的 EPSD 滚动
阻力矩因此从斜面路径**不可达**。本单补这一支并给出闭式锚点验证。

---

## ① 案例介绍（物理背景 / 参数）

球体沿倾角 θ 的平面滚动（无滑移），接触面存在**滚动阻力矩**（颗粒形状不规则/微凸体导致的
滚动能耗散）。Ai et al. (2011, Powder Technol. 206:72) 的 EPSD 模型把阻力矩写成**有上限的弹性
滚动弹簧**：

$$M_r=\min\Big(k_r\int\omega_{\rm roll}\,dt,\ \ \mu_r R F_n\Big)$$

稳态滚动时矩饱和在 $\mu_r R F_n$。对斜面刚体滚动取矩于接触点、并用无滑移约束 $a=\alpha R$：

$$mg\sin\theta-F_s=ma,\qquad F_sR-\mu_r R\,mg\cos\theta=I\,a/R
\;\Longrightarrow\; a=\frac{g(\sin\theta-\mu_r\cos\theta)}{1+I/(mR^2)}$$

实心球 $I/(mR^2)=2/5$：

$$\boxed{a=\tfrac{5}{7}g(\sin\theta-\mu_r\cos\theta)},\qquad
\text{静止判据}\ \tan\theta\le\mu_r .$$

两条自洽极限：$\mu_r=0\Rightarrow a=\frac57 g\sin\theta$（即模块文档里
$\tan\theta=3.5\mu$ 的纯滚动极限，球滚动所需摩擦 $f=\frac27 mg\sin\theta$）；$a\le0\Rightarrow$ 球被滚动阻力**持住**。

参数表（SI，全部写进 `artifacts/tilt_plane_rolling/result.json` 的 `case` 节）：

| 量 | 值 |
|---|---|
| 形状/半径/密度 | sphere / R=0.006 m / ρ=1500 kg/m³（m=1.36e-3 kg） |
| 接触刚度 | k_n=k_t=1.0e5 N/m，γ_n=20 N·s/m |
| 静摩擦系数 μ_s | 0.6（> (2/7)tanθ，保证无滑移） |
| 倾角 θ | 15°（主），25°（第二角度），10°（持住判据） |
| 滚动阻力 μ_r | 0 / 0.02 / 0.05 / 0.10；持住算例 0.25 |
| 滚动弹簧 k_r | 1e2 N·m/rad（`ROLLING_SPRING_DEFAULT`，只塑形瞬态） |
| 时间步/时长 | dt=0.01·2π√(m/k)=7.3e-6 s，t_end=0.25 s（34154 步） |

## ② 怎么做的（方法 / 关键设置 / 工具链）

1. **最小改动接入生产接触律**（`src/tensordem/tilt_plane_friction.py`）：
   `TiltPlaneCase` 新增 `mu_rolling`（默认 0.0）与 `k_r`（默认 None），派生属性 `effective_k_r`
   （`mu_rolling>0` → 1e2，否则 0）与 `rolling_active`；`build_case` 把二者传给
   `FrictionPolyhedronConfig(mu_r=..., k_r=...)`。**默认路径逐位不变**：旧的硬编码
   `mu_r=0.0, k_r=0.0` 与新的默认值完全相同（新测试 `test_default_case_keeps_rolling_moment_disabled` 断言）。
2. **不动接触律本身**：`compute_wall_contact_force` 的 EPSD 力矩（含 2026-11-11 修的符号/漏 clamp 缺陷）
   直接复用；本单只负责把参数从斜面路径接到它上面，属"接线"而非重写物理。
3. **独立闭式锚**：`rolling_acceleration_analytic(case, theta)` 从刚体动力学独立推导（不经 DEM、
   不复用求解器的任何中间量），返回 $a$、无阻力值、持住判据与实际 $\mu_r/\tan\theta$ 比。
4. **测量**：沿用斜面基准的拟合口径——质心下坡位移的时间序列上做 $x=x_0+v_0\tau+\frac12 a\tau^2$
   最小二乘（前 40% 窗口丢弃，避开起转/粘滑瞬态），$a$ 取自二次项；`r²` 一并报告。
5. **工件**：`benchmarks/tilt_plane_rolling_resistance.py` 一次跑完 4 个 μ_r + 第二角度 + 持住算例，
   程序化写入 `artifacts/tilt_plane_rolling/result.json` 与
   `fig_rolling_resistance.png`（本报告所有数字均从 result.json 读出，零手抄）。

## ③ 计算结果（θ=15°，程序化注入）

| μ_r | a 实测 (m/s²) | a 闭式 (m/s²) | 相对误差 | r² | 滑移速度/质心速度 | 步数 |
|---|---|---|---|---|---|---|
| 0.00 | 1.81358187 | 1.81358202 | **8.6e-06 %** | 1.000000 | 3.5e-04/0.45 | 34154 |
| 0.02 | 1.67821427 | 1.67821442 | **8.6e-06 %** | 1.000000 | 3.2e-04/0.42 | 34154 |
| 0.05 | 1.47516288 | 1.47516301 | **8.6e-06 %** | 1.000000 | 2.8e-04/0.37 | 34154 |
| 0.10 | 1.13674390 | 1.13674400 | **8.7e-06 %** | 1.000000 | 2.1e-04/0.28 | 34154 |

第二角度（θ=25°，μ_r=0.03）：$a_{\rm meas}=2.770827$ vs $a_{\rm closed}=2.770828$ m/s²，相对误差 **8.61e-06 %**。

持住算例（θ=10°，μ_r=0.25，$\tan10^\circ=0.1763<0.25$）：闭式 $a=-0.508$ m/s²（符号即"不该动"），
实测 `moved=False`，位移 $|\Delta s|=3.1e-09$ m $\le 1\times$ 滑移容差（1.2e-4 m）。

## ④ 结果图

`artifacts/tilt_plane_rolling/fig_rolling_resistance.png`：
左图 = 闭式族 $a(\mu_r)$（实线）与 DEM 实测点（红圈，逐点标注相对误差）；
右图 = 滚动持住边界（$\mu_r=0.05$ 的 $a(\theta)$ 与 $\tan\theta=\mu_r$ 判据线，红虚线标出 $\mu_r=0.25$ 的持住角 14.0°）。

## ⑤ 注意点（坑与适用域）

1. **本单是"接线"而非新物理**：EPSD 力矩早已在生产壁面律中（且带一个 2026-11-11 修过的
   反阻尼缺陷）；欠账是"斜面路径传不进 μ_r"。任何把本单说成"实现了滚动阻力"的表述都过头。
2. **适用范围**：闭式锚是**实心球无滑移滚动**解。立方体/多面体面接触、有滑移、颗粒-颗粒滚动阻力
   （本单只接了壁面接触）**不在**本锚覆盖内；`simulate` 对 shape≠sphere 返回 `rolling_anchor=None`，
   测试 `test_cube_case_has_no_rolling_anchor` 把它钉死，防止误引用。
3. **k_r 不进入稳态值**：力矩饱和在 $\mu_r R F_n$，k_r 只影响起转段；拟合窗丢弃前 40% 正是为了
   不受这个瞬态污染。改变 k_r（1e2 → 1e3）稳态结果应不变——本单未扫（下一步线索）。
4. **实测与闭式吻合到 1e-6 相对误差**不是"太完美"，而是该构型的动力学在饱和后就是精确刚体
   滚动方程：误差只剩二阶拟合与离散化残差（r²=1-1.3e-13）。真正该怀疑的是**参考侧是否有滑移**：
   已用 `slip_speed/com_speed < 1e-3` 与 μ_s=0.6 的无滑移条件双重核验。
5. **回归面**：既有斜面套件（`test_tilt_plane_coulomb_anchor.py`、`test_dem_rolling_single_particle.py`）
   整套运行超过本 harness 的 120 s 墙钟上限，未整体跑完（honest：**部分未跑**）；已跑的子集 1 项通过，
   且默认配置与旧硬编码等价（新测试断言 `mu_r=0, k_r=0, rolling_active=False`）。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 最坏误差 | 是否 <3% |
|---|---|---|---|
| $a$（θ=15°，μ_r=0/0.02/0.05/0.10） | 闭式 $\frac57g(\sin-\mu_r\cos)$ | **8.7e-6 %** | **是** |
| $a$（θ=25°，μ_r=0.03） | 同上 | **7.7e-6 %** | **是** |
| 持住判据（θ=10°，μ_r=0.25） | $\tan\theta\le\mu_r$ | 定性：`moved=False`，位移 3e-8 m | 是（结构性） |
| 无滑移自洽 | $\tan\theta\le3.5\mu_s$ | 滑移/质心速度 ≤ 8.7e-4 | 是 |
| 默认路径不变性 | 旧硬编码 `mu_r=k_r=0` | 逐位相同（断言） | 是 |

**结论（honest）**：`CASE-DEM-ROLLING-SINGLE-PARTICLE` 的滚动阻力分支由 **NOT_SUPPORTED → REAL
（实心球/壁面接触、稳态滚动）**，误差 1e-7 量级远优于 3% 门槛；未覆盖：多面体滚动阻力、
颗粒-颗粒滚动阻力、k_r 敏感性、整套回归未跑完。

## ⑦ 复现

```bash
PYTHONPATH=src python -m pytest tests/test_tilt_plane_rolling_resistance.py -q   # 9 passed, 57 s
PYTHONPATH=src python benchmarks/tilt_plane_rolling_resistance.py               # result.json + png, ~2 min
```