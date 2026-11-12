# CASE-DEM-PLANE-SHEAR-MUIDI 验证报告

**锚点**: `CASE-DEM-PLANE-SHEAR-MUIDI`
**任务**: m16694
**日期**: 2026-11-12
**验证人**: AI agent (selfstart)

---

## ① 案例介绍

### 物理背景
稠密颗粒流表现出率依赖摩擦律，即 μ(I) 流变律（GDR MiDi 2004; da Cruz et al. 2005）：

$$\mu(I) = \mu_s + \frac{\mu_2 - \mu_s}{I_0/I + 1}$$

$$\phi(I) = \phi_c + \frac{\phi_j - \phi_c}{I_0/I + 1}$$

其中：
- $\mu(I)$：有效摩擦系数（$\tau/P$）
- $I$：惯性数 $= \dot{\gamma} d / \sqrt{P/\rho_p}$
- $\phi(I)$：体积分数

### 参考文献
- da Cruz et al. 2005, Phys. Rev. E 72:021309, DOI: 10.1103/PhysRevE.72.021309
- GDR MiDi 2004, Eur. Phys. J. E 14:341, DOI: 10.1140/epje/i2004-10049-0

### 参数表

| 参数 | 值 | 单位 |
|---|---|---|
| 法向应力 P | 62500 | Pa |
| 接触刚度 kn | 62500 | N/m |
| 接触摩擦 μ_contact | 0.5 | — |
| 滚动摩擦 μ_rolling | 0.05 | — |
| 恢复系数 e | 0.3 | — |
| 颗粒半径 r | 0.005 | m |
| 颗粒密度 ρ | 2500 | kg/m³ |
| 参考 μ_s | 0.36 | — |
| 参考 μ_2 | 0.84 | — |
| 参考 I_0 | 0.30 | — |
| 参考 φ_c | 0.58 | — |
| 参考 φ_j | 0.64 | — |

### 验收判据
- μ(I) 曲线 vs 参考值误差 < 10%
- φ(I) 曲线诚实对标（报告偏移）

---

## ② 方法与流程

### 实现模块
- `src/tensordem/mu_i_rheology.py`：ServoShearCase 伺服剪切盒模拟
- `benchmarks/plane_shear_mu_i/`：扫描驱动与分析
- `benchmarks/plane_shear_mu_i/test_plane_shear_mu_i.py`：验证测试

### 关键设置
- 伺服控制法向应力（恒压剪切）
- 周期性边界条件（剪切方向）
- 刚性壁面（横向）
- 6 点惯性数扫描：I_target = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

### 工具链
- Python 3.11 + PyTorch (CPU)
- pytest 9.2.0
- numpy

---

## ③ 计算结果

### μ(I) 曲线对标

| I_target | I_measured | μ_measured | μ_ref | 相对误差 | 判定 |
|---|---|---|---|---|---|
| 0.001 | 0.00101 | 0.3295 | 0.3616 | 8.89% | ✓ <10% |
| 0.003 | 0.00309 | 0.3289 | 0.3649 | 9.86% | ✓ <10% |
| 0.01 | 0.01085 | 0.3850 | 0.3768 | 2.20% | ✓ <10% |
| 0.03 | 0.0346 | 0.4074 | 0.4096 | 0.54% | ✓ <10% |
| 0.1 | 0.1179 | 0.4302 | 0.4954 | 13.16% | ✗ >10% |
| 0.3 | 0.3283 | 0.4916 | 0.6108 | 19.52% | ✗ >10% |

### φ(I) 曲线对标

| I_target | φ_measured (phi_bed) | φ_ref | 备注 |
|---|---|---|---|
| 0.001 | 0.5218 | 0.5802 | 偏低 ~10% |
| 0.003 | 0.5218 | 0.5806 | 偏低 ~10% |
| 0.01 | 0.5218 | 0.5839 | 偏低 ~11% |
| 0.03 | 0.5218 | 0.5933 | 偏低 ~12% |
| 0.1 | 0.5218 | 0.6133 | 偏低 ~15% |
| 0.3 | 0.5218 | 0.6333 | 偏低 ~18% |

**注**：φ_bed 在所有 I 点均为 0.5218，未体现剪切膨胀效应。`phi_bed_steady` 字段在旧扫描数据中缺失（round-3 fix 已添加该字段，但 shipped sweep.json 为旧版本）。

### 测试执行结果

```
benchmarks/plane_shear_mu_i/test_plane_shear_mu_i.py::test_reference_law_limits PASSED
benchmarks/plane_shear_mu_i/test_plane_shear_mu_i.py::test_sweep_artifact_is_well_formed PASSED
benchmarks/plane_shear_mu_i/test_plane_shear_mu_i.py::test_mu_curve_low_mid_I_within_10pct_of_reference PASSED
benchmarks/plane_shear_mu_i/test_plane_shear_mu_i.py::test_phi_curve_confronted_honestly PASSED
benchmarks/plane_shear_mu_i/test_plane_shear_mu_i.py::test_phi_bed_steady_captures_dilatancy SKIPPED
benchmarks/plane_shear_mu_i/test_plane_shear_mu_i.py::test_mu_monotone_in_I PASSED
benchmarks/plane_shear_mu_i/test_plane_shear_mu_i.py::test_cache_is_live_two_point_rerun SKIPPED
```

**5 passed, 2 skipped, 0 failed**

---

## ④ 结果图

### μ(I) 比较曲线

```
μ(I) 曲线: 测量值 vs 参考值 (da Cruz et al. 2005)

I (log)    μ_measured    μ_ref        误差
0.001      ● 0.330       ○ 0.362      -8.9%
0.003      ● 0.329       ○ 0.365      -9.9%
0.01       ● 0.385       ○ 0.377      +2.2%
0.03       ● 0.407       ○ 0.410      -0.5%
0.1        ● 0.430       ○ 0.495     -13.2%
0.3        ● 0.492       ○ 0.611     -19.5%

趋势: 低 I 区 (I<0.03) 吻合良好 (<10%)；高 I 区 (I>0.1) 显著偏低
```

### φ(I) 比较

```
φ(I) 曲线: 测量值 vs 参考值

I (log)    φ_measured    φ_ref        偏差
0.001      ● 0.522       ○ 0.580      -10.0%
0.003      ● 0.522       ○ 0.581      -10.1%
0.01       ● 0.522       ○ 0.584      -10.6%
0.03       ● 0.522       ○ 0.593      -12.0%
0.1        ● 0.522       ○ 0.613      -14.9%
0.3        ● 0.522       ○ 0.633      -17.6%

注: φ_bed 为沉降后高度计算，未反映剪切膨胀；phi_bed_steady 需重新扫描获取
```

---

## ⑤ 注意点与局限

### 已识别问题

1. **高 I 区 μ 偏差超标**：I > 0.1 时误差 >10%（13-20%），可能原因：
   - 伺服控制响应在高剪切率下不足
   - 颗粒惯性效应导致应力波动增大
   - 采样窗口可能不够长以达到稳态

2. **φ(I) 曲线未体现剪切膨胀**：当前 shipped sweep.json 使用 `phi_bed`（沉降高度），该值在所有 I 点恒定。`phi_bed_steady` 字段已在 round-3 fix (m16568) 中添加，但需要重新运行扫描才能获取。

3. **测试覆盖缺口**：
   - `test_phi_bed_steady_captures_dilatancy`：SKIPPED（需要 MU_I_SWEEP=1 重新扫描）
   - `test_cache_is_live_two_point_rerun`：SKIPPED（需要实时重跑）

### 适用范围
- 本验证针对 I ∈ [0.001, 0.03] 范围有效（误差 <10%）
- I > 0.1 的结果需声明为"未达标"，不可用于工程标定

### 改进建议
1. 重新运行全 6 点扫描（MU_I_SWEEP=1），获取 `phi_bed_steady` 数据
2. 对高 I 点增加采样步数或调整伺服增益
3. 考虑增加颗粒数以减小有限尺寸效应

---

## ⑥ 误差判定

| 关键量 | 基准 | 实测 | 误差 | 是否 <10% |
|---|---|---|---|---|
| μ(I=0.001) | 0.3616 | 0.3295 | 8.89% | ✓ |
| μ(I=0.003) | 0.3649 | 0.3289 | 9.86% | ✓ |
| μ(I=0.01) | 0.3768 | 0.3850 | 2.20% | ✓ |
| μ(I=0.03) | 0.4096 | 0.4074 | 0.54% | ✓ |
| μ(I=0.1) | 0.4954 | 0.4302 | 13.16% | ✗ |
| μ(I=0.3) | 0.6108 | 0.4916 | 19.52% | ✗ |

**结论**：低-中惯性数区（I ≤ 0.03）满足 <10% 容差；高惯性数区（I ≥ 0.1）超出容差，需进一步调参或声明局限。

### 工件指针
- 扫描数据：`benchmarks/plane_shear_mu_i/runs/sweep.json`
- 测试代码：`benchmarks/plane_shear_mu_i/test_plane_shear_mu_i.py`
- 实现代码：`src/tensordem/mu_i_rheology.py`
- 测试执行：`pytest benchmarks/plane_shear_mu_i/test_plane_shear_mu_i.py -v` → 5 passed, 2 skipped

---

*本报告由 AI agent 自动生成，所有数值从 sweep.json 程序化提取，零手抄。*
