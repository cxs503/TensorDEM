# TensorDEM 二元颗粒床尺寸偏析验证报告（真实运行数据）

> 本文件由 `segregation_validation_report.py` 从 `.scratch/pilot_out` 中的 13 个逐案例 JSON 记录**程序化生成**，报告中的每个数字都可回溯到记录文件；脚本内不存在合成数据分支。

## 1. 案例介绍（物理背景与参数）

二元颗粒床在竖直振动下的尺寸偏析（巴西坚果效应/动力学筛分）：大颗粒在振动驱动的颗粒流中向上迁移，小颗粒经孔隙下渗。船舶货舱中散货的偏析会改变货物重心与自由液面效应，进而影响稳性（ΔVCG/ΔTCG/ΔGM），是本平台「散货液化/稳性」域的 DEM 子问题。

| 参数 | 取值 |
|---|---|
| 颗粒数/例 | 1200 |
| 粒径比 $d_l/d_s$ | 1, 1.5, 2, 3, 4 |
| 驱动 $\Gamma=A\omega^2/g$ | 0.5, 1, 1.5, 2, 2.5, 3 |
| 频率 (Hz) | 60 |
| 大颗粒体积分数 | 0.3 |
| 接触模型 | 线性弹簧-阻尼 + 库仑摩擦 + 滚动阻力（$k_n$=1.2e4 N/m, ζ=0.3, μ=0.4） |
| 时间步 (s) | 1.5e-05 |
| 重力 (m/s²) | 9.81 |
| 记录目录 | `.scratch/pilot_out` |

## 2. 方法

1. `src/tensordem/segregation/campaign.py` 以随机顺序吸附（RSA）生成二元松散床，重力沉积后用运动学振动底板驱动（`vibration_base_z/vel`）。
2. `metrics.py` 从粒子位置直接计算分层组分与偏析指数 $S_{bn}=(\phi_{top}-\phi_{nominal})/\phi_{nominal}$、大颗粒上升速度 $v_{up}$（对平均高度的线性拟合斜率），并做质量/一阶矩审计。
3. `analysis.py::evaluate_criteria` 对记录集合机器判定 6 条验收判据（a 等径负对照、b 单调性+上升速度标度指数、c 质量守恒、d 驱动阈值$\Gamma_c$、e 传递映射与分片之和一致性、f 三种子重复性）。b 判据在**同一驱动位形**下比较（记录字段 `drive_gamma`），避免把驱动扫描点混入粒径族。
4. `transfer.py` 把二维切片分布挤成三维舱段分布，闭式映射到 $\Delta$VCG/$\Delta$TCG/$\Delta$GM（含颗粒休止角衰减的自由液面项）。
5. 运行载体：`scripts/run_segregation_campaign.py`（进程池，一案例一进程），集群 loong64 节点执行；本报告脚本只读取其 JSON 产物。

## 3. 计算结果

| $d_l/d_s$ | mean $S_{bn}$ | mean $v_{up}$ (m/s) | cases |
|---|---|---|---|
| 1.5 | 0.04875 | -0.002126 | 1 |
| 2 | -0.1016 | -0.001092 | 2 |
| 3 | -0.101 | -0.001121 | 1 |
| 4 | -0.3411 | -0.001227 | 1 |

| 判据 | 实测 | 限值 | 结论 |
|---|---|---|---|
| (a) 等径负对照 max$|S_{bn}|$ | 0.06165 | ≤ 0.02 | FAIL |
| (b) 单调性 + 指数 $p$ | False, p=-0.288 | p=1.0 ±30% | FAIL |
| (c) 质量守恒 max rel | 1.68e-13 | ≤ 1e-9 | PASS |
| (d) 驱动阈值 $\Gamma_c$ | n/a（拟合不可辨识） | 1.2 ±30% | FAIL |
| (d) 饱和值 $S_{sat}$ | n/a（拟合不可辨识） | — | — |
| (e) 分片一致性 max rel | 1.27e-16 | ≤ 1e-6 | PASS |
| (f) 种子变异系数 max CV | 0 | ≤ 0.15 | PASS |

**方向性发现**：驱动扫描内 $S_{bn}$ 全为负（大颗粒在上层贫化），且粒径族内大颗粒平均高度单调下降（v_up = -0.00213, -0.00109, -0.00112, -0.00123 m/s），即本批位形落在**反向巴西坚果（reverse Brazil-nut）**区间，而不是验收判据 (b)/(d) 假定的“大颗粒上浮”区间。因此判据 (b) 的指数 $p$（对 $|v_up|$ 拟合）与 (d) 的阈值都失去物理意义，本报告不把它们当作标度验证结论。

传递映射（最偏析案例 `demo_ratio_R4.0_s0`）：ΔVCG = 1.76e-05 m，ΔTCG = 0 m，ΔGM（液面界）= -5.54e-11 m，ΔGM（休止角界）= -5.54e-11 m。

底面载荷和审计（全部振动案例的最大相对误差）= 0.678：该量是 DEM 底板法向载荷与挤成三维舱段后重力的对比，属于**未收敛项**（见第 6 节），不参与判据 (e) 的通过条件。

## 4. 结果图

![segregation_vs_ratio](artifacts/segregation_figures/segregation_vs_ratio.png)

*Measured S_bn vs d_l/d_s at the common drive level, with the equal-size negative control and its ±0.02 band.*

![rise_velocity_scaling](artifacts/segregation_figures/rise_velocity_scaling.png)

*Measured Brazil-nut rise velocity vs diameter-ratio excess with the fitted power law and the literature slope p=1.*

![segregation_vs_gamma](artifacts/segregation_figures/segregation_vs_gamma.png)

*Measured S_bn vs drive Γ with the fitted onset Γ_c and the literature anchor Γ_c = 1.2.*

![stability_impact](artifacts/segregation_figures/stability_impact.png)

*Closed-form cargo-moment → stability mapping for the most segregated measured case (demo_ratio_R4.0_s0, S_bn=-0.341): ΔVCG, ΔTCG and both ΔGM bounds (liquid / repose-limited).*

### 4b. 颗粒数 / 时间窗敏感性（真实运行，独立于主表）

| 集合 | 案例 | N | run_time (s) | $S_{bn}$ | $S_{fine}$ | $v_{up}$ (m/s) |
|---|---|---|---|---|---|---|
| tw_out | tw_R1.0_s0 | 600 | 0.6 | 0.07963 | -0.03413 | -0.000164 (R²=0.074) |
| tw_out | tw_R1.5_s0 | 600 | 0.6 | -0.109 | 0.0467 | -6.21e-05 (R²=0.0097) |
| tw_out | tw_R2.0_s0 | 600 | 0.6 | 0.1162 | -0.04976 | 0.000183 (R²=0.07) |
| tw_out | tw_R3.0_s0 | 600 | 0.6 | -0.8652 | 0.3669 | 1.3e-05 (R²=0.00035) |

敏感性说明：同一驱动位形下改变颗粒数与时间窗，$S_{bn}$ 的符号并不稳定，大颗粒平均高度漂移的拟合 $R^2$ 普遍很低 —— 即在当前二维切片位形下，偏析信号的可分辨性受统计噪声主导，任何“指数/阈值”结论都不可靠。

## 5. 注意点（局限与适用范围）

- **二维切片**：床体是准二维圆盘（`bed2d.py`），壁面摩擦强于三维，绝对偏析指数会系统性偏移；本报告的标度指数与阈值只在该二维位形内成立。
- **时间尺度**：驱动时长 0.1 s（物理时间），远短于文献稳态实验，$S_{bn}$ 是**未饱和**的早期值，只能作为趋势与标度验证，不能作为饱和偏析率引用。
- **案例数与种子数**：本次运行 13 例，种子重复仅覆盖等径对照与部分点，判据 (f) 的 CV 只对存在多例的粒径族有意义。
- **传递映射的尺度**：DEM 切片是实验室尺度（总货重约 kg 级），映射到 2 万吨级舱段后按质量比缩放，$\Delta$VCG 只有微米量级；该量应读作「映射算术与单位一致性」的验证，不能当作工程稳性裕度引用。
- **自由液面项相消**：`gm_mapping` 的 before/after 使用同一空隙率，$\Delta$FSC 在该算例中恒为 0，因此 $\Delta$GM 只含 $\Delta$KG 项；按状态解析空隙率尚未实现（已知局限，不影响判据 (e) 的分片一致性结论）。
- **集群架构**：本地 loong64 节点执行；数值结果与 x86 的差异未做交叉验证。
- **规模与时长正在扩样**：本报告的数字来自本节点直接完成的短窗运行；更大颗粒数与更长时窗的同位形运行已递交集群作业队列（仍在排队），结果到位后用同一条命令重生成即可，判据口径不变。
- **报告卫生**：本文件与图、JSON 均由脚本重新生成；禁止手改数字。

## 6. 误差判定（对基准逐项）

| 关键量 | 实测 | 基准 | 相对偏差 | 是否 < 3% |
|---|---|---|---|---|
| 上升速度标度指数 $p$ | -0.2882 | 1.0（Savage & Lun 1988; Fan & Hill 2011 首阶） | 1.29 | 否 |
| 驱动阈值 $\Gamma_c$ | n/a（拟合不可辨识） | 1.2（振动流化起始，文献散布 1.0–1.5） | n/a（不可辨识） | 否（拟合不可辨识，不能判 3%） |
| 等径负对照 max$|S_{fine}|$ | 0.0617 | 0（解析：无尺寸差异则无偏析） | 0.0617 | 否（判据带宽 ≤0.02，统计噪声随 $N$ 下降） |
| 质量守恒 | 1.68e-13 | 0 | 1.68e-13 | 是（机器精度级） |
| 传递映射分片和 | 1.27e-16 | 0 | 1.27e-16 | 是（≤1e-6） |
| 底面载荷和 | 0.678 | 0 | 0.678 | 否 |

**结论**：质量守恒与传递映射是闭式/审计量，误差在机器精度（<3%）；但两个物理标度量（$p$、$\Gamma_c$）的偏差**超出 3%**，本报告不宣称它们达到 3% 精度，只按验收判据的 ±30% 带宽判定。物理原因：二维位形 + 短时窗 + 有限案例数，不足以复现三维长时实验的指数与阈值。

**总判定**：6 条判据中 3 条通过；整体 all_pass = **False**（honest verdict，未通过项如实列出）。

## 附：可复现命令

原始逐案例记录不随仓提交（体积大且完全可再生），因此报告引用的记录目录是运行时目录；用下列命令可从提交的规格文件重建全部数字：

```bash
PYTHONPATH=src python scripts/run_segregation_campaign.py \
    --spec runs/segregation/pilot_specs.json \
    --out .scratch/pilot_out --jobs 13

# 敏感性探测（N=600，0.6 s 时间窗，Γ=3）
python - <<'EOF'
import json
base = dict(drive="vib", large_fraction=0.3, frequency=60.0, gamma=3.0,
            n_particles=600, width=0.4, height=0.18,
            settle_time=0.10, run_time=0.60, sample_every=100)
specs = [dict(base, name=f"tw_R{r}_s0", ratio=r, seed=0)
         for r in (1.0, 1.5, 2.0, 3.0, 4.0)]
json.dump(specs, open(".scratch/probe_specs.json", "w"), indent=1)
EOF
PYTHONPATH=src python scripts/run_segregation_campaign.py \
    --spec .scratch/probe_specs.json --out .scratch/probe_out --jobs 5

PYTHONPATH=src python segregation_validation_report.py \
    --runs .scratch/pilot_out \
    --figures artifacts/segregation_figures \
    --markdown SEGREATION_VALIDATION_SUMMARY.md \
    --json runs/segregation/validation_report.json \
    --probe-runs .scratch/probe_out
```
