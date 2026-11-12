# CASE-DEM-PLANE-SHEAR-MUIDI round 2 — the φ(I) FAIL was a metric defect, not device physics

**锚点**：`CASE-DEM-PLANE-SHEAR-MUIDI`（算例池·平面剪切 mu(I) 流变律标定，status 仍 open）
**任务**：m16427（tp-impl-5 自找活）｜**仓库**：TensorDEM｜**分支** agent/m16427
**接棒对象**：m16080 `/root/tp-agent/work/m16427/docs/CASE_DEM_PLANE_SHEAR_MUIDI.md`（round 1，
honest FAIL：`fail_stage = high_I_mu + phi_flat`）

## ① 案例与方法

装置 = 仓内恒压伺服直剪盒 `tensordem.mu_i_rheology.simulate_servo_shear`（x 周期、y 光滑侧壁、
z 摩擦底板 + 粗糙压板；固结伺服 → 以 `gamma_dot = I·sqrt(P/ρ)/d` 剪切到应变 0.2，后 60% 样本取均值）。
本单**不改物理、不改参数**（round 1 配置逐位复现），只查一件事：

> round 1 判 FAIL 的 "φ(I) 恒定 0.5218" 到底是材料的性质，还是度量口径？

参数表（round 1 同款，`runs/sweep_baseline.json → config`）：P=62.5 kPa，kn=6.25e4 N/m
（P/kn=1），μ_p=0.5，μ_r=0.05，e=0.3，r=5 mm，ρ=2500 kg/m³，盒 0.12×0.05×0.06 m，
N=294，dt=3.2e-5 s，种子 5279，I∈{1e-3,3e-3,1e-2,3e-2,1e-1,3e-1}。

## ② 怎么做的（关键：找到度量缺陷）

读 `simulate_servo_shear` 源码发现：

```python
plate_height = plate_z        # 固结段结束、剪切段开始前的一次快照
...剪切循环中 plate_z 被伺服持续更新，但结果字典只报这个快照...
```

round 1 的床层体积分数 `phi_bed = N·V_p/(W·D·plate_height)` 用的就是这个**剪切前冻结**的床高。
而固结段与剪切率无关（同种子、无剪切）→ 六个 I 的 `plate_height` **必然逐位相同** → φ(I) 恒定
是**构造性恒等式**，不是测量结果。实证：`analysis_r2.json → metric_defect`：

| 量 | 值 |
|---|---|
| `phi_bed_settled` 唯一值个数 | **1**（6 点全 0.5218） |
| `plate_height_settled` 唯一值个数 | **1** |
| `phi_bed_settled` 极差 | **0.0**（精确） |
| `phi_bed_steady` 极差 | 0.0159 |

**修复（最小改动，`src/tensordem/mu_i_rheology.py`）**：剪切段样本已逐帧记录 `plate_z`，
故在结果里补三项尾均值口径，旧键全保留（向后兼容）：

- `plate_height_settled`（= 旧 `plate_height`，显式别名）
- `plate_height_steady` = 剪切段后 60% 样本 `plate_z` 均值
- `phi_bed_steady` = N·V_p/(W·D·plate_height_steady)，`dilatancy` = h_steady/h_settled − 1

测试 `tests/test_servo_shear_steady_state_phi.py`（4 项，16 s）把两半都钉住：旧口径**必须**
与剪切率无关（缺陷留证），新口径**必须**依赖剪切率，且记账恒等式精确成立。

## ③ 计算结果（程序化注入 `runs/sweep_baseline.json` → `analysis_r2.json`，零手抄）

| I 实测 | μ | μ vs ref | φ (旧口径/settled) | φ (新口径/steady) | φ vs ref | dilatancy | 配位数 |
|---|---|---|---|---|---|---|---|
| 1.01e-3 | 0.3295 | −8.9% | 0.5218 | **0.5356** | −7.7% | −2.58% | 4.02 |
| 3.09e-3 | 0.3289 | −9.9% | 0.5218 | **0.5342** | −8.0% | −2.32% | 3.77 |
| 1.09e-2 | 0.3850 | +2.2% | 0.5218 | **0.5287** | −9.2% | −1.31% | 3.65 |
| 3.46e-2 | 0.4074 | −0.5% | 0.5218 | **0.5241** | −10.6% ✗ | −0.44% | 3.50 |
| 1.18e-1 | 0.4302 | −13.2% ✗ | 0.5218 | **0.5231** | −12.4% ✗ | −0.25% | 3.30 |
| 3.28e-1 | 0.4916 | −19.5% ✗ | 0.5218 | **0.5197** | −15.0% ✗ | +0.40% | 3.05 |

μ 六点与 round 1 **逐位相同**（0.3295/0.3289/0.3850/0.4074/0.4302/0.4916）——本单只动记账，未动物理。

**方向诊断（新发现，`direction` 节）**：φ_steady 随 log I 的斜率 = **−6.59e-3 / decade**
（标准误 5.83e-4，≈11σ，即显著下降）；旧口径斜率 −6.3e-17（数值零，再次印证退化）。
参考律隐含方向为**上升**（φ_c=0.58 → φ_j=0.64，即 I↑ φ↑，斜率 ≈ +4.3e-2/decade）。

图：`benchmarks/plane_shear_mu_i/figures/phi_of_I_r2.png`（旧口径 × 与新口径 ■ 同图对照 + 参考律）、
`figures/error_vs_I_r2.png`（μ/φ 误差带 ±10%）。

## ④ 注意点（两个缺陷 + 一个不可辨识）

1. **D1 度量缺陷（已修）**：φ(I) 的 round-1 判据是退化量，其 FAIL 不含材料信息 → round 1 的
   "装置未复现剪切致密化" 结论**撤回**，改为「φ 从未在稳态下被测过」。
2. **D2 参考方向存疑（未修，如实挂账）**：参考 φ_c=0.58→φ_j=0.64（I↑ φ↑）在仓内**无文献锚**
   （grep `brash_ice_rheology.mu_i` / `granular_column.scaling` 只有 μ(I) 三元组
   `LITERATURE_JOP=(0.36,0.84,0.30)`），是 round 1 手抄常数；且与恒压剪切**剪胀**（I↑ φ↓，
   本次实测方向）相反。**本单不改参考值**（无外网、不能在线核验），只在 `analysis_r2.json`
   `ref_constants_in_repo` 里把它标为 PROVENANCE-UNVERIFIED。若参考方向本身有误，则 φ 判据的
   全部 6 点误差符号无意义 → φ 门应判 **WITHHELD**，而不是 FAIL。
3. **φ 水平偏低（−8%~−15%）的候选根因**：床 φ≈0.52–0.536 vs 参考 0.58；P/kn=1 意味着接触很软，
   固结段压缩达 18%（0.060→0.049 m），初始 HCP 格子被打散成松散随机堆积。**未做**刚度档对照
   （`--arm stiff`，kn 10×——本机 load15 ≈110–156/128，只跑了基线臂；该臂一次性失败于
   "servo 未达目标应力"，因 dt 随 kn 缩小 3.2× 而 settle_steps 未相应放大，留作下一轮）。
4. **μ 高 I 段 −13%/−19.5% 未消**（round 1 已归因：e=0.3 偏低 + 盒体有限 + μ_2/I_0 不可辨识）。

## ⑤ 误差判定（逐项，是否 <3%）

| 关键量 | 基准 | 最坏误差 | <3%? |
|---|---|---|---|
| μ(I) 六点 | da Cruz 2005 / GDR MiDi 2004（仓内锚 `LITERATURE_JOP`） | **19.5%**（I=0.33）；4/6 点 <10% | **否** |
| φ(I) 六点（新口径） | 手抄 φ_c/φ_j（PROVENANCE-UNVERIFIED，方向与实测相反） | **15.0%** | **否（且基准不可信 → 建议 WITHHELD）** |
| φ(I) 六点（旧口径） | 同上 | −10.1%~−14.5%，但**量本身退化** | **不适用（判据无效）** |
| 度量缺陷证据 | 旧口径唯一值个数 = 1、极差 = 0.0 | 精确 | 是（结构性证明） |
| 记账恒等式 | 与体积/床高公式逐位对照 | rel 1e-12 | 是（测试断言） |

**结论（honest）**：锚点 `CASE-DEM-PLANE-SHEAR-MUIDI` **不关闭**。本单贡献 = 撤销一条错误归因 +
补上正确的稳态度量 + 把参考侧的出处欠账显式挂账（D2）。`fail_stage` 更新为
`high_I_mu + thin_bed_phi_level + phi_ref_provenance`。

## ⑥ 复现

```bash
PYTHONPATH=src python -m pytest tests/test_servo_shear_steady_state_phi.py -q     # 4 passed, 16 s
PYTHONPATH=src python benchmarks/plane_shear_mu_i/run_parallel.py --arm baseline --jobs 6  # 6 点 ~100 s（6 核）
PYTHONPATH=src python benchmarks/plane_shear_mu_i/analyse_r2.py                    # tables + figures
```

工件：`benchmarks/plane_shear_mu_i/runs/{sweep_baseline.json, analysis_r2.json}`、
`figures/{phi_of_I_r2.png, error_vs_I_r2.png}`。
环境：loongarch64 / numpy 2.x / load15≈110–156（128 核，本单峰值 6 核）。

## ⑦ 给下一轮的线索

1. `--arm stiff`（kn=6.25e5，P/kn=0.1）+ `settle_steps`、`min_shear_steps` 同倍放大 3.2× →
   判定床 φ 偏低是否纯软接触伪影（预期 φ 升向 0.58）。
2. 若参考侧 φ 方向最终核定为**剪胀**（I↑φ↓），把 `run_sweep.PHI_C/PHI_J` 换成有出处的一组常数并
   重判——这是本轮唯一能改变 φ 门结论的一步。
3. 盒体 y=5d、z≈6 层偏薄（m14960 已量 3 层偏低 19%）；8 层对照仍欠账。