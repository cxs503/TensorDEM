# 直剪盒宏观摩擦测量：死代码修复 + 诚实 FAIL（m13798，锚点 GAP-DEM-FRICTION）

**结论先行**：`tensordem.shear_cell` 此前是**从未执行过的死代码**（第一步就 `TypeError`），
其"摩擦角"拟合在方法上是**病态**的（单次试验内 σ_n 近似常数却去拟合斜率，且用 `Σ|f_z|`
冒充法向应力）。本轮把它修成可运行的三维直剪装置（法向应力伺服压板 + 逐级 τ/σ + 跨级
Mohr-Coulomb 包络 + 有效性门），并**用测试锁住拟合器代数与"拒绝单级拟合"的结构修复**。
**但物理稳态未达成**：压板在床面持续压密/塌落时失接触，σ_n 归零 ⇒ μ 无定义。按纪律记为
**honest FAIL**，有效性门会拒绝出数，而不是给出 NaN/0 的"摩擦角"。
锚点 `GAP-DEM-FRICTION`（"摩擦系数 vs 实验 <10%"）**仍未关闭**，本单只消除了它下面的一颗地雷。

---

## 1. 案例与装置

| 项 | 值 | 说明 |
|---|---|---|
| 颗粒 | r = 5 mm，ρ = 2500 kg/m³ | 单分散球，HCP 初始装填 |
| 盒体 | 0.080 × 0.032 × 0.050 m（N ≈ 78） | 侧壁无摩擦（n法向弹簧） |
| 底板 / 压板 | 粗糙，μ = μ_s | Cundall–Strack 切向弹簧 + 滚动弹簧 |
| 接触 | kn = 1e5 N/m，kt = 0.5 kn，e = 0.3 | dt = 2π√(m/2kn)/20 ≈ 2.5e-5 s |
| 协议 | 重力沉降 → 应力伺服固结 → 压板横移剪切 | 切变率 γ̇ = v/H = 1.0 s⁻¹，I ≈ 0.018 |
| 测量 | τ = −Σf_t,x/A（压板），σ_n = Σf_n/A | 稳态窗后半段 + 分块标准误 |

## 2. 修复了什么（对照修复前）

| # | 缺陷（修复前） | 证据 | 修复 |
|---|---|---|---|
| D1 | 装置**从未跑通**：`acc = force / mass[:, None]`，`mass` 是 float | `TypeError: 'float' object is not subscriptable` @ `shear_cell.py:255`（复现于 m13798 首跑） | 标量质量/惯量 + 通用 `_plane_contact()` 平面壁接触 |
| D2 | **病态拟合**：单次试验内对 σ_n 拟合 τ = c + σ tanφ；σ_n 只有噪声级变化 | 代码审查：`lstsq([ss_sigma, 1], ss_tau)` 在同一 σ 水平内 | 每级只给点值 μ = τ/σ；包络只在**不同** σ 级之间拟合，单级/重复级**直接拒绝** |
| D3 | 法向应力用 `Σ|f_z|` 冒充 | `normal_force_measured = Σ abs(force[top,2]) / n_top` | σ_n 取压板法向反力 Σf_n/A |
| D4 | 无稳压：静载压板在床面膨胀时脱开（下游 upscaling 直剪同类失效已记录为 NaN） | `docs/coarse_graining_upscaling_laws.md` §4.2"死载压板脱开，σ_n=0 ⇒ μ 无定义" | 两段式伺服压板（脱离期快速逼近 + 近目标力控）+ **有效性门** |

## 3. 已验证（可机器复现）

`PYTHONPATH=src python -m pytest tests/test_shear_cell_friction.py -q` → **7 passed（7.4 s）**

| 判据 | 阈值 | 结果 |
|---|---|---|
| 包络拟合器在合成直线 τ=0.5σ 上精确 | tanφ 误差 < 1e-12 | PASS（实测 < 1e-12） |
| 含黏聚合成线 (c=100 Pa, tanφ=0.3) 反演 | c 误差 < 1e-6 Pa | PASS |
| 单级拟合被拒绝（D2 的结构修复） | 抛 `ValueError` | PASS |
| 重复 σ 级被拒绝 | 抛 `ValueError` | PASS |
| 装置端到端可运行（D1 回归） | 返回完整记录、有限步数 | PASS（此前必崩） |
| 有效性门自洽（`invalid_reason=="" ⇔ valid`） | 恒成立 | PASS |

## 4. honest FAIL：物理稳态未达成

一次 400/1600 Pa 双级运行（2154 步 + 沉降/固结，settle 0.02 s，固结 0.03 s，剪切 0.02 s）：

| 观测量 | 数值 | 判读 |
|---|---|---|
| 压板 z | 0.05184 → 0.04764 m | 伺服在跟 |
| 床面最高点 z_top | 0.04674 → 0.03848 m | **比压板多掉 4.1 mm ≈ 0.8 r** |
| 最大瞬时压板载荷 | 2.09 N（目标 2.048 N，约第 790 步） | 伺服**曾**到位 |
| 剪切窗内 σ_n 均值 | **0.0 Pa** | 压板已失接触 |
| μ = τ/σ | **无定义（NaN→门拒绝）** | 不做假数 |

**未判定的三个候选成因**（本轮未做判别实验，按纪律不选边）：
1. 底板阻尼用单颗粒 ζ 公式 `γ_w = 2ζ√(kn m)`，在床层等效质量下过阻尼/欠阻尼，导致床层整体沉降失控；
2. 初始 HCP（jitter=0.01）本身是膨胀态，塌落速率（自由落体量级 0.14–0.4 m/s）远超伺服逼近速度上限（0.05 m/s）；
3. 四个侧壁共用一个 `_WallState`，切向状态键只按颗粒编号（本轮 μ_wall=0 时无影响，但属结构隐患）。

**下一步（建议，不属本单范围）**：宏观摩擦包络改走**已验证的周期 Lees–Edwards 直剪**
（`tensordem.mts_contact.shear_cell_run`，其两臂压力一致性 ≤2% 已有测试），或先修底板阻尼/床层
初始致密度再回本装置。修好后本模块的 `measure_macro_friction()` 与有效性门可原样使用。

## 5. 下游影响（审计提示）

`scripts/run_coarse_graining_campaign.py:233-260` 直接调用 `simulate_shear_cell` 并读
`result['friction_angle_deg']` / `['r_squared']`。该路径**在本单之前必然崩**（D1），
故凡以该脚本 shear 支路为依据的"摩擦角"数字**都缺证据**，应回溯核查（本单未改该脚本；
`friction_angle_deg` 现在返回的是点值 atan(τ/σ)，仅在 `valid=True` 时有意义）。

## 6. 锚点状态与误差判定

| 项 | 判定 |
|---|---|
| `GAP-DEM-FRICTION`（摩擦系数 vs 实验 <10%） | **未关闭（in_progress）**：本轮未取得任何可比实验的摩擦系数 |
| 是否 <3% | **不适用/未达成**：无有效测量，不做粉饰 |
| 本单可复现产物 | 代码 `src/tensordem/shear_cell.py`；测试 `tests/test_shear_cell_friction.py`；诊断 `artifacts/shear_cell_friction/shear_cell_diagnosis.json` |

## 7. 复现

```bash
PYTHONPATH=src python -m pytest tests/test_shear_cell_friction.py -q
PYTHONPATH=src python scripts/diagnose_shear_cell_m13798.py   # 重生成诊断 JSON
```
