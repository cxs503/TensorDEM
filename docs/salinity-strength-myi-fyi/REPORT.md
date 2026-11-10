# P54-W3-N01 — MYI/FYI 盐度-强度分化模型：Cox & Weeks 卤水体积 + Timco–Frederking 孔隙强度

**仓库**：TensorDEM · **模块**：`tensordem.salinity_strength`（新建领域目录）
**节点**：P54-W3-N01 · **分支**：`agent/m9109`
**产物**：`artifacts/salinity_strength/results.json`、`artifacts/salinity_strength/figures/*.png`、
`docs/salinity-strength-myi-fyi/TABLES.md`（程序化生成，零手抄）、`tests/test_salinity_strength.py`（23 例）

---

## ① 案例介绍（物理背景 / 参数）

海水冰是**纯冰基体 + 液态卤水囊**的两相材料。卤水囊是弱相：冰的宏观抗压强度几乎完全由
**卤水体积分数 ν_b** 控制，而 ν_b 取决于**冰温 T**（经海水冰液相线决定卤水盐度 S_b）与
**整体（冰芯实测）盐度 S**。所以"给 T、S 就能给出 σ_c"这条链本身是物理闭合的：

```
T  ──(液相线)──▶  S_b(T)  ──(质量守恒)──▶  ν_b  ──(孔隙率强度律)──▶  σ_c
```

多年冰（MYI）与首年冰（FYI）的差别正是沿这条链的两处：**脱盐**（S 由 5–10 ‰ 降到
0.5–2 ‰，主效应）与**孔结构粗化**（同 ν_b 下强度更高，次效应）。

**模型输入/参数表（逐项钉源，见 `TABLES.md` §1）**

| 符号 | 含义 | 取值 | 出处 |
|---|---|---|---|
| A₁, A₂ | 海水冰液相线 T_f = −(A₁S + A₂S²) | 0.05411, 7.594e−5 | Cox & Weeks (1983) |
| ρ_i | 无气泡纯冰密度 | 0.917 g/cm³ | Weeks (2010) Tab. 1.1 |
| ρ_b | 卤水密度 1 + 0.8·S_b/1000 | — | Weeks (2010) eq. 2.9 |
| σ₀ | 零孔隙率柱状冰强度（−10 °C, 1e−3 s⁻¹） | 5.5 MPa (5–8) | Timco & Weeks (2010) §5 |
| ν_p | 强度归零的卤水体积分数 | 0.27 (0.25–0.30) | Timco & Weeks (2010) Fig. 9 |
| m | 孔隙率强度律指数 | 1.5 (1.0–2.0) | Timco & Frederking (1984) 数据曲率 |
| κ_MY | MYI 孔结构增强因子 | 1.15 (1.10–1.20) | **工程估计**（非拟合常数，见 ⑥） |
| S_FY / S_MY | 典型整体盐度 | 5 / 1 ‰ (3–10 / 0.5–2) | Weeks (2010) Ch. 2 |

## ② 怎么做的（方法 / 流程 / 工具链）

1. **卤水盐度**（`brine_salinity`）：把 Cox & Weeks (1983) 海水冰点多项式
   T_f = −(0.05411·S + 7.594e−5·S²) 对 S 解析反解（正根），T ≥ 0 返回 0（无冰相）。
   该函数在 T < 0 上严格单调递增（越冷卤水越咸），这是后续两条单调性判据的物理来源。
2. **卤水体积分数**（`brine_volume_fraction`）：Cox & Weeks 质量守恒
   ν_b = (ρ_i/ρ_b)·S/S_b(T)；卤水密度 ρ_b = 1 + 0.8·S_b/1000。S = 0 精确给 0；T ≥ 0 给 1（全液态）。
3. **强度**（`compressive_strength`）：Timco & Frederking (1984) 孔隙率律
   σ_c = κ · σ₀ · (1 − ν_b/ν_p)^m，ν_b ≥ ν_p 时截断为 0（不出现负强度）。
   κ = κ_MY（MYI）或 1（FYI）。
4. **不确定性**（`monte_carlo_strength`）：对 σ₀、ν_p、m、κ_MY 在**文献区间上取均匀先验**
   （区间是"界"不是标准差，用正态先验会谎报信息量），叠加冰芯盐度测量误差
   N(S, 0.5 ‰)，20 000 次抽样，报 p05/p50/p95。固定 seed=20261110，结果位级可复现（有测试）。
5. **出口**：`scripts/run_salinity_strength.py` 计算 17×16 强度曲面、温度/盐度扫描、
   MC 置信带、文献对照表 → `results.json` + 5 张图；
   `scripts/emit_salinity_tables.py` 从 JSON 程序化生成 `TABLES.md`（**零手抄**）。
6. **工具链**：numpy 2.4.6 / matplotlib (Agg) / pytest 8；Python 3.11.6，loongarch64。

## ③ 计算结果（数字均引自 `TABLES.md`，程序注入）

**判据汇总**

| 判据 | 模型值 | 基准 | 偏差 | 阈值 | 判定 |
|---|---|---|---|---|---|
| (a) ν_b(T=−10 °C, S=5 ‰) | 0.02684 | 0.0270（Timco & O'Brien 1994 诺模图） | **0.59 %** | < 2 % | **PASS** |
| (b) σ_c FYI(−10 °C, 5 ‰) | 4.701 MPa | 4.40 MPa（Timco & Frederking 1984） | **6.83 %** | < 10 % | **PASS** |
| (c) MYI/FYI 序 | σ_MY 5.406 MPa vs σ_FY 4.701 MPa，比 1.150 | 路径(3) 要求 MY > FY | — | — | **CONFLICT**（见 ⑤） |
| (d) T: −40 → 0 °C，17 档 | Spearman ρ = **−1.0**，严格单调下降 | \|ρ\|=1，≥10 档 | — | — | **PASS** |
| (e) S: 0 → 15 ‰，16 档 | Spearman ρ = **−1.0**，严格单调下降 | \|ρ\|=1，≥10 档 | — | — | **PASS** |
| (f) 负控 | σ(S=0)=5.500 MPa=σ₀；ν_b(S=0)=0；σ(T=0)=0 | 0 / 0 / 0 | 精确 | — | **PASS** |
| (g) 工程交付 | 23 例 pytest 全绿 + results.json + 5 图 + 本报告 | ≥15 例 | — | — | **PASS** |

**强度随温度（FYI, S=5 ‰）**：−40 °C 5.274 MPa → −20 °C 5.076 → −10 °C 4.701 → −5 °C 3.992 → 0 °C 0。
**强度随盐度（FYI, T=−10 °C）**：0 ‰ 5.500 MPa → 5 ‰ 4.701 → 10 ‰ 3.944 → 15 ‰ 3.233 MPa。
**MYI 典型（−10 °C, S=1 ‰, κ=1.15）**：σ_c 由 FYI 的 4.701 提升到 **6.137 MPa**（比 1.306）；
**同 (T,S) 等盐度下** κ 单独贡献 1.15（分解：脱盐 ×1.1353，孔结构 ×1.1500，见 `TABLES.md` §2b）。

## ④ 结果图（`artifacts/salinity_strength/figures/`）

| 图 | 内容 | 说明 |
|---|---|---|
| `sigma_surface_fy.png` | σ_c(T,S) 三维曲面 | 全域单调：冷/淡最强，暖/咸趋零 |
| `myi_vs_fyi.png` | MYI 典型 / MYI 等盐度 / FYI 三条曲线 | 分离"脱盐"(主)与"孔结构"(次)两种贡献 |
| `brine_volume.png` | ν_b(T) 半对数，多盐度 + ν_p 极限线 | 解释强度悬崖的来源 |
| `mc_band.png` | MC 5–95 % 置信带 | 暖端带变宽（逼近强度悬崖，非线性放大） |
| `literature_comparison.png` | 模型点 vs T&F 锚点 + FYI 现场 1–3 MPa 带 | 见 ⑥"是否 <3%" |

每张图 sha256 记入 `results.json["figures"]`（provenance 可审计）。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **判据 (c) 是自相矛盾的（如实记录，不粉饰）。** 节点元数据**路径 (3)** 要求
   "σ_c_MY > σ_c_FY 在相同 T,S 条件下"，而**判据 (c)** 要求 "σ_c_MY < σ_c_FY（物理一致性）"，
   二者不可能同时成立。物理上（Timco & Weeks 2010：MY 冰 σ_c 5–10 MPa vs FY 1–3 MPa）
   与路径 (3) 一致：MY 冰更强。本实现按路径 (3)+文献走，判据 (c) 按字面读法记
   **CONFLICT_NOT_MET**，并用 `test_criterion_c_conflict_memorandum` 把冲突钉住，
   防止后人"顺手改绿"。
2. **κ_MY = 1.15 是工程估计，不是拟合常数。** 已明确标注来源与区间 1.10–1.20，
   并进 MC 先验；若日后拿到同 ν_b 下 MY/FY 强度实测对，应重标此参数。
3. **液相线在 T < −23 °C 高估卤水盐度**：真实液相线在 Na₂SO₄·10H₂O（−8 °C）、
   NaCl·2H₂O（−22.9 °C）析盐线处折点。表 3 中 S_b(−40 °C)=452 ‰ 明显偏高，
   故 −23 °C 以下的强度偏保守（高估）。判据 (d) 只用单调性，不受影响。
4. **无应变率依赖**：T&F 数据对应 1e−3 s⁻¹ 的准静态高应变率端；韧性区强度低约 2 倍。
   模型是"强度上包络"，用于船-冰挤压/平整冰阻力上限，不可直接用于蠕变问题。
5. **无各向异性**（柱状冰横柱方向更弱）、**无气泡孔隙**（MYI 实际密度 0.87–0.90，与脱盐增强反向）。
6. 判据 (a) 的基准是诺模图读数（±0.001 绝对，约 4 %），所以"<2 %"实际部分落在读数精度内——
   已在 `results.json` 的 note 中声明，不宣称这是盲验证。

## ⑥ 误差判定（逐项回答"是否 <3%"）

| 关键量 | 基准 | 相对误差 | 是否 < 3 % |
|---|---|---|---|
| ν_b(T=−10 °C, S=5 ‰) | 0.0270（T&O 1994 诺模图读数） | 0.59 % | **是**（但基准读数不确定度 ~4 %，见 ⑤-6） |
| σ_c FYI(T=−10 °C, S=5 ‰) | 4.40 MPa（T&F 1984 编译值） | 6.83 % | **否**；<10 % 达标。差距来源：σ₀ 取文献区间下端 5.5 MPa 且 m=1.5，未对锚点做拟合 |
| σ_c FYI 落在现场区间 1–3 MPa | Timco & Weeks (2010) | 模型 4.70 MPa **高于**该带 | **否**；模型是实验室高应变率强度，现场大尺度试件含缺陷更弱——这属于适用范围差异，不是数值误差 |
| 单调性 (d)(e) | Spearman \|ρ\|=1 | 0 | **是**（精确，解析性质） |
| MC 位级可复现 | 同 seed 两次调用 | 0（dict 全等） | **是** |

**结论**：可机器裁的三条精度判据 (a)(b) 均达标；对 (b) 与现场强度带 6.8 %/超出带的偏差
如实列出，未做参数回拟（避免"调参到靶"）。σ₀、ν_p、m、κ 全部保留文献区间并进入 MC 传播，
因此 `mc_band.png` 的 5–95 % 带（−40 °C 半宽 ±21 %，−2 °C 半宽 ±44 %）才是诚实的预测不确定度，
中心值不应被当成 3 % 精度的结论。

## 参考文献

1. Cox, G.F.N. & Weeks, W.F. (1983). *Equations for determining the gas and brine volumes in
   sea-ice samples.* J. Glaciology 29(102), 306–316.
2. Timco, G.W. & Frederking, R.M.W. (1984). *The uniaxial compressive strength of sea ice.*
   Proc. IAHR Ice Symposium, Vol. 2, 251–262.
3. Timco, G.W. & O'Brien, S. (1994). *A practical guide for using the brine volume of sea ice
   to predict ice strength.* NRC Canada, Report 94-11.
4. Timco, G.W. & Weeks, W.F. (2010). *A review of the engineering properties of sea ice.*
   Cold Regions Science and Technology 60(2), 107–129.
5. Weeks, W.F. (2010). *On Sea Ice.* University of Alaska Press.
6. Frankenstein, G. & Garner, R. (1967). *Equations for determining the brine volume of sea ice
   from −0.5 to −22.9 °C.* J. Glaciology 6(48), 943–944.