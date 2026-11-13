# XVAL-DEM-BIAXIAL-CELL — R2 验证欠账清零：应力定义差的定量归因

**锚点** `XVAL-DEM-BIAXIAL-CELL-NEGATIVE-RESULT`（TensorDEM）。
R1（`docs/XVAL_DEM_BIAXIAL_CELL_NEGATIVE_RESULT.md`）把轴向 σ_v 的墙-vs-内部差
（+24 %）记为 **WITHHELD（未定位）**，并列出三个候选原因：
(a) 只取**底板**反力、顶端驱动板反力未进 σ_v；(b) 边界半层分支向量缺失（内部 virial 漏墙项）；
(c) 峰值时刻非平衡。本单**把三者分开、逐个量化**，并补回 R1 缺失的可复跑工件。

**一句话结论（honest, mixed）。** 24 % 的轴向差不含物理错误，是**测量定义**造成的：
其中 **+35.7 %** 来自"只读一块板"（(a)，驱动板比固定板多承担 298 kPa，因为侧墙摩擦把轴向荷载
分走了 183 kPa、外体力（全局阻尼）再分走 115 kPa）、**−1.2 %** 来自内部 virial 漏墙项（(b)，量化后
很小，**假设被否定**）、**−10.4 %** 是闭合残差（(c)，含外体力 virial 未计入）；
三项相加 = 24.0 % 精确对上旧缺口。**没有任何物理量对实验锚达到 3 %**，本单只宣称"缺口被归因"。

## ① 案例介绍（几何 / 参数 / 基准）

平应变双轴剪切盒，24×48 抖动三角点阵，粒径 r=0.75 mm，试样 35.7×62.6 mm，厚度 1，
冻结键 2084 条（峰值前已全部破断），四壁 servo 围压 100 kPa，轴向应变率 0.05 s⁻¹，
时间步 dt=4.97×10⁻⁴ s，加载至 ε_a=0.09（3795 步）。两臂**只改侧墙摩擦**：
`rough`（μ=0.55，材料值）与 `smooth`（μ=0，三轴惯例）。基准锚：
密实 Hostun 砂剪切带 60°（Desrues & Viggiani 2004，文献值 55–60°）。

## ② 怎么做的（方法 / 工具链）

1. **补回边界数据**：`benchmarks/biaxial_cell_xval/run_xval.py` 的 `case` 子命令逐步记录
   四壁**法向与切向 traction**（按各自的向内法向统一成"压缩为正"）、顶端板反力、
   `(virial + virial_wall)` 全 virial、外体力（`body_force` + 全局速度阻尼 `-drag·v`）及其 virial、
   竖向力平衡残差、颗粒平均速度。
2. **生产内核只加不改**：`src/tensordem/ridge_dem.py` 新增 `virial_wall`（墙接触
   `f ⊗ (x_contact - x_center)`），与原有 `virial`（仅内部接触）**并列返回**，默认路径与历史结果不变。
3. **可加分解**（`decompose_stress_definitions`，基准内，不进生产 API）：
   `gap_old = (I−B)/B` 被拆成
   `part_a = (M−B)/B`（漏驱动板）、`part_b = (I−F)/B`（漏墙 virial）、
   `residual = (F−M)/B`（散度定理闭合残差，准静态下必须趋零），三项**恒等相加**于 `gap_old`。
   B=底板读数，M=两块板均值，I=内部 virial，F=全 virial。
4. **判据先于动手**：沿用 `XVAL_TOLERANCE = 0.15`；闭合要求 `|residual| ≤ 15 %`，
   严格报告里另外回答"是否 <3 %"。

## ③ 计算结果（全部由工件程序化产出，零手抄）

### 3.1 复现 R1（同一配置、同一步长、同一峰值规则）

| 量 | R1 记录 | 本单重跑 | 差 |
|---|---|---|---|
| σ_v 墙（底板，峰值 kPa） | 418 | 418.2 | 0.05 % |
| σ_v 内部（virial，kPa） | 519 | 518.6 | 0.08 % |
| σ_h 墙（kPa） | 114 | 113.9 | 0.09 % |
| σ_h 内部（kPa） | 206 | 205.6 | 0.19 % |
| φ 墙 / φ 内部 | 34.87° / 25.61° | 34.87° / 25.61° | 0 % |
| 轴向相对差 | +24.0 % | +24.02 % | 0.02 pp |
| 侧向相对差 | +80.4 % | +80.43 % | 0.03 pp |

→ R1 的负结果**可复现**（同峰值步 ε_a=0.0851）。

### 3.2 峰值步的应力定义分解（`case_p100_*_analysis.json: stress_definition`）

| 臂 | σ_v 底板 | σ_v 顶板 | 两板均值 M | 内部 I | 全 virial F | 旧缺口 |
|---|---|---|---|---|---|---|
| rough (μ=0.55) | 418.2 | 716.3 | 567.2 | 518.6 | 523.6 | **+24.02 %** |
| smooth (μ=0) | 516.3 | 585.9 | 551.1 | 514.1 | 515.2 | **−0.44 %** |

| 臂 | part_a 漏驱动板 | part_b 漏墙 virial | residual 闭合 | 三项和 | ‖residual‖≤15 % |
|---|---|---|---|---|---|
| rough | **+35.65 %** | −1.19 % | −10.44 % | +24.02 % ✓ | ✓（但不为 0） |
| smooth | +6.74 % | −0.21 % | −6.97 % | −0.44 % ✓ | ✓ |

**读法**
- **(b) 被否定（定量）**：墙 virial 项只有 0.2–1.2 %，与解析预估
  `2(r−overlap)/H ≈ 2×0.75/62.6 = 2.4 %` 同量级 —— 边界半层分支向量**解释不了**几十个百分点。
  单粒子/三粒链的精确恒等式测试（`tests/test_biaxial_stress_definition.py`）证明：
  全 virial 与板反力在静定链上**逐位相等（1e-9）**，内部 virial 的偏差恰为边界层占比 `2(r−p)/H`。
- **(a) 被确认并定量**：驱动板承 716 kPa、固定板只承 418 kPa，差 298 kPa。
  竖直平衡 `dσ_yy/dy = ρg + 2τ/W` 由实测 τ=62.7 kPa 预测 **183 kPa**；
  余下 115 kPa 由**外体力**（全局速度阻尼 `-drag·v`，其合力 ≈ 边界荷载的 22 %）承担。
  无摩擦臂（μ=0）该差缩到 69.6 kPa、旧缺口 −0.44 % —— **一个开关同时关掉轴向差和侧向差**，
  与 R1 的 A/B 结论一致。
- **(c) 闭合残差**：带上外体力后**竖向力平衡残差在峰值为 6.3 %（rough）/ 3.0 %（smooth）**；
  若不计外体力则升到 ~28 %。但全 virial 与两板均值的**应力恒等式**仍差 −10.4 %：
  外体力的 virial 一阶矩只补回 5.0 kPa（523.6 vs 518.6），故 −10.4 % 归为
  "非准静态/单元体不封闭"残留 → 记为 **WITHHELD**，不粉饰。

### 3.3 侧向与 φ

侧向旧缺口 +80.4 %（rough）在加入墙 virial 后变为 **+92.7 %**（更差，即墙 virial 不是侧向的原因）；
无摩擦臂侧向缺口降到 **+12.4 % ✓（<15 %）**，**复现 R1 的 12.4 %**。
φ（Mohr–Coulomb）：底板定义 34.87° → 全 virial 定义 24.15°（−44 %），
**并未因定义修正而收敛** → φ 的偏差仍是**侧向**应力定义/侧墙摩擦问题，不是轴向问题（honest）。

## ④ 结果图

`artifacts/biaxial_cell_xval/stress_definition.png`（`benchmarks/biaxial_cell_xval/make_stress_figure.py`）
- (a) 三条 σ_v–ε_a 曲线：底板/顶板/两板均值 + 内部 virial 与全 virial。两板从 μ 生效起就劈开，
  内部 virial 贴着两板均值走（证明 virial 本身没坏），底板读数系统偏低；
- (b) 缺口分解条形图（两臂 × 四项，含 ±15 % 判据线）；
- (c) 板间落差：实测 298 kPa vs 侧墙剪切预测 183 kPa（差值＝外体力项）。

## ⑤ 注意点 / 局限

1. **生产代码只做加法**：`ridge_dem` 只多返回 `virial_wall`，`biaxial_shear_band` 未改，
   历史结果与其默认行为不变（R1 的 418/519 复现即证据）。
2. **单位制**：本仓 DEM 的长度/力单位是 mm 量级（width0=35.7、velocity≈2.9 "m/s" 实为 mm/s），
   因此"外体力合力 ≈ 3.5e6（内部单位）vs 边界荷载 1.6e7"必须按同一单位制读，**不可跨单位比较**。
3. **residual −10.4 % 未定位**（WITHHELD）：候选为非准静态（颗粒平均速度 1.82 内部单位 ≈ 板速的 64 %，
   说明试样在加载中确有流动）、外体力 virial 的分布形式、以及单元体未封闭（颗粒被压出测量窗外）。
4. **侧向仍不闭合**：墙 virial 使侧向更差（+92.7 %），真正原因是侧墙摩擦（无摩擦臂 12.4 % 已证），
   与 R1 相同；**本单不宣称 φ 被修正**。
5. **p=200 kPa 臂未跑**、峰值判据仍按 `≥3 快照/散度<10°` 判 unresolved（本单重跑尾部仅 1–2 快照），
   故**不改判** R1/R2 的带倾角结论。
6. 报告里的全部数字来自 `artifacts/biaxial_cell_xval/*_analysis.json`，
   由 `run_xval.py analyze` 从 `.npz` 计算生成（零手抄）；`.npz` 与本文件同批入库。

## ⑥ 误差判定（逐项回答）

| 关键量 | 基准 | 误差 | <3 % ? | <15 % ? |
|---|---|---|---|---|
| 全 virial ≡ 板反力（静定三粒链） | 解析恒等式 | 1e-9 | **是** | 是 |
| 内部 virial 偏差 = 边界层占比 `2(r−p)/H` | 解析 | <1e-6 相对 | **是** | 是 |
| 旧缺口三项可加性 `gap_old = a+b+residual` | 恒等式 | 1e-12 | **是** | 是 |
| rough 轴向闭合残差 | 两板均值（散度定理） | 10.4 % | 否 | **是** |
| smooth 轴向闭合残差 | 同上 | 7.0 % | 否 | **是** |
| rough 侧向闭合残差 | 侧墙法向均值 | 92.7 % | 否 | **否**（WITHHELD） |
| smooth 侧向缺口（=R1 的 12.4 %） | 侧墙法向均值 | 12.4 % | 否 | **是** |
| 板间落差（侧墙剪切预测） | 实测 | 62.8 %（余项=外体力） | 否 | 否（已归因） |
| 带倾角 vs 60° 锚 | Desrues & Viggiani 2004 | 本单不重判（unresolved） | — | — |

**是否 <3 %？** 只有**恒等式类**（全 virial≡板反力、边界层解析式、三项可加性）达到 3 %；
**没有任何物理量对实验锚 <3 %**。本单宣称的是"**一个未定位的测量缺口被拆成可复算的三项**"，
不含任何精度提升声明。

## 复现

    PYTHONPATH=src python benchmarks/biaxial_cell_xval/run_xval.py case --strain-target 0.09 \
        --out artifacts/biaxial_cell_xval/case_p100_rough.npz
    PYTHONPATH=src python benchmarks/biaxial_cell_xval/run_xval.py case --strain-target 0.09 \
        --wall-friction 0.0 --out artifacts/biaxial_cell_xval/case_p100_smooth.npz
    PYTHONPATH=src python benchmarks/biaxial_cell_xval/run_xval.py analyze \
        --inp artifacts/biaxial_cell_xval/case_p100_rough.npz \
        --out artifacts/biaxial_cell_xval/case_p100_rough_analysis.json
    PYTHONPATH=src python benchmarks/biaxial_cell_xval/make_stress_figure.py
    PYTHONPATH=src python -m pytest tests/test_biaxial_stress_definition.py \
        tests/test_biaxial_xval.py tests/test_biaxial_shear_band.py -q   # 29 passed

单次 case ≈ 78 s（torch 单线程，1128 颗粒，CPU load15≈100 时仍按容量纪律只跑单线程短作业）。