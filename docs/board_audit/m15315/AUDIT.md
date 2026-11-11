# 交差审计 · m15279 / CASE-BIAXIAL-SHEAR-BAND（TensorDEM）

审计单：m15315（任务榜交差核验单，非重做）· 审计人：board-audit · 日期：2026-11-11

被审交付：mission m15279，commit `5060946e`（branch `agent/m15279`，**非 main 祖谱**），
`src/tensordem/biaxial_shear_band.py`(505 行) + `benchmarks/biaxial_shear_band/{run_benchmark,make_figure}.py`
+ `tests/test_biaxial_shear_band.py`(115 行) + `docs/BIAXIAL_SHEAR_BAND.md` + `artifacts/biaxial_shear_band/*`。

## 0. 结论（disposition）

| 项 | 判定 |
|---|---|
| 三件套（run_artifacts / ref-vs-got / repro_cmd） | **真实存在且可跑**（见 §2），无造假迹象 |
| 报告六段 + 显式误差判定 | 齐全，且 self-declared honest FAIL，**不粉饰** |
| 报告的核心科学结论（44.25°/44.67°，误差 26%，材料性质） | **被本审计证伪**：误差是测量算子缺陷，不是材料响应 |
| 等级 | **L2**（独立功能实现）；L3 不成立（未接入、未导出于包、未合入 main）；L4 不成立（基准判据按 invalid 处理，需复测） |
| 处置 | **request changes → reopened**：修 `radon_band_angle` 各向异性归一化（单行级），重跑 4 配置再判 L4 |

## 1. 取证：三件套逐项核验

| 检查 | 证据 | 结果 |
|---|---|---|
| run_artifacts | `artifacts/biaxial_shear_band/results.json`(1656 行, 4 配置全记录)、`final_field.npz`(893 粒子 x/y/gamma/angle_deg/strain)、`shear_band.png`(1610×728 PNG)、`logs/biaxial_shear_band_bench.log`；均在 commit `5060946e` 中被 git 跟踪（`git ls-tree` 已核） | ✅ 真实 |
| ref vs got | ref θ=60°，tol 0.15（接受区间 51–69°）；got：r24x48_p100 **44.25°**(rel 0.2625)、r24x48_p200 **44.67°**(rel 0.2556)、r16x32×2 unresolved(NaN, 如实标 resolved=false) | ✅ 数字与台账一致；判据 **FAIL** |
| repro_cmd | 本审计实跑 `PYTHONPATH=src python benchmarks/biaxial_shear_band/run_benchmark.py --quick --workers 1` → 正常出 JSON（`all_passed:false`，rc=1 为设计语义）；全量配置本地需 ~127 s/配置（本审计 shell 120 s 硬限被杀，已改走集群 cj1060，排队中未取回 = not_run）；确定性：同一配置两次调用应力路径与 band history **逐位相同** | ✅ 可执行 / 可确定性复现；全量独立复跑 not_run |
| tests | 本审计实跑 `pytest tests/test_biaxial_shear_band.py -q` → **15 passed in 14.03 s** | ✅ 通过（但覆盖有盲区，见 F2） |
| benchmark 表 | `SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BIAXIAL-SHEAR-BAND'` → **0 行**（全表 `ILIKE '%shear%'` 亦 0 行） | ⚠️ 机器裁判**未登记**，判据只存在于仓内 docstring |
| 报告 | report_id=14385，六段齐全（①问题②方法③参数表④对照表⑤敏感性⑥失败模式），并显式回答「是否<3%」→ 26.3%/25.6%，不<3% 也不<15% | ✅ 齐全 |
| integration_callers | `grep -rn biaxial` 除本单 5 个文件外**零引用**；`src/tensordem/__init__.py` 未导出；commit `5060946e` 不是 main(HEAD) 祖先 | ❌ 无集成 |

## 2. 判级依据（L0–L5）

- **L2 = 独立功能实现**：真代码输入→计算→输出。`run_biaxial_shear_band` 用 `ridge_dem.RidgeAssembly/Plane`
  真跑四壁伺服双轴 DEM（1128 粒子、4839 步、wall 110–127 s），测 φ/ψ、Radon 倾角、理论对照 —— 不是 MOCK/STUB。
- **L3 不成立**：未进 `__init__`、无 CLI/API、无包内调用者，且**交付 commit 尚未合入 main**（能力在主线不存在）。
- **L4 不成立**：完整算例+误差分析+可复现是有的，但 **got 数值本身被本审计判为无效**（见 §3 F1），基准判据不能按 26% 失败结案。

## 3. 发现（按严重度）

### F1 [blocking · 已复现] 倾角算子存在各向异性归一化缺陷，把 60° 读成 44–51°

`radon_band_angle` 把 x、y **各自**按自身极差归一化后再投影：
`xs=(xs-xs.min())/ptp(xs)`、`ys=(ys-ys.min())/ptp(ys)`（`src/tensordem/biaxial_shear_band.py:180-181`）。
对宽高比 A=h/w 的窗口，这等价于对试样做了各向异性缩放，真实倾角 θ 被报成
**arctan(tanθ / A)**。双轴试样正是竖长窗口（24×48 → A=1.73；实测窗口 A=1.404）：
θ=60° → 45.0°/51.0°，与 m15279 报的 44.25°/44.67° 量级一致。

复现（`audit_biaxial_estimator.py`，写于运行前的判据 P1/P2/P3）：

| 探针 | 结果 |
|---|---|
| P1 已知倾角合成脊（24×48 晶格，与试样同构） | library 算子偏差 **−12.0 … −16.0°**（60°→44.5°）；只把归一化改成单一各向同性尺度 → 偏差 **≤0.5°** |
| P1 窗口宽高比扫描 | library 偏差随 A 变号：A=1.73 → −15°、A=0.87 → +4°、A=0.43 → +22°（系统性、可预测） |
| P2 根因隔离 | 预测式 `arctan(tanθ/A)` 与 library 输出**最大残差 0.5°**；仅改归一化一行即消除 → 根因=各向异性归一化，**与 m15279 报告归咎的「全局仿射剥离」和「35% 键合比例」无关**（纯权重脊、无位移场时偏差照样出现） |
| P3 用 m15279 自己的存储场重测 `final_field.npz`（893 粒子，A=1.404） | library=51.0°；**各向同性算子=60.0°**（η²=0.57, z=10.6）；相对 60° 锚点误差 **0.0%**，落在 51–69° 接受区间内 |

即：**同一份 DEM 场，算子修好后就是 60.0°**。m15279 的「honest FAIL 26%」是测量伪影，
其失败机理分析（仿射剥离/键合比例）与证据不符 —— 这部分是误判，但属诚实误判（未粉饰数字）。

图：`docs/board_audit/m15315/m15315_estimator_bias.png`（(a) 24×48 窗口回收曲线；(b) `arctan(tanθ/A)` 根因拟合与两个真实窗口；(c) 同一存储场两算子对比 + 接受区间）。

### F2 [major] 合成验证的几何盲区让缺陷逃过 15 条测试

`tests/test_biaxial_shear_band.py::synthetic_band` 用 **30×30 各向同性随机云**（A=1.0），
该几何下 A=1 时 `arctan(tanθ/1)=θ`，缺陷自然不可见；且测试只直接喂权重到 `radon_band_angle`，
不走 `local_shear_strain`（真实链路）。→ 测试断言的是「算子能测合成脊」，而非「在试样窗口几何下无偏」。

### F3 [minor] 报告与工件的数字不一致

报告 ③ 写「24×48（**1104** 粒子）」，`results.json` 实为 **1128**；servo 表格只写了 +22%/+16%，
`r24x48_p200` 的 `relative_offset=0.164` 与文末 +16% 对应，属一致但易误读。

### F4 [minor] 判据未登记到机器裁判

`tensor_benchmark_cases` 无本锚点行 → gate 无法自动复核 got。判据仅存于 `run_benchmark.py` docstring（预登记，可接受），建议补登记（ref_source=Desrues & Viggiani 2004，metric=theta_deg，tolerance=±15%）。

### F5 [note] 材料类别与锚点的可比性

2D 冰键合圆盘（r=0.75 m 量级）对比 Hostun 砂文献值，本身是跨材料类比；本审计的结论不依赖这一点（F1 已在算子层闭环），但 L4 复测时应在文档里显式声明该类比的适用边界。

## 4. 未跑 / 限制

- 全量 4 配置独立复跑：集群 cj1060 已递交（stage_dir=m15279 工作树，`--only r24x48_p100`），审计关账时仍 **queued** → reproducibility **not verified by independent full run**（只验了 quick 可执行 + 同进程两次调用逐位一致）。
- 修正后角度 60.0° 来自 **cumulative 场**（`final_field.npz`），而 m15279 的 headline 是 **post-peak incremental 场均值**；两者口径不同。因此本审计的判定是「基准 FAIL 不可信、需重测」，**不是**直接宣布 60° PASS（重测 = 用修正算子跑 4 配置、按同一 incremental 口径出数）。
- 本审计未修改被审仓任何文件；所有探针在 `docs/board_audit/m15315/` 内、以 `TENSORDEM_REPO` 指向被审工作树只读运行。

## 5. 复现

```bash
# 探针（P1/P2/P3 一体，判据写死在脚本 docstring）
TENSORDEM_REPO=/root/tp-agent/work/m15279 \
  python docs/board_audit/m15315/audit_biaxial_estimator.py
# 图
python docs/board_audit/m15315/make_figure.py
# 被审单测试
cd $TENSORDEM_REPO && PYTHONPATH=src python -m pytest tests/test_biaxial_shear_band.py -q   # 15 passed
```

## 6. 返修建议（最小、可验证）

1. `radon_band_angle` 改为**单一各向同性尺度**归一化（`scale=max(ptp(x),ptp(y))`），保持 η² 判据/分箱/折叠不变；
2. 测试补「非等轴窗口」参数化用例（A∈{0.43,0.87,1.0,1.73} × θ∈{45,60,70}），断言回收 ≤1°；
3. 用修正算子重跑 4 配置，按原 incremental 口径出 got，重判 60°±15%（预期转 PASS）；
4. 把本锚点登记进 `tensor_benchmark_cases`；修正报告 ③ 的粒子数 1104→1128。