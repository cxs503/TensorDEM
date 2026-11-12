# 交差审计：m16074 · CASE-DEM-BRAZIL-NUT-SEGREGATION（TensorDEM）

- **审计单**：m16134（`board-audit`，任务榜交差核验，非重做）
- **被审单**：m16074（repo=TensorDEM，status=succeeded，finished 2026-11-12 09:36:57Z）
- **锚点**：CASE-DEM-BRAZIL-NUT-SEGREGATION · 振动分选巴西坚果效应
- **参考**：Möbius, Lauderdale, Nagel & Jaeger, *Nature* **414**, 270 (2001), DOI:10.1038/35104697（Fig. 1）；密度带 0.47–0.50 取自同组 *PRE* **72**, 011306 (2005) Fig. 3
- **裁定**：**L3（系统集成）**，`gate` 复核后 `confirmed`。benchmark 自身判据 **FAIL**（honest-FAIL，非造假），L4 不成立。
- **产物**：`artifacts/audit_m16074/`（独立重算脚本+输出、2 图、evidence.json）

---

## ① 审计对象与判据（先定判据，再动手）

交差三件套要求：run_artifacts 真存在 / benchmark ref vs got 数值 / repro_cmd 可执行。
级别判据（maturity_gate.py `SCHEME`，机器可核，非审计者意见）：

| 级 | 机器判据 |
|---|---|
| L1 | tests 非空 |
| L2 | impl_files 非占位 + entry + run_artifacts 存在 |
| L3 | L2 + integration_callers（生态仓 grep 复核调用方） |
| L4 | L3 + benchmark（ref vs got + 误差数字）+ repro_cmd 可复现 |
| L5 | L4 + external_anchor（文献/实验锚） |

`tensor_benchmark_cases` 里 **没有**本锚点的行（`anchor_slug LIKE '%BRAZIL%'` → 0 行，全表仅 25 行）。因此"基准判据是唯一裁判"这一层对本单**无法自动裁**——裁判表本身缺行（见 F7）。本轮改为对照**被审单自己预注册的判据**（`prereg.json`：定性方向 + T_rise 类趋势描述量 <20%）+ 原文 DOI 锚点逐项核。

## ② 怎么做的（审计方法）

1. 读 m16074 的 `result`（`tensor_dev_missions`）、报告（`tensor_reports` #15211/#15213，正文 8280 字，5 图）、工作树 `/root/tp-agent/work/m16074`。
2. **不信任交付自己的聚合器**：`artifacts/audit_m16074/verify_recompute.py` 只读原始 `runs/*.json` 的拍序列，独立重算升速、方向、密度族抛物线顶点 ρ*、与参考带的相对误差，最后才与 `result.json` 对表（脚本不 import `analyse.py`，独立性由构造保证）。
3. 复现：用交付的求解器入口按 `grid_reduced` 的同一参数重跑 `R6/ρ=3.24/seed 0/4 拍`，比对 `audit.digest` 与升速。
4. 取证：`sha256` 逐一钉住 impl/工件/文档；核对 `git` 提交与集群作业状态（cj1238）。
5. 报告六段与预注册偏差逐条核，按 severity 分类，写 disposition。

## ③ 核验结果

### 3.1 三件套是否真存在

| 项 | 结论 | 证据 |
|---|---|---|
| impl_files 真实、非占位 | ✅ | `benchmarks/case_brazil_nut_moebius2001/run_case.py`（16.6 kB，`run_case()` L228、`embed_intruder()` L125、`tap_drive()` L164）、`analyse.py`（37.2 kB）、`sweep.py`（7.6 kB）；`sha256` 见 `evidence.json` 与 `verify_recompute.json` |
| entry | ✅ | `sweep.py:main()`（L152，`--grid {pilot,reduced,full}`）、`run_case.py:main()`（L343，逐案例 CLI） |
| run_artifacts 真实运行工件 | ✅ | 6 个 `runs/*.json`（每例 36–74 kB：完整拍序列 + 快照 + digest + 墙钟）；聚合 `artifacts/.../result.json`（13.2 kB）；报告 `report.md`（11.0 kB）+ 5 图。独立统计：**114 648 步积分、346.1 s 墙钟**，非合成数据 |
| 求解器真被调用（L3） | ✅ | `run_case.py:50-51` `from tensordem.granular2d.engine import DiskConfig` / `from tensordem.segregation.bed2d import BidisperseBed, channel_walls`；生态仓内调用方存在：`scripts/segregation_campaign_report.py`、`tests/test_segregation_bed.py`、`benchmarks/mts_contact_integration.py` |
| tests | ✅（仅账本层） | `tests/test_case_brazil_nut_moebius2001.py`，`pytest -q` → **6 passed in 6.4 s**；全部为度量定义/几何比值/密度覆写/预注册网格完整性，**不含物理断言** |
| repro_cmd 可执行 | ⚠️ 部分 | 逐案例命令**位精确**复现：`run_case(size_ratio=6.0, rho_ratio=3.24, seed=0, max_taps=4, settle_time=0.02, dt=2.5e-5)` → digest `045696266a…2bb436` 与存档**完全一致**，升速 −0.264824455 d/tap 一致，墙钟 44.5 s。**但**交付文档里的"一键复现"是 `sweep.py --grid full`，该网格从未跑过（见 F1） |

### 3.2 benchmark：ref vs got（预注册判据逐项）

| 判据 | ref | got | 判定 |
|---|---|---|---|
| H1 重颗粒（ρ/ρ_m=3.24）上升 | 上升（Brazil-nut） | **下沉**：4 拍净 −0.79 d（seed0）/ −0.72 d（seed1），全部 6 例 direction=sink | ❌ FAIL |
| H2 最慢上升密度 ρ*（=T_rise 峰位） | 0.47–0.50 | 两 seed 拟合 2.125（相对误差 **338.1%**）；**seed0 单 seed 拟合 3.045（528%）** | ❌ FAIL |
| H3 尺寸依赖（T_rise 随 D/d 减小） | D/d 至 50.8 | 仅 D/d=4/6 两档（10 档未跑），升速 −0.11 / −0.26 d/tap | ❌ FAIL（且不可判） |
| T_rise 绝对值 | 仅图 1 图形给出 | 6/6 例 `surfaced_at_tap=null`（4 拍预算内从未起浮） | **WITHHELD**（原文未给数，如实标注，未像素猜读） |
| 分层指数 | 趋势量 | 初 → 末：−0.139 → −0.146（重颗粒） | 有数，方向同"下沉" |
| 气压分支（90 torr） | 原文核心结论 | 2D 干颗粒无气相 | NOT_SUPPORTED（已声明） |

**噪声量化（本轮审计新增，独立重算）**：ρ=3.24 处两 seed 差 **0.0232 d/tap**，而整个密度族 0.47→3.24 的趋势跨度只有 **0.0209 d/tap** ——**种子噪声大于全部密度趋势**。因此 ρ* 与"338% 误差"是拟合权重（x=3.24 被两个 seed 重复计入，R²=0.344）造成的伪像，不是可分辨的物理量。正确表述应是 **unresolved / noise-limited（WITHHELD 类）**，而不是"测到了 2.125"。

### 3.3 报告六段齐全性

| 段 | 状态 |
|---|---|
| ①问题+ref 出处 | ✅ DOI + Fig.1 图注参数 + 同组 PRE 补充来源，逐项列 |
| ②方法与实现 | ✅ 求解器模块、建床/置换/逐拍驱动流程、算纲三件 |
| ③参数表（无量纲完整） | ✅ 16 项（Γ、f、A/d、D/d、D/H₀、W/H₀、dt/t_c、m 比…） |
| ④对照表 ref vs got + 打勾叉 | ✅ 有表且敢打叉（但含 F4/F5 的标签错误） |
| ⑤收敛性/参数敏感性（≥2 点） | ❌ **实质为空**：`dt_convergence=[]`（未跑）；"Γ 敏感性"表两行都是 Γ=5（其实是两个 seed） |
| ⑥失败模式与工件 | ✅ honest-FAIL + fail_stage + failure_reason + 下一轮路径 + 工件清单 |

## ④ 发现（按严重度）

**HIGH**

- **F1 documented repro_cmd 指向未执行的网格。** 交付文档"一键复现"= `sweep.py --grid full`（22 例、D/d=8、30 拍）。审计时集群作业 **cj1238 仍 `queued`（age 17.4 min，node 未派发）**，`cj1237` 同批；`runs/` 里只有 `grid_reduced` 的 6 个文件。→ 按文档复现得不到报告里任何数字；且 `grid_reduced` 编码 8 例，实跑 6 例（缺 `R10_ρ3.24_s0`、`R6_ρ3.24_s0_G7`）。
- **F2 任务书点名的指标从未产生观测。** 目标指标="大颗粒上升时间 T_rise 与分层指数"。6/6 例 `surfaced_at_tap=null`（4 拍 vs 预注册 30 拍），T_rise **零观测**，只能用"每拍升速"代用。交付虽在 §⑤/⑥ 如实写了删失，但未把这个"主指标零样本"提到判据层。
- **F3 §⑤ 呈现了不存在的研究。** `analyse.py:215 gamma_family()` 只按 size/ρ/dt 过滤，不按 Γ 过滤；reduced 网格里 Γ=7 的案例没跑，于是"Γ 敏感性"表打印 `Γ: 5 | 5`——把两个 seed 当成了两个驱动强度。dt 收敛为空表。任务要求"至少两点"敏感性 → **实际 0 点**。这是本单最典型的"报告好看不算数"项。

**MEDIUM**

- **F4 ρ* 不可复现于同类可辩护分析。** `analyse.py:136 fit_slowest_density()` 用 5 点、x=3.24 重复（两 seed），R²=0.344；seed0-only 重算 ρ*=3.045。两个数差 0.92，对应"误差"338% vs 528%。结论（FAIL）不变，但**报告给出的误差数字本身是拟合选择的函数**（且小于种子噪声，见 3.2）。应在报告中给出拟合敏感性区间或直接判 unresolved。
- **F5 H1 的 "physics FAIL" 表述越过了尺度缺口。** 执行 D/d=6，参考 D/d=50.8；同组论文明确指出 D/d 小时尺寸分离效应减弱。在 1/8.5 的尺度比下**无法证伪参考分支**，只能证伪"在这个尺度、这个驱动强度下会上升"。报告 §⑥ 写了尺度缺口，但 verdicts 里 H1/H3 只写 "pass: false"，未带这条限定。
- **F7 平台侧：基准裁判表缺行。** `tensor_benchmark_cases` 无本锚点记录（`anchor_slug LIKE '%BRAZIL%'` → 0 行），任务书指定的"唯一裁判"对本锚点不可用；查得全表 25 行均为其他族。→ 建议由算例池维护方按 prereg 判据补登记（本条为平台级 friction，非 m16074 责任）。

**LOW**

- **F6 图文标签错。** `fig1` 说明写 "D/d = 8"（实际 6，`analyse.py:321` 硬编码）；`fig3` 图例写 "dt = 2e-5 s (production)"（实际 2.5e-5，`analyse.py:366`），并承诺 "half-step points overlaid" 而半步步长案例不存在；相似性表对 D/H_bed（0.295 vs 0.177）、W/H_bed（0.953 vs 0.572）标"对齐"，实际差 40%。
- **F8 `f·√(d/g)` 差 2 倍**（0.0928 vs 0.1856，因床粒 d=2 mm vs 参考 0.5 mm）未列入偏差声明；只声明了 D/d 缺口。
- **F9 交付分支未落仓。** `result.head_commit=null`，生态仓 `/root/tensor-ecosystem/TensorDEM` 内无 `benchmarks/case_brazil_nut_moebius2001`；交付只存在于工作树（工作树内单 commit `b575cf05`，工作区 clean）。审计取证因此全部指向工作树路径。

**核验通过、非问题（避免误伤）**

- 数据是真的：`runs/*.json` 含 114 648 步、逐拍快照、digest；重跑**位精确**一致（DEM 确定性），说明"可复现"在案例级成立。
- 交付聚合与原始序列**自洽**：独立重算 6 例升速/方向与 `result.json` 逐一相等（0 mismatch）。
- honest-FAIL 是真诚实：三条假设全打叉、fail_stage 归因清楚、未把失败包装成成功；`server_quality_audit.gate_valid=false` 是平台侧门缺失告警，与本单交付质量无关（本轮我来补上 verdict）。
- tests 6 passed 无造假，但属账本层，不能作为物理验证证据。

## ⑤ 级别判定（L0–L5）

| 级 | 是否成立 | 依据 |
|---|---|---|
| L1 | ✅ | tests 6 passed |
| L2 | ✅ | 真代码 harness（输入→计算→输出）+ 入口 + 6 个真实运行工件 |
| L3 | ✅ | 调用库内求解器 `tensordem.granular2d.engine` / `tensordem.segregation.bed2d`（生态仓 grep 有调用方），未自造物理 |
| L4 | ❌ | 基准**未通过**自身预注册判据；T_rise 零观测；网格 1/8 尺度、4/30 拍；无 dt 收敛 → "完整算例+误差分析+可复现"三项缺两项 |
| L5 | ❌ | 无 L4 基础 |

**裁定：L3。** 百页报告与 5 图不改变级别——本单的实质是"一个能真跑库求解器、工件位精确可复现的算例 harness + 一份诚实的否定性结果"，而不是"巴西坚果效应被复现/验证"。负结果本身有价值（已写 friction 与下一轮：逐粒 `drag_time`），但**不构成 L4**。

## ⑥ 处置与返修建议

- disposition：**approve with notes（L3 confirmed，benchmark 判为 honest-FAIL 保留）**；不建议整单返修（实现是真的），建议**在算例池层补一轮 reduced-grid 收尾**：
  1. 文档与报告的一键复现改为 `sweep.py --grid reduced --workers N`（或补跑 `grid_full` 后更新），并把 `grid_reduced` 缺的 2 例跑完（R10、G7）。
  2. 敏感性补齐：dt 减半对（`grid_reduced` 已有 `dt` 字段能力）、真 Γ=7 案例 → 让 §⑤ 的两点真正存在。
  3. ρ* 判据改为"区间/不可分辨"表述，或加 seed 扩样（≥4 seed）后重拟抛物线；报告中给出拟合敏感性。
  4. 修 fig1/fig3 标签与相似性表"对齐"注记。
  5. 平台侧：给 `tensor_benchmark_cases` 补本锚点行（metric/tolerance 取 prereg），否则交差核验仍无机器裁判。
- 未执行的检查（如实声明 **not_run**）：`grid_full` 全量网格、D/d=50.8 尺度实验、气压（90 torr）分支、逐粒 drag_time 变体、3D 圆球对比。本审计未重跑 `sweep.py`（只做单案例位精确复现 + 全量工件独立重算）。

## ⑦ 图表（`artifacts/audit_m16074/`）

- `fig_rate_vs_rho.png`：升速 vs ρ/ρ_m；两种可辩护拟合的 ρ*（2.12 / 3.05）+ 参考带；标注"种子散布 0.0232 > 密度趋势跨度 0.0209"。
- `fig_z_history.png`：6 例侵入体 z/d 随拍历史——全部下降，4 拍内无一起浮。
**MEDIUM（平台侧，审计机制）**

- **F10 `maturity_gate` 的 L4 判据是"存在性"而非"通过性"。** 本单 `judge(4, evidence)` 在机械复核下**也会返回 4**：`benchmark` 只查 `has_numbers()`（有 ref/got 数字即过），`repro_cmd` 只查非空字符串——一个**基准判据全 FAIL** 的交付只要把失败数字写进 benchmark，就能被机器判成 L4。本审计按功能实质判 L3（不声明 L4），但机制本身无法区分"benchmark 通过"与"benchmark 做了但失败"。建议 gate 增加 `benchmark.pass` 布尔位（PASS 才计 L4；FAIL/WITHHELD 计 L3 并保留负结果入账），否则"honest-FAIL 也算 L4"会成为新的虚标通道。

---

## ⑧ 交差物索引

- verdict 落表：`tensor_maturity_verdicts` id=**424**（repo=TensorDEM, scope=`m16074 CASE-DEM-BRAZIL-NUT-SEGREGATION`, mission_id=16134, level=**3**, verdict_by=`board-audit`, gate_state=pending → 由 `maturity_gate.py` 30 min 周期机器复核；提交前已在进程内预演 `judge(3, evidence, 'TensorDEM', 16134)` → **`(3, [])`**，即 L2/L3 证据全部机械通过、L4 不声明）
- 独立重算：`artifacts/audit_m16074/verify_recompute.py` → `verify_recompute.json`
- 图表/证据：`artifacts/audit_m16074/{fig_rate_vs_rho.png, fig_z_history.png, evidence.json}`
- 论文级进展报告：`tensor_reports` id=**15232**（更正版；15230 因噪声量化行取错样本被本单自行更正并重发，属自查自纠，留痕不删）
