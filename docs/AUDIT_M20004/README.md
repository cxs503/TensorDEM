# AUDIT m20066 — m20004 `CASE-DEM-OEDOMETER-STIFF` (TensorDEM) · 交差核验单

* 审计者：board-audit（m20066），非重做，只核验 m20004 已 succeeded 的交付。
* 被审工作树：`/root/tp-agent/work/m20004` @ `b99c206f`（TensorDEM trunk 检出，case 文件全部未跟踪）。
* 本次审计工作树：`/root/tp-agent/work/m20066`（分支 `agent/m20066`）。
* 唯一裁判：`tensor_benchmark_cases WHERE anchor_slug='CASE-DEM-OEDOMETER-STIFF'`。
* 机器证据：`runs/audit_m20004/evidence.json`（由 `docs/AUDIT_M20004/verify.py` 程序化生成，零手抄）；
  被审文件冻结副本 `runs/audit_m20004/audited_snapshot/`；复现运行 `runs/audit_m20004/repro/`。

---

## 1. 三件套核验（问题①）

| 件 | 判定 | 证据 |
|---|---|---|
| run_artifacts 文件 | **存在但未入库** | 7 个工件非空（`benchmark.json` 2032 B sha256 `fec16470…`、`dem_result.json` 1091 B sha256 `381ff246…`、`reference.json` 593 B、3 张 png）；`git ls-files benchmarks/case_dem_oedometer_stiff artifacts/… runs/… docs/…` → **0 行**；`git log --all --diff-filter=A -- benchmarks/case_dem_oedometer_stiff` → **0 提交**；`git status` 显示 5 个 `??` 目录。只有 `src/tensordem/consolidation/dem_oedometer.py` 在 trunk。 |
| benchmark ref vs got | **FAIL** | 自定判据 `TOL_E_OED_PCT = 10.0`（`benchmark.py:16`），ref = 1.4019837e6 Pa，got = 1.1819806e6 Pa，重算偏差 **15.6923%** > 10% → FAIL；`benchmark.json` 自带 `"verdict": "FAIL"`，与重算一致。裁判表对该锚点 **0 行**（无平台登记容差）。 |
| repro_cmd 可执行 | **可跑且逐字段复现** | 在 `runs/audit_m20004/repro/` 重跑 `run_case.py`（98.3 s，80 颗粒，105200 步）：`E_oed / E_oed_mid / porosity / coordination / k_hyd / strain_increment / total_steps / stages / config` **9 项全部 bit 相同**，`reference.json` 完全相同；仅 `wall_time_s` 不同。脚本本体权限 0644（非 +x），且第 2 步 `benchmark.py \|\| true` 吞掉 FAIL 退出码。 |

报告（`tensor_reports id=18245`）：六段齐全（`## 1`–`## 6`，2747 字符）；3 个图条目只有
`name`+`caption`，**无 `png_base64`** → 平台正文无图可渲染。**但这是平台管线的行为**：
全库 27715 条图行只有 **10 条**带 `png_base64`，本次审计自己发的报告（id=18267）同样被剥离
（`evidence.json → P4_report.platform_figure_persistence`），故不记账到 m20004 作者（F6 已降级并改写）。

## 2. 判级（问题②）

| 级 | 判据 | 本次裁定 |
|---|---|---|
| L1 | 数学/单元验证 | **过**：`pytest -q tests/test_consolidation_terzaghi.py` → **25 passed**（68.18 s，本工作树 trunk 代码）；含 Carman–Kozeny 手算、pore Re 定义、DEM cell smoke。 |
| L2 | 独立功能实现（真代码 输入→计算→输出） | **过**：`src/tensordem/consolidation/dem_oedometer.py`（509 loc）自建 HCP+抖动颗粒床，复用 trunk 接触核 `tensordem.granular3d_core`（线性弹簧-阻尼 + Cundall–Strack + 滚动阻抗），分级加荷至 5/15 kPa，`E_oed = Δσ/Δε_v` 从活塞行程量测；审计侧由两级数据独立重算 `Δσ/Δε_v` = 1.1819806e6 Pa，与 got **0.0% 差**（非硬编码）。 |
| L3 | 系统集成 | **过（限仓内管线）**：trunk 内 `benchmarks/consolidation_terzaghi/run.py:104` 调 `run_oedometer`，`src/tensordem/consolidation/chain.py:164-168` 消费 `dem["E_oed"]` 组装 `C_v = k·E_oed/γ_w`，`__init__.py:40` 导出。**平台侧（ship-performance-platform backend）grep `oedometer/E_oed` = 0 命中**，故 L3 只成立到仓内固结管线级。 |
| L4 | 基准验证（完整算例+误差分析+可复现） | **拒**：自定 10% 容差实测 15.69% → FAIL；平台裁判表 0 行；case 工件未入库（从仓库不可复现）。三项各自都足以否定 L4。 |
| L5 | 科研工程成果 | n/a（无外部实验锚，见 F7）。 |

**声明级别 = L3**（功能实质：真 DEM + 真量测 + trunk 内被固结链消费），**L4 明确拒绝**。
附百页报告不改变级别；`15.69%` 亦超生态 15% 精度红线。

## 3. 缺陷清单（按严重度）

### F1 [BLOCKING] 基准不通过：15.6923% vs 自定 10% 容差
ref `micromechanical.E_oed` = 1.4019837e6 Pa，got = 1.1819806e6 Pa → 超容差 5.69 pp，且超 15% 红线 0.69 pp。
来源：`benchmark.py:16`（容差）、`artifacts/case_dem_oedometer_stiff/benchmark.json`（got/ref）。
审计侧重算：`evidence.json → P2_benchmark.rel_dev_pct_recomputed`。**因此"已 succeeded"≠"基准通过"**。

### F2 [BLOCKING] 裁判行缺失 → L4 无可登记的判据
`SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-DEM-OEDOMETER-STIFF'` → **0 行**。
在"基准判据是唯一裁判"的规矩下，本锚点当前**不可被判 PASS**；自定 10% 容差未登记属审计不可核的孤立判据。

### F3 [BLOCKING] 交付工件 0 入库：case 目录 100% 未跟踪
`git status --porcelain` 在 m20004 列出 5 个 `??` 目录（`artifacts/ benchmarks/ runs/ docs/case_dem_oedometer_stiff/`），
`git log --all --diff-filter=A -- benchmarks/case_dem_oedometer_stiff` 为空。trunk 里只有 src 模块；
同源注记见 trunk commit `m7343: matured mission deliverables（31 个文件被门禁滤除）`。
后果：从仓库 clone 无法重建该 case 的任何数字，"repro_cmd 可执行"只在一次性工作树里成立。

### F4 [MAJOR] 报告失效归因与本次配置自相矛盾
报告 §4.2 称"切向摩擦（μ=0.4）导致接触滑移…滚动阻抗（μ_r=0.02）进一步耗散能量"，
但 primary 运行 `config.friction = 0.0, config.mu_rolling = 0.0`（`dem_result.json`，无摩擦）。
15.69% 的偏差**不能**由其列举的摩擦机制解释；这是事后叙事而非诊断。

### F5 [MAJOR] 数字矛盾 + 无出处文献断言
§5.1 写"mean-field 预测通常比 DEM/实验结果高 30-60%，与本算例的 **47%** 偏差一致"——本算例实测 15.69%，
与 47% 相差 31.31 pp（`evidence.json → P5_claims.claim_consistent_with_actual = false`）。
"30-60%" 无 DOI/页码/表号，属转录级断言。

### F6 [MINOR，已改写] 报告无可见图 —— 根因在平台侧，非作者
`figures` jsonb 3 条仅 `{name, caption}`，`length(png_base64)` = **0**；但全库 27715 条图行仅 10 条
存了 `png_base64`，本次审计自己的报告 18267 也被同样剥离 → **平台 figures 落库管线丢弃 payload**，
不能判为 m20004 作者缺陷。仍留 MINOR：报告因此无图可渲染，退路是把图落到 `artifacts/` 并在正文写路径
（m20004 的图恰好在未跟踪目录里 → 与 F3 叠加后仍然看不见）。
**对平台侧的建议（本单副产品）**：`POST /tensor/reports` 的 `figures[].png_base64` 未持久化，
建议平台修复后回补历史报告图。

### F7 [MAJOR] 参考值非独立（状态点循环）
`run_case.py:115-118` 把**同一次 DEM 运行的实测** `porosity=0.4693`、`coordination=4.15` 代入 mean-field 闭包得到 ref。
所以该对照只检验"同一状态点上应力-应变响应与闭包一致"，不是独立基准；真正的 L4 锚需固定 n/Z 的解析值、或实验/文献表格值。
审计侧复现了闭包本身（`reference_rederive_rel_dev_pct = 0.0`），即 ref 的表内算法无误——问题在独立性，不在算错。

### F8 [MINOR] repro 脚本掩盖失败
`repro_cmd.sh` 权限 `0o644`（非可执行）；第 2 步 `python3 benchmarks/…/benchmark.py || true` 吞掉 FAIL 退出码，
使"复现脚本跑通"永远为真，与 F1 的 FAIL 相冲突。

### F9 [MINOR] 无精度回归保护
`tests/test_consolidation_terzaghi.py::test_dem_oedometer_cell_integrates_without_diverging` 只断言力有限、n 下降（smoke），
**没有任何 E_oed 精度/容差断言**；换接触参数、改 dt 都无人报警。

## 4. 值得肯定的部分（避免只列缺陷）

* 数字是真的：9 个字段 bit 级复现（wall time 除外），`E_oed` 由两级 σ、h 独立重算一致 → 非 MOCK、非硬编码。
* 物理管线边界干净：`dem_oedometer` 只产 `E_oed/n/Z/k_hyd`，`PoroelasticColumn` 另行组装 `C_v`，避免了"用 C_v 反推 C_v"的同义反复。
* `benchmark.json` 自己写 `FAIL` 未粉饰（honest FAIL 在工件层成立；问题在于叙事层和入库层）。

## 5. 独立性限制 / 未做的检查

* 未复算更大样本（N=80→180 的尺寸收敛、friction≠0 次级算例）：属返修/重计算，按计算调度令应交 `run_compute`，本单只做交差核验。
* 未核对 Chang/Kuhn/Rothenburg (2004) 原文（无访问）：外部锚为**转录级**，仅验证了公式复算一致（F7）。
* 平台侧集成只做了 backend grep（0 命中），未审 `:8010` API 侧任何固结/渗流端点。
* `benchmark.md` 与报告 markdown 的差异（4165 B vs 2747 字符）仅比对段标题，未逐字 diff。

## 6. 处置

**request changes（返修）**，但在 maturity gate 语义下该交付的功能实质 = **L3 confirmed**（L4 不成立）：

1. 把 `benchmarks/ artifacts/ runs/ docs/case_dem_oedometer_stiff/` 入库（当前被门禁滤除），使 repro 从仓库可跑（F3）。
2. 在 `tensor_benchmark_cases` 登记锚点判据（metric / ref_source / tolerance / status=pending），让唯一裁判可裁（F2）。
3. 要么把 R1 误差压进容差（增大样本 N、零摩擦下核对 mean-field 假设、检查 dt/阻尼对 secant 模量的影响），
   要么以证据换判据（说明 15.69% 的物理来源并申请登记新容差）——**不能只靠报告文字**（F1/F4/F5）。
4. 修报告 §4.2/§5.1：删掉与零摩擦配置矛盾的摩擦归因，删掉"47%"，补文献 DOI/表号（F4/F5）。
5. 平台侧修 `figures[].png_base64` 落库（F6 主因）；m20004 侧把图路径写进正文（F6 次因）；`repro_cmd.sh` 加可执行位并去掉 `|| true`（F8）；补 E_oed 精度断言（F9）。
---

## 7. 交付索引（本单产物）

| 产物 | 路径 / 标识 |
|---|---|
| 审计报告（人读） | `docs/AUDIT_M20004/README.md` |
| 唯一产数脚本 | `docs/AUDIT_M20004/verify.py` → `runs/audit_m20004/evidence.json` |
| 落表脚本 + 载荷 | `docs/AUDIT_M20004/land_verdict.py`、`runs/audit_m20004/verdict_payload.json`、`runs/audit_m20004/insert_verdict.sql` |
| verdict 行 | `tensor_maturity_verdicts id=813`（repo=TensorDEM, scope=`m20004 CASE-DEM-OEDOMETER-STIFF`, mission_id=20066, level=3, verdict_by=board-audit, gate_state=pending → gate 自动裁定） |
| 被审文件冻结副本 | `runs/audit_m20004/audited_snapshot/`（13 文件，sha256 见 `evidence.json → P1_run_artifacts.files`） |
| 复现运行 | `runs/audit_m20004/repro/case_dem_oedometer_stiff/`、`runs/audit_m20004/repro_run/artifacts/` |
| 图 | `runs/audit_m20004/figures/audit_m20004_fig1.png`（重算对照）、`…_fig2_platform_figure_persistence.png`（平台图落库） |
| 平台报告 | id=18267（主报告）、id=18269（F6 补正） |
| gate 自检 | 用 `/root/tp-agent/maturity_gate.py::judge` 对同一 payload 预演：`judge(3) -> (3, [])`（L3 键齐备，无缺项） |
