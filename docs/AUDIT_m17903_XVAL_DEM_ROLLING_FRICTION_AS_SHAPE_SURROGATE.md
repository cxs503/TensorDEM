# 交差核验：m17903 XVAL-DEM-ROLLING-FRICTION-AS-SHAPE-SURROGATE（board audit, m18083）

**核验对象**：m17903（TensorDEM，status=succeeded）。交付 commit `95618e52`（local branch `agent/m17903`）
＝ 本审计单 cherry-pick 进 `agent/m18083` 的 `43f8321e`（内容未改一字）。
**判级结论：L2（独立功能实现）**；L3/L4/L5 均不成立，理由见 §4 逐级核。
**裁决：approve with notes**（交付可留主线；不得升 L4；返修建议见 §7）。

---

## 一、独立性限制（必须声明）

* 审计者未参与 m17903 的开发，但**无法独立获得文献原始数值**：外部锚
  Wensrich & Katterfeld (2012, *Powder Technol.* **217**:409–417) 的 PDF 不在本环境，
  全部 ref 值来自仓内文献表 `docs/GAP_DEM_CLUMP_ROUND2.md:32`（sha256 `4767513195d2…`，
  审计在 worktree 内实测一致）。因此「文献 ref 本身是否正确」在本单只能 **WITHHELD**，
  审计能核的是「报告是否忠实使用该 ref + got 是否在门内」。
* `tensor_benchmark_cases` **无本锚点行**（84 行全表，TensorDEM 已有锚点仅 5 条），
  机器裁判缺席 → 由本审计按报告自声明的 `gate_rel_error=0.15` 与 5 条判据自行裁。
* 未重跑全部 8 个 DEM 点（每点 56–159 s，报告称合计 ~19 min）；聚合/出图/单点端到端已实测（§3）。

## 二、三件套取证（run_artifacts / benchmark 数值 / repro_cmd）

| 项 | 结论 | 证据 |
|---|---|---|
| run_artifacts | **真存在**，8 点 sha256 与 `result.json.provenance.point_files` 逐一对上 | `artifacts/xval_rolling_shape_surrogate/points/*.json`、`result.json`、`fig_shape_surrogate_calibration.png`（119 KB） |
| benchmark ref vs got | **存在但不过门**（3/5 判据 FAIL，见 §3） | `result.json.checks` / `literature_comparison` / `sensitivity` |
| repro_cmd 可执行 | **可执行且对已登记输入集可复现**；但对「树里现在这副文件集」**不可复现**（见 §5 F2） | 实测：`aggregate` 用 8 点 → 与 `result.json` 数值逐一相同（仅 `wall_seconds`/`provenance` 变） |
| 报告六段 | 齐（①–⑥ 全在，5788 字符），但 **④ 图未内嵌**（`figures[]` 只有 name/caption，无 `png_base64`） | `GET /tensor/reports/16865` |
| 落库状态 | m17903 标 succeeded，但 commit **不在 origin/main**、**无任何远端分支**、生态仓 checkout 停在 m9876 | `git merge-base --is-ancestor 95618e52 origin/main` → false；`ls-remote --heads`(625) 无 `agent/m17903` |

## 三、结果核验（数字全部取自 `result.json`，零手抄）

基准门：`gate_rel_error = 0.15`（报告自声明），5 条判据：

| 判据 | ref | got | 相对误差 | 门 | 判 |
|---|---|---|---|---|---|
| C1 单调性 | — | Spearman = 1.000 | — | — | ✔ |
| C2a μ_r=0 → θ | 21.0° | 16.0999°（饱和拟合外推） | **+23.33 %** | <15 % | ✘ |
| C2b μ_r=0.5 → θ | 32.0° | 36.8437°（实测） | **+15.14 %** | <15 % | ✘（临界超门） |
| C3 灵敏度 dθ/dμ_r | 22.0 °/unit | 41.4875 °/unit | **+88.58 %** | <15 % | ✘ |
| C4 反演 21°/32° | 21°, 32° | 32°→μ_r\*=0.3075；21° **不可达**（实测最小 21.06°） | — | 可反演 | ✘ |
| C5 样本外迁移（唯一非循环精度） | hold-out 实测 | 拟合预测 | 最差 **10.65 %**（N=800）/ 2.10 %（seed23） | <15 % | ✔ |

`all_checks_pass=false` —— 报告如实写了 **honest FAIL**，没有粉饰、没有把 10.65 % 说成「通过验收」，
也没有「<3 %」的表述（唯一小量级是 C5 的 2.10 %，属样本外点之一，非锚点）。**报告诚实性核验通过**。
关键：**是否 <3 %？否**（C2/C3 量级 15–89 %），审计与该声明一致。

## 四、逐级核（级别=功能实质，不看报告篇幅）

* **L1（数学/单元验证）✔**：`tests/test_sandpile_rolling_calibration.py` 15 passed（10.2 s，工作树内实测），
  含合成参数回归（拟合法能恢复已知 θ∞/A/μ_c）、反演精确性、迁移打分、文献解析器在缺文件/缺行时 loud-fail。
* **L2（独立功能实现）✔**：`src/tensordem/protocols/sandpile_calibration.py` 455 行真实现
  （饱和拟合 / 单调反演 / 样本外打分 / 文献正则解析，非 placeholder）；入口 `benchmarks/xval_rolling_shape_surrogate/run_sweep.py`
  能 point→aggregate→figures 三段跑通并产出工件。
* **L3（系统集成）✘**：`run_sweep.py:37` 真的 import 该模块，且 shipped 协议 `sandpile.py` 增加了
  `calibration_sweep` 门（`:183/:202/:310/:357`，先验门只对 mu_s/mu_r 开洞、control 必须 baseline、
  记录层三重标记不会误读成 baseline，`test_calibration_sweep_cannot_be_a_negative_control` 覆盖）——
  这是**库级 + 协议门钩子 + 基准驱动**，但**不是生产求解入口的调用路径**；且交付当时不在任何远端分支，
  平台集成 grep 在生态仓（`ad75394d`=m9876）里必然失效。故按「已落地产品的集成」只给 L2。
  （本次审计已把交付 cherry-pick 进本分支，落地后 grep `sandpile_calibration` 即可复核 L3。）
* **L4（基准验证）✘**：完整算例 ✔、误差分析 ✔、可复现 ✔，但**基准裁判本身判 FAIL**（3/5 判据不过），
  且锚点在 `tensor_benchmark_cases` 无登记行。基准不过 ≠ 基准验证达标，故不给 L4。
* **L5 ✘**：外部锚为仓内二手表、原文 WITHHELD，无独立科研工程交付。

## 五、发现（按严重度）

* **F1（blocking，L4 阻塞）基准不过 + 基准未登记**：C2a 23.33 %、C2b 15.14 %、C3 +88.58 %、C4 不可达；
  `tensor_benchmark_cases` 0 行。复现：`python -c` 读 `result.json['checks']`。
* **F2（major，可复现性）发布的聚合件与树内工件集不一致**：
  `points/hold_dthalf_mu0p30.json`（dt/2）写于 **10:31**，晚于 `result.json`（10:30），不在 provenance；
  把它放进输入后 `aggregate` 得到不同结论面：`protocol_spread` 2.47°(8.0 %) → **4.11°(13.0 %)**，
  C5 最差 10.65 % → 2.95 %。即：**照树复现 ≠ 已发布 result.json**。
  复现：把该点移出后再 `aggregate` → 与 `result.json` 完全一致（审计已实测）。
* **F3（major，科学）dt 敏感性 5.16 % 且报告未提**：μ_r=0.3、N=1600、seed=11，
  dt=1.038e-4 → θ=31.81°，dt/2=5.188e-5 → **θ=33.45°**（+1.64°，+5.16 %）。
  该量与 C2b 的超门幅度（15.14 % vs 15 %）同阶，说明**协议在 dt 上未收敛即已把结论写成「文献对应不可移植」**；
  物理结论方向可能不错，但**归因强度被高估**，且这一检查没写进报告的「注意点」。
* **F4（minor，报告规范）④ 图未内嵌**：`figures[]` 无 `png_base64`，只有 name/caption。
* **F5（minor，溯源）`git_rev: "unknown"`**：`result.json` 与每个点文件都无法绑定到产生它的 commit。
* **F6（minor，PNG 非同字节复现）**：重跑 `figures.py` 得 sha256 `8f162cf4…`，已提交为 `2874b0e9…`
  （matplotlib 版本相关，未做像素级比对 → 仅记「字节不可复现」）。
* **F7（背景，非本单责任）**：`tests/test_sandpile_degenerate_measure_m17676.py::test_flat_deposit_keeps_the_angle_and_reports_packing_gap`
  在**基线 da3d59e6 同样失败**（`git archive` 干净副本实测），与本交付无关，属主线既有红灯。

## 六、未做的检查（如实标注 not_run）

* 8 个 DEM 点未全量重跑（只做聚合/出图/单点端到端）；因此「同 seed 下 DEM 内核逐比特确定」未证。
* 侧壁线性拟合算子 `repose_angle_from_profile` 未独立重实现比对（只抽验 r²≥0.987、h/R≥3 与报告表一致）。
* 原始论文数值不可得（WITHHELD），无法判「文献表转写是否正确」。
* ECO 生态仓全量 grep/CI 未跑（checkout 停在 m9876，无参考价值）。

## 七、处置：approve with notes + 返修建议（不自行改锚点）

1. **登记基准**：`tensor_benchmark_cases` 补一行 anchor 同名，`metric` 必须覆盖**全部 5 条判据**
   （不能只登记已通过的 C5），`tolerance` 写明 15 %；当前状态应为 failed（honest FAIL 入账）。
2. **把 dt/2 点纳入结论面**：要么进 `aggregate` 并给出 dt 收敛表（θ(dt)），要么显式声明排除理由；
   当前「树里有、报告里没有、聚合里没有」三处不一致必须消除（F2/F3）。
3. **报告补图**：`figures[].png_base64` 内嵌（F4）；`provenance.git_rev` 写真实 commit（F5）。
4. 落地后由后续审计复核 L3（grep `sandpile_calibration` ∈ TensorDEM/），基准登记后再谈 L4。

## 八、本审计自身产物

* `artifacts/audit_m17903/evidence_m17903.json`（机器可读 evidence，落 `tensor_maturity_verdicts` 的 JSON）
* `artifacts/audit_m17903/make_audit_figure.py` + `fig_audit_m17903.png`（三联：C2 ref/got 与 ±15 % 门、
  标定面+拟合+文献对应、dt 敏感性；数字全部运行时读 `result.json` / `points/*.json`，零手抄）
* 分支 `agent/m18083` 含交付本体（cherry-pick `43f8321e`，未改动内容）+ 本审计文档。