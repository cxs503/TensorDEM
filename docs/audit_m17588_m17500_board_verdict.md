# 交差审计 · m17500 `XVAL-DEM-MU-I-CONTACT-PROXY-NEGATIVE`（TensorDEM）

**审计单**：m17588（board-audit）｜**被审单**：m17500（status=succeeded，auto-approve 交付验收）
**裁定单**：`tensor_maturity_verdicts` id=**593**，level=**2**，verdict_by=board-audit
**审计器**：`scripts/audit/m17588_verify.py`（机器核证据）、`scripts/audit/m17588_write_verdict.py`（evidence 由前者产物程序化装配）、`scripts/audit/m17588_optin_bitwise_probe.py`（默认关逐位不变探针）
**原始证据**：`artifacts/audit_m17588/evidence.json`、`optin_bitwise_probe.json`、`results_m17500_copy.json`（sha256 见 verdict）

---

## 0. 结论摘要（disposition = approve with notes / 级别 L2）

三件套**真实存在**、数字**可复现、内部自洽**：8/8 单测在审计端重跑通过（5.15 s）；results.json 的
`i_mapping_error`/`mu_error`/`shortfall` 用原始量重算，5/5 行一致到 1e-9；报告 #16473 表格数字与
results.json 5/5 行一致到 0.1 pp；六段齐全。**但**两处交付性缺口使 L3/L4 不成立：

| 缺口 | 机器证据 |
|---|---|
| 成果从未入库 | `git ls-remote --heads origin agent/m17500` = 空；`git merge-base --is-ancestor agent/m17500 HEAD` rc=**1**；main `f2e227f6` 下 `src/tensordem/mu_i_contact_proxy.py` 等 6 个文件**全不存在** |
| 平台裁判缺失 | `SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-DEM-MU-I-CONTACT-PROXY-NEGATIVE'` → **0 行**（TensorDEM 已有 8 条其他锚点的登记） |

裁定 **L2（独立功能实现）**：真代码 + 真运行工件 + 单测，功能实质成立；L3 需"接入主求解器"，而唯一
调用点在**未合并分支**上，生态仓主干 grep 无引用；L4 需登记的基准裁判，锚点 0 行 → 按"基准判据是唯一
裁判"纪律，L4 不可判。

---

## 1. 取证：三件套逐项核

| 项 | 断言 | 审计核实 | 结论 |
|---|---|---|---|
| run_artifacts | `artifacts/xval_mu_i_contact_proxy_m17500/{results.json,fig_mu_vs_I.png,fig_errors.png}` | 8807 / 56239 / 39960 B，sha256 记录入 verdict；拷入 `artifacts/audit_m17588/results_m17500_copy.json` 防工作树回收 | ✅ 真 |
| benchmark ref vs got | 5 行 I_bulk/I_c/mu_bulk/mu_proxy | 用 `results.json` 原始量重算 `|I_c/I_bulk-1|`、`|mu_proxy/mu_bulk-1|`、`(0.5-mu(I_c))/0.5`（closure 取仓内 `jop_bulk_mu`）→ 5/5 行 1e-9 内一致 | ✅ 真 |
| repro_cmd 可执行 | `PYTHONPATH=src pytest tests/test_xval_mu_i_contact_proxy.py -q` | 审计端重跑 **8 passed in 5.15s**；脚本入口 `scripts/xval_mu_i_contact_proxy_m17500.py` 存在（全量 5 算例 346 s 未在审计预算内重跑，见 §5） | ✅ 单测级可执行 |
| 报告六段 | ①案例 ②方法 ③结果 ④图 ⑤注意点 ⑥误差判定 | 报告 #16473（2719 字符）六段关键词全部命中；⑤显式写 honest-FAIL 与设备局限；⑥逐项回答"是否<3%" | ✅ 齐 |
| 平台裁判 | 容差内？ | **锚点 0 行登记** → 无裁判，只有仓内 `prereg/`(DEFAULT_TOL=0.15) 自定阈值可依 | ❌ 缺 |

报告表格 vs JSON：`base_I0.01/+152.6%/+35.8%`、`base_I0.1/+42.4%/+31.0%`、`base_I0.6/+59.3%/+5.2%`、
`sens_kn4e3_I0.1/+38.0%/+9.4%`、`sens_P8k_I0.1/+46.1%/+8.1%` —— 5/5 行与 JSON 百分比一致到 0.1 pp
（`evidence.json:report_vs_json`），**零手抄**。

---

## 2. 判级（L0–L5）

| 级 | 判据 | m17500 | 依据 |
|---|---|---|---|
| L0 文档 | — | 过 | docs 101 行 + prereg 17 行 |
| L1 数学/单元 | 解析/恒等式测试 | 过 | 8 单测重跑通过（含 rigid spin 2ωr、f_n 加权手算、I_c→I_bulk 恒等、末窗选择、默认关） |
| **L2 独立功能** | 真代码 输入→计算→输出 | **过（裁定级）** | `mu_i_contact_proxy.py` 270 行非占位；真实 DEM 状态→逐接触 I_c→f_n 加权 mu(I_c)→误差表；工件真 |
| L3 系统集成 | 接主求解器 | **不过** | 调用点 `mu_i_rheology.py:1077/1269` 只在 `agent/m17500` 上；主干/生态检出无 `mu_i_contact_proxy` → grep 无引用 |
| L4 基准验证 | 登记裁判 + 容差内 | **不过** | 锚点 0 行登记；且本轮自身参考门 FAIL（见 F3），绝对摩擦比较按其 prereg 自动作废 |
| L5 科研工程 | 外部锚 + 全面认证 | 不过 | 3/5 算例 `servo_valid=False`；外部锚仅 OSS 台账 #421 + Jop 2006 引用 |

---

## 3. 严重度排序的发现（adversarial review）

**F1 · 阻断 · 交付**：成果全在本地分支 `agent/m17500`（68f1622c），**未推送、未合并**。
→ 主干 CI 永远测不到、任何按 main 复现的读者都拿不到文件。复现：
`git -C /root/tp-agent/work/m17588 merge-base --is-ancestor agent/m17500 HEAD; echo $?` → 1；
`git ls-remote --heads origin agent/m17500` → 空。修复归属=返修单/推送环节，非本审计可代做。

**F2 · 阻断 · 裁判**：`tensor_benchmark_cases` 无该锚点行 → 平台无法机器判"容差内"。
锚点名为 `-NEGATIVE`，建议登记时把 metric 写成"复现负结论的 got 值带"，tolerance 语义另立（否则负结论
在"误差<15% 为过"的裁判表里天然判死）。

**F3 · 重要 · 科学性**：参考门 FAIL（体相路线 vs Jop 2006 常数 22.9% / 33.0% / 24.0%），
其 prereg 明写"参考合格门失败 → 绝对摩擦比较作废，只留无量纲映射结论"。报告 ③/⑥ 仍把 `mu_error 35.8%`
列为"裁决 FAIL"的结果行 —— **自相矛盾**（应标注"作废（void）"）。无量纲映射结论（I_c/I_bulk=2.54/0.576/
0.407）不受影响，仍是有效的负结论。

**F4 · 重要 · 有效性 + 文案不符**：5 行中 **3 行** `servo_valid=False`
（`pressure_control_error_final` = **−5.3% / −7.4% / −19.3%**，`base_I0.01/base_I0.1/sens_kn4e3`）。
被审文案（报告 ⑤ 与 mission summary）写"**4/5** 算例 servo_valid=False（末窗压力偏差 **−7.4%…**）"——
**行数多算 1 行、量级少报 12 个百分点**（真值最差 −19.3%，疑把 `rows_capped_by_proxy=4` 混作 invalid 行数）。
"同一次稳态"的说法因此未获认证。

**F5 · 次要 · 已被审计关闭**：`积分器逐位不变` 原只有结构性断言（`test_device_diagnostic_is_opt_in`
仅查 dataclass 默认 False）。审计探针实跑同算例 off/on：`dt`、`pressure_control_error_final`、
`shear_rate_measured`、`mu_wall_measured`、`normal_stress_measured_pa`、`coordination`、`phi`、
`mu_full_over/wall` **8/8 位级相同**，`contact_proxy` 键关时不存在、开时存在 →
`artifacts/audit_m17588/optin_bitwise_probe.json`。**该断言成立**（审计端补证，非原单自证）。

**F7 · 次要 · 数字口径**：报告 ③/摘要写 `I_c/I_bulk = 2.54`，由工件复算为
`i_proxy_fn_weighted/i_bulk_measured = 0.024108/0.009544 = **2.526**`（差 0.5%）。其余两值 0.576/0.407 正确；
建议统一由工件注入避免手抄滑移。

**F6 · 次要 · 数据卫生**：`results.json` 内有裸 `NaN`（`n_contacts`、`kappa`）→ 非严格 JSON，
严格解析器（非 Python）会崩；建议 `null`。

---

### 被审文案的过度声明（overstated claims）清单

| 文案 | 工件真值 | 判定 |
|---|---|---|
| "4/5 算例 servo_valid=False" | 3/5（`servo_valid=false` 行=base_I0.01/base_I0.1/sens_kn4e3） | 过度 |
| "末窗压力偏差 −7.4%…" | 最差 −19.3%（sens_kn4e3） | 少报 |
| "I_c/I_bulk = 2.54" | 2.526 | 轻微滑移 |
| "mu 误差最差 35.8% → 裁决 FAIL 结果" | 参考门 FAIL ⇒ 按其 prereg 绝对摩擦比较**作废** | 口径矛盾（F3） |
| "积分器与库仑帽逐位不变" | 审计探针 8/8 位级相同 | **成立**（F5，审计补证） |
| "8 passed 单测" | 审计端重跑 8 passed in 5.15 s | 成立 |

## 4. 独立性声明与局限

- 审计者未参与 m17500；也未修改被审工作树（`git status --porcelain` 干净），仅只读 + 重跑。
- 独立性局限：数字来源单一（同一 `results.json`），无法对 DEM 轨迹本身做第三方重算；
  F5 的位级检查用的是**缩短的算例**（settle 20 + 20 步，≈30 s/run），不是原 2782–11619 步轨迹。
- 未经审计者裁决的部分：物理结论的对错（proxy 是否真不能替代体相 I）——审计只核证据链与自洽性。

## 5. 未做的检查（unknown / not_run，不等于 pass）

1. **全量 5 算例 346 s 再生**：未重跑（shell 单次预算所限）；因此"数值可复现到同样小数位"未证，
   只证了"单测可跑通 + 算术自洽 + 报告与工件一致"。
2. 3 个 `servo_valid=False` 算例的压力控制认证。
3. 集群架构（loong64）上的一致性未验（本审计在 99 控制面 x86 环境执行）。

## 6. 处置（disposition）

- **级别裁定 L2**（水平如实，不虚标）；判决书 id=593 已落表，`gate_state=pending` 交 maturity_gate
  机器复核（本地预演 `judge(2, evidence, 'TensorDEM', 17588)` → `gate_level=2, reasons=[]`，即无缺项、
  不触发返修）。
- **建议（不阻断 L2）**：①把 68f1622c 推送/合并进 main（否则本单功能实存但产品不可见）；
  ②按 `-NEGATIVE` 语义补登 benchmark case；③报告 ③/⑥ 把 mu 误差标为 void，主结论收敛到无量纲映射。
- 本审计单的产物：`artifacts/audit_m17588/`（证据 JSON×3 + 图 + results 副本）、
  `scripts/audit/m17588_*.py`（可复跑审计器）。

![m17500 audit](../artifacts/audit_m17588/fig_m17500_audit.png)