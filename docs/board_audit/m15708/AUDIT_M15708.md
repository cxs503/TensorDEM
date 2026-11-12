# 交差审计 · m15708 审 m15649（TensorDEM / CASE-ROTATING-DRUM 第 2 轮）

**审计对象**：m15649（repo=TensorDEM，commit `074148fc`，分支 `agent/m15649`，status=succeeded，
finished_at 2026-11-12T02:00:00Z）　**审计单**：m15708　**基准裁判**：
`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-ROTATING-DRUM'`
→ **0 行**（该锚点在机读判据表里根本没有登记行）。

**审计者独立性说明**：审计者为另一 worktree（m15708，分支 `agent/m15708`）上的独立执行体，
只读 m15649 工作树，未参与其实现；所有数字由脚本从交付工件复算（`verify_r2.py`），不誊抄。

---

## 0. 复算与独立执行（做了什么）

| 检查 | 方法 | 结果 |
|---|---|---|
| C1 基准角复算 | 由冻结闭式 `θ_d = arccos(-Fr)`（Fr≥1 饱和 180°）重算，与 12 个 case JSON 的 `theta_ref_deg` 逐项比对 | **全部一致**（≤1e-6） |
| C2 误差复算 | 由 ref/got 重算相对误差，与 `theta_rel_err` 比对 | **全部一致**（≤1e-9） |
| C3 表 vs JSON | 解析 `artifacts/case_rotating_drum_r2/table_main.md` 9 行，与 per-case JSON 比对 | **9/9 一致，0 处不符**（无手改表） |
| C4 判据适用性 | 按 prereg `acceptance.quantitative`（仅 Fr<1 有限释放角）分类 | 7 个有限角算例 **7/7 <20%**；Fr=1.0/1.1 两行为饱和引用 |
| C5 dt 敏感性 | dt_base vs dt_half（同 Fr=0.3、同 6 圈） | 9.09% → 1.12%（**×0.124**） |
| C6 rev20/平台 | 盘上 case 文件 vs `merged.json`/`summary.json` | rev20 **盘上有、merged 无**；`plateau={}` |
| C7 报告六段 | 扫 `docs/CASE_ROTATING_DRUM_R2_FROUDE.md` 章节 | **仅 ①②③④，缺 ⑤ ⑥** |
| C8 图件 | PNG 魔数+字节 | fig1 64 047 B / fig2 88 599 B 均合法；**fig3 缺**（审计重跑后生成） |
| C9 测试实跑 | `PYTHONPATH=src pytest -q tests/test_drum_froude_r2.py`（在 `runs/m15708/repro` 独立副本） | **5 passed / 1 failed**（确定性，重跑 2 次同结果） |
| C9b repro_cmd 实跑 | `python benchmarks/rotating_drum_froude_r2.py --fr 0.7 --revs 10` | 93.9 s，θ=136.9°（ref 134.4，1.81%），r_p05=0.35，cataracting，`resting_ok=True`，**与交付 JSON 完全一致** |
| C9c 报告脚本实跑 | `python benchmarks/rotating_drum_froude_r2_report.py` | 重生成 table/fig；table 与交付**逐字节相同**；fig3 补出 |

复算产物：`docs/board_audit/m15708/verdict.json`；复现副本：`runs/m15708/repro/`；
交付物快照（含 sha256_16）：`docs/board_audit/m15708/reviewed/`。

---

## 1. 发现（按严重度）

### B1（阻断·L4 不可成立）机读基准行缺席
`tensor_benchmark_cases` 在 `anchor_slug='CASE-ROTATING-DRUM'` 下 **0 行**（本审计单指定的
唯一裁判）。因此本锚点的 "benchmark ref vs got" 没有平台侧机读判据可核；任务的
"容差：定性+<20%" 只存在于任务 goal 与 prereg 里。→ 基准维证据按 **WITHHELD** 记，不按通过记。

### B2（阻断）交付的测试 1/6 失败（交付方自陈"未执行 pytest"，被独立复现证实）
`tests/test_drum_froude_r2.py::test_centrifuging_at_froude_above_one_is_measured_not_asserted`
断言 `r_p05_over_R > 0.75`，实测 **0.31808972654624623** → AssertionError。local 重跑两次同值
（simulate 确定性，seed=0）。即：该测试所表达的"Fr>1 载荷被钉壁成环"在本实现里**不成立**。

### B3（重大）"9/9 <20% PASS" 中 2 行用的是饱和/退化引用
prereg 明写 `fr_limit: Fr >= 1 -> theta_d = 180 deg (load pinned, no release point)`，
且 `acceptance.quantitative` 限定 "at every **Fr < 1 with a finite release angle**"。
报告表仍把 Fr=1.0（err 18.53%）与 Fr=1.1（15.0%）标 `PASS`。**有效结论应为 7/7 通过、
最大误差 11.32%（Fr=0.05）**；两行应标 N/A（无释放角），而非 PASS。18.53% 已贴 20% 线，
且该行误差多半来自离散化（见 B4），不能当作"接近理论"的证据。

### B4（重大）生产 dt 未证明收敛，误差由离散化支配
Fr=0.3、6 圈、dt=1e-4：err = **9.088%**；dt=5e-5：err = **1.123%**（比值 ×0.124）。
主算例 7 个有限角算例的误差区间 1.8%–11.3%，其量级与 dt 误差同阶 → **"θ 与精确闭式差 7–11%"
主要是数值误差，不是物理偏差**；反过来说，dt=1e-4 的生产算例并不能支撑"<5% 目标"，
也没有在主算例上做 dt 收敛研究（唯一 dt 对照在 Fr=0.3、6 圈，且圈数短于生产 10 圈）。

### B5（重大）P5（两转数平台）未交付，且其真实结果否定"平台"
rev20 算例（Fr=0.1，20 圈）**盘上存在**（`artifacts/.../cases/case_rev20.json`，err 13.32%，
wall 557 s，未 tracked），但 `merged.json`/`summary.json` 未纳入，`summary.plateau = {}`。
审计用交付物重跑报告脚本后得到：`S_rev5=0.5109, S_rev10=0.2927, S_rev20=0.5007`（**非单调**），
并补出 fig3。即 prereg 的 P5「短窗口不算稳态证据」在交付里既没证明，其数据反而显示
**无稳态平台**（10→20 圈释放角误差 8.80%→13.32% 亦非收敛方向）。此项应作 **failed prediction** 如实登记。

### B6（重大）离心化起点验收项未判定，却以 null/NaN 静默呈现
`summary.centrifuging_onset_measured = null`、`rel_err = NaN`；Fr=1.0/1.1 实测流态
`cataracting`（r_p05 0.37/0.38 ≪ 0.75 冻结判据），即按冻结分类器**未测得离心化**。
prereg 的该项验收（"onset within 20% of Fr=1"）实为 **未达成/未判定**，报告与 summary 均未明示 FAIL。

### B7（重大）报告六段不齐
交付文档只有 ①问题设定 ②方法 ③参数表 ④对照表；**缺 ⑤收敛性/参数敏感性、⑥失败模式与工件**。
owner 报告纪律要求六段齐全，且 ⑤ 恰好是 B4/B5 的地方，⑥ 恰好是 repro_cmd 与工件索引。

### B8（次要）冻结几何被跑后修订，敏感性子集偏离冻结清单
`prereg.amendment_1`（02:25Z）把 `d_s` 0.006→0.010 m（理由如实：6 mm 床 20 轮淬火内达不到休止判据）。
阈值/锚点未变，属合法修订；但后果是：冻结清单里的 `rev10coarse`（d_s=0.01, 10 圈）**未跑**，
而主算例与冻结值（0.006）不同 → 报告 "frozen_before_runs: true" 的措辞需配合修订说明读。

### B9（次要）`oss_solutions` 未落库
交付方在 result.checks 自陈未落；调研文档 §③ 只给了字段草案。跨任务复用回路（第 2 轮"沉淀"目标）缺口。

### B10（次要·诚实项）量化基准是单颗粒闭式，非实测床形角
`θ_d = arccos(-Fr)` 是单颗粒径向力平衡的解析根，不含床面/填充分数/粒径效应；作为锚点合法，
但"与文献/实验对照"的强度有限。调研文档已明写"零手抄 = 脚本内计算闭式，不是查表"，**属诚实披露**。

---

## 2. 通过核验的部分（不能因上面否定）

* 数字可复算、表与 JSON 零偏差（C1–C3）；`repro_cmd` 真能跑，且 **新跑 fr=0.7 十圈与交付值逐位一致**（C9b）→ 交付**不是**伪造/手抄。
* `resting_ok` 9/9 True、`settle_rms_max=3.92e-3 < 5e-3`：前轮 m15329 的休止闸门致死点**已修好**，且是测量不是断言。
* 方向预测 P1/P2/P3 Spearman = 0.633 / 0.983 / 0.800，**符号全部如预期**。
* 实现是真代码（`kernel2d._contact_forces` 逐粒子接触计数 + 运动学观测量采样 + `regimes.classify_measured`），非占位。

---

## 3. 判级（L0–L5）

**L2 独立功能实现**。理由：真实现 + 入口 + 运行工件齐（impl/entry/run_artifacts 全过），
但 **未达 L3**（`kernel2d` 只被 benchmark/script 调用，库内无主求解器/平台侧集成调用；
`drum_flow.regimes` 的调用方也全是 benchmarks/scripts），
**未达 L4**（机读基准行缺席 B1；主算例 dt 未收敛 B4；2 行饱和引用 B3；P5 未交付 B5；
报告缺 ⑤⑥ B7；交付测试 1/6 失败 B2）。

## 4. 处置建议（disposition）

**request changes**（可确认 L2，返修项=Benchmark/收敛面）：
1. 去 `tensor_benchmark_cases` 为本锚点补登记行（case_name/metric/tolerance），令基准可机读；
2. 把 Fr=1.0/1.1 从定量判据中摘出（标 N/A），报告改为 "7/7 <20%，max 11.32%"；
3. 主算例至少一个 Fr 做 dt 收敛 + 20 圈稳态判定，写进 ⑤；
4. 修 `test_centrifuging_at_froude_above_one_*`（要么提高 Fr/圈数使载荷真钉壁，要么把断言改成冻结分类器能过的事实）；
5. 把 rev20 纳入 merged/summary，P5 按 failed/hold 如实写进 ⑥；
6. `oss_solutions` 落库 `feature_slug='CASE-ROTATING-DRUM'`。

## 5. 未执行的检查（unknown，不记 pass）

* 3D 验证（本内核是 2D 切片；3D 类比未做，属方法边界不是缺陷）。
* 与**实测**滚筒实验（如 Parker 1997 / Nakagawa 数据）对照：交付未做，审计亦未做（无数据源在仓内）。
* 集群节点（loong64）上的重跑：本审计只在控制面（99）复跑单算例与测试，未在集群复算 11 算例全集。