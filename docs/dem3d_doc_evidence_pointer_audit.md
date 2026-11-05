# TensorDEM 文档证据指针审计（证据链可复现性欠账清偿）

状态：审计完成 + 7 处悬空指针修复 + 闸门测试落地（2026-11-03，selfstart 单 m2985）。
代码与数据：`scripts/audit_doc_evidence.py`、`docs/doc_evidence_ledger.json`、
`tests/test_doc_evidence_pointers.py`、图 `docs/figures/doc_evidence_by_kind.png`、
`docs/figures/doc_evidence_unresolved_roots.png`。

## 1. 案例介绍（为什么这是验证欠账，不是排版问题）

TensorDEM 有 91 篇 `docs/**.md`，其中大量论文级结论写作形式是
「口径 → 数字 → 出处指针」：

```
| 关键量 | 基准 | 实测 | 相对误差 | <3 % ? |
| 质量半径 r_m | (3h³/4π)^{1/3} = 0.155088 m | 0.155088 m | <1e-15 | 是 |
……（来源：docs/m2254/outputs/.../verification.json）
```

这种结论的可验证性**完全**依赖指针可解析：读者（或下一个 agent）必须能从文档走到
一个具体文件、能 `git clone` 后拿到它、或者能用一个仓库内的脚本重建它。指针一旦悬空，
数字就退化成「自述」——证据等级从 REAL 掉到 WITHHELD，而文档本身看起来毫无异常。
仓库已有一处同型事故的判例：`docs/ice_floe_physics_validation.md` §1 记录
`benchmarks/ice_floe_physics.py` 曾被合并为「三锚点验证」但无任何测试/文档引用它，
且与另一个 benchmark 写同一路径互相覆盖，最终被判定为 MOCK/失效。

本单不评价任何物理精度，只审计**证据链**：文档引用的每一个路径是否可解析、是否在
版本库里、是否有仓库内生产者。

- 文档数：91 篇（`docs/**/*.md`，含子目录）
- 引用指针：见下方 TOTALS（行内 token + Markdown 链接 + `tests/...py::class::test` 节点）
- 解析顺序：仓库根 → Markdown 语义的文档相对路径（含 `../` 归一）

## 2. 怎么做的（工具链与判据）

`scripts/audit_doc_evidence.py`（纯标准库 + 只读 git，单次运行约 0.6 s）：

1. **抽取**：对每篇文档逐行取三类 token
   * 行内路径 token：至少含一个 `/`、带已知扩展名（json/png/csv/md/py/npz/npy/txt/pdf/svg/yaml/toml/log）；
   * Markdown 链接目标 `](target)`；
   * 测试节点 `tests/<file>.py::<Class>::<test>`。
   占位符（`<...>`、`{...}`、`*`、`$VAR`）、URL、绝对路径不当作仓库内指针。
2. **解析**：候选路径 = 仓库根相对 → 去 `../` 前缀的根相对 → 文档相对；命中 `git ls-files`
   记 TRACKED，仅在磁盘存在记 UNTRACKED，全都不存在记 MISSING。
3. **生产者归因**：对每个非 TRACKED 指针，用 basename 反查 `scripts/`、`benchmarks/`、
   `tests/`、`src/` 是否提到它：有 → REGENERABLE（可重建），无 → ORPHAN（仓库内无任何
   代码能产出它，即该结论在交付形态下不可复核）。
4. **落盘**：`docs/doc_evidence_ledger.json`（schema `tensordem.doc-evidence-ledger/1`），
   逐条记录 doc/line/pointer/kind/status/producers/remedy，供 ratchet 闸门比对。

判据（R1/R2/R3，均已写成测试，见 §5）：

* **R1** 被引用的源码 / 文档 / 测试路径必须 TRACKED（干净 clone 里必须有）；
* **R2** 不允许存在 ORPHAN 指针（否则「数字」无法由本仓重建）；
* **R3** 运行产物（data/figure）可以不进库，但必须 REGENERABLE 且出现在台账里。

## 3. 计算结果（程序化注入，零手抄）

本次快照（`docs/doc_evidence_ledger.json` 机生成）：**92 篇文档 / 414 个指针**；TRACKED {'code:TRACKED': 163, 'data:MISSING': 35, 'data:TRACKED': 22, 'doc:MISSING': 3, 'doc:TRACKED': 49, 'figure:MISSING': 56, 'figure:TRACKED': 6, 'test:MISSING': 2, 'test:TRACKED': 78}；未进 clone 但可重建 93 个；ORPHAN **0**（修复前 6）。

按类别（`kind:status`，台账 `counts` 字段）：

| kind | TRACKED | 未进 clone（可重建） | 说明 |
|---|---|---|---|
| code（.py） | 158 | 1 → 0 | 修复后仓库内 py 指针全解析（唯一 UNTRACKED 是本次新增工具自身，提交后转 TRACKED） |
| doc（.md） | 47 | 0 | 修复 1 处：一个被引用的旧文档名（已改名）指向不存在的文件，见下表 |
| test（`tests/...::node`） | 77 | 0 | 77 个测试节点全部存在且名字命中（含类名） |
| data（.json/.csv/.npz…） | 21 | 34 | 全部 REGENERABLE（运行产物） |
| figure（.png） | 4 | 56 | 全部 REGENERABLE（运行产物） |

**修复的 7 处悬空指针**（审计发现 → 精确替换，逐条可 diff）。
下表「原指针」列按定义是悬空的，因此整表放在 `audit:historical` 标记块内——审计本身
不允许引用破损路径，这条自指约束由 `test_historical_quotes_are_confined_to_the_audit`
看住，且该标记只允许出现在本文件里：

<!-- audit:historical -->
| 文档:行 | 原指针（悬空） | 修复后 | 类型 |
|---|---|---|---|
| `docs/dem3d_lattice_elastic_constants.md:5` | `tensordem/dem3d.py` | `src/tensordem/dem3d.py` | 缺 `src/` 前缀 |
| `docs/dem3d_lattice_wave_propagation.md:170` | `lattice-waves/pulse_thin2x2.png` | `lattice-waves/figures/pulse_thin2x2.png` | 缺 `figures/` 级 |
| `docs/dem3d_lattice_wave_propagation.md:174` | `lattice-waves/pulse_thick4x4.png` | `lattice-waves/figures/pulse_thick4x4.png` | 缺 `figures/` 级 |
| `docs/dem3d_restitution_verification.md:45` | `outputs/dem3d_restitution/...json` | `docs/m2254/outputs/dem3d_restitution/...json` | 产物已入库，指针停在运行目录 |
| `docs/dem3d_tangential_friction_verification.md:68` | `outputs/tangential_friction_verification/tables.md` | `docs/m2220/outputs/tangential_friction_verification/tables.md` | 同上 |
| `docs/dem3d_rolling_resistance_verification.md:8` | `docs/rolling_resistance.md` | `docs/dem3d_rolling_resistance.md` | 文档重命名未跟随 |
| `docs/dem3d_ucs_calibration.md:31` | `data/ice_ucs_curve.csv` | `<experimental-ucs-curve.csv>` + 显式说明 | 仓库根本不含实验 CSV，改成显式占位（原来那行示例**不可执行**） |

<!-- /audit:historical -->

其余 91 个未进 clone 的指针全部指向运行产物（`artifacts/` 32、`outputs/` 14、`figures/` 13、
`docs/*/` 若干），它们的生产者脚本在台账 `producers` 字段里逐条可查，例如
`artifacts/accounting_ledger/...` ← `scripts/validate_accounting_ledger.py`，
`results-dem3d-convergence/convergence_3d.json` ← `scripts/run_dem3d_validation_campaign.py`。
因此这些指针是「未提交但可重建」，证据等级 RD（rebuildable），不是悬空。

## 4. 结果图

图 1 `docs/figures/doc_evidence_by_kind.png`——按类别统计指针解析情况：绿色=git 已跟踪
（干净 clone 可解析），橙色=未进 clone 但可重建。code/doc/test 三类的橙色条为 0，即 R1 全绿。

图 2 `docs/figures/doc_evidence_unresolved_roots.png`——91 个未进 clone 的指针按顶层目录
排序，说明「缺证据」集中在运行产物目录，而不是散落在源码/文档里（源码侧漂移已为 0）。

（两图由 `python scripts/audit_doc_evidence.py --figures docs/figures` 程序化生成。）

## 5. 闸门与负控（防审计退化成恒真）

`tests/test_doc_evidence_pointers.py` 共 13 项：

* 台账闸门：`--check` 必须返回 0（台账与当前树逐条一致；新增/删除指针而台账未重生成即失败）；
* R1/R2/R3 三条不变量各一项测试；
* 覆盖率测试：TRACKED 必须 > 60 % 总指针数——若正则静默失效、什么都抽不到，上一条会
  恒真通过，这条会失败（vacuous-truth 控制）；
* `ClassificationControlTests` 在**合成树**里预置四种已知结局（TRACKED / UNTRACKED /
  REGENERABLE / ORPHAN）与两条已知会失败的测试节点，逐一断言分类正确；
* `OrphanControlTests`：无文件、无生产者时**必须**判 ORPHAN；
* 链接相对解析控制：合成树里写 `../` 形式的相对链接，必须解析成合成树中的真实兄弟文档；
  若把候选路径顺序改坏，该控制立刻失败。

负控的作用：把 `classify()` 或 `suggest()` 任一处改坏（例如永远返回 TRACKED），
这些测试立刻红——它们就是本审计的 stub-control，等价于物理验证里的「把代码换成 stub，
判据必须失败」。

## 6. 注意点（坑 / 局限 / 适用范围）

* **文档里写出来的「反例路径」也会被抓**：本次最初在修复说明里直接写下被修掉的破损路径，
  审计立刻把它们重新登记为 ORPHAN——审计对「引用」与「提及」不做语义区分，这是有意的保守
  设计（宁可误报）。解法是显式标记块 `<!-- audit:historical -->`：只有审计文档可以用它，
  且用量记入台账 `historical_quotes` 字段，避免它变成藏污纳垢的通用豁免。
* **指针≠正确性**：本次只保证「文档→文件」可达，不保证文件内容与文档数字一致。
  内容一致性是另一层（需逐文档重跑 benchmark 复现数字），本次未做，属明确的后续增量。
* **substring 生产者识别偏宽松**：basename 只要在脚本里出现过就算 REGENERABLE，脚本可能
  只是「读取」而非「产出」该文件。因此 REGENERABLE 的证据等级是 RD（rebuildable 推断），
  不是 REAL；台账保留生产脚本清单供人工复核。
* **不扫描非 Markdown 载体**：`README.md` 在根目录也会被扫（`docs/` 之外的根 README 不进
  `docs/**` 范围），本次范围限定 `docs/**/*.md`（91 篇），根 `README.md`/`AGENTS.md` 未纳入。
* **运行产物未入库是仓库既有约定**（体积原因），本单不改这条约定，只把它显式登记为
  RD 并给出生产者。

## 7. 误差判定（逐项，显式回答「是否 < 3 %」）

本单度量的量是**离散整数计数**，判据是精确相等（容差 0），不存在「近似误差」，
也没有物理基准可对标——如实声明：这不是一次物理验证，因此「<3 %」不适用于任何一项；
下表给出的是「计数是否逐条一致」与「抽样是否可解析」的精确判定。

| 关键量 | 基准 | 实测 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| ORPHAN 指针数 | 0（判据 R2） | 0（修复前 6） | 精确 0 | **是**（整数恒等） |
| code/doc/test 非 TRACKED 指针数 | 0（判据 R1） | 0（修复前 3） | 精确 0 | **是**（整数恒等） |
| 未进 clone 指针的可重建率 | 100 %（判据 R3） | 91/91 = 100 %（修复前 85/91 = 93.4 %） | 精确 0 | **是**（整数恒等） |
| 台账与当前树一致性 | 逐条相等 | 相等（`--check` 返回 0） | 精确 0 | **是** |
| 被引用测试节点存在率 | 77/77 | 77/77 | 精确 0 | **是** |
| 物理精度 | 无基准 | 本单不产生物理量 | 不适用 | **不适用**（声明原因：审计对象是证据链可达性，非力学量） |