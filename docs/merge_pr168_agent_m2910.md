# PR #168（agent/m2910）→ main 三方合并记录

m3931 治愈单。PR #168 在 #137 之后变脏：`main` 先后合入 #138…#203，其中 m2185 /
m2935 / m3317 已把 m2910 的 `fracture_gen` 契约修复以更细化的版本重新实现，于是同一
契约在两侧各有一份实现，形成 17 处文本冲突。本次按「语义并集」解冲：两侧的行为契约
都保留，靠机器可核的 API 集合差与测试套件证明无静默丢失。

## 1. 冲突文件清单与逐文件解法

| 文件 | 冲突块 | 解法 | 保留下来的两侧内容 |
|---|---|---|---|
| `src/tensordem/fracture_gen.py` | 17 | 以 main 实现为骨架（其后继版本含 `cdist` 分块建键、`normalise_mass_mapping`/`normalise_face_extent` 归一化、`mass_accounting(continuum_volume=, tolerance=)`、`_lattice_volume`、`plot` 侧键），再把 m2910 独有的两处 API 并回 | main：全部函数（`face_extent` 属性、`_lattice_volume`、`mass_mapping_report`、块化 `build_bond_topology`）；m2910：`FractureLedger.lattice_volume()`（公开拼写，`docs/fracture_gen_contract_restoration.md` §2 引用）、`mass_accounting()["absolute_relative_error"]`（同值别名，m2910 的 `scripts/fracture_gen_contract_report.py` 与 `tests/test_fracture_gen_audit.py` 读它） |
| `tests/test_fracture_gen_audit.py` | 1 | 并集：同时断言 `relative_error`（带符号）、`relative_error_magnitude`（main/`benchmarks/fracture_gen_audit.py` 读）与 `absolute_relative_error`（m2910 读） | 两侧断言语义完全相同，仅键名不同，故三个键都断言 |
| `tests/test_fracture_gen_mass_footprint.py` | 2 | 冲突块两侧只是同一映射的两种拼写（`"sphere"` vs `"equal_volume_spheres"`），别名表同时接受 → 取 `"sphere"` 并在注释中保留 m2910 的拼写与文档出处；m2910 新增的 `test_default_mapping_is_volume_consistent` 保留，但断言从「默认 `mass_mapping` 字符串等于 `cell_volume`」改为「默认 `mass_mode == "cell"` + 零残差 + 描述性拼写被接受且同质量」 | main：`mass_mapping="sphere"` 的冻结映射审计；m2910：默认映射体量一致性的测试意图与 §4 文档指针 |
| `benchmarks/fracture_gen_mass_footprint_audit.py` | 1 | 取 main 版本（含 m2910 没有的 `plot_footprint_field_map` 三面板图函数）；冲突注释中保留 m2910 的拼写说明 | main：`plot_footprint_field_map`；m2910：冻结映射别名可读性 |

非冲突自动合入（m2910 独有新增，全部保留）：`docs/fracture_gen_contract_restoration.md`、
`scripts/fracture_gen_contract_report.py`、`benchmarks/fracture_gen_audit_report.json` 的两侧
数值更新。

## 2. 无静默丢失的机器核验

对四个冲突文件，取「main / m2910 / 合并结果」三份的函数定义集合做差集：

| 文件 | main 独有且合并后缺失 | m2910 独有且合并后缺失 | 合并结果新增 |
|---|---|---|---|
| `src/tensordem/fracture_gen.py` | 无 | 无 | `lattice_volume` |
| `tests/test_fracture_gen_audit.py` | 无 | 无 | — |
| `tests/test_fracture_gen_mass_footprint.py` | 无 | 无 | `test_default_mapping_is_volume_consistent` |
| `benchmarks/fracture_gen_mass_footprint_audit.py` | 无 | 无 | — |

（`comm` 双向差集，main 侧 `plot_footprint_field_map`、`_lattice_volume`、
`normalise_*`、`face_extent` 属性均在合并结果中。）

### 2.1 唯一一处实现分歧：footprint 判据用「键中点」还是「键与平面交点」

main 用键中点、m2910 用平面交点投影来判「是否落在面 footprint 圆盘内」。二者只在
非垂直入射的键上可能不同；审计格点是简单立方 6 邻位，所有跨面键都垂直于面，两点在
面内投影恒等，故计数完全一致（复算，n=8 格点、R=0.1 m、面在两层之间）：

| R [m] | 键中点判据（main，保留） | 平面交点判据（m2910） | 解析 πR²/h² |
|---|---|---|---|
| 0.20 | 4 | 4 | 3.14 |
| 0.40 | 12 | 12 | 12.57 |
| 0.60 | 32 | 32 | 28.27 |
| 0.70 | 32 | 32 | 38.48 |

两种判据在已发表的审计数字上不可区分，故保留 main 的实现（被主干测试覆盖），并在
本节留证该等价性，而不是引入第二条未经测试的代码路径。

## 3. 测试证据

| 套件 | main 基线 | 合并结果 |
|---|---|---|
| 全量 `pytest tests`（1235 项 + 152 subtests） | 1225 passed, 3 skipped, 6 errors（`test_doc_evidence_pointers` 因基线树是 `git archive` 快照、无 git 元数据而报错） | **1232 passed, 2 skipped, 1 failed** |
| `tests/test_fracture_gen_audit.py` + `tests/test_fracture_gen_mass_footprint.py` + `tests/test_fracture_gen_footprint_breakage.py` | 全部通过 | 50 passed |
| `scripts/fracture_gen_contract_report.py`（m2910 交付脚本，在合并后 API 上） | 不存在于 main | 正常产出 `artifacts/fracture_gen_contract_restoration/`（含 3 张图），质量表 `cell_volume` 残差 0.000 %、`equal_volume_spheres` −47.64 % |

唯一失败项 `test_doc_evidence_pointers.py::test_committed_ledger_is_current` 是 m2910 新
文档引入的指针未登记到 `docs/doc_evidence_ledger.json`：本次已用仓库自带生成器
`scripts/audit_doc_evidence.py` 重算账本并一并提交，复跑该文件 14 项全绿。合并后合并
结果不低于 main 基线。

## 4. 注意点与局限

- 合并保留了 main 的默认映射报告拼写（`mass_mode` 规范名，除非调用方显式给了描述性
  拼写），因此 m2910 原测试里「默认 `mass_mapping == "cell_volume"`」的字面断言被改写为
  物理等价断言；这是拼写口径差异，不是数值差异（默认映射残差仍为 0）。
- footprint 判据的实现分歧在非垂直入射键（未来若引入非立方格点或倾斜面）会显形，
  届时需按 §2.1 的解析 πR²/h² 与交点评分重新定夺，本记录只证明在当前审计配置下等价。
- `artifacts/fracture_gen_contract_restoration/` 由 `scripts/fracture_gen_contract_report.py`
  程序化生成（数值零手抄），可直接复算。
## 5. 复验周期内的第二次合流（main 前进到 #166 治愈后）

首次解冲用的 main 基线是 `e12a2ab`。合并完成后 `main` 又前进了两个提交
（`806b96f` 治愈 PR #166 / m2942 的同类语义并集 + `6c6f98b` merge-governor 记账），
其中 #166 也改到同一个 `src/tensordem/fracture_gen.py`（`MASS_RADIUS_FACTOR` 加类型
注解、`MASS_RADIUS_FACTOR`/`face_footprint_radius` 单一定义处的注释折叠）。本次把
`6c6f98b` 再并入：`fracture_gen.py` 自动合并且与本节 §1 的解冲结果一致（两侧对
`(6/π)^(1/3) == 2(3/(4π))^(1/3)` 为同一常量的判断相同），唯一冲突是机器生成的
`docs/doc_evidence_ledger.json`，用仓库自带生成器 `scripts/audit_doc_evidence.py`
重算解决（账本不手改）。复验：`tests/test_doc_evidence_pointers.py` 14 passed、
`test_fracture_gen_audit.py`+`test_fracture_gen_mass_footprint.py`+
`test_fracture_gen_footprint_breakage.py` 50 passed。
