# PR #166 (branch `agent/m2942`) — main 变脏的三方合并与逐文件解冲

**日期**：2026-11-05
**任务**：TensorDEM #166（治愈单 m3915）
**仓库**：`TensorDEM`
**旧 SHA**：`agent/m2942 = 4b54f9bbb037adc092bcdcba7bdca94f6cc0e41e`
**main（并发基线）**：`e12a2abd636ab5a2998054dbe0fcf3c18b6a8564`
**merge-base**：`8cc88f7f1bc681cb028269014e61c1fc265d9d2e`

## 1. 为什么会脏

`git merge-tree --write-tree origin/main origin/agent/m2942` 报 `mergeable=false`，冲突面只有一个文件：

```
src/tensordem/fracture_gen.py
CONFLICT (content): Merge conflict in src/tensordem/fracture_gen.py
```

两侧自 merge-base 起的改动规模极不对称：

| 侧 | 文件数 | 内容 |
|---|---|---|
| `main` | 92 文件（+21912/−430） | DEM 验证波次成果：`contact_invariants`、`shear_convergence`、`threefield/lbm_transport`、`fracture_gen` 的 **完整重写**（mass mapping / footprint / on-plane 三套判据 + 别名归一化），以及对应 docs/scripts/tests |
| `agent/m2942` | 3 文件（+67/−1） | `fracture_gen.py` 补 `MASS_RADIUS_FACTOR` + `face_footprint_radius` + `extent_radius` 字段；`docs/dem_analytic_verification_debt_restoration.md` 追加 §6；根 `report.json` |

**冲突的实质**：m2942 是一次「只为让测试能 collect」的最小补救——它手工补了两个符号；
而 main 后来独立地把同一功能做成了完整实现（包括同名符号、同名 dataclass 字段）。
即：**两侧是同一意图的重复引入，不是互斥改动**。

## 2. 逐文件解法

### 2.1 `src/tensordem/fracture_gen.py`（唯一真冲突，UU）

自动合并后的文件里出现 **重复定义**（机械 overlay 的典型后果）：

```
 40: MASS_RADIUS_FACTOR = 2.0 * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)    # main
117: MASS_RADIUS_FACTOR: float = (6.0 / math.pi) ** (1.0 / 3.0)           # m2942
120: def face_footprint_radius(...)                                        # m2942（定义在 FailureFace 之前）
434: def face_footprint_radius(...)                                        # main（定义在 FailureFace 之后）
```

同时 `FailureFace.extent_radius` 字段本身两侧一致（m2942 新增 / main 新增），
只剩字段注释一处 3-way 冲突标记。

解法（**语义并集，逐条对齐，无一侧内容被静默丢弃**）：

| 冲突点 | main | m2942 | 取用 | 理由 |
|---|---|---|---|---|
| `MASS_RADIUS_FACTOR` | `2(3/4π)^{1/3}`，无类型标注 | `(6/π)^{1/3}: float` | 单一定义＝main 表达式 **＋ m2942 的 `: float` 标注** | 两式为恒等式 `2(3/4π)^{1/3} = (8·3/4π)^{1/3} = (6/π)^{1/3} = 1.2407009817988002`（程序验证差 0.0）。保留带文档推导的 main 位置，吸收 m2942 的类型标注 |
| `face_footprint_radius` | 定义在 `FailureFace` 之后，校验 `positive and finite` | 定义在 `FailureFace` 之前，校验 `non-negative` | **保留 main 的单一定义**；删除 m2942 的前置副本，并在原位留下显式注释指针 | 语义等价（有 `extent_radius` 用之，否则 `sqrt(area/π)`）。main 版是严格超集：额外拒绝非有限值与 `extent_radius==0`（零足迹半径切不断任何键，属退化输入）。m2942 的副本写在 dataclass 之前，靠 `from __future__ import annotations` 才免于 `NameError`，属实现顺序缺陷；且模块级 `def` 后写覆盖前写，保留双份会让「哪份生效」依赖行号，正是静默陷阱 |
| `FailureFace.extent_radius` 字段 | 有字段 + 文档注释 | 只有字段行 | 取 main（注释是 m2942 内容的超集） | 字段两侧一致，唯一差异是注释 |
| m2942 的「Analytic constants & helpers」小节标题与推导注释 | — | 有 | 折叠为指向 canonical 定义的注释块，并在本文档留档 | 推导内容 `4/3 π r_m³ = h³ = (2r)³` 已完整包含在 main 的常量注释里 |

**验证无静默丢失**：合并后对模块做 AST 顶层重复定义扫描，结果 `{}`（无重复）；
`python -c "import tensordem.fracture_gen"` 成功，两个符号可取用，数值与解析式一致（见 §4）。

### 2.2 `docs/dem_analytic_verification_debt_restoration.md`（自动合并，无冲突）

m2942 追加 §6 记录「collect error 已修、13 个测试仍红＝诚实债务」。
main 侧未触碰该文件，git 三方自动合并保留全部新增段落（`\ No newline at end of file` 同时被消除）。
**保留原文不删改**——它记录的是 m2942 当时点的真实状态；本单在其后追加 §7 记录债务已被 main 的实现清偿，避免文档与 HEAD 事实冲突（见下）。

### 2.3 `report.json`（根目录，m2942 新增，自动合并）

以 `A`（新增）状态无冲突并入。按「任何一侧新增不得静默丢失」原则原样保留，
不在本单改动其内容（它是 m2942 的报告快照，被覆盖才是丢证据）。

## 3. 合并方式

在 `agent/m3915` 工作树（基于 `origin/main`）上执行：

```bash
git merge origin/agent/m2942 --no-ff
# → 仅 fracture_gen.py 冲突，手工解冲后
git commit
```

产物提交的父提交为 `e12a2ab`（main）与 `4b54f9b`（agent/m2942），
因此对 `agent/m2942` 是 **fast-forward**（`4b54f9b` 是祖先），可安全推回同分支；
对 main 是三方可合并（PR #166 由 `mergeable=false` 变 `mergeable=true`）。

## 4. 测试证据

命令与结果见 §5「复现」；摘要：

| 套件 | 结果 |
|---|---|
| `tests/test_fracture_gen_mass_footprint.py`（m2942 的 13 个「诚实债务」红测试） | **全绿** —— main 的完整实现到位后债务清偿 |
| `tests/test_fracture_gen_footprint_breakage.py` + `tests/test_fracture_gen_audit.py` | 全绿 |
| 上述三文件合计 | `49 passed` |
| 全量 `tests/` | 见 §5 表格（合并树 vs main 基线逐项对比） |

## 5. 复现

```bash
git merge-tree --write-tree origin/main origin/agent/m2942   # 解冲前：报告冲突
python3 -m pytest tests/test_fracture_gen_mass_footprint.py \
                 tests/test_fracture_gen_footprint_breakage.py \
                 tests/test_fracture_gen_audit.py -q          # 49 passed
python3 -m pytest tests/ -q -p no:randomly                    # 全量
```