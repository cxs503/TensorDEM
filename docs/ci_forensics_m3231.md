# CI 红因取证报告：TensorDEM main@f5de971c「6 个测试文件收集失败」

任务书（m3231）声称：`tests` workflow 在 main@f5de971c 死于 6 个测试文件的 **收集错误**
（import 断裂），疑因最近合并移动/重命名了 `lattice_elasticity3d` 模块族而测试未随迁；
另一条 `DEM3D benchmark cases` workflow @8c7f316f 疑同根因。

本报告给出取证结论：**该根因假设被证伪**；并给出 main 上真实红因的定位、已修部分、
未修部分的清单与归因。

---

## 1. 结论摘要

| 声明 | 判定 | 证据 |
|---|---|---|
| 命名 6 个测试文件在 f5de971c 收集失败（ERROR collecting） | **NOT_REPRODUCED（证伪）** | §3.1 |
| `lattice_elasticity3d` 模块族被移动/重命名，测试未随迁 | **NOT_REPRODUCED（证伪）** | §3.2：`src/tensordem` 全历史 rename/delete 事件数 = 0 |
| 6 个命名文件当前状态 | 收集正常，**63 passed** | §3.1 |
| main 真实红因（全量 pytest） | **45 failed / 1187 passed / 2 skipped**（基线） | §4 |
| 本轮修复 | 45 → **35 failed**（10 个测试转绿），修的是真缺陷、未改任何断言 | §5 |
| 「全量 pytest 绿」 | **honest-FAIL：未达成**，剩余 35 个红分布在 3 个簇 | §6 |

---

## 2. 方法与环境

* 工作树：`/root/tp-agent/work/m3231`，分支 `agent/m3231`，基线 commit `f5de971c03f6363afa701063a833a3766c874eca`。
* 环境：Python 3.11、torch 2.7.1、pytest 8.x；`tests/conftest.py` 把本 checkout 的 `src` 前置进 `sys.path`，
  因此测试解析到的是本树代码（不是 site-packages 里其他 worktree 的安装）。
* 与 CI 的已知差异：CI 额外安装 `git+https://github.com/cxs503/TensorPhysics.git`（本地缺），
  `pyproject.toml` 的 `torch==2.14.1` 在本机解析为 2.7.1。差异影响见 §6.3 的交叉核对。
* 复现命令（本报告所有数字均由下列命令直接产出，日志留存在 `runs/ci-m3231/`）：

```bash
python -m pytest --collect-only -q                 # 收集面：1234 collected, 0 error
python -m pytest -q tests/test_dem3d_born_huang.py \
  tests/test_dem3d_brazilian_static.py \
  tests/test_dem3d_lattice_elasticity3d.py \
  tests/test_dem3d_shear_cross_section_convergence.py \
  tests/test_dem3d_shear_static_reference.py \
  tests/test_lattice_elasticity3d_modulus_factor.py    # 63 passed
python -m pytest -q > runs/ci-m3231/pytest_baseline.log   # 全量基线
```

---

## 3. 根因取证

### 3.1 「收集失败」不可复现

```
$ python -m pytest tests/ --collect-only -q | tail -1
1234 tests collected in 13.92s          # 0 个 error

$ python -m pytest -q <6 个命名文件>
63 passed, 2 warnings in 27.67s
```

6 个文件与其依赖模块在 `HEAD` 的 git 对象层均存在（`git cat-file -e HEAD:<path>` 逐个通过）：

```
tests/test_dem3d_born_huang.py                    OK   src/tensordem/born_huang.py          OK
tests/test_dem3d_brazilian_static.py              OK   scripts/benchmark_dem3d_brazilian.py OK
tests/test_dem3d_lattice_elasticity3d.py          OK   src/tensordem/lattice_elasticity3d.py OK
tests/test_dem3d_shear_cross_section_convergence.py OK src/tensordem/shear_convergence.py   OK
tests/test_dem3d_shear_static_reference.py        OK   src/tensordem/shear_static.py        OK
tests/test_lattice_elasticity3d_modulus_factor.py OK   src/tensordem/shear_specimen.py      OK
```

→ 任务书的症状描述与本树 `HEAD` 不符。**最可能的解释是:那张 CI 红快照对应的工作树/批次已过期
（冷启动包 WORLD 行同源风险）**，或该 workflow 红在别处（见 §4：真实红是 45 个 test failure，
不是 collection error）。任何基于「模块被移动」的修复都会是空修复。

### 3.2 「移动/重命名」假设被证伪

```
$ git log --diff-filter=ADR --name-status -- src/tensordem/{lattice_elasticity3d,shear_static,
      shear_convergence,shear_specimen,born_huang,dem3d_lattice_elastic}.py
A  src/tensordem/lattice_elasticity3d.py   cf47740  (m2170)
A  src/tensordem/shear_static.py           0fd8dd9  (m2586)
A  src/tensordem/shear_convergence.py      654c95b  (m2917)
A  src/tensordem/shear_specimen.py         b1f195a  (m2477)
A  src/tensordem/born_huang.py             1c28a28  (m2579)
A  src/tensordem/dem3d_lattice_elastic.py  72a0260  (m2289)

$ git log --diff-filter=DR --oneline -- src/tensordem | wc -l
0
```

六次事件全是 `A`（新增），无一次 `D`/`R`；`src/tensordem` 全历史**零次**删除/重命名。
即：正典名从未变过，import 路径无需迁移。

---

## 4. main 的真实红因（基线，45 failed）

| 簇 | 文件 | 数量 | 首个可判据错误 | 归因 |
|---|---|---:|---|---|
| C1 | `test_ice_floe_physics.py` | 16 | `module 'ice_floe_physics' has no attribute 'analytic_submerged_area' / 'anchor_*' / 'run_all'` | `benchmarks/ice_floe_physics.py` 缺测试所引用的锚点函数（基准脚本落后于验证套件） |
| C2 | `test_fracture_gen_audit.py` | 14 | `FractureGenConfig.__init__() got an unexpected keyword argument 'face_extent'`；`'FractureField' object has no attribute 'volume_fraction'` | `fracture_gen` 落后于审计套件：缺 `face_extent` 契约、`volume_fraction`、ledger `face_extent` 键 |
| C3 | `test_dem3d_moving_wedge_bow.py` (3)、`test_dem3d_wedge_resistance_sensitivity.py` (1)、`test_threefield.py` (2) | 6 | `MovingWedgeBowIceDEM3D.forces() got an unexpected keyword argument 'update_history'` | **覆盖方法签名漂移**：基类钩子新增关键字，子类 override 未随迁（详见 §5.1） |
| C4 | `test_dem3d_restitution_verification.py` | 4 | `0.6667242852845159 not less than 0.015` 等 | 数值参考断言：恢复系数映射与解析参考的偏差远超门限，需单独物理复核 |
| C5 | `test_wake_lbm_analytic_verification.py` | 1 | Smagorinsky 闭包 vs 独立求值不一致 | 同上，数值/实现一致性 |
| C6 | `test_doc_evidence_pointers.py` | 4 | `docs/doc_evidence_ledger.json is stale`；3 个 ORPHAN 指针 | 文档证据台账未随新文档刷新 + 3 处悬空引用（详见 §5.2） |

---

## 5. 本轮修复（fix-forward，未改任何断言）

### 5.1 C3 覆盖方法签名漂移（6 个测试）

根因：`IceDEM3D.forces()` 的关键字契约先后长出 `update_fracture` 与 `update_history`
（后者保证「探针不推进摩擦历史」，见 `dem3d.py` 的 docstring）。两个子类 override 停在旧签名，
并且**只在多态调用点崩**——基类 `diagnostics()` 正是这样调的
（`forces(external_forces, update_fracture=False, update_history=False)`），于是任何监控路径、
能量审计、不变性探针在子类族里全部 TypeError，而只调用构造与 `step` 的测试仍然是绿的。

```
src/tensordem/dem3d_wedge.py    MovingWedgeBowIceDEM3D.forces  -> 增 update_history 并透传
src/tensordem/rudder_ice_load.py RudderIceLoadDEM.forces       -> 同上
```

修复后：`test_dem3d_moving_wedge_bow` + `test_dem3d_wedge_resistance_sensitivity` +
`test_threefield`（`BowContactDEM` 继承该 override）+ `test_rudder_ice_load` 共 57 passed（原 6 failed）。

### 5.2 C6 文档证据台账（4 个测试）

按仓库自带流程重生成台账，并修正 3 处悬空引用：

* `docs/wake_lbm_analytic_verification.md`：引用的模块路径缺 `src/tensordem/` 前缀（旧写法指向不存在的相对路径）→ 改为完整包路径。
* `docs/dem3d_contact_impact_dashpot_convention.md`：引用了从未提交的一次性计时探针（该文件已不存在于仓库）
  → 改引其已提交的复现件 `tests/test_contact_impact_dashpot_convention.py`。
* `docs/dem3d_fracture_footprint_breakage.md`：引用了未提交的审计 PNG → 改为文字复述读数并注明图未随仓提交
  （数字仍取自同文档 §3 的表，未改动任何结论）。
* `python scripts/audit_doc_evidence.py` 重生成 `docs/doc_evidence_ledger.json`（612 增 22 删，纯机械）；
  ORPHAN 3 → 0，`test_doc_evidence_pointers.py` 14 passed。

### 5.3 新增两道结构性守卫（防同类问题静默复发）

* `tests/test_import_surface.py`（320 个参数化用例）——AST 扫描 `src/ tests/ scripts/ examples/`
  全部 `import tensordem.x` / `from tensordem.x import y`，逐个 import 校验模块与符号存在；
  失败的报错直接点名**哪个导入方被落在后面**，把「收集错误」变成带定位的测试失败。
  含 3 个负控：扫描面非空、故意植入的陈旧 import 必被检出、命名族公共符号逐个钉住。
* `tests/test_override_signature_parity.py`——枚举包内所有对 `forces/diagnostics/step` 的 override，
  要求其关键字参数集合/种类/默认值 ⊇ 基类最近祖先；含负控（人为窄化签名必被判红）。
  这条守卫正是 §5.1 那个 bug 的通用版本：以后基类再加关键字，任何未迁移的 override 立刻变红。

---

## 6. 未完成部分（honest-FAIL）

### 6.1 剩余 35 个红

| 簇 | 数量 | 状态 | 下一步（建议单独立项） |
|---|---:|---|---|
| C1 `benchmarks/ice_floe_physics.py` 锚点缺失 | 16 | **未修** | 基准脚本补 `analytic_submerged_area / analytic_equilibrium_height / anchor_archimedes_static / anchor_hydrostatic_equilibrium / anchor_heave_period / anchor_friction_slip / anchor_restitution_mapping / run_all`，逐一对应 `docs/ice_floe_physics_validation.md` 的解析式 |
| C2 `fracture_gen` 审计契约缺失 | 14 | **未修** | 需实现 `FractureGenConfig(face_extent∈{disc,plane})`、`FractureField.volume_fraction`、
`identify_bonds_crossing_face(..., face_extent=)`、ledger `mass_accounting()` 的 `face_extent` 键与 `cell_volume` 拼写别名；属功能实现，非一行修补 |
| C4/C5 数值参考 | 5 | **未修** | 恢复系数映射与 Smagorinsky 闭包需按解析参考重做验证（禁止改断言洗红） |

### 6.2 为什么不强行凑绿

任务书禁止「改断言洗红」。C1/C2 是**源侧功能缺失**：把测试改窄或加 `xfail` 只是把欠账藏起来；
C4/C5 的偏差量级（0.667 vs 0.015 门限）远大于环境噪声，必须按物理复核而不是放宽门限。

### 6.3 与 CI 环境的交叉核对（残余不确定性，WITHHELD）

本机缺 CI 额外安装的 `TensorPhysics`；`test_threefield` / `test_wake_lbm_*` / `run_manifest_s3`
中存在可选依赖路径。本轮已排除其对 C3 的影响（C3 的 6 个测试修后本地全绿，且修复点与可选依赖无关），
但 C4/C5 的数值偏差是否与 torch 版本（2.7.1 vs CI 的 `torch==2.14.1` 声明）相关**未证**——
留作 WITHHELD，需 CI 侧日志确认。

---

## 7. 误差/判据说明

本任务无物理量输出，交付判据是**测试计数**与**结构契约**，全部可程序化复算：

| 指标 | 基线 f5de971c | 修复后 | 判定 |
|---|---:|---:|---|
| 收集错误数（ERROR collecting） | 0 | 0 | 任务书声称的 6 个 → 未复现 |
| 命名 6 文件用例 | 63 passed | 63 passed | 绿 |
| 全量 failed | 45 | **35** | 未达标（honest-FAIL） |
| 全量 passed | 1187 | **1517** | +330（新增 320 守卫用例 + 10 个转绿） |
| 文档证据 ORPHAN 指针 | 3 | 0 | 绿 |
| 新增结构守卫 | — | 320 用例全绿（含负控） | 绿 |

「是否 <3%」判据不适用（无解析解/实验基准量）；此处等价判据是**红数下降 10/45 = 22.2%**
且下降全部来自真缺陷修复，无一条断言被改宽、无一条被 xfail。

三份全量日志与图的产物均取自同一脚本：

```
artifacts/ci_triage_m3231/pytest_baseline.log   45 failed, 1187 passed   (基线 f5de971c)
artifacts/ci_triage_m3231/pytest_final.log      35 failed, 1517 passed   (本分支 HEAD)
artifacts/ci_triage_m3231/failure_clusters.png     由 scripts/plot_ci_triage_m3231.py 解析上述两份日志生成
```

![failure clusters](artifacts/ci_triage_m3231/failure_clusters.png)

*图 1* 左：各红簇在基线（红）与修复后（蓝）的失败数，计数直接由两份 pytest 日志解析，非手抄；
右：本轮关闭数（绿=C3/C6，共 10）与待办数（灰=C1 16 / C2 14 / C4 4 / C5 1）。

### 7.1 `DEM3D benchmark cases` workflow（@8c7f316f）

该 workflow 的判定面是脚本链（`scripts/run_dem3d_benchmark_suite.py` → `visualize_dem3d_benchmarks.py`
→ `run_dem3d_mechanics_benchmarks.py` → …），与「收集错误」不是同一失败族。本树上这些脚本与配套测试
（`tests/test_dem3d_benchmark_case_suite.py`、`tests/test_dem3d_visualization.py`、
`tests/test_dem3d_mechanics_benchmarks.py`、`tests/test_dem3d_specimen_response_benchmarks.py`、
`tests/test_dem3d_platen_compression.py`）在全量套件中均为**绿**；但其 workflow 的完整脚本链未在本机重跑，
其绿色状态判 **WITHHELD**（需 CI 侧 `dem3d-benchmark-results-*` 产物确认）。
本轮改动不触碰 `benchmarks/dem3d/**` 与上述脚本，因此不改变该 workflow 的触发路径集合。