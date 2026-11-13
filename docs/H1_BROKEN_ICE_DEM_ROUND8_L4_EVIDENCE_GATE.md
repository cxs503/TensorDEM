# H1-BROKEN-ICE-DEM · 第 8 轮：L4 证据门结构修复 + C1/C3 证据入库

锚点：`H1-BROKEN-ICE-DEM`。上游：`docs/H1_BROKEN_ICE_DEM_ROUND3.md`（休止角量测修复）、
`docs/H1_BROKEN_ICE_DEM_ROUND7_GATE.md`、审计 `docs/AUDIT_M14948_H1_BROKEN_ICE_DEM.md`（m14948/m15029）。
本轮**不碰物理、不调参数、不改判据门**：只修「判据/证据机器」本身——它坏了，所以前三轮的 PASS/FAIL 读不出来。

---

## 1. 四个结构性缺陷（都是「仪器坏了」，不是「物理不对」）

### 1.1 L4 门的布尔优先级 bug：门永远到不了 L4

`integration_callers.check_h1_maturity` 的 L4 子句原文：

```python
if "ref" not in case or "got" not in case and name != "repose":
```

Python 里 `and` 比 `or` 紧，等价于 `"ref" not in case or ("got" not in case and name != "repose")`。
于是这条子句的实际语义与注释写的（「每个算例都要有 ref/got 数值对照」）**相反**：

| 算例 | 有 `ref` | 有 `got` | 旧子句判定 | 应然 |
|---|---|---|---|---|
| buoyancy | 是 | 是 | 通过 | 通过 |
| **channel** | 是 | **否**（1.2 缺陷） | **记 gap** | 记 gap（对，但原因是缺 `got` 而非缺 ref） |
| **repose** | 是 | 否 | **静默豁免** | 应记 gap |

两侧都错：`channel` 因缺 `got` 被记 gap——**这本来是对的**，但它与 1.2 一起构成死锁；
`repose` 缺实测块被**静默豁免**——一个没有测量值的算例可以拿 L4。实测（修前，quick 跑）：

```json
"maturity": {"level_achieved": "L1", "passed": false,
             "gaps": ["case channel: verdict not PASS",
                      "L4: case channel has no ref/got numeric pair"]}
```

即：**三个算例全 PASS 也到不了 L4**，因为 `channel` 从来没有 `got` 块。

### 1.2 `channel_clearing_case` 从不输出 `got`：L4 对照缺半边

`buoyancy_equilibrium_case` 有 `ref`/`got`/`error` 三元组，`channel_clearing_case` 只有
`ref` + `summary` + `criteria`——实测值藏在 `summary` 里，门看不见。修法不是「把门放宽」，
而是让**测量值出现在它该在的位置**：新增 `got` 块（8 个数值键 + 时间/比例），`ref` 保持原样
（判据常量 + 声明的判据来源）。

### 1.3 `quick` 冒烟模式结构性地必然 FAIL

`validate(quick=True)` 给 channel 的预算：`n_steps=300, dt=5e-4, speed=0.4` → 航程 `0.06 m`；
而走廊 `corridor = 1.0 m`。判据 `sweep_complete = (advance >= corridor)` 于是在冒烟模式里
**恒假**，「航道是否被清空」这条判据一次也没被行使——冒烟只证明「能跑」「跑出来必挂」。

修后：冒烟模式的走廊按预算缩放（`corridor = min(1.0, 0.8 · advance_max)`），并且把
`steps_for_corridor` / `sweep_budget_ratio` / `sweep_budget_sufficient` 一并写进结果，
任何调用者覆盖 `n_steps` 时都能一眼看出预算与走廊不匹配。修后 quick：

```json
"setup":  {"corridor_m": 0.048, "steps_for_corridor": 240, "sweep_budget_ratio": 1.25}
"criteria": {"sweep_budget_sufficient": true, "sweep_complete": true,
             "lane_halved": true, "mass_conserved": true, "energy_closed": true}
```

冒烟运行仍然**不能**当 L4 证据（走廊只有 4.8 cm）：`check_h1_maturity` 现在对 `quick` 运行
显式记 gap `"L4: quick (smoke) run is not L4 evidence - rerun with quick=False ..."`，
但 L3 级仍判 PASS。

### 1.4 提交的 C1/C3 证据文件不在树里：三个测试长期 skip

`tests/test_h1_broken_ice_dem.py` 从 `runs/h1_broken_ice_dem/results_c1c3.json`
读 C1/C3 证据，三个测试（`test_committed_buoyancy_evidence_matches_archimedes`、
`test_committed_channel_evidence_is_self_consistent`、`test_committed_evidence_verdicts_are_not_over_claimed`）
在文件缺失时 `pytest.skip`。审计 m15029 第 3 项（「run artifacts 必须进树」）因此**看起来闭了环，实际没闭**：
CI 里这三个断言从未执行过。上一轮交付的 `runs/` 工件被合并门滤除，文件就丢了。

本轮新增 `benchmarks/h1_broken_ice_dem/record_c1c3_evidence.py`：跑完 C1+C3 后写**裁剪过**的证据
（C3 时间序列每 20 步留 1 点），文件从 ~13 MB 降到 ~100 KB 量级——体积本身就是它上次被门滤除的原因。

---

## 2. 改动（最小面，7 文件）

| 文件 | 改动 |
|---|---|
| `src/tensordem/polyhedron_floe3d/h1_cases.py` | channel：`got` 块 + 走廊预算诊断（`steps_for_corridor`/`sweep_budget_ratio`/`sweep_budget_sufficient` 判据）+ `passed = all(criteria.values())`；repose：`got` 块；`validate(quick=True)`：走廊按预算缩放 |
| `src/tensordem/integration_callers.py` | `check_h1_maturity`：L4 子句改写为「每个算例都要非空 `ref` **且** 非空 `got`」（显式括号、不再豁免任何算例名）＋ quick 冒烟不作 L4 证据 ＋ 新增 `ref_got_pairs` 审计字段 ＋ `_merge_repose_shards` 补 `got` |
| `tests/test_h1_broken_ice_dem.py` | 新增 7 项测试：门的正例/缺 `got` 反例/repose 不再豁免/quick 不作 L4 但过 L3/失败算例与缺 repro_cmd/channel 的 `got`+预算/quick 走廊自洽/repose 的 ref-got 对 |
| `benchmarks/h1_broken_ice_dem/record_c1c3_evidence.py` | 新：证据录制（含 markdown 表生成，数字零手抄） |
| `benchmarks/h1_broken_ice_dem/make_figures_round8.py` | 新：从证据 JSON 画 C1/C3 两幅图（唯一输入是证据文件，图不能与数字打架） |
| `runs/h1_broken_ice_dem/results_c1c3.json` + `..._table.md` | 新：提交的 C1/C3 证据与它的表 |
| `docs/H1_BROKEN_ICE_DEM_ROUND8_L4_EVIDENCE_GATE.md` | 本文 |

**热文件声明**：本轮改到 `src/tensordem/integration_callers.py`（近 6 日内 m17651/m17470 动过，
已合并）与 `polyhedron_floe3d/h1_cases.py`（近 6 日内只有 api-sync 与 m15203 合并）。
改动为**追加式**（新键、新判据、新测试），不改任何已有键名与物理路径，冲突面仅限文件头尾。

---

## 3. 计算结果（程序化生成，零手抄）

证据：`runs/h1_broken_ice_dem/results_c1c3.json`；表：`runs/h1_broken_ice_dem/results_c1c3_table.md`
（由录制脚本从同一次运行写出）。图：`docs/figures/h1_round8_c1_buoyancy.png`（浮沉收敛到解析吃水）、
`docs/figures/h1_round8_c3_channel.png`（航道浓度时程，含旧质心代理指标对照）。

C1（自由浮冰静平衡）与 Archimedes 解析值：

| 量 | 测值 | 解析值 | 相对误差 |
|---|---|---|---|
| 浸没体积分数 | 0.894555 | 0.894634（= ρ_ice/ρ_water = 917/1025） | 0.009 % |
| 吃水 | 0.107360 m | 0.107356 m | 0.003 % |

C3（航道清空，`n_floes=24, n_steps=6000, dt=5e-4, speed=0.4 m/s, kn=5e4, mu=0.35`）：

| 判据 | 值 | 门 | 判定 |
|---|---|---|---|
| 走廊内冰浓度（精确裁剪体积 / 走廊层体积） | 0.8998 → 0.0（ratio 0.000） | ratio < 0.5 | PASS |
| 船体推进距离 | 1.200 m | ≥ corridor 1.0 m | PASS |
| 质量残差 | 0.0 | < 1e-12 | PASS |
| 能量残差 | 0.01375 | < 0.05 | PASS |
| 半清空时间 | 1.051 s | — | 诊断量 |
| 稳态阻力均值 | 0.048 N | — | **不作对标**（见 §5.3） |

运行时间 297.3 s（单核，128 核机器上单进程）。以上数值一律由
`record_c1c3_evidence.py` 写出，表由 `_markdown_table()` 渲染——文档不手抄。

---

## 4. 复现

```bash
# 证据录制（约 5.5 min，单核；数字全部程序化写入）
PYTHONPATH=src python benchmarks/h1_broken_ice_dem/record_c1c3_evidence.py
# 图
PYTHONPATH=src python benchmarks/h1_broken_ice_dem/make_figures_round8.py
# 本轮新增/相关的测试（含 3 个曾经 skip 的证据测试，现在真跑）
PYTHONPATH=src python -m pytest tests/test_h1_broken_ice_dem.py tests/test_h1_repose_estimator.py -q
```

---

## 5. 注意点 / 局限

1. **本轮没有产生新的物理结论**：C1/C3 的 PASS 是第 2 轮就有的量级结论，本轮把它们从
   「文档里说的」变成「CI 每次都会重算的」。S 级判读应记 **defect_fix**（判据/证据机器），
   不是 `new_capability`。
2. **休止角 C2 仍是 WITHHELD**：`repose` 的 `ref` 是文献带 25–40°（无原始数据集），
   本轮只给它补了 `got`（角度数组 + 统计），**没有**把它推成 PASS；文献带的证据等级不变。
3. **C3 没有实验阻力基准**：`resistance_mean_steady_N = 0.048 N` 在小数点后第二位，
   24 个 12 cm 冰块的算例里它受接触刚度/步长支配，**不能**当阻力基准用。C3 判的是
   「航道被清空 + 账本闭合」，ref 块里已显式声明 `no experimental resistance benchmark is claimed`。
4. **quick 模式的走廊只有 4.8 cm**：它只证明管线通，不证明清空能力；门已把它挡在 L4 之外。
5. **证据文件的裁剪**：C3 序列每 20 步留 1 点（`series.stride = 20`），被丢弃的样本不进文件但
   `n_samples_full` 记录原始长度；判据与账本全部用**全程**数据算，不受裁剪影响（裁剪只作用于
   写盘的那份拷贝）。

---

## 6. 误差判定（逐项回答「是否 < 3%」）

| 关键量 | 基准 | 误差 | < 3% ? |
|---|---|---|---|
| C1 浸没体积分数 | Archimedes ρ_ice/ρ_water（解析，非拟合） | **0.009 %** | **是** |
| C1 吃水 | Archimedes `(ρ_i/ρ_w)·t`（解析） | **0.003 %** | **是** |
| C3 走廊冰浓度终值 | 「清空」= 0（判据侧声明，无实验基准） | 0.0000 绝对值 | 是（对判据），但**无实验基准**故不作精度主张 |
| C3 质量残差 | 0（守恒律） | 0.0 | 是 |
| C3 能量残差 | 0（耗散账，门 0.05） | 0.01375（占门 27.5 %） | 门内，但**非 0**，见前轮 §5.5 的冲击期记账误差 |
| 休止角 | 文献带 25–40° | 不可判（WITHHELD） | **不作主张** |

结论：几何/静力学量（C1）误差 0.009 %/0.003 %，远优于 3 % 门；C3 无实验基准，
只主张「判据通过 + 账本闭合」，不主张阻力精度；休止角仍 WITHHELD。**没有任何一项被粉饰为达标。**