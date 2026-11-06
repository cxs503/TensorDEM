# 2-D energy/fracture ledger: negative controls that are green when they should be

修复目标：`tests/test_dem2d_energy_fracture_ledger_negative.py` 在 main 上
**设计性红灯**（7 个负控用例必然失败），导致整仓 CI 无法转绿。本文件记录
失效机理、修复方式，以及负控自身的检出裕度（全部由脚本注入，无手抄）。

## 1. 案例与物理背景

工况是 2-D 冰-工具压入（tool indentation）：7×3 的 bonded-particle 冰梁，
工具以 0.2 m/s 压入，0.12 s 内压断 21 条 bond，台账记录 5 个能量通道：

| 通道 | 含义 | 本工况量级 (J) |
|---|---|---|
| `tool_work_J` | 工具做功 | 1.031e-01 |
| `damping_dissipation_J` | 阻尼耗散 | 1.266e-02 |
| `fracture_release_J` | 断键释放的键弹性能 | 1.330e-02 |
| `pair_contact_acquired_J` | 接触对获得的能量 | 4.799e-03 |
| `external_work_J` | 外驱动做功（本工况休眠，恒为 0） | 0.000e+00 |

平衡恒等式（`src/tensordem/dem.py` 的 `IceDEM.accounting` 逐时步累加）：

```
dE_mech - W_tool - W_external + Q_damping + E_fracture - E_pair_contact == 0
```

泊松/量纲层面不涉及 Cb/L/B 船型几何，本单不触发几何前置校验。

## 2. 失效机理（为什么 main 红）

原实现的负控把「门必须亮红」写错了对象：

```python
# 原实现：在污染数据上直接断言「平衡闭合」
assert rel_err < 0.005, "... this test is expected to FAIL (red)"
```

断言写的是「闭合」，数据却是被污染的，所以断言必然失败——**红灯就是测试本身**，
不是被测系统的信号。CI 只有通过/失败两态，没有「预期失败」这一档，于是 7 个用例
永久红。正确写法是把「门亮红」作为断言对象（`pytest.raises`），让**绿 = 负控成立**。

同时发现第二个缺陷：`external_work_J` 在本工况恒为 0，原来的
`test_negative_external_work_negated`（× -1）对 0 取负仍是 0，负控**空转**，
即使平衡门完全瞎掉对该通道也照样"通过"。修复后加了空转守卫（见 §4）。

## 3. 怎么修

1. `balance_check(sim, tol)`：把平衡门单独抽成会 `assert` 的函数（与正控同一判据）。
2. `_assert_detects(sim, channel, factor|offset)`：三个必要条件
   - 注入必须**非空转**（通道值真的变了）；
   - 干净台账必须先**闭合**（保证亮红可归因于注入而非基线坏了）；
   - 污染数据上门**必须** `pytest.raises(AssertionError)`，且残差 > 0.5 %。
3. `external_work_J` 改用**加性**故障（+1×`tool_work_J`），并在测试里钉住
   「本工况该通道为 0」的前提，附一个守卫测试断言「对休眠通道做符号翻转会被判空转」。
4. 顺带修掉 main 上另外两处红灯（同一「CI 全绿」判据下无法回避）：
   - `docs/test_coverage_expansion_m4729.md` 里把兄弟仓 TensorSPH 的一个基线模块路径
     写成行内代码指针，`scripts/audit_doc_evidence.py` 判定为 MISSING code 指针，
     连带 4 个证据门用例红；改为不含路径扩展名的散文表述，并按 `--check` 要求
     重生成 `docs/doc_evidence_ledger.json`。
   - `tests/test_threefield_lbm_transport.py::test_streaming_is_mass_conserving`
     的绝对阈值 1e-12 低于 `sum(f)`（≈1.6e4）的 1 ulp，属求和顺序偶然性；
     改为按舍入预算 `8·eps·Σ|f|` 判据 + 相对漂移 < 1e-14 的物理判据
     （`src/tensordem/threefield/lbm_transport.py` 的 stream 只做重采样/累加，
     实数运算下严格守恒）。

## 4. 结果（程序注入，`scripts/plot_m4812_ledger_negative_controls.py`）

干净台账残差 `2.796e-03`（0.28 %，< 0.5 % 判据，正控通过）。逐通道单故障检出：

| 通道 | 注入 | 污染后残差 | 是否检出 | 裕度 (×tol) |
|---|---|---|---|---|
| `fracture_release_J` | ×0 | 1.117e-01 | 是 | 22.3 |
| `fracture_release_J` | ×3 | 1.731e-01 | 是 | 34.6 |
| `pair_contact_acquired_J` | ×0 | 3.997e-02 | 是 | 8.0 |
| `pair_contact_acquired_J` | ×3 | 7.154e-02 | 是 | 14.3 |
| `tool_work_J` | ×0 | 3.988e+00 | 是 | 797.6 |
| `damping_dissipation_J` | ×0 | 1.056e-01 | 是 | 21.1 |
| `external_work_J` | +1.031e-01 J | 7.962e-01 | 是 | 159.2 |

原始数据 `artifacts/m4812_ledger_negative_controls/ledger_negative_controls.json`，
点检日志 `artifacts/m4812_ledger_negative_controls/ledger_negative_controls.log`，
图 `artifacts/m4812_ledger_negative_controls/ledger_negative_controls.png`
（残差柱状图 + 判据线 + 干净基线，对数纵轴）。

## 5. 误差判定

| 关键量 | 基准 | 误差 | 是否 < 3 % |
|---|---|---|---|
| 干净台账相对残差 | 0（恒等式闭式） | 2.796e-03 | 是（0.28 %） |
| 最小负控残差 | 判据 0.5 % | 3.997e-02 | 是（为判据 8.0 倍，方向正确） |
| 质量守恒残差（3-D LBM） | 0 | 2.206e-16 相对（1 ulp） | 是（机器精度级） |

## 6. 注意点

- 负控的"绿"只证明**检出灵敏度**，不证明台账物理完备（未计摩擦/滚阻等其它耗散通道）。
- 0.12 s / 7×3 网格是脚本级规模，非工程尺度；结论限于台账身份与门禁灵敏度。
- `external_work_J` 工况依赖：若将来该场景接入外驱动，应改回乘性符号翻转，
  测试已用 `assert sim.accounting["external_work_J"] == 0.0` 钉住前提并在报错信息中提示。
- 判据 0.5 % 沿用正控原值，未放宽度；负控裕度最紧的通道仍有 8 倍。
## 7. 复现与产物（m4790 复算）

本节的数字全部由脚本注入，无手抄：

- 负控套件 `tests/test_dem2d_energy_fracture_ledger_negative.py`
  （`pytest -q tests/test_dem2d_energy_fracture_ledger_negative.py`
  → 8 passed, 1 skipped；skip 为 `external_work_J` 休眠通道的乘性空转守卫）。
- 图与逐通道残差 `scripts/plot_m4790_sensitivity.py`
  → `artifacts/m4790/ledger_corruption_sensitivity.png`（残差柱状图 + 0.5 % 判据线
  + 3 % 参考线；干净基线 0.2796 %）。
- 报告装配 `scripts/post_m4790_report.py` → `artifacts/m4790/report.md`。

复算得到与本文件 §4 一致的残差（干净台账 0.2796 %；检出裕度最紧的通道
`pair_contact_acquired_J` ×0 为 4.0 %≈8×判据，最宽 `tool_work_J` ×0 为 398.8 %；
休眠通道 `external_work_J` 用加性故障 +`tool_work_J` 得 79.6 %）。逐通道数字落在
`artifacts/m4790/sensitivity.json`，报告由 `scripts/post_m4790_report.py` 从该 JSON
注入后发布，无手抄。注意本文件的 §1–§6 成文于 m4790；此后 main 上的负控测试已升级为
含 `balance_check` 门函数、空转守卫与 `external_work_J` 加性故障的 10 用例版本，
本节的复算与之口径一致（判据 0.5 %、同一残差定义）。
