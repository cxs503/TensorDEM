# 直剪盒惯性数记账的重复字典键缺陷（m16114）

**锚点**：`GAP-DEM-FRICTION` 的未竟部分（直剪盒宏观摩擦测量面）。本单不改物理，修的是
m14840 速率记账修复**在结果层被静默作废**的缺陷（defect_fix）。

## 1. 缺陷（F1）

`src/tensordem/shear_cell.py::simulate_shear_cell` 返回的字典**同一个字面量里写了两次
`"inertial_number"`**：

1. 靠前的一处（m14840 新增）用**实测床高** `bed_height_mean` 换算的
   `shear_rate_actual` 与**实测**法向应力 `sig_s["mean"]` 计算；
2. 结尾一处（历史遗留）用**名义** `case.shear_rate`（分母为名义 `fill_height`）与
   **目标**应力 `case.normal_stress` 计算。

Python dict 字面量重复键后写覆盖前写且**不报警**，因此 m14840 的速率记账修正对所有
消费者（`measure_macro_friction`、报告脚本、测试）**完全不可见**——任何人读
`results["inertial_number"]` 拿到的都是名义值，正是 m14840 §5.4 记为「报告级错误」的那条口径。

**红/绿基线（AST 静态判定，不依赖任何重算力）**
`artifacts/m16114/inertial_key_dup_ast.json`：

| 版本 | 返回字典键数 | 重复键 |
|---|---|---|
| HEAD（修复前） | 33（有效 32） | `["inertial_number"]` |
| 工作树（修复后） | 33（有效 33） | `[]` |

## 2. 修法（最小改动）

1. 抽出纯函数 `inertial_number(shear_rate, radius, density, normal_stress)`：
   `I = γ̇ · d · √(ρ/σ)`，`d = 2r`（da Cruz et al. 2005 *Phys. Rev. E* 72:021309；
   GDR MiDi 2004 *Eur. Phys. J. E* 14:341）。任一输入非正 ⇒ 返回 `nan`，
   保证「未加载记录」不能凭空造出一个惯性数。
2. 结果字典里两处**分别命名**：主键 `inertial_number` = 实测口径；
   名义口径改名 `inertial_number_nominal`（保留可追溯性，不再互相覆盖）。
3. 加**结构性回归门**：`test_simulate_shear_cell_result_dict_has_no_duplicate_keys`
   用 `ast` 解析该函数返回的字典字面量，重复键即失败——这一类「物理修正被重复键吃掉」
   的缺陷从此不可能再静默复发。

## 3. 影响量化（由已入库数字推导，非新跑）

m14840 审计的实测记录：压实后床高 0.0293–0.0297 m vs 名义 `fill_height` 0.05 m，
法向应力保持误差 <1%。

⇒ 名义口径把剪切率（及惯性数）**低估 `0.05 / 0.0295 = 1.695` 倍**；即修复后
`inertial_number ≈ 1.70 × inertial_number_nominal`（应力口径差 <1% 可忽略）。
这也解释了 m14721 报告里 I ≈ 0.14–0.28 与本文参数不可复算的现象：那条数字既用了名义 γ̇，
又与本文 `I = γ̇·d·√(ρ/σ)` 的口径混淆。

## 4. 验证

```bash
python3 -m pytest tests/test_shear_cell_friction.py -q     # 13 passed（新增 3 例）
```

新增 3 例：`test_inertial_number_matches_the_published_definition`（手算值 + `1/√σ` 标度 +
5 组非法输入 ⇒ `nan`）、`test_result_reports_measured_not_nominal_inertial_number`
（真跑 tiny case：主键 == 用 `shear_rate_actual`/实测 σ 复算值；且 `shear_rate_actual ≥ shear_rate`）、
`test_simulate_shear_cell_result_dict_has_no_duplicate_keys`（AST 结构门）。

## 5. 注意点 / 未完成（honest）

1. **本单不解决非稳态**：m14840 的 F3（78 颗粒、压板接触 3–10 个、τ 在平均窗内仍漂移）
   依旧成立，`mu_macro` 仍**不可**用于「μ vs 实验 <10%」的判定。要过那条线必须先按 m14840 §5.1
   把床层做到 ≥6–8 层、~10³ 颗粒，并开 `stationarity_tol` 门；这是算力活，本单按容量纪律
   （load15 ≈ 100/128 核）未启动。
2. 修复只改**记账**，不触碰接触力、伺服或积分，因此所有历史物理数字逐位不变
   （唯一的数值变化是 `inertial_number` 键的值变成实测口径，名义值移到
   `inertial_number_nominal`）。
3. 消费方若原先读 `results["inertial_number"]` 当名义值用，需改读 `inertial_number_nominal`；
   仓内 `measure_macro_friction` 只做透传与 Mohr-Coulomb 拟合，不含 I 的算术，无兼容性影响。