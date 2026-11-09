# EDIT_WHITELIST=docs/,scripts/,tests/
# Hertz Peak Overlap Verification Fix

## 1 案例介绍

**物理背景。** `docs/dem3d_hertz_restitution_closure.md` §3.1 报告弹性极限下峰值重叠的实测值与闭式解偏差 74.1%，记录为 WITHHELD 欠账。本单判定该偏差的来源。

**问题。** 弹性 Hertz 接触的峰值重叠闭式解为：

```
delta_max = (15 m_eff v_rel^2 / (16 E* sqrt(R_eff)))^(2/5)
```

其中 `v_rel` 是两球的**相对闭速度**。在验证脚本的两球对撞设置中，每球速度为 `speed`，相向而行，故 `v_rel = 2 * speed`。

## 2 怎么做的

审查 `scripts/verify_dem3d_hertz_restitution.py` 的 `analytic_peak_overlap` 函数，发现其使用 `speed` 而非 `v_rel = 2*speed`：

```python
# 错误版本
def analytic_peak_overlap(speed: float) -> float:
    return ((15.0 * p["m_eff_kg"] * speed ** 2)
            / (16.0 * e_star * math.sqrt(r_eff))) ** 0.4
```

正确版本应为：

```python
# 修正版本
def analytic_peak_overlap(speed: float) -> float:
    v_rel = 2.0 * speed  # 两球相向，闭速度为每球速度的两倍
    return ((15.0 * p["m_eff_kg"] * v_rel ** 2)
            / (16.0 * e_star * math.sqrt(r_eff))) ** 0.4
```

**数值验证。** 以 `speed = 0.5 m/s` 为例：

| 版本 | delta_max [m] |
|---|---|
| 错误（用 speed） | 1.056e-04 |
| 正确（用 2*speed） | 1.838e-04 |
| 实测 | 1.838e-04 |
| 偏差（错误版本） | 74.1 % |
| 偏差（正确版本） | < 0.1 % |

比值 `1.838e-04 / 1.056e-04 = 1.741 ≈ 4^(2/5) = 2^(4/5)`，与理论预期完全一致。

## 3 计算结果

修正后重新运行验证脚本，弹性极限的峰值重叠偏差从 74.1 % 降至机器精度量级（< 0.1 %）。

| v [m/s] | e | 峰值重叠实测 [m] | 峰值重叠闭式 [m] | 相对误差 |
|---|---|---|---|---|
| 0.1 | 1.000000 | 5.072e-05 | 5.072e-05 | < 0.1 % |
| 0.5 | 1.000000 | 1.838e-04 | 1.838e-04 | < 0.1 % |
| 1.0 | 1.000000 | 3.200e-04 | 3.200e-04 | < 0.1 % |

**判定：峰值重叠的 74.1 % 偏差是验证脚本的解析公式错误，不是求解器的缺陷。修正后弹性极限完全闭合。**

## 4 结果图

无新增图（纯公式修正，数值验证通过表格展示）。

## 5 注意点与局限

- **求解器本身无缺陷**：`hertz_normal_force` 和 `hertz_secant_stiffness` 的实现正确，峰值重叠的实测值与正确的解析解一致。
- **验证脚本的解析公式有 bug**：`analytic_peak_overlap` 使用每球速度而非相对闭速度，导致解析解偏小 4^(2/5) ≈ 1.741 倍。
- **恢复系数偏差（16.4 %）仍然存在**：这是独立的建模问题（线性弹簧-阻尼器理论映射到非线性 Hertz 接触的近似误差），不在本单修复范围内，仍记录为已知局限。
- **标度指数偏差（27.5 %）仍然存在**：同样是恢复系数映射的近似效应，不在本单范围。

## 6 误差判定（逐项，回答「是否 < 3%」）

| 关键量 | 基准 | 修正前相对误差 | 修正后相对误差 | 是否 < 3 % |
|---|---|---|---|---|
| 弹性峰值重叠 | 闭式 delta_max | 74.1 % | < 0.1 % | **是**（修正后） |
| 恢复系数 e 闭合 | e_target | 16.4 % | 16.4 % | 否（已知局限） |
| 标度指数 | -0.2 | 27.5 % | 27.5 % | 否（已知局限） |

**结论：峰值重叠的验证欠账已清零；恢复系数与标度指数的偏差是建模近似，非实现缺陷。**

## 7 复现

```bash
# 修正后的验证脚本
PYTHONPATH=src python scripts/verify_dem3d_hertz_restitution.py --render
PYTHONPATH=src python -m pytest tests/test_dem3d_hertz_restitution_verification.py -q
```

## 8 修改清单

- `scripts/verify_dem3d_hertz_restitution.py`: 修正 `analytic_peak_overlap` 函数，使用 `v_rel = 2*speed`
- `tests/test_dem3d_hertz_restitution_verification.py`: 更新测试断言，移除 74.1 % 偏差的 pin（已不再是欠账）
- `docs/dem3d_hertz_restitution_closure.md`: 更新 §3.1 表格与 §6 误差判定

---

**协议版本**: `tensordem-dem3d-hertz-restitution-closure-v2`（修正解析公式）
