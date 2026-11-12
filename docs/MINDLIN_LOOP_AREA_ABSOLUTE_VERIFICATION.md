# Mindlin–Deresiewicz 滞回环面积绝对值验证（m16526）

## 验证欠账
`tensordem.mindlin_partial_slip.reversal_loop_area` 此前仅验证了
**量纲缩放**（`μ²F_n²/k_t`，见 `test_reversal_branch_endpoints_and_slope`），
未验证**绝对值**。本单补上该欠账。

## 解析闭式
全幅双振幅 Mindlin–Deresiewicz 循环（0→δ_s 加载，再全反转至 -δ_s）的
滞回环面积（=每周期摩擦耗散能）为：

```
W_loop = 6 (μ F_n)² / k_t
```

其中 `k_t = 8 G* a` 为无滑移切向刚度。推导：对加载支
`F = μF_n[1-(1-δ/δ_s)^{3/2}]` 与全反转支
`F = μF_n - 2μF_n[1-(1-Δ/2δ_s)^{3/2}]` 在 [0, 2δ_s] 上直接求积，
环面积 = 2·∫(F_load − F_rev)dδ = 6(μF_n)²/k_t。

## 验证结果
四组参数 (μ,F_n) = (0.3,1),(0.5,2),(0.1,0.5),(0.7,3)：
`reversal_loop_area` vs 闭式相对误差均 **<1.3e-11**（远小于 3% 阈值）。

## 注意
- 闭式 `6(μF_n)²/k_t` 对应**全幅双振幅**循环（+μF_n→−μF_n）。
- 单幅 0→δ_s→0 循环的经典耗散能为 `(2/5)(μF_n)²/k_t`，二者物理含义不同，勿混用。

## 证据
- 测试：`tests/test_mindlin_partial_slip.py::test_reversal_loop_area_absolute_closed_form`（11 passed）
- 报告：`/tensor/reports` id=15579
- 图：`post_report_m16526.py` 生成（滞回环 + 闭式一致线）
