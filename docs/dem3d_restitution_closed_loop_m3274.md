# C4 验证欠账清偿：3-D 恢复系数闭环（harness 静默丢弃 γ_n 的修复落主干）

Protocol: `tensordem-restitution-closed-loop-v1`（承接 m2536 欠账清单 §4 与 m3231 红簇 C4）
Status: **已清偿**（4 红转绿，求解器 vs 独立 RK4 钳位参考最大 0.27 %）

## 1. 欠账是什么

`docs/dem3d_restitution_verification.md`（已合并）与
`tests/test_dem3d_restitution_verification.py`（已合并）钉住一条闭环契约：
3-D 求解器 `IceDEM3D` 的压缩侧接触律

    F_n = max(k_n·overlap − γ_n·v_n, 0)

必须复现独立 RK4 积分的钳位一维冲击参考（`clamped_restitution_ratio`），
且 `damping_for_target_restitution` 的逆映射必须在真实求解器里命中目标恢复系数。

`main` 上该文件 4 例长期红（m3231 红簇 C4，0.667 vs 0.015 门限等），
根因在 harness 而非引擎：`measure_dem3d_impact` 把待审的绝对阻尼
`contact_damping=gamma_n` 写进 `DEM3DConfig`，却未声明阻尼约定。引擎默认
`contact_damping_model="restitution"` 且 `restitution_coefficient=1.0`，
于是 pair 分支走 `gamma_n = 0.0` —— **γ_n 被静默丢弃，每次冲击都变成完全弹性**，
`e_measured ≈ 1.0` 与任何 γ_n 无关（m2536 记录的「引擎返回 e≈1.0」即此）。

## 2. 修复（最小改动，7 行）

`src/tensordem/contact_restitution.py::measure_dem3d_impact` 显式钉
`contact_damping_model="absolute"`，与 tool/platen 分支及 2-D harness（m3129）同约定：
γ_n = contact_damping [N s/m]，参考解为钳位律 RK4 积分。
引擎两条约定分支（`restitution`/`absolute`）本身均合法，缺陷只在 harness 未声明约定。

## 3. 验证（程序化注入，零手抄）

`python3 scripts/verify_restitution_closed_loop.py` →
`artifacts/restitution_closed_loop_m3129/evidence.json` + 双联图。

| 量 | 基准 | 求解器 | 相对误差 | <3 % ? |
|---|---|---|---|---|
| e(γ=1.0) | 0.914802 | 0.915808 | 1.10e-3 | 是 |
| e(γ=3.0) | 0.772607 | 0.774698 | 2.71e-3 | 是 |
| e(γ=5.0) | 0.659722 | 0.658328 | 2.11e-3 | 是 |
| e(γ=10.0) | 0.462764 | 0.463225 | 9.97e-4 | 是 |
| 闭环 e_target=0.6（校准 γ=6.26603） | 0.6 | 0.597486 | 4.19e-3 | 是 |
| 同目标经典映射（γ=5.56045） | 0.6 | 0.632947 | 5.49e-2 | 否（预期，模型差） |
| 等质量对 m_eff | m/2=0.0300088 | 0.0300088 | 0 | 是 |
| 等质量对 e(γ=2.0) | 0.783533 | 0.786562 | 3.87e-3 | 是 |
| dt 减半收敛（γ=10） | 0.463225 | 0.462447 | 1.68e-3 | 是 |

经典闭式 `exp(−πζ/√(1−ζ²))` 在 ζ=0.289 处偏离钳位参考 16.2 %，
与文档「弱阻尼极限才可用」的口径一致（本表即该口径的数值证据）。

**可复现性**：本单重新生成的 evidence.json 与 m3129 分支原件逐字段比对，
数值差为 0（0 个字段漂移）——修复落主干后结果与当年分支完全一致。

## 4. 复现

```
python -m pytest tests/test_dem3d_restitution_verification.py -q   # 9 passed
python3 scripts/verify_restitution_closed_loop.py
```

## 5. 局限

- 本单只清偿 C4（恢复系数闭环）。m3231 剩余红簇 C1（floe 锚点 16）、
  C2（fracture_gen API 14）、C5（LBM 闭式 1）与
  `test_threefield_lbm_transport.py::test_streaming_is_mass_conserving`
  （3.6e-12 vs 1e-12 门限，float64 求和顺序量级，属门限标定问题）不在本单范围。
- 2×2×2 雕刻双球只验证 pair 接触分支；tool/platen 分支的绝对阻尼约定
  由既有测试覆盖，本单未触碰。
