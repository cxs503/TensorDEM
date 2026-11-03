# TensorDEM 三维接触模型专项审计（m2425）

**仓**：TensorDEM（`src/tensordem/dem3d.py` + `src/tensordem/contact_tangential.py`）
**触发**：外部审计指出三维接触摩擦需专项核查（①无转动自由度却算滚动阻力矩；②`_tangential_state` 的法向旋转投影/接触消失清理/再接触初始化三处存疑；③friction/rolling/restitution 三系数未独立验证）。

**结论速览**：①外部质疑成立——旧滚动项是"零做功的正交力偶"，已改为可证的耗散型等效模型，并把无自旋自由度的残留缺口如实记录；②三处疑点全部确证为真缺陷并修复（另发现 2 处：活键期积累切向历史、`diagnostics()` 会推进状态）；③friction 与（新）rolling 通过一阶原理校验，**restitution 在强阻尼区实测系统性偏高（e=0.3 目标实测 0.396），判诚实 FAIL，未修复（附修复建议）**。

---

## 1. 案例介绍（物理背景与算例设计）

审计对象是碎冰 DEM 的三维接触本构：法向线性弹簧-阻尼（由恢复系数反解）、切向 Cundall-Strack 增量库仑摩擦、以及"滚动阻力"。难点是这三项都在同一个 `forces()` 里耦合，任何一项语义错误都会被其它项掩盖，所以必须造**可解析的单接触算例**。

**test bed**：`2×2×2` 点阵中把两颗粒放到受控中心距 `d = 2R − overlap`（法向恰为 `+x`），其余 6 颗粒移到 50 m 之外、所有键断开（仅"断键"组保留 `(0,1)` 键）。这样系统只剩一个接触。

| 参数 | 值 |
|---|---|
| 半径 R | 0.025 m |
| 法向/切向刚度 k_n = k_t | 5000 N/m |
| 单颗粒质量 m | 6.001751e-02 kg |
| 法向力 F_n = k_n·overlap | 5 / 10 / 50 N（按算例） |
| 时间步 dt | 1e-5 s（≈ recommended_dt/13） |
| 密度 / 阻尼 / 外部阻力 | 917 kg/m³ / 0 / 0 |

**两种协议**（都是接触模型验证的标准做法）：

1. **规定运动学**（力级别）：冻结位置、强加相对速度，法向恒为 `+x`，逐次直接调用力核 → 库仑锥、成对性、切向状态机严格可算，不受几何漂移污染。
2. **导轨 + 平衡法向载荷**（能量级别）：particle 0 固定，外载 `−F_n x` 精确抵消法向接触力，接触轴每步复位到 x 轴；只剩切向运动自由度，外载垂直于运动、做功为零，因此**动能下降必须等于接触功**。

**容差（相对）**：代数恒等式 1e-10；状态复位/投影 1e-12；能量账目 3%；恢复系数 2%。

## 2. 审计方法与工具链

* 探针脚本 `scripts/audit_dem3d_contact_3d.py`（力核 + 引擎 `IceDEM3D`，无第三方求解器）→ 机读 JSON/CSV；
* 判定测试 `tests/test_dem3d_contact_audit.py`（28 项 + 2 项 `xfail(strict=True)`）；
* 本报告与本单所有数字由 `scripts/report_dem3d_contact_audit.py` 从 JSON 注入（零手抄）；
* 旧实现缺陷用 `legacy_*` 探针**重放修复前核**量化，而非口头断言。

## 3. 计算结果（逐项定谳表）

| 审计项 | 判定 | 关键数字（实测 vs 解析 / 不变量） |
|---|---|---|
| a) 无摩擦接触 | **PASS** | F_t = 0.0e+00 N（恒零）；F_n 相对误差 8.9e-16 |
| a) 纯滑动 | **PASS** | \|F_t\| = 4.000000000 N vs 4.000000000 N（rel 8.9e-16） |
| a) 纯滚动 | **NOT_SUPPORTED** | 无自旋自由度；零滑移时 μ_r 阻力 0.0 N |
| a) 静止接触 | **PASS** | 漂移 0.0e+00 m、\|v\| 1.5e-16 m/s；承载力占锥 0.50 |
| b) 作用反作用 | **PASS** | 四工况 max \|F_i+F_j\|/F_n = 0.0e+00 |
| b) 成对力偶预算 | **结构性限制** | 合力 0.0e+00 N；力偶 0.1920 N·m = d·μF_n（0.1920 N·m） |
| c) 摩擦功耗散账目 | **PASS** | W = 3.078185e-04 J vs ΔKE 3.075986e-04 J（闭合 rel 7.2e-04）；峰值不越锥 True |
| c) 旧状态机缺陷量化 | **PASS** | 反向行程 projectd 0.004005 m（rel 1.2e-03）vs legacy 0.040005 m（rel 9.00，10.0x） |
| d) 断键后接触 | **PASS** | 键存续期历史 0.0e+00 m；断键首步 0.0250 N vs 单增量 0.0250 N |
| d) 再接触初始化 | **PASS** | 剪除 True；再接触力 0.0e+00 N vs 陈旧值 4.000 N |
| d) 法向旋转投影 | **PASS** | 伪法向残留 8.9e-15 N；切向残留 0.0e+00 N vs 陈旧值 4.000 N |
| Q1 滚动阻力（修复后） | **PASS** | \|F\| 0.5000 N vs 0.5000 N；功 2.6676e-04 J vs ΔKE 2.6668e-04 J（rel 3.1e-04）；力偶轴 z 分量 -1.000 |
| Q1 滚动阻力（修复前） | **FAIL_LEGACY（旧实现）** | 功率 0.0e+00 W（正交力，零耗散）；力偶轴 = sliding axis (drilling torque) |
| Q3 恢复系数 e=0.9 | **PASS** | 实测 0.90199（rel 2.21e-03） |
| Q3 恢复系数 e=0.5 | **FAIL** | 实测 0.55064（rel 1.01e-01） |
| Q3 恢复系数 e=0.3 | **FAIL** | 实测 0.39624（rel 3.21e-01） |

> `FAIL_LEGACY` 表示"旧实现确证失败、当前实现已通过"，保留作为缺陷尺寸的量化证据。

### 3.1 恢复系数独立验证（诚实 FAIL）

| e_target | e_measured (dt = 40 µs) | e_measured (dt = 10 µs) | 相对误差 | 判定 |
|---|---|---|---|---|
| 0.9 | 0.90199 | 0.90199 | 2.208e-03 | PASS |
| 0.5 | 0.54976 | 0.55064 | 1.013e-01 | FAIL |
| 0.3 | 0.39624 | 0.39624 | 3.208e-01 | FAIL |

dt 加密 4 倍后回弹比几乎不变（e=0.5 位移 0.0009），说明这是**模型效应**：解析映射 `e = exp(−πζ/√(1−ζ²))` 假设双边线性振子，而引擎的阻尼力在 `F_n ≤ 0` 时被单边钳位（`clamp_min(0)`），释放阶段阻尼被截断，回弹速度系统性偏高。

## 4. 已确认并修复的缺陷

| # | 位置 | 缺陷 | 修复 |
|---|---|---|---|
| 1 | `forces()` 切向块 | 存**未钳位**试探位移 → 滑动积累无界弹性记忆；反向时力要滑回全部累积滑移才反向 | 改用 `advance_tangential_contact`，只存 `F_t/k_t` |
| 2 | `_get_tangential_state` | 全局坐标历史不做切平面投影 → 法向转动后产生伪法向力 | 投影到当前切平面 |
| 3 | `_tangential_state` 生命周期 | 接触消失后历史常驻（无界增长 + 再接触继承陈旧滑移） | 每次力评估剪除未出现对 |
| 4 | 活键期 | 活键对仍积分切向历史 → 断键瞬间弹出饱和力 | 以 `contact_mag`（活键为 0）驱动状态机 |
| 5 | `_compute_rolling_resistance_torque` | 正交力偶：零做功、钻磨力偶、伪侧向加速度 | 改为沿 ±v̂_t 的反对称耗散等效阻力 |
| 6 | `diagnostics()` | 只读评估也推进切向历史 → 每步多吃一个增量 | 仅 `update_fracture=True` 提交 |
| 7 | `energy()` / `diagnostics()` | 能量账缺切向弹簧能 ½k_t·|δ|²（竖线已转义） | 新增 `particle_tangential_energy_J` 并计入机械能 |
| 8 | `contact_tangential` | 缺 `advance_tangential_contact` 等 5 个 API（仓内 2 个验证套件 collection error，CI 红） | 按库文档与既有验证脚本语义补齐 |
| 9 | `compute_tangential_force` | 文档称返回钳位位移，实现返回未钳位力；无参数校验 | 返回 `F_t/k_t`，补校验 |

## 5. 注意点（坑与局限）

* **几何漂移会污染切向结论**：让颗粒自由积分时，滑移本身会改变球-球接触法向，进而把"切向"结论污染成几度到十几度的伪法向分量；因此力级别探针必须冻结几何（或导轨复位），这是本单第二次迭代才定位的坑。
* **成对力矩不守恒是结构性的**：平动模型里切向力偶 2R·μF_n 没有自旋吸收器，属设计限制而非缺陷；必须写成 `STRUCTURAL_LIMITATION` 并纳入测试，否则日后会被当成回归反复"修"。
* **纯滚动不可表达**：无 ω 自由度时 μ_r 在零滑移处为零，即**无静滚动阻力**；要真滚动阻力需接 `tensordem.rolling_resistance` 的 EPSD/CDT（需自旋状态）。
* **旧符号约定陷阱**：把滚动阻力写成 `−μ_rF_n v̂_t` 时，若施加到 `ci` 上会变成"加速"而非"阻碍"（引擎的成对约定是 `F_on_i` 沿 `v_rel`）；必须用成对功率 ≤ 0 校验。

## 6. 误差判定（关键量的显式回答）

| 关键量 | 基准 | 误差 | 是否 < 3% |
|---|---|---|---|
| 饱和切向力 | μF_n（解析） | 8.9e-16 | 是 |
| 法向力 | k_n·overlap | 8.9e-16 | 是 |
| 成对合力 | 0（牛顿第三定律） | 0.0e+00 | 是 |
| 滑动耗散功 | μF_n·s_slip | 7.2e-04 | 是 |
| 弹簧储能 | (μF_n)²/(2k_t) | 1.6e-15 | 是 |
| 滚动阻力大小 | μ_rF_n | 8.9e-16 | 是 |
| 滚动耗散功 | μ_rF_n·s | 3.1e-04 | 是 |
| 断键首步切向力 | k_t·v·dt（单增量） | 0.00 | 是 |
| 恢复系数 e=0.9 | 引擎目标值 | 2.21e-03 | 是 |
| 恢复系数 e=0.5 | 引擎目标值 | 1.01e-01 | **否**（模型效应，见 §3.1） |
| 恢复系数 e=0.3 | 引擎目标值 | 3.21e-01 | **否**（模型效应，见 §3.1） |

## 7. 验证与回归

```
PYTHONPATH=src python3 -m pytest tests/test_dem3d_contact_audit.py -q          # 28 passed, 2 xfailed
PYTHONPATH=src python3 -m pytest tests -q --ignore=tests/test_dem3d_hertz_contact.py     --ignore=tests/test_dem3d_restitution.py --ignore=tests/test_fracture_gen_mass_footprint.py
                                                                              # 916 passed, 44 failed
PYTHONPATH=src python3 scripts/audit_dem3d_contact_3d.py                      # 产物 artifacts/dem3d_contact_audit/
```

* 全量套件 44 个失败**全部为既存欠账**：把本单改动 stash 后在干净 `origin/main` 上逐文件复跑同样失败（`test_dem3d_contact_impact.py` 5、`test_dem3d_platen_contact_mask.py` 3、`test_dem3d_restitution_verification.py` 4、`test_restitution.py` 1、`test_fracture_gen_audit.py` 13、`test_ice_floe_physics.py` 16；另 3 个文件为 collection error）。本单改动**未新增任何失败**，并让 `test_friction_cone_verification.py`、`test_tangential_friction_verification.py` 由 collection error 转为全绿。
* 既存红根因：早年任务提交时 `src/` 侧文件被门禁滤除（tests/scripts 进仓、实现未进），缺 `MASS_RADIUS_FACTOR`、`clamped_contact_restitution` 与整套 `hertz_*`。建议平台另派修复单。

## 8. 修复建议（优先级）

1. **P0**：`IceDEM3D` 增自旋自由度后接 `rolling_resistance` 的 EPSD；在此之前把 `rolling_resistance_coefficient` 文档化为"等效滑动阻力系数"。
2. **P1**：法向阻尼改用 `restitution.damping_for_clamped_restitution` 反向标定（同步更新所有以 e 为输入的基线），届时 §3.1 的 xfail 应翻案。
3. **P1**：补齐 `hertz_*`、`clamped_contact_restitution`、`MASS_RADIUS_FACTOR`，让 CI 转绿。
4. **P2**：`_tangential_state` 张量化（pair-key 张量 + `index_add`），去掉逐对 `.item()` 循环。
5. **P2**：耗散量（摩擦/滚动）进入累计能量账，便于长期能量审计。

## 附：产物

`scripts/audit_dem3d_contact_3d.py`（探针+机读产物）、`tests/test_dem3d_contact_audit.py`（判定测试）、`artifacts/dem3d_contact_audit/`（JSON/CSV/PNG）、`src/tensordem/dem3d.py` 与 `src/tensordem/contact_tangential.py`（修复）。
