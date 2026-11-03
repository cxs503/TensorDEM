# DEM3D 颗粒对接触阻尼与恢复系数闭合：m670 回归的定谳与修复

本文件由 `scripts/verify_dem3d_pair_damping.py` 生成的
`outputs/dem3d_pair_damping/audit.json` 程序化渲染（数字零手抄）。

## 1. 案例与物理背景

颗粒对法向接触是 DEM 冰力学里唯一决定碰撞恢复系数与能量耗散的通道。
`IceDEM3D` 的三条接触分支本来共用同一套「线性弹簧 + 阻尼器、法向力截断到零」的
单边（unilateral）本构：

    F_n = max(0, k_n · δ - c_n · v_n),   δ = 2R - |x_j - x_i| > 0

工具分支与压板分支都直接用**绝对阻尼系数**（`contact_damping` / `platen_damping`），
唯独颗粒-颗粒分支在 m670（提交 `a40e29a`）把 `contact_damping` 换成了由
`restitution_coefficient` 派生的阻尼，且默认 `restitution_coefficient = 1.0` 时
该阻尼被置为 **0**。后果：默认配置下自由颗粒对碰撞变成完全弹性（e≈1.0），
`contact_damping` 成为静默空参（MOCK 化的"已验证未落地"能力），
并同时把下列**已合并的验证锚点**打成红：

* `tests/test_dem3d_contact_impact.py`（5 条，解析冲击基准）
* `tests/test_dem3d_restitution_verification.py`（4 条，恢复系数验证）
* `tests/test_restitution.py::EngineClosureTests`（2 条，引擎闭合）

| 参数 | 值 |
|---|---|
| 颗粒半径 R | 0.025 m |
| 密度 ρ | 917.0 kg/m³ |
| 质量 m | 0.060018 kg |
| 有效质量 m_eff = m/2 | 0.0300088 kg |
| 法向刚度 k_n | 5000.0 N/m |
| 冲击速度 | 0.5 m/s（两球各半，相对 1.0 m/s） |
| 参考解 | `tensordem.restitution.unilateral_restitution`（同一条截断律的 RK4 独立解，4000 采样/接触周期） |

## 2. 怎么做（方法/流程/工具链）

1. **代码修复（最小 hunk，`src/tensordem/dem3d.py`）**
   * 颗粒-颗粒法向分支：`restitution_coefficient == 1.0` 时用 `gamma_n = cfg.contact_damping`
     （恢复 m670 之前的、与工具/压板分支一致的绝对阻尼语义）。
   * 线性接触的恢复系数反演改用 `restitution.damping_for_restitution`
     （截断律的精确逆），并按 (e, k_n, m_eff) 缓存，避免每步每对做数值求解；
     Hertz 分支保留原来的双边小阻尼式（已在文档中声明为局限）。
2. **验证脚本**：扫 `c_n ∈ {0,1,2,5,10,20}` 与 `target e ∈ {0.9,0.7,0.5,0.3}`，
   各配置在引擎内做正面撞击、测量回弹比，与截断律参考解逐点比对；再扫时间步
   （1.0/0.5/0.25 × `recommended_dt`）与冲击速度（0.1/0.5/2.0 m/s）做负控。
3. **机器判据**：新增 `tests/test_dem3d_pair_damping.py`（5 条）把契约钉住——
   含"接触阻尼不得是空参"的回归指纹（c=0 → e≈1.0，c=5 → e<0.75）。

## 3. 计算结果

### 3.1 原始阻尼模式：引擎 vs 截断律参考

| c_n [N·s/m] | engine e | clipped RK4 reference | rel. err | bilateral formula | contact [ms] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.99987 | 1.00000 | -0.013% | 1.0000 | 7.65 |
| 1.0 | 0.88252 | 0.88249 | +0.003% | 0.8796 | 7.65 |
| 2.0 | 0.78333 | 0.78353 | -0.026% | 0.7731 | 7.65 |
| 5.0 | 0.56636 | 0.56594 | +0.074% | 0.5195 | 7.80 |
| 10.0 | 0.35481 | 0.35747 | -0.745% | 0.2454 | 8.16 |
| 20.0 | 0.17495 | 0.17540 | -0.257% | 0.0118 | 9.18 |

### 3.2 请求恢复系数模式：截断律反演 vs 旧双边映射

| target e | inverted c_n | engine e | rel. err | old bilateral c_n -> e | rel. err |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 0.839 | 0.90009 | +0.010% | 0.9021 | +0.23% |
| 0.7 | 2.994 | 0.69971 | -0.042% | 0.7178 | +2.55% |
| 0.5 | 6.254 | 0.49860 | -0.279% | 0.5456 | +9.12% |
| 0.3 | 12.182 | 0.29690 | -1.035% | 0.3948 | +31.59% |

`c_n = 0` 一行是负控：无阻尼必须给 e = 1，实测吻合。旧双边反演在低恢复系数下
系统性欠给阻尼（clip 移除耗散），实测 e 偏高最多 **31.59%**，
这正是 3.2 表中第二列误差的来源。

### 3.3 时间步与冲击速度负控（c_n = 5）

| sweep | value | engine e | reference | rel. err |
| --- | ---: | ---: | ---: | ---: |
| dt_scale | 1.0 | 0.56636 | 0.56594 | +0.074% |
| dt_scale | 0.5 | 0.56430 | 0.56594 | -0.290% |
| dt_scale | 0.25 | 0.56608 | 0.56594 | +0.025% |
| impact_speed | 0.1 | 0.56275 | 0.56594 | -0.563% |
| impact_speed | 0.5 | 0.56636 | 0.56594 | +0.074% |
| impact_speed | 2.0 | 0.56940 | 0.56594 | +0.611% |

## 4. 结果图

* `outputs/dem3d_pair_damping/e_vs_contact_damping.png`——引擎回弹比随 `contact_damping`
  的曲线，与截断律参考解重合，双边公式整段偏离（clip 偏置）。
* `outputs/dem3d_pair_damping/target_closure.png`——请求恢复系数闭合误差柱状图：
  新（截断律反演）vs 旧（双边映射），附 ±3% 判据线。

## 5. 误差判定（逐项，是否 <3%）

| 关键量 | 基准 | 最差相对误差 | <3% ? |
|---|---|---|---|
| 引擎 e（阻尼模式，6 组 c_n） | 截断律 RK4 参考解 | 0.745% | 是 |
| 引擎 e（目标模式，4 组 e） | 请求值本身 | 1.035% | 是 |
| 旧双边反演的 e 闭合 | 请求值本身 | 31.59% | 否（已废弃路径） |
| 时间步/冲击速度敏感度 | 同律参考解 | 0.611% | 是 |

## 6. 注意点与局限

* **修复严格向后兼容于非物理默认之外**：全量测试（`tests/`，跳过 m2536 已声明的
  `test_fracture_gen_mass_footprint.py` 收集错误）在本修复后为 **33 failed / 1009 passed**（基线 44 failed / 993 passed），
  与修复前基线 **44 failed** 相比，修复 **11 条**、**新增 0 条**失败（同一批 33 条红为本单
  之外的既有欠账：`test_ice_floe_physics`、`test_fracture_gen_audit`、
  `test_dem3d_platen_contact_mask`）。
* 默认 `contact_damping = 5` 现在**真实生效**：默认自由颗粒对碰撞 e ≈
  0.5664，
  与 m670 之前的语义一致；任何过去"默认配置下弹性碰撞"的结果需要重跑。
* 仅一维法向冲击：无切向摩擦、无转动、无黏附；Hertz 分支的恢复系数仍用双边
  小阻尼近似，未做截断律反演（Hertz 的 k 随重叠变化，需要非线性逆解）。
* 参考解是**同一条律**的独立 RK4 解，因此通过证明的是*实现保真度*，不是冰材料
  真实恢复系数的物理标定；有效质量取孤立对的 m/2，有完整键的邻居不适用。
* `classical_restitution` / `restitution_to_damping`（双边教科书关系）仍然是**公式正确**
  的，但不能用来设置本引擎的绝对阻尼——这一点在 `docs/dem3d_contact_restitution.md`
  已有说明，本单把代码改到与该结论一致。

## 7. 复现

```
python -m pytest tests/test_dem3d_pair_damping.py tests/test_restitution.py \
    tests/test_dem3d_contact_impact.py tests/test_dem3d_restitution_verification.py -q
python scripts/verify_dem3d_pair_damping.py --output outputs/dem3d_pair_damping
```
