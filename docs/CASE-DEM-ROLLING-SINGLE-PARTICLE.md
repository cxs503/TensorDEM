# CASE-DEM-ROLLING-SINGLE-PARTICLE — 单颗粒斜面纯滚动加速度（TensorDEM 算例池）

* case_id: `CASE-DEM-ROLLING-SINGLE-PARTICLE`
* 生成时间(UTC): 2026-11-12T10:09:57Z
* 判定: **PASS**（容差 2.0 %）
* 预注册: `prereg/CASE-DEM-ROLLING-SINGLE-PARTICLE.json`（阈值先于生产运行声明）
* 工件: `artifacts/case_rolling_single_particle/result.json` + `case_rolling_single_particle.png`
* 能力复用: `tensordem.rolling_resistance.simulate_sphere_on_incline`（未新写求解器）

## ① 问题与参考值出处

单颗实心球在倾角 theta 的斜面上**无滑移纯滚动**，经典刚体动力学给出

```
a = g sin(theta) / (1 + I/(m R^2))          alpha = a / R
I/(m R^2) = 2/5  ->  a = (5/7) g sin(theta)
```

本算例取 theta = 15 deg、无滑移（mu 远大于所需值）。次级参考为滚动阻力模型
Ai, Chen, Rotter, Ooi (2011), *Assessment of rolling resistance models in discrete element
simulations*, Powder Technology **206**, 269-282, doi:10.1016/j.powtec.2010.11.003：
该文比较 CDT / viscous / EPSD 三类滚动阻力模型，滚动阻力矩取 `M_r = mu_r |F_n| r_eff`，
由此斜坡上纯滚动的平衡临界角为 `theta_c = arctan(mu_r)`、稳态加速度为
`a = g (sin t - mu_r cos t)/(1 + I/(m R^2))`。

**参考值出处与零手抄声明（如实）**：本报告与工件中的全部数字由
`benchmarks/case_rolling_single_particle/run_case.py` 内的闭式关系**程序化计算**
（参数 g/theta/R/rho 与 I/(mR^2)=0.4 全部取自代码而非手输常数），不是从论文表格抄录。
本执行环境**无法访问 Ai et al. 2011 原文/表格**（无该 PDF，也无在线检索），
因此**不宣称**引用其原表原页数值，仅引用其模型关系式与临界角结论——这一点在
`provenance.reference.access_note` 中机器可查。误差判定完全针对闭式解析解。

无滑移参考 alpha 的符号遵循 harness 约定：`v > 0` 为沿坡向下、`omega` 为 +z 分量，
无滑移即 `v + R omega = 0`，故 `alpha = -a / R`。

## ② 方法与实现

* 复用仓库现有能力 `tensordem.rolling_resistance.simulate_sphere_on_incline`
  （Cundall-Strack 切向弹簧/库仑锥 + 法向弹簧-阻尼 + rolling_resistance 模块的 CDT 滚动阻力矩），
  半隐式（symplectic）Euler 时间推进；**未新写求解器**（避免重复造轮子）。
* 观测算子（测量定义先于运行声明）：
  * `a` = 稳态窗内 v(t) 的最小二乘斜率（扫过 ~195 个切向振荡周期，振荡平均效果收敛）；
  * `alpha` = 稳态窗内 omega(t) 的最小二乘斜率；
  * 无滑移闭合 = `|a + R alpha| / |a|`；
  * 滑移速率 rms = `rms(v + R omega)`；
  * 瞬时加速度 a(t) 用中心差分给出，仅作诊断（见 ⑤ 的振荡机理）。
* 负控与判别力：
  1. `mu = 0.01`（低于纯滚动所需 mu ≈ 0.051）→ 必须滑移、必须被判失败；
  2. 无惯性修正伪参考 `g sin t`（比真值高 `I/(mR^2)/(1+I/(mR^2))` ≈ 28.6 %）→ 必须被 2 % 门拒绝；
  3. `theta = 5 deg < arctan(mu_r) = 5.711 deg`（mu_r = 0.1）→ 必须近似静止。
* 计算规模：单核本地执行，最贵点（Delta t = 5.0e-06）用时
  15.5 s，全案 < 1 分钟，无需集群。

## ③ 参数表（含无量纲数）

| 参数 | 值 | 单位 | 来源/说明 |
|---|---|---|---|
| 倾斜角 theta | 15.0 | deg | case definition (pool anchor) |
| 球半径 R | 0.01 | m | case definition |
| 密度 rho (冰) | 900 | kg/m^3 | case definition |
| 质量 m | 0.00376991 | kg | m = 4/3 pi R^3 rho (code) |
| 转动惯量 I | 1.50796e-07 | kg m^2 | I = 2/5 m R^2 (code) |
| I / (m R^2) | 0.4 | - | solid sphere |
| 重力 g | 9.81 | m/s^2 | harness default |
| 法向刚度 k_n | 200000 | N/m | contact |
| 切向刚度 k_t | 200000 | N/m | k_t = k_n |
| 滑动摩擦 mu | 0.9 | - | Cundall-Strack cone |
| 滚动阻力 mu_r | 0 | - | Ai et al. 2011 CDT branch (case A: off) |
| 时间步 Delta t | 2e-05 | s | semi-implicit Euler |
| 模拟时长 / 拟合窗 | 0.12 / 0.09 | s | LS fit window |
| 步数 | 6000 | - | duration / dt |

无量纲数：

| 无量纲数 | 值 | 说明 |
|---|---|---|
| I / (m R^2) | 0.4000 | 1.4 -> a/(g sin t) = 0.7143 |
| a / (g sin theta) | 0.714286 | = 1/(1+I/mR^2) |
| friction demand / cone | 0.0851 | << 1 -> no slip |
| k_n / (m g cos t / R) | 5.599e+04 | contact stiffness ratio |
| omega_n Delta t | 0.1457 | normal mode |
| omega_slip Delta t | 0.2725 | tangential (slip) mode; ~23 steps/period |
| pure_rolling_feasible(theta, mu_r, mu) | True | in-code check |

## ④ 对照表（ref vs got，误差逐项）

基准点 theta = 15 deg、mu_r = 0：

| 指标 | ref（闭式解） | got（TensorDEM） | 相对误差 | 判据 <2% |
|---|---|---|---|---|
| 球心平动加速度 a | 1.813582 m/s^2 | 1.813580 m/s^2 | 0.0001 % | ✔ |
| 转动角加速度 alpha | -181.3582 rad/s^2 | -181.3587 rad/s^2 | 0.0003 % | ✔ |
| 无滑移闭合 norm(a + R alpha)/a | 0 (exact) | 3.659e-06 | 0.0004 % | ✔ |
| 瞬时加速度振荡幅值 | 0.7254 m/s^2 | 0.7323 m/s^2 | 0.9415 % | ✔ |
| 滚动阻力分支 (mu_r=0.1) a | 1.136744 m/s^2 | 1.136739 m/s^2 | 0.0004 % | ✔ |
| 滚动阻力分支 (mu_r=0.1) alpha | -113.6744 rad/s^2 | -113.6755 rad/s^2 | 0.0010 % | ✔ |
| 负控1: mu=0.01 滑动分支 a | 2.444258 m/s^2 | 2.444258 m/s^2 | 0.0000 % | ✔ |
| 负控1: 与无滑移参考的偏差 (必须 >2%) | 1.813582 m/s^2 | 2.444258 m/s^2 | 34.7751 % | ✔ |
| 负控2: 无惯性修正伪参考 g sin t (必须被拒) | 2.539015 m/s^2 | 1.813580 m/s^2 | 28.5715 % | ✔ |
| 静态保持 theta=5 deg < theta_c=5.711 deg: a | 0 (static) | 3.276e-06 m/s^2 | creep = 3.832e-06 g sin t | ✔ |

补充诊断（基准点）：v(t) 拟合 R^2 = 0.99999935、
omega(t) 拟合 R^2 = 0.99999594、滑移速率 rms = 1.330e-04 m/s
（最大 1.881e-04 m/s，相当于 v 的量级 1e-4）。

**关键两问**：球心平动加速度误差 **0.0001 %**，转动角加速度误差
**0.0003 %** —— 均 **< 2 %（<3 %）**，判定 PASS。

![case figure](case_rolling_single_particle.png)

图说明：(a) v(t) 与 -R omega(t) 重合 = 无滑移闭合；(b) 瞬时 a(t) 在解析均值上下振荡，
虚线为解析均值与实测包络（无阻尼切向弹簧的必然结果，见 ⑤）；
(c) 时间步收敛：均值误差始终 ~1e-6（红/蓝），振荡幅值误差随 Delta t 一阶收敛；
(d) 四项指标相对各自参考的归一化（全部落在 100 % 附近，偏差标注在点上）。

## ⑤ 收敛性与参数敏感性（三点）

**(1) 时间步收敛**（基准点，仅 Delta t 变化）：

| Delta t [s] | 步数 | omega_slip Delta t | a [m/s^2] | a 误差 | alpha 误差 | 振荡幅值 | 幅值误差 |
|---|---|---|---|---|---|---|---|
| 4.0e-05 | 3000 | 0.545 | 1.813581 | 0.0000 % | 0.0001 % | 0.7540 | 3.9342 % |
| 2.0e-05 | 6000 | 0.273 | 1.813580 | 0.0001 % | 0.0003 % | 0.7323 | 0.9415 % |
| 1.0e-05 | 12000 | 0.136 | 1.813583 | 0.0000 % | 0.0001 % | 0.7271 | 0.2329 % |
| 5.0e-06 | 24000 | 0.068 | 1.813580 | 0.0001 % | 0.0002 % | 0.7259 | 0.0581 % |

* 均值 a、alpha 在全部 Delta t 上误差 ~1e-6 —— **不敏感**（均值是守恒量级的解析结果，误差被振荡平均吸收）；
* **瞬时振荡幅值**才是一阶收敛量（幅值误差随 Delta t 单调下降），这是该系统真正的离散误差度量。

**(2) 接触刚度 k_n 敏感性**（k_t = k_n，Delta t 固定）：

| k_n [N/m] | a [m/s^2] | a 误差 | a(t) 标准差 |
|---|---|---|---|
| 5e+04 | 1.813592 | 0.0005 % | 0.5141 |
| 2e+05 | 1.813580 | 0.0001 % | 0.5179 |
| 8e+05 | 1.813581 | 0.0000 % | 0.5333 |

* 均值 a 在 k_n 变化 5e+04~
  8e+05 N/m 范围内散布 < 1e-5（相对），
  与「参考解不含 k_n」一致 → 刚度无关性成立；
* 瞬时幅值随 k_n 略增（0.5141 →
  0.5333），即刚度越大、同一 Delta t 下振荡越欠解析，
  这正是 (1) 中 Delta t 判据要盯幅值的原因。

**(3) 拟合窗长度敏感性**：

| 拟合窗 [s] | a [m/s^2] | a 误差 | alpha 误差 |
|---|---|---|---|
| 0.045 | 1.813600 | 0.0010 % | 0.0024 % |
| 0.090 | 1.813580 | 0.0001 % | 0.0003 % |
| 0.180 | 1.813582 | 0.0000 % | 0.0000 % |

* 窗长从 0.045 s 增到
  0.180 s，a 变化 < 1e-5 相对 → 估计量无窗长偏置，
  本算例用 0.09 s 窗即可（远小于窗长自由度的代价）。

## ⑥ 失败模式、局限与工件

失败模式（已实跑并记录在 result.json）：

* `points.NEG_low_friction`（mu = 0.01）：实测 a = 2.444258 m/s^2，
  与无滑移参考 1.813582 m/s^2 差 34.7751 %（>2 %，**被判失败**），
  而与滑动分支 g(sin t - mu cos t) 一致到 0.0000 %
  → 模型在滑移区间同样正确，2 % 门具备判别力。
* `points.C_below_critical`（theta = 5 deg < theta_c = 5.711 deg）：
  实测 a = 3.276e-06 m/s^2 = 3.832e-06 g sin t，
  即 CDT 无静摩擦分支但在该状态下亦无可见蠕滑（该处静力由切向摩擦承担，滚动阻力矩只在滚动建立后起作用）。
  注意这是本算例的构型结论，不能推广为「CDT 处处可持静」。
* 无惯性修正伪参考：相对误差 28.5715 %，被 2 % 门拒绝。

局限与适用范围：

* 参考值是**闭式刚体动力学**（无滑移、刚性平面、点接触），验证的是求解器的运动学/动力学闭环与接触律实现，
  **不是**对任何真实材料参数的实验标定；Ai et al. 2011 的原表数值未取得（见 ①）。
* 切向弹簧**无阻尼**，因此 a(t) 必然在均值上下振荡（幅值 = `f g sin t/(1+f)`，`f = I/(m R^2)`；
  注意它与地面滑动摩擦 mu 无关，只由惯性和重力倾角决定）；任何单点瞬时采样都会带来 ~±40 % 的假误差 —— 本算例用窗内 LS 斜率作正式指标，
  并在表中同时给出解析与实测幅值（0.9415 %）。
* 滚动阻力分支仅验证 CDT 模型；EPSD/viscous 分支在 `docs/dem3d_rolling_resistance.md` 的既有验证中覆盖，本算例不重复。
* 半隐式 Euler 一阶精度：均值误差极小是因为振幅被平均，若把小 Delta t 下的瞬时接触力当物理量使用，须先做幅值收敛。

复现（一键）：

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python benchmarks/case_rolling_single_particle/run_case.py --out artifacts/case_rolling_single_particle/result.json && PYTHONPATH=src python benchmarks/case_rolling_single_particle/figures.py && PYTHONPATH=src python -m pytest tests/test_case_rolling_single_particle.py -q
```

工件：

* `artifacts/case_rolling_single_particle/result.json` —— 全部点位、无量纲数、收敛/敏感性表、负控与 provenance；
* `artifacts/case_rolling_single_particle/case_rolling_single_particle.png` —— 四联图；
* `prereg/CASE-DEM-ROLLING-SINGLE-PARTICLE.json` —— 预注册设计（判据/负控/停止规则）；
* `tests/test_case_rolling_single_particle.py` —— CI 守卫（6 项，含两个负控）。

## 生态摩擦（如实记录）

1. **知识基座三查部分不可用**：`GET /api/v1/knowledge/search` 对 `TensorDEM`、`DEM rolling`、
   `inclined plane rolling`、`Ai 2011` 均返回 0 命中（`available: true`，store 为空）；
   `knowledge/capabilities`、`knowledge/negative_ledger`、`knowledge/oss_solutions` 返回 404。
   → 本单的三查退化为：仓内 grep（找到 `rolling_resistance` 既有能力）+ 本报告记录。
2. **Ai et al. 2011 原文不可得**：环境内无该 PDF，无在线检索 → 只能引用模型关系式，
   「原表原页数值」**WITHHELD**（已写入 provenance，不粉饰）。
3. **性能缺陷（已修）**：`simulate_sphere_on_incline` 每步把张量 append 进 Python 列表，
   实测单步耗时随步数增长（12k 步时 ~5.8 ms/步 vs 首段 ~0.6 ms/步，超线性，长跑被拖慢 ~9×）。
   改为预分配张量写入后回到线性 0.637 ms/步（12k 步 7.7 s），数值结果不变（54 项既有测试通过）。
4. **线程数**：torch 默认线程数把该单列小张量循环拖慢 10× 以上，需显式 `torch.set_num_threads(1)`
   （已在脚本中固定并在 prereg 记录）。
5. **docstring 符号约定不一致（已修）**：`simulate_sphere_on_incline` 原写 "omega 正值顺时针，omega = -v/R"，
   与代码（omega 为 +z 分量，无滑移即 omega = -v/R < 0，即逆时针向下滚）矛盾；
   若不核对，alpha 会给出 200 % 的假误差。已改为 +z 分量表述。
6. **报告通道摩擦**：约定路径 `/tmp/report.json` 是**多 agent 共享**的——本单构建 payload 期间被另一 agent 覆盖
   （实测读回的是别的任务的 `underice_pack_closure.png` payload），导致首次发布缺图；
   已改用 `/tmp/report_m16142.json` 并二次发布（report id 15262，mission 16142）。
   另：`POST /tensor/reports` 对 (mission_id, md5(markdown)) 建唯一索引，内容微调重发会 500，
   而 `PATCH /tensor/reports/{id}` 返回 405 → 报告不可原地更新，只能改 markdown 变体重发（产生重复条目）。
7. **figures 的 png_base64 在 GET 响应里恒为空串**（本单两次发布均如此，name/caption 正常）→ 图片本体只能以
   worktree 内 `artifacts/case_rolling_single_particle/case_rolling_single_particle.png` 交付，平台侧不可自证。
