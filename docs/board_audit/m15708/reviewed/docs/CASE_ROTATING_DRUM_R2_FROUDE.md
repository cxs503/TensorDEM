# CASE-ROTATING-DRUM 第 2 轮：Froude 数基准（m15649）

仓库 TensorDEM · 锚点 `CASE-ROTATING-DRUM` · 判据：**定性 + <20%** · 冻结判据见
`prereg/CASE_ROTATING_DRUM_R2.json` · 调研/谱系/选型见 `docs/OSS_DRUM_R2_RESEARCH.md`。

## ① 问题设定 + 基准出处

回转滚筒（R = 0.15 m，fill = 0.30，双分散 d_s = 10 mm / d_l = 20 mm，N = 209）中颗粒床的
横向运动由壳体 Froude 数 `Fr = ω²R/g` 控制（Haff 1983, JFM 132:401–430）。本轮的**定量基准**
只用精确闭式（零手抄：脚本内计算，不是誊抄表格）：

* 释放角 `θ_d = arccos(−Fr)`（6 点钟起算），由贴壁颗粒法向力 `N = m(ω²R + g cosθ) = 0` 解出；
* 离心化起点 `Fr = 1`（上式在 θ = 180° 的根）。

流态分带（Mellmann 2001, Powder Technol. 118:251–270）只作**定性**标签——其中间分界是量级
分界，不作数值基准。

## ② 方法与实现

`tensordem.drum_segregation.kernel2d` 的二维软球 DEM（法向弹簧-阻尼 + Cundall-Strack 切向 +
滚动阻力 + 旋转壁面 `v_wall = ω ẑ × x`，半隐式欧拉，dt = 1e-4 s；Cundall & Strack 1979）。
本轮在**不换物理**的前提下补了三件测量面（前轮的致死点）：

1. 逐粒子接触计数 `n_contacts`（`_contact_forces` 返回第 5 项）→ 自由飞行分数；
2. 逐样本运动学观测量：壁接触弧上端角（θ 的 95 分位）、壁层切向速度比、上半区飞行分数、
   r/R 的 5 分位、床层角速度比（`kernel2d.series`）；
3. 休止相**收敛循环**（淬火轮数上限 20、实测 rms/轮数记账、`resting_ok` 布尔判据）。

关键代码指针：`src/tensordem/drum_segregation/kernel2d.py::_contact_forces` / `::simulate`；
`src/tensordem/drum_flow/regimes.py::MELLMANN_BANDS` / `::classify_measured`；
`benchmarks/rotating_drum_froude_r2.py`；复现：

```
PYTHONPATH=src python benchmarks/rotating_drum_froude_r2.py --subset all --out artifacts/case_rotating_drum_r2/cases
PYTHONPATH=src python benchmarks/rotating_drum_froude_r2_report.py
PYTHONPATH=src pytest -q tests/test_drum_froude_r2.py
```

## ③ 参数表（无量纲数完整）

| 量 | 值 |
|---|---|
| R [m] | 0.15 |
| fill / 大颗粒面积分数 | 0.30 / 0.30 |
| d_l/d_s | 2.0（10 / 20 mm），N = 209 |
| ρ_l/ρ_s | 1.0 |
| e / μ / μ_r | 0.30 / 0.40 / 0.10 |
| Fr | 0.05 / 0.1 / 0.2 / 0.3 / 0.5 / 0.7 / 0.9 / 1.0 / 1.1 |
| ω [rad/s] | 1.81–8.48 |
| dt / 圈数 | 1e-4 s / 10 |

## ④ 对照表（ref vs got，基准判据逐条打勾/叉）

由 `benchmarks/rotating_drum_froude_r2_report.py` 从 JSON 工件**程序化生成**
（`artifacts/case_rotating_drum_r2/table_main.md`，零手抄）：


### 生成表（程序化注入，零手抄）

| Fr | N | theta_ref [deg] | theta_got [deg] | err | <20% | r_p05/R | wall ratio | upper flight | regime (measured / literature) | resting_ok | S | mode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 209 | 92.9 | 103.4 | 11.3% | PASS | 0.33 | 0.70 | 0.024 | cascading / cascading | True | 0.412 | mixed |
| 0.1 | 209 | 95.7 | 104.2 | 8.8% | PASS | 0.33 | 0.75 | 0.044 | rolling / cataracting | True | 0.462 | mixed |
| 0.2 | 209 | 101.5 | 110.4 | 8.7% | PASS | 0.33 | 0.76 | 0.066 | cataracting / cataracting | True | 0.394 | mixed |
| 0.3 | 209 | 107.5 | 116.9 | 8.8% | PASS | 0.33 | 0.72 | 0.096 | cataracting / cataracting | True | 0.415 | mixed |
| 0.5 | 209 | 120.0 | 128.3 | 6.9% | PASS | 0.31 | 0.72 | 0.133 | cataracting / cataracting | True | 0.479 | mixed |
| 0.7 | 209 | 134.4 | 136.9 | 1.8% | PASS | 0.35 | 0.73 | 0.163 | cataracting / cataracting | True | 0.511 | mixed |
| 0.9 | 209 | 154.2 | 148.9 | 3.4% | PASS | 0.35 | 0.78 | 0.204 | cataracting / cataracting | True | 0.516 | core |
| 1 | 209 | 180.0 | 146.6 | 18.5% | PASS | 0.37 | 0.76 | 0.201 | cataracting / centrifuging | True | 0.541 | mixed |
| 1.1 | 209 | 180.0 | 153.1 | 15.0% | PASS | 0.38 | 0.81 | 0.234 | cataracting / centrifuging | True | 0.594 | mixed |

### summary.json
```json
{
 "n_main_cases": 9,
 "theta_pass_20pct": "9/9",
 "theta_max_err": 0.1853074656122257,
 "centrifuging_onset_measured": null,
 "centrifuging_onset_rel_err": NaN,
 "P1_spearman_wall_ratio_vs_fr": 0.6333333333333333,
 "P2_spearman_upper_flight_vs_fr": 0.9833333333333333,
 "P3_spearman_r_p05_vs_fr": 0.8,
 "regime_family_agreement": "6/9",
 "regime_pairs": [
  [
   "cascading",
   "cascading"
  ],
  [
   "rolling",
   "cataracting"
  ],
  [
   "cataracting",
   "cataracting"
  ],
  [
   "cataracting",
   "cataracting"
  ],
  [
   "cataracting",
   "cataracting"
  ],
  [
   "cataracting",
   "cataracting"
  ],
  [
   "cataracting",
   "cataracting"
  ],
  [
   "cataracting",
   "centrifuging"
  ],
  [
   "cataracting",
   "centrifuging"
  ]
 ],
 "all_cases_resting_ok": true,
 "settle_rms_max": 0.00392400000000003,
 "plateau": {},
 "sensitivity": [
  {
   "case": "dt_base",
   "fr": 0.3,
   "revs": 6.0,
   "dt": 0.0001,
   "n_total": 209,
   "theta_rel_err": 0.09088047114321496,
   "r_p05_over_R": 0.32423917418619613,
   "S_steady": 0.4179325287674763,
   "mode": "mixed"
  },
  {
   "case": "dt_half",
   "fr": 0.3,
   "revs": 6.0,
   "dt": 5e-05,
   "n_total": 209,
   "theta_rel_err": 0.011227630725752008,
   "r_p05_over_R": 0.3527315112958767,
   "S_steady": 0.4301463651381145,
   "mode": "mixed"
  }
 ]
}```
