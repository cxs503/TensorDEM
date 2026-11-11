# P152W1-N04 锚点 A/B 收口：预注册休止角基准（协议 v2，TensorDEM）

- 节点：`P152W1-N04-repose-v2` · 预注册 `prereg/P152W1_N04_repose_protocol_v2_prereg.json` (sha256 `c80ca8b356a60000…`)
- 工件：`artifacts/P152W1-N04/repose_benchmark.json` (schema `tensordem.friction_surface.repose_benchmark/v1`) · commit `6bfda4dd9751`
- 复现命令：`python -m tensordem.friction_surface --particles 150 --bisection-steps 3 --jobs 12 --out artifacts/P152W1-N04/repose_benchmark.json`

## 1 案例介绍

休止角（angle of repose）是颗粒材料摩擦/滚动阻力的宏观指纹，也是 DEM 微参数标定的经典标定面。本节点把 TensorDEM 的摩擦接触面（切向 `F_t = min(mu F_n, k_t delta_t) + c_t v_t`、滚动阻力矩）放到「散体从固定高度喷嘴倾泻成堆」的实验室类比中检验：

| 参数 | 值 |
|---|---|
| `particles` | `150` |
| `bisection_steps` | `3` |
| `seeds` | `[15204, 15205]` |
| `source_mode` | `fixed` |
| `max_time` | `None` |
| 材料 | `glass_beads` (mu=0.30), `quartz_sand` (mu=0.50) |
| 负控 | mu = mu_rolling = 0，判据 theta_r < 5.00 deg |
| 判据 | 相对误差 ≤ 10.00 %，文献带逐项给出 |

预注册阈值、材料微参数（文献值，非本仓拟合）与实验带全部从 `prereg/P152W1_N04_repose_protocol_v2_prereg.json` 读取，SHA-256 写入工件，避免「跑完再调门限」。

## 2 怎么做的

1. **协议 v2 倾泻**：喷嘴固定在 `drop_height`，撒料节奏 = 电荷穿出喷嘴区的自由落体时间（`nozzle_clear_time()`），因此不接受「抬高喷嘴」这类被本节点 §5.3 否定过的做法；沉积体填满喷嘴板层后停料且不复开。
2. **测量**：上包络径向分箱 + 15–90 % 坡面窗口线性拟合（`heap_surface_angle`），与 v1 一字未改，因此本轮数字与历史数字可比。
3. **标定**：以 `glass_beads` 为标定材料，对单一形状参数 `mu_rolling` 在 [0.02, 0.40] 上做 3 步二分，目标 23.00 deg；得到的 `mu_rolling*` 原封不动用于砂的样本外预测。
4. **规模/算力**：150 球 × 2 种子 × 12 进程；墙钟 367.1 s（36cb7828dbd5，Python 3.11.6，numpy 2.4.6）。
5. **工件**：每例的 `reached_rest` / `pour_closed` / `nozzle_z_max` / `bursts_released` 全部落盘，判定可由第三方脚本重算（判据来自预注册，非代码常量）。

## 3 计算结果

| 材料 | 参考 theta [deg] | 文献带 | 实测 theta [deg] | std | n | 相对误差 | 带内 | 状态 |
|---|---|---|---|---|---|---|---|---|
| glass_beads | 23.00 | 21.00–25.00 | 21.51 | 3.56 | 2 | +6.46 % | 是 | **PASS** |
| quartz_sand | 32.00 | 30.00–34.00 | 24.61 | 4.43 | 2 | +23.08 % | 否 | **FAIL** |

负控（mu=mu_rolling=0）：theta_r = 0.00 deg（判据 < 5.00 deg，PASS），逐次 ['0.00', '0.00']。

`mu_rolling*` = 0.2813；二分历史：['22.30', '23.45', '22.55'] deg。

总体判定：**FAIL**（2/3 判据 PASS）。

## 4 结果图

- `fig1_deposits.png`：见正文表格（程序化生成，数据来自工件）
- `fig2_calibration.png`：见正文表格（程序化生成，数据来自工件）
- `fig3_reference_vs_sim.png`：见正文表格（程序化生成，数据来自工件）

## 5 注意点

1. 文献带不是本仓实测：玻璃珠 21–25 deg、石英砂 30–34 deg 取自常规干态单分散体系，属于外部基准，不能反过来当作本仓的标定数据。
2. 单形状参数（`mu_rolling`）代替真实颗粒形状；砂是**样本外**预测，若失败即为形状近似的证伪，不作任何事后调参。
3. 粒子数 150 是为算力选取，未做有限尺寸收敛研究；本轮不声明尺寸收敛性。
4. 停料由几何（喷嘴板层被沉积体填满）决定，停料后不复开——若堆体尚未稳定而停料，测到的是「停料时」的表面角，`reached_rest` 字段逐例记录该风险。
5. 5 个 `theta(mu)` 面点只用于图示趋势，不参与任何判据。

## 6 误差判定

| 判据 | 基准 | 实测 | 误差 | 阈值 | 是否达标 |
|---|---|---|---|---|---|
| B/glass_beads | 23.00 deg 文献值 | 21.51 deg | +6.46 % | ≤ 10.00 % | **是** |
| B/quartz_sand | 32.00 deg 文献值 | 24.61 deg | +23.08 % | ≤ 10.00 % | **否** |
| A/负控 | theta_r < 5.00 deg | 0.00 deg | — | ≤ 5.00 deg | **是** |

**是否 <3%？** 否：本轮没有任何材料达到 3 % 以内；glass_beads = +6.46 %、quartz_sand = +23.08 %。（判据线本身是 10 %，3 % 是更严的自检线；站在 10 % 门限上不等于满足 3 %。）

> WITHHELD means the anchor produced no adjudicable number (NaN angle or no run reached rest); it is neither a pass nor a pass by default.

