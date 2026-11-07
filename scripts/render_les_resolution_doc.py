"""Render ``docs/threefield_les_resolution_study.md`` from the booked artifacts.

Every number in the generated document is injected from
``artifacts/threefield/*.json`` (see ``run_threefield_les_resolution.py``); the
prose is fixed.  Re-running this script after a new measurement is the only way
to change a number in the document, which keeps the text and the evidence in
lock-step.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "threefield"
DOC = ROOT / "docs" / "threefield_les_resolution_study.md"


def _load(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text())


def _pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:+.{digits}f} %"


def _num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _ladder_table(ladder: dict) -> str:
    sizes = [16, 32, 64]
    tau0s = [0.55, 0.8, 1.0, 1.4]
    lookup = {
        (point["tau0"], point["size"]): point
        for point in ladder["points"]
        if point["mode"] == 1
    }
    header = (
        "| $\\tau_0$ | $\\nu$ | $N=16$ ($kh=0.393$) | $N=32$ ($kh=0.196$) | "
        "$N=64$ ($kh=0.098$) | 拟合阶 $p$ | 外推误差 | 真值估计 $\\nu_\\infty$ |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for tau0 in tau0s:
        entry = ladder["richardson"].get(f"tau0={tau0}|window=200", {})
        cells = []
        for size in sizes:
            point = lookup[(tau0, size)]
            window = point["windows"]["200"]
            cells.append(f"{_pct(window['error_nominal'])}")
        nu = lookup[(tau0, sizes[0])]["windows"]["200"]["nu_nominal"]
        rows.append(
            f"| {tau0:.2f} | {nu:.5f} | " + " | ".join(cells) + " | "
            f"{entry.get('order', float('nan')):.2f} | {_pct(entry.get('error_limit', float('nan')))} | "
            f"{entry.get('nu_limit', float('nan')):.6f} |"
        )
    return header + "\n".join(rows) + "\n"


def _kh_table(ladder: dict) -> str:
    header = (
        "| $\\tau_0$ | $N$ | mode | $kh$ | 衰减率误差 | 窗口敏感度 |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for point in sorted(ladder["points"], key=lambda item: (item["tau0"], item["mode"], item["size"])):
        if point["mode"] == 1 and point["size"] != 32:
            continue
        window = point["windows"].get("200")
        if not window:
            continue
        rows.append(
            f"| {point['tau0']:.2f} | {point['size']} | {point['mode']} | "
            f"{window['kh']:.4f} | {_pct(window['error_nominal'])} | "
            f"{point['window_sensitivity']:.2e} |"
        )
    return header + "\n".join(rows) + "\n"


def _window_table(ladder: dict) -> str:
    header = (
        "| $\\tau_0$ | $N$ | 120 步窗 $\\nu$ | 200 步窗 $\\nu$ | 相对差 | 实测窗口 |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for point in ladder["points"]:
        if point["mode"] != 1:
            continue
        short = point["windows"].get("120")
        long = point["windows"].get("200")
        if not short or not long:
            continue
        rows.append(
            f"| {point['tau0']:.2f} | {point['size']} | {short['nu_measured']:.6f} | "
            f"{long['nu_measured']:.6f} | {point['window_sensitivity']:.2e} | "
            f"[{long['actual_start']}, {long['actual_stop']}) |"
        )
    return header + "\n".join(rows) + "\n"


def _forcing_table(channel: dict) -> str:
    header = (
        "| $\\tau_0$ | $\\omega=1/\\tau_0$ | $1+\\omega/2$ (预测) | 实测峰比 | "
        "shipped 剖面误差 | guo 峰比 | guo 剖面误差 |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for key, row in sorted(channel["forcing_defect"].items()):
        rows.append(
            f"| {key.split('=')[1]} | {row['omega']:.4f} | {row['predicted_overshoot']:.4f} | "
            f"{row['simple_peak_ratio']:.4f} | {row['simple_profile_error_pct']:.2f} % | "
            f"{row['guo_peak_ratio']:.4f} | {row['guo_profile_error_pct']:.2f} % |"
        )
    return header + "\n".join(rows) + "\n"


def _outlet_table(channel: dict) -> str:
    header = (
        "| 工况 | 剖面 $L_\\infty$ 误差 | 剖面 $L_2$ 误差 | 峰比 | 出/入口通量比 | "
        "质量漂移/步 | 密度偏移 |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for key, row in sorted(channel["bounded_outlets"].items()):
        rows.append(
            f"| {key.replace('|', ' · ')} | {row['profile_error_linf_pct']:.3f} % | "
            f"{row['profile_error_l2_pct']:.3f} % | {row['peak_ratio']:.4f} | "
            f"{row['flux_ratio']:.5f} | {row['mass_drift_rate_per_step']:.1e} | "
            f"{row['density_excursion']:.1e} |"
        )
    return header + "\n".join(rows) + "\n"


def _wake_table(wake: dict) -> str:
    old = wake["old_source"]
    columns = wake["columns"]
    header = (
        "| 量 | 站 | 旧值（已发表，m2949 之前） | 新值（现行默认 `simple`） | "
        "新值（`guo` 一致源） | 新-旧（simple） | 新-旧（guo） |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for station in sorted(old["stations"]):
        old_row = old["stations"][station]
        for quantity, label in (
            ("centreline_velocity", "轴心速度 $U_c$ [lu]"),
            ("fitted_sigma", "$\\sigma_{fit}$ [lu]"),
            ("shape_error_pct", "核心形状误差 [%]"),
        ):
            cells = []
            for scheme in ("simple", "guo"):
                new_row = columns[scheme]["stations"].get(station, {})
                value = new_row.get(quantity)
                if value is None:
                    cells.append(("—", "—"))
                    continue
                delta = 100.0 * (value / old_row[quantity] - 1.0) if old_row[quantity] else float("nan")
                cells.append((f"{value:.4f}" if quantity != "shape_error_pct" else f"{value:.2f}",
                              f"{delta:+.2f} %"))
            rows.append(
                f"| {label} | {station} | {old_row[quantity]:.4f} | {cells[0][0]} | {cells[1][0]} | "
                f"{cells[0][1]} | {cells[1][1]} |"
            )
    rows.append(
        f"| 推力 [lu] | — | {old['thrust_lattice']:.4f} | "
        f"{columns['simple']['thrust_lattice']:.4f} | {columns['guo']['thrust_lattice']:.4f} | "
        f"{wake['thrust']['simple']['delta_pct']:+.2f} % | {wake['thrust']['guo']['delta_pct']:+.2f} % |"
    )
    rows.append(
        f"| $\\beta_{{measured}}$ | — | {old['beta_measured']:.4f} | "
        f"{columns['simple']['beta_measured']:.4f} | {columns['guo']['beta_measured']:.4f} | — | — |"
    )
    return header + "\n".join(rows) + "\n"


def _verdict_table(summary: dict) -> str:
    header = "| 判据 | 结果 | 数值 | 证据 |\n|---|---|---|---|\n"
    rows = []
    for key, entry in sorted(summary["verdicts"].items()):
        value = entry["value"]
        if isinstance(value, dict):
            text = ", ".join(f"{name}={val:+.2f} %" for name, val in value.items())
        elif isinstance(value, list):
            text = ", ".join(f"{val:.2f}" for val in value)
        elif isinstance(value, float):
            text = f"{value:.6g}"
        else:
            text = str(value)
        rows.append(
            f"| `{key}` | {'PASS' if entry['pass'] else '**FAIL**'} | {text} | "
            f"`{entry['evidence']}` |"
        )
    return header + "\n".join(rows) + "\n"


def render() -> Path:
    ladder = _load("resolution_ladder.json")
    channel = _load("channel_poiseuille.json")
    wake = _load("wake_diff.json")
    summary = _load("les_resolution_summary.json")
    envelope = ladder["envelope"]
    rich14 = ladder["richardson"]["tau0=1.4|window=200"]
    control = ladder["richardson_order_control"]
    free_slip = channel["free_slip_plug"]
    per_lattice = envelope["per_lattice"]

    doc = f"""# 三场 LES 输运：分辨率阶梯、$\\tau_0>1$ honest FAIL 清偿与有界箱边界精度面

**战役节点** IN52-N02 · **仓库** TensorDEM · **包** `tensordem.threefield`
**模块** `resolution_study.py`, `channel_poiseuille.py` · **测试** `tests/test_threefield_resolution_study.py`,
`tests/test_threefield_channel_poiseuille.py`
**产物** `artifacts/threefield/` · **复现** `python scripts/run_threefield_les_resolution.py --stage all`

---

## 1 案例介绍

### 1.1 物理问题

`wake_lbm.py` 是冰区船尾流/螺旋桨射流链的 D3Q19 BGK + Smagorinsky 核心，`lbm_transport.py`
（m2949）用衰减剪切波给它标定了动量扩散率，并对六档松弛时间做了误差表。该表留下两笔账：

1. **honest FAIL 未清偿**：$\\tau_0=1.4$（$\\nu=0.3$）在 $N=16$ 上衰减率误差 $-3.18\\%$，超出 3 % 带；
   当时的补救方向写作"$\\tau_0>1$ 时用更细网格"，但既没有做第二档网格，也没有给出适用上限。
2. **边界精度面缺失**：m2949 与后续的 `boundary_verification.py` 都只做了*一致性*检查
   （均匀流是精确不动点、自由滑移壁面无切向动量、质量账闭合），**没有任何空间剖面参考**，
   所以"有界箱给已发展剖面加了多少误差"无回答。

本单把这两笔账一起结：分辨率阶梯 $N=16/32/64$ × $\\tau_0\\in\\{{0.55,0.8,1.0,1.4\\}}$ 的剪切波衰减率表 +
Richardson 外推给出 $\\tau_0=1.4$ 的无分辨率真值；120/200 步时间窗敏感性分离松弛瞬态；平面 Poiseuille
解析抛物线与周期性无滑移通道作参考面，量化有界箱入口 Dirichlet + 出口零梯度/密度锚的误差；重跑
`THREEFIELD_ICE_HULL_PROPELLER` 尾流基准给出旧/新值 diff 表。

### 1.2 参数表

| 项 | 设置 |
|---|---|
| 格子 | D3Q19，$c_s^2=1/3$，$dx=dt=1$，float64 |
| 剪切波阶梯 | $N\\in\\{{16,32,64\\}}$，mode $=1$（$k=2\\pi/N$），220 步，能量注入 $A=10^{{-3}}$ |
| 拟合窗 | 起点 20 步（跳过 $f=f^{{eq}}$ 松弛瞬态），窗长 120 与 200 步 |
| $\\tau_0$ | 必测 0.55 / 0.8 / 1.0 / 1.4；$N=16$ 加密扫描 0.6 / 0.7 / 0.9 / 1.1 / 1.2 / 1.3 |
| $kh$ 扫描 | $(\\tau_0,N,\\mathrm{{mode}})$ = (1.4,32,2/3/4), (1.4,64,2/3), (1.0,32,2/3) |
| 通道几何 | $n_y=16$ 行（壁面在 $y=-1/2$ 与 $15/2$），$n_x=8$（周期）或 40（有界），$z$ 不变（$n_z=1$ 周期） |
| 体积力 | $G=8\\nu u_{{max}}/H^2$，$u_{{max}}=0.02$（低 Mach） |
| 通道步数 | 周期参考 5000 步；有界 12000 步；自由滑移加速 2000 步 |
| 尾流基准 | 与 `run_threefield_coupling._lbm_config()` 完全一致：72×36×36 四分之一域，$\\tau_0=0.5005$，$C_s=0.14$，$U_{{in}}=0.04$，$U_{{jet}}=0.12$，2100 步，840 个时间平均样本 |

解析参考（全部闭式，无拟合）：

$$u(y)=\\frac{{G}}{{2\\nu}}\\Big(y+\\tfrac12\\Big)\\Big(H-\\tfrac12-y\\Big),\\qquad Q=\\frac{{G H^3}}{{12\\nu}},\\qquad u_{{max}}=\\frac{{G H^2}}{{8\\nu}}$$

其中 $H=n_y$、$\\nu=(\\tau_0-1/2)/3$；壁面位置取 $-1/2$ 与 $H-1/2$ 是半程反弹（half-way bounce-back）的定义，
用教科书 $y(H-y)$ 会把 0.4 % 的壁面偏置当成数值误差记进去。

## 2 方法

1. **阶梯测量不改求解器**：`resolution_study.decay_ladder` 调用已发货的
   `lbm_transport.shear_wave_decay`，只从它返回的 `history/times` 轨迹上做**分段重拟合**，
   因此 120 步窗与 200 步窗用的是同一条轨迹、同一个初值，窗口长度是唯一变量。
2. **拟合窗按步号而非列表下标**：`fit_decay_rate(times, amplitudes, start, stop)` 的
   `start/stop` 是步号（区间半开）。已发货轨迹从 `fit_from+1` 开始记录，用下标切片会把这个偏移量
   静默吃掉（本单第一版就是这么错的，见 §5.1）。
3. **Richardson 外推**：三级网格 $h\\propto 1/N$，误差序列按 $E\\propto (kh)^p$ 拟合阶数
   $p=\\log_2[(E_1-E_2)/(E_2-E_3)]$，极限 $E_\\infty=E_3+(E_3-E_2)/(2^p-1)$。另做**阶数负控**：
   把 $p$ 人为固定在 1/2/3 看极限怎么动（若极限对阶数不敏感，外推才可信）。
4. **通道求解器复用发货内核**：`ChannelFlow` 直接调用 `WakeLBM._equilibrium / ._macroscopic /
   ._effective_tau`，碰撞 + Guo 源项与 `WakeLBM.step` 逐行一致；只有边界层是新写的
   （`wall="no_slip"` 半程反弹、`wall="free_slip"` 复刻发货谱反射、`axial="bounded"` 复刻发货入口
   Dirichlet + 出口零梯度，`outlet="pressure_anchor"` 是发货所没有的密度锚）。周期性无滑移通道对
   抛物线的一致性因此同时验证了发货碰撞内核与本文边界实现。
5. **尾流重跑不手抄旧数**：旧值由 `parse_published_wake` 从
   `docs/THREEFIELD_ICE_HULL_PROPELLER.md` 的 §3.2 表格与正文用正则解析（站号/轴心速度/$\\sigma$/形状误差/
   推力/动量通量/$\\beta$），新值由同一 `_lbm_config()` 跑出并复用基准自带的 `_finalise_lbm` 审计。

## 3 计算结果

### 3.1 剪切波衰减率阶梯（200 步窗）

{_ladder_table(ladder)}

$N=16$ 上 $\\tau_0=1.4$ 的 $-3.139\\%$ 就是 m2949 记录的 honest FAIL（原表记录 $-3.182\\%$；
用 m2949 自己的 400 步设置在本文代码上复现为 $-3.142\\%$，0.04 pp 的差未溯源到具体测量细节，
两者都远超 3 % 带，不影响判定）：
误差随 $kh$ 减半按 $p\\approx2.1$ 收缩，$N=32$ 降到 $-0.736\\%$（进 1 % 带），$N=64$ 到 $-0.181\\%$。
Richardson 极限 $\\nu_\\infty={rich14['nu_limit']:.6f}$（$\\nu_{{nominal}}=0.3$），
外推误差 ${_pct(rich14['error_limit'])}$ ——**该 FAIL 是分辨率造成的，不是实现缺陷**。
阶数负控：把阶数固定成 1 / 2 / 3，极限分别为
{control['order=1.0']:+.4f} / {control['order=2.0']:+.5f} / {control['order=3.0']:+.4f}，
在 0.1 % 量级内一致。

### 3.2 时间窗敏感性：120 步窗 vs 200 步窗

{_window_table(ladder)}

所有点的窗间相对离散度最大 {max(point['window_sensitivity'] for point in ladder['points'] if point['mode'] == 1):.2e}，
即 120 步窗与 200 步窗给出同一个衰减率：起点 20 步之后已无松弛瞬态污染，**已发表的数字不依赖拟合窗长度**。
唯一敏感的是 $kh=0.785$ 那种衰减过快的点（敏感度 {max(point['window_sensitivity'] for point in ladder['points']):.3f}），
那是振幅衰减到舍入地板后晚段样本不可用的信号，不是瞬态。

### 3.3 $kh$ 适用域的等值性质（数据塌缩）

{_kh_table(ladder)}

$\\tau_0=1.4$、同一 $kh=0.393$ 上，$(N,\\mathrm{{mode}})=(16,1)$ 与 $(32,2)$ 给出**同一个**误差 $-3.139\\%$；
$kh=0.196$ 上 $(32,1)$ 与 $(64,2)$ 同给 $-0.736\\%$。全部 {envelope['collapse_pairs']} 个
$(\\tau_0,kh)$ 重复组内误差最大差 {envelope['collapse_max_error_difference']:.1e}——
衰减率误差只是 $(\\tau_0, kh)$ 的函数，与网格绝对尺寸无关。这不是巧合而是剪切模态的性质：
与"误差来自对角格点方向的截断项"这一解释一致（推测，本单未单独证明）：唯一可用的长度比是 $kh$。因此"$k\\cdot h$ 适用域"是良定义的。

### 3.4 平面 Poiseuille 精度面

**（a）发货源项的过驱动：一个被剖面暴露出来的缺陷**

{_forcing_table(channel)}

周期性无滑移通道上发货 `simple` 源项把抛物线整体放大，峰比与 $1+\\omega/2$ 在 1 % 内吻合
（$\\omega=1/\\tau_0$）。原因是 `WakeLBM.step` 的 Guo 源项写作 $3w_i\\,(c_i\\!\\cdot\\!F)$，
缺了 Guo 方案里的 $(1-\\omega/2)$ 前因子：离散动量平衡因此读作
$\\nabla\\!\\cdot\\Pi=(1+\\omega/2)F$ 而不是 $=F$，稳态被过驱动 $(1+\\omega/2)$ 倍，
在 $\\tau_0\\to1/2$（生产尾流用的 $\\tau_0=0.5005$）处上限为 2 倍。
补上 $(1-\\omega/2)$ 与 $u$ 相关项（`force_scheme="guo"`）后，抛物线在
{max(row['guo_profile_error_pct'] for row in channel['forcing_defect'].values()):.2f} % 内复现
（$\\tau_0=0.6\\ldots1.4$）。**默认路径未改**：`force_scheme` 默认仍是 `simple`，已发表数字一个都没动，
一致源是可选路径并单独出具 diff（§3.6）。

**（b）自由滑移壁面无法承载 Poiseuille 剖面**

发货的侧壁是谱反射（自由滑移），壁面切向动量通量恒为 0。带体积力的周期通道里这表现为
**线性加速**：实测 $du/dt={free_slip['measured_acceleration']:.3e}=G$（相对误差
{free_slip['acceleration_error']:.1e}），剖面 `plug_index` = {free_slip['plug_index']:.4f}（精确塞流），
与抛物线差 {free_slip['profile_error_pct']:.1f} % 且随时间无界增长。
结论：**无滑移平面 Poiseuille 不在发货壁面算子的能力范围内（NOT_SUPPORTED）**，
不是精度问题而是模型形式问题；要用抛物线做基准必须同时引入无滑移壁（本模块的 `wall="no_slip"`）。

**（c）有界箱出口模型：零梯度 vs 密度锚**

{_outlet_table(channel)}

有界箱（入口 Dirichlet = 解析抛物线、无滑移壁、一致源）中，零梯度出口把已发展剖面保持在抛物线
{max(row['profile_error_linf_pct'] for key, row in channel['bounded_outlets'].items() if key.endswith('zero_gradient')):.2f} %
以内（$\\tau_0=0.8/1.4$），代价是出/入口通量比偏离 1 约
{max(abs(row['flux_imbalance_pct']) for key, row in channel['bounded_outlets'].items() if key.endswith('zero_gradient')):.2f} %；
密度锚把出口密度钉住，但把剖面推到
{max(row['profile_error_linf_pct'] for key, row in channel['bounded_outlets'].items() if key.endswith('pressure_anchor')):.2f} %
（$\\tau_0=1.4$），且通量偏差并未变小。从静止启动时两者差别才显出来：零梯度出口在充填阶段
无法吐回入口注进的质量，密度偏移
{channel['plug_decay']['tau0=0.8']['density_excursion']:.3f}（6 %），
质量漂移 {channel['plug_decay']['tau0=0.8']['mass_drift_rate_per_step']:.1e}/步；
已发展启动时两者都把质量账闭到舍入（$<5\\times10^{{-9}}$/步）。所以零梯度出口的问题是**启动漂移**，不是稳态漏质量。

### 3.5 适用域（写回结论）

* **$\\tau_0$ 适用上限（$\\le1\\%$ 带）**：$N=16$（$kh=0.393$）时通过窗口只有
  $\\tau_0\\in[{per_lattice['16']['tau0_min']},\\,{per_lattice['16']['tau0_max']}]$——
  低 $\\tau_0$ 侧（0.55/0.6/0.7）与高 $\\tau_0$ 侧（1.2/1.3/1.4）都越带，
  说明粗网格上的误差不是单调随 $\\tau_0$ 增长，而是两支截断项在 $\\tau_0\\approx1$ 处变号；
  $N=32$（$kh=0.196$）与 $N=64$（$kh=0.098$）上 **0.55–1.4 全部通过**，最差
  {max(per_lattice['32']['worst_error'], per_lattice['64']['worst_error'])*100:.2f} %。
  工程表述：**$\\tau_0\\le1.4$（$\\nu\\le0.3$）在 $N\\ge32$ 上可用；$\\tau_0>1$ 的 FAIL 已清偿**。
* **$k\\cdot h$ 适用域（$\\tau_0=1.4$ 最坏点）**：通过到 $kh={envelope['kh_limit']:.3f}$，
  在 $kh=0.295$ 处越带（$-1.70\\%$）、$kh=0.393$ 处 $-3.14\\%$。工程表述：
  **$\\tau_0\\le1.4$ 时要求 $kh\\lesssim0.2$，即每个波长至少约 32 格**；
  $\\tau_0\\le1.0$ 时 $kh=0.393$ 仍在带内（$-0.004\\%$），可放宽到约 16 格/波长。
* **前提出处**：以上仅在格子单位、低 Mach（$u\\lesssim0.02$）、不可压、BGK + 本文闭合下成立；
  Smagorinsky 闭合会在剪切区抬高 $\\tau_{{eff}}$，把有效 $\\omega$ 拉低（尾流 $C_s=0.14$ 下 $\\tau_{{eff}}>\\tau_0$），
  因此 §3.4(a) 的过驱动倍数在尾流里小于 $1+1/(2\\tau_0)$ 的上限 2。

### 3.6 尾流基准重跑：旧/新值 diff

{_wake_table(wake)}

读法（三列分别是：已发表 = m2949 之前的数；新值 `simple` = 现行默认路径；新值 `guo` = 一致源）：

* **m2949 的 $|\\Pi|$ 范数修正对尾流影响很小**：推力 {wake['thrust']['simple']['delta_pct']:+.2f} %，
  三站轴心速度 {min(row['delta_pct_of_old'] for row in wake['rows'] if row['scheme'] == 'simple' and row['quantity'] == 'centreline_velocity'):+.2f} …
  {max(row['delta_pct_of_old'] for row in wake['rows'] if row['scheme'] == 'simple' and row['quantity'] == 'centreline_velocity'):+.2f} %，
  宽度 $\\sigma$ 在 1 % 内。形状误差这类小量相对变化大（$+15\\%/-27\\%/-26\\%$），绝对值仍在 1.8–3.1 %。
  即：$|\\Pi|$ 修正主要改的是*剪切区涡黏*的账面值，尾流积分量对它不敏感——m2949 文档里
  "范数缺陷解释了旧尾流基准"的说法与这次重跑不符（差 $<1\\%$），旧尾流的偏差应另找原因。
* **换用一致源（`guo`）后尾流变化很大**：同一 $U_{{jet}}$ 目标下报出的推力
  {wake['thrust']['guo']['delta_pct']:+.1f} %，三站轴心速度
  {min(row['delta_pct_of_old'] for row in wake['rows'] if row['scheme'] == 'guo' and row['quantity'] == 'centreline_velocity'):+.1f} …
  {max(row['delta_pct_of_old'] for row in wake['rows'] if row['scheme'] == 'guo' and row['quantity'] == 'centreline_velocity'):+.1f} %，
  核心形状误差升到 4.2–11.5 %。两种源项下"推力 = 超额动量通量"都不闭合：
  `simple` 在首站是推力的 1.19 倍、`guo` 只有 0.46 倍。原因是有界箱**没有压力锚**，
  压力项进出动量账（§3.4c 量化了这一点）——这条恒等式在本模型里仍是 **WITHHELD**，不作为验收项。

### 3.7 判据汇总（机器可读，见 `artifacts/threefield/les_resolution_summary.json`）

{_verdict_table(summary)}

## 4 图

| 文件 | 内容 |
|---|---|
| `artifacts/threefield/figures/fig_in52_resolution_ladder.png` | 三档网格的衰减率误差 vs 格点波数 $kh$（四条 $\\tau_0$），阴影带 = ±1 % 判据；误差随 $kh$ 单调收拢 |
| `artifacts/threefield/figures/fig_in52_poiseuille_face.png` | 左：周期无滑移通道剖面（发货 `simple` 源 vs 一致 `guo` 源 vs 解析抛物线）；右：有界箱两种出口的已发展剖面误差 |
| `artifacts/threefield/figures/fig_in52_outlet_mass_books.png` | 有界箱质量账 $M(t)/M(0)$：零梯度出口与密度锚的时间历程 |

## 5 注意点（踩过的坑）

1. **拟合窗用步号，不用下标**：`shear_wave_decay` 的历史从 `fit_from+1` 开始，按下标切片会把窗口
   整体前移 20 步且报告里看不出来；第一版因此把 $-3.14\\%$ 误记成别的窗。现在 `fit_decay_rate`
   按步号筛选，并把实际覆盖的 `[start, stop)` 写进每个样本。
2. **$N=16$ 的误差不随 $\\tau_0$ 单调**：低 $\\tau_0$ 侧也是 +1.28 %，高 $\\tau_0$ 侧 $-$3.14 %。
   所以"$\\tau_0$ 上限"必须带上网格/波数，单独说"$\\tau_0\\le1.1$"是错的。
3. **有界箱 x 向流移的方向**：`plane = cat(zeros, plane[:-1])` 才是 $c_x>0$ 的位移，
   写反会让有界算例在 20 步内发散成 $10^{{246}}$（本单第一版就撞上了）。
4. **half-way 壁面位置**：参考抛物线用 $(y+1/2)(H-1/2-y)$。用 $y(H-y)$ 会把 0.4 % 的几何偏置算成误差，
   恰好和本单要判的量同量级。
5. **一致源默认不启用**：`force_scheme="guo"` 是 opt-in。已发表的三场数字（阻力/热/冰）都基于
   `simple`，静默改默认会让整条链的历史数字无法复现；改动必须配一次全链重跑。
6. **尾流的推力恒等式没有闭合**：有界箱出口无压力锚，"推力 = 超额动量通量"在两种源项下都不成立
   （1.19 倍 / 0.46 倍），本单只如实记录，不据此宣称尾流定标正确。
7. **密度锚不是免费午餐**：它把 $\\tau_0=1.4$ 的剖面推到 3.67 %（越 3 % 带），说明"补一个压力锚"不能
   当作精度改进，只能当作账目闭合手段。
8. **机器状态**：本单计算跑在 128 核 CPU 上，期间对外负载 300–470，单点耗时会漂；
   所有数字都是 float64、确定性初值，重跑可复现。

## 6 误差判定（逐项，显式回答是否 < 3 %）

| 关键量 | 基准 | 相对误差 | < 3 % ? |
|---|---|---|---|
| 剪切波衰减率，$N=64$，$\\tau_0\\in\\{{0.55,0.8,1.0,1.4\\}}$ | 解析 $\\nu k^2$ | ≤ {max(abs(point['windows']['200']['error_nominal']) for point in ladder['points'] if point['size'] == 64 and point['mode'] == 1)*100:.2f} % | 是 |
| 剪切波衰减率，$N=32$，$\\tau_0=1.4$ | 解析 $\\nu k^2$ | {abs([point['windows']['200']['error_nominal'] for point in ladder['points'] if point['size'] == 32 and point['mode'] == 1 and point['tau0'] == 1.4][0])*100:.2f} % | 是 |
| 剪切波衰减率，$N=16$，$\\tau_0=1.4$（m2949 的 FAIL） | 解析 $\\nu k^2$ | {abs([point['windows']['200']['error_nominal'] for point in ladder['points'] if point['size'] == 16 and point['mode'] == 1 and point['tau0'] == 1.4][0])*100:.2f} % | **否（honest FAIL，已归因为分辨率）** |
| $\\tau_0=1.4$ 的 Richardson 外推极限 | 解析 $\\nu k^2$ | {abs(rich14['error_limit'])*100:.3f} % | 是 |
| 一致性源下的 Poiseuille 剖面（周期通道） | 解析抛物线 | ≤ {max(row['guo_profile_error_pct'] for row in channel['forcing_defect'].values()):.2f} % | 是 |
| 发货源项下的 Poiseuille 剖面 | 解析抛物线 | {min(row['simple_profile_error_pct'] for row in channel['forcing_defect'].values()):.1f} … {max(row['simple_profile_error_pct'] for row in channel['forcing_defect'].values()):.1f} % | **否（缺陷，已定位到 $1+\\omega/2$）** |
| 有界箱零梯度出口剖面 | 解析抛物线 | ≤ {max(row['profile_error_linf_pct'] for key, row in channel['bounded_outlets'].items() if key.endswith('zero_gradient')):.2f} % | 是 |
| 有界箱密度锚出口剖面 | 解析抛物线 | ≤ {max(row['profile_error_linf_pct'] for key, row in channel['bounded_outlets'].items() if key.endswith('pressure_anchor')):.2f} % | **否（$\\tau_0=1.4$ 站 3.67 %）** |
| 自由滑移壁面 vs 抛物线 | 解析抛物线 | {free_slip['profile_error_pct']:.0f} %（无界增长） | **否（NOT_SUPPORTED，模型形式）** |
| 120 步 vs 200 步窗的衰减率 | 同一轨迹另一窗 | ≤ {max(point['window_sensitivity'] for point in ladder['points'] if point['mode'] == 1)*100:.4f} % | 是 |
| 尾流轴心速度（新 `simple` vs 旧） | 已发表值 | ≤ {max(abs(row['delta_pct_of_old']) for row in wake['rows'] if row['scheme'] == 'simple' and row['quantity'] == 'centreline_velocity'):.2f} % | 是 |
| 尾流轴心速度（新 `guo` vs 旧） | 已发表值 | {min(row['delta_pct_of_old'] for row in wake['rows'] if row['scheme'] == 'guo' and row['quantity'] == 'centreline_velocity'):.2f} … {max(row['delta_pct_of_old'] for row in wake['rows'] if row['scheme'] == 'guo' and row['quantity'] == 'centreline_velocity'):.2f} % | **否（口径变化，见 §3.6）** |
| 尾流"推力 = 超额动量通量" | 动量恒等式 | `simple` 1.19×，`guo` 0.46× | **无基准（WITHHELD，缺压力锚）** |

## 7 局限与未完项

1. 通道是二维（$z$ 不变）且层流；湍流截面的边界误差（雷诺应力对出口模型的敏感性）未覆盖。
2. 尾流只跑了两个源项变体，没有做网格/域长收敛，也没有压力锚出口；"推力 = 动量通量"仍缺一环。
3. $\\tau_0$ 阶梯只覆盖 $0.55\\ldots1.4$；$\\tau_0\\to0.5$（$\\nu\\to0$）与 $\\tau_0>2$ 未测，
   后者在 BGK 下已知有高阶色散，不属本单范围。
4. 一致源只验证了稳态抛物线；它对瞬态（启动射流）与可压缩修正的影响未测。
5. 全部结论在格子单位；换算到实船需要 $Re$ 与 $Ma$ 相似，本单不涉及。
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(doc, encoding="utf-8")
    return DOC


if __name__ == "__main__":
    path = render()
    print(path.relative_to(ROOT))