### 表 A1 — 层数与分支（该分支由归档原文给出，未由任何轮次判定）

| 量 | 值 | 依据 |
|---|---|---|
| $h_{eff}/d$（自由盘层数） | 15.00 | 算例 `gap=0.150`, $d=0.0100$ |
| $\nu_{max}$ 厚层（$\mu=0.4$） | 0.81 | Eq.13 / `reference_2d` |
| $\nu_{max}$ 薄层（$\mu=0.4$） | 0.78 | 归档原文 `thin_layer_packing` |
| 分支分隔层数 | 原文未给出 | the archived excerpt does not give the layer count that separates its 'thin' from its 'thick' case; the branch is therefore carried as an envelope, not resolved |

### 表 A2 — $\phi$ 逐点在两个分支下的判定（容差 10%）

| 文件 | $I$ | $I$ 来源 | $\phi$ | $\nu_{ref}$ 厚 | err %% | 厚 | $\nu_{ref}$ 薄 | err %% | 薄 |
|---|---|---|---|---|---|---|---|---|---|
| prof_r31_w0.5_I0.05_s0_smoke.json | 0.11057 | I_bulk | 0.8043 | 0.7768 | +3.53 | PASS | 0.7468 | +7.69 | PASS |
| prof_r31_w0.5_I0.2_s0_smoke.json | 0.45053 | I_bulk | 0.7839 | 0.6748 | +16.15 | FAIL | 0.6448 | +21.56 | FAIL |

### 表 A3 — 分支敏感性结论

| 项 | 结论 |
|---|---|
| $\phi$ 判定（厚/薄） | FAIL / FAIL |
| $\phi$ 最差绝对误差（厚/薄） | 16.15 % / 21.56 % |
| 分支敏感？ | 否 |
| $\phi$ 结论 | BRANCH-INSENSITIVE: both nu_max branches give FAIL, so the thin/thick ambiguity does not change the phi verdict. |
| $\mu$ 结论 | the friction law is thickness-invariant in the archived text ('the dilatancy and friction laws are not affected'), so no mu verdict may be attributed to the branch choice |
