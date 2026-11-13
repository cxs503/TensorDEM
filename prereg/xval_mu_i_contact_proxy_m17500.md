# 预注册 · XVAL-DEM-MU-I-CONTACT-PROXY-NEGATIVE (m17500)

**冻结时间**：跑数前写入（脚本 `scripts/xval_mu_i_contact_proxy_m17500.py` 内 `DEFAULT_TOL=0.15` 同步冻结）

| 项 | 内容 |
|---|---|
| 决策 | 接触局部惯性数代理 I_c 能否在 15 % 内替代体相惯性数 I（即 mu(I_c) 闭合能否代表床体实际摩擦） |
| 备择 | H_ref=能（代理即同一无量纲数）；H_neg=不能（oss_solutions#421 的负结论） |
| 结果量 1 | `i_mapping_error = abs(I_c/I_bulk - 1)`，I_c 取 f_n 加权均值，I_bulk 取测量剪切率剖面 + Love-Weber 板压 |
| 结果量 2 | `mu_error = abs(mu_proxy/mu_bulk - 1)`，mu_proxy=f_n 加权 mu(I_c)，mu_bulk=tau/P |
| 结果量 3 | `shortfall = (mu_c - mu(I_c))/mu_c`，mu_c=0.5（DEM 接触库仑系数），结构性指标 |
| 判据 | 全部行 `i_mapping_error < 0.15` 且 `mu_error < 0.15` → 代理成立；任一行超出 → 代理被否 |
| 参考合格门 | 体相路线自身 vs Jop(2006) 常数须 <15 %，否则**绝对摩擦比较作废**，只保留无量纲映射结论（预注册） |
| 样本 | I_target ∈ {0.01, 0.1, 0.6}（跨准静态→碰撞带），P=2 kPa，kn=1e3，r=5 mm，rho=2500，mu_c=0.5，seed=5279 |
| 敏感性 | S1 P: 2 kPa→8 kPa @I=0.1；S2 kn: 1e3→4e3 @I=0.1；S3 数值：末 20 % 窗 vs 尾 60 % 窗 |
| 停止规则 | 若参考合格门失败即只报映射误差，不再追加算例；预算 7200 s，实耗 346 s |
| 分析 | 逐行对照表 + 最差误差；不做事后阈值调整 |
