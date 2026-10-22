# 固定 B0 v2：有限重复性通过，长期因果未证

本摘要整理于 2026-10-22，只引用既有证据，不是新实验，也不改变原协议/旧 Gate。

- 前置 3 项通过；两相机各 5 次真实原 backward，共 10 次；另有前置 10 次，无额外热身、无训练更新。
- 20 配对、80 项张量比较全部在原定容差内，非有限为 0；约 90.13% 非零配对元素差异≤4 ULP。配对不是独立重复样本，元素数不是 Gaussian 数。
- 初始化显式加载原扩展并单独审计，测量期间输入/RNG 不变，不恢复 RNG 掩盖副作用。
- 六点 Adam 解释来自历史 A/A，不是本轮 B1 回传产生的新 P2/P3；B1 容差内不豁免下游超容差。
- 不证明全局误差上界、唯一微观机制或长期失衡因果。旧 Gate1 未追认，原六分支未恢复。

## 可追溯证据

- [运行身份与逐文件 SHA](../records/fixed-b0-results-20261022-v2.json)
- [historical_assessment](../../../../../../../outputs/analysis/fixed-b0-results-20261022-v2/assessment.json)
- [report.md](../../../../../../../outputs/analysis/fixed-b0-results-20261022-v2/report.md)
- [independent-metrics.json](../../../../../../../outputs/analysis/fixed-b0-results-20261022-v2/independent-metrics.json)
- [frozen_protocol_copy](../../../../../../../outputs/analysis/fixed-b0-results-20261022-v2/incoming/protocol.json)

本地引用已校验；远端未重查。历史原报告保持原字节，描述“当时尚未修复”的历史语句不倒写。
