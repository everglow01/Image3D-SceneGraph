# 真实 12k 配对：尺寸修复不是最终答案

本摘要整理于 2026-10-22，只引用既有证据，不是新实验，也不改变原协议/旧 Gate。

- 原库与目标相机尺寸修复在冻结初始化、配置和相机序列下完成 12k。
- 12k raw PSNR 原库约 22.4463、修复约 22.1288；搬迁前总死亡数约 267090/270200。弱 owner 从 rank1 变为 rank0。
- 上游正确性修复不等于恢复不对称消失；12k 不能替代旧 15k 以后的长期退化复现。
- 监控面板结束标记缺失是历史限制，保留独立终态证据，不把面板状态补写成 finished。

## 可追溯证据

- [运行身份与逐文件 SHA](../records/camera-size-scene-pair-results-20261021-v1.json)
- [historical_assessment](../../../../../../../outputs/analysis/camera-size-scene-pair-results-20261021-v1/assessment.json)
- [report.md](../../../../../../../outputs/analysis/camera-size-scene-pair-results-20261021-v1/report.md)
- [frozen_protocol_copy](../../../../../../../outputs/analysis/camera-size-scene-pair-results-20261021-v1/incoming/protocol.json)

本地引用已校验；远端未重查。历史原报告保持原字节，描述“当时尚未修复”的历史语句不倒写。
