# 固定 B0 v1：前置失败，真实回放未开始

本摘要整理于 2026-10-22，只引用既有证据，不是新实验，也不改变原协议/旧 Gate。

- 执行与唯一监控均 finished=true、exit_code=1。前置 1 失败、2 通过；真实 B0 回放调用为 0。
- 解析一像素的 rank0 首次 backward 返回后 RNG guard 失败；该 trial 未验收，不补发五次通过。
- 变化在 Python RNG，NumPy/Torch CPU/CUDA RNG 未观察到变化；源码与离线状态推演吻合首次导入中的 getrandbits(24)。没有动态调用栈，不能声称唯一来源被动态证明。
- 真实实验未 prepare、未生成真实回放协议。v2 是独立新尝试，不改写本次失败。

## 可追溯证据

- [运行身份与逐文件 SHA](../records/fixed-b0-results-20261022-v1.json)
- [historical_assessment](../../../../../../../outputs/analysis/fixed-b0-results-20261022-v1/assessment.json)
- [report.md](../../../../../../../outputs/analysis/fixed-b0-results-20261022-v1/report.md)

本地引用已校验；远端未重查。历史原报告保持原字节，描述“当时尚未修复”的历史语句不倒写。
