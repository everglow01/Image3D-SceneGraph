# 五臂对照：颜色改善不等于恢复失衡已解决

本摘要整理于 2026-10-22，只引用既有证据，不是新实验，也不改变原协议/旧 Gate。

- 五臂均执行：baseline/raw_loss 完成 30k；floor、目标矩清零、位置日程失败。不是五臂全通过，也不是失败臂被跳过。
- 同次比较 raw_loss 的 raw PSNR 平均提升约 4.839 dB，display PSNR 约 0.337 dB；弱分片低透明度仍严重。
- floor 是尺度非有限；目标矩臂是 tile 分配 CUDA OOM；位置日程臂触发 18GiB reserved 保护，不能统称 CUDA OOM。
- 单场景单 seed 的历史证据，不自动推广候选，也没有证明唯一退化机制。

## 可追溯证据

- [运行身份与逐文件 SHA](../records/mcmc-suite-results-20261017-v1.json)
- [historical_assessment](../../../../../../../outputs/analysis/mcmc-suite-results-20261017-v1/assessment.json)

本地引用已校验；远端未重查。历史原报告保持原字节，描述“当时尚未修复”的历史语句不倒写。
