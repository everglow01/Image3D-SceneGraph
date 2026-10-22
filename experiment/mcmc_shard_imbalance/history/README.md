# MCMC 历史档案

**2026-10-22，C 第一批：文档归档与维护退役登记。代码删除 0 个。**

这里保存原始决策与冻结研究合同，不替代 [主课题的现行认识](../README.md)。原文中的“未执行”“仅实现”是当时状态；不能因后续成功而重写历史，也不能将过去授权解释为新的执行许可。

## 档案

| 来源 | 归档内容 | 保留在原处 |
|---|---|---|
| [codex.md](../../../codex.md) | [72 条 MCMC 原始决策](codex-decisions-20261013-20261022.md)，2026-10-13—2026-10-22 | 项目总体计划、A/B/C 决策、独立 Project 诊断、用户原导航与其他历史 |
| [Trainer 合同](../../../docs/gaussian-trainer-contract.md) | [19 节研究历史合同](trainer-research-contracts-20261022.md) | Model/rendering 等产品条款及原 19 个标题/跳转 |
| [Checkpoint 合同](../../../docs/gaussian-checkpoint-contract.md) | [5 节研究历史边界](checkpoint-research-contracts-20261022.md) | checkpoint 格式、发布/保留/恢复条款及原 5 个标题/跳转 |

## 完整性与权威边界

[archive-manifest.json](archive-manifest.json) 保存整理前来源 SHA、原区段边界、每片段的顺序/字节数/SHA 和稳定锚点。片段之外添加导航，片段内部逐字保留。应用归档时已将三个原文件反向重构并与事前全文核对，区段外仅 codex 新增一条 C 决策。

历史路径仍以仓库根解释。原 H2/H3 标题保留使旧章节锚点仍可到达原位置，但文件行号可能变化。**原冻结条款继续有效**，具体运行仍受原 protocol 和对应历史合同约束；不以归档取消阈值、预算或 Gate。

本地 25 份已存 protocol 未发现这三个文档路径的精确绑定；未做远端全量审计。没有改运行代码、旧产物或科学结论，也没有新实验。

## 减负范围

- `codex.md`：450,661 → 362,768 bytes，入口正文减少 87,893 bytes。
- `docs/gaussian-trainer-contract.md`：80,774 → 39,546 bytes，入口正文减少 41,228 bytes。
- `docs/gaussian-checkpoint-contract.md`：16,240 → 9,847 bytes，入口正文减少 6,393 bytes。

这是内容归位，不是删除证据或节省等量总磁盘空间。codex 的其他课题历史本批不处理。

## 维护退役，不是物理删除

固定 B0 v2 的两份旧本地分析不再作为维护入口，现役实现是 [E004/analyze.py](../branches/b1_observation_reliability/experiments/e004_fixed_b0/analyze.py)。旧文件仍被来源清单绑定，原位保留。C 第一批当时尚未迁移 GPU 入口，因此该批代码删除清单为空。此后 [固定 B0 代码已迁入](../branches/b1_observation_reliability/experiments/e004_fixed_b0/README.md)，但远端验证未运行且仍依赖旧代码；产品共享资源代码也仍在使用。

下一批真正减少 scripts/src 文件，需要先完成相应 B 迁移闭包和验证，再确认新的 C 清单。个人文档、原锁、旧 protocol/complete/报告/张量均不在清理范围。

## 本地验证

```bash
python3 -I -B -m unittest discover -s experiment -p 'test_history.py' -v
```

仅标准库检查；完整回归按 [研究工作区](../../README.md) 的 experiment 测试范围运行，不启动训练。
