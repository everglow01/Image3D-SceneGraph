# B 阶段试点：本地分析迁移验证

这次是既有固定 B0 v2 文本的分析复算，不是新的 GPU 实验，不产生新的 P2/P3。原脚本与证据保持原位。

## 结果

- 输入 33 份文本，共 3,367,798 bytes；按冻结来源清单核对 SHA。
- assessment、sensitivity、download-verification 三份 JSON 的结构和值与历史参考完全一致。
- repetition-metrics.csv 逐字一致；报告的非空行逐行一致，允许且仅允许空白行布局差异。
- 新入口把输入和输出分离，只有实验 runs 下的全新目录可写；核验/计算完成后才创建输出，最后写完成记录。
- 8 项新增标准库测试通过：小型标量/汇总、语义拒绝、输入/路径、来源身份、输出/失败保护、隔离 CLI 与优化模式、静态依赖、历史结果一致性。
- GPU runner、12 文件冻结 core、旧 tests、旧协议与运行记录均未迁移或重新执行；旧 Gate1 不追认。

## 身份与复现

- [逐输出比较与来源 SHA](analysis-migration-20261022-v1.json)
- [冻结分析来源清单](../analysis-source.json)
- [当前分析 CLI 与范围](../README.md)

原始复算产物在 `runs/analysis-migration-20261022-v1/`，该目录不进入 Git，其他 checkout 未必存在。上述 JSON 保存了完成记录与每份输出的 SHA，不能据此宣称远端仍可用或已重新核验。
