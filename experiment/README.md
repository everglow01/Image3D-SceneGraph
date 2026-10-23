# 研究工作区

按“主问题 → 子问题／支线 → 实验 → 运行证据”组织研究，不是训练平台。

- [MCMC 分片恢复不对称](mcmc_shard_imbalance/README.md)：问题、结论、20 项实验索引与下一步。
- [迁移边界](mcmc_shard_imbalance/migration.md)：现役实现、冻结依赖与保留原因。
- [历史档案](mcmc_shard_imbalance/history/README.md)：原始决策与冻结研究合同。
- [codex.md](../codex.md)：项目级计划。

## 文件职责

- 课题 README：现行认识与研究路线；实验 README：目的、结论、限制及使用入口。
- experiment.json：唯一实验身份、依赖和代码/证据指针。JSON 路径相对仓库根，Markdown 链接相对所在文件。
- records/：一次执行或只读分析的状态和来源 SHA；跨问题引用同一记录，不重复计数。历史记录与新迁移验收分列，不把记录条数当成新增科学实验次数。
- reports/：审核结论；history/：不可改写的历史载荷。原 protocol 仍是预算、判据及源绑定权威。
- runs/<run-id>/：新入口的独立产物目录，和 scratch/incoming 一样不入 Git；旧 outputs/analysis 原位保留，不擅改旧冻结入口路径。

源码、测试、索引和审核文档可进入 Git；不忽略整个 experiment/。不提交模型、数据集、大张量、日志或凭据。Git 历史不是产物备份。

## 执行边界

1. 先冻结假设、对照、指标、预算及停止条件，再申请执行。同步完成、目录或协议存在都不是授权。
2. 本机仅标准库、AST/Ruff、NoTorch 轻量检查；真实数组、CUDA、训练需远端单独授权。新代码只经 Git，由用户手动同步。
3. 保留指定主机/项目、GPU lease、防重复、唯一执行和只读监控、显存/磁盘/时限保护；主存无限额，具体值以获授权协议为准。
4. 执行成功、数据有效、数值验收、假设支持分开记录。新代码/范围是新身份，不借历史成功晋级；失败独立保留，不覆盖旧产物或恢复 RNG 掩盖副作用。
5. 不自动启动 Test/Validation、训练或旧六分支。local_verified 只指本地文件/SHA 核验；remote_not_rechecked 不代表远端当前可用；remote_verified_at_record_time 仅表示记录时核验；null 必须读其缺失原因。
6. 产品代码仍在 src/backend，产品不反向依赖 experiment；正式 CLI 留 scripts。无实际复用不提取公共框架。

## 本机检查

在仓库根运行索引与历史检查（标准库，不导入产品/Torch）：

```bash
IMAGE3D_REQUIRE_LOCAL_EVIDENCE=1 python3 -I -B -m unittest discover -s experiment -p 'test_*.py' -v
```

旁置检查用现有 pytest 显式选择 experiment；默认 testpaths 已包含 tests 和 experiment，不为目录整理在本机运行整个产品/张量测试集：

```bash
IMAGE3D_REQUIRE_LOCAL_EVIDENCE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider experiment
```

其他 checkout 缺 ignored outputs 时，可不设置 IMAGE3D_REQUIRE_LOCAL_EVIDENCE；索引会警告，历史 golden 对比会跳过，不得当作完整证据验收。远端模块默认跳过，不能记作 CUDA 通过。
