# 固定 B0 GPU runner 代码迁移

**状态：代码与本机 NoTorch 检查完成，真实远端验证未运行。** 日期：2026-10-22。

## 本批实际迁入

- [run.py](../run.py)：prepare、协议校验、worker、五次 backward 回放、监控与 compare 的新维护实现，不调用旧 main/run/worker 执行实验。
- [tensor_analysis.py](../tensor_analysis.py)：原解码、选点、历史点分析与重复统计；计算函数 AST 与原实现相同。
- [legacy.py](../legacy.py) / [gpu-source.json](../gpu-source.json)：集中锁定 13 个旧 scripts 及必要 core/helper/原测试，共 32 份文件。旧数据验真和产品资源模块延迟加载，避免轻量入口顺带加载 NumPy；小型 cgroup/core-hash 函数体保持原样。
- 两份旁置测试：本机 NoTorch 与独立远端 opt-in。旧脚本、测试和原始证据全部保留。

## 本地证据

- 新 GPU 入口与原固定 B0/T1/边界/lifecycle 关联回归：76 passed，1 个远端模块 skipped。
- 连同目录、历史归档与本地分析的完整指定回归：96 passed，30 个子测试通过，1 个远端模块 skipped。
- 重复统计/选点/Adam 标量函数及 replay/load_context/初始化/worker/compare 的计算 AST 与原版一致；资源门禁/core-hash 函数体一致。
- 新旧科学合同除 profile 和冻结桥来源身份外一致；容差、配对方向、五次预算、零 forward/optimizer、初始化审计与旧 Gate 状态不改。
- 冻结清单/依赖篡改、缺文件/路径逃逸、旧协议、错误主机/未授权、输出覆盖、缺 lease、worker module 命令与旧 opt-in 隔离均有检查。
- 独立进程导入和 --help 不加载 Torch/NumPy/gsplat。没有执行真实数组计算、CUDA、prepare 或远端命令。

逐文件身份与状态见 [本机迁移核验记录](gpu-migration-local-20261022-v1.json)。Ruff/AST 检查通过。AST 相同不证明新的进程生命周期/调度已经 CUDA 等价。

## 新旧边界

新 profile 为 `mcmc_fixed_b0_five_repeats_workspace_v1`，新 unit 为 `image3d-fixed-b0-workspace-v1.service`；输出限定本实验 `runs/<new-run-id>/`。子进程通过同一 Python 的 `-m` 启动新模块，保留 cwd、PYTHONPATH 清理、GPU lease、主存无限额及原显存/磁盘/超时保护。

原 v2 协议不能用于新入口，旧 v2 成功不追认为新入口通过。原 12 文件 core、安装库和 checkpoint 不改。当前仍依赖受控冻结桥，不能据此删除旧 scripts 或声称完全解耦。

本地 [analyze.py](../analyze.py) 仍只支持历史 v2 文本，不能用于这条新 GPU 运行身份。新 compare 产生的 comparison.json 归新协议；未来实际运行后的本地整理需按新来源登记。

## 尚未执行

新远端前置、真实新协议五次回放、完整同源新旧数值比较均为 **not_run**。需要用户手动 Git 同步及单独具名执行授权；这份代码迁移不启动任何任务，不恢复原六分支，不改变旧 Gate。
