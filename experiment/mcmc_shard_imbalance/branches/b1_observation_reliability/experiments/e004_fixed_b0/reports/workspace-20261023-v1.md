# 固定 B0 新入口远端迁移验收

2026-10-23 · **MCMC-B1-E004** · [运行记录](../records/workspace-20261023-v1.json)

## 结论

**迁移验收阶段已完成。执行、数据核验和数值验收通过；唯一监控任务的面板终态缺失，作为已知审计缺项保留。** 不把这次工程验收当成长期失衡的新因果证据，不追认旧 Gate1。

- 代码：`4c49e8fa4a9070ee7ef94e259a773d5e9f2c8520`，新 profile `mcmc_fixed_b0_five_repeats_workspace_v1`。
- 执行：`20261023-103744-3d60`，面板 `finished=true / exit_code=0`；前置、prepare、run、compare 和 suite 退出码均为 0。
- 唯一监控：`20261023-103744-ebc6`，日志读到 suite 退出码 0 和最终 comparison SHA，但面板 `running=false / finished=false / exit_code=null`。不推断监控成功，未补标志、未重启、未重跑实验；原因未确定。
- 执行 UTC：2026-10-23 02:37:44—02:43:47；来自执行日志，不从面板 ID 推算。
- 新协议 SHA：`34126cd50d0c7d02fe508cfec5701cc1c4768cf7e5b0d6efa2f09a5b01025ce0`。
- comparison SHA：`cab6a91e20999442fb8d70f0d80285f1b8e4b6a2e55ddae209ed5ec5915037b8`。

## 分层验收

| 层次 | 实际证据 |
|---|---|
| 来源 | 远端 main/提交、32 份冻结依赖、12 文件 core、B0 来源协议/比较和原 gsplat 安装库匹配；新协议绑定 636 个来源/执行文件。 |
| 微型前置 | 3 passed，11.38 秒；两卡合计 10 次合成 backward，零 warmup forward/backward，与真实回放预算分列。 |
| 真实执行 | 两相机各 5 次 backward，共 10 次；零新 forward、optimizer 更新和相机采样。内部 replay 阶段耗时约 21.02 秒，不含来源校验与统计。 |
| 数据有效性 | worker/complete/初始化/协议绑定一致；每相机五次输入与测量期 RNG 固定；远端调用冻结快照加载器校验新旧共 20 份 trial 的文件、形状、dtype 与 canonical SHA，核对新统计 32 个数组文件 SHA；24 份下载文本与远端 SHA 一一匹配。 |
| 本次重复性 | 80 项张量配对全部有限、无超容差；存在非位级相同，结论为 numerically_equivalent。容差仍为 `2e-6 + 2e-4*abs(reference)`。 |
| 新旧迁移数值 | 用冻结旧 helper 在同一份新原始输出上复算 80 项基础统计，逐字段等于新报告；新旧入口按相机/同序号试次对齐的 40 项张量比较均有限且在原容差内。不是跨所有试次的全局保证，也不要求两次运行统计量逐字相同。 |
| 历史六点 | historical_point_analysis 与旧 v2 报告完全一致；仍来自历史 A/A，不是本次 B1 的下游传播。 |
| 资源／服务 | 两卡峰值 reserved 558 / 546 MiB，主存无限额；保留 18 GiB/卡、20 GiB 准入、4 GiB 停止、超时和 lease。终态只读核验时无 GPU 计算进程，image3d-cloud 为 active，原锁和旧数据未改。 |

旧 helper 复算复用了旧统计代码，并非第二份独立统计实现；微型前置另含独立 CPU 参考检查。没有读取 Test/Validation 图像，没有重启旧六分支。

## 位置与下一步

远端项目：`/usr/local/3dgs_new/Image3D-SceneGraph`。本实验目录下：

- `runs/workspace-20261023-v1/`：协议、真实回放与 comparison。
- `runs/workspace-20261023-v1-preflight/`：微型前置及执行阶段日志。
- 本地 `runs/workspace-20261023-v1-local/`：启动命令、任务身份、隔离下载文本和只读核验输出；不入 Git，逐文件 SHA 在运行记录中。

现役实现仍依赖冻结旧文件，不能删除旧 scripts。下一步是设计连通 B1→P2→P3 与独立 A/A 的新协议；尚未实现、冻结或执行，不以本次通过替代。监控终态问题须在下次启动前处理，但不为补标志重跑本次 GPU 实验。
