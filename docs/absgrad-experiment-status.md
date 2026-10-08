# 绝对梯度消融：当前状态与执行边界

更新日期：2026-10-08。此文是结果索引；`codex.md`仍是唯一执行计划，训练合同见[gaussian-trainer-contract.md](gaussian-trainer-contract.md)。名称仅为“AbsGS启发的绝对梯度消融”，不是完整AbsGS复现。

## 已完成／未完成

| 项目 | 当前状态 |
|---|---|
| 双卡实现、SH3门禁 | CPU102项、每rank58项数值对照及12步合成trainer通过；不是质量结论 |
| signed正式基线 | 30k更新/60k样本→SOR→377-view Validation→2k/4k Train-only完成 |
| 观察区域 | 6个Validation ROI、16个Train相机/24个独立Train ROI已冻结并诊断 |
| 重复推理 | 同模型同16个Train视角一次额外推理，显示uint8逐通道与raw/display指标差均0 |
| 磁盘准入 | 2026-10-08 01:56:33Z可用302.30GiB，原20GiB线通过；每次执行仍须复核 |
| 质量／资源增长门禁 | 用户授权按上述计划验证后训练；新建批准合同，历史草案保持原样；不提供训练seed方差 |
| absolute执行控制 | 候选入口与运行保护已实现，31项轻量回归通过；尚未远端验证或执行 |
| absolute正式臂 | 已授权验证通过后启动；实际任务状态以独立启动审计为准，不从signed续训 |
| Test、生产默认推广 | 未执行、未授权 |

## signed证据

- 训练提交：`9cf78ad84abdc2da61f03512af09a3484b7e75bd`；seed20260729、双卡、最长边1920。
- 双臂内部schema11仅差`densification.absgrad`；主预算、增密阈值/结束点、recovery-prune、SOR及Train-only保持一致。
- 主训练3016个Train相机各19–20次，包含第15000步的增密前各9–10次；Train-only各1–2次。与历史Project两段采样序列相同，不等于每ROI有效可见次数相同。
- SOR后Validation：raw PSNR21.916840/SSIM0.803270；2k Train-only后：22.147273/0.806472，均377/377成功。
- Train-only角色为`held_out_after_train_only_control`且`selection_eligible=false`，不能替代原selection。
- 模型SHA256：`9113b393646a4b1e254355791b79b498cadec8969f0d2bbd7f791427bd7f2475`。
- 隔离gsplat rendering SHA256：`f1e345b2c943f2d5541d91d5b7d89e76c91467ab419a5dfe6448d2a1ba2ada9e`；生产依赖未改。

3670组Train椅脚/地毯仍有细节欠拟合。新旧关闭态全局均值接近但局部漂移较大，且代码/环境不同，不能当作重复seed噪声或AbsGrad效应。额外推理仅确认16个相机的显示像素和指标复现；旧raw预测张量未保存，未做raw像素逐元素比较。

## 结果索引

以下路径相对仓库根；生成产物不入Git。远端根为`/usr/local/3dgs_new/Image3D-SceneGraph`。

| 目录 | 入口 |
|---|---|
| `outputs/experiments/absgrad-signed-20260930-v1/experiment/` | `protocol.json`、`complete.json`及signed原生模型/评估 |
| `outputs/analysis/absgrad-signed-completion-20261008-v1/` | `REPORT.md`、`summary.json`、`sampling-audit.json`、`roi-audit.json` |
| `outputs/analysis/absgrad-signed-train-roi-20261008-v1/` | `REPORT.md`、`train-roi.json`、`train-roi-metrics.json`、`contact-sheets/` |
| `outputs/analysis/absgrad-signed-render-repeatability-20261008-v1/` | `REPORT.md`、`result.json`、`contract.json` |
| `outputs/analysis/absgrad-stage2b-gate-proposal-20261008-v1/` | 先读`CALIBRATION_UPDATE.md`，再读原`PROPOSAL.md`与`gate-proposal.json`；仍是草案 |
| `outputs/analysis/absgrad-results-sync-20261008-v1/` | 本轮同步`manifest.json`、最新`next-step.json`；以远端校验结果确认同步完成 |

历史报告保留其生成时状态，不改写原哈希证据。低磁盘/待Train渲染/校准首次拒绝是历史事实，当前状态以上表为准。

## 已授权执行顺序

1. 按用户授权另建批准合同：主目标Train/Validation两ROI各+0.5dB/+0.01 SSIM等为本轮工程门禁；不能冒充统计置信界。推理稳定性与视觉门禁可用于限定单场景探索，但训练波动仍未知。
2. 资源增长限制采用高斯数、主训练wall和每rank reserved相对signed的2倍。终止/失败门禁不得变成新的剪枝或高斯cap算法。既有signed runner保持原样；新候选入口已实现数量、逐rank显存与主阶段时间保护，尚未完成远端GPU验证。
3. 用户已授权absolute fresh 30k/60k、SOR、Validation和2k/4k；须先通过验证。不从signed成品续训，不重复跑signed runner（它始终只跑关闭态）。
4. 候选执行控制的本地轻量回归已完成；经授权Git同步并完成远端验证后才启动正式臂；所有6个Validation与24个Train ROI、377个全局逐视角结果及失败均保留，不能事后改框/挑阶段。
5. 全部门禁通过至多`PASS_FOR_REPLICATION`；第二seed/场景、Test与默认推广均需独立授权。

此前4d673b4仅同步文档与结果。当前候选准备新增执行入口和资源监控，并为主训练／final-fit CLI增加显式启用的遥测参数；训练核心、配置、隔离库与signed执行脚本未改。未来协议分别记录实际执行提交和signed历史提交，并核对源码/配置/隔离库身份；不得重写旧实验protocol以伪造相同Git HEAD。


## 候选执行入口（验证后启动已授权）

`scripts/run_absgrad_candidate.py`有三个显式子命令：

- `gate-template --output <新文件>`：只生成`DRAFT_NOT_APPROVED`合同，不加载模型、不开GPU、不批准实验。
- `preflight --signed-experiment <signed实验根> --gate-contract <批准后的新合同> --gate-sha256 <合同SHA256> --output-dir <新实验目录>`：只读核对合同、signed证据、源文件和单变量配置，不创建候选目录；不是GPU环境验收。
- `execute`：参数与preflight相同；只有已批准合同才允许继续，限制主机`i-94B8D131`及既有远端仓库。批准标记是执行前置条件，不代替用户授权；不得自动把草案改为批准。

本地草案和轻量验证记录位于`outputs/analysis/absgrad-candidate-preparation-20261008-v1/`，不入Git。批准时另建合同文件，不覆盖草案或历史报告。模板绑定质量草案的SHA及其`proposed_quality_gates`、Train ROI SHA、两倍资源草案与原20/8/4GiB政策；不采用旧报告中的18GiB选项。若修改阈值，应先改合同/实现/测试并重新审核，不能仅改一个运行参数。

### 运行保护与身份

- 外层持有`outputs/.gpu.lock`；目录必须全新且位于`outputs/experiments/`下。没有自动resume。
- 启动核对干净且受Git跟踪的代码、训练核心相对于signed提交的差异、模型/配置/源协议哈希、训练code/environment hash、隔离rendering和现场两张L2。新CLI/监控代码由实际执行提交记录，不宣称全仓库与signed同提交。
- 保持fresh 30k/60k→SOR→377-view Validation→固定拓扑2k/4k Train-only；每步两个相机，主训练与Train-only相机序列摘要分别必须等于signed。评估不完整或held-out角色错误会阻止写完成标记。
- 仅候选显式传入`--resource-telemetry-dir`。每rank独立线程每10秒保存累计`torch.cuda.max_memory_reserved`，不改变训练循环、梯度、随机种子或策略；不使用`nvidia-smi used_memory`冒充PyTorch reserved。
- 监控增量读取每步高斯数和两个rank遥测，每10秒检查一次；进程正常退出也做最后检查。主训练2倍wall、主训练及Train-only的高斯数量/逐rank显存超限均失败，不加入cap或剪枝。
- 显存采样包括模型合并阶段，比signed原result中的训练峰值统计窗口更宽；这是保守停止保护。公平资源对照仍分别使用原生result的同口径峰值与新增监控峰值，不能混称严格同口径2倍测量。SOR及独立selection阶段沿用磁盘/6小时保护，没有逐rank reserved保护。
- 初始遥测最多等待600秒，已有遥测超过120秒未刷新、损坏、缺rank或退出时不完整均失败。这是待批准合同中的运行健全性限制，不是算法参数。10秒轮询不是硬实时显存上限，不能保证先于OOM拦截。
- 失败优先写取消标记，等待120秒，再向本任务进程组TERM，30秒后必要时KILL。主训练走既有checkpoint取消路径；final-fit原本不保存阶段checkpoint，不宣称能够恢复。保留失败目录，不重启服务、不停止其他任务。

### 验证范围

31项本地轻量回归覆盖配置单变量、草案拒绝、身份篡改、缺失/陈旧/损坏遥测、资源边界、增量日志、采样摘要、正常/异常子进程、停止升级、四阶段调度及默认入口。只运行标准库模拟、配置逻辑与短CPU子进程，没有模型训练、渲染、Test或GPU验证；Ruff通过。

完成标记为`absolute_training_complete_quality_pending`，不是质量通过。六个Validation ROI、24个Train ROI、全部视角指标与视觉否决仍须在获得候选结果后完整评审；本入口不自动渲染额外ROI、不挑选最好阶段、不自动产生`PASS_FOR_REPLICATION`。

2026-10-08授权补充：用户明确要求提交、Git同步、远端验证并允许开始训练。旧草案不改写；批准合同另存于`outputs/analysis/absgrad-candidate-launch-20261008-v1/`。现有SH3冒烟新增显式`--resource-telemetry`，验证真实双rank累计reserved、12步/24样本进度与结束记录后才进入正式训练。最新运行状态由该独立目录及面板任务记录确认，不能把授权或本地测试当作已启动证据。
