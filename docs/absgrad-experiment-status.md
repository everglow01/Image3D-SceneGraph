# 绝对梯度消融：当前状态与执行边界

更新日期：2026-10-09。此文是结果索引；`codex.md`仍是唯一执行计划，训练合同见[gaussian-trainer-contract.md](gaussian-trainer-contract.md)。名称仅为“AbsGS启发的绝对梯度消融”，不是完整AbsGS复现。

## 已完成／未完成

| 项目 | 当前状态 |
|---|---|
| 双卡实现、SH3门禁 | CPU102项、每rank58项数值对照及12步合成trainer通过；不是质量结论 |
| signed正式基线 | 30k更新/60k样本→SOR→377-view Validation→2k/4k Train-only完成 |
| 观察区域 | 6个Validation ROI、16个Train相机/24个独立Train ROI已冻结并诊断 |
| 重复推理 | 同模型同16个Train视角一次额外推理，显示uint8逐通道与raw/display指标差均0 |
| 磁盘准入 | 2026-10-08 01:56:33Z可用302.30GiB，原20GiB线通过；每次执行仍须复核 |
| 质量／资源增长门禁 | 用户授权按上述计划验证后训练；新建批准合同，历史草案保持原样；不提供训练seed方差 |
| absolute执行控制 | 31项本地轻量回归、132项远端回归及SH3／真实双rank遥测冒烟通过 |
| absolute正式臂 | 已中断：5418步，高斯数及rank 0显存越界；取消期间发生主机OOM。无完整质量结果，不自动重试 |
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
2. 资源增长限制采用高斯数、主训练wall和每rank reserved相对signed的2倍。终止/失败门禁不得变成新的剪枝或高斯cap算法。既有signed runner保持原样；新候选入口已实现数量、逐rank显存与主阶段时间保护，已通过远端合成GPU与遥测验证；正式场景已资源越界并被主机OOM中断，见下方终态。
3. 用户已授权absolute fresh 30k/60k、SOR、Validation和2k/4k；须先通过验证。不从signed成品续训，不重复跑signed runner（它始终只跑关闭态）。
4. Git同步和远端验证已完成，正式臂曾启动但已在5418步中断；所有6个Validation与24个Train ROI、377个全局逐视角结果及失败均保留，不能事后改框/挑阶段。
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

### 正式启动观察

- 训练提交：`65513f129e99e2cc906bd52d7b75cb73b8700300`；后续文档更新不改变实验protocol。
- 执行任务：`20261008-163110-7086`；只读监控：`20261008-163110-d0f7`。
- 产物：`outputs/experiments/absgrad-absolute-20261008-v1/`；启动审计：`outputs/analysis/absgrad-candidate-launch-20261008-v1/startup.json`。
- 2026-10-08 08:33:50Z观察640更新、1,038,293高斯，两rank峰值reserved为1,486,880,768／1,507,852,288 bytes；两次观察从7步推进到640步。
- 132项远端回归、每rank58项SH3数值检查、12步/24样本合成trainer及真实显存遥测均通过。主训练、SOR、全Validation和Train-only是否完成，仍以之后的退出记录及完整评估为准。

## 最新终态：资源越界及主机OOM中断

2026-10-08 09:01:50Z复核，本轮已停止，GPU空闲，服务恢复。第5400步高斯数达到3,048,410（上限2,975,056）；最后rank 0 reserved为8.75GiB（上限约7.60GiB）。08:51:35取消标记已写入，08:52:17内核global_oom杀死rank 0 PID306870；面板服务随后以oom-kill退出并自动重启。不是助手重启，也不是整机重启。

最后记录5418更新／10836样本，无训练result、阶段exit记录或完整checkpoint；SOR、完整Validation和Train-only未执行。外层shell留下的exit-code=0不能替代完成证据，面板任务实际无正常结束标志。资源门禁已不通过，质量无法评估；不降低门禁、不自动resume或重跑。

证据入口：`outputs/analysis/absgrad-candidate-interruption-20261008-v1/{REPORT.md,terminal-audit.json}`。原启动观察保留为历史。运行中主机RAM保护和面板服务cgroup隔离暴露缺口；目前只做审计，未修改服务/训练逻辑，也未证明具体哪次内存分配导致OOM。


## 2026-10-09：另立完整质量探索协议

用户选择继续完整质量实验，并批准新质量探索计划。原候选资源失败结论保留，新协议不是原门禁的放宽后“通过”，也不产生`PASS_FOR_REPLICATION`。

- 新候选CLI显式加`--quality-exploration`（gate-template/preflight/execute）；独立schema2合同与旧profile互不接受，质量/ROI合同不改，原两倍资源只作报告。
- 仍fresh、同replay/初始化/seed1920/增密阈值与预算，只改absgrad；8个核心文件、生产库、signed runner不改，因此不追加signed训练。
- 新停止线：6,000,000高斯、每rank18GiB reserved、每阶段6h；磁盘20/8/4GiB不变。数量线只停止，不截断增密或剪枝。
- 主机RAM：启动/阶段须≥22GiB MemAvailable；运行低于6GiB或任务memory.current≥14GiB紧急停止。每2秒检查所有阶段；逐rankreserved仍每10秒采集，不宣称硬实时GPU保护。
- 新任务必须处于唯一瞬态systemd service，实际MemoryMax=16GiB、SwapMax=0、memory.oom.group=1，不能仍在gpu-panel.service里；不修改既有服务配置。紧急停止仅自身进程组TERM，5秒后必要时KILL，不写cancel文件进入完整checkpoint保存。旧profile保持原120/30秒行为。
- 先落不可覆盖的stage.failure.json再终止，外部只读监控记录systemd/面板退出与完成文件；不以shell零退出码代替实验完成。正常最终checkpoint/合并仍可能触发新预算，届时保留失败，不擅改训练核心。
- `scripts/evaluate_absgrad_pair.py`逐端点报告全部24Train/6Validation ROI、377逐视角raw差、全局均值/P10、原资源倍率，输出固定三列ROI图片；引用/渲染身份严格核验。数值通过时仍待全部视觉审查，不自动填通过；raw全局与uint8裁剪指标分开。
- `scripts/smoke_absgrad_resources.py`仅限远端受限CPU隔离检查与128MiB小分配/96MiB任务cgroup OOM冒烟，不制造整机OOM、不使用GPU。

本地39项轻量回归、Ruff及diff检查通过，不包含模型训练/渲染/验证。2026-10-09只读复核两张L2空闲、实际面板/cloud/TURN active；主机可用约19.3GiB，低于22GiB。新正式任务尚未启动；远端验证、准入与启动以后置实际记录为准。无完整质量结果，实验仍未完成。旧中断报告上传拒绝未重试或绕过。


### 远端验收与当前阻塞（2026-10-09）

- 执行代码`8e15887dc297e6da791f8b1b2299d18ecbe82e4c`已Git快进同步。用户补充点名授权后才执行；首次权限拦截未绕过。
- **141项远端回归通过**（10.42s），只读preflight通过。独立cgroup隔离exit0；128MiB小分配/96MiB cgroup约束得到`oom-kill`/信号9，内核为`CONSTRAINT_MEMCG`，仅杀该测试unit。面板/cloud/TURN active且原启动时间未改变。
- 批准合同SHA`610e9002fba4c2c80c05707be11e75d5144292136b79b38bfeff072f59b19885`及明确授权经文件API上传，5份小日志取回SHA一致；旧中断审计上传没有重试。
- 01:49:52Z可用RAM **19.18GiB < 22GiB**，还差约2.82GiB；因此新SH3/GPU验证与正式训练尚未运行，正式输出目录尚不存在。最长20分钟只读等待准入，不创建训练/修改远端；满足后仍需重查所有准入与SH3，超时不降低门槛。
- 证据：`outputs/analysis/absgrad-quality-preparation-20261009-v1/{REPORT.md,verification.json,remote-*.log}`。该本地REPORT/verification尚未上传，远端同目录有原始日志、新合同/授权；完整质量实验仍未完成。


## 保存峰值定位与流式优化（2026-10-09）

旧诊断`outputs/analysis/checkpoint-memory-old-20261009-v1/`由e971ba1执行，6个case中3M正常/取消均受限OOM；控制器exit0只说明诊断收集完，不代表case全通过。1.5M主峰出现在gather和rank再打包，匿名内存为主。

新诊断`outputs/analysis/checkpoint-memory-stream-20261009-v1/`由57dfe50执行，0.5M/1.5M/3M/6M正常与取消全部成功；13高斯双rank全部状态/RNG解码等价。正常保存总/匿名采样峰值：1.5M为5.176/2.631GB（旧8.158/7.968GB），3M为8.463/3.378GB，6M为11.918/4.999GB。各指标最大值可能不同时间，不能相加；代表负载不是完整场景峰值保证。旧/新都在独立12GiB无swap cgroup，服务未重启。

148项远端回归通过；公开checkpoint合同与旧读取兼容，内部rank容器版本改变、训练核心hash改变。下一轮新增`run_absgrad_matched_pair.py`同提交fresh signed+absolute，schema3独立合同及匹配收据，旧signed仅历史参照；不绕过旧candidate身份门禁。真实SH3/12步模型等价及取消验收仍待运行。正式22GiB准入未改、完整配对尚未启动，不宣称质量通过。


### 新匹配实验预算已明确批准

2026-10-09真实SH3与12步小trainer/取消验收通过（任务20261009-103842-1e50）；历史小模型最大参数差1.28616e-6，在原容差内，生产gsplat未改。用户随后批准仅新schema3匹配实验改用**12GiB任务上限/18GiB主机准入/6GiB整机余量/11.5GiB任务停止/无swap**。旧schema1/2的22GiB政策不改，也不把旧资源失败翻成通过。新协议SHA`793b8e75e165503f3190d8c26220328eb9c623820ecdca6d3b737834f9392223`绑定已验证保存/SH3证据；所有新鲜signed/absolute阶段同代码、同环境，训练core为9f3878de…，完整预算、ROI/quality、6M/18GiB/6h界限不变。

新预算仅是受限执行决策：600万代表负载成功不保证完整训练一定完成，触线仍停止且不自动加预算。正式启动结果另记。


### 新同代码配对实验已启动（2026-10-09 02:57:49Z观察）

- 实际执行提交`b2c7ac443564f7fcca331eb0ef7f76f7fddf505e`；151项远端回归通过，合同/来源/保存验收身份及现场18GiB准入通过。
- 执行`20261009-105648-27e3`、只读监控`20261009-105707-0863`；目录`outputs/experiments/absgrad-streaming-matched-20261009-v1/`。
- fresh signed主训练已推进至108更新/216样本、999,901高斯。独立cgroup实际12GiB/无swap，观测memory.current约7.897GiB、oom事件0；面板/cloud/TURN active。这不是完整训练或质量结果。
- 顺序为新signed完整四阶段→同代码absolute完整四阶段→两个端点冻结ROI/377-view配对；旧signed/失败保留。后台等待终态，不自动重试、放宽限制或加载Test。
- 启动审计`outputs/analysis/absgrad-streaming-matched-launch-20261009-v1/startup.json`本地记录。活动远端保持执行HEAD；后续仅文档提交暂不向运行中仓库快进，避免阶段身份门禁失败。实验完成后再同步最新状态。


### 最新终态：新signed触及任务内存停止线（2026-10-09 03:05:21Z）

新匹配实验尚未完成。fresh signed在3005更新/6010样本、1,183,595高斯处停止；保护采样记录3000步。`train.failure.json`原因为`task_memory_at_11.5_gib`：memory.current为12,419,674,112 bytes（11.5667GiB），整机仍可用11.9626GiB。memory.events的max/oom/oom_kill均0，内核本轮无OOM；子进程TERM退出（-15）、外层exit-code1，面板/cloud/TURN均active且启动时间未变。GPU已空闲。

第3000步训练内Validation已完成377/377，两rank best-model文件存在；不是SOR后selection或最终checkpoint。无完整checkpoint、主训练result或signed/pair complete；absolute尚未创建，后续SOR/正式Validation/Train-only/配对质量未执行。

代表负载保存优化通过不等于12GiB覆盖完整训练。此次触线发生于首次Validation/best-model写出后，尚未进入最终流式checkpoint；缺少触线时memory.stat与RSS/PSS，具体内存组成未定位，不据此修改预算或宣称保存优化无效。面板任务缺结束标志，但原生退出/失败记录与systemd一致；其元数据缺失原因未确认。

证据：`outputs/analysis/absgrad-streaming-matched-interruption-20261009-v1/{REPORT.md,terminal-audit.json,train.failure.json,train.exit.json,driver.log,train.log}`，仅本地归档，4份下载文件SHA与远端一致。旧失败保留，不自动重跑/resume/加预算，质量不可评估，Test与默认未动。远端保持执行提交b2c7ac4，未同步后续文档或修改原始产物。

### 代码审查修复完成，实验保持停止

2026-10-09，用户要求按review修复、梳理代码，并明确“先不要启动实验”。已完成峰值统计、模型文件I/O、正式内存时间线、准入/后验终态及同代码资源分母修复；共享实验逻辑移入absgrad_experiment，candidate/matched变为薄CLI，新增版本化systemd启动/只读watch脚本。128项本地非模型检查通过，模型/GPU等价与实际资源复测未运行；不宣称真实训练内存问题已解决。

按用户对精确清单的授权，仅删除两次失败运行的4个best模型分片，合计928,323,568 bytes，删除后已核验；日志、协议、进度、成功signed及诊断证据保留。历史终态审计中的“文件存在”是删除前事实，新增清理回执单独记录。

代码核心哈希已改变且新增模块纳入身份范围，旧SH3/保存验收不得直接沿用；旧合同及历史产物未改。远端仍保持原执行代码，本轮没有Git同步或新任务。详见`docs/absgrad-code-review-20261009.md`后续修复节。
