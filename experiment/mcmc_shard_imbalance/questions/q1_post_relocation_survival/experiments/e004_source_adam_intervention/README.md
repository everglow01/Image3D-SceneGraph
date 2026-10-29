# 来源Adam状态短窗干预

2026-10-29 · **MCMC-Q1-E004 / Q1-B** · **远端合成前置失败；真实公共前缀、Gate 0及两臂未启动。**

## 1. 问题与证据边界

[Q1-A家族配对](../e003_minimal_survival/reports/family-pair-readonly-20261028-v1.md)确认：实际relocation后、noise前S/T六参数位级一致，但初始存活家族等权后，S的20/100步首次死亡风险仍比T高约35—37个百分点。这不足以证明Adam清零致死。

本实验只检验：**同一完整公共起点下，保留来源S迁移前的Adam两矩，是否改变其短期更新和死亡，同时对T/U与重建loss产生什么代价。** 干预对象是六组参数的来源两矩，不只是opacity；首步opacity代数对照只能解释对应的局部更新，不能代替六组状态干预的系统响应。

一个公共前缀不是充分独立重复。结果不能自动证明统计显著性、算法总体改善、长期owner不对称，或15k—30k的百万级分片退化根因。E007与旧Gate失败不追认，生产默认不变。

## 2. 两阶段，分别授权

### 阶段一：100步新公共前缀与Gate 0

- 从冻结12000父checkpoint恢复，名义30000步日程不变，执行12001—12100一次。
- 12100 Adam后、relocation前保存六参数、Adam两矩/step/超参数、strategy、RNG、相机order/cursor的完整typed相位快照。它不是已经完成该步策略的可直接续训checkpoint。
- 原采样、relocation恰好调用一次，保存真实T→S、唯一S、k与T/S/U。relocation后、noise前保存完整相位快照及初始alpha，检查每条T→S六参数位级一致。
- Gate 0核S两矩清零、T/U两矩位级保留、step未改且六组一致、原生Adam类与Parameter映射正确、状态完整且有限。仅支持冻结的非AMSGrad、非fused、非capturable、非differentiable、无weight decay的原Adam配置；实际lr/betas/eps与foreach选项从状态读取，不伪造默认值。
- 来源清零前与目标迁移后的两矩非零比例、符号与绝对值分位数逐组报告。`m/(sqrt(v)+eps)`仅是历史状态尺度，**不是尚未发生的更新**。目标未被此函数清零，不意味着其实际历史必然非零。
- 允许原12100 noise执行一次；保存S/T位置扰动。在Validation入口截停，复用原streaming writer原子保存完整12100 checkpoint，含scheduler、history、camera游标、strategy和四类RNG。不做Validation或导出，不伪造评价值。
- 产生 `gate0.json`，状态为`structural_verified_review_required`。结构通过与动量信号分布分开；不设临时“够大”阈值。如果来源全部六组两矩原本都为0，`effective_intervention_available=false`阻止两臂准备。
- **前缀任务到此退出、释放GPU，绝不自动进入A/B。** Gate 0分析是只读，但生成新前缀需要100次新更新授权，不能使用旧T0/T1六参数快照冒充完整Adam历史。

### 阶段二：用户审阅Gate 0后，A→B各20步

两臂从同一12100完整checkpoint恢复，执行12101—12120：

| 对象 | A：原生 | B：保留来源历史 |
|---|---|---|
| 模型六参数、相机、strategy、RNG | 公共checkpoint | 完全相同 |
| S的exp_avg/exp_avg_sq | 原生清零结果 | 恢复为12100 Adam后、relocation前的值 |
| T/U两矩 | 原生状态 | 位级不变 |
| step、lr、betas、eps等 | 原生状态 | 位级不变 |

原`inject_noise_to_position`不读取Adam两矩，因此共享同一次12100 noise后再恢复S两矩，不会改变这个公共noise结果。**S/T起点相等性检查在noise前；noise后S/T位置可以不同，但A/B同槽位起点仍必须相同。**

恢复完整状态的摘要必须匹配prefix；B白名单只允许唯一S的两个moment字段变化，不能改step、其他状态或参数。checkpoint恢复是显式原生恢复；观察守卫不能靠恢复RNG掩盖消耗。

两臂不重抽映射、不重跑公共前缀、不重复12100 noise。20步内禁止新relocation/拓扑改变。实际相机batch、SH、学习率、原生RNG/noise抽取日程须一致，但**不要求干预后的模型、梯度、Adam长期SHA一致**。

## 3. 首步机制证据

12101的原backward只执行一次，不注入或替换梯度，不做额外forward/backward：

- 六组记录实际更新和梯度摘要；opacity保存逐点总参数梯度、解析正则梯度、两矩、step、超参数及更新前后logit。
- 这里的总参数梯度含重建及正则，不能称为独立测得的纯图像梯度。
- 对A臂已取得的同一份opacity总梯度，离线代入“清零/保留”两种历史两矩计算Adam更新，区分**固定梯度的局部状态效应**与**实际训练系统响应**。
- 每臂用自己的实际梯度、step与状态预测真实更新，对参数、更新量、两矩使用原容差核对。代数不匹配则标记机制解释不可用，不修改原优化器或放宽门槛。
- 保存源/目标原生位置noise前后值；同时保存1/5/10/20步opacity实际更新及moment状态，并将首步更新/梯度按后来首次死亡或右删失分组，只作描述。

首步跨臂梯度仍可能存在CUDA浮点差异；后续干预也会通过模型反馈改变其他Gaussian的梯度，不能把全部差异算成直接局部Adam效应。

## 4. 生存、风险转移与loss相位

- 20步逐点alive/cutoff位图；1/5/10/20步报告首次alpha≤.005的风险。初始未活单列，恢复不抹去首死，未见死亡右删失。诊断死亡不是点被删除。
- 主结果为各owner的S风险`B−A`；并列T（目标点数与家族等权口径）、U、混合目标结局及逐目标死亡先后。
- 同一个新公共前缀允许跨臂配对slot；不能与旧T0/T1槽位配对。目标家族保留0…k个死亡成员分布，不将多个T压成一个二元状态。
- 报告各臂owner1−owner0差距及其变化，不把owner与硬件GPU或相机执行rank混同。
- 原生loss/L1/SSIM/正则按相同步骤和相机配对。**12101 loss在首次干预更新前；12102才可能反映第一次更新。12120 loss仍在第20次更新前**，本轮不额外渲染更新后终点。
- S改善而T/U或loss变差时必须报告代价；不自动宣布算法改善、不临时挑允许恶化阈值、不追加C臂或第三次重复。

## 5. 预算与保护

| 阶段 | 全局更新 | 双rank相机forward | loss backward | Adam参数组step |
|---|---:|---:|---:|---:|
| 公共前缀 | 100 | 200 | 200 | 1200 |
| A | 20 | 40 | 40 | 240 |
| B | 20 | 40 | 40 | 240 |
| 合计 | 140 | 280 | 280 | 1680 |

正式仅12100一次全局relocation（两owner调用），额外渲染0。公共前缀与两臂各30分钟上限，人工审阅间隔不持GPU；collective120秒。单卡reserved≤18GiB，RAM max/high/swap无限额。新输出≤16GiB，准入20GiB并始终保留4GiB；prepare计入全状态快照与原子checkpoint暂存峰值。

合成前置独立：每owner8点SH3，公共前缀4步、A/B各2步，共8更新、16 forward/backward、96 Adam组step，真实图像解码0；另每rank两次一次性位置noise等价性fixture调用，不增加模型forward/backward。上限15分钟，主存无限额。rank1设置零目标，验证rank局部采样跳过不会导致共同边界死锁。

各阶段新目录、once、GPU lease、单执行＋唯一只读监控、原子终态；失败不自动重启，不覆盖或清理旧结果。只允许新实验自身原子checkpoint暂存的原生收尾，不碰用户文件或其他attempt。

## 6. 实现与前置

- `probe.py`复用typed snapshot、原生resume与optimizer hook模式、共同边界consensus和位图编码，不安装旧完整观察器，不修改生产trainer或gsplat。
- `analyze.py`远端CPU按绑定快照重验Gate 0、首步Adam代数、家族风险及原生loss；本机fresh import不加载Torch/NumPy/gsplat。
- `run.py`分别提供prefix与arms授权，绑定新提交、64份执行依赖、父checkpoint与独立前置。旧Q1或E007前置不能替代。
- `tests/test_intervention_remote.py`仅指定主机且 `IMAGE3D_AUTHORIZE_SOURCE_ADAM_TESTS=1` 才收集；报告路径变量为 `IMAGE3D_SOURCE_ADAM_PREFLIGHT_REPORT`，报告和pytest basetemp必须为本实验新runs目录。
- 前置验证原生checkpoint写入/恢复、两矩白名单、step与T/U保留、首步代数、映射/家族分布/删失、原调用预算、观察不变、RNG负例、零目标owner及双rank拒绝；测试代码存在不代表已经通过。

## 7. 执行流程与后续授权边界

1. 本地实现/轻量验证完成 → 用户授权提交 → 用户手动Git同步。
2. 明确授权新合成前置与100步公共前缀后，核目标主机/提交/GPU/磁盘，使用独立unit执行。前置unit为 `image3d-source-adam-preflight-v1.service`，prefix为 `image3d-source-adam-prefix-v1.service`，arms为 `image3d-source-adam-arms-v1.service`，三者主存限制均无限额。
3. 模块 `experiment.mcmc_shard_imbalance.questions.q1_post_relocation_survival.experiments.e004_source_adam_intervention.run`：
   - `--prepare --authorize-prepare --preflight-report <新报告> --output-root <新adam-intervention目录>`。
   - `--prefix --authorize-prefix --protocol-sha256 <prefix协议SHA> --output-root <同目录>`，只能在prefix unit内运行。完成后停在Gate 0。
4. 用户审阅实际moment分布、明确批准两臂后：
   - `--prepare-arms --authorize-prepare-arms --acknowledge-gate0-review --gate0-sha256 <已审阅SHA> --protocol-sha256 <prefix SHA> --output-root <同目录>`。
   - `--arms --authorize-arms --arms-protocol-sha256 <新arms协议SHA> --protocol-sha256 <prefix SHA> --output-root <同目录>`，只能在arms unit内运行。
5. 各阶段唯一只读监控用 `--monitor --monitor-phase prefix|arms --protocol-sha256 <prefix SHA> --output-root <同目录>`，不会停止或重启执行。

两阶段之间保持同一实现提交与执行依赖；若实现需要修复，应另立尝试与前置，不能覆盖已有once或把旧Gate 0绑定到变化后的代码。用户已提交并同步e98c828，2026-10-29按授权执行前置，但在合成A初始化失败，真实公共前缀未启动；下面记录此次终态，不自动按上面的流程重试。


## 8. 2026-10-29前置失败

[失败报告](reports/adam-intervention-20261029-v1-preflight.md)：执行与唯一监控均exit1，pytest 1 failed/73.05秒。8点合成prefix已完成4全局更新，但合成A在零来源rank1上对空moment张量求摘要时触发`stride(-1)=0`的Float→Byte view错误；rank0收到共同拒绝。完整前置未通过，不能用局部合成Gate 0放行真实数据。

没有preflight通过报告、正式协议或正式输出目录；真实100步公共前缀与A/B更新均0。原失败保留，未修代码或补跑。建议在本实验局部处理空来源摘要、保留冻结helper及零来源用例，再经提交/手动同步与新尝试授权进行验证。
